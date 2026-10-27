"""Verbatim AST-selected Nezha definitions; see restoration manifest."""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .config import Config

from .kernels import jit_integrate_trajectory

@dataclass
class RNAForceFieldParams:
    # ---- base-pair attraction (Gaussian around r0) ----
    bp_r0: float = 1.05          # 目标距离（粗粒度单位）
    bp_sigma: float = 0.25       # 宽度
    bp_gain: float = 1.0         # 总强度（乘在 J 上）
    bp_tail_enable: bool = False # 启用长程弱尾巴
    bp_tail_sigma: float = 1.20  # 尾巴宽度（越大越长程）
    bp_tail_gain: float = 0.20   # 尾巴强度（相对 bp_gain 的系数）

    # ---- excluded volume (softplus barrier) ----
    rep_r: float = 0.35          # 小于此距离开始强烈排斥
    rep_soft: float = 0.08       # softplus 平滑尺度
    rep_gain: float = 0.35       # 排斥强度
    rep_power: float = 2.0       # soft barrier 阶数；2.0 对应二次穿透惩罚
    stack_k: float = 0.5         # 法线堆积力 (默认值改为 0.5)

    # ---- backbone bond (harmonic) ----
    bond_r0: float = 0.58        # 相邻点目标距离
    bond_k: float = 2.0          # 键强度

    # ---- bending (discrete curvature) ----
    bend_k: float = 0.25         # 曲率惩罚强度（越大越“硬”）

    # ---- masks / normalization ----
    min_seq_sep: int = 4         # |i-j| < 4 直接禁止 pair（避免近邻不物理短路）
    energy_norm: bool = True     # 是否按有效项数归一化能量（避免长序列 loss 爆）

def _make_seqsep_mask(L: int, device: torch.device, dtype: torch.dtype, min_sep: int) -> torch.Tensor:
    """
    返回 [1,L,L] mask：当 |i-j| >= min_sep 时为 1，否则 0（对角线也会是 0）。
    """
    idx = torch.arange(L, device=device)
    dist = (idx[None, :] - idx[:, None]).abs()
    m = (dist >= int(min_sep)).to(dtype=dtype)
    eye = torch.eye(L, device=device, dtype=dtype)
    m = m * (1.0 - eye)
    return m.unsqueeze(0)  # [1,L,L]

def _energy_norm_denom(m: torch.Tensor, norm_mode: str, energy_norm_default: bool) -> torch.Tensor:
    """
    返回每个样本的能量归一化分母（[B]）。
    norm_mode 为空时：energy_norm_default=True -> pair；False -> length。
    """
    mode = str(norm_mode or "").strip().lower()

    if mode in ("none", "raw", "off"):
        return m.new_ones(m.shape[0])

    pair_mask = m[:, :, None] * m[:, None, :]
    valid_pairs = (pair_mask.sum(dim=(1, 2)) - m.sum(dim=1)).clamp_min(1.0)

    if mode in ("pair", "pairs", "pairwise", "l2"):
        return valid_pairs
    if mode in ("sqrt_pair", "sqrt_pairs", "sqrt", "sqrt_l2"):
        return valid_pairs.sqrt()
    if mode in ("residue", "length", "len", "l1"):
        return m.sum(dim=1).clamp_min(1.0)
    if mode:
        # 未知模式：回退到 pair
        return valid_pairs

    if bool(energy_norm_default):
        return valid_pairs
    return m.sum(dim=1).clamp_min(1.0)

def _energy_norm_mode_from_config() -> str:
    return str(getattr(Config, "RNA_ENERGY_NORM_MODE", "")).strip().lower()

def _normalize_energy_per_sample(
    V: torch.Tensor,
    m: torch.Tensor,
    energy_norm_default: bool = True,
    norm_mode: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一能量归一化入口：
    - 返回逐样本归一化能量 V_norm_per=[B]
    - 返回逐样本分母 denom_per=[B]
    """
    mode = _energy_norm_mode_from_config() if norm_mode is None else str(norm_mode).strip().lower()
    denom_per = _energy_norm_denom(m, mode, bool(energy_norm_default)).clamp_min(1.0)
    return V / denom_per, denom_per

def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    acc_dtype = torch.float64 if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else x.dtype
    x_acc = x.to(dtype=acc_dtype)
    mask_acc = mask.to(dtype=acc_dtype)
    num = (x_acc * mask_acc).sum(dim=dim, keepdim=True)
    den = mask_acc.sum(dim=dim, keepdim=True).clamp_min(float(eps))
    return (num / den).to(dtype=x.dtype)

def _masked_center(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    center = _masked_mean(x, mask, dim=dim, eps=eps)
    acc_dtype = torch.float64 if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else x.dtype
    centered = (x.to(dtype=acc_dtype) - center.to(dtype=acc_dtype)) * mask.to(dtype=acc_dtype)
    return centered.to(dtype=x.dtype)

def compute_rna_force_field_fast(
    q: torch.Tensor,        # [B, L, 6] (Pos:0-3, Ori:3-6)
    J: torch.Tensor,        # [B, L, L] in [0,1], 3D contact J
    pad_mask: torch.Tensor, # [B, L] 1=valid
    params: RNAForceFieldParams,
    contact_rc: float = 0.8,
    return_aux: bool = False,
    nff_k: torch.Tensor = None,   # 神经力场强度
    nff_d0: torch.Tensor = None,  # 神经力场平衡距离
    nff_j_mix: float = None,      # NFF 的 J 门控混合系数（1=全门控, 0=无门控）
):
    """
    [Physics Engine Core] 快速解析力场计算
    
    改进点：
    1. 掩码分离：排斥力(Repulsion)使用全掩码，吸引力(Attraction)使用远程掩码。
    2. 刚体动力学修正：法线力(Orientation Force)增加了切空间投影，防止模长漂移。
    """
    device = q.device
    dtype = q.dtype
    B, L, D = q.shape
    m = pad_mask.to(dtype=dtype)
    
    # =====================================================================
    # 1. 基础几何与掩码准备 (Geometry & Masks)
    # =====================================================================
    
    # --- 拆分坐标与法线 ---
    q_pos = q[:, :, :3]
    q_ori = q[:, :, 3:]
    # 必须对法线归一化，防止模长漂移导致计算错误
    q_ori = F.normalize(q_ori, p=2, dim=-1, eps=1e-8)

    # Improve numerical translation invariance before pairwise differencing.
    q_pos = _masked_center(q_pos, m, dim=1)

    # --- 距离矩阵计算 ---
    q2 = (q_pos * q_pos).sum(dim=-1)                         # [B,L]
    dot = torch.bmm(q_pos, q_pos.transpose(1, 2))            # [B,L,L]
    dist_sq = (q2[:, :, None] + q2[:, None, :] - 2.0 * dot).clamp_min(0.0)
    dist = torch.sqrt(dist_sq + 1e-6)

    # --- 掩码构造 (Mask Construction) [Critical Fix] ---
    # 基础 Padding 掩码
    base_mask = (m[:, :, None] * m[:, None, :])  # [B,L,L]
    eye = torch.eye(L, device=device, dtype=dtype).unsqueeze(0)
    not_eye = (1.0 - eye)

    # --- 方向向量 (Unit Vectors) ---
    # invd: 距离倒数; dir_ij: j 指向 i 的单位向量
    safe_dist = dist + 1e-6 + (1.0 - base_mask) * 1e3
    invd = 1.0 / safe_dist
    dir_ij = (q_pos[:, :, None, :] - q_pos[:, None, :, :]) * invd[:, :, :, None]

    # [改进] 拆分为两类掩码：
    # 1. mask_rep: 用于排斥力 (Repulsion)。近邻原子也必须排斥，防止穿模。
    mask_rep = base_mask * not_eye
    
    # 2. mask_long: 用于吸引力/配对 (Attraction/Pairing)。过滤掉近邻(|i-j|<min_sep)。
    min_seq_sep = int(getattr(params, "min_seq_sep", 2))
    sep_mask = _make_seqsep_mask(L, device=device, dtype=dtype, min_sep=min_seq_sep)
    mask_long = mask_rep * sep_mask 

    # --- J 矩阵处理 ---
    # Jw: unified contact weights for analytical terms, NFF, and contact stats.
    Jw = 0.5 * (J + J.transpose(1, 2))
    Jw = Jw.to(dtype=dtype) * mask_long

    # =====================================================================
    # 2. 力场分项计算 (Force Components)
    # =====================================================================

    # --- 1) Base-Pair Attraction (Watson-Crick) ---
    r0 = float(getattr(params, "bp_r0", 1.05))
    sig = float(getattr(params, "bp_sigma", 0.50))
    bp_gain = float(getattr(params, "bp_gain", 1.0))

    z = (dist - r0) / max(sig, 1e-6)
    g = torch.exp(-0.5 * z * z)                        
    V_bp_ij = -0.5 * bp_gain * Jw * g
    V_bp = V_bp_ij.sum(dim=(1, 2))                     

    # Force uses full derivative to stay consistent with V_bp_ij's 0.5 double-counting fix.
    dVdd_bp = bp_gain * Jw * g * (z / max(sig, 1e-6))
    F_bp_i = -(dVdd_bp[:, :, :, None] * dir_ij).sum(dim=2) # [B,L,3]
    
    # [可选] 长程弱尾巴：更宽的高斯，提供远距微弱吸引
    if bool(getattr(params, "bp_tail_enable", False)):
        sig_t = float(getattr(params, "bp_tail_sigma", sig))
        gain_t = float(getattr(params, "bp_tail_gain", 0.0))
        if (sig_t > 0.0) and (gain_t != 0.0):
            z_t = (dist - r0) / max(sig_t, 1e-6)
            g_t = torch.exp(-0.5 * z_t * z_t)
            V_bp_ij = V_bp_ij + (-0.5 * bp_gain * gain_t * Jw * g_t)
            dVdd_bp = dVdd_bp + (bp_gain * gain_t * Jw * g_t * (z_t / max(sig_t, 1e-6)))
            V_bp = V_bp_ij.sum(dim=(1, 2))
            F_bp_i = -(dVdd_bp[:, :, :, None] * dir_ij).sum(dim=2)

    # --- 2) Short-Range Repulsion [Fix: Use mask_rep] ---
    # 修正：使用 mask_rep 而不是 mask_long，确保相邻残基不会重叠
    rep_r = float(getattr(params, "rep_r", 0.35))
    rep_soft = float(getattr(params, "rep_soft", 0.08))
    rep_gain = float(getattr(params, "rep_gain", 0.35))
    rep_power = float(getattr(params, "rep_power", 2.0))
    if (not math.isfinite(rep_power)) or rep_power < 1.0:
        rep_power = 1.0

    u = (rep_r - dist) / max(rep_soft, 1e-6)
    sp = F.softplus(u)
    sig_u = torch.sigmoid(u)
    
    # 这里的 mask_rep 保证了即使是相邻原子，距离过近也会产生排斥
    rep_shape = torch.pow(sp, rep_power)
    V_rep_ij = 0.5 * rep_gain * rep_shape * mask_rep 
    V_rep = V_rep_ij.sum(dim=(1, 2))

    rep_grad_shape = torch.pow(sp, rep_power - 1.0)
    A_rep = (rep_gain * rep_power / max(rep_soft, 1e-6)) * rep_grad_shape * sig_u * mask_rep
    F_rep_i = (A_rep[:, :, :, None] * dir_ij).sum(dim=2)

    # --- 3) Backbone Bond (Harmonic) ---
    bond_k = float(getattr(params, "bond_k", 2.0))
    bond_r0 = float(getattr(params, "bond_r0", 0.58))
    V_bond = torch.zeros(B, device=device, dtype=dtype)
    F_bond = torch.zeros(B, L, 3, device=device, dtype=dtype)

    if L >= 2:
        dq = q_pos[:, 1:, :] - q_pos[:, :-1, :]              # [B,L-1,3]
        d = torch.sqrt((dq * dq).sum(dim=-1) + 1e-6)         # [B,L-1]
        dd = (d - bond_r0)
        edge_mask = (m[:, 1:] * m[:, :-1])
        V_bond = 0.5 * bond_k * ((dd * dd) * edge_mask).sum(dim=1)

        dVdd = bond_k * dd
        # Force uses negative potential gradient: F = -dV/dq
        f = (dVdd[:, :, None] * dq / (d[:, :, None] + 1e-6))
        f = f * edge_mask[:, :, None]
        F_bond[:, :-1, :] += f
        F_bond[:, 1:,  :] -= f

    # --- 4) Backbone Bending (Stiffness) ---
    bend_k = float(getattr(params, "bend_k", 0.25))
    V_bend = torch.zeros(B, device=device, dtype=dtype)
    F_bend = torch.zeros(B, L, 3, device=device, dtype=dtype)

    if L >= 3:
        c = q_pos[:, :-2, :] - 2.0 * q_pos[:, 1:-1, :] + q_pos[:, 2:, :]
        c2 = (c * c).sum(dim=-1)
        m_mid = (m[:, :-2] * m[:, 1:-1] * m[:, 2:])
        V_bend = 0.5 * bend_k * (c2 * m_mid).sum(dim=1)

        g = bend_k * c
        g = g * m_mid[:, :, None]
        F_bend[:, :-2, :] += -g
        F_bend[:, 1:-1, :] += 2.0 * g
        F_bend[:, 2:,  :] += -g

    # --- 5) Neural Pairwise Force ---
    F_neural = torch.zeros(B, L, 3, device=device, dtype=dtype)
    V_neural = torch.zeros(B, device=device, dtype=dtype)

    if nff_k is not None and nff_d0 is not None:
        k_eff = nff_k.to(dtype=dtype).clamp_min(0.0)
        use_j = bool(getattr(Config, "RNA_NFF_USE_J_WEIGHT", True))
        j_pow = float(getattr(Config, "RNA_NFF_J_POWER", 1.0))

        # 神经力场通常用于长程约束，使用 mask_long
        w = mask_long
        if use_j:
            if "Jw" in locals(): jw = Jw
            else: jw = 0.5 * (J + J.transpose(1, 2))
            jw = jw.clamp(0.0, 1.0) ** j_pow
            j_floor = float(getattr(Config, "RNA_NFF_J_FLOOR", 0.0))
            if j_floor > 0.0:
                jw = j_floor + (1.0 - j_floor) * jw
            if nff_j_mix is not None:
                mix = float(nff_j_mix)
                if mix < 0.0: mix = 0.0
                if mix > 1.0: mix = 1.0
                # mix=1: 全门控; mix=0: 无门控
                jw = jw * mix + (1.0 - mix)
            w = w * jw

        diff = dist - nff_d0.to(dtype=dtype)
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)

        # V_neural_ij has 0.5 factor while summing both (i,j) and (j,i),
        # so force needs a factor 2.0 to satisfy F = -dV/dq exactly.
        f_mag = -2.0 * k_eff * diff * w
        V_neural_ij = 0.5 * k_eff * (diff.pow(2)) * w
        V_neural = V_neural_ij.sum(dim=(1, 2))
        F_neural = (f_mag.unsqueeze(-1) * dir_ij).sum(dim=2)

    # --- 6) Stacking Force (Orientation) [Improved Physics] ---
    # 目标：让相邻或配对碱基的法线(Orientation)趋向平行
    stack_k = float(getattr(params, "stack_k", 0.5))
    
    # 法线夹角余弦: dot > 0 (平行), < 0 (反平行)
    normal_dot = torch.bmm(q_ori, q_ori.transpose(1, 2)) # [B, L, L]
    
    # 构造权重 (Neighbors + Pairs)
    idx_tensor = torch.arange(L, device=device)
    # 邻居掩码 |i-j|==1
    ngb_mask = ((idx_tensor[None, :] - idx_tensor[:, None]).abs() == 1)
    ngb_mask = ngb_mask.to(dtype).unsqueeze(0) * base_mask
    
    # 权重组合：相邻堆积(1.5) + 接触堆积(1.0)
    align_weight = (ngb_mask * 1.5) + (Jw * 1.0)
    
    # [Fix] 对称化权重矩阵，确保力的作用与反作用平衡
    W_align = 0.5 * (align_weight + align_weight.transpose(1, 2))
    
    # 势能: V = 0.5 * k * sum( W * (1 - dot) )
    V_stack_ij = 0.5 * stack_k * W_align * (1.0 - normal_dot)
    V_stack = V_stack_ij.sum(dim=(1, 2))

    # [Fix] 力的计算与切空间投影 (Tangent Space Projection)
    # 原始力 F_raw = k * sum(W_ij * n_j)
    # 物理含义：n_i 倾向于指向邻居 n_j 的加权平均方向
    F_stack_raw = stack_k * torch.bmm(W_align, q_ori)
    
    # 投影：去除 F_raw 在 n_i 方向上的分量
    # 因为 n_i 是单位向量，受力必须垂直于 n_i 才能只产生旋转而不改变模长
    # F_tangent = F_raw - (F_raw . n_i) * n_i
    dot_self = (F_stack_raw * q_ori).sum(dim=-1, keepdim=True)
    F_stack_ori = F_stack_raw - (dot_self * q_ori)

    # =====================================================================
    # 3. 力场汇总 (Total Force Assembly)
    # =====================================================================
    
    # 位置力 F3 (前3维)
    F3_pos = (F_bp_i + F_rep_i + F_bond + F_bend + F_neural) * m[:, :, None]
    
    # 法线力 F3_ori (后3维)
    F3_ori = F_stack_ori * m[:, :, None]
    
    # 拼接为 6维 力向量
    F_total = torch.zeros(B, L, 6, device=device, dtype=dtype)
    F_total[:, :, :3] = F3_pos
    F_total[:, :, 3:] = F3_ori 

    # 总势能
    V_total = (V_bp + V_rep + V_bond + V_bend + V_neural + V_stack)
    energy_norm_default = bool(getattr(params, "energy_norm", True))

    if not return_aux:
        return F_total, V_total

    # --- 辅助统计信息 (Aux Stats) ---
    k_contact = torch.sigmoid((float(contact_rc) - dist) / 0.1)
    k_contact = k_contact.to(dtype=dtype) * mask_long # 使用 mask_long 统计接触
    
    contact_i = (Jw * k_contact).sum(dim=-1)
    Z = mask_long.sum(dim=-1).clamp_min(1.0)
    contact_i = contact_i / Z 

    c_mean = (contact_i * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
    c_var = ((contact_i - c_mean[:, None]) ** 2 * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
    c_std = torch.sqrt(c_var + 1e-8)
    force_rms = torch.sqrt(((F3_pos * F3_pos).sum(dim=-1) * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0)) + 1e-8)
    V_norm, _ = _normalize_energy_per_sample(
        V=V_total, m=m, energy_norm_default=energy_norm_default
    )
    
    aux = {
        "V_bp": V_bp.detach(),
        "V_norm": V_norm.detach(),
        "contact_mean": c_mean.detach(),
        "contact_std": c_std.detach(),
        "force_rms": force_rms.detach(),
        "V_stack": V_stack.detach()
    }
    return F_total, V_total, aux

def residue_invariants(q: torch.Tensor, J: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """
    [Physics Upgrade v4.6] 
    1) Fix Neighbor Feature Contamination: 严格检查坐标有效性 (避免 0.0 坐标导致的巨大伪向量)
    2) Mask-aware robust normal estimation
    3) Explicit Triangular Features
    """
    B, L, D = q.shape
    device = q.device
    dtype = q.dtype

    # --- [Step 1: 构建严格的有效性 Mask] ---
    # 1. 序列 Mask (pad_mask)
    m_bool = (pad_mask > 0.5)
    
    # 2. 坐标 Mask (coord_valid): 检查坐标是否非原点
    # 对于混合训练，Reactivity数据的坐标是 0.0，必须过滤掉
    is_coord_valid = (q.abs().sum(dim=-1) > 1e-6) # [B, L]
    
    # 3. 综合 Mask: 既要是序列的一部分，又要有有效坐标
    valid_atom = m_bool & is_coord_valid
    
    m = valid_atom.to(dtype)
    m_exp = m.unsqueeze(-1)

    # --- [Step 2: 计算安全的差分向量] ---
    pos = _masked_center(q[:, :, :3], m, dim=1)

    # A. Forward Diff: p[i+1] - p[i]
    # 只有当 i 和 i+1 都是 valid_atom 时，这条边才物理存在
    pos_pad_end = F.pad(pos, (0, 0, 0, 1)) # [B, L+1, 3]
    raw_diff_fwd = pos_pad_end[:, 1:] - pos_pad_end[:, :-1]
    
    # 构建边 Mask
    valid_next = F.pad(valid_atom, (0, 1), value=False)[:, 1:] # Shift Left
    has_fwd_valid = valid_atom & valid_next # i 和 i+1 都有效
    
    # 清洗：无效的边强制置 0
    diff_fwd = raw_diff_fwd * has_fwd_valid.unsqueeze(-1).to(dtype)

    # B. Backward Diff: p[i] - p[i-1]
    # 只有当 i 和 i-1 都是 valid_atom 时有效
    pos_pad_start = F.pad(pos, (0, 0, 1, 0)) # [B, L+1, 3]
    raw_diff_bwd = pos_pad_start[:, 1:] - pos_pad_start[:, :-1]
    
    # 构建边 Mask
    valid_prev = F.pad(valid_atom, (1, 0), value=False)[:, :-1] # Shift Right
    has_bwd_valid = valid_atom & valid_prev # i 和 i-1 都有效
    
    # 清洗：无效的边强制置 0
    diff_bwd = raw_diff_bwd * has_bwd_valid.unsqueeze(-1).to(dtype)

    # --- [Step 3: 邻居掩码准备 (用于后续计算)] ---
    # 此时 has_fwd_valid 和 has_bwd_valid 已经排除了 "幽灵坐标"
    has_next = has_fwd_valid
    has_prev = has_bwd_valid
    
    has_next_u = has_next.unsqueeze(-1)
    has_prev_u = has_prev.unsqueeze(-1)

    # 4) v_main (切向向量)
    # 优先用 forward，没有则用 backward
    v_main = diff_fwd.clone()
    v_main = torch.where(has_next_u, v_main, diff_bwd)
    
    # 标记孤立点 (既没有前驱也没有后继的有效点)
    mask_isolated = ~(has_next_u | has_prev_u)

    # -----------------------------
    # Triangular Features (u, v from i-1->i and i->i+1)
    # -----------------------------
    vec_u = torch.where(has_prev_u, diff_bwd, torch.zeros_like(diff_bwd))  # i-1 -> i
    vec_v = torch.where(has_next_u, diff_fwd, torch.zeros_like(diff_fwd))  # i -> i+1

    u_norm = F.normalize(vec_u, dim=-1, eps=1e-4)
    u_norm = torch.nan_to_num(u_norm, nan=0.0, posinf=0.0, neginf=0.0)
    v_norm = F.normalize(vec_v, dim=-1, eps=1e-4)
    v_norm = torch.nan_to_num(v_norm, nan=0.0, posinf=0.0, neginf=0.0)

    tri_valid = (has_prev & has_next).unsqueeze(-1).to(dtype)

    tri_cos = (u_norm * v_norm).sum(dim=-1, keepdim=True) * tri_valid  # [B,L,1]

    cross = torch.cross(u_norm, v_norm, dim=-1)
    cross = torch.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0)
    n_norm = torch.norm(cross, dim=-1, keepdim=True)                    # [B,L,1]
    n_norm_safe = n_norm.clamp_min(1e-4)
    tri_sine = n_norm * tri_valid                                       # [B,L,1]

    # -----------------------------
    # Robust normal construction (v4.4 logic)
    # -----------------------------
    if D >= 6:
        normal = q[:, :, 3:6]
        normal = F.normalize(normal, dim=-1, eps=1e-6)
        normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        valid = (has_prev & has_next).unsqueeze(-1) & (n_norm > 1e-3)
        normal0 = torch.where(valid, cross / n_norm_safe, torch.zeros_like(cross))

        # Forward fill (autograd-safe via torch.where + stack)
        if L == 1:
            normal_fwd = normal0
        else:
            fwd_list = [normal0[:, 0]]
            for i in range(1, L):
                cur = normal0[:, i]
                prev = fwd_list[-1]
                cur_invalid = (torch.norm(cur, dim=-1, keepdim=True) < 1e-6)
                prev_valid = (torch.norm(prev, dim=-1, keepdim=True) > 1e-6)
                fill = cur_invalid & prev_valid
                fwd_list.append(torch.where(fill, prev, cur))
            normal_fwd = torch.stack(fwd_list, dim=1)

        # Backward fill
        if L == 1:
            normal_bwd = normal_fwd
        else:
            bwd_list = [normal_fwd[:, -1]]
            for i in range(L - 2, -1, -1):
                cur = normal_fwd[:, i]
                nxt = bwd_list[-1]
                cur_invalid = (torch.norm(cur, dim=-1, keepdim=True) < 1e-6)
                next_valid = (torch.norm(nxt, dim=-1, keepdim=True) > 1e-6)
                fill = cur_invalid & next_valid
                bwd_list.append(torch.where(fill, nxt, cur))
            bwd_list.reverse()
            normal_bwd = torch.stack(bwd_list, dim=1)

        normal = normal_bwd

        # Degenerate fallback (ensure v_main is non-zero before building fallback)
        n_mag = torch.norm(normal, dim=-1, keepdim=True)

        v_mag = torch.norm(v_main, dim=-1, keepdim=True)
        const_z = torch.zeros_like(v_main); const_z[..., 2] = 1.0
        v_main_safe = torch.where(v_mag < 1e-6, const_z, v_main)
        v_main_n = F.normalize(v_main_safe, dim=-1, eps=1e-8)

        ref_x = torch.zeros_like(v_main_n); ref_x[..., 0] = 1.0
        ref_z = torch.zeros_like(v_main_n); ref_z[..., 2] = 1.0
        dot_z = (v_main_n * ref_z).sum(dim=-1, keepdim=True).abs()
        ref = torch.where(dot_z > 0.9, ref_x, ref_z)

        fallback = torch.cross(v_main_n, ref, dim=-1)
        fallback = F.normalize(fallback, dim=-1, eps=1e-6)
        fallback = torch.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)

        normal = torch.where(n_mag < 1e-6, fallback, normal)
        normal = F.normalize(normal, dim=-1, eps=1e-6)
        normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)

    normal = normal * m_exp

    # -----------------------------
    # Local frame (Gram-Schmidt)
    # -----------------------------
    v_mag = torch.norm(v_main, dim=-1, keepdim=True)
    safe_dir = normal.clone()
    safe_dir_mag = torch.norm(safe_dir, dim=-1, keepdim=True)
    const_z = torch.zeros_like(safe_dir); const_z[..., 2] = 1.0
    safe_dir = torch.where(safe_dir_mag < 1e-6, const_z, safe_dir)
    v_main = torch.where(mask_isolated | (v_mag < 1e-6), safe_dir, v_main)

    z_axis = normal
    proj = (v_main * z_axis).sum(dim=-1, keepdim=True) * z_axis
    x_axis = v_main - proj
    x_axis = F.normalize(x_axis, dim=-1, eps=1e-6)
    x_axis = torch.nan_to_num(x_axis, nan=0.0, posinf=0.0, neginf=0.0)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    R = torch.stack([x_axis, y_axis, z_axis], dim=-2)

    # -----------------------------
    # Local relative coordinates
    # -----------------------------
    vec_to_prev = -diff_bwd
    vec_to_prev = torch.where(has_prev_u, vec_to_prev, torch.zeros_like(vec_to_prev))
    local_prev = torch.matmul(R, vec_to_prev.unsqueeze(-1)).squeeze(-1)

    vec_to_next = diff_fwd
    vec_to_next = torch.where(has_next_u, vec_to_next, torch.zeros_like(vec_to_next))
    local_next = torch.matmul(R, vec_to_next.unsqueeze(-1)).squeeze(-1)

    # -----------------------------
    # Scalar features
    # -----------------------------
    curvature = torch.zeros(B, L, 1, device=device, dtype=dtype)
    if L >= 3:
        m_tri = m_bool[:, :-2] & m_bool[:, 1:-1] & m_bool[:, 2:]
        c = pos[:, :-2] - 2.0 * pos[:, 1:-1] + pos[:, 2:]
        curvature[:, 1:-1] = torch.norm(c, dim=-1, keepdim=True) * m_tri.unsqueeze(-1).to(dtype)

    J_masked = J * m.unsqueeze(1) * m.unsqueeze(2)
    contact = J_masked.sum(dim=-1, keepdim=True)

    seq_pos = torch.arange(L, device=device, dtype=dtype).view(1, L, 1).expand(B, -1, -1)
    rel_pos = seq_pos / max(L, 1.0)

    norm_dot_prev = torch.zeros(B, L, 1, device=device, dtype=dtype)
    norm_dot_next = torch.zeros(B, L, 1, device=device, dtype=dtype)
    d_bwd = (normal[:, 1:] * normal[:, :-1]).sum(dim=-1, keepdim=True)
    norm_dot_prev[:, 1:] = d_bwd * has_prev[:, 1:].unsqueeze(-1).to(dtype)
    norm_dot_next[:, :-1] = d_bwd * has_next[:, :-1].unsqueeze(-1).to(dtype)

    # -----------------------------
    # Concatenate invariants (dim=13)
    # -----------------------------
    inv = torch.cat([
        torch.tanh(local_prev),        # [3]
        torch.tanh(local_next),        # [3]
        torch.tanh(curvature),         # [1]
        torch.tanh(contact / 5.0),     # [1]
        rel_pos,                       # [1]
        norm_dot_prev,                 # [1]
        norm_dot_next,                 # [1]
        tri_cos,                       # [1]
        tri_sine                       # [1]
    ], dim=-1)

    inv = torch.nan_to_num(inv, nan=0.0) * m_exp
    return inv.float()

def project_orientation(q: torch.Tensor, eps: float = 1e-8, no_grad: bool = False) -> torch.Tensor:
    """
    将 q 的后 3 维(orientation)归一化到单位向量，避免数值漂移。
    关键点：q_pos 必须保持可导；orientation 是否可导由 no_grad 控制。

    Args:
        q: [B, L, 6]  (pos[0:3], ori[3:6])
        eps: 数值稳定项
        no_grad: True 则 orientation 不反传；False 则全部可导（排障期推荐 False）
    Returns:
        q_proj: [B, L, 6]
    """
    q_pos = q[:, :, :3]
    q_ori = q[:, :, 3:]

    if no_grad:
        with torch.no_grad():
            n = torch.sqrt((q_ori * q_ori).sum(dim=-1, keepdim=True) + eps)
            q_ori = q_ori / n
    else:
        n = torch.sqrt((q_ori * q_ori).sum(dim=-1, keepdim=True) + eps)
        q_ori = q_ori / n

    return torch.cat([q_pos, q_ori], dim=-1)

class BioSystem:
    def _fold_precompute_sums_and_contacts(self, J: torch.Tensor, m: torch.Tensor):
        """预计算 sum_m / c_mean / c_max。"""
        sum_m = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        sum_m_3d = sum_m.unsqueeze(-1)
        sum_m_scalar = sum_m.squeeze(-1)

        with torch.no_grad():
            row_sum = (J * (m.unsqueeze(2) * m.unsqueeze(1))).sum(dim=-1)  # [B,L]
            # Normalize by valid neighbors to remove sequence-length bias in modulator inputs.
            deg_denom = (sum_m - 1.0).clamp_min(1.0)
            row_density = row_sum / deg_denom
            c_mean = (row_density * m).sum(dim=1, keepdim=True) / sum_m     # [B,1], ~[0,1]
            c_max = (row_density + (1.0 - m) * -1e9).max(dim=1, keepdim=True).values

        return sum_m, sum_m_3d, sum_m_scalar, c_mean, c_max

    def _fold_build_obs_vec(
        self,
        q: torch.Tensor,
        F_bio: torch.Tensor,
        V_bio: torch.Tensor,
        J: torch.Tensor,
        m: torch.Tensor,
        m_3d: torch.Tensor,
        sum_m_3d: torch.Tensor,
        sum_m_scalar: torch.Tensor,
        c_mean: torch.Tensor,
        c_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """"构建 obs_vec：宏观 5 维摘要 + 几何不变量。需保持梯度畅通！"""
        B, L, _ = q.shape
        energy_grad_scale = 0.0
        if torch.is_grad_enabled():
            energy_grad_scale = self._get_grad_gate_scale(
                "RNA_OBS_ENERGY_GRAD_MAX_SCALE",
                "RNA_OBS_ENERGY_GRAD_WARMUP_STEPS",
                "RNA_OBS_ENERGY_GRAD_INFER_FULL_SCALE",
            )
        self._last_obs_energy_grad_scale = float(energy_grad_scale)

        # [Fix] 移除了 torch.no_grad()，允许梯度流过观察向量
        # 1. 质心（位置维度）
        q_pos = q[:, :, :3]
        cm = _masked_mean(q_pos, m, dim=1)

        # 2. 回转半径 Rg（仅由坐标决定，避免被法线分量污染）
        dq_cm = _masked_center(q_pos, m, dim=1)
        rg_sq = (dq_cm.pow(2).sum(dim=-1) * m).sum(dim=1) / sum_m_scalar
        rg = torch.sqrt(rg_sq + 1e-8).view(B, 1)

        # 3. 受力 RMS (保留梯度，因为 F_bio 包含关于 q 的导数)
        f_sq_sum = ((F_bio * m_3d).pow(2)).sum(dim=(1, 2))
        denominator = sum_m_scalar.view(-1) * 3.0
        f_mean_sq = f_sq_sum / denominator.clamp_min(1.0)

        f_rms = torch.sqrt(f_mean_sq + 1e-8).view(B, 1)

        # 4. 能量：使用按样本归一化能量并做稳定压缩，减少长度偏置与尺度飘移。
        energy_norm_default = bool(getattr(self.ff_params, "energy_norm", True))
        obs_energy_mode = str(getattr(Config, "RNA_OBS_ENERGY_NORM_MODE", "")).strip().lower()
        V_obs_src = V_bio.float().view(B)
        V_obs_src = self._apply_grad_gate(V_obs_src, energy_grad_scale)
        V_obs_norm, _ = _normalize_energy_per_sample(
            V=V_obs_src,
            m=m,
            energy_norm_default=energy_norm_default,
            norm_mode=(obs_energy_mode if obs_energy_mode else None),
        )
        e_scale = float(getattr(Config, "RNA_OBS_ENERGY_ASINH_SCALE", 10.0))
        if e_scale <= 0.0:
            e_scale = 10.0
        e_val = torch.asinh((V_obs_norm / e_scale)).view(B, 1)

        # 5. 组装
        # c_mean, c_max 也是预计算好的 detach 变量，保持原样即可
        summary_macro = torch.cat([rg, c_mean, c_max, f_rms, e_val], dim=-1)

        micro_feats = None 
        if self.use_geom_invariants:
            micro_feats = residue_invariants(q, J, m)
            denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            micro_mean = (micro_feats * m.unsqueeze(-1)).sum(dim=1) / denom
            obs_vec = torch.cat([summary_macro, micro_mean], dim=-1)
        else:
            obs_vec = summary_macro

        return obs_vec, micro_feats

    def _get_grad_gate_scale(
        self,
        max_scale_name: str,
        warmup_name: str,
        infer_full_name: str,
        default_max: float = 0.0,
        default_warmup: int = 0,
        default_infer_full: bool = False,
    ) -> float:
        max_scale = float(getattr(Config, max_scale_name, default_max))
        if (not math.isfinite(max_scale)) or (max_scale < 0.0):
            max_scale = 0.0
        max_scale = min(max_scale, 1.0)
        if max_scale <= 0.0:
            return 0.0

        warm_steps = int(getattr(Config, warmup_name, default_warmup))
        infer_full = bool(getattr(Config, infer_full_name, default_infer_full))

        warm = 1.0
        if warm_steps > 0:
            if (not self.encoder.training) and infer_full:
                warm = 1.0
            else:
                warm = min(1.0, float(getattr(self, "global_step", 0)) / float(max(1, warm_steps)))

        return float(max_scale * warm)

    class _RMSGradPrecondition(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, grad_scale: float):
            rms = x.detach().float().pow(2).mean().sqrt()
            coeff = float(grad_scale) / float(torch.sqrt(1.0 + (rms * rms)).item())
            ctx.coeff = float(max(0.0, min(float(grad_scale), coeff)))
            return x

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output * ctx.coeff, None

    @staticmethod
    def _apply_grad_gate(x: Optional[torch.Tensor], grad_scale: float):
        if x is None or (not isinstance(x, torch.Tensor)):
            return x
        scale = float(max(0.0, min(1.0, grad_scale)))
        if not x.requires_grad:
            return x
        if scale <= 0.0:
            return x.detach()
        return BioSystem._RMSGradPrecondition.apply(x, scale)

    def _fold_update_hallucination_loss(
        self,
        total_hallucination_loss: torch.Tensor,
        last_phys_pred: Optional[torch.Tensor],
        obs_vec: torch.Tensor,
    ) -> torch.Tensor:
        """计算 hallucination loss：对宏观前 5 维做 smooth_l1。"""
        if last_phys_pred is None:
            return total_hallucination_loss

        target_raw = obs_vec.detach()
        target_norm = target_raw.clone()

        target_norm[..., 0] = torch.log1p(target_raw[..., 0])
        target_norm[..., 3] = torch.log1p(target_raw[..., 3])
        # obs energy term is already normalized + compressed in _fold_build_obs_vec.
        target_norm[..., 4] = target_raw[..., 4]

        h_loss = F.smooth_l1_loss(last_phys_pred, target_norm[..., :5])
        total_hallucination_loss = total_hallucination_loss + h_loss
        return total_hallucination_loss

    def _get_nff_j_mix(self) -> Optional[float]:
        """NFF 的 J 门控混合系数（1=全门控, 0=无门控）。"""
        if not bool(getattr(Config, "RNA_NFF_J_SCHEDULE_ENABLE", False)):
            return None
        start = float(getattr(Config, "RNA_NFF_J_SCHEDULE_START", 0.0))
        end = float(getattr(Config, "RNA_NFF_J_SCHEDULE_END", 1.0))
        steps = int(getattr(Config, "RNA_NFF_J_SCHEDULE_STEPS", 0))
        if steps <= 0:
            mix = end
        else:
            t = float(getattr(self, "global_step", 0)) / float(steps)
            if t < 0.0:
                t = 0.0
            if t > 1.0:
                t = 1.0
            mix = start + (end - start) * t
        if mix < 0.0:
            mix = 0.0
        if mix > 1.0:
            mix = 1.0
        return float(mix)

    def _langevin_step_simple(
        self,
        q: torch.Tensor,              # [B, L, 6] (带有梯度)
        J: torch.Tensor,
        m: torch.Tensor,
        temp: float,
        dt: float,
        nff_k: torch.Tensor = None,
        nff_d0: torch.Tensor = None,
        *,
        generator: torch.Generator = None,
        force_clip: float = None,
        max_drift: float = None,
        pos_only_noise: bool = True,
        noise_clip_sigma: float = None,
        detach_force: bool = False,
    ) -> torch.Tensor:
        """
        [Physics] 极简 Langevin 步进 (Robust Out-of-place Version)
        """
        # ---- 0) 数据清洗：必须保留 q 的梯度状态 ----
        # 使用 nan_to_num 确保数值安全，同时 float() 确保类型
        q32 = torch.nan_to_num(q.float(), nan=0.0, posinf=0.0, neginf=0.0)

        # J 和 m 是常数/条件，不需要梯度，直接 detach
        J32 = torch.nan_to_num(J.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        m32 = torch.nan_to_num(m.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        B, L, D = q32.shape
        if D != 6:
            raise ValueError(f"_langevin_step_simple expects q shape [B,L,6], got {q32.shape}")

        # ---- 1) 计算力场：逻辑核心点 ----
        # [Fix 1] 如果开启 detach_force，仅在计算力场时使用分离的坐标
        # 这样 F_total 不会产生关于 q 的导数（即阻断了 Hessian）
        q_for_force = q32.detach() if detach_force else q32

        nff_j_mix = None
        if (nff_k is not None) and (nff_d0 is not None):
            nff_j_mix = self._get_nff_j_mix()

        F_total, _ = compute_rna_force_field_fast(
            q_for_force, J32, m32, self.ff_params,
            return_aux=False,
            nff_k=nff_k, nff_d0=nff_d0,
            nff_j_mix=nff_j_mix,
        )
        # [Optimization] 加上 posinf/neginf 防御极端物理重叠
        F_total = torch.nan_to_num(F_total, nan=0.0, posinf=0.0, neginf=0.0)

        # 即使 q 已经分离，F_total 可能还连着 nff_k/feat 的梯度，彻底切断
        if detach_force:
            F_total = F_total.detach()

        # ---- 2) 力 Clip (Out-of-place) ----
        fc = float(force_clip) if force_clip is not None else float(getattr(Config, "RNA_LANGEVIN_FORCE_CLIP", 0.0))
        if fc > 0.0:
            max_abs = F_total.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            # 使用乘法生成新 Tensor，不修改原 F_total
            F_total = F_total * torch.clamp(fc / max_abs, max=1.0)

        # ---- 3) 漂移项与 Clip (Out-of-place) ----
        dt_f = float(dt)
        drift = F_total * dt_f

        md = float(max_drift) if max_drift is not None else float(getattr(Config, "RNA_LANGEVIN_MAX_DRIFT", 0.0))
        if md > 0.0:
            # [Fix 2] 使用 torch.cat 替代切片赋值，彻底杜绝 In-place 错误
            drift_pos = drift[..., :3]
            drift_ori = drift[..., 3:]

            dp_max = drift_pos.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            s = torch.clamp(md / dp_max, max=1.0)

            # cat 生成新 Tensor
            drift = torch.cat([drift_pos * s, drift_ori], dim=-1)

        # ---- 4) 噪声项 (Out-of-place) ----
        T = float(temp)
        sigma = math.sqrt(max(0.0, 2.0 * T * dt_f))

        # 注意：使用 q32.device 和 dtype 保证一致性
        noise = torch.randn(q32.shape, device=q32.device, dtype=q32.dtype, generator=generator) * sigma
        noise = torch.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)

        # Noise Clip
        if noise_clip_sigma is not None and noise_clip_sigma > 0:
             limit = noise_clip_sigma * sigma
             noise = noise.clamp(-limit, limit)

        if pos_only_noise:
            # [Fix 3] 使用 cat 保证计算图不被原地破坏
            noise_pos = noise[..., :3]
            noise_ori = torch.zeros_like(noise[..., 3:])
            noise = torch.cat([noise_pos, noise_ori], dim=-1)

        # ---- 5) 更新 (核心：q32 必须是带梯度的那个) ----
        # q_new = q (带梯度) + drift (detach过的) + noise (常数)
        # 梯度路径：Loss -> q_new -> q32 -> Encoder
        q_new = q32 + drift + noise

        # 最后强制投影与 Mask
        q_new = project_orientation(
            q_new,
            eps=1e-8,
            no_grad=bool(getattr(Config, "RNA_ORI_NORM_NO_GRAD", False)),
        )
        q_new = q_new * m32.unsqueeze(-1)

        return q_new
