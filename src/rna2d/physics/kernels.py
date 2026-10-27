"""Verbatim AST-selected Nezha definitions; see restoration manifest."""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from .config import Config

class RWKVCellLite(nn.Module):
    """
    轻量级 RWKV 循环单元 (Minimalist RWKV-style Recurrent Cell).
    用于嵌入到 SNN 物理循环中，提供时间记忆能力。
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        key_mode: str = "softplus_zero",
        pair_state_cap: Optional[float] = None,
        pair_state_cap_mode: str = "hard",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.key_mode = str(key_mode).strip().lower()
        self.pair_state_cap = None if pair_state_cap is None else float(max(pair_state_cap, 1e-6))
        self.pair_state_cap_mode = str(pair_state_cap_mode)
        
        # 核心参数: Receptance, Key, Value, Gate
        self.receptance = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(input_dim, hidden_dim, bias=False)
        
        # 时间衰减参数 (Time decay)
        self.time_decay = nn.Parameter(torch.zeros(hidden_dim))
        self.time_first = nn.Parameter(torch.zeros(hidden_dim))
        
        # 输出层
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ln = nn.LayerNorm(hidden_dim)
        
        # Keep PyTorch's default Linear initialization. In the continuous-front-end
        # Ghost controller this matches the empirically better legacy RWKV behavior
        # more closely than the orthogonal / zero-gate override.
        nn.init.uniform_(self.time_decay, -6.0, -3.0)
        nn.init.zeros_(self.time_first)

    def _map_key(self, k_raw: torch.Tensor) -> torch.Tensor:
        if self.key_mode == "softplus_zero":
            # Align the positive key to zero-input behavior: softplus(0)=log(2), so
            # subtract that baseline and clamp to keep the denominator accumulator
            # non-negative without writing a constant offset during silence.
            return torch.clamp(F.softplus(k_raw) - math.log(2.0), min=0.0)
        if self.key_mode == "softplus":
            return F.softplus(k_raw)
        if self.key_mode == "exp":
            return torch.exp(k_raw.clamp(-20.0, 20.0))
        if self.key_mode == "raw":
            return k_raw
        raise ValueError(f"unknown RWKV key_mode: {self.key_mode}")

    def _apply_pair_state_cap(self, a_new: torch.Tensor, b_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pair_state_cap is None:
            return a_new, b_new

        pair_rms = torch.sqrt(a_new.pow(2).mean(dim=-1, keepdim=True) + b_new.pow(2).mean(dim=-1, keepdim=True) + 1e-12)
        cap = a_new.new_tensor(self.pair_state_cap)

        if self.pair_state_cap_mode == "soft":
            ratio = pair_rms / cap
            scale = torch.where(ratio > 1.0, torch.tanh(ratio) / ratio.clamp_min(1e-6), torch.ones_like(ratio))
            return a_new * scale, b_new * scale

        scale = torch.clamp(pair_rms / cap, min=1.0)
        return a_new / scale, b_new / scale

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        x: [Batch, input_dim]
        state: Tuple(numerator, denominator) or None. 形状均为 [Batch, hidden_dim]
        """
        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        # 1. 状态初始化 (如果是第一步)
        if state is None:
            a = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
            b = torch.zeros(B, self.hidden_dim, device=device, dtype=dtype)
        else:
            a, b = state
            
        # 2. 线性投影
        r = torch.sigmoid(self.receptance(x))
        k_raw = self.key(x)
        # The denominator/state accumulator acts like a positive weight bank in the
        # stabilized Ghost RWKV path, so map the raw key before all WKV updates.
        k_weight = self._map_key(k_raw)
        v = self.value(x)
        g = torch.sigmoid(self.gate(x))
        
        # 3. 准备 RWKV 时间常数
        u = self.time_first
        w = -torch.exp(self.time_decay) # 强制为负数
        
        # 4. WKV 注意力计算 (Attention mechanism)
        e_u = torch.exp(u)
        e_w = torch.exp(w)
        
        kv = k_weight * v
        
        # 当前步的输出 (WKV)
        # 加上 1e-6 防止除以零
        wkv = (a + e_u * kv) / (b + e_u * k_weight + 1e-6)
        
        # 5. 更新状态 (传递给下一步)
        # a_new = decay * a + k * v
        # b_new = decay * b + k
        a_new = e_w * a + kv
        b_new = e_w * b + k_weight
        a_new, b_new = self._apply_pair_state_cap(a_new, b_new)
        
        # 6. 最终输出
        # 去掉 LayerNorm 在零输入处学出的静态偏置，避免“零驱动也有控制输出”。
        core = self.ln(r * wkv)
        core_zero = self.ln(torch.zeros_like(core))
        y = self.output(core - core_zero) * g
        
        return y, (a_new, b_new)

@dataclass
class ProtocolSpec:
    phy_dim: int
    proto_dim: int
    device: torch.device
    dtype: torch.dtype

    @property
    def side_dim(self) -> int:
        return int(self.phy_dim - self.proto_dim)

    def validate(self):
        if int(self.phy_dim) < int(self.proto_dim):
            raise ValueError("phy_dim must be >= proto_dim")

class _CouplingMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_mult=0.25):
        super().__init__()
        h = max(8, int(round(hidden_mult * max(in_dim, out_dim))))
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.SiLU(),
            nn.Linear(h, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class USBLink(nn.Module):
    def __init__(self, spec: ProtocolSpec, coupling_layers: int = 2, hidden_mult: float = 0.25, enable_learnable_isometry: bool = False):
        super().__init__()
        self.spec = spec
        self.phy_dim = int(spec.phy_dim)
        self.proto_dim = int(spec.proto_dim)
        self.side_dim = int(spec.side_dim)
        self.mode = "identity"

        spec.validate()

        self._W_param = None
        if self.side_dim == 0:
            self.mode = "identity"
            if enable_learnable_isometry:
                self.mode = "isometry"
                W = torch.eye(self.phy_dim, device=spec.device, dtype=spec.dtype)
                self._W_param = nn.Parameter(W)
            return

        self.mode = "coupling"
        L = int(max(1, coupling_layers))
        self.L = L
        self.f_list = nn.ModuleList()
        self.g_list = nn.ModuleList()
        for _ in range(L):
            self.f_list.append(_CouplingMLP(self.side_dim, self.proto_dim, hidden_mult=hidden_mult))
            self.g_list.append(_CouplingMLP(self.proto_dim, self.side_dim, hidden_mult=hidden_mult))

        self.to(device=spec.device, dtype=spec.dtype)

    def _split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = x[..., :self.proto_dim]
        x1 = x[..., self.proto_dim:]
        return x0, x1

    def _merge(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return torch.cat([x0, x1], dim=-1)

    def encode_phy(self, x_phy: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        target_dtype = next(self.parameters()).dtype if len(list(self.parameters())) > 0 else self.spec.dtype
        x = x_phy.to(device=self.spec.device, dtype=target_dtype)

        if self.mode == "identity":
            return x, None
        if self.mode == "isometry":
            W = self._W_param
            return x @ W.T, None

        x0, x1 = self._split(x)
        for i in range(self.L):
            x0 = x0 + self.f_list[i](x1)
            x1 = x1 + self.g_list[i](x0)
        return x0, x1

    def decode_phy(self, x_proto: torch.Tensor, z_sideband: Optional[torch.Tensor]) -> torch.Tensor:
        x0 = x_proto.to(device=self.spec.device, dtype=self.spec.dtype)

        if self.mode == "identity":
            return x0
        if self.mode == "isometry":
            W = self._W_param
            return x0 @ W

        if not isinstance(z_sideband, torch.Tensor):
            x1 = torch.zeros(*x0.shape[:-1], self.side_dim, device=x0.device, dtype=x0.dtype)
        else:
            x1 = z_sideband.to(device=x0.device, dtype=x0.dtype)

        for i in reversed(range(self.L)):
            x1 = x1 - self.g_list[i](x0)
            x0 = x0 - self.f_list[i](x1)
        return self._merge(x0, x1)

@torch.jit.script
def smooth_force_bound_jit(
    force: torch.Tensor,
    force_clip: float,
    eps: float,
) -> torch.Tensor:
    if force_clip <= 0.0:
        return force

    norm = torch.sqrt((force * force).sum(dim=-1, keepdim=True) + eps)
    cap = max(force_clip, eps)
    sharpness = 16.0
    ratio = norm / cap
    scale = torch.pow(1.0 + torch.pow(ratio, sharpness), -1.0 / sharpness)
    return force * scale

@torch.jit.script
def scale_forces_scalar_clip_jit(
    F_cons: torch.Tensor,
    F_field: torch.Tensor,
    F_port: torch.Tensor,
    force_clip: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Only the learned actuator is bounded. Conservative field forces are left
    # untouched so the trajectory still follows F = -∇U without post-hoc force saturation.
    F_port = smooth_force_bound_jit(F_port, force_clip, eps)
    F_total = F_cons + F_field + F_port
    return F_total, F_field, F_port

@torch.jit.script
def jit_nose_hoover_step(
    q: torch.Tensor,
    p: torch.Tensor,
    xi: torch.Tensor,
    r: torch.Tensor,
    F_field: torch.Tensor,
    F_port: torch.Tensor,
    g_inv: torch.Tensor,
    dt: torch.Tensor,
    k: float,
    lam: float,
    target_temp: torch.Tensor, 
    eps: float,
    port_force_clip: float,
    max_exp_arg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dt_half = 0.5 * dt
    K0 = 0.5 * (p.pow(2) * g_inv).sum(dim=-1, keepdim=True)
    xi = xi + dt_half * (K0 - target_temp)

    if max_exp_arg > 0.0:
        xi_safe = max_exp_arg * torch.tanh(xi / max_exp_arg)
    else:
        xi_safe = xi
    scale_0 = torch.exp(-xi_safe * dt_half)

    F_cons0 = -k * (q - r) - lam * q
    F_total0, _, _ = scale_forces_scalar_clip_jit(F_cons0, F_field, F_port, port_force_clip, eps)

    p = p + dt_half * F_total0
    p = p * scale_0

    q = q + dt * (p * g_inv)

    F_cons1 = -k * (q - r) - lam * q
    F_total1, F_field1, F_port1 = scale_forces_scalar_clip_jit(F_cons1, F_field, F_port, port_force_clip, eps)

    p = p * scale_0
    p = p + dt_half * F_total1

    K1 = 0.5 * (p.pow(2) * g_inv).sum(dim=-1, keepdim=True)
    xi = xi + dt_half * (K1 - target_temp)

    v1 = p * g_inv
    power_port = (F_port1 * v1).sum(dim=-1).mean()

    V = 0.5 * k * (q - r).pow(2).mean() + 0.5 * lam * q.pow(2).mean()
    H_val = K1.mean() + V

    return q, p, xi, H_val, F_total1, power_port

@torch.jit.script
def jit_integrate_trajectory(
    q: torch.Tensor,
    p: torch.Tensor,
    xi: torch.Tensor,
    r: torch.Tensor,
    F_field: torch.Tensor,
    F_port: torch.Tensor,
    g_inv: torch.Tensor,
    dt_step: torch.Tensor,
    steps: int,
    k: float,
    lam: float,
    target_temp: torch.Tensor,
    eps: float,
    gamma_dyn: torch.Tensor,
    port_force_clip: float,
    max_exp_arg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    H_acc = torch.tensor(0.0, device=q.device, dtype=torch.float32)
    V_acc = torch.tensor(0.0, device=q.device, dtype=torch.float32)
    W_port_acc = torch.tensor(0.0, device=q.device, dtype=torch.float32)

    curr_q = q
    curr_p = p
    curr_xi = xi
    last_F = torch.zeros_like(q)

    zero = dt_step.new_zeros(())
    one = dt_step.new_ones(())

    gamma_pos = torch.clamp(gamma_dyn, min=0.0)
    safe_gamma = gamma_pos * dt_step
    safe_gamma_clip = torch.clamp(safe_gamma, max=max_exp_arg)
    damp = torch.exp(-safe_gamma_clip)
    damp = torch.where(safe_gamma > max_exp_arg, zero, damp)

    base = torch.clamp(one - damp * damp, min=0.0)
    noise_factor = torch.sqrt(base)
    
    # [Fix] 引入质量修正：noise ~ sqrt(M * T) = sqrt(T / g_inv)
    mass_factor = torch.rsqrt(g_inv + 1e-8) 
    
    # noise_scale_vec: [B*L, 1] * [1, D] -> Broadcast -> [B*L, D]
    # Use eps floor for sqrt input to avoid d/dx sqrt(x)|_{x=0} singularity.
    # Keep exact zero noise on non-positive temperature entries via where-mask.
    temp_pos = target_temp > 0.0
    temp_sqrt = torch.sqrt(target_temp.clamp_min(eps))
    temp_sqrt = torch.where(temp_pos, temp_sqrt, torch.zeros_like(temp_sqrt))
    noise_scale_vec = noise_factor * temp_sqrt * mass_factor
    

    for _ in range(steps):
        curr_q, curr_p, curr_xi, H_val, last_F, power_port = jit_nose_hoover_step(
            curr_q,
            curr_p,
            curr_xi,
            r,
            F_field,
            F_port,
            g_inv,
            dt_step,
            k,
            lam,
            target_temp, # 传入 Tensor
            eps,
            port_force_clip,
            max_exp_arg,
        )

        # --- [修改点3] 应用向量化噪声 ---
        noise = torch.randn_like(curr_p) * noise_scale_vec
        curr_p = curr_p * damp + noise
        # ---------------------------

        H_acc = H_acc + H_val.to(dtype=H_acc.dtype)

        diff = curr_q - r
        V_val = 0.5 * k * (diff * diff).sum(dim=-1) + 0.5 * lam * (curr_q * curr_q).sum(dim=-1)
        V_acc = V_acc + V_val.mean()

        W_port_acc = W_port_acc + (power_port.to(dtype=W_port_acc.dtype) * dt_step)

    step_float = float(steps)

    return (
        curr_q,
        curr_p,
        curr_xi,
        H_acc / step_float,
        last_F,
        V_acc / step_float,
        W_port_acc,
    )
