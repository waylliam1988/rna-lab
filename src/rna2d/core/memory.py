# Extracted verbatim by scripts/migrate_core.py; see docs/MIGRATION.md.
import math
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

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
