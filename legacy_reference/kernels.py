# kernels.py
# -*- coding: utf-8 -*-
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# === 第三方库导入 ===
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 本地模块导入 ===
from config import Config


# -----------------------------
# numeric helpers
# -----------------------------

def py_eps() -> float:
    return float(Config.py_eps())


def eps_of(dtype: torch.dtype = None) -> float:
    return float(Config.eps(dtype))


def max_exp_arg_of(dtype: torch.dtype = None, safety: float = 0.95) -> float:
    return float(Config.max_exp_arg(dtype) * float(safety))


def tail_prob_default(n: int) -> float:
    n = max(1, int(n))
    return float(1.0 / float(n * n))


def alpha_from_tau(dt: float, tau: float) -> float:
    if tau is None or tau <= 0:
        return 1.0
    return 1.0 - math.exp(-float(dt) / float(tau))


def rms(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return torch.sqrt(torch.mean(x * x) + Config.eps_like(x))


# -----------------------------
# stats
# -----------------------------

class EMARMS(nn.Module):
    def __init__(self, dt: float = 1.0, tau: float = 64.0):
        super().__init__()
        self.dt = float(dt)
        self.tau = float(max(float(tau), py_eps()))
        self.alpha = float(alpha_from_tau(self.dt, self.tau))
        self.register_buffer("m2", torch.tensor(0.0))
        self.register_buffer("step", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return torch.sqrt(self.m2 + 1e-6)
        x = x.detach()
        if self.m2.device != x.device:
            self.m2 = self.m2.to(x.device)
        v = torch.mean(x.float() * x.float())
        self.step += 1.0
        self.m2.mul_(1.0 - self.alpha).add_(self.alpha * v)
        return torch.sqrt(self.m2 + self.m2.new_tensor(eps_of(self.m2.dtype)))

    def forward(self, x):
        return self.update(x)


class EMAMean(nn.Module):
    def __init__(self, tau: float = 128.0, init: float = 0.0):
        super().__init__()
        self.tau = float(max(1.0, tau))
        self.register_buffer("_m", torch.tensor(float(init)))

    def alpha(self, dt: float = 1.0) -> float:
        return float(1.0 - math.exp(-float(dt) / self.tau))

    @torch.no_grad()
    def update(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        if self._m.device != x.device:
            self._m = self._m.to(x.device)
        a = self.alpha(dt)
        val = x.detach().float().mean()
        val = torch.nan_to_num(val, nan=0.0)
        self._m.mul_(1.0 - a).add_(a * val)
        return self._m

    def value(self) -> float:
        return float(self._m.item())


class AdaptiveGain(nn.Module):
    def __init__(
        self,
        target: float,
        tau: float = 256.0,
        gain_init: float = 1.0,
        gain_min: float = 1e-4,
        gain_max: float = 100.0,
        lr: float = 1.0,
    ):
        super().__init__()
        self.target = float(max(1e-12, target))
        self.tau = float(max(1.0, tau))
        self.lr = float(lr)
        self.gain_min = float(max(1e-9, gain_min))
        self.gain_max = float(max(self.gain_min * 2.0, gain_max))
        self._mean = EMAMean(tau=max(4.0, tau / 4.0), init=target)
        safe_init = max(gain_init, gain_min)
        self.register_buffer("_log_gain", torch.tensor(float(math.log(safe_init))))

    def _clamp_log_gain_(self):
        lo = math.log(self.gain_min)
        hi = math.log(self.gain_max)
        self._log_gain.clamp_(min=lo, max=hi)

    def gain(self) -> torch.Tensor:
        self._clamp_log_gain_()
        return torch.exp(self._log_gain)

    @torch.no_grad()
    def step(self, measured: torch.Tensor, dt: float = 1.0):
        m = self._mean.update(measured, dt=dt)
        current_gain = self.gain()
        prediction = m * current_gain
        err = (prediction - self.target) / self.target
        a = float(1.0 - math.exp(-float(dt) / self.tau))
        step_val = torch.clamp(a * self.lr * err, -0.1, 0.1)
        self._log_gain.sub_(step_val)
        self._clamp_log_gain_()


# -----------------------------
# SNN core
# -----------------------------

def z_from_tail_prob(tail_prob: float) -> float:
    floor = Config.py_eps()
    p = max(min(float(tail_prob), 1.0 - floor), floor)
    t = torch.tensor(p, dtype=torch.float32)

    if hasattr(torch.special, "erfcinv"):
        val = torch.special.erfcinv(t)
    else:
        val = torch.special.erfinv(1.0 - t)

    z = math.sqrt(2.0) * float(val.item())
    return z


@torch.jit.script
def compute_trace_jit(spikes: torch.Tensor, decay: float) -> torch.Tensor:
    batch, steps, dim = spikes.shape
    trace = torch.zeros_like(spikes)
    curr = torch.zeros((batch, dim), device=spikes.device, dtype=spikes.dtype)

    for t in range(steps):
        curr = curr * decay + spikes[:, t, :]
        trace[:, t, :] = curr

    return trace


class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha * torch.exp(-alpha * torch.abs(x))
        return grad_output * grad, None


def jit_lif_forward(
    x_seq: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    v_mem: torch.Tensor,
    threshold: torch.Tensor,
    ref_counter: torch.Tensor,
    decay: torch.Tensor,
    current_alpha: float,
    ref_period: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    linear_out = torch.matmul(x_seq, weight.t()) + bias
    
    batch_size = x_seq.shape[0]
    time_steps = x_seq.shape[1]
    out_dim = weight.shape[0]

    spike_list: List[torch.Tensor] = []
    total_spikes = torch.zeros(batch_size, out_dim, device=x_seq.device, dtype=x_seq.dtype)

    for t in range(time_steps):
        i_t = linear_out[:, t, :]
        
        active_mask = (ref_counter <= 0).float()
        v_mem = (v_mem * decay + i_t) * active_mask
        spike = (v_mem - threshold > 0).float()
        v_mem = v_mem - (spike * threshold)
        ref_counter = torch.where(spike > 0, ref_period, ref_counter - 1.0)
        
        spike_list.append(spike)
        total_spikes += spike

    spikes_out = torch.stack(spike_list, dim=1)
    return spikes_out, v_mem, ref_counter, total_spikes


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


class SpikingLIFLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        tau=2.0,
        base_threshold=1.0,
        stdp_config=None,
        refractory_period=2,
        time_steps: int = 8,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

        self.tau = float(max(1.0, tau))
        self.refractory_period = int(max(1, refractory_period))
        self.time_steps = int(max(1, time_steps))

        self.register_buffer("threshold", torch.ones(out_features) * float(base_threshold))

        target_spikes_per_window = float(getattr(Config, "GHOST_ACTIVITY_TARGET_PER_WINDOW", 0.0))
        if target_spikes_per_window <= 0.0:
            target_spikes_per_window = math.sqrt(float(self.time_steps)) * 0.5
        target_rate = float(target_spikes_per_window / float(self.time_steps))
        self.register_buffer("target_rate", torch.tensor(target_rate))
        self.register_buffer("running_rate", torch.ones(out_features) * target_rate)

        homeostasis_tau_steps = float(getattr(Config, "GHOST_ACTIVITY_HOMEOSTASIS_TAU_STEPS", 256.0))
        self.homeo_mean = EMAMean(tau=homeostasis_tau_steps, init=target_rate)

        hetero_cv = float(getattr(Config, "GHOST_ACTIVITY_HETEROGENEITY_CV", 0.2))
        hetero_cv = float(max(0.0, min(0.8, hetero_cv)))
        if hetero_cv > 0:
            sigma = math.sqrt(math.log(1.0 + hetero_cv ** 2))
            mu = math.log(self.tau) - 0.5 * sigma * sigma
            tau_samples = torch.exp(mu + sigma * torch.randn(out_features))
            self.register_buffer("tau_vec", tau_samples)
        else:
            self.register_buffer("tau_vec", torch.ones(out_features) * self.tau)

        if hetero_cv > 0:
            sigma = math.sqrt(math.log(1.0 + hetero_cv ** 2))
            mu = math.log(float(self.refractory_period)) - 0.5 * sigma * sigma
            ref_samples = torch.exp(mu + sigma * torch.randn(out_features))
            self.register_buffer("ref_vec", ref_samples)
        else:
            self.register_buffer("ref_vec", torch.ones(out_features) * float(self.refractory_period))

        self.spike_fn = SurrogateHeaviside.apply
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

        if stdp_config is None:
            stdp_config = getattr(Config, "GHOST_ACTIVITY_STDP", None)
        get_val = lambda d, k, default: d.get(k, default) if isinstance(d, dict) else default
        self.learning_rate_pos = float(get_val(stdp_config, "stdp_lr_pos", 0.005))
        self.learning_rate_neg = float(get_val(stdp_config, "stdp_lr_neg", 0.004))

        tau_trace = float(getattr(Config, "GHOST_ACTIVITY_TRACE_TAU_STEPS", 8.0))
        tau_trace = float(max(1.0, tau_trace))
        self.trace_decay = float(math.exp(-1.0 / tau_trace))

        self._v_rms = EMARMS(dt=1.0, tau=float(getattr(Config, "GHOST_ACTIVITY_VRMS_TAU", 64.0)))
        self._trace_rms = EMARMS(dt=1.0, tau=float(getattr(Config, "GHOST_ACTIVITY_TRACE_RMS_TAU", 64.0)))
        self._mod_rms = EMARMS(dt=1.0, tau=float(getattr(Config, "GHOST_ACTIVITY_MOD_RMS_TAU", 64.0)))
        self.tau_min_steps = float(getattr(Config, "GHOST_ACTIVITY_TAU_MIN_STEPS", 1.0))

    def forward(self, x, v_mem=None, ref_counter=None, time_steps=8, current_alpha=2.0, force_grad=False):
        batch_size = x.shape[0]
        out_dim = self.fc.out_features

        if x.dim() == 2:
            x_seq = x.unsqueeze(1).repeat(1, time_steps, 1)
        else:
            x_seq = x
            time_steps = x.shape[1]

        if v_mem is None:
            v_mem = torch.zeros(batch_size, out_dim, device=x.device)
        if ref_counter is None:
            ref_counter = torch.zeros(batch_size, out_dim, device=x.device)

        tau_vec_local = torch.clamp(self.tau_vec.to(x.device), min=float(self.tau_min_steps))
        decay_vec = 1.0 - (1.0 / tau_vec_local)
        ref_vec_local = self.ref_vec.to(x.device)

        if self.training or force_grad:
            current_input = self.fc(x_seq)
            spike_train = []
            total_spikes = torch.zeros(batch_size, out_dim, device=x.device)
            calc_threshold = self.threshold.clone().detach()

            for t in range(time_steps):
                i_t = current_input[:, t, :]
                active_mask = (ref_counter <= 0).float()
                v_mem = (v_mem * decay_vec + i_t) * active_mask
                spike = self.spike_fn(v_mem - calc_threshold, current_alpha)
                v_mem = v_mem - (spike * calc_threshold)
                ref_counter = torch.where(spike > 0, ref_vec_local, ref_counter - 1)
                spike_train.append(spike)
                total_spikes += spike

            if self.training:
                with torch.no_grad():
                    current_batch_rate = total_spikes.mean(dim=0) / float(time_steps)
                    self.homeo_mean.update(current_batch_rate)
                    self.running_rate.copy_(self.homeo_mean._m)
                    alpha = self.homeo_mean.alpha(dt=1.0)
                    delta = alpha * (self.running_rate - self.target_rate)
                    gain = float(getattr(Config, "GHOST_ACTIVITY_HOMEOSTASIS_GAIN", 0.5))
                    self.threshold.add_(delta * gain)

                    v_scale = float(self._v_rms.update(v_mem.detach()).item() + py_eps())
                    tail_prob = float(getattr(Config, "GHOST_ACTIVITY_THRESH_TAIL_PROB", 0.0))
                    if tail_prob <= 0.0:
                        tail_prob = tail_prob_default(self.fc.out_features)
                    z = float(z_from_tail_prob(tail_prob))

                    thr_min = float(getattr(Config, "GHOST_ACTIVITY_THRESH_MIN", 0.0))
                    if thr_min <= 0.0:
                        thr_min = max(py_eps(), 0.25 * v_scale)
                    thr_max = float(getattr(Config, "GHOST_ACTIVITY_THRESH_MAX", 0.0))
                    if thr_max <= 0.0:
                        thr_max = max(thr_min, z * v_scale)

                    self.threshold.clamp_(min=thr_min, max=thr_max)

            return torch.stack(spike_train, dim=1), v_mem, ref_counter

        target_dtype = x.dtype
        weight = self.fc.weight.to(dtype=target_dtype)
        bias = self.fc.bias if self.fc.bias is not None else torch.zeros(out_dim, device=x.device, dtype=target_dtype)
        bias = bias.to(dtype=target_dtype)

        if v_mem.dtype != target_dtype:
            v_mem = v_mem.to(dtype=target_dtype)
        if ref_counter.dtype != target_dtype:
            ref_counter = ref_counter.to(dtype=target_dtype)

        threshold = self.threshold.to(dtype=target_dtype)
        decay = decay_vec.to(dtype=target_dtype)
        ref_period = ref_vec_local.to(dtype=target_dtype)

        jit_alpha = current_alpha.item() if isinstance(current_alpha, torch.Tensor) else float(current_alpha)

        spikes, v_mem, ref_counter, _total_spikes = jit_lif_forward(
            x_seq,
            weight,
            bias,
            v_mem,
            threshold,
            ref_counter,
            decay,
            jit_alpha,
            ref_period,
        )
        return spikes, v_mem, ref_counter

    def process_stdp(self, pre_spikes, post_spikes, modulation=1.0):
        if pre_spikes is None or post_spikes is None:
            return
        try:
            pre_spikes = pre_spikes.float()
            post_spikes = post_spikes.float()
            batch_size = pre_spikes.shape[0]
            pre_trace = compute_trace_jit(pre_spikes, self.trace_decay)
            post_trace = compute_trace_jit(post_spikes, self.trace_decay)

            pre_trace_masked = pre_trace
            post_trace_masked = post_trace

            ltp = torch.einsum("bti,bto->oi", pre_trace_masked, post_spikes)
            ltd = torch.einsum("bti,bto->oi", pre_spikes, post_trace_masked)

            ltp = ltp / float(batch_size)
            ltd = ltd / float(batch_size)

            mod = float(modulation)
            self.fc.weight.data.add_(self.learning_rate_pos * mod * ltp)
            self.fc.weight.data.sub_(self.learning_rate_neg * mod * ltd)
        except Exception:
            pass


# -----------------------------
# protocol link
# -----------------------------

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


# -----------------------------
# io
# -----------------------------

_ATOMIC_REPLACE_SUPPORTED = True


def atomic_torch_save(obj, path: str, *, replace_retries: int = 8, retry_delay: float = 0.10):
    global _ATOMIC_REPLACE_SUPPORTED
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not _ATOMIC_REPLACE_SUPPORTED:
        with open(path_obj, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        return

    fd, tmp_path = tempfile.mkstemp(prefix=path_obj.name + ".", suffix=".tmp", dir=path_obj.parent)
    fd_open = True

    try:
        with os.fdopen(fd, "wb") as f:
            fd_open = False
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())

        retries = max(1, int(replace_retries))
        last_replace_exc = None
        for attempt in range(retries):
            try:
                os.replace(tmp_path, str(path_obj))
                return
            except PermissionError as e:
                last_replace_exc = e
                if attempt + 1 >= retries:
                    break
                time.sleep(float(retry_delay) * float(attempt + 1))

        # Fallback for environments where rename/replace is blocked or heavily locked.
        _ATOMIC_REPLACE_SUPPORTED = False
        with open(path_obj, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        if last_replace_exc is not None:
            return
    except Exception:
        if fd_open:
            try:
                os.close(fd)
            except OSError:
                pass
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def safe_torch_load(path: str, map_location=None, *, expect_type=None, max_mb: int = 2048):
    abspath = os.path.abspath(path)
    project_root = os.path.dirname(os.path.abspath(__file__))

    if ".." in path:
        print(f"[Security] Suspicious path: {path}")

    try:
        if os.path.commonpath([project_root, abspath]) != project_root:
            raise ValueError(f"Unsafe path detected: {abspath} (escapes root {project_root})")
    except ValueError:
        raise ValueError(f"Invalid path check: {abspath} (system root mismatch)")

    max_bytes = int(max_mb) * 1024 * 1024
    if os.path.exists(abspath):
        size = os.path.getsize(abspath)
        if size > max_bytes:
            raise RuntimeError(f"File too large: {size} bytes > {max_bytes} bytes")

    try:
        obj = torch.load(abspath, map_location=map_location, weights_only=True)
    except TypeError:
        obj = torch.load(abspath, map_location=map_location)
    except Exception as e:
        raise RuntimeError(f"Safe load failed: {e}")

    if expect_type is not None and not isinstance(obj, expect_type):
        raise TypeError(f"Unexpected checkpoint type: {type(obj)} expect {expect_type}")

    return obj


# -----------------------------
# physics integrator
# -----------------------------

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


# -----------------------------
# RNA/Bio upgrade: Adaptive Riemannian metric + force-scale diagnostics
# (ported & simplified from kernels1.py HamiltonianSystem)
# -----------------------------
class RiemannianMetricController(nn.Module):
    """
    Maintains an EMA estimate of per-dimension force RMS and provides:
      - g_inv (inverse diagonal metric / mass preconditioner)
      - force_clip (diagnostic/suggested force scale; no longer applied to F_field)
      - max_exp_arg (safe clamp for exp() / xi)
    Designed to be called under torch.no_grad().
    """
    def __init__(
        self,
        dim: int,
        beta: float = 0.99,
        tail_prob: float = 0.0,
        ginv_clip: Tuple[float, float] = (0.05, 20.0),
        force_clip_floor: float = 0.25,
        force_clip_cap: float = 50.0,
        curvature_mass_gain: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = int(dim)
        self.beta = float(beta)
        self.tail_prob = float(tail_prob)
        self.ginv_clip = (float(ginv_clip[0]), float(ginv_clip[1]))
        self.force_clip_floor = float(force_clip_floor)
        self.force_clip_cap = float(force_clip_cap)
        self.curvature_mass_gain = float(curvature_mass_gain)

        m2 = torch.ones(self.dim, device=device, dtype=dtype)
        self.register_buffer("_m2", m2)  # EMA of E[F^2] per dim

        self.register_buffer("_g_inv", torch.ones(1, self.dim, device=device, dtype=dtype))
        self.register_buffer("_force_scale", torch.tensor(1.0, device=device, dtype=dtype))
        self.register_buffer("_force_clip", torch.tensor(self.force_clip_cap, device=device, dtype=dtype))

        mea = float(getattr(Config, "PHYS_MAX_EXP_ARG", 0.0))
        self.max_exp_arg = float(mea) if mea and mea > 0.0 else float(max_exp_arg_of(dtype, safety=0.95))

    @torch.no_grad()
    def update(
        self,
        force: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        curvature_mean: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float, float, float]:
        """
        force: [N, D]
        mask:  [N, 1] in {0,1} (optional) to ignore padded residues
        curvature_mean: optional scalar, slows down when curvature high
        returns: (g_inv[1,D], force_clip_scalar, max_exp_arg_scalar, force_scale_scalar)
        """
        if force is None:
            return self._g_inv, float(self._force_clip.item()), float(self.max_exp_arg), float(self._force_scale.item())

        f = torch.nan_to_num(force.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
        f2 = f.pow(2)

        if mask is not None:
            m = torch.nan_to_num(mask.detach().float(), nan=0.0).clamp(0.0, 1.0)
            denom = m.sum().clamp_min(py_eps())
            m2 = (f2 * m).sum(dim=0) / denom
        else:
            m2 = f2.mean(dim=0)

        b = float(min(max(self.beta, 0.0), 0.999999))
        self._m2.mul_(b).add_((1.0 - b) * m2)

        rms = torch.sqrt(self._m2 + py_eps())
        g_inv = 1.0 / rms
        g_inv = torch.nan_to_num(g_inv, nan=1.0, posinf=1.0, neginf=1.0)
        g_inv = torch.clamp(g_inv, self.ginv_clip[0], self.ginv_clip[1])

        if curvature_mean is not None and self.curvature_mass_gain > 0.0:
            cm = float(max(0.0, curvature_mean))
            g_inv = g_inv / (1.0 + self.curvature_mass_gain * cm)

        self._g_inv = g_inv.view(1, -1).to(self._g_inv.dtype)

        force_scale = float(torch.sqrt(self._m2.mean() + py_eps()).item())
        self._force_scale.fill_(force_scale)

        tp = self.tail_prob if self.tail_prob and self.tail_prob > 0.0 else float(getattr(Config, "PHYS_FORCE_TAIL_PROB", 0.0))
        if tp <= 0.0:
            tp = tail_prob_default(self.dim)

        z = float(z_from_tail_prob(tp))
        clip = max(self.force_clip_floor, z * force_scale)
        clip = min(clip, self.force_clip_cap)
        self._force_clip.fill_(clip)

        return self._g_inv, float(clip), float(self.max_exp_arg), float(force_scale)


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
