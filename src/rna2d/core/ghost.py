"""Explicit-state adaptation of the old continuous Ghost cortex.

Preserves the two-stage dynamics, trace encoder and RWKV recurrence. This
module is not yet coupled to the RNA pair baseline, and makes no physical
time or learned-world-model claim.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .control import ContinuousControlLayer
from .memory import RWKVCellLite


@dataclass(frozen=True)
class GhostSettings:
    input_dim: int = 64
    hidden_dim: int = 64
    inner_steps: int = 8
    rate_bound: float = 8.0
    key_mode: str = 'softplus_zero'
    pair_state_cap: float = 1.0
    pair_state_cap_mode: str = 'soft'
    prediction_dim: int = 5


class GhostDynamics(nn.Module):
    STATE_KEYS = ('control_stage1_state', 'control_stage1_aux',
                  'control_stage2_state', 'control_stage2_aux',
                  'rwkv_num_state', 'rwkv_den_state')

    def __init__(self, settings=None):
        super().__init__()
        self.settings = settings or GhostSettings()
        c = self.settings
        if min(c.input_dim, c.hidden_dim, c.inner_steps, c.prediction_dim) < 1:
            raise ValueError('Dimensions and inner_steps must be positive')
        if not math.isfinite(c.rate_bound) or c.rate_bound <= 0:
            raise ValueError('rate_bound must be finite and positive')
        self.control_stage1 = ContinuousControlLayer(c.input_dim, c.hidden_dim, tau=2.0)
        self.control_stage2 = ContinuousControlLayer(c.hidden_dim, c.hidden_dim, tau=2.0)
        self.trajectory_encoder = nn.Sequential(nn.Flatten(), nn.Linear(c.inner_steps * c.hidden_dim, c.hidden_dim),
                                                nn.LayerNorm(c.hidden_dim), nn.SiLU())
        nn.init.zeros_(self.trajectory_encoder[1].bias)
        self.rwkv_memory = RWKVCellLite(c.hidden_dim, c.hidden_dim, key_mode=c.key_mode,
                                       pair_state_cap=c.pair_state_cap, pair_state_cap_mode=c.pair_state_cap_mode)
        # Generic placeholder; prediction targets must be defined before training it.
        self.prediction_head = nn.Linear(c.hidden_dim, c.prediction_dim)
        nn.init.zeros_(self.prediction_head.weight)
        nn.init.zeros_(self.prediction_head.bias)

    def init_state(self, batch_size, device=None):
        weight = self.control_stage1.fc.weight
        return {key: torch.zeros(batch_size, self.settings.hidden_dim,
                                 device=device or weight.device, dtype=weight.dtype) for key in self.STATE_KEYS}

    @staticmethod
    def detach_state(state):
        return {key: value.detach() for key, value in state.items()}

    def forward(self, x, state=None, dt=0.1, error_current=None, alpha_scale=1.0):
        c = self.settings
        if x.ndim != 2 or x.shape[1] != c.input_dim:
            raise ValueError('Expected [batch, input_dim]')
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError('dt must be finite and positive (algorithmic scale, not seconds)')
        state = self.init_state(x.shape[0], x.device) if state is None else state
        if set(state) != set(self.STATE_KEYS):
            raise ValueError('Invalid state keys')
        for value in state.values():
            if value.shape != (x.shape[0], c.hidden_dim) or value.device != x.device or value.dtype != x.dtype:
                raise ValueError('State shape/device/dtype mismatch')
        drive = x if error_current is None else x + 4.0 * torch.tanh(error_current)
        alpha = x.new_tensor(2.0) * torch.as_tensor(alpha_scale, device=x.device, dtype=x.dtype).mean().clamp(0.5, 2.0)
        phase = torch.linspace(0.0, 1.0, c.inner_steps, device=x.device, dtype=x.dtype).view(1, c.inner_steps, 1)
        drive = (drive * (5.0 * dt)).unsqueeze(1) * (1.0 + 0.05 * torch.cos(2.0 * math.pi * phase))
        activity1, mem1, aux1 = self.control_stage1(drive, state['control_stage1_state'], state['control_stage1_aux'],
                                                   current_alpha=alpha)
        activity2, mem2, aux2 = self.control_stage2(activity1, state['control_stage2_state'], state['control_stage2_aux'],
                                                   current_alpha=alpha)
        encoded = self.trajectory_encoder(activity2) - self.trajectory_encoder(torch.zeros_like(activity2))
        rate, (num, den) = self.rwkv_memory(encoded, (state['rwkv_num_state'], state['rwkv_den_state']))
        rate = c.rate_bound * torch.tanh(rate / c.rate_bound)
        next_state = dict(zip(self.STATE_KEYS, (mem1, aux1, mem2, aux2, num, den)))
        return rate, next_state, self.prediction_head(rate)
