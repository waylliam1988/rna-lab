# Extracted verbatim by scripts/migrate_core.py; see docs/MIGRATION.md.
import math
import torch
from torch import nn

class ContinuousControlLayer(nn.Module):
    """
    Bounded continuous control stage.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 2.0,
        base_threshold: float = 1.0,
        refractory_period: int = 2,
        time_steps: int = 8,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.mem_norm = nn.LayerNorm(out_features)
        self.out_norm = nn.LayerNorm(out_features)

        self.tau = float(max(1.0, tau))
        self.refractory_period = int(max(1, refractory_period))
        self.time_steps = int(max(1, time_steps))

        thr = max(1e-3, float(base_threshold))
        self.register_buffer("threshold", torch.ones(out_features) * thr)
        self.register_buffer("tau_vec", torch.ones(out_features) * self.tau)
        self.register_buffer("ref_vec", torch.ones(out_features) * float(self.refractory_period))
        self.register_buffer("running_activity", torch.zeros(out_features))

        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x,
        v_mem=None,
        ref_counter=None,
        time_steps=8,
        current_alpha=1.0,
        force_grad=False,
    ):
        batch_size = x.shape[0]
        out_dim = self.fc.out_features

        if x.dim() == 2:
            x_seq = x.unsqueeze(1).repeat(1, time_steps, 1)
        else:
            x_seq = x
            time_steps = x.shape[1]

        if v_mem is None:
            v_mem = torch.zeros(batch_size, out_dim, device=x.device, dtype=x_seq.dtype)
        if ref_counter is None:
            ref_counter = torch.zeros(batch_size, out_dim, device=x.device, dtype=x_seq.dtype)

        tau_vec_local = torch.clamp(self.tau_vec.to(device=x.device, dtype=x_seq.dtype), min=1.0)
        ref_vec_local = torch.clamp(self.ref_vec.to(device=x.device, dtype=x_seq.dtype), min=1.0)
        mem_decay = torch.exp(-1.0 / tau_vec_local)
        ref_decay = torch.exp(-1.0 / ref_vec_local)
        thr = torch.clamp(self.threshold.to(device=x.device, dtype=x_seq.dtype), min=1e-3)

        if isinstance(current_alpha, torch.Tensor):
            alpha_t = current_alpha.to(device=x.device, dtype=x_seq.dtype)
        else:
            alpha_t = x_seq.new_tensor(float(current_alpha))

        current_input = self.fc(x_seq)
        activity_seq = []
        total_activity = torch.zeros(batch_size, out_dim, device=x.device, dtype=x_seq.dtype)

        for t in range(time_steps):
            drive = current_input[:, t, :]
            v_mem = mem_decay * v_mem + (1.0 - mem_decay) * drive
            mixed_state = self.mem_norm(v_mem + 0.25 * ref_counter)
            raw = (alpha_t * mixed_state) / thr
            act = torch.tanh(raw)
            ref_counter = ref_decay * ref_counter + (1.0 - ref_decay) * act
            act = torch.tanh(self.out_norm(act + 0.10 * ref_counter))
            activity_seq.append(act)
            total_activity = total_activity + act.abs()

        activity = torch.stack(activity_seq, dim=1)

        if self.training or force_grad:
            with torch.no_grad():
                mean_activity = total_activity.mean(dim=0) / float(time_steps)
                self.running_activity.lerp_(mean_activity.detach(), 0.05)

        return activity, v_mem, ref_counter
