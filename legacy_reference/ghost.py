# ghost.py
# -*- coding: utf-8 -*-
import math
from typing import Dict, Optional

# === 第三方库导入 ===
import torch
import torch.nn as nn

# === 本地模块导入 ===
from config import Config
from kernels import ProtocolSpec, RWKVCellLite, USBLink


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

# =====================================================================
# Neural Cortex (Continuous Control Core)
# =====================================================================
class NeuralCortex(nn.Module):
    """
    Continuous control cortex for RNA-only training.

    Canonical semantics:
    - `stage1` / `stage2` are continuous bounded control stages
    - cached traces are continuous activities, not binary spikes
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.time_steps = int(getattr(Config, "GHOST_CORTEX_TIME_STEPS", 8))
        self.rate_bound = float(getattr(Config, "GHOST_CONTROL_ACTIVITY_BOUND", 8.0))

        self.control_stage1 = ContinuousControlLayer(
            self.input_dim,
            self.hidden_dim,
            tau=2.0,
            base_threshold=1.0,
        )
        self.control_stage2 = ContinuousControlLayer(
            self.hidden_dim,
            self.hidden_dim,
            tau=2.0,
            base_threshold=1.0,
        )

        # Trajectory encoder: flatten the continuous control trace over the inner
        # rollout window so the controller can condition on local temporal shape.
        flatten_dim = self.time_steps * self.hidden_dim
        self.trajectory_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU()
        )
        if self.trajectory_encoder[1].bias is not None:
            nn.init.zeros_(self.trajectory_encoder[1].bias)

        self.rwkv_memory = RWKVCellLite(
            self.hidden_dim,
            self.hidden_dim,
            key_mode=str(getattr(Config, "GHOST_RWKV_KEY_MODE", "softplus_zero")).lower(),
            pair_state_cap=float(getattr(Config, "GHOST_RWKV_PAIR_STATE_CAP", 1.0)),
            pair_state_cap_mode=str(getattr(Config, "GHOST_RWKV_PAIR_STATE_CAP_MODE", "soft")).lower(),
        )

        # Predict [Rg, Contact_Mean, Contact_Max, Force_RMS, Energy].
        self.phys_hallucination = nn.Linear(self.hidden_dim, 5)
        nn.init.zeros_(self.phys_hallucination.weight)
        nn.init.zeros_(self.phys_hallucination.bias)

        self.last_drive_input = None
        self.last_stage1_output = None
        self.last_stage2_drive = None
        self.last_stage2_output = None
        self.control_stage1_state = None
        self.control_stage1_aux = None
        self.control_stage2_state = None
        self.control_stage2_aux = None

    @property
    def device(self):
        return self.control_stage1.fc.weight.device

    @property
    def stage1(self):
        return self.control_stage1

    @property
    def stage2(self):
        return self.control_stage2

    @property
    def last_input(self):
        return self.last_drive_input

    @property
    def last_stage1_activity(self):
        return self.last_stage1_output

    @property
    def last_stage2_input(self):
        return self.last_stage2_drive

    @property
    def last_stage2_activity(self):
        return self.last_stage2_output

    def _safe_softsign(self, x_scaled: torch.Tensor) -> torch.Tensor:
        x_scaled = torch.nan_to_num(x_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        probs = x_scaled / (1.0 + x_scaled.abs())
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=-1.0)
        return probs

    def _pdm_pulse_train(self, p: torch.Tensor, time_steps: int, rounding: bool = True) -> torch.Tensor:
        p = torch.clamp(p, 0.0, 1.0).float()
        t = torch.arange(time_steps, device=p.device, dtype=torch.float32).view(1, time_steps, 1)
        delta = 0.5 if rounding else 0.0
        cum_next = torch.floor((t + 1.0) * p.unsqueeze(1) + delta)
        cum_prev = torch.floor(t * p.unsqueeze(1) + delta)
        pulses = (cum_next - cum_prev).clamp(0.0, 1.0)
        return pulses

    def _encode_activity_drive(self, x, time_steps, gain=1.0, mode=None, return_preact=False):
        if mode is None:
            mode = (
                getattr(Config, "GHOST_ACTIVITY_ENCODER_MODE_TRAIN", "sample")
                if self.training
                else getattr(Config, "GHOST_ACTIVITY_ENCODER_MODE_INFER", "rate")
            )

        x_scaled = x * gain
        probs_signed = self._safe_softsign(x_scaled)

        probs_pos = torch.clamp(probs_signed, min=0.0)
        probs_neg = torch.abs(torch.clamp(probs_signed, max=0.0))

        probs_pos_t = probs_pos.unsqueeze(1).repeat(1, time_steps, 1)
        probs_neg_t = probs_neg.unsqueeze(1).repeat(1, time_steps, 1)

        if mode == "sample":
            event_pos = torch.bernoulli(torch.clamp(probs_pos_t, 0.0, 1.0))
            event_neg = torch.bernoulli(torch.clamp(probs_neg_t, 0.0, 1.0))
            drive = event_pos - event_neg
            preact = drive
        elif mode == "rate":
            drive = torch.clamp(probs_pos_t, 0.0, 1.0) - torch.clamp(probs_neg_t, 0.0, 1.0)
            enabled = bool(getattr(Config, "GHOST_ACTIVITY_PRETRACE_ENABLED_WHEN_RATE", True))
            if not enabled:
                preact = drive
            else:
                pre_mode = str(getattr(Config, "GHOST_ACTIVITY_PRETRACE_MODE_WHEN_RATE", "pdm")).lower()
                if pre_mode == "drive":
                    preact = drive
                elif pre_mode == "hard":
                    thr = float(getattr(Config, "GHOST_ACTIVITY_DETERMINISTIC_THRESHOLD", 0.5))
                    event_pos = (probs_pos_t > thr).float()
                    event_neg = (probs_neg_t > thr).float()
                    preact = event_pos - event_neg
                elif pre_mode == "pdm":
                    rounding = bool(getattr(Config, "GHOST_ACTIVITY_PDM_ROUNDING", True))
                    pulse_pos = self._pdm_pulse_train(probs_pos, time_steps, rounding=rounding).to(drive.device)
                    pulse_neg = self._pdm_pulse_train(probs_neg, time_steps, rounding=rounding).to(drive.device)
                    preact = pulse_pos - pulse_neg
                else:
                    raise ValueError(f"[NeuralCortex] Unknown STDP pre mode when rate: {pre_mode}")
        elif mode == "hard":
            thr = float(getattr(Config, "GHOST_ACTIVITY_DETERMINISTIC_THRESHOLD", 0.5))
            event_pos = (probs_pos_t > thr).float()
            event_neg = (probs_neg_t > thr).float()
            drive = event_pos - event_neg
            preact = drive
        else:
            raise ValueError(f"[NeuralCortex] Unknown encoder mode: {mode}")

        if return_preact:
            return drive, preact
        return drive

    def forward(self, x, current_state=None, dt=0.1, record_state=True, use_grad=False, error_current=None, alpha_scale=None):
        return self.forward_dynamic(
            x,
            current_state=current_state,
            dt=dt,
            record_state=record_state,
            use_grad=use_grad,
            error_current=error_current,
            alpha_scale=alpha_scale,
        )

    def forward_dynamic(self, x, current_state=None, dt=0.1, record_state=True, use_grad=False, error_current=None, alpha_scale=None):
        x = x.float()
        time_steps = self.time_steps

        if current_state is not None:
            self.control_stage1_state = current_state.get("control_stage1_state")
            self.control_stage1_aux = current_state.get("control_stage1_aux")
            self.control_stage2_state = current_state.get("control_stage2_state")
            self.control_stage2_aux = current_state.get("control_stage2_aux")
        else:
            self.control_stage1_state = None
            self.control_stage1_aux = None
            self.control_stage2_state = None
            self.control_stage2_aux = None

        rwkv_state = None
        if current_state is not None:
            if "rwkv_num_state" in current_state and "rwkv_den_state" in current_state:
                rwkv_state = (current_state["rwkv_num_state"], current_state["rwkv_den_state"])

        if error_current is not None:
            safe_error = 2.0 * torch.tanh(error_current)
            input_with_pain = x + (safe_error * 2.0)
        else:
            input_with_pain = x

        current_alpha = torch.tensor(2.0, device=x.device, dtype=x.dtype)
        if alpha_scale is not None:
            try:
                if isinstance(alpha_scale, torch.Tensor):
                    a = alpha_scale.mean()
                else:
                    a = torch.tensor(float(alpha_scale), device=x.device, dtype=x.dtype)
                a = torch.clamp(a, 0.5, 2.0)
                current_alpha = current_alpha * a
            except Exception:
                pass

        if isinstance(dt, torch.Tensor):
            dt_t = dt.to(device=x.device, dtype=x.dtype)
        else:
            dt_t = torch.tensor(float(dt), device=x.device, dtype=x.dtype)
        scaled_input = input_with_pain * (5.0 * dt_t)
        phase = torch.linspace(0.0, 1.0, self.time_steps, device=x.device, dtype=x.dtype).view(1, self.time_steps, 1)
        envelope = 1.0 + 0.05 * torch.cos(2.0 * math.pi * phase)
        drive_sequence = scaled_input.unsqueeze(1) * envelope
        if record_state:
            self.last_drive_input = scaled_input.detach()

        stage1_activity, self.control_stage1_state, self.control_stage1_aux = self.control_stage1(
            drive_sequence,
            v_mem=self.control_stage1_state,
            ref_counter=self.control_stage1_aux,
            time_steps=time_steps,
            current_alpha=current_alpha,
            force_grad=use_grad,
        )
        if not use_grad and self.control_stage1_state is not None:
            self.control_stage1_state = self.control_stage1_state.detach()
            self.control_stage1_aux = self.control_stage1_aux.detach()
        if record_state:
            self.last_stage1_output = stage1_activity.detach()

        stage2_drive = stage1_activity
        if record_state:
            self.last_stage2_drive = stage2_drive.detach()

        stage2_activity, self.control_stage2_state, self.control_stage2_aux = self.control_stage2(
            stage2_drive,
            v_mem=self.control_stage2_state,
            ref_counter=self.control_stage2_aux,
            time_steps=time_steps,
            current_alpha=current_alpha,
            force_grad=use_grad,
        )
        if not use_grad and self.control_stage2_state is not None:
            self.control_stage2_state = self.control_stage2_state.detach()
            self.control_stage2_aux = self.control_stage2_aux.detach()

        if record_state:
            self.last_stage2_output = stage2_activity.detach()

        rwkv_input = self.trajectory_encoder(stage2_activity)
        # Keep the zero-drive condition anchored at zero instead of a learned encoder bias.
        rwkv_input = rwkv_input - self.trajectory_encoder(stage2_activity.new_zeros(stage2_activity.shape))
        control_rate, new_rwkv_state = self.rwkv_memory(rwkv_input, state=rwkv_state)
        if math.isfinite(self.rate_bound) and self.rate_bound > 0.0:
            control_rate = self.rate_bound * torch.tanh(control_rate / self.rate_bound)
        phys_pred = self.phys_hallucination(control_rate)
        new_state_pack = {
            "control_stage1_state": self.control_stage1_state,
            "control_stage1_aux": self.control_stage1_aux,
            "control_stage2_state": self.control_stage2_state,
            "control_stage2_aux": self.control_stage2_aux,
            "rwkv_num_state": new_rwkv_state[0],
            "rwkv_den_state": new_rwkv_state[1],
        }
        return control_rate, new_state_pack, phys_pred

    def init_state(self, batch_size, device):
        dtype = torch.float32
        return {
            "control_stage1_state": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "control_stage1_aux": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "control_stage2_state": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "control_stage2_aux": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "rwkv_num_state": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
            "rwkv_den_state": torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype),
        }


# =====================================================================
# Ghost (minimal RNA-only core)
# =====================================================================
class Ghost(nn.Module):
    def __init__(self, state_dict: Optional[dict] = None, memory_ref=None, device=None):
        super().__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device(getattr(Config, "GHOST_DEVICE", "cpu"))

        self.cortex = NeuralCortex(
            input_dim=int(Config.PROTO_DIM),
            hidden_dim=int(getattr(Config, "GHOST_CORTEX_DIM", 512)),
        ).to(self.device)

        bus_spec = ProtocolSpec(
            phy_dim=int(getattr(Config, "GHOST_CORTEX_DIM", 512)),
            proto_dim=int(Config.PROTO_DIM),
            device=self.device,
            dtype=torch.float32,
        )
        bus_spec.validate()
        self.control_bus = USBLink(bus_spec, coupling_layers=2, hidden_mult=0.25).to(self.device)
        self.intent_norm = nn.LayerNorm(Config.PROTO_DIM).to(self.device)
        self.state_proto_proj = nn.Linear(Config.PROTO_DIM, Config.PROTO_DIM, bias=False).to(self.device)
        self.goal_proto_proj = nn.Linear(Config.PROTO_DIM, Config.PROTO_DIM, bias=False).to(self.device)
        self.goal_input_norm = nn.LayerNorm(Config.PROTO_DIM).to(self.device)
        self.goal_input_scale = float(getattr(Config, "GHOST_GOAL_INPUT_SCALE", 1.0))
        nn.init.eye_(self.state_proto_proj.weight)
        nn.init.orthogonal_(self.goal_proto_proj.weight, gain=0.5)

        if state_dict:
            try:
                if "model_state" in state_dict:
                    self.load_state_dict(state_dict["model_state"], strict=False)
                elif "state_dict" in state_dict:
                    self.load_state_dict(state_dict["state_dict"], strict=False)
                else:
                    self.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

    def step_forward_micro(self, sensory_input, prev_control_state, error_vec=None, dt=0.1, alpha_scale=None, use_grad=False, goal_input=None):
        x = sensory_input.to(self.device).float()
        if goal_input is None:
            goal = torch.zeros_like(x)
        else:
            goal = goal_input.to(self.device).float()
            if goal.shape != x.shape:
                raise ValueError(f"goal_input shape {tuple(goal.shape)} incompatible with sensory_input {tuple(x.shape)}")
        goal_scale = self.goal_input_scale
        if (not math.isfinite(goal_scale)) or goal_scale < 0.0:
            goal_scale = 1.0
        x = self.goal_input_norm(self.state_proto_proj(x) + (goal_scale * self.goal_proto_proj(goal)))
        err = error_vec.to(self.device).float() if isinstance(error_vec, torch.Tensor) else None

        rate, new_state_pack, phys_pred = self.cortex(
            x,
            current_state=prev_control_state,
            error_current=err,
            dt=dt,
            record_state=False,
            alpha_scale=alpha_scale,
            use_grad=use_grad,
        )
        return rate, new_state_pack, phys_pred

    def init_state(self, batch_size=1, device=None) -> Dict[str, torch.Tensor]:
        target_device = device if device is not None else self.device
        return self.cortex.init_state(batch_size, target_device)
