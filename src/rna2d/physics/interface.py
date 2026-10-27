"""Explicit adapter around the restored physical definitions.

This adapter supplies state/observation/feedback contracts. It does not recreate
the original encoder, policy heads, adaptive metric or trained checkpoints.
"""
from dataclasses import dataclass, fields
from types import SimpleNamespace
import math
import torch
from .config import Config
from .reference import BioSystem, RNAForceFieldParams, compute_rna_force_field_fast, project_orientation
from .kernels import jit_integrate_trajectory


@dataclass
class PhysicalState:
    q: torch.Tensor       # [B,L,6], positions nm, orientations dimensionless
    p: torch.Tensor       # [B,L,6], reduced generalized momentum
    xi: torch.Tensor      # [B*L,1], reduced thermostat state


def original_force_params():
    names = {'min_seq_sep':'RNA_MIN_SEQ_SEP', 'energy_norm':'RNA_ENERGY_NORM'}
    return RNAForceFieldParams(**{f.name:getattr(Config, names.get(f.name, 'RNA_'+f.name.upper())) for f in fields(RNAForceFieldParams)})


class PhysicalInterface(BioSystem):
    def __init__(self, use_geom_invariants=True):
        self.ff_params = original_force_params()
        self.use_geom_invariants = use_geom_invariants
        self.global_step = 0
        self.encoder = SimpleNamespace(training=True)

    def initial_state(self, q, mask):
        if q.ndim != 3 or q.shape[-1] != 6 or mask.shape != q.shape[:2]:
            raise ValueError('Expected q[B,L,6] and mask[B,L]')
        if not torch.isfinite(q).all() or (mask.sum(1)<=0).any():
            raise ValueError('Finite coordinates and at least one observed residue required')
        q = project_orientation(q) * mask[...,None]
        return PhysicalState(q, torch.zeros_like(q), q.new_zeros((q.shape[0]*q.shape[1],1)))

    def observe(self, state, contact, mask):
        force, energy = compute_rna_force_field_fast(state.q, contact, mask, self.ff_params)
        _, sum_m3, sum_m, cmean, cmax = self._fold_precompute_sums_and_contacts(contact, mask)
        obs, micro = self._fold_build_obs_vec(state.q, force, energy, contact, mask, mask[...,None], sum_m3, sum_m, cmean, cmax)
        return obs, micro, force, energy

    def transition(self, state, contact, mask, port_force=None, dt=None, temperature=None, gamma=None):
        """One original integration step, then original orientation projection.

        Field is recomputed at every outer step. Identity inverse mass is an
        explicit test fixture, not the old adaptive metric controller.
        Temperature/time/energy are reduced simulation units, not K/seconds.
        """
        q=state.q; b,l,d=q.shape
        _,_,force,_=self.observe(state, contact, mask)
        port=torch.zeros_like(q) if port_force is None else port_force
        if port.shape!=q.shape or not torch.isfinite(port).all():raise ValueError('Invalid port force')
        delta=Config.RNA_DT if dt is None else float(dt)
        temp=Config.RNA_TARGET_TEMP if temperature is None else float(temperature)
        friction=Config.RNA_GAMMA_DYN if gamma is None else float(gamma)
        if not all(math.isfinite(v) for v in (delta,temp,friction)) or delta<=0 or temp<0 or friction<0:raise ValueError('Invalid integration parameters')
        flat=lambda x:x.reshape(b*l,d)
        max_exp=math.log(torch.finfo(q.dtype).max)*.95
        result=jit_integrate_trajectory(flat(q),flat(state.p),state.xi,flat(q.detach()),flat(force),flat(port*mask[...,None]),q.new_ones((1,d)),q.new_tensor(delta),1,0.,0.,mask.reshape(b*l,1)*temp,1e-8,q.new_tensor(friction),Config.RNA_GHOST_FORCE_CLIP,max_exp)
        q1,p1,xi,h,_,v,work=result
        q1=project_orientation(q1.reshape(b,l,d))*mask[...,None]
        xi=xi.detach() if Config.RNA_XI_DETACH else xi
        return PhysicalState(q1,p1.reshape(b,l,d)*mask[...,None],xi), {'integrator_H':h,'integrator_V':v,'port_work':work}

    def feedback_loss(self, prediction, next_observation):
        return self._fold_update_hallucination_loss(next_observation.new_zeros(()),prediction,next_observation)


def normalized_macro(observation):
    target=observation[...,:5].detach().clone()
    target[...,0]=torch.log1p(target[...,0])
    target[...,3]=torch.log1p(target[...,3])
    return target
