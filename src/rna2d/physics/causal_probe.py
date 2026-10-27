"""Residual probes with original Ghost core and explicit new conditioning.

Momentum input is a summary, not the full molecular phase-space state.
Action specifies a radial force amplitude held for the requested future window.
"""
import math
import torch
from torch import nn
from .forecast import PhysicalForecast

VARIANTS={'ghost_obs':('ghost',False,False),'ghost_momentum':('ghost',True,False),'ghost_action':('ghost',False,True),'ghost_full':('ghost',True,True),'ghost_no_rwkv_memory':('ghost',True,True),'mlp_obs':('mlp',False,False),'mlp_full':('mlp',True,True)}


def radial_force(state,amplitude):
    x=state.q[...,:3];direction=torch.nn.functional.normalize(x-x.mean(1,keepdim=True),dim=-1)
    return torch.cat([float(amplitude)*direction,torch.zeros_like(direction)],dim=-1)


def momentum_summary(state):
    x=state.q[...,:3];v=state.p[...,:3];n=state.p[...,3:]
    dx=x-x.mean(1,keepdim=True);dv=v-v.mean(1,keepdim=True)
    rg=torch.sqrt(dx.square().sum(-1).mean(1)+1e-8)
    rgdot=(dx*dv).sum(-1).mean(1)/rg
    return torch.cat([v.mean(1),v.square().mean((1,2)).sqrt()[:,None],n.square().mean((1,2)).sqrt()[:,None],rgdot[:,None],state.xi.reshape(x.shape[0],-1).mean(1,keepdim=True),state.xi.reshape(x.shape[0],-1).abs().amax(1,keepdim=True)],dim=-1)


class ResidualProbe(nn.Module):
    def __init__(self,variant,device='cpu',matched_parameters=None):
        super().__init__();kind,self.use_momentum,self.use_action=VARIANTS[variant]
        self.kind=kind;self.rwkv_only_reset=variant=='ghost_no_rwkv_memory';d=19+8*self.use_momentum+self.use_action
        if kind=='ghost':self.network=PhysicalForecast(d,device=device)
        else:
            if matched_parameters is None:raise ValueError('Explicit matched capacity required')
            # Two hidden layers, parameter-count-only selection before fitting.
            width=min(range(1500,2501),key=lambda w:abs(w*w+(d+7)*w+5-matched_parameters))
            self.width=width
            self.network=nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,5)).to(device)
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,obs,momentum,action,horizon,obs_mean,obs_scale,mom_mean,mom_scale,shuffle=False,reset=False):
        parts=[(obs-obs_mean)/obs_scale]
        if self.use_momentum:parts.append((momentum-mom_mean)/mom_scale)
        if self.use_action:parts.append(action/.4)
        h=torch.log2(horizon.float()).view(-1,1,1)/4
        parts.append(h.expand(-1,obs.shape[1],1));x=torch.cat(parts,dim=-1)
        if shuffle:x=x[:,[6,0,5,1,4,2,3,7]]
        if self.kind=='mlp':return self.network(x[:,-1])
        if self.rwkv_only_reset:
            g=self.network.ghost;state=g.init_state(x.shape[0],x.device)
            for t in range(x.shape[1]):
                if reset:state=g.init_state(x.shape[0],x.device)
                state=dict(state,rwkv_num_state=torch.zeros_like(state['rwkv_num_state']),rwkv_den_state=torch.zeros_like(state['rwkv_den_state']))
                _,state,pred=g.step_forward_micro(self.network.perception(x[:,t]),state,dt=.05,use_grad=torch.is_grad_enabled())
            return pred
        return self.network(x,reset_each_step=reset)[:,-1]
