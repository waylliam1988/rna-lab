"""Versioned input-only factorial probe. Physics, targets and J remain fixed."""
import torch
from torch import nn
from .forecast import PhysicalForecast

PROFILES={'keep_static_no_geom':(True,False),'drop_static_no_geom':(False,False),'keep_static_geom':(True,True),'drop_static_geom':(False,True)}


class RepresentationProbe(nn.Module):
    def __init__(self,kind,profile,device='cpu',matched_parameters=None):
        super().__init__();self.kind=kind;self.profile=profile
        self.keep_static,self.geometry=PROFILES[profile]
        if kind=='ghost':self.network=PhysicalForecast(30,device=device)
        elif kind=='mlp':
            if matched_parameters is None:raise ValueError('Explicit parameter reference required')
            width=min(range(1500,2501),key=lambda w:abs(w*w+37*w+5-matched_parameters))
            self.width=width
            self.network=nn.Sequential(nn.Linear(30,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,5)).to(device)
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        else:raise ValueError(kind)

    def input_vector(self,obs,mom,action,horizon,mean,scale,mm,ms):
        if obs.shape[-1]!=20:raise ValueError('Version2 requires20 observation slots')
        mask=torch.ones(20,device=obs.device,dtype=obs.dtype)
        if not self.keep_static:mask[[1,2]]=0
        if not self.geometry:mask[[18,19]]=0
        z=((obs-mean)/scale)*mask
        h=torch.log2(horizon.float()).view(-1,1,1)/4
        return torch.cat([z,(mom-mm)/ms,action/.4,h.expand(-1,obs.shape[1],1)],dim=-1)

    def forward(self,obs,mom,action,horizon,mean,scale,mm,ms):
        x=self.input_vector(obs,mom,action,horizon,mean,scale,mm,ms)
        if self.kind=='mlp':return self.network(x[:,-1])
        return self.network(x)[:,-1]
