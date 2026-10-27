"""Algorithmic graph refinement; iterations are not physical folding times."""
import torch
from torch import nn
from .model import PairBaseline
from .core.memory import RWKVCellLite


class PairDynamics(nn.Module):
    def __init__(self,hidden=32,steps=4,mode='iterative'):
        super().__init__()
        if mode not in ('iterative','rwkv','rwkv_reset'):raise ValueError('Unvalidated mechanism')
        self.hidden=hidden;self.steps=steps;self.mode=mode
        self.base=PairBaseline(hidden)
        self.feature=nn.Sequential(nn.Linear(2*hidden+1,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.update_q=nn.Linear(hidden,hidden,bias=False);self.update_k=nn.Linear(hidden,hidden,bias=False)
        self.last_trace=[]
        if mode.startswith('rwkv'):
            self.memory=RWKVCellLite(hidden,hidden,key_mode='softplus_zero',pair_state_cap=1.0,pair_state_cap_mode='soft')
            self.next_head=nn.Linear(hidden,hidden)

    def candidate_mask(self,tokens):
        table=tokens.new_tensor([[a+b in ('AU','UA','GC','CG','GU','UG') for b in 'ACGUX'] for a in 'ACGUX'],dtype=torch.bool)
        n=tokens.shape[1];pos=torch.arange(n,device=tokens.device)
        return table[tokens[:,:,None],tokens[:,None,:]] & ((pos[:,None]-pos[None,:]).abs()>=4)

    def features(self,h,state,allowed):
        weights=state.sigmoid()*allowed
        context=torch.matmul(weights,h)/weights.sum(-1,keepdim=True).clamp_min(1)
        density=weights.sum(-1,keepdim=True)/allowed.sum(-1,keepdim=True).clamp_min(1)
        return self.feature(torch.cat((h,context,density),dim=-1))

    def forward_with_aux(self,tokens):
        h=self.base.encode(tokens);base=self.base.score(h);state=base
        allowed=self.candidate_mask(tokens);valid=tokens.ne(4).unsqueeze(-1)
        self.last_trace=[]
        memory_state=None;prediction_losses=[];persistence_losses=[]
        for _ in range(self.steps):
            features=self.features(h,state,allowed)*valid
            current_features=features
            if self.mode.startswith('rwkv'):
                rate,memory_state=self.memory(features.reshape(-1,self.hidden),None if self.mode=='rwkv_reset' else memory_state)
                features=(features+rate.reshape_as(features))*valid
            proposal=torch.matmul(self.update_q(features),self.update_k(features).transpose(-1,-2))/self.hidden**0.5
            proposal=(proposal+proposal.transpose(-1,-2))*0.5
            target=base+2*proposal.tanh()
            delta=0.5*(target-state);state=state+delta
            if self.mode.startswith('rwkv'):
                next_features=(self.features(h,state,allowed)*valid).detach()
                prediction=self.next_head(features)*valid
                count=(valid.sum()*self.hidden).clamp_min(1)
                prediction_losses.append(((prediction-next_features).square()*valid).sum()/count)
                persistence_losses.append(((current_features-next_features).square()*valid).sum()/count)
            if not self.training:
                denom=allowed.sum().clamp_min(1)
                self.last_trace.append(dict(mean_abs_update=float((delta.abs()*allowed).sum()/denom),
                                            positive_fraction=float(((delta>0)&allowed).sum()/denom),
                                            negative_fraction=float(((delta<0)&allowed).sum()/denom)))
        aux={}
        if prediction_losses:
            aux=dict(prediction_mse=torch.stack(prediction_losses).mean(),persistence_mse=torch.stack(persistence_losses).mean())
        return state,aux

    def forward(self,tokens):return self.forward_with_aux(tokens)[0]
