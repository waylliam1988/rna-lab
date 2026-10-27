"""Algorithmic graph refinement; iterations are not physical folding times."""
import torch
from torch import nn
from .model import PairBaseline


class PairDynamics(nn.Module):
    def __init__(self,hidden=32,steps=4,mode='iterative'):
        super().__init__()
        if mode!='iterative':raise ValueError('Unvalidated mechanism')
        self.hidden=hidden;self.steps=steps;self.mode=mode
        self.base=PairBaseline(hidden)
        self.feature=nn.Sequential(nn.Linear(2*hidden+1,hidden),nn.GELU(),nn.LayerNorm(hidden))
        self.update_q=nn.Linear(hidden,hidden,bias=False);self.update_k=nn.Linear(hidden,hidden,bias=False)
        self.last_trace=[]

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
        for _ in range(self.steps):
            features=self.features(h,state,allowed)*valid
            proposal=torch.matmul(self.update_q(features),self.update_k(features).transpose(-1,-2))/self.hidden**0.5
            proposal=(proposal+proposal.transpose(-1,-2))*0.5
            target=base+2*proposal.tanh()
            delta=0.5*(target-state);state=state+delta
            if not self.training:
                denom=allowed.sum().clamp_min(1)
                self.last_trace.append(dict(mean_abs_update=float((delta.abs()*allowed).sum()/denom),
                                            positive_fraction=float(((delta>0)&allowed).sum()/denom),
                                            negative_fraction=float(((delta<0)&allowed).sum()/denom)))
        return state,{}

    def forward(self,tokens):return self.forward_with_aux(tokens)[0]
