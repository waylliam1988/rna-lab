"""Controlled sequence-only encoders with one common pair head.

RWKV is Nezha's Lite cell, bidirectional, cap=None, bounded decay and a
chunked exact-linear-recurrence implementation. EGNN uses synthetic sequence
position coordinates, never native RNA coordinates. Neither is a full published
RNA foundation model or a benchmark of RNA3D folding.
"""
import math
import torch
from torch import nn
from .core.memory import RWKVCellLite


def position_features(n,device,dtype,hidden=64):
    p=torch.arange(n,device=device,dtype=dtype)
    freq=torch.exp(torch.arange(0,hidden,2,device=device,dtype=dtype)*(-math.log(10000.)/hidden))
    angles=p[:,None]*freq[None];pe=torch.stack([angles.sin(),angles.cos()],dim=-1).reshape(n,hidden)
    coords=torch.stack([p.cos(),p.sin(),p/128.],dim=-1)
    return pe,coords


def lite_sequence(cell,x,valid,chunk=32):
    """Equivalent to the un-capped cell within decay bounds, up to roundoff.

    Prefix products/sums are chunked to avoid long-sequence exponential overflow.
    The scalar decay parameter is clamped to[-12,0] for stable chunk inverses.
    Invalid positions keep state unchanged and output zero.
    """
    r=cell.receptance(x).sigmoid();k=cell._map_key(cell.key(x));v=cell.value(x);g=cell.gate(x).sigmoid()
    decay=torch.exp(-torch.exp(cell.time_decay.clamp(-12.,0.)));first=cell.time_first.exp()
    a=x.new_zeros((x.shape[0],cell.hidden_dim));b=torch.zeros_like(a);values=[]
    for start in range(0,x.shape[1],chunk):
        stop=min(start+chunk,x.shape[1]);m=valid[:,start:stop,None]
        kk=k[:,start:stop]*m;kv=kk*v[:,start:stop]
        factors=torch.where(m,decay[None,None,:],torch.ones_like(kk));product=factors.cumprod(1)
        aa=product*(a[:,None,:]+(kv/product).cumsum(1));bb=product*(b[:,None,:]+(kk/product).cumsum(1))
        ap=torch.cat([a[:,None,:],aa[:,:-1]],dim=1);bp=torch.cat([b[:,None,:],bb[:,:-1]],dim=1)
        values.append((ap+first*kv)/(bp+first*kk+1e-6));a=aa[:,-1];b=bb[:,-1]
    core=cell.ln(r*torch.cat(values,dim=1));core=core-cell.ln(torch.zeros_like(core))
    return cell.output(core)*g*valid[...,None]


class FeedForward(nn.Module):
    def __init__(self,h,w):
        super().__init__();self.norm=nn.LayerNorm(h);self.net=nn.Sequential(nn.Linear(h,w),nn.GELU(),nn.Linear(w,h))
    def forward(self,x,mask):return (x+self.net(self.norm(x)))*mask[...,None]


class CNNBlock(nn.Module):
    def __init__(self,h,w):
        super().__init__();self.norm=nn.LayerNorm(h);self.net=nn.Sequential(nn.Conv1d(h,w,5,padding=2),nn.GELU(),nn.Conv1d(w,h,5,padding=2))
    def forward(self,x,mask):
        y=self.norm(x)*mask[...,None];y=y.transpose(1,2)
        for layer in self.net:y=layer(y)*mask[:,None,:]
        return (x+y.transpose(1,2))*mask[...,None]


class RWKVBlock(nn.Module):
    def __init__(self,h,w):
        super().__init__();self.norm=nn.LayerNorm(h);self.forward_cell=RWKVCellLite(h,h);self.reverse_cell=RWKVCellLite(h,h)
        self.merge=nn.Linear(2*h,h);self.ff=FeedForward(h,w)
    def forward(self,x,mask):
        z=self.norm(x)*mask[...,None]
        a=lite_sequence(self.forward_cell,z,mask);b=lite_sequence(self.reverse_cell,z.flip(1),mask.flip(1)).flip(1)
        return self.ff((x+self.merge(torch.cat([a,b],dim=-1)))*mask[...,None],mask)


class TransformerBlock(nn.Module):
    def __init__(self,h,w):
        super().__init__();self.norm=nn.LayerNorm(h);self.attention=nn.MultiheadAttention(h,4,dropout=0.,batch_first=True);self.ff=FeedForward(h,w)
    def forward(self,x,mask):
        z=self.norm(x);y,_=self.attention(z,z,z,key_padding_mask=~mask,need_weights=False)
        return self.ff((x+y)*mask[...,None],mask)


class EGNNBlock(nn.Module):
    def __init__(self,h,w):
        super().__init__();self.norm=nn.LayerNorm(h)
        self.edge=nn.Sequential(nn.Linear(2*h+2,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU())
        self.coordinate=nn.Sequential(nn.Linear(w,w),nn.SiLU(),nn.Linear(w,1))
        self.node=nn.Sequential(nn.Linear(h+w+1,w),nn.SiLU(),nn.Linear(w,h))
    def forward(self,h,x,mask):
        b,n,d=h.shape;z=self.norm(h);delta=x[:,:,None,:]-x[:,None,:,:];distance=delta.square().sum(-1,keepdim=True)
        pos=torch.arange(n,device=h.device);sep=torch.log1p((pos[:,None]-pos[None,:]).abs().to(h.dtype))/math.log(513.)
        inputs=torch.cat([z[:,:,None,:].expand(-1,-1,n,-1),z[:,None,:,:].expand(-1,n,-1,-1),distance,sep[None,:,:,None].expand(b,-1,-1,-1)],dim=-1)
        edges=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,device=h.device,dtype=torch.bool)[None]
        msg=self.edge(inputs)*edges[...,None];degree=edges.sum(2).clamp_min(1)[...,None]
        update=(delta/torch.sqrt(distance+1e-6)*self.coordinate(msg).tanh()*edges[...,None]).sum(2)/degree
        x=(x+.05*update)*mask[...,None]
        center=x.sum(1,keepdim=True)/mask.sum(1).clamp_min(1)[:,None,None]
        radial=torch.sqrt((x-center).square().sum(-1,keepdim=True)+1e-6)
        # Classification readout uses an invariant of updated coordinates, so
        # the final coordinate-update module is not an unused parameter block.
        h=(h+self.node(torch.cat([z,msg.sum(2)/degree,radial],dim=-1)))*mask[...,None]
        return h,x


class SequencePairModel(nn.Module):
    def __init__(self,kind,width,hidden=64,depth=2):
        super().__init__();self.kind=kind;self.hidden=hidden
        # Shared components are constructed before architecture-dependent modules:
        # resetting seed gives identical embeddings/position projection/pair head.
        self.embedding=nn.Embedding(5,hidden,padding_idx=4);self.position=nn.Linear(3,hidden,bias=False);self.input_norm=nn.LayerNorm(hidden)
        self.query=nn.Linear(hidden,hidden,bias=False);self.pair_bias=nn.Parameter(torch.tensor(-2.));self.distance=nn.Embedding(32,1);nn.init.zeros_(self.distance.weight)
        block={'cnn':CNNBlock,'rwkv_lite':RWKVBlock,'transformer':TransformerBlock,'egnn_seed':EGNNBlock}[kind]
        self.layers=nn.ModuleList([block(hidden,width) for _ in range(depth)])

    def encode(self,tokens):
        mask=tokens.ne(4);pe,coords=position_features(tokens.shape[1],tokens.device,self.embedding.weight.dtype,self.hidden)
        h=self.input_norm(self.embedding(tokens)+pe[None]+self.position(coords)[None])*mask[...,None]
        x=coords[None].expand(tokens.shape[0],-1,-1)*mask[...,None]
        for layer in self.layers:
            if self.kind=='egnn_seed':h,x=layer(h,x,mask)
            else:h=layer(h,mask)
        return h

    def forward(self,tokens):
        h=self.encode(tokens);q=self.query(h);logits=(q@h.transpose(1,2))/self.hidden**.5;logits=(logits+logits.transpose(1,2))*.5
        p=torch.arange(h.shape[1],device=h.device);bins=(torch.log2((p[:,None]-p[None,:]).abs().float()+1)*4).floor().long().clamp_max(31)
        return logits+self.distance(bins).squeeze(-1)+self.pair_bias


def choose_width(kind,target=150000):
    low,high=8,2048
    while high-low>1:
        middle=(low+high)//2;count=sum(p.numel() for p in SequencePairModel(kind,middle).parameters())
        if count<target:low=middle
        else:high=middle
    candidates=[(abs(sum(p.numel() for p in SequencePairModel(kind,w).parameters())-target),w) for w in (low,high)]
    return min(candidates)[1]
