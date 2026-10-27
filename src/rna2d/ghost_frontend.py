"""Ghost continuous frontend with explicit state, separated from RWKV memory."""
import math
import torch
from torch import nn
from .core.ghost import GhostDynamics,GhostSettings


class GhostFrontEnd(nn.Module):
    def __init__(self,hidden,inner_steps=4):
        super().__init__();self.hidden=hidden;self.inner_steps=inner_steps
        reference=GhostDynamics(GhostSettings(input_dim=hidden,hidden_dim=hidden,inner_steps=inner_steps,prediction_dim=hidden))
        # Register only the continuous stages and trace encoder, not the
        # reference object's RWKV/head. Full mode adds exactly one RWKV cell.
        self.stage1=reference.control_stage1;self.stage2=reference.control_stage2;self.encoder=reference.trajectory_encoder

    def forward(self,x,state=None):
        if state is None:state=tuple(torch.zeros_like(x) for _ in range(4))
        phase=torch.linspace(0,1,self.inner_steps,device=x.device,dtype=x.dtype).view(1,self.inner_steps,1)
        drive=(x*0.5).unsqueeze(1)*(1+0.05*torch.cos(2*math.pi*phase))
        a,m1,u1=self.stage1(drive,state[0],state[1],current_alpha=x.new_tensor(2.0))
        b,m2,u2=self.stage2(a,state[2],state[3],current_alpha=x.new_tensor(2.0))
        encoded=self.encoder(b)-self.encoder(torch.zeros_like(b))
        return encoded,(m1,u1,m2,u2)
