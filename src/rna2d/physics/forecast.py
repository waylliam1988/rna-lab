"""Next physical-observation probe using original Ghost and a new input adapter.

Teacher-forced forecasting from current 3D observations is not sequence folding.
The 18->256 perception projection is new; original GVP is not claimed restored.
"""
import torch
from torch import nn
from .ghost import Ghost


class PhysicalForecast(nn.Module):
    def __init__(self, input_dim=18, device='cpu'):
        super().__init__()
        self.perception=nn.Sequential(nn.Linear(input_dim,256),nn.LayerNorm(256)).to(device)
        self.ghost=Ghost(device=device)

    def forward(self, observations, reset_each_step=False):
        state=self.ghost.init_state(observations.shape[0],observations.device)
        outputs=[]
        for t in range(observations.shape[1]):
            if reset_each_step:state=self.ghost.init_state(observations.shape[0],observations.device)
            x=self.perception(observations[:,t])
            _,state,pred=self.ghost.step_forward_micro(x,state,dt=.05,use_grad=torch.is_grad_enabled())
            outputs.append(pred)
        return torch.stack(outputs,dim=1)


def window_batch(episodes, indices, starts, length):
    # All roles use exactly length+1 observations: the last state is target only.
    return torch.stack([episodes[i,s:s+length] for i,s in zip(indices,starts)]), torch.stack([episodes[i,s+1:s+length+1] for i,s in zip(indices,starts)])
