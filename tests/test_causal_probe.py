import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.physics.causal_probe import ResidualProbe,momentum_summary
from rna2d.physics.forecast import PhysicalForecast
from rna2d.physics.interface import PhysicalState


class CausalProbeTests(unittest.TestCase):
    def test_zero_residual_and_capacity_match(self):
        torch.set_num_threads(1);torch.manual_seed(18)
        reference=PhysicalForecast(19);count=sum(p.numel() for p in reference.parameters())
        ghost=ResidualProbe('ghost_obs');mlp=ResidualProbe('mlp_obs',matched_parameters=count)
        self.assertLess(abs(sum(p.numel() for p in mlp.parameters())-count)/count,.01)
        obs=torch.randn(2,8,18);mom=torch.randn(2,8,8);action=torch.zeros(2,8,1);h=torch.tensor([1,16])
        stats=(torch.zeros(18),torch.ones(18),torch.zeros(8),torch.ones(8))
        for model in (ghost,mlp):
            with torch.no_grad():y=model(obs,mom,action,h,*stats)
            self.assertTrue(torch.equal(y,torch.zeros_like(y)))

    def test_stateless_control_cannot_use_history(self):
        model=ResidualProbe('mlp_full',matched_parameters=4082181).eval()
        torch.nn.init.normal_(model.network[-1].weight,std=.001)
        obs=torch.randn(2,8,18);mom=torch.randn(2,8,8);action=torch.zeros(2,8,1);h=torch.tensor([1,16]);stats=(torch.zeros(18),torch.ones(18),torch.zeros(8),torch.ones(8))
        with torch.no_grad():
            a=model(obs,mom,action,h,*stats);obs[:,:-1]+=100;mom[:,:-1]-=100
            b=model(obs,mom,action,h,*stats)
        self.assertTrue(torch.equal(a,b))

    def test_momentum_summary_detects_radial_direction(self):
        q=torch.zeros(1,3,6);q[0,:,0]=torch.tensor([-1.,0.,1.]);q[:,:,5]=1
        p=torch.zeros_like(q);p[0,:,0]=torch.tensor([-.2,0.,.2]);xi=torch.zeros(3,1)
        self.assertGreater(momentum_summary(PhysicalState(q,p,xi))[0,5],0)
        self.assertLess(momentum_summary(PhysicalState(q,-p,xi))[0,5],0)

    def test_rwkv_only_control_preserves_first_step_and_changes_history(self):
        torch.manual_seed(17);full=ResidualProbe('ghost_full');control=ResidualProbe('ghost_no_rwkv_memory')
        torch.nn.init.normal_(full.network.ghost.cortex.phys_hallucination.weight,std=.01)
        control.load_state_dict(full.state_dict());full.eval();control.eval()
        obs=torch.randn(2,8,18);mom=torch.randn(2,8,8);action=torch.zeros(2,8,1);h=torch.tensor([1,16]);stats=(torch.zeros(18),torch.ones(18),torch.zeros(8),torch.ones(8))
        with torch.no_grad():
            a=full(obs[:,:1],mom[:,:1],action[:,:1],h,*stats);b=control(obs[:,:1],mom[:,:1],action[:,:1],h,*stats)
            self.assertTrue(torch.equal(a,b))
            a=full(obs,mom,action,h,*stats);b=control(obs,mom,action,h,*stats)
            self.assertGreater((a-b).abs().max().item(),0.)


if __name__=='__main__':unittest.main()
