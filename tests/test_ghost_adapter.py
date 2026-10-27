import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.core.ghost import GhostDynamics,GhostSettings
from rna2d.ghost_frontend import GhostFrontEnd
from rna2d.iterative import PairDynamics


class GhostAdapterTests(unittest.TestCase):
    def test_frontend_matches_migrated_ghost_before_memory(self):
        torch.manual_seed(1);g=GhostDynamics(GhostSettings(input_dim=8,hidden_dim=8,inner_steps=4,prediction_dim=8));f=GhostFrontEnd(8)
        f.stage1.load_state_dict(g.control_stage1.state_dict());f.stage2.load_state_dict(g.control_stage2.state_dict());f.encoder.load_state_dict(g.trajectory_encoder.state_dict())
        g.eval();f.eval();captured=[]
        hook=g.rwkv_memory.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
        x=torch.randn(2,8);g(x);encoded,_=f(x);hook.remove()
        self.assertTrue(torch.equal(encoded,captured[0]))
        self.assertTrue(torch.equal(f(torch.zeros_like(x))[0],torch.zeros_like(x)))

    def test_both_adapters_reset_and_backpropagate(self):
        x=torch.tensor([[0,1,2,3,0,2,1,3]])
        for mode in ('ghost_only','full'):
            m=PairDynamics(8,4,mode);m.eval();a=m(x);m(x.flip(-1));self.assertTrue(torch.equal(a,m(x)))
            padded=m(torch.cat((x,torch.full((1,3),4)),1))[:,:8,:8]
            self.assertTrue(torch.allclose(a,padded,atol=1e-6))
            m.train();y,aux=m.forward_with_aux(x);(y.square().mean()+0.01*aux['prediction_mse']).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertGreater(m.ghost.stage1.fc.weight.grad.abs().sum(),0)


if __name__=='__main__':unittest.main()
