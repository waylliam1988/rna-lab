import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.iterative import PairDynamics


class IterationTests(unittest.TestCase):
    def test_rwkv_history_ablation_and_auxiliary_gradient(self):
        torch.manual_seed(9);m=PairDynamics(8,1,'rwkv');r=PairDynamics(8,1,'rwkv_reset');r.load_state_dict(m.state_dict());m.eval();r.eval()
        x=torch.tensor([[0,1,2,3,0,2,1,3]])
        self.assertTrue(torch.allclose(m(x),r(x)))
        m.steps=r.steps=4
        self.assertFalse(torch.allclose(m(x),r(x),atol=1e-7))
        before=m(x);m(torch.flip(x,[-1]));self.assertTrue(torch.equal(before,m(x)))
        m.train();logits,aux=m.forward_with_aux(x);(logits.square().mean()+aux['prediction_mse']).backward()
        self.assertGreater(m.next_head.weight.grad.abs().sum(),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_padding_reset_symmetry_and_finite_gradient(self):
        torch.manual_seed(7);model=PairDynamics(8,4);model.eval()
        x=torch.tensor([[0,1,2,3,0,2,1,3]])
        y=model(x);padded=model(torch.cat((x,torch.full((1,3),4)),1))[:,:8,:8]
        self.assertTrue(torch.allclose(y,padded,atol=1e-6))
        self.assertTrue(torch.allclose(y,y.transpose(-1,-2)))
        model(x);self.assertTrue(torch.equal(y,model(x)))
        model.train();model(x).square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.feature[0].weight.grad.abs().sum(),0)


if __name__=='__main__':unittest.main()
