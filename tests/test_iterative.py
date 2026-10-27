import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.iterative import PairDynamics


class IterationTests(unittest.TestCase):
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
