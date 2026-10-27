import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.sequence_encoders import SequencePairModel,choose_width,lite_sequence,EGNNBlock
from rna2d.core.memory import RWKVCellLite


class SequenceEncoderTests(unittest.TestCase):
    def test_chunked_rwkv_matches_cell_outputs_and_gradients(self):
        torch.set_num_threads(1);torch.manual_seed(18);cell=RWKVCellLite(8,8);other=RWKVCellLite(8,8);other.load_state_dict(cell.state_dict())
        x=torch.randn(2,73,8,requires_grad=True);xx=x.detach().clone().requires_grad_();mask=torch.ones(2,73,dtype=torch.bool)
        outputs=[];state=None
        for t in range(73):y,state=cell(x[:,t],state);outputs.append(y)
        a=torch.stack(outputs,1);b=lite_sequence(other,xx,mask)
        self.assertTrue(torch.allclose(a,b,atol=2e-5,rtol=2e-4))
        a.square().sum().backward();b.square().sum().backward()
        self.assertTrue(torch.allclose(x.grad,xx.grad,atol=2e-4,rtol=2e-3))

    def test_padding_symmetry_and_gradients_for_all_encoders(self):
        tokens=torch.tensor([[0,1,2,3,0,1,2,3],[0,1,2,3,4,4,4,4]])
        for kind in ('cnn','rwkv_lite','transformer','egnn_seed'):
            torch.manual_seed(42);model=SequencePairModel(kind,16).eval()
            logits=model(tokens);alone=model(tokens[1:2,:4]);self.assertEqual(logits.shape,(2,8,8))
            self.assertTrue(torch.allclose(logits,logits.transpose(1,2),atol=1e-6))
            self.assertTrue(torch.allclose(logits[1,:4,:4],alone[0],atol=2e-5,rtol=2e-4),kind)
            logits.square().mean().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_parameter_match_and_shared_components(self):
        shared=None
        for kind in ('cnn','rwkv_lite','transformer','egnn_seed'):
            width=choose_width(kind);torch.manual_seed(8);model=SequencePairModel(kind,width)
            self.assertLess(abs(sum(p.numel() for p in model.parameters())-150000)/150000,.01)
            state={k:v for k,v in model.state_dict().items() if not k.startswith('layers.')}
            if shared is None:shared=state
            else:self.assertTrue(all(torch.equal(v,shared[k]) for k,v in state.items()))

    def test_egnn_layer_equivariance(self):
        torch.manual_seed(9);layer=EGNNBlock(8,16);h=torch.randn(2,7,8);x=torch.randn(2,7,3);m=torch.ones(2,7,dtype=torch.bool)
        rotation,_=torch.linalg.qr(torch.randn(3,3));shift=torch.tensor([2.,-1.,.5])
        a,xa=layer(h,x,m);b,xb=layer(h,x@rotation+shift,m)
        self.assertTrue(torch.allclose(a,b,atol=2e-5,rtol=1e-4));self.assertTrue(torch.allclose(xa@rotation+shift,xb,atol=2e-5,rtol=1e-4))


if __name__=='__main__':unittest.main()
