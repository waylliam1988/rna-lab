import io
from pathlib import Path
import sys
import tempfile
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.losses import balanced_pair_bce
from rna2d.sequence_encoders import SequencePairModel
from rna2d.thermodynamics import PriorStore
from rna2d.thermo_fusion import ThermoResidualModel, add_prior_logits


class FusionTests(unittest.TestCase):
    def test_alpha_zero_exact_output_and_gradient_without_prior(self):
        class ForbiddenStore:
            def batch(self,tokens):raise AssertionError('Disabled prior accessed')
        torch.manual_seed(42)
        cnn=SequencePairModel('cnn',16,hidden=8,depth=1)
        clone=SequencePairModel('cnn',16,hidden=8,depth=1)
        clone.load_state_dict(cnn.state_dict())
        model=ThermoResidualModel(clone,ForbiddenStore(),alpha=0)
        tokens=torch.tensor([[2,2,2,0,0,0,1,1,1]])
        a,b=cnn(tokens),model(tokens)
        self.assertTrue(torch.equal(a,b))
        a.sum().backward();b.sum().backward()
        for x,y in zip(cnn.parameters(),clone.parameters()):
            self.assertTrue(torch.equal(x.grad,y.grad))
        self.assertEqual(sum(p.numel() for p in cnn.parameters()),sum(p.numel() for p in model.parameters()))

    def test_zero_prior_finite_and_overridable_with_gradient(self):
        residual=torch.tensor([[[10.0]]],requires_grad=True)
        prior=torch.zeros_like(residual,requires_grad=True)
        result=add_prior_logits(residual,prior,alpha=1)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(result.sigmoid().item(),0.5)
        result.sum().backward()
        self.assertEqual(residual.grad.item(),1)
        self.assertIsNone(prior.grad)

    def test_unknown_labels_never_enter_loss(self):
        residual=torch.zeros((1,6,6),requires_grad=True)
        p=torch.full_like(residual,0.2)
        mask=torch.zeros_like(residual,dtype=torch.bool);mask[0,0,4]=True
        first=torch.zeros_like(residual);first[0,0,4]=1
        second=torch.ones_like(residual);second[0,0,4]=1
        logits=add_prior_logits(residual,p,alpha=0.25)
        a=balanced_pair_bce(logits,first,mask);b=balanced_pair_bce(logits,second,mask)
        self.assertTrue(torch.equal(a,b))
        a.backward()
        self.assertEqual(torch.count_nonzero(residual.grad).item(),1)

    def test_variable_length_prior_alignment_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            store=PriorStore(folder)
            store.get('GGGAAACCC',create=True);store.get('AAAAA',create=True)
            tokens=torch.tensor([[2,2,2,0,0,0,1,1,1],[0,0,0,0,0,4,4,4,4]])
            cnn=SequencePairModel('cnn',16,hidden=8,depth=1)
            model=ThermoResidualModel(cnn,store,alpha=0.25)
            logits=model(tokens)
            expected=add_prior_logits(cnn(tokens),store.batch(tokens),alpha=0.25)
            self.assertTrue(torch.equal(logits,expected))
            self.assertTrue(torch.equal(logits,logits.transpose(1,2)))
            buffer=io.BytesIO()
            torch.save({'model':model.state_dict(),'alpha':model.alpha,'epsilon':model.epsilon},buffer)
            buffer.seek(0);ckpt=torch.load(buffer,weights_only=True)
            rebuilt=ThermoResidualModel(SequencePairModel('cnn',16,hidden=8,depth=1),store,
                                        alpha=ckpt['alpha'],epsilon=ckpt['epsilon'])
            rebuilt.load_state_dict(ckpt['model'],strict=True)
            self.assertTrue(torch.equal(logits,rebuilt(tokens)))

    def test_invalid_inputs(self):
        x=torch.zeros((1,2,2))
        for alpha,epsilon in [(-1,1e-4),(float('nan'),1e-4),(1,0),(1,0.5)]:
            with self.assertRaises(ValueError):add_prior_logits(x,x,alpha=alpha,epsilon=epsilon)
        for p in [torch.zeros((1,3,3)),torch.full_like(x,float('nan')),torch.full_like(x,1.1)]:
            with self.assertRaises(ValueError):add_prior_logits(x,p,alpha=1)


if __name__=='__main__':unittest.main()
