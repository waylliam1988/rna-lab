import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.physics.representation_v2 import RepresentationProbe


class RepresentationTests(unittest.TestCase):
    def test_removed_channels_have_no_information_and_other_static_retained(self):
        torch.set_num_threads(1);model=RepresentationProbe('ghost','drop_static_no_geom')
        obs=torch.randn(2,8,20);mom=torch.randn(2,8,8);action=torch.zeros(2,8,1);h=torch.tensor([1,16]);stats=(torch.zeros(20),torch.ones(20),torch.zeros(8),torch.ones(8))
        a=model.input_vector(obs,mom,action,h,*stats);changed=obs.clone();changed[:,:,[1,2,18,19]]+=1000;b=model.input_vector(changed,mom,action,h,*stats)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a[:,:,12:14],obs[:,:,12:14]))
        self.assertEqual(a.shape[-1],30)

    def test_geometry_addition_is_input_only_and_parameter_matched(self):
        torch.manual_seed(42);a=RepresentationProbe('ghost','keep_static_no_geom')
        torch.manual_seed(42);b=RepresentationProbe('ghost','keep_static_geom')
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values())))
        obs=torch.randn(2,8,20);mom=torch.zeros(2,8,8);action=torch.zeros(2,8,1);h=torch.tensor([1,16]);stats=(torch.zeros(20),torch.ones(20),torch.zeros(8),torch.ones(8))
        xa=a.input_vector(obs,mom,action,h,*stats);xb=b.input_vector(obs,mom,action,h,*stats)
        self.assertTrue(torch.equal(xa[:,:,:18],xb[:,:,:18]));self.assertFalse(torch.equal(xa[:,:,18:20],xb[:,:,18:20]))


if __name__=='__main__':unittest.main()
