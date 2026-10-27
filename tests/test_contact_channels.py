import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.physics.interface import PhysicalInterface
from rna2d.physics.reference import compute_rna_force_field_fast


class ContactChannelTests(unittest.TestCase):
    def test_coordinates_change_aux_contact_but_not_input_J_density(self):
        engine=PhysicalInterface();q=torch.zeros(1,8,6);q[0,:,0]=torch.arange(8)*.2;q[:,:,5]=1
        m=torch.ones(1,8);j=torch.ones(1,8,8)*.1;j[:,torch.arange(8),torch.arange(8)]=0
        a=engine.initial_state(q,m);q2=q.clone();q2[:,:,:3]*=2;b=engine.initial_state(q2,m)
        with torch.no_grad():
            oa=engine.observe(a,j,m)[0];ob=engine.observe(b,j,m)[0]
            self.assertTrue(torch.equal(oa[:,1:3],ob[:,1:3]))
            self.assertTrue((oa[:,1:3]>0).all())
            _,_,aa=compute_rna_force_field_fast(a.q,j,m,engine.ff_params,return_aux=True)
            _,_,ab=compute_rna_force_field_fast(b.q,j,m,engine.ff_params,return_aux=True)
            self.assertGreater(float(aa['contact_mean'][0]),float(ab['contact_mean'][0]))
            half=engine.observe(a,j*.5,m)[0]
            self.assertTrue(torch.allclose(half[:,1:3],oa[:,1:3]*.5))


if __name__=='__main__':unittest.main()
