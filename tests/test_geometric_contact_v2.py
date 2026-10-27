import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.physics.interface import PhysicalInterface
from rna2d.physics.geometric_contact_v2 import geometric_contacts


class GeometricContactTests(unittest.TestCase):
    def test_current_geometry_and_translation_invariance(self):
        engine=PhysicalInterface();q=torch.zeros(1,8,6);q[0,:,0]=torch.arange(8)*.2;q[:,:,5]=1;m=torch.ones(1,8);j=torch.ones(1,8,8)*.1
        a=engine.initial_state(q,m);shift=q.clone();shift[:,:,:3]+=3.;b=engine.initial_state(shift,m)
        expanded=q.clone();expanded[:,:,:3]*=2;c=engine.initial_state(expanded,m)
        with torch.no_grad():
            x=geometric_contacts(engine,a,j,m);y=geometric_contacts(engine,b,j,m);z=geometric_contacts(engine,c,j,m)
        self.assertEqual(x.shape,(1,2));self.assertTrue(torch.allclose(x,y,atol=1e-6,rtol=1e-5))
        self.assertGreater(float(x[0,0]),float(z[0,0]));self.assertTrue(torch.isfinite(x).all())


if __name__=='__main__':unittest.main()
