import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.physics.diagnostics import baseline_at_horizon,skill


class SignalTests(unittest.TestCase):
    def test_linear_baseline_uses_only_current_and_previous(self):
        # Contacts/energy remain linear; Rg/force transform uses log1p.
        a=np.zeros((1,33,5));a[0,:,4]=np.arange(33)*.01
        current,target,pred=baseline_at_horizon(a,8)
        self.assertTrue(np.allclose(pred,target))
        b=a.copy();b[:,17:]+=1
        c,t,p=baseline_at_horizon(b,8)
        self.assertTrue(np.array_equal(c,current));self.assertTrue(np.array_equal(p,pred))
        self.assertFalse(np.array_equal(t,target))

    def test_constant_baseline_skill_is_undefined(self):
        self.assertIsNone(skill(0,0));self.assertEqual(skill(1,2),.5)


if __name__=='__main__':unittest.main()
