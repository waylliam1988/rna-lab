import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_gate3_probe import score,action_effect


class CausalMetricTests(unittest.TestCase):
    def setUp(self):
        self.meta=[dict(split_group_id=str(i//3),amplitude=[-.4,0.,.4][i%3]) for i in range(6)]
        self.truth=np.zeros((6,4,5))
        for i,m in enumerate(self.meta):
            for h in range(4):self.truth[i,h,[0,3,4]]=m['amplitude']*(h+1)
        self.scale=np.ones((4,5));self.current=np.zeros((6,5))

    def test_perfect_prediction_and_persistence_skill(self):
        exact=score(self.truth,self.truth,self.meta,self.scale,self.current)
        zero=score(np.zeros_like(self.truth),self.truth,self.meta,self.scale,self.current)
        for a,b in zip(exact,zero):
            self.assertEqual(a['skill_vs_persistence'],1.)
            self.assertEqual(b['skill_vs_persistence'],0.)
            self.assertEqual(a['sign_accuracy_by_dynamic_channel'],[1.,1.,1.])

    def test_action_effect_zero_and_perfect(self):
        exact=action_effect(self.truth,self.truth,self.meta,self.scale)
        zero=action_effect(np.zeros_like(self.truth),self.truth,self.meta,self.scale)
        for a,b in zip(exact,zero):
            self.assertEqual(a['skill_vs_zero_action_effect'],1.)
            self.assertEqual(b['skill_vs_zero_action_effect'],0.)


if __name__=='__main__':unittest.main()
