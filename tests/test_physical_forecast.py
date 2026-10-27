import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.physics.forecast import PhysicalForecast,window_batch


class ForecastTests(unittest.TestCase):
    def test_window_target_is_next_state_and_no_future_in_input(self):
        episodes=torch.arange(20).reshape(2,10,1)
        x,y=window_batch(episodes,[0,1],[2,4],3)
        self.assertTrue(torch.equal(y,x+1))
        self.assertEqual(x[:,0,0].tolist(),[2,14])

    def test_reset_matches_independent_calls_and_memory_affects_output(self):
        torch.set_num_threads(1);torch.manual_seed(22);model=PhysicalForecast().eval()
        # Original prediction head is zero-initialized; nonzero probe exposes recurrence.
        torch.nn.init.normal_(model.ghost.cortex.phys_hallucination.weight,std=.01)
        x=torch.randn(2,3,18)
        with torch.no_grad():
            reset=model(x,True)
            independent=torch.cat([model(x[:,i:i+1]) for i in range(3)],dim=1)
            carry=model(x)
        self.assertTrue(torch.equal(reset,independent))
        self.assertTrue(torch.equal(carry[:,0],reset[:,0]))
        self.assertGreater((carry[:,1:]-reset[:,1:]).abs().max().item(),0.)
        self.assertTrue(torch.equal(model(x),model(x)))


if __name__=='__main__':unittest.main()
