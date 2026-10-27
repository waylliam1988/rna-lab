import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_encoder_lr_scan import learning_rate


class LearningRateTests(unittest.TestCase):
    def test_fixed_lr_and_one_warmup(self):
        self.assertEqual(learning_rate(.001,1,0),.001)
        self.assertAlmostEqual(learning_rate(.001,1,55),.001/55)
        self.assertEqual(learning_rate(.001,55,55),.001)
        self.assertEqual(learning_rate(.001,550,55),.001)


if __name__=='__main__':unittest.main()
