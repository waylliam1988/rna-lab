import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_gate1_contract import contract_checks


class Gate1Tests(unittest.TestCase):
    def test_chronology_counterfactual_and_no_future(self):
        self.assertTrue(all(v for k,v in contract_checks().items() if k!='source_training_id'))


if __name__=='__main__':unittest.main()
