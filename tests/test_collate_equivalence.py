import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.training import collate
from rna2d.data import CANONICAL


class CollateTests(unittest.TestCase):
    def test_vectorized_mask_matches_declared_label_contract(self):
        rows=[dict(sequence='GAAACUGCAU',pairs=[[0,4]],observed_mask=[True]*7+[False]*3),
              dict(sequence='AGCUA',pairs=[],observed_mask=[True,False,True,True,True])]
        tokens,labels,mask,_=collate(rows)
        for b,r in enumerate(rows):
            for i in range(tokens.shape[1]):
                for j in range(tokens.shape[1]):
                    expected=(i<len(r['sequence']) and j<len(r['sequence']) and j-i>=4 and
                              r['sequence'][i]+r['sequence'][j] in CANONICAL and r['observed_mask'][i] and r['observed_mask'][j])
                    self.assertEqual(bool(mask[b,i,j]),expected)


if __name__=='__main__':unittest.main()
