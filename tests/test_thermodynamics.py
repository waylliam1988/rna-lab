import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.thermodynamics import (PriorStore, compute_prior, pair_complementarity,
                                validate_probabilities, ThermodynamicModel)


class ThermodynamicTests(unittest.TestCase):
    def test_one_based_upper_triangle_is_mirrored_correctly(self):
        p,_=compute_prior('GGGAAACCC')
        self.assertEqual(p.shape,(9,9))
        self.assertGreater(p[0,8],0.5)
        self.assertEqual(p[0,8],p[8,0])
        self.assertTrue(np.all(p.sum(1)<=1.00001))

    def test_cache_padding_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            store=PriorStore(folder)
            p=store.get('GGGAAACCC',create=True)
            store.get('AAAAA',create=True)
            tokens=torch.tensor([[2,2,2,0,0,0,1,1,1],[0,0,0,0,0,4,4,4,4]])
            scores=ThermodynamicModel(store)(tokens).sigmoid()
            self.assertTrue(np.allclose(scores[0],p,atol=1e-7))
            self.assertEqual(float(scores[1,5:].sum()),0)
            matrix=next(Path(folder).glob('*.npy'))
            matrix.write_bytes(b'corrupt')
            fresh=PriorStore(folder)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                for sequence in ['GGGAAACCC','AAAAA']:fresh.get(sequence)

    def test_unknown_pairs_not_false_positives_or_truth(self):
        row={'sequence':'GGGAAACCC','observed_mask':[True]*8+[False],'pairs':[[0,8],[1,7]]}
        result=pair_complementarity(row,[[0,8],[1,7]],[[1,7],[2,6]])
        self.assertEqual(result['counts']['both_correct'],1)
        self.assertEqual(result['counts']['cnn_only_false_positive'],1)
        self.assertEqual(sum(result['counts'][k] for k in ['both_correct','thermo_only_correct','cnn_only_correct','neither_recovers']),1)

    def test_invalid_matrices_rejected(self):
        for p in [np.ones((3,3)),np.full((3,3),np.nan),np.zeros((2,3)),np.triu(np.ones((3,3)),1)]:
            with self.assertRaises(ValueError):validate_probabilities(p,3)


if __name__ == '__main__':unittest.main()
