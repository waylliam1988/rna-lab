import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_comparative_candidate_relations import compare_pair_sets, pair_signature


class ComparativeCandidateRelationTests(unittest.TestCase):
    def test_pair_signature_is_order_independent(self):
        left = pair_signature([[1, 8], [3, 6]], 10)
        right = pair_signature([[6, 3], [8, 1]], 10)
        self.assertEqual(left, right)
        self.assertEqual(compare_pair_sets(left, right), "exact_pair_set_match")

    def test_pair_signature_rejects_invalid_indices_and_duplicates(self):
        for pairs in ([[0, 10]], [[2, 2]], [[0, 4], [4, 0]]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                pair_signature(pairs, 10)

    def test_raw_pair_difference_is_not_called_a_label_verdict(self):
        status = compare_pair_sets(pair_signature([[0, 9]], 10), pair_signature([[1, 8]], 10))
        self.assertEqual(status, "different_pair_set")


if __name__ == "__main__":
    unittest.main()
