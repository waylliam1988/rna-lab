from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rna2d.data import parse_bpseq, pair_metrics, crossing_pairs, greedy_decode, observable_pairs


class DataContractTests(unittest.TestCase):
    def test_parse_and_normalize(self):
        seq, pairs, ts = parse_bpseq('# header\n1 G 5\n2 A 0\n3 A 0\n4 A 0\n5 T 1')
        self.assertEqual((seq, pairs, ts), ('GAAAU', [[0, 4]], 1))

    def test_reject_asymmetric_and_missing_indices(self):
        for text in ['1 G 2\n2 C 0', '1 A 0\n3 U 0', '1 A 1']:
            with self.assertRaises(ValueError): parse_bpseq(text)

    def test_exact_pair_metrics(self):
        m = pair_metrics([[0, 5], [1, 4]], [[0, 5], [2, 6]])
        self.assertEqual((m['tp'], m['fp'], m['fn'], m['f1']), (1, 1, 1, 0.5))
        self.assertEqual(pair_metrics([], [])['f1'], 1.0)
        self.assertEqual(pair_metrics([], [[0, 4]])['f1'], 0.0)

    def test_unknown_endpoints_are_not_counted_as_false_positives(self):
        pred = [[0, 5], [1, 4]]
        mask = [True, True, True, True, False, True]
        assessed = observable_pairs(pred, mask)
        self.assertEqual(assessed, [[0, 5]])
        self.assertEqual(pair_metrics(assessed, [[0, 5]])['fp'], 0)

    def test_crossing_not_nested(self):
        self.assertEqual(crossing_pairs([[0, 5], [2, 7], [3, 4]]), {(0, 5), (2, 7)})
        self.assertEqual(crossing_pairs([[0, 7], [2, 5]]), set())

    def test_decoder_one_partner_and_crossings(self):
        scores = [[0.0] * 8 for _ in range(8)]
        scores[0][5] = 0.9; scores[2][7] = 0.8; scores[0][7] = 0.7
        self.assertEqual(greedy_decode(scores, 'GAGAAUAU'), [[0, 5], [2, 7]])


if __name__ == '__main__':
    unittest.main()
