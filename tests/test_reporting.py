from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rna2d.reporting import group_macro_f1


class GroupMacroReportingTests(unittest.TestCase):
    def test_groups_are_equally_weighted_after_rna_averaging(self):
        rows = [
            {'id': 'a1', 'split_group_id': 'a'},
            {'id': 'a2', 'split_group_id': 'a'},
            {'id': 'b1', 'split_group_id': 'b'},
        ]
        details = [
            {'id': 'a1', 'f1': 1.0},
            {'id': 'a2', 'f1': 0.0},
            {'id': 'b1', 'f1': 1.0},
        ]
        score, per_group = group_macro_f1(details, rows)
        self.assertEqual(per_group, {'a': 0.5, 'b': 1.0})
        self.assertEqual(score, 0.75)

    def test_rejects_unmatched_or_empty_evaluation(self):
        with self.assertRaisesRegex(ValueError, 'corresponding group'):
            group_macro_f1([{'id': 'unknown', 'f1': 1.0}], [])
        with self.assertRaisesRegex(ValueError, 'without evaluated'):
            group_macro_f1([], [])


if __name__ == '__main__':
    unittest.main()
