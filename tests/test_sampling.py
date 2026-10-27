import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rna2d.sampling import group_balanced_sample_indices, sample_schedule_sha256


class GroupSamplingTests(unittest.TestCase):
    def test_schedule_is_reproducible_and_batch_aligned(self):
        groups = ["a"] * 7 + ["b"] * 2 + ["c"] * 4
        first = group_balanced_sample_indices(groups, updates=25, batch_size=4, seed=42)
        second = group_balanced_sample_indices(groups, updates=25, batch_size=4, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 25)
        self.assertTrue(all(len(batch) == 4 for batch in first))
        self.assertTrue(all(0 <= i < len(groups) for batch in first for i in batch))

    def test_group_uniform_schedule_does_not_follow_group_size(self):
        groups = ["large"] * 90 + ["small"] * 10
        schedule = group_balanced_sample_indices(groups, updates=2000, batch_size=1, seed=7)
        large = sum(groups[batch[0]] == "large" for batch in schedule)
        self.assertGreater(large, 900)
        self.assertLess(large, 1100)

    def test_schedule_hash_uses_record_order(self):
        groups = ["a", "b"]
        schedule = group_balanced_sample_indices(groups, updates=4, batch_size=2, seed=2)
        ids = ["rna0", "rna1"]
        self.assertEqual(sample_schedule_sha256(schedule, ids), sample_schedule_sha256(schedule, ids))
        self.assertNotEqual(sample_schedule_sha256(schedule, ids), sample_schedule_sha256(schedule, ids[::-1]))

    def test_rejects_empty_or_invalid_inputs(self):
        for kwargs in ({"updates": 0, "batch_size": 1}, {"updates": 1, "batch_size": 0}):
            with self.assertRaises(ValueError):
                group_balanced_sample_indices(["a"], seed=1, **kwargs)
        with self.assertRaisesRegex(ValueError, "empty"):
            group_balanced_sample_indices([], updates=1, batch_size=1, seed=1)
        with self.assertRaisesRegex(ValueError, "split group"):
            group_balanced_sample_indices([None], updates=1, batch_size=1, seed=1)


if __name__ == "__main__":
    unittest.main()
