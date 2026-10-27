import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_profile_and_history_relations import (
    classify_history_relation,
    classify_profile_relation,
)


class ProfileHistoryRelationTests(unittest.TestCase):
    def test_profile_relation_never_grants_pair_training(self):
        for split in (None, "quarantine", "train", "validation", "test"):
            status, allowed = classify_profile_relation(split)
            self.assertFalse(allowed)
            self.assertTrue(status)

    def test_prior_test_is_always_marked_exposed(self):
        for current in (None, "quarantine", "train", "validation", "test"):
            status, allowed = classify_history_relation("test", current)
            self.assertEqual(status, "historically_exposed_holdout")
            self.assertFalse(allowed)

    def test_exact_non_test_alias_is_still_not_trainable(self):
        for split in ("train", "validation", "test", "quarantine"):
            status, allowed = classify_history_relation("train", split)
            self.assertIn("exact_sequence_alias", status)
            self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
