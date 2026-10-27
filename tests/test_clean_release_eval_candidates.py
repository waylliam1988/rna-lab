import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_clean_release_eval_candidates import classify_candidate_group, DisjointSet


class CleanReleaseEvaluationCandidateTests(unittest.TestCase):
    def test_frozen_split_alias_or_relation_never_becomes_new_test(self):
        for split in ("train", "validation", "test"):
            status = classify_candidate_group({split}, False, True, True, True)
            self.assertEqual(status, f"connected_to_current_{split}")

    def test_historical_test_exposure_takes_precedence(self):
        status = classify_candidate_group({"train"}, True, True, True, True)
        self.assertEqual(status, "historically_test_exposed")

    def test_missing_independence_evidence_stays_quarantined(self):
        status = classify_candidate_group(set(), False, False, True, False)
        self.assertEqual(status, "no_known_current_link_but_independence_evidence_incomplete")

    def test_fully_linked_unseen_candidate_still_requires_manual_review(self):
        status = classify_candidate_group(set(), False, True, True, True)
        self.assertEqual(status, "unlinked_candidate_requires_manual_review")

    def test_disjoint_set_merges_transitive_relations(self):
        groups = DisjointSet(4)
        groups.union(0, 1)
        groups.union(1, 2)
        self.assertEqual(groups.find(0), groups.find(2))
        self.assertNotEqual(groups.find(0), groups.find(3))


if __name__ == "__main__":
    unittest.main()
