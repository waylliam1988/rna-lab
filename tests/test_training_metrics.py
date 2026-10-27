import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rna2d.training import collate, evaluate, multi_partner_conflict_diagnostic


class MultiPartnerConflictDiagnosticTests(unittest.TestCase):
    def test_counts_all_legal_candidates_without_label_masking(self):
        sequence = "ACGAUUG"
        scores = [[0.9] * len(sequence) for _ in sequence]
        result = multi_partner_conflict_diagnostic(scores, sequence, threshold=0.5)
        # Position 0 has two legal AU candidates (4, 5); the denominator is
        # all seven nucleotides, including positions without multiple options.
        self.assertEqual(result["multi_partner_conflict_endpoints"], 1)
        self.assertAlmostEqual(result["multi_partner_conflict_endpoint_rate"], 1 / 7)

    def test_threshold_and_candidate_rules(self):
        sequence = "ACGAUUG"
        scores = [[0.0] * len(sequence) for _ in sequence]
        scores[0][4] = 0.8
        scores[0][5] = 0.49
        result = multi_partner_conflict_diagnostic(scores, sequence, threshold=0.5)
        self.assertEqual(result["multi_partner_conflict_endpoints"], 0)
        self.assertEqual(result["multi_partner_conflict_endpoint_rate"], 0.0)

    def test_evaluation_diagnostic_includes_unknown_positions(self):
        row = {
            "id": "unknown-mask",
            "sequence": "ACGAUUG",
            "observed_mask": [False] * 7,
            "pairs": [],
        }
        tokens, labels, mask, rows = collate([row])
        logits = torch.full((1, 7, 7), -10.0)
        logits[0, 0, 4] = logits[0, 4, 0] = 10.0
        logits[0, 0, 5] = logits[0, 5, 0] = 10.0

        class FixedModel(torch.nn.Module):
            def forward(self, _tokens):
                return logits

        summary, details = evaluate(FixedModel(), [(tokens, labels, mask, rows)], "cpu")
        self.assertEqual(details[0]["multi_partner_conflict_endpoints"], 1)
        self.assertAlmostEqual(summary["macro_multi_partner_conflict_endpoint_rate"], 1 / 7)
        self.assertEqual(details[0]["evaluated_predicted_pairs"], [])

    def test_evaluation_crops_batch_padding_per_sequence(self):
        short = {
            "id": "short",
            "sequence": "ACGAUUG",
            "observed_mask": [False] * 7,
            "pairs": [],
        }
        long = {
            "id": "long",
            "sequence": "ACGAUUGCA",
            "observed_mask": [False] * 9,
            "pairs": [],
        }
        tokens, labels, mask, rows = collate([short, long])
        logits = torch.full((2, 9, 9), -10.0)
        logits[0, 0, 4] = logits[0, 4, 0] = 10.0
        logits[0, 0, 5] = logits[0, 5, 0] = 10.0

        class FixedModel(torch.nn.Module):
            def forward(self, _tokens):
                return logits

        summary, details = evaluate(FixedModel(), [(tokens, labels, mask, rows)], "cpu")
        self.assertEqual([detail["id"] for detail in details], ["short", "long"])
        self.assertEqual(details[0]["multi_partner_conflict_endpoints"], 1)
        self.assertAlmostEqual(details[0]["multi_partner_conflict_endpoint_rate"], 1 / 7)
        self.assertEqual(details[1]["multi_partner_conflict_endpoints"], 0)
        self.assertAlmostEqual(summary["macro_multi_partner_conflict_endpoint_rate"], 1 / 7 / 2)

    def test_rejects_mismatched_matrices_and_thresholds(self):
        with self.assertRaisesRegex(ValueError, "matching the sequence"):
            multi_partner_conflict_diagnostic([[0.5]], "ACGU")
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            multi_partner_conflict_diagnostic([[0.5] * 4 for _ in range(4)], "ACGU", 1.1)


if __name__ == "__main__":
    unittest.main()
