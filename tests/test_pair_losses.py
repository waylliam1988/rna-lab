import unittest
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rna2d.losses import balanced_pair_bce, pair_prediction_loss, partner_competition_loss


def reference_balanced_loss(logits, labels, mask):
    rows = []
    for b in range(len(logits)):
        pos, neg = mask[b] & labels[b].bool(), mask[b] & ~labels[b].bool()
        terms = []
        if pos.any():
            terms.append(F.softplus(-logits[b][pos]).mean())
        if neg.any():
            terms.append(F.softplus(logits[b][neg]).mean())
        if terms:
            rows.append(torch.stack(terms).mean())
    if not rows:
        raise ValueError("No usable supervision")
    return torch.stack(rows).mean()


class PairLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)

    def test_shared_bce_matches_old_formula_and_gradients(self):
        logits_a = torch.randn(3, 7, 7, requires_grad=True)
        logits_b = logits_a.detach().clone().requires_grad_()
        labels = torch.zeros_like(logits_a)
        mask = torch.zeros_like(logits_a, dtype=torch.bool)
        mask[0, 0, 5] = mask[0, 1, 6] = mask[1, 2, 6] = True
        labels[0, 0, 5] = 1
        labels[1, 2, 6] = 1
        old_loss = reference_balanced_loss(logits_a, labels, mask)
        new_loss = balanced_pair_bce(logits_b, labels, mask)
        old_loss.backward()
        new_loss.backward()
        torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
        torch.testing.assert_close(logits_b.grad, logits_a.grad, rtol=0, atol=0)

    def test_competition_uses_left_and_right_partners_and_ignores_unknown(self):
        logits = torch.zeros(1, 7, 7, requires_grad=True)
        labels = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        positives = [(2, 5)]
        negatives = [(0, 2), (2, 4), (2, 6), (1, 5), (3, 5), (5, 6)]
        for i, j in positives + negatives:
            mask[0, i, j] = True
        labels[0, 2, 5] = 1
        with torch.no_grad():
            logits[0, 0, 2] = 0.0
            logits[0, 2, 4] = 1.0
            logits[0, 2, 6] = -1.0
            logits[0, 1, 5] = -0.5
            logits[0, 3, 5] = 0.0
            logits[0, 5, 6] = 0.4
            logits[0, 2, 5] = 0.5
            logits[0, 1, 2] = 1000.0  # Unknown despite a large score.
        loss, details = partner_competition_loss(logits, labels, mask, return_details=True)
        left = torch.logsumexp(torch.tensor([0.0, -0.5, 0.5, -1.5]), dim=0)
        right = torch.logsumexp(torch.tensor([0.0, -1.0, -0.5, -0.1]), dim=0)
        torch.testing.assert_close(loss, (left + right) / 2)
        self.assertEqual(details["applicable_rnas"], 1)
        self.assertEqual(details["applicable_endpoints"], 2)
        self.assertEqual(details["candidate_negatives"], 6)
        self.assertEqual(details["positive_edges"], 1)

    def test_competition_gradients_favor_true_partner(self):
        labels = torch.zeros(1, 6, 6)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[0, 0, 4] = True
        mask[0, 0, 5] = True
        mask[0, 1, 4] = True
        labels[0, 0, 4] = 1
        logits = torch.zeros_like(labels, requires_grad=True)
        loss = partner_competition_loss(logits, labels, mask)
        loss.backward()
        self.assertLess(logits.grad[0, 0, 4].item(), 0)
        self.assertGreater(logits.grad[0, 0, 5].item(), 0)
        raised_true = logits.detach().clone()
        raised_true[0, 0, 4] += 1
        lowered_negative = logits.detach().clone()
        lowered_negative[0, 0, 5] -= 1
        self.assertLess(partner_competition_loss(raised_true, labels, mask), loss.detach())
        self.assertLess(partner_competition_loss(lowered_negative, labels, mask), loss.detach())

    def test_no_applicable_endpoint_is_zero_and_backward_safe(self):
        logits = torch.randn(2, 5, 5, requires_grad=True)
        labels = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, 0, 4] = True  # Reliable negative only; no positive edge.
        labels[1, 1, 4] = 1  # Positive edge has no reliable negative candidate.
        mask[1, 1, 4] = True
        loss, details = partner_competition_loss(logits, labels, mask, return_details=True)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(details["applicable_rnas"], 0)
        self.assertEqual(details["skipped_endpoints_without_negative"], 2)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_unknown_pairs_do_not_affect_competition_and_extremes_are_finite(self):
        labels = torch.zeros(1, 6, 6)
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[0, 0, 4] = mask[0, 0, 5] = mask[0, 1, 4] = True
        labels[0, 0, 4] = 1
        logits = torch.zeros_like(labels)
        base = partner_competition_loss(logits, labels, mask)
        changed_unknown = logits.clone()
        changed_unknown[0, 1, 5] = -10000
        torch.testing.assert_close(partner_competition_loss(changed_unknown, labels, mask), base)
        extremes = logits.clone()
        extremes[0, 0, 4] = 10000
        extremes[0, 0, 5] = -10000
        extremes.requires_grad_()
        extreme_loss = partner_competition_loss(extremes, labels, mask)
        extreme_loss.backward()
        self.assertTrue(torch.isfinite(extreme_loss))
        self.assertTrue(torch.isfinite(extremes.grad).all())

    def test_zero_weight_is_exact_bce_and_positive_weight_is_reported(self):
        logits = torch.randn(1, 6, 6, requires_grad=True)
        labels = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, 0, 4] = mask[0, 0, 5] = mask[0, 1, 4] = True
        labels[0, 0, 4] = 1
        bce, bce_details = pair_prediction_loss(logits, labels, mask, return_details=True)
        base = balanced_pair_bce(logits, labels, mask)
        torch.testing.assert_close(bce, base, rtol=0, atol=0)
        self.assertEqual(bce_details["competition_weight"], 0.0)
        mixed, mixed_details = pair_prediction_loss(logits, labels, mask, competition_weight=0.2, return_details=True)
        torch.testing.assert_close(mixed, mixed_details["loss_bce"] + 0.2 * mixed_details["loss_competition"])
        self.assertEqual(mixed_details["applicable_endpoints"], 2)

    def test_rejects_multiple_positive_partners(self):
        logits = torch.zeros(1, 6, 6)
        labels = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        labels[0, 0, 4] = labels[0, 0, 5] = 1
        mask[0, 0, 4] = mask[0, 0, 5] = True
        with self.assertRaisesRegex(ValueError, "at most one"):
            partner_competition_loss(logits, labels, mask)

    def test_rejects_invalid_competition_weight_and_empty_bce(self):
        logits = torch.zeros(1, 4, 4)
        labels = torch.zeros_like(logits)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "No usable supervision"):
            balanced_pair_bce(logits, labels, mask)
        mask[0, 0, 3] = True
        for weight in (-1.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                pair_prediction_loss(logits, labels, mask, competition_weight=weight)


if __name__ == "__main__":
    unittest.main()
