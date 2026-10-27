"""Reusable masked objectives for canonical RNA pair prediction."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_pair_tensors(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
    if logits.ndim != 3 or logits.shape[-1] != logits.shape[-2]:
        raise ValueError("Pair logits must have shape (batch, length, length)")
    if labels.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("Logits, labels and pair mask must have identical shapes")
    if not logits.is_floating_point():
        raise TypeError("Pair logits must be floating point")
    if not labels.is_floating_point() and labels.dtype != torch.bool:
        raise TypeError("Pair labels must be floating point or boolean")


def balanced_pair_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Class-balanced BCE per RNA, then mean over RNAs with known labels.

    `mask` is expected to contain the upper-triangular supervised pair set used
    by :func:`rna2d.training.collate`. Unknown pairs must have mask=False.
    A positive-only sample contributes its positive term; an entirely
    unsupervised batch is rejected instead of silently becoming a zero loss.
    """
    _validate_pair_tensors(logits, labels, mask)
    losses = []
    supervised = mask.bool()
    positive = labels > 0
    for b in range(logits.shape[0]):
        pos = supervised[b] & positive[b]
        neg = supervised[b] & ~positive[b]
        terms = []
        if pos.any():
            terms.append(F.softplus(-logits[b][pos]).mean())
        if neg.any():
            terms.append(F.softplus(logits[b][neg]).mean())
        if terms:
            losses.append(torch.stack(terms).mean())
    if not losses:
        raise ValueError("No usable supervision")
    return torch.stack(losses).mean()


def partner_competition_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    return_details: bool = False,
):
    """Rank each known partner above reliable negative partners at both ends.

    For every known positive edge i-j, each endpoint with at least one
    explicitly supervised negative candidate k receives
    ``logsumexp([0, z(i,k)-z(i,j) for k in negatives])``. Pair indices are
    normalized from the upper-triangular mask, so partners to the left and
    right of a nucleotide are both considered. Unknown pairs never enter the
    candidate set.

    The mean is taken over applicable endpoints within each RNA and then over
    applicable RNAs. With no applicable endpoints the loss is a graph-connected
    zero and the details report zero denominators.
    """
    _validate_pair_tensors(logits, labels, mask)
    batch = logits.shape[0]
    valid_upper = torch.triu(mask.bool(), diagonal=1)
    positive_upper = valid_upper & (labels > 0)
    valid_symmetric = valid_upper | valid_upper.transpose(1, 2)
    positive_symmetric = positive_upper | positive_upper.transpose(1, 2)
    negative_symmetric = valid_symmetric & ~positive_symmetric

    # The experimental label contract is one canonical partner per endpoint.
    partner_counts = positive_symmetric.sum(dim=-1)
    if torch.any(partner_counts > 1):
        raise ValueError("Partner competition requires at most one known positive partner per endpoint")

    endpoint_has_positive = positive_symmetric.any(dim=-1)
    endpoint_has_negative = negative_symmetric.any(dim=-1)
    applicable = endpoint_has_positive & endpoint_has_negative
    # The model's pair head is symmetric. Mirror the declared upper triangle so
    # candidate partners on either side use the same canonical pair logit.
    pair_logits = torch.triu(logits, diagonal=1)
    pair_logits = pair_logits + pair_logits.transpose(1, 2)
    true_partner = positive_symmetric.to(torch.int64).argmax(dim=-1)
    true_logits = pair_logits.gather(2, true_partner.unsqueeze(-1)).squeeze(-1)
    differences = pair_logits - true_logits.unsqueeze(-1)
    differences = differences.masked_fill(~negative_symmetric, -torch.inf)
    with_zero = torch.cat((torch.zeros_like(differences[..., :1]), differences), dim=-1)
    endpoint_loss = torch.logsumexp(with_zero, dim=-1)
    applicable_float = applicable.to(endpoint_loss.dtype)
    endpoint_counts = applicable_float.sum(dim=-1)
    rna_loss = (endpoint_loss * applicable_float).sum(dim=-1) / endpoint_counts.clamp_min(1)
    rna_has_applicable = endpoint_counts > 0
    if rna_has_applicable.any():
        loss = rna_loss[rna_has_applicable].mean()
    else:
        loss = logits.reshape(-1)[:0].sum()
    details = {
        "applicable_rnas": int(rna_has_applicable.sum().item()),
        "applicable_endpoints": int(applicable.sum().item()),
        "candidate_negatives": int((negative_symmetric & endpoint_has_positive.unsqueeze(-1)).sum().item()),
        "positive_edges": int(positive_upper.sum().item()),
        "skipped_endpoints_without_negative": int((endpoint_has_positive & ~endpoint_has_negative).sum().item()),
    }
    return (loss, details) if return_details else loss


def pair_prediction_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    competition_weight: float = 0.0,
    return_details: bool = False,
):
    """Balanced BCE with an optional partner-ranking auxiliary term.

    `competition_weight=0` is the unchanged project baseline. This utility
    does not choose a weight or imply that the auxiliary term improves results.
    """
    if not math.isfinite(competition_weight) or competition_weight < 0:
        raise ValueError("competition_weight must be finite and non-negative")
    bce = balanced_pair_bce(logits, labels, mask)
    if competition_weight == 0:
        total = bce
        competition = logits.reshape(-1)[:0].sum()
        stats = {"applicable_rnas": 0, "applicable_endpoints": 0, "candidate_negatives": 0,
                 "positive_edges": 0, "skipped_endpoints_without_negative": 0}
    else:
        competition, stats = partner_competition_loss(logits, labels, mask, return_details=True)
        total = bce + competition_weight * competition
    details = {"loss_bce": bce, "loss_competition": competition, "competition_weight": competition_weight, **stats}
    return (total, details) if return_details else total
