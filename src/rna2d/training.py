import hashlib
import json
import math
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .data import CANONICAL, crossing_pairs, greedy_decode, pair_metrics, observable_pairs


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


class Records(Dataset):
    def __init__(self, path, limit=None):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f'Empty dataset: {path}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows):
    n = max(len(r['sequence']) for r in rows)
    tokens = torch.full((len(rows), n), 4, dtype=torch.long)
    labels = torch.zeros(len(rows), n, n)
    mask = torch.zeros(len(rows), n, n, dtype=torch.bool)
    for b, r in enumerate(rows):
        seq = r['sequence']
        tokens[b, :len(seq)] = torch.tensor(['ACGU'.index(c) for c in seq])
        ids=tokens[b,:len(seq)]
        compatibility=torch.tensor([[a+c in CANONICAL for c in 'ACGU'] for a in 'ACGU'])
        observed=torch.tensor(r['observed_mask'],dtype=torch.bool)
        mask[b,:len(seq),:len(seq)]=torch.triu(compatibility[ids[:,None],ids[None,:]] &
                                                       observed[:,None] & observed[None,:],diagonal=4)
        for i, j in r['pairs']:
            labels[b, i, j] = labels[b, j, i] = 1
    return tokens, labels, mask, rows


def multi_partner_conflict_diagnostic(scores, sequence, threshold=0.5):
    """Count endpoints scoring at least two legal partners above threshold.

    This diagnostic deliberately uses every canonical, minimum-separation
    candidate, independent of observed-label masks or the true pair set. The
    rate denominator is every nucleotide in the sequence.
    """
    n = len(sequence)
    if n == 0 or len(scores) != n or any(len(row) != n for row in scores):
        raise ValueError("Scores must be a non-empty square matrix matching the sequence")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Probability threshold must be finite and between 0 and 1")
    conflict_endpoints = 0
    for i in range(n):
        high_scoring_partners = sum(
            float(scores[i][j]) >= threshold
            for j in range(n)
            if abs(i - j) >= 4 and sequence[i] + sequence[j] in CANONICAL
        )
        conflict_endpoints += high_scoring_partners >= 2
    return {
        "multi_partner_conflict_endpoints": int(conflict_endpoints),
        "multi_partner_conflict_endpoint_rate": conflict_endpoints / n,
    }


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    details, tp, fp, fn = [], 0, 0, 0
    for tokens, labels, mask, rows in loader:
        scores = model(tokens.to(device)).sigmoid().cpu().tolist()
        for matrix, r in zip(scores, rows):
            pred = greedy_decode(matrix, r['sequence'], threshold)
            assessed_pred = observable_pairs(pred, r['observed_mask'])
            truth = observable_pairs(r['pairs'], r['observed_mask'])
            metrics = pair_metrics(assessed_pred, truth)
            cross = pair_metrics(crossing_pairs(assessed_pred), crossing_pairs(truth))
            conflict = multi_partner_conflict_diagnostic(matrix, r['sequence'], threshold)
            details.append({'id': r['id'], 'length': len(r['sequence']), **metrics,
                            'crossing': cross, 'predicted_pairs': pred, 'evaluated_predicted_pairs': assessed_pred,
                            **conflict,
                            'evaluated_position_fraction': sum(r['observed_mask']) / len(r['observed_mask']),
                            'true_crossing_pairs': len(crossing_pairs(truth))})
            tp += metrics['tp']; fp += metrics['fp']; fn += metrics['fn']
    cross_rows = [r for r in details if r['true_crossing_pairs']]
    summary = {'samples': len(details), 'threshold': threshold,
               'evaluation_scope': 'observable label positions only; decode first, then mask unknown endpoints',
               'mean_evaluated_position_fraction': sum(r['evaluated_position_fraction'] for r in details) / len(details),
               'macro_f1': sum(r['f1'] for r in details) / len(details),
               'macro_precision': sum(r['precision'] for r in details) / len(details),
               'macro_recall': sum(r['recall'] for r in details) / len(details),
               'macro_multi_partner_conflict_endpoint_rate':
                   sum(r['multi_partner_conflict_endpoint_rate'] for r in details) / len(details),
               'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
               'crossing_macro_f1_on_positive_samples':
                   sum(r['crossing']['f1'] for r in cross_rows) / len(cross_rows) if cross_rows else None,
               'crossing_positive_samples': len(cross_rows), 'length_bins': {}}
    for lower, upper in [(0, 64), (65, 128), (129, 256), (257, 512), (513, 1000000)]:
        group = [r for r in details if lower <= r['length'] <= upper]
        if group:
            summary['length_bins'][f'{lower}-{upper}'] = {'samples': len(group), 'macro_f1': sum(r['f1'] for r in group) / len(group)}
    return summary, details


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
