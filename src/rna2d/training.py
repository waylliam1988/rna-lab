import hashlib
import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .data import CANONICAL, crossing_pairs, greedy_decode, pair_metrics


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
        for i in range(len(seq)):
            for j in range(i + 4, len(seq)):
                mask[b, i, j] = (seq[i] + seq[j] in CANONICAL and
                                 r['observed_mask'][i] and r['observed_mask'][j])
        for i, j in r['pairs']:
            labels[b, i, j] = labels[b, j, i] = 1
    return tokens, labels, mask, rows


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    details, tp, fp, fn = [], 0, 0, 0
    for tokens, labels, mask, rows in loader:
        scores = model(tokens.to(device)).sigmoid().cpu().tolist()
        for matrix, r in zip(scores, rows):
            pred = greedy_decode(matrix, r['sequence'], threshold)
            metrics = pair_metrics(pred, r['pairs'])
            cross = pair_metrics(crossing_pairs(pred), crossing_pairs(r['pairs']))
            details.append({'id': r['id'], 'length': len(r['sequence']), **metrics,
                            'crossing': cross, 'predicted_pairs': pred,
                            'true_crossing_pairs': len(crossing_pairs(r['pairs']))})
            tp += metrics['tp']; fp += metrics['fp']; fn += metrics['fn']
    cross_rows = [r for r in details if r['true_crossing_pairs']]
    summary = {'samples': len(details), 'threshold': threshold,
               'macro_f1': sum(r['f1'] for r in details) / len(details),
               'macro_precision': sum(r['precision'] for r in details) / len(details),
               'macro_recall': sum(r['recall'] for r in details) / len(details),
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
