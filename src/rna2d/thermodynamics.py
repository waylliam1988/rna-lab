"""Sequence-only ViennaRNA priors and audits; never produce training labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import observable_pairs, pair_metrics, sequence_hash

SETTINGS = {
    'engine': 'ViennaRNA', 'version': '2.7.2', 'parameters': 'RNA_Turner2004',
    'temperature_C': 37.0, 'salt_M': 1.021, 'dangles': 2,
    'min_loop_size': 3, 'noLP': 0, 'noGU': 0, 'noGUclosure': 0,
    'circular': 0, 'gquad': 0, 'compute_bpp': 1,
    'model': 'single isolated linear RNA; no experimental constraints or ligand/Mg binding model',
    'condition_status': 'assumed fixed conditions, not recovered experimental buffer',
}


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_probabilities(matrix, length):
    p = np.asarray(matrix)
    if p.shape != (length, length) or not np.isfinite(p).all():
        raise ValueError('Prior must be a finite sequence-length square matrix')
    if (p < 0).any() or (p > 1).any() or not np.allclose(p, p.T, atol=1e-7):
        raise ValueError('Prior must be symmetric probabilities in [0,1]')
    if not np.allclose(np.diag(p), 0) or (p.sum(axis=1) > 1.00001).any():
        raise ValueError('Invalid base-pair probability diagonal or endpoint probability mass')


def compute_prior(sequence):
    import RNA
    if RNA.__version__ != SETTINGS['version']:
        raise ValueError('ViennaRNA version differs from the frozen prior engine')
    if not sequence or set(sequence) - set('ACGU'):
        raise ValueError('Expected a nonempty canonical RNA sequence')
    RNA.params_load_RNA_Turner2004()
    md = RNA.md()
    for name, value in {'temperature': SETTINGS['temperature_C'], 'salt': SETTINGS['salt_M'],
                        **{key: SETTINGS[key] for key in ('dangles', 'min_loop_size', 'noLP',
                           'noGU', 'noGUclosure', 'gquad', 'compute_bpp')}}.items():
        setattr(md, name, value)
    md.circ = SETTINGS['circular']
    fc = RNA.fold_compound(sequence, md)
    structure, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    _, ensemble_energy = fc.pf()
    # SWIG bpp() is 1-based and upper triangular, NOT a symmetric N x N matrix.
    upper = np.triu(np.asarray(fc.bpp(), dtype=np.float64)[1:, 1:], k=1)
    matrix = (upper + upper.T).astype(np.float32)
    validate_probabilities(matrix, len(sequence))
    return matrix, {'mfe_structure_diagnostic_only': structure, 'mfe_kcal_mol': mfe,
                    'ensemble_free_energy_kcal_mol': ensemble_energy}


class PriorStore:
    """Content-addressed matrices; loading verifies receipt and sequence hashes."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.memo = {}

    def get(self, sequence, *, create=False):
        key = sequence_hash(sequence)
        if key in self.memo:
            return self.memo[key]
        stem = self.folder / f'{json_hash(SETTINGS)}_{key}'
        receipt_path, matrix_path = stem.with_suffix('.json'), stem.with_suffix('.npy')
        if not receipt_path.exists() and create:
            if matrix_path.exists():
                raise ValueError('Orphan cache matrix; review instead of overwriting')
            matrix, info = compute_prior(sequence)
            np.save(matrix_path, matrix, allow_pickle=False)
            receipt = {'settings': SETTINGS, 'sequence_sha256': key, 'length': len(sequence),
                       'matrix_sha256': hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                       'role': 'computed input feature, NOT experimental structure label', **info}
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['settings'] != SETTINGS or receipt['sequence_sha256'] != key or receipt['length'] != len(sequence):
            raise ValueError('Prior receipt mismatch')
        if hashlib.sha256(matrix_path.read_bytes()).hexdigest() != receipt['matrix_sha256']:
            raise ValueError('Prior matrix hash mismatch')
        matrix = np.load(matrix_path, allow_pickle=False)
        validate_probabilities(matrix, len(sequence))
        self.memo[key] = matrix
        return matrix

    def batch(self, tokens):
        matrices = []
        for row in tokens.detach().cpu().tolist():
            n = sum(t != 4 for t in row)
            if not n or any(t == 4 for t in row[:n]) or any(t != 4 for t in row[n:]):
                raise ValueError('Expected nonempty sequence followed only by padding')
            seq = ''.join('ACGU'[t] for t in row[:n])
            padded = np.zeros((len(row), len(row)), dtype=np.float32)
            padded[:n, :n] = self.get(seq)
            matrices.append(padded)
        return torch.as_tensor(np.stack(matrices), device=tokens.device)


class ThermodynamicModel(torch.nn.Module):
    """Adapter to the existing identical sigmoid/greedy-decode evaluator."""
    def __init__(self, store):
        super().__init__()
        self.store = store

    def forward(self, tokens):
        return torch.logit(self.store.batch(tokens))


def pair_type(sequence, edge):
    bases = ''.join(sorted(sequence[i] for i in edge))
    return {'AU': 'AU', 'CG': 'GC', 'GU': 'GU'}.get(bases, 'outside_AU_GC_GU')


def pair_complementarity(row, thermo_pred, cnn_pred):
    """Decode BEFORE applying observed-mask; report recovery and false positives."""
    truth = {tuple(sorted(p)) for p in observable_pairs(row['pairs'], row['observed_mask'])}
    thermo = {tuple(sorted(p)) for p in observable_pairs(thermo_pred, row['observed_mask'])}
    cnn = {tuple(sorted(p)) for p in observable_pairs(cnn_pred, row['observed_mask'])}
    categories = {'both_correct': truth & thermo & cnn,
                  'thermo_only_correct': (truth & thermo) - cnn,
                  'cnn_only_correct': (truth & cnn) - thermo,
                  'neither_recovers': truth - (thermo | cnn),
                  'both_false_positive': (thermo & cnn) - truth,
                  'thermo_only_false_positive': thermo - truth - cnn,
                  'cnn_only_false_positive': cnn - truth - thermo}
    return {'counts': {k: len(v) for k, v in categories.items()},
            'edges': {k: [list(p) for p in sorted(v)] for k, v in categories.items()},
            'by_pair_type': {kind: {k: sum(pair_type(row['sequence'], p) == kind for p in v)
                                   for k, v in categories.items()}
                             for kind in ('AU', 'GC', 'GU', 'outside_AU_GC_GU')},
            'oracle_union_recall': len(truth & (thermo | cnn)) / len(truth) if truth else None,
            'thermo': pair_metrics(thermo, truth), 'cnn': pair_metrics(cnn, truth)}
