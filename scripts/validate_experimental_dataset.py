"""Validate label contracts and declared split isolation before any training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.data import CANONICAL
from rna2d.evidence import pairing_evidence_issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/experimental/paired_v0_2')
    args = p.parse_args()
    seen = {key: {} for key in ['sequence', 'pdb_id', 'split_group_id', 'homology_component']}
    ids_to_split, summary = {}, {}
    all_rows = []
    for split in ['train', 'validation', 'test']:
        rows = [json.loads(line) for line in (args.dataset_dir / f'{split}.jsonl').read_text().splitlines() if line]
        if not rows: raise ValueError(f'Empty split: {split}')
        for r in rows:
            issues = pairing_evidence_issues(r)
            if issues: raise ValueError(f'{r["id"]}: {issues}')
            seq, mask = r['sequence'], r['observed_mask']
            if len(seq) != len(mask) or not set(seq) <= set('ACGU'): raise ValueError('Invalid sequence/mask')
            if hashlib.sha256(seq.encode()).hexdigest() != r['sequence_sha256']: raise ValueError('Sequence hash mismatch')
            if r['cif_sha256'] != r['source_download_receipt']['sha256']: raise ValueError('Coordinate provenance mismatch')
            used = set()
            for i, j in r['pairs']:
                if not 0 <= i < j < len(seq) or j-i < 4: raise ValueError('Invalid pair indices')
                if seq[i] + seq[j] not in CANONICAL or not mask[i] or not mask[j]: raise ValueError('Pair/label mask mismatch')
                if i in used or j in used: raise ValueError('Multiple canonical partners')
                used.update((i, j))
            if not r['homology_components']: raise ValueError('Unknown homology entered training splits')
            for field, values in [('sequence', [seq]), ('pdb_id', [r['pdb_id']]),
                                  ('split_group_id', [r['split_group_id']]), ('homology_component', r['homology_components'])]:
                for value in values:
                    previous = seen[field].get(value)
                    if previous and previous != split: raise ValueError(f'Cross-split leak in {field}')
                    seen[field][value] = split
            ids_to_split[r['id']] = split
        summary[split] = {'samples': len(rows), 'pairs': sum(len(r['pairs']) for r in rows),
                          'mean_label_coverage': sum(sum(r['observed_mask']) / len(r['sequence']) for r in rows) / len(rows)}
        all_rows.extend(rows)
    for line in (args.dataset_dir / 'similarity_edges.jsonl').read_text().splitlines():
        edge = json.loads(line)
        if ids_to_split[edge['a']] != ids_to_split[edge['b']]: raise ValueError('Declared sequence similarity edge crosses splits')
    report = {'passed': True, 'splits': summary, 'sample_count': len(all_rows),
              'checks': ['evidence gate', 'canonical pair legality', 'unknown position masking', 'source hashes',
                         'sequence/PDB/component/group split isolation', 'declared similarity-edge split isolation'],
              'note': 'Does not prove absence of unannotated remote structural homology or guarantee local electron density quality.'}
    (args.dataset_dir / 'validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
