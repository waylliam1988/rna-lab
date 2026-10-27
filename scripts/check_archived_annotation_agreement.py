"""Diagnostic agreement, never a ground-truth accuracy estimate."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.data import CANONICAL, observable_pairs, pair_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/experimental/paired_v0_2')
    args = p.parse_args()
    reference = defaultdict(list)
    for line in (ROOT / 'data/experimental/bprna_pdb_candidates_NOT_GOLD.jsonl').read_text().splitlines():
        r = json.loads(line)
        reference[r['sequence']].append(r)
    results = []
    for line in (args.dataset_dir / 'all_annotated_chains.jsonl').read_text().splitlines():
        r = json.loads(line)
        for ref in reference[r['sequence']]:
            pairs = [p for p in ref['pairs'] if p[1]-p[0] >= 4 and r['sequence'][p[0]] + r['sequence'][p[1]] in CANONICAL]
            metrics = pair_metrics(r['pairs'], observable_pairs(pairs, r['observed_mask']))
            results.append({'coordinate_id': r['id'], 'archived_id': ref['id'], **metrics})
    summary = {'sequence_matched_comparisons': len(results),
               'mean_pair_f1_agreement': sum(r['f1'] for r in results) / len(results) if results else None,
               'perfect_agreements': sum(r['f1'] == 1 for r in results),
               'note': 'Agreement only. Archived annotation provenance not fully verified; identical sequence need not mean identical experiment or conditions. Not model accuracy, not an acceptance filter.'}
    (args.dataset_dir / 'archived_agreement.json').write_text(json.dumps({'summary': summary, 'comparisons': results}, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
