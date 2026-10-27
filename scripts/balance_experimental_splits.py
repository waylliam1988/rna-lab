"""Preserve the first draft; freeze a new split without repeating annotation."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from build_experimental_pairs import assign_groups, save_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT / 'data/experimental/paired_v0_1')
    p.add_argument('--output', type=Path, default=ROOT / 'data/experimental/paired_v0_2')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    for split in ['train', 'validation', 'test']:
        for line in (args.source / f'{split}.jsonl').read_text().splitlines():
            r = json.loads(line)
            groups[r['split_group_id']].append(r)
    allocated = assign_groups(list(groups.values()))
    report = json.loads((args.source / 'summary.json').read_text())
    report['previous_split_draft'] = str(args.source)
    report['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
    report['split_protocol'] = 'largest-group-first target-sample-deficit, 80/10/10, seed42'
    report['benchmark_ready'] = False
    report['training_ready'] = report.get('pilot_limit') is None and all(allocated.values())
    report['splits'] = {}
    for split, assigned in allocated.items():
        rows = [dict(r, split=split) for group in assigned for r in group]
        save_jsonl(args.output / f'{split}.jsonl', rows)
        (args.output / f'{split}_sequences.fasta').write_text(''.join(f'>{r["id"]}\n{r["sequence"]}\n' for r in rows), encoding='utf-8')
        report['splits'][split] = {'samples': len(rows), 'groups': len(assigned),
                                  'sha256': hashlib.sha256((args.output / f'{split}.jsonl').read_bytes()).hexdigest()}
    for name in ['all_annotated_chains.jsonl', 'annotation_rejected.jsonl', 'unknown_homology_NOT_FOR_TRAINING.jsonl', 'sequence_conflicts.jsonl', 'similarity_edges.jsonl']:
        shutil.copy2(args.source / name, args.output / name)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['splits'], indent=2))


if __name__ == '__main__':
    main()
