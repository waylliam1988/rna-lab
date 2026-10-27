"""Normalize benchmark labels without inventing experimental provenance."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.data import CANONICAL, parse_bpseq, sequence_hash, crossing_pairs


def determine_split(archive, member):
    if archive == 'archiveII.tar.gz':
        return 'archiveII'
    if archive == 'bpRNAnew.tar.gz':
        return 'test_external'
    for folder, split in [('TR0', 'train'), ('VL0', 'validation'), ('TS0', 'test')]:
        if folder in member.split('/'):
            return split
    raise ValueError(f'Unrecognized split: {member}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min-length', type=int, default=20)
    p.add_argument('--max-length', type=int, default=256)
    args = p.parse_args()
    if not 1 <= args.min_length <= args.max_length:
        p.error('Invalid length range')
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / 'data/reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    counts, rejects, grouped = Counter(), [], defaultdict(list)
    archives = [ROOT / 'data/raw' / name for name in
                ['archiveII.tar.gz', 'bpRNA.tar.gz', 'bpRNAnew.tar.gz']]
    if not all(path.exists() for path in archives):
        raise SystemExit('Run scripts/download_data.py first')
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in archives}
    with (out / 'normalized_all.jsonl').open('w', encoding='utf-8') as normalized:
        for archive in archives:
            with tarfile.open(archive) as tar:
                for member in tar:
                    if not member.isfile() or not member.name.lower().endswith('.bpseq'):
                        continue
                    counts[f'raw:{archive.name}'] += 1
                    ident = archive.name + ':' + member.name
                    try:
                        if member.size > 10_000_000:
                            raise ValueError('Oversized BPSEQ')
                        seq, pairs, ts = parse_bpseq(tar.extractfile(member).read().decode('utf-8-sig'))
                        split = determine_split(archive.name, member.name)
                    except (ValueError, UnicodeError) as exc:
                        rejects.append({'id': ident, 'reason': 'parse_error', 'detail': str(exc)})
                        continue
                    record = {'id': ident, 'sequence': seq, 'sequence_sha256': sequence_hash(seq),
                              'pairs': pairs, 'observed_mask': [True] * len(seq),
                              'source_archive': archive.name, 'source_member': member.name,
                              'source_url': 'https://github.com/mxfold/mxfold2/releases/tag/v0.1.0',
                              'label_method': 'author_distributed_benchmark_BPSEQ; original provenance pending',
                              'preprocessing_note': 'bpRNA archive path explicitly says canonicals; original contacts may already be removed upstream',
                              'family': None, 'homology_cluster': None,
                              'conditions': {'mg_mM': None, 'na_mM': None, 'k_mM': None,
                                             'temperature_C': None, 'pH': None},
                              't_to_u_count': ts, 'official_split': split}
                    normalized.write(json.dumps(record) + '\n')
                    counts['parsed'] += 1
                    reason = None
                    if set(seq) - set('ACGU'):
                        reason = 'unsupported_bases'
                    elif not args.min_length <= len(seq) <= args.max_length:
                        reason = 'length'
                    elif any(seq[i] + seq[j] not in CANONICAL for i, j in pairs):
                        reason = 'unsupported_pairs'
                    elif any(j - i < 4 for i, j in pairs):
                        reason = 'short_separation'
                    if reason:
                        rejects.append({'id': ident, 'reason': reason})
                        continue
                    grouped[(split, record['sequence_sha256'])].append(record)
    deduped = defaultdict(dict)
    for (split, seqhash), records in sorted(grouped.items()):
        signatures = {tuple(tuple(x) for x in r['pairs']) for r in records}
        if len(signatures) > 1:
            rejects.extend({'id': r['id'], 'reason': 'same_sequence_conflicting_labels'} for r in records)
            continue
        record = records[0]
        record['merged_ids'] = [r['id'] for r in records]
        deduped[split][seqhash] = record
        counts['duplicates_merged'] += len(records) - 1
    seen = {}
    split_counts = {}
    for split in ['test_external', 'test', 'validation', 'archiveII', 'train']:
        selected = []
        for seqhash, record in sorted(deduped[split].items()):
            if seqhash in seen:
                rejects.append({'id': record['id'], 'reason': 'cross_split_exact_sequence',
                                'kept_split': seen[seqhash]})
                continue
            seen[seqhash] = split
            selected.append(record)
        path = out / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in selected), encoding='utf-8')
        split_counts[split] = {'samples': len(selected),
                              'with_crossing_pairs': sum(bool(crossing_pairs(r['pairs'])) for r in selected),
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (report_dir / 'excluded.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rejects), encoding='utf-8')
    report = {'protocol': 'v0.1', 'length': [args.min_length, args.max_length],
              'source_sha256': source_hashes, 'counts': dict(counts),
              'exclusions': dict(Counter(r['reason'] for r in rejects)), 'splits': split_counts,
              'limitations': ['Exact sequence isolation only; homology audit NOT complete.',
                             'Author-preprocessed canonical benchmark labels, not pure experimental truth.',
                             'ArchiveII and external sets must not be used to tune hyperparameters.',
                             'Every source retains null family/conditions until independently verified.']}
    (report_dir / 'prepare_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
