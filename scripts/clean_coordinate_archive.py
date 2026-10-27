"""Read the local PDB ZIP without extraction; resumable per-entry audit."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.coordinate_qc import parse_coordinate_entry

_ZIP = None
_IDS = None


def initialize(path, ids):
    global _ZIP, _IDS
    _ZIP, _IDS = zipfile.ZipFile(path), set(ids)


def process_member(name):
    try:
        data = _ZIP.read(name)
        info, records = parse_coordinate_entry(data, name, _IDS)
        return {'member': name, 'entry': info, 'chains': records, 'error': None}
    except Exception as exc:
        return {'member': name, 'error': f'{type(exc).__name__}: {exc}', 'entry': None, 'chains': []}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=Path('D:/E backup/下载/PDB_RNA.zip'))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    p.add_argument('--output', type=Path, default=ROOT / 'data/experimental/coordinate_qc_v1')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((ROOT / 'data/reports/current_experimental_rna_pdb.json').read_text(encoding='utf-8'))
    ids = frozen['entry_ids']
    with zipfile.ZipFile(args.archive) as z:
        members = [m for m in z.infolist() if m.filename.lower().endswith('.cif') and not m.is_dir()]
        names = sorted(m.filename for m in members)
        directory = '\n'.join(f'{m.filename}\t{m.CRC}\t{m.file_size}\t{m.compress_size}' for m in sorted(members, key=lambda m: m.filename))
    fingerprint = hashlib.sha256(directory.encode()).hexdigest()
    run_config = {'archive': str(args.archive.resolve()), 'archive_bytes': args.archive.stat().st_size,
                  'zip_directory_sha256': fingerprint, 'member_count': len(names), 'limit': args.limit,
                  'frozen_pdb_query_sha256': frozen['response_sha256'], 'protocol': 'coordinate-qc-v1',
                  'source_module_sha256': hashlib.sha256((ROOT / 'src/rna2d/coordinate_qc.py').read_bytes()).hexdigest()}
    config_path = args.output / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        p.error('Output was generated with different inputs/protocol. Choose a new --output directory.')
    config_path.write_text(json.dumps(run_config, indent=2), encoding='utf-8')
    if args.limit:
        names = names[:args.limit]
    completed = set()
    spool = args.output / 'entry_audit.jsonl'
    if spool.exists():
        # A truncated last line is not silently accepted; preserve and report it.
        for line in spool.read_text(encoding='utf-8').splitlines():
            if line:
                completed.add(json.loads(line)['member'])
    remaining = [name for name in names if name not in completed]
    start = time.monotonic()
    print(f'Audit {len(names)} CIF members; resume skips {len(completed)}; remaining {len(remaining)}', flush=True)
    with spool.open('a', encoding='utf-8') as f, ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(args.archive), ids)) as pool:
        for n, row in enumerate(pool.map(process_member, remaining, chunksize=1), 1):
            f.write(json.dumps(row, allow_nan=False) + '\n')
            if n % 100 == 0 or n == len(remaining):
                f.flush()
                print(f'Processed {n}/{len(remaining)} new entries in {time.monotonic()-start:.1f}s', flush=True)
    entries, chain_count, passed, reasons, methods, lengths, parse_errors = 0, 0, 0, Counter(), Counter(), Counter(), []
    candidates = args.output / 'screen_pass_chains.jsonl'
    local_ids, unique = set(), set()
    with candidates.open('w', encoding='utf-8') as out, spool.open(encoding='utf-8') as inp:
        for line in inp:
            row = json.loads(line)
            entries += 1
            if row['error']:
                parse_errors.append({'member': row['member'], 'error': row['error']})
                continue
            local_ids.add(row['entry']['pdb_id'])
            methods.update(row['entry']['experimental_methods'])
            for chain in row['chains']:
                chain_count += 1
                reasons.update(chain['quality_screen_reasons'])
                if chain['coordinate_screen_pass']:
                    out.write(json.dumps(chain) + '\n')
                    passed += 1
                    unique.add(chain['sequence'])
                    lengths['<=256' if chain['length'] <= 256 else '257-512' if chain['length'] <= 512 else '>512'] += 1
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'config': run_config,
              'cif_entries_processed': entries, 'parse_errors': parse_errors, 'rna_chains_screened': chain_count,
              'screen_pass_chains': passed, 'screen_pass_unique_sequences': len(unique), 'pass_length_distribution': dict(lengths),
              'screen_reasons': dict(reasons), 'entry_methods': dict(methods),
              'frozen_current_ids_missing_locally': sorted(set(ids) - local_ids),
              'local_ids_outside_frozen_search': sorted(local_ids - set(ids)),
              'pair_labels_approved': 0, 'training_ready': False,
              'limitations': ['Coordinate screening only, not final experimental gold labels.',
                              'Only first coordinate model screened; NMR quarantined separately.',
                              'Modified bases preserved but quarantined for dedicated mapping.',
                              'No homology split or cross-chain base pair context audit yet.']}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('config', 'frozen_current_ids_missing_locally', 'local_ids_outside_frozen_search', 'parse_errors')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
