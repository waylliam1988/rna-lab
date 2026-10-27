"""Index acquired RNA sequence/pair candidates without promoting them to gold."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.data import parse_bpseq, CANONICAL


def identify(member):
    stem = Path(member).stem
    match = re.match(r'^PDB_([A-Za-z0-9]{8})_(\d+)_(.+)$', stem)
    if not match: return None, None, None, None
    extended, model, chain = match.groups()
    pdb_id = extended[-4:].upper() if extended.startswith('0000') else 'PDB_' + extended.upper()
    fragment = '__' in chain
    return pdb_id, int(model), chain, fragment


def fasta_records(text):
    name, parts = None, []
    for line in text.splitlines():
        if line.startswith('>'):
            if name is not None: yield name, ''.join(parts).upper()
            name, parts = line[1:].strip(), []
        elif line.strip(): parts.append(line.strip())
    if name is not None: yield name, ''.join(parts).upper()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', type=Path, default=ROOT / 'data/raw/additional_experimental_acquisition_v1')
    p.add_argument('--output', type=Path, default=ROOT / 'data/experimental/internet_annotation_candidates_v1')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.source_dir / 'plan.json').read_text(encoding='utf-8'))
    query_snapshot = json.loads((ROOT / 'data/raw/experimental_acquisition_v1/rcsb_query.json').read_text(encoding='utf-8'))
    current = {row['identifier'].upper() for row in query_snapshot['response']['result_set']}
    database = sqlite3.connect(args.output / 'candidates.sqlite3')
    database.execute('PRAGMA journal_mode=WAL')
    database.execute('CREATE TABLE IF NOT EXISTS artifacts (name TEXT PRIMARY KEY, sha256 TEXT NOT NULL)')
    database.execute('CREATE TABLE IF NOT EXISTS sequences (source_key TEXT PRIMARY KEY, artifact TEXT, member TEXT, header TEXT, pdb_id TEXT, model INTEGER, chain_spec TEXT, fragment INTEGER, sequence TEXT, sequence_sha256 TEXT, alphabet_valid INTEGER, current_experimental INTEGER)')
    database.execute('CREATE TABLE IF NOT EXISTS pairings (source_key TEXT PRIMARY KEY, artifact TEXT, member TEXT, pdb_id TEXT, model INTEGER, chain_spec TEXT, fragment INTEGER, sequence TEXT, sequence_sha256 TEXT, pairs_json TEXT, pair_count INTEGER, noncanonical_pair_count INTEGER, t_to_u_count INTEGER, current_experimental INTEGER)')
    errors = []
    for job in plan['jobs']:
        if not job['id'].startswith('rnasolo:') or job.get('format') not in ('fasta', 'bpseq'): continue
        path = args.source_dir / job['relative_path']
        receipt = path.with_suffix(path.suffix + '.receipt.json')
        if not path.exists() or not receipt.exists():
            errors.append({'artifact': job['id'], 'reason': 'not_downloaded_and_verified'})
            continue
        manifest = json.loads(receipt.read_text(encoding='utf-8'))
        previous = database.execute('SELECT sha256 FROM artifacts WHERE name=?', (path.name,)).fetchone()
        if previous:
            if previous[0] != manifest['sha256']: raise ValueError('Previously indexed artifact changed')
            continue
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir(): continue
                key = hashlib.sha256((manifest['sha256'] + ':' + member.filename).encode()).hexdigest()
                try:
                    text = archive.read(member).decode('utf-8-sig')
                    pdb_id, model, chain, fragment = identify(member.filename)
                    if job['format'] == 'bpseq':
                        seq, pairs, ts = parse_bpseq(text)
                        invalid = sum(seq[i] + seq[j] not in CANONICAL for i, j in pairs)
                        database.execute('INSERT OR REPLACE INTO pairings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                         (key, path.name, member.filename, pdb_id, model, chain, fragment,
                                          seq, hashlib.sha256(seq.encode()).hexdigest(), json.dumps(pairs), len(pairs), invalid, ts,
                                          int(pdb_id in current)))
                    else:
                        for index, (header, seq) in enumerate(fasta_records(text)):
                            record_key = key + ':' + str(index)
                            database.execute('INSERT OR REPLACE INTO sequences VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                             (record_key, path.name, member.filename, header, pdb_id, model, chain, fragment,
                                              seq, hashlib.sha256(seq.encode()).hexdigest(), int(bool(seq) and set(seq) <= set('ACGU')), int(pdb_id in current)))
                except (ValueError, UnicodeError) as exc:
                    errors.append({'artifact': path.name, 'member': member.filename, 'reason': str(exc)})
        database.execute('INSERT INTO artifacts VALUES (?,?)', (path.name, manifest['sha256']))
        database.commit()
        print('Indexed', path.name, flush=True)
    def count(sql): return database.execute(sql).fetchone()[0]
    summary = {'sequence_records_across_overlapping_views': count('SELECT COUNT(*) FROM sequences'),
               'unique_sequence_strings': count('SELECT COUNT(DISTINCT sequence_sha256) FROM sequences'),
               'unique_unambiguous_ACGU_current_experimental_candidates': count('SELECT COUNT(DISTINCT sequence_sha256) FROM sequences WHERE alphabet_valid=1 AND current_experimental=1'),
               'pairing_records_across_overlapping_views': count('SELECT COUNT(*) FROM pairings'),
               'unique_pairing_sequence_strings': count('SELECT COUNT(DISTINCT sequence_sha256) FROM pairings'),
               'unique_current_experimental_pairing_sequence_candidates': count('SELECT COUNT(DISTINCT sequence_sha256) FROM pairings WHERE current_experimental=1'),
               'pairing_records_not_in_frozen_current_experimental_set': count('SELECT COUNT(*) FROM pairings WHERE current_experimental=0'),
               'pairing_records_needing_nucleotide_normalization_review': count('SELECT COUNT(*) FROM pairings WHERE t_to_u_count>0'),
               'errors_or_pending': errors, 'training_approved_records': 0,
               'limitations': ['Repository views overlap; strings, chains, fragments, models and structures are different units.',
                               'Fragment coordinates, combined-chain boundaries, modified bases and annotation policies require review.',
                               'Only matches frozen experimental entry IDs; does not certify residue-level quality.',
                               'No new training/test split or automatic gold promotion.']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    database.close()
    print(json.dumps({k:v for k,v in summary.items() if k!='errors_or_pending'}, indent=2))


if __name__ == '__main__': main()
