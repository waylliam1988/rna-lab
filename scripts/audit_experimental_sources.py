"""Evidence-first inventory. Candidates are never silently promoted to gold.

Audits existing local bpRNA; snapshots official RNA3DB and RMDB catalogs;
exports experimental-coordinate-associated sequences for 1D use. Does not
infer missing base pairs or turn probing profiles into pairing truth.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.data import parse_bpseq, sequence_hash, crossing_pairs


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def quality_reasons(record):
    """Provisional screening, NOT final coordinate/label validation."""
    reasons = []
    method = record.get('structure_method', '').lower()
    resolution = record.get('resolution')
    seq = record.get('sequence', '').upper()
    if not seq or set(seq) - set('ACGU'):
        reasons.append('ambiguous_or_modified_sequence')
    if method == 'x-ray diffraction':
        if not isinstance(resolution, (float, int)) or not 0 < resolution <= 3.0:
            reasons.append('xray_resolution_gt3_or_missing')
    elif method == 'electron microscopy':
        if not isinstance(resolution, (float, int)) or not 0 < resolution <= 3.5:
            reasons.append('em_resolution_gt3_5_or_missing')
    elif 'nmr' in method:
        reasons.append('nmr_requires_restraint_and_ensemble_quality_audit')
    else:
        reasons.append('method_requires_separate_review')
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpseq-zip', type=Path, default=Path('D:/E backup/下载/bpseqFiles.zip'))
    parser.add_argument('--offline', action='store_true', help='Use cached catalogs/archives')
    args = parser.parse_args()
    raw = ROOT / 'data/raw/source_catalog'
    out = ROOT / 'data/experimental'
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers['User-Agent'] = 'Nezha-RNA2D-evidence-audit/0.1'
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    downloads = []

    def fetch(url, path, refresh=False):
        if not args.offline and (refresh or not path.exists()):
            response = session.get(url, timeout=(15, 120))
            response.raise_for_status()
            tmp = path.with_suffix(path.suffix + '.partial')
            tmp.write_bytes(response.content)
            tmp.replace(path)
        if not path.exists():
            raise FileNotFoundError(path)
        downloads.append({'url': url, 'path': str(path.relative_to(ROOT)),
                          'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'checked_at_utc': datetime.now(timezone.utc).isoformat()})
        return path

    rna_releases = json.loads(fetch('https://api.github.com/repos/marcellszi/rna3db/releases', raw / 'rna3db_releases.json', refresh=True).read_text(encoding='utf-8'))
    release = next(r for r in rna_releases if not r['prerelease'] and not r['draft'])
    asset = next(a for a in release['assets'] if a['name'] == 'rna3db-jsons.tar.gz')
    archive = fetch(asset['browser_download_url'], raw / ('rna3db-' + release['tag_name'] + '-jsons.tar.gz'))
    # Authoritative release metadata; filenames do not imply current PDB coverage.
    with tarfile.open(archive) as tar:
        member = next(m for m in tar if m.isfile() and m.name.endswith('/split.json'))
        split_data = json.load(tar.extractfile(member))
        parse_member = next(m for m in tar.getmembers() if m.name.endswith('/parse.json'))
        parsed_data = json.load(tar.extractfile(parse_member))
    split_membership = {}
    for split, components in split_data.items():
        for component, clusters in components.items():
            for cluster, chains in clusters.items():
                for chain in chains:
                    split_membership[chain] = (split, component, cluster)
    selected, quarantined, all_rows = [], [], []
    for chain, record in parsed_data.items():
        record = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in record.items()}
        split, component, cluster = split_membership.get(chain, (None, None, None))
        row = {'id': chain, **record, 'official_split': split, 'homology_component': component,
               'sequence_cluster': cluster, 'source_release': release['tag_name'],
               'evidence_class': 'experimental_coordinate_associated_sequence',
               'pair_labels_available': False, 'coordinate_quality_verified': False}
        all_rows.append(row)
        reasons = quality_reasons(record)
        if reasons:
            quarantined.append({**row, 'reasons': reasons})
        else:
            selected.append(row)
    for filename, rows in [('rna3db_all_chain_metadata.jsonl', all_rows),
                           ('rna3db_sequence_candidates.jsonl', selected),
                           ('rna3db_quality_quarantine.jsonl', quarantined)]:
        (out / filename).write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    unique = {}
    for row in selected:
        unique.setdefault(row['sequence'], []).append(row['id'])
    with (out / 'experimental_associated_sequences.fasta').open('w', encoding='utf-8') as f:
        for seq, ids in sorted(unique.items(), key=lambda x: sequence_hash(x[0])):
            f.write(f'>{ids[0]} seqsha256={sequence_hash(seq)} structures={len(ids)} evidence=coordinate_associated_only\n{seq}\n')
    rna_summary = {'release': release['tag_name'], 'chains_in_parse_snapshot': len(all_rows),
                   'chains_in_official_split': len(split_membership),
                   'pdb_entries': len({r['id'].split('_')[0] for r in all_rows}),
                   'metadata_screen_pass_chains': len(selected), 'unique_sequences': len(unique),
                   'all_unique_sequences': len({r['sequence'] for r in all_rows}),
                   'methods': dict(Counter(r['structure_method'] for r in all_rows)),
                   'quarantine_reasons': dict(Counter(reason for r in quarantined for reason in r['reasons'])),
                   'quality_status': 'metadata-screened candidates; residue coverage and pairing annotations not yet validated'}
    print('RNA3DB:', json.dumps(rna_summary), flush=True)

    source_counts, pdb_candidates, pdb_invalid = Counter(), [], []
    if args.bpseq_zip.exists():
        with zipfile.ZipFile(args.bpseq_zip) as z:
            for name in z.namelist():
                if not name.endswith('.bpseq'):
                    continue
                source = Path(name).stem.split('_')[1]
                source_counts[source] += 1
                if source != 'PDB':
                    continue  # Comparative analysis is NOT direct per-molecule experimental truth.
                try:
                    seq, pairs, ts = parse_bpseq(z.read(name).decode('utf-8-sig'))
                    pdb_candidates.append({'id': Path(name).stem, 'sequence': seq, 'pairs': pairs,
                                           'sequence_sha256': sequence_hash(seq),
                                           'crossing_pair_count': len(crossing_pairs(pairs)),
                                           'evidence_class': 'reported_pdb_derived_pairing',
                                           'provenance_verified': False, 'quality_verified': False,
                                           'source_zip': str(args.bpseq_zip), 'source_member': name,
                                           'reasons_pending': ['resolve_original_PDB_and_chain', 'check_method_resolution_coverage',
                                                               'check_pair_extractor_and_annotation_policy']})
                except (ValueError, UnicodeError) as exc:
                    pdb_invalid.append({'id': name, 'reason': str(exc)})
        (out / 'bprna_pdb_candidates_NOT_GOLD.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in pdb_candidates), encoding='utf-8')
    bprna_summary = {'local_archive': str(args.bpseq_zip), 'source_counts': dict(source_counts),
                     'reported_pdb_candidates_parsed': len(pdb_candidates),
                     'unique_pdb_candidate_sequences': len({r['sequence'] for r in pdb_candidates}),
                     'invalid': pdb_invalid,
                     'excluded_from_strict_pool_comparative_records': sum(v for k, v in source_counts.items() if k != 'PDB'),
                     'gold_approved': 0,
                     'note': 'Not-direct-experimental is not synonymous with fraudulent or model-generated.'}
    print('bpRNA:', json.dumps(bprna_summary), flush=True)

    rmdb = json.loads(fetch('https://api.github.com/repos/DasLab/rmdb.github.io/releases', raw / 'rmdb.github.io_releases.json', refresh=True).read_text(encoding='utf-8'))
    rmdb_assets = []
    for rel in rmdb:
        for asset in rel['assets']:
            if asset['name'].endswith('.rdat'):
                rmdb_assets.append({'name': asset['name'], 'bytes': asset['size'], 'url': asset['browser_download_url'],
                                    'digest': asset.get('digest'), 'category': rel['tag_name'],
                                    'updated_at': asset['updated_at'], 'evidence_class': 'experimental_probing_container',
                                    'per_profile_quality_verified': False,
                                    'structure_field_is_not_automatically_experimental_truth': True})
    dump(raw / 'rmdb_assets.json', rmdb_assets)
    # Fetch explicit small empirical examples, not model-generated secondary structures.
    sample_ids = ['16SRRNA_H44_0001.rdat', 'ADDRSW_DMS_0001.rdat', 'NEIL1_DMS_0020.rdat']
    sample_info = []
    for sample in sample_ids:
        asset = next(a for a in rmdb_assets if a['name'] == sample)
        path = raw / sample
        fetch(asset['url'], path)
        if asset.get('digest'):
            if asset['digest'] != 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('RMDB upstream checksum mismatch')
        text = path.read_text(encoding='utf-8-sig')
        sample_info.append({'name': sample, 'bytes': path.stat().st_size,
                            'sequence_field': any(l.startswith('SEQUENCE ') or l.startswith('SEQUENCE\t') for l in text.splitlines()),
                            'reactivity_rows': sum(l.startswith('REACTIVITY:') for l in text.splitlines()),
                            'structure_rows': sum(l.startswith('STRUCTURE') for l in text.splitlines()),
                            'note': 'RDAT structure fields may be supplied hypotheses; keep separate from measured reactivities.'})
    rmdb_summary = {'rdat_files_cataloged': len(rmdb_assets), 'total_bytes': sum(a['bytes'] for a in rmdb_assets),
                    'by_category': dict(Counter(a['category'] for a in rmdb_assets)), 'sample_downloads': sample_info,
                    'gold_pair_labels_from_reactivities': 0,
                    'note': 'Construct counts, unique sequences, conditions and data points are different units. Library experiments may include designed sequences.'}
    report = {'as_of_utc': datetime.now(timezone.utc).isoformat(), 'rna3db': rna_summary,
              'bprna': bprna_summary, 'rmdb': rmdb_summary, 'strict_pair_training_ready': False,
              'exclusions': ['all model-predicted secondary structures', 'model-imputed probing positions',
                             'comparative labels in direct-experiment-only pool', 'unverified provenance in strict gold pool'],
              'limitations': ['Not a claim to cover all existing experimental data.',
                              'No coordinate-derived pairing extraction or residue-level quality audit yet.',
                              'Do not use current benchmark train files as strict experimental training data.']}
    dump(ROOT / 'data/reports/experimental_sources_audit.json', report)
    dump(raw / 'audit_download_manifest.json', downloads)
    print('RMDB:', json.dumps(rmdb_summary), flush=True)


if __name__ == '__main__':
    main()
