"""Versioned, resumable acquisition of official experimental RNA resources.

Raw acquisition is not label approval. Reuses the audited local PDB archive,
downloads only missing coordinate entries, and collects RMDB's original RDAT
release assets with upstream digests where available.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import tarfile
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
_LOCAL = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(4 * 1024 * 1024): digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8', newline='\n')
    os.replace(tmp, path)


def session():
    if not hasattr(_LOCAL, 'session'):
        _LOCAL.session = requests.Session()
        _LOCAL.session.headers.update({'User-Agent': 'RNA-Lab-experimental-acquisition/1.0', 'Accept-Encoding': 'identity'})
    return _LOCAL.session


def request_json(url, params=None):
    response = session().get(url, params=params, timeout=(20, 120))
    response.raise_for_status()
    return response.json()


def build_plan(run_dir):
    query = {'query': {'type': 'terminal', 'service': 'text', 'parameters':
                      {'attribute': 'rcsb_entry_info.polymer_entity_count_RNA', 'operator': 'greater', 'value': 0}},
             'return_type': 'entry', 'request_options': {'return_all_hits': True, 'results_content_type': ['experimental']}}
    pdb = request_json('https://search.rcsb.org/rcsbsearch/v2/query', {'json': json.dumps(query)})
    ids = sorted(r['identifier'].upper() for r in pdb['result_set'])
    if len(ids) != pdb['total_count']: raise ValueError('PDB result count mismatch')
    write_json(run_dir / 'rcsb_query.json', {'query': query, 'response': pdb, 'fetched_at_utc': now()})
    archived = {}
    audit_path = ROOT / 'data/experimental/coordinate_qc_v1/entry_audit.jsonl'
    with audit_path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['entry'] and not any('THEORETICAL' in m.upper() for m in row['entry']['experimental_methods']):
                archived[row['entry']['pdb_id']] = row['member']
    audit_config = json.loads((audit_path.parent / 'run_config.json').read_text(encoding='utf-8'))
    local_archive = Path(audit_config['archive'])
    if not local_archive.exists() or local_archive.stat().st_size != audit_config['archive_bytes']:
        raise ValueError('Audited local archive no longer matches recorded size')
    coverage, jobs = [], []
    for pdb_id in ids:
        if not re.fullmatch(r'[A-Z0-9_]+', pdb_id): raise ValueError('Unsafe PDB identifier')
        old = ROOT / 'data/raw/experimental_cif_v1' / (pdb_id.lower() + '.cif')
        receipt = old.with_suffix('.receipt.json')
        if old.exists() and receipt.exists():
            saved = json.loads(receipt.read_text(encoding='utf-8'))
            if digest_file(old) != saved['sha256']: raise ValueError('Existing authoritative CIF checksum differs')
            coverage.append({'pdb_id': pdb_id, 'status': 'existing_authoritative_cif', 'path': str(old.relative_to(ROOT)),
                             'sha256': saved['sha256']})
        elif pdb_id in archived:
            coverage.append({'pdb_id': pdb_id, 'status': 'existing_audited_local_archive',
                             'archive': str(local_archive), 'member': archived[pdb_id],
                             'note': 'ID and experimental metadata checked; archive is reused, not newly downloaded.'})
        else:
            rel = f'pdb/{pdb_id.lower()}.cif.gz'
            jobs.append({'id': 'pdb:' + pdb_id, 'kind': 'pdb_cif_gz', 'pdb_id': pdb_id,
                         'url': f'https://files.rcsb.org/download/{pdb_id}.cif.gz', 'relative_path': rel,
                         'expected_bytes': None, 'expected_sha256': None})
            coverage.append({'pdb_id': pdb_id, 'status': 'scheduled_authoritative_download', 'relative_path': rel})
    releases = request_json('https://api.github.com/repos/DasLab/rmdb.github.io/releases', {'per_page': 100})
    write_json(run_dir / 'rmdb_releases.json', releases)
    seen_names = set()
    for release in releases:
        if not release['tag_name'].startswith('data-'): continue
        for asset in release['assets']:
            name = asset['name']
            if not name.endswith('.rdat'): continue
            if not re.fullmatch(r'[A-Za-z0-9_.+-]+\.rdat', name): raise ValueError('Unsafe RDAT filename')
            if name in seen_names: raise ValueError('Duplicate RMDB identifier needs explicit review')
            seen_names.add(name)
            digest = asset.get('digest')
            if digest and not digest.startswith('sha256:'): raise ValueError('Unexpected upstream digest type')
            jobs.append({'id': 'rmdb:' + name, 'kind': 'rdat', 'category': release['tag_name'],
                         'url': asset['browser_download_url'], 'relative_path': 'rmdb/' + name,
                         'expected_bytes': asset['size'], 'expected_sha256': digest.split(':', 1)[1] if digest else None,
                         'upstream_updated_at': asset['updated_at'],
                         'evidence_class': 'experimental_probing_container_not_pair_truth'})
    plan = {'protocol': 'experimental-acquisition-v1', 'created_at_utc': now(),
            'pdb_entry_count': len(ids), 'pdb_coverage': coverage,
            'rmdb_file_count': len(seen_names), 'jobs': jobs,
            'known_download_bytes': sum(j['expected_bytes'] or 0 for j in jobs),
            'coordinate_quality_approval': False, 'probing_profile_quality_approval': False}
    write_json(run_dir / 'plan.json', plan)
    return plan


def acquire(job, run_dir):
    # Create only the two fixed collection directories before resolving targets;
    # avoid concurrent resolution while the directory is still being created.
    relative = Path(job['relative_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe artifact relative path')
    target = (run_dir / job['relative_path']).resolve()
    if not target.is_relative_to(run_dir.resolve()): raise ValueError(f'Path escapes acquisition directory: {job["id"]}')
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = target.with_suffix(target.suffix + '.receipt.json')
    if target.exists():
        if not receipt_path.exists(): return {'id': job['id'], 'status': 'quarantined', 'reason': 'Existing file lacks receipt'}
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if target.stat().st_size != receipt['bytes'] or digest_file(target) != receipt['sha256']:
            return {'id': job['id'], 'status': 'quarantined', 'reason': 'Existing file checksum mismatch'}
        if job.get('expected_sha256') and receipt['sha256'] != job['expected_sha256']:
            return {'id': job['id'], 'status': 'quarantined', 'reason': 'Receipt differs from frozen upstream checksum'}
        return dict(receipt, status='verified_existing')
    partial = target.with_suffix(target.suffix + '.partial')
    partial_state = partial.with_suffix(partial.suffix + '.json')
    last_error = None
    for attempt in range(4):
        try:
            if shutil.disk_usage(run_dir).free < 50 * 1024**3: raise RuntimeError('Less than 50GiB free; preserve disk reserve')
            offset = partial.stat().st_size if partial.exists() else 0
            saved = json.loads(partial_state.read_text()) if partial_state.exists() else {}
            headers = {}
            if offset and saved.get('validator'):
                headers = {'Range': f'bytes={offset}-', 'If-Range': saved['validator']}
            response = session().get(job['url'], stream=True, headers=headers, timeout=(20, 60))
            response.raise_for_status()
            if response.status_code == 206:
                content_range = response.headers.get('Content-Range', '')
                if not content_range.startswith(f'bytes {offset}-'): raise ValueError('Invalid range response')
                mode = 'ab'
            elif response.status_code == 200:
                mode = 'wb'
                offset = 0
            else:
                raise ValueError('Unexpected download response')
            validator = response.headers.get('ETag') or response.headers.get('Last-Modified')
            write_json(partial_state, {'validator': validator, 'url': job['url']})
            with response, partial.open(mode) as out:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk: out.write(chunk)
            size = partial.stat().st_size
            if job.get('expected_bytes') is not None and size != job['expected_bytes']:
                raise ValueError(f'Download size mismatch: {size} vs {job["expected_bytes"]}')
            digest = digest_file(partial)
            if job.get('expected_sha256') and digest != job['expected_sha256']:
                bad = partial.with_suffix('.invalid-' + digest[:12])
                partial.rename(bad)
                raise ValueError('Upstream checksum mismatch; bytes preserved in quarantine')
            format_note = None
            if job['kind'] == 'pdb_cif_gz':
                with gzip.open(partial, 'rb') as f:
                    prefix = f.read(4096)
                    if b'data_' not in prefix: raise ValueError('Not a mmCIF payload')
                    while f.read(4 * 1024 * 1024): pass  # Validate gzip CRC/end-of-stream.
            elif job['kind'] == 'rdat':
                with partial.open('rb') as f: prefix = f.read(65536)
                if b'RDAT_VERSION' not in prefix: format_note = 'RDAT header requires separate review'
            elif job['kind'] == 'interaction_csv_gz':
                with gzip.open(partial, 'rb') as f:
                    prefix = f.read(65536)
                    if b'unit_id' not in prefix: format_note = 'Interaction header requires review'
                    while f.read(4 * 1024 * 1024): pass
            elif job['kind'] == 'zip':
                with zipfile.ZipFile(partial) as archive:
                    if archive.testzip() is not None: raise ValueError('ZIP member CRC mismatch')
            elif job['kind'] == 'tar':
                with tarfile.open(partial) as archive:
                    if not any(m.isfile() for m in archive): raise ValueError('TAR has no regular data files')
            elif job['kind'] == 'reference_gzip':
                with gzip.open(partial, 'rb') as f:
                    prefix = f.read(65536)
                    if job.get('reference_type') == 'fasta' and not prefix.lstrip().startswith(b'>'):
                        raise ValueError('Expected FASTA reference header')
                    if job.get('reference_type') == 'gff' and b'\t' not in prefix:
                        raise ValueError('Expected GFF annotation columns')
                    while f.read(4 * 1024 * 1024): pass
            receipt = {'id': job['id'], 'kind': job['kind'], 'url': job['url'],
                       'path': str(target.relative_to(ROOT)), 'bytes': size, 'sha256': digest,
                       'upstream_sha256_verified': bool(job.get('expected_sha256')),
                       'source_etag': response.headers.get('ETag'),
                       'source_last_modified': response.headers.get('Last-Modified'),
                       'downloaded_at_utc': now(), 'format_note': format_note,
                       'quality_approved': False, 'status': 'verified_download'}
            partial.replace(target)
            write_json(receipt_path, receipt)
            return receipt
        except (requests.RequestException, ValueError, OSError, EOFError, RuntimeError) as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            if attempt < 3: time.sleep(2 ** attempt)
    return {'id': job['id'], 'status': 'failed', 'reason': last_error, 'url': job['url']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, default=ROOT / 'data/raw/experimental_acquisition_v1')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--kind', choices=['all', 'pdb_cif_gz', 'rdat'], default='all')
    p.add_argument('--retry-failed', action='store_true', help='Reuse verified results from the previous completed run; retry unfinished artifacts only')
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for collection in ['pdb', 'rmdb']:
        (run_dir / collection).mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else build_plan(run_dir)
    print(json.dumps({'pdb_current_entries': plan['pdb_entry_count'],
                      'pdb_missing_to_download': sum(j['kind']=='pdb_cif_gz' for j in plan['jobs']),
                      'rmdb_files': plan['rmdb_file_count'], 'known_download_GiB': round(plan['known_download_bytes']/1024**3, 3)}), flush=True)
    if args.plan_only: return
    jobs = [j for j in plan['jobs'] if args.kind == 'all' or j['kind'] == args.kind]
    report_path = ROOT / 'data/reports' / (run_dir.name + '_' + args.kind + '.json')
    results = []
    if args.retry_failed and report_path.exists():
        previous = json.loads(report_path.read_text(encoding='utf-8'))
        if not previous['complete']: p.error('Do not start a retry while the original run is still active')
        if previous['plan_sha256'] != digest_file(plan_path): p.error('Frozen acquisition plan changed')
        results = [r for r in previous['results'] if r['status'].startswith('verified')]
    already_verified = {r['id'] for r in results}
    remaining_jobs = [j for j in jobs if j['id'] not in already_verified]
    start, last = time.monotonic(), time.monotonic()

    def checkpoint():
        status = {'updated_at_utc': now(), 'plan_sha256': digest_file(plan_path),
                  'requested_jobs': len(jobs), 'finished_jobs': len(results),
                  'verified_jobs': sum(r['status'].startswith('verified') for r in results),
                  'failed_or_quarantined': [r for r in results if not r['status'].startswith('verified')],
                  'verified_bytes': sum(r.get('bytes', 0) for r in results if r['status'].startswith('verified')),
                  'complete': len(results) == len(jobs), 'results': results,
                  'note': 'Raw retrieval/checksums only; not residue/probing/label quality approval.'}
        write_json(report_path, status)
        return status
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(acquire, job, run_dir): job for job in remaining_jobs}
        while pending:
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try: results.append(future.result())
                except Exception as exc: results.append({'id': job['id'], 'status': 'failed', 'reason': str(exc)})
            if time.monotonic() - last >= 15 or not pending:
                report = checkpoint()
                print(f'Finished {len(results)}/{len(jobs)}; verified {report["verified_jobs"]}; failures {len(report["failed_or_quarantined"])}; completed {report["verified_bytes"]/1024**3:.2f}GiB; elapsed {time.monotonic()-start:.0f}s', flush=True)
                last = time.monotonic()
    final = checkpoint()
    print(json.dumps({k: v for k, v in final.items() if k not in {'results', 'failed_or_quarantined'}}, indent=2), flush=True)
    if final['failed_or_quarantined']: raise SystemExit(1)


if __name__ == '__main__':
    main()
