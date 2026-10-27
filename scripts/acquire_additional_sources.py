"""Fetch frozen RNAsolo/FR3D/original RASP artifacts without approving labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode

from acquire_experimental_corpus import acquire, write_json, digest_file, now

ROOT = Path(__file__).resolve().parents[1]


def amended_plan():
    old_dir = ROOT / 'data/raw/additional_experimental_catalog_v1'
    directory = ROOT / 'data/raw/additional_experimental_acquisition_v1'
    directory.mkdir(parents=True, exist_ok=True)
    plan_path = directory / 'plan.json'
    if plan_path.exists(): return directory, json.loads(plan_path.read_text(encoding='utf-8'))
    old = json.loads((old_dir / 'plan.json').read_text(encoding='utf-8'))
    score_rows, contact_rows = {}, {}
    for path in old_dir.glob('rasp_*_scores.json'):
        for row in json.loads(path.read_text(encoding='utf-8')).get('bw_files', []):
            if str(row.get('tracktype', '')).lower() == 'raw':
                remote = str(row.get('bed_file_path') or '').strip()
                url = 'https://rasp2.zhanglab.net/api/downloadfile/?' + urlencode({'filetype':'bed', 'filelist':remote})
                score_rows.setdefault(url, []).append(row)
    for path in old_dir.glob('rasp_*_contacts.json'):
        for row in json.loads(path.read_text(encoding='utf-8')).get('bedpe_files', []):
            remote = str(row.get('file_path') or row.get('bedpe_file_path') or '').strip()
            url = 'https://rasp2.zhanglab.net/api/downloadfile/?' + urlencode({'filetype':'bedpe', 'filelist':remote})
            contact_rows.setdefault(url, []).append(row)
    jobs = []
    for job in old['jobs']:
        updated = dict(job)
        if job['url'] in score_rows: updated['source_track_rows'] = score_rows[job['url']]
        if job['url'] in contact_rows: updated['source_track_rows'] = contact_rows[job['url']]
        jobs.append(updated)
    plan = dict(old, jobs=jobs, source_catalog_sha256=digest_file(old_dir/'plan.json'),
                amendment='Preserve all display/strand metadata rows when identical download URLs are deduplicated.',
                acquisition_created_at_utc=now())
    write_json(plan_path, plan)
    return directory, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    directory, plan = amended_plan()
    for job in plan['jobs']:
        (directory / job['relative_path']).parent.mkdir(parents=True, exist_ok=True)
    report_path = ROOT / 'data/reports/additional_experimental_acquisition_v1.json'
    results = []
    if args.retry_failed and report_path.exists():
        old = json.loads(report_path.read_text(encoding='utf-8'))
        if not old['complete']: parser.error('Original run still active; do not duplicate it')
        if old['plan_sha256'] != digest_file(directory/'plan.json'):
            parser.error('Frozen acquisition plan changed')
        results = [r for r in old['results'] if r['status'].startswith('verified')]
    verified = {r['id'] for r in results}
    jobs = [j for j in plan['jobs'] if j['id'] not in verified]
    print('Additional artifacts scheduled:', len(jobs), flush=True)
    last = time.monotonic()

    def checkpoint():
        report = {'updated_at_utc': now(), 'plan_sha256': digest_file(directory/'plan.json'),
                  'requested_jobs': len(plan['jobs']), 'finished_jobs': len(results),
                  'verified_jobs': sum(r['status'].startswith('verified') for r in results),
                  'failures': [r for r in results if not r['status'].startswith('verified')],
                  'verified_bytes': sum(r.get('bytes',0) for r in results if r['status'].startswith('verified')),
                  'complete': len(results) == len(plan['jobs']), 'results': results,
                  'note': 'No model-imputed tracks requested. Raw measured signals/coordinate annotations still require per-record QC and mapping.'}
        write_json(report_path, report)
        return report
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(acquire, job, directory): job for job in jobs}
        while pending:
            done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try: results.append(future.result())
                except Exception as exc: results.append({'id':job['id'],'status':'failed','reason':str(exc)})
            if time.monotonic()-last > 15 or not pending:
                report = checkpoint()
                print(f'Additional verified {report["verified_jobs"]}/{len(plan["jobs"])}; failures {len(report["failures"])}; completed {report["verified_bytes"]/1024**3:.2f}GiB', flush=True)
                last = time.monotonic()
    report = checkpoint()
    if report['failures']: raise SystemExit(1)


if __name__ == '__main__': main()
