"""Fetch official coordinates for ZIP-screened candidates, then repeat fine QC.

An archive filename/experimental header alone is never provenance approval.
Existing receipt-backed official inputs are reused. The large ZIP is not fetched.
"""
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import json
from pathlib import Path
import time
from acquire_experimental_corpus import ROOT,acquire,now,write_json
from extend_coordinate_audit import process as parse_official
from clean_coordinate_pairs import process as fine_process,initialize


def main():
    preliminary=ROOT/'data/experimental/fine_qc_v1'
    out=ROOT/'data/experimental/verified_coordinate_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():
        print('Official-source verification already finalized; keeping its version unchanged.')
        return
    raw=ROOT/'data/raw/coordinate_provenance_verification_v1';raw.mkdir(parents=True,exist_ok=True)
    snapshot=json.loads((ROOT/'data/raw/experimental_acquisition_v1/plan.json').read_text())
    ids=[r['pdb_id'] for r in snapshot['pdb_coverage']]
    jobs={};results={};official_results={};pending={};last=0
    with ThreadPoolExecutor(max_workers=6) as pool:
        while True:
            spool=preliminary/'fine_entry_results.jsonl'
            if spool.exists():
                with spool.open(encoding='utf-8') as f:
                    for line in f:
                        if not line.endswith('\n'): break
                        r=json.loads(line);ident=r['pdb_id']
                        if not r['one_d']: continue
                        if r['one_d'][0]['source_kind']=='official_cif': official_results[ident]=r;continue
                        if ident in jobs: continue
                        job=dict(id='pdb-verification:'+ident,pdb_id=ident,kind='pdb_cif_gz',
                                 url=f'https://files.rcsb.org/download/{ident}.cif.gz',relative_path=ident.lower()+'.cif.gz')
                        jobs[ident]=job;pending[pool.submit(acquire,job,raw)]=ident
            for future in list(pending):
                if future.done():
                    ident=pending.pop(future)
                    try: results[ident]=future.result()
                    except Exception as exc: results[ident]=dict(id=jobs[ident]['id'],status='failed',reason=str(exc))
            if time.monotonic()-last>15:
                write_json(raw/'plan.json',dict(created_at_utc=now(),provisional=not (preliminary/'summary.json').exists(),jobs=list(jobs.values())))
                write_json(out/'source_verification_report.json',dict(updated_at_utc=now(),complete=False,
                           requested=len(jobs),verified=sum(r['status'].startswith('verified') for r in results.values()),
                           results=list(results.values())))
                last=time.monotonic()
            if (preliminary/'summary.json').exists() and not pending: break
            time.sleep(2)
    write_json(raw/'plan.json',dict(created_at_utc=now(),provisional=False,jobs=list(jobs.values())))
    failures=[r for r in results.values() if not r['status'].startswith('verified')]
    write_json(out/'source_verification_report.json',dict(updated_at_utc=now(),complete=True,
               requested=len(jobs),verified=len(jobs)-len(failures),failures=failures,results=list(results.values())))
    fresh=[]
    paths=[ROOT/r['path'] for r in results.values() if r['status'].startswith('verified')]
    audit=out/'official_coordinate_audit.jsonl'
    parsed=set()
    if audit.exists():
        with audit.open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);parsed.add(r['path'])
    with audit.open('a',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(parse_official,((str(p),ids) for p in paths if str(p.relative_to(ROOT)) not in parsed)):
            r['source_kind']='official_cif';f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
    with audit.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if not r['error']: fresh.append(r)
    spool=out/'fine_entry_results.jsonl';done=set()
    if spool.exists():
        with spool.open(encoding='utf-8') as f: done={json.loads(l)['pdb_id'] for l in f}
    with spool.open('a',encoding='utf-8') as f:
        for ident,r in official_results.items():
            if ident not in done: f.write(json.dumps(r,allow_nan=False)+'\n');done.add(ident)
        f.flush()
        with ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
            for i,r in enumerate(pool.map(fine_process,[r for r in fresh if r['entry']['pdb_id'] not in done]),1):
                f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
                if i%100==0: print('Official-source fine QC',i,'/',len(fresh),flush=True)
    write_json(out/'completed.json',dict(created_at_utc=now(),download_failures=failures,
               processed_official_entries=len(fresh)+len(official_results),
               note='Final pool must be consolidated separately; preliminary ZIP-only labels are not approved.'))


if __name__=='__main__': main()
