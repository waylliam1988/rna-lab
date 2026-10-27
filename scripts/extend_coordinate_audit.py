"""Screen newly acquired official coordinates; preserve the original ZIP audit."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import sys
from acquire_experimental_corpus import ROOT, now, write_json, digest_file

sys.path.insert(0,str(ROOT/'src'))
from rna2d.coordinate_qc import parse_coordinate_entry


def process(item):
    path, ids = item
    path=Path(path)
    try:
        data=gzip.open(path,'rb').read() if path.suffix=='.gz' else path.read_bytes()
        name=path.name.removesuffix('.gz')
        entry,chains=parse_coordinate_entry(data,name,set(ids))
        return dict(path=str(path.relative_to(ROOT)),entry=entry,chains=chains,error=None)
    except Exception as exc:
        return dict(path=str(path.relative_to(ROOT)),entry=None,chains=[],error=str(exc))


def main():
    out=ROOT/'data/experimental/fine_qc_v1'
    out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'data/raw/experimental_acquisition_v1/plan.json').read_text(encoding='utf-8'))
    ids=[r['pdb_id'] for r in plan['pdb_coverage']]
    paths=[ROOT/'data/raw/experimental_acquisition_v1'/j['relative_path'] for j in plan['jobs'] if j['kind']=='pdb_cif_gz']
    paths+=list((ROOT/'data/raw/experimental_cif_v1').glob('*.cif'))
    spool=out/'fresh_coordinate_audit.jsonl'
    done=set()
    if spool.exists():
        with spool.open(encoding='utf-8') as f:
            done={json.loads(line)['path'] for line in f}
    todo=[p for p in paths if str(p.relative_to(ROOT)) not in done]
    with spool.open('a',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=2) as pool:
        for i,row in enumerate(pool.map(process,((str(p),ids) for p in todo)),1):
            f.write(json.dumps(row,allow_nan=False)+'\n')
            f.flush()
            if i%100==0: print('Fresh coordinate audit',i,'/',len(todo),flush=True)
    by_id={}
    with (ROOT/'data/experimental/coordinate_qc_v1/entry_audit.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['entry'] and r['entry']['pdb_id'] in ids:
                r['source_kind']='audited_local_zip'
                by_id[r['entry']['pdb_id']]=r
    errors=[]
    with spool.open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['error']: errors.append(r); continue
            r['source_kind']='official_cif'
            by_id[r['entry']['pdb_id']]=r
    with (out/'coordinate_entries.jsonl').open('w',encoding='utf-8') as f:
        for ident in sorted(by_id): f.write(json.dumps(by_id[ident],allow_nan=False)+'\n')
    write_json(out/'coordinate_audit_summary.json',dict(created_at_utc=now(),entries=len(by_id),
               current_entries=len(ids),parse_errors=errors,source_plan_sha256=digest_file(ROOT/'data/raw/experimental_acquisition_v1/plan.json'),
               chains=sum(len(r['chains']) for r in by_id.values()),
               screen_pass_chains=sum(c['coordinate_screen_pass'] for r in by_id.values() for c in r['chains']),
               note='Coordinate screen only; local geometry and paired-label review are separate.'))
    print('Extended audit complete',len(by_id),flush=True)


if __name__=='__main__': main()
