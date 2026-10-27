"""Publish source URLs/checksums and counts, without redistributing raw data."""
import json
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file


def main():
    reports=['data/reports/experimental_acquisition_v1_all.json',
             'data/reports/additional_experimental_acquisition_v1.json',
             'data/reports/rasp_reference_acquisition_v1.json',
             'data/reports/rasp_bigwig_fallback_v1.json',
             'data/experimental/verified_coordinate_v1/source_verification_report.json']
    records={}
    for name in reports:
        report=json.loads((ROOT/name).read_text(encoding='utf-8'))
        if not report['complete']: raise ValueError('Acquisition still active: '+name)
        for r in report['results']:
            if not r['status'].startswith('verified'): continue
            record={k:r[k] for k in ('id','kind','url','bytes','sha256','downloaded_at_utc','upstream_sha256_verified','source_last_modified') if k in r}
            records[(record['url'],record['sha256'])]=record
    for path in (ROOT/'data/raw/experimental_cif_v1').glob('*.receipt.json'):
        r=json.loads(path.read_text(encoding='utf-8'))
        record=dict(id='existing-cif:'+path.name.split('.')[0].upper(),kind='pdb_cif',
                    url=r['url'],sha256=r['sha256'],downloaded_at_utc=r['downloaded_at_utc'])
        records[(record['url'],record['sha256'])]=record
    directory=ROOT/'data/manifests';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'raw_artifacts_v1.jsonl'
    text=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for _,r in sorted(records.items()))
    if path.exists() and path.read_text(encoding='utf-8')!=text:
        raise ValueError('Frozen source manifest differs; choose a new version')
    path.write_text(text,encoding='utf-8',newline='\n')
    write_json(directory/'raw_artifacts_v1_summary.json',dict(artifacts=len(records),manifest_sha256=digest_file(path),
               source_reports={p:digest_file(ROOT/p) for p in reports},
               note='Official artifact URLs and byte hashes only. Raw datasets/environments/weights are not committed; local ZIP references are not promoted by filename.'))
    print('Frozen official artifact receipts:',len(records))


if __name__=='__main__': main()
