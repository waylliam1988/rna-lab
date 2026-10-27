"""Collect official raw BigWig alternatives for unavailable BED downloads.

Keep failed original URLs visible; alternate formats are separate artifacts.
"""
import json
from urllib.parse import urlencode
from acquire_experimental_corpus import ROOT, acquire, now, write_json


def main():
    report=json.loads((ROOT/'data/reports/additional_experimental_acquisition_v1.json').read_text(encoding='utf-8'))
    source=json.loads((ROOT/'data/raw/additional_experimental_acquisition_v1/plan.json').read_text(encoding='utf-8'))
    failed={r['id'] for r in report['failures']}
    directory=ROOT/'data/raw/rasp_bigwig_fallback_v1'
    directory.mkdir(parents=True,exist_ok=True)
    plan_path=directory/'plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
    else:
        jobs=[]
        for original in source['jobs']:
            if original['id'] not in failed or not original['id'].startswith('rasp-scores:'): continue
            for i,row in enumerate(original['source_track_rows']):
                if row['tracktype']!='raw': raise ValueError('Non-original track')
                jobs.append(dict(id=original['id']+':bw:'+str(i),kind='tar',
                                 url='https://rasp2.zhanglab.net/api/downloadfile/?'+urlencode({'filetype':'bw','filelist':row['file_path']}),
                                 relative_path=original['id'].replace(':','_')+'_'+str(i)+'.tar',
                                 original_unavailable_artifact=original['id'],source_track=row,
                                 evidence_class='original_experimental_signal_bigwig_not_pair_truth'))
        plan=dict(created_at_utc=now(),jobs=jobs)
        write_json(plan_path,plan)
    results=[acquire(j,directory) for j in plan['jobs']]
    write_json(ROOT/'data/reports/rasp_bigwig_fallback_v1.json',
               dict(complete=True,requested_jobs=len(results),
                    verified_jobs=sum(r['status'].startswith('verified') for r in results),
                    failures=[r for r in results if not r['status'].startswith('verified')],results=results))
    print([(r['id'],r['status']) for r in results])


if __name__=='__main__': main()
