"""Reannotate high-quality single-RNA candidates; preserve strict core v1.

The extension approves only per-pair intersections and masks all disagreement
endpoints. No independent annotation, no positive consensus, or <50% usable
labels still rejects the sample. It is a different, explicitly named cohort.
"""
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import json
from acquire_experimental_corpus import ROOT,write_json,digest_file
from clean_coordinate_pairs import initialize,process
from build_joint_splits import load


def main():
    out=ROOT/'data/experimental/annotation_review_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Already finalized')
    source=load(ROOT/'data/experimental/clean_release_v1/coordinate_sequence_records.jsonl')
    selected={r['pdb_id'] for r in source if r['context']=='single_RNA'}
    receipts={r['pdb_id']:r['source_download_receipt'] for r in source}
    entries={}
    for p in [ROOT/'data/experimental/fine_qc_v1/coordinate_entries.jsonl',ROOT/'data/experimental/verified_coordinate_v1/official_coordinate_audit.jsonl']:
        for e in load(p):
            if e.get('entry') and e['entry']['pdb_id'] in selected and e.get('source_kind')=='official_cif':
                e['min_pair_jaccard']=0.0;entries[e['entry']['pdb_id']]=e
    variants=defaultdict(list)
    for r in load(ROOT/'data/experimental/clean_release_v1/paired_deduplicated.jsonl'):
        r['annotation_agreement_policy']='strict_chain';variants[r['sequence_sha256']].append(r)
    counts=Counter();errors=[];n=0
    with (out/'entry_audit.jsonl').open('w',encoding='utf-8',newline='\n') as f,ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
        for result in pool.map(process,entries.values()):
            n+=1;f.write(json.dumps(result)+'\n');f.flush()
            if result['error']:errors.append(result['error'])
            counts.update(a['reason'] for a in result['audit'])
            for r in result['two_d']:
                r['source_download_receipt']=receipts[r['pdb_id']]
                variants[r['sequence_sha256']].append(r)
    accepted=[];conflicts=[]
    for h,rs in variants.items():
        shared=[all(r['observed_mask'][i] for r in rs) for i in range(len(rs[0]['sequence']))]
        signatures={tuple(tuple(p) for p in r['pairs'] if shared[p[0]] and shared[p[1]]) for r in rs}
        if len(signatures)!=1 or not next(iter(signatures)) or sum(shared)/len(shared)<0.5:
            conflicts.append(h);continue
        r=dict(min(rs,key=lambda r:(r.get('annotation_agreement_policy')!='strict_chain',r['context']!='single_RNA',r['resolution_A'],r['id'])))
        r.update(observed_mask=shared,pairs=[list(p) for p in next(iter(signatures))],training_split_ready=False,
                 alternative_structure_ids=sorted({a for x in rs for a in x.get('alternative_structure_ids',[x['id']])}))
        accepted.append(r)
    path=out/'paired.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in sorted(accepted,key=lambda r:r['sequence_sha256'])),encoding='utf-8',newline='\n')
    summary=dict(entries_reviewed=n,source_single_RNA_candidates=len(selected),unique_2d=len(accepted),
                 cohorts=dict(Counter(r['annotation_agreement_policy'] for r in accepted)),
                 contexts=dict(Counter(r['context'] for r in accepted)),conflicting_sequences=len(conflicts),
                 annotation_outcomes=dict(counts),errors=errors,paired_sha256=digest_file(path),
                 scope='Strict core plus explicitly masked partial-consensus single-RNA extension; no fabricated pairs or unverified FR3D-missing labels.')
    write_json(out/'summary.json',summary);write_json(ROOT/'data/manifests/annotation_review_v1.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
