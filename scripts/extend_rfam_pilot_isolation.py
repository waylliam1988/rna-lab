"""Confirm RNase P fragment relations missed by an exact-only initial screen."""
import json
import sys
from pathlib import Path
from Bio.Align import PairwiseAligner
from acquire_experimental_corpus import ROOT, digest_file, write_json
from audit_rfam_pilot_isolation import relation_decision
sys.path.insert(0,str(ROOT/'src'))
from rna2d.comparative_alignment import read_stockholm, ungap


def main():
    output=ROOT/'data/manifests/rfam_family_pilot_isolation_v2.json'
    if output.exists():
        raise FileExistsError('Extended isolation review exists')
    base=ROOT/'data/manifests/rfam_family_pilot_isolation_v1.json'
    first=json.loads(base.read_text())
    seeds=read_stockholm(ROOT/'data/raw/rfam_family_pilot_v1/RF00010.sto')['sequences']
    aligner=PairwiseAligner(mode='local',match_score=2,mismatch_score=-1,open_gap_score=-3,extend_gap_score=-.5)
    checked=0;hits=[]
    for split in ['validation','test','archiveII']:
        source=ROOT/'data/processed'/(split+'.jsonl')
        assert digest_file(source)==first['dataset_files_sha256']['legacy_processed:'+split]
        rows=[json.loads(line) for line in source.open() if line.strip()]
        # Names only triage which existing sequences to align. A name alone
        # never grants family identity, independence or label approval.
        rows=sorted([r for r in rows if 'RNP_' in r['id'].upper() or 'RNASEP_' in r['id'].upper()],key=lambda r:r['id'])
        for row in rows:
            found=False
            for accession,aligned in sorted(seeds.items()):
                sequence=ungap(aligned)
                if set(sequence)-set('ACGU'):
                    continue
                al=aligner.align(sequence,row['sequence'])[0];counts=al.counts()
                den=counts.identities+counts.mismatches+counts.gaps
                identity=counts.identities/den if den else 0.
                coverage=sum(y-x for x,y in al.aligned[0])/min(len(sequence),len(row['sequence']))
                checked+=1
                if identity>=.8 and coverage>=.8:
                    hits.append(dict(family='RF00010',dataset='legacy_processed:'+split,record_id=row['id'],seed_accession=accession,identity=identity,shorter_coverage=coverage));found=True;break
            if found:
                break
    detail=ROOT/'data/reports/rfam_family_pilot_v1/rnasep_targeted_alignment.json'
    recorded=json.loads(detail.read_text())
    assert recorded['comparisons']==checked and recorded['hits']==hits
    final=json.loads(json.dumps(first))
    rnasep=next(f for f in final['families'] if f['family']=='RF00010')
    assert hits,'Do not invent a positive relation if fragment search finds none'
    rnasep.update(positive_evaluation_relation_found=True,decision=relation_decision(True,True),targeted_fragment_relation_count=len(hits),fragment_relations=[{k:v for k,v in hit.items() if k not in ['record_id','seed_accession']} for hit in hits])
    final.update(stage='2_isolation_review_extended',parent_v1_sha256=digest_file(base),targeted_comparisons=checked,targeted_detail_sha256=digest_file(detail),all_three_blocked=True,representatives_selected=0,test_performance_read=False,source_script_sha256=digest_file(Path(__file__)))
    final['limitations'].append('Initial v1 exact/component screen left RNase P unresolved; targeted alignment of historical RNP/RNaseP-labelled fragments established a positive relation. Non-hits are not independence evidence.')
    write_json(output,final)
    print(json.dumps(dict(comparisons=checked,positive_fragment_relations=len(hits),all_three_blocked=True,eligible_families=0)))


if __name__=='__main__':
    main()
