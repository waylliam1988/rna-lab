"""Freeze conservative study/construct/homology isolation without model scores.

Unanchored RNA and probing libraries with unresolved parents are quarantined.
No 1D pretraining is enabled by this first conservative release.
"""
import argparse
from collections import defaultdict,Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
from Bio.Align import PairwiseAligner
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
import rna2d
import gemmi
from rna2d.coordinate_qc import rows
from build_experimental_pairs import assign_groups


def load(path):
    with path.open(encoding='utf-8') as f:return [json.loads(l) for l in f]


def study_map(records):
    locations={}
    for report in [ROOT/'data/reports/experimental_acquisition_v1_all.json',ROOT/'data/experimental/verified_coordinate_v1/source_verification_report.json']:
        for r in json.loads(report.read_text())['results']:
            if r.get('kind')=='pdb_cif_gz' and r['status'].startswith('verified'):
                locations[r['id'].split(':')[-1]]=ROOT/r['path']
    for p in (ROOT/'data/raw/experimental_cif_v1').glob('*.cif'):locations[p.stem.upper()]=p
    result={}
    ids={a.split('_')[0] for r in records for a in r.get('alternative_structure_ids',[r['id']])}
    for ident in sorted(ids):
        path=locations.get(ident)
        if not path:result[ident]=[];continue
        data=gzip.open(path,'rb').read() if path.suffix=='.gz' else path.read_bytes()
        block=gemmi.cif.read_string(data.decode('utf-8')).sole_block()
        citations=list(rows(block,['_citation.id','_citation.pdbx_database_id_PubMed','_citation.pdbx_database_id_DOI']))
        primary=[c for c in citations if c[0].lower()=='primary']
        # Missing primary citation is unknown, not an invented study ID.
        keys=[]
        for _,pmid,doi in primary:
            if pmid and pmid not in ('.','?'):keys.append('pmid:'+pmid)
            if doi and doi not in ('.','?'):keys.append('doi:'+doi.lower())
        result[ident]=sorted(set(keys))
    return result


def sequence_edges(records):
    aligner=PairwiseAligner(mode='local',match_score=2,mismatch_score=-1,open_gap_score=-3,extend_gap_score=-0.5)
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j]
            alignments=aligner.align(a['sequence'],b['sequence'])
            if not alignments:continue
            al=alignments[0];counts=al.counts()
            denominator=counts.identities+counts.mismatches+counts.gaps
            identity=counts.identities/denominator if denominator else 0
            covered=sum(y-x for x,y in al.aligned[0])
            coverage=covered/min(len(a['sequence']),len(b['sequence']))
            if identity>=0.8 and coverage>=0.8:
                yield dict(a=a['id'],b=b['id'],identity=identity,shorter_coverage=coverage)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pairs',type=Path,default=ROOT/'data/experimental/clean_release_v1/paired_deduplicated.jsonl')
    parser.add_argument('--output',type=Path,default=ROOT/'data/experimental/joint_split_v1')
    parser.add_argument('--frozen',type=Path)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Frozen split already exists')
    records=load(args.pairs);components=defaultdict(set)
    for r in load(ROOT/'data/experimental/rna3db_all_chain_metadata.jsonl'):
        c=r.get('homology_component')
        if c and c!='component_0':components[r['sequence']].add(c)
    studies=study_map(records);parent=list(range(len(records)));owner={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):parent[find(j)]=find(i)
    for i,r in enumerate(records):
        r['homology_components']=sorted(components[r['sequence']])
        pdbs=sorted({a.split('_')[0] for a in r.get('alternative_structure_ids',[r['id']])})
        r['source_pdb_ids']=pdbs;r['study_ids']=sorted({s for pdb in pdbs for s in studies.get(pdb,[])})
        r['study_metadata_complete']=all(studies.get(pdb) for pdb in pdbs)
        for key in r['homology_components']+['pdb:'+p for p in pdbs]+r['study_ids']:
            if key in owner:union(i,owner[key])
            owner[key]=i
    byid={r['id']:i for i,r in enumerate(records)};edges=list(sequence_edges(records))
    for e in edges:union(byid[e['a']],byid[e['b']])
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[find(i)].append(r)
    eligible=[];quarantine=[];prior={}
    if args.frozen:
        for split in ('train','validation','test'):
            for r in load(args.frozen/(split+'.jsonl')):prior[r['sequence_sha256']]=split
    assigned=defaultdict(list);newgroups=[]
    for group in groups.values():
        old={prior[r['sequence_sha256']] for r in group if r['sequence_sha256'] in prior}
        if len(old)>1:
            for r in group:r['quarantine_reason']='new_evidence_bridges_frozen_splits'
            quarantine.extend(group);continue
        usable=[]
        for r in group:
            if not r['homology_components'] or not r['study_metadata_complete']:
                r['quarantine_reason']='unverified_homology_or_primary_study';quarantine.append(r)
            else:usable.append(r)
        if usable:
            if old:assigned[next(iter(old))].append(usable)
            else:newgroups.append(usable)
    if prior:
        # Keep all previous assignments; new independent groups use a stable hash.
        for group in newgroups:
            value=int(hashlib.sha256(min(r['sequence_sha256'] for r in group).encode()).hexdigest()[:8],16)%10
            assigned['train' if value<8 else 'validation' if value==8 else 'test'].append(group)
    else:assigned=assign_groups(newgroups,42)
    splits={s:[] for s in ('train','validation','test')}
    for split,gs in assigned.items():
        for group in gs:
            h=hashlib.sha256('\n'.join(sorted(r['sequence_sha256'] for r in group)).encode()).hexdigest()
            for r in group:r.update(split=split,split_group_id=h,training_split_ready=True);splits[split].append(r)
    for split,rs in splits.items():
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in sorted(rs,key=lambda r:r['id'])),encoding='utf-8',newline='\n')
    (out/'quarantine.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in quarantine),encoding='utf-8',newline='\n')
    write_json(out/'sequence_edges.json',edges);write_json(out/'study_map.json',studies)
    # Entire unresolved probing libraries stay out of all pretraining. A source
    # file is not silently treated as a validated biological family.
    with (out/'joint_1d_assignments.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        assignment={r['sequence_sha256']:(s,r['split_group_id']) for s,rs in splits.items() for r in rs}
        import sqlite3
        db=sqlite3.connect(ROOT/'data/experimental/clean_release_v1/sequences.sqlite3')
        counts=Counter()
        for (h,) in db.execute('SELECT sha256 FROM sequences ORDER BY sha256'):
            split,group=assignment.get(h,('quarantine',None));counts[split]+=1
            f.write(json.dumps(dict(sequence_sha256=h,split=split,group=group,reason='paired_verified_anchor' if group else 'unresolved_parent_study_or_homology'))+'\n')
    seen=defaultdict(set)
    for split,rs in splits.items():
        for r in rs:
            for k in [r['sequence_sha256'],r['split_group_id']]+r['homology_components']+r['study_ids']+['pdb:'+p for p in r['source_pdb_ids']]:seen[k].add(split)
    assert all(len(v)==1 for v in seen.values())
    assigned_ids={r['id']:s for s,rs in splits.items() for r in rs}
    assert all(assigned_ids[e['a']]==assigned_ids[e['b']] for e in edges if e['a'] in assigned_ids and e['b'] in assigned_ids)
    summary=dict(protocol='joint-isolation-v1',paired_source_sha256=digest_file(args.pairs),
                 splits={s:len(rs) for s,rs in splits.items()},groups={s:len({r['split_group_id'] for r in rs}) for s,rs in splits.items()},
                 quarantined_2d=len(quarantine),joint_1d_assignments=dict(counts),
                 validation_passed=True,training_ready=all(splits.values()),one_d_pretraining_enabled=False,
                 seed=42,identity_min=0.8,shorter_aligned_coverage_min=0.8,
                 limitations=['Known RNA3DB components plus specified local similarity and primary-citation controls; not proof of absence of remote undiscovered homology.',
                              'All unanchored probing constructs remain quarantined; no 1D pretraining is performed.',
                              'Contextual-chain labels are evaluated separately from claims about isolated RNA; small sample sizes remain exploratory.'])
    write_json(out/'summary.json',summary);write_json(ROOT/'data/manifests'/(out.name+'.json'),summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
