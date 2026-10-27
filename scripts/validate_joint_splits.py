"""Validate evidence, masks, declared isolation and preserved frozen assignments."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
from rna2d.evidence import pairing_evidence_issues
from rna2d.data import CANONICAL


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--previous',type=Path);args=p.parse_args()
    seen=defaultdict(set);ids={};hashes={};counts={}
    for s in ('train','validation','test'):
        path=args.directory/(s+'.jsonl');rs=[json.loads(l) for l in path.open(encoding='utf-8')];assert rs
        for r in rs:
            assert not pairing_evidence_issues(r);assert len(r['sequence'])==len(r['observed_mask'])
            partners=set()
            for i,j in r['pairs']:
                assert 0<=i<j<len(r['sequence']) and j-i>=4
                assert r['sequence'][i]+r['sequence'][j] in CANONICAL
                assert r['observed_mask'][i] and r['observed_mask'][j]
                assert i not in partners and j not in partners;partners.update((i,j))
            for key in [r['sequence_sha256'],r['split_group_id']]+r['study_ids']+r['homology_components']+r['source_pdb_ids']:seen[key].add(s)
            ids[r['id']]=s;hashes[r['sequence_sha256']]=s
        counts[s]=len(rs)
    assert all(len(x)==1 for x in seen.values())
    for e in json.loads((args.directory/'sequence_edges.json').read_text()):
        if e['a'] in ids and e['b'] in ids:assert ids[e['a']]==ids[e['b']]
    for line in (args.directory/'joint_1d_assignments.jsonl').open():
        r=json.loads(line)
        if r['split']!='quarantine':assert hashes[r['sequence_sha256']]==r['split']
    if args.previous:
        for s in counts:
            for line in (args.previous/(s+'.jsonl')).open(encoding='utf-8'):
                r=json.loads(line)
                if r['sequence_sha256'] in hashes:assert hashes[r['sequence_sha256']]==s
    result=dict(passed=True,counts=counts,files={s:digest_file(args.directory/(s+'.jsonl')) for s in counts},
                checks=['evidence and canonical label contract','mask and partner legality','component/study/construct/sequence isolation',
                        'declared local-alignment edges','joint 1D exclusion','previous assignment preservation'])
    write_json(args.directory/'validation.json',result);write_json(ROOT/'data/manifests'/(args.directory.name+'_validation.json'),result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
