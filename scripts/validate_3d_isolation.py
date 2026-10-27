"""Check frozen 3D artifacts and replay physical interfaces, without fitting."""
import json
import sys
from collections import defaultdict
from acquire_experimental_corpus import ROOT, write_json, digest_file
from build_joint_splits import load
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.interface import PhysicalInterface


def main():
    torch.set_num_threads(1)
    root=ROOT/'data/experimental/three_d_isolated_v1'
    summary=json.loads((root/'summary.json').read_text())
    exposed={r['sequence_sha256'] for r in load(ROOT/'data/experimental/joint_split_v2/test.jsonl')}
    seen=defaultdict(set);engine=PhysicalInterface();checks=[]
    for split in ('train','validation','test'):
        assert digest_file(root/(split+'.jsonl'))==summary['output_sha256'][split]
        for r in load(root/(split+'.jsonl')):
            assert r['sequence_sha256'] not in exposed
            assert r['training_split_ready'] and not r['folding_trajectory_labels']
            for key in [r['sequence_sha256'],r['split_group_id']]+r['study_ids']+r['homology_components']:seen[key].add(split)
            q=torch.tensor(r['q_nm']).unsqueeze(0);m=torch.ones(q.shape[:2]);n=q.shape[1]
            assert q.shape[-1]==6 and torch.isfinite(q).all()
            assert torch.allclose(q[...,3:].norm(dim=-1),m,atol=1e-5)
            # Sequence-only uniform allowed-pair prior, never native distance labels.
            seq=r['sequence'];j=torch.tensor([[float(a+b in ('AU','UA','GC','CG','GU','UG') and abs(i-k)>=4) for k,b in enumerate(seq)] for i,a in enumerate(seq)]).unsqueeze(0)*.1
            s=engine.initial_state(q,m)
            with torch.no_grad():
                obs,_,_,energy=engine.observe(s,j,m)
                next_s,_=engine.transition(s,j,m)
                assert torch.isfinite(obs).all() and torch.isfinite(next_s.q).all()
            checks.append(dict(id=r['id'],split=split,length=n,energy=float(energy[0]),max_step_displacement_nm=float((next_s.q[...,:3]-s.q[...,:3]).abs().max())))
    assert all(len(v)==1 for v in seen.values())
    report=dict(checked_structures=len(checks),all_checks_passed=True,fresh_test_available=bool(summary['splits']['test']['structures']),trajectory_labels_are_experimental=False,checks=checks)
    write_json(root/'validation.json',report)
    public={k:v for k,v in report.items() if k!='checks'}
    public['report_sha256']=digest_file(root/'validation.json')
    write_json(ROOT/'data/manifests/three_d_isolation_validation_v1.json',public)
    print(json.dumps(public,indent=2))


if __name__=='__main__':main()
