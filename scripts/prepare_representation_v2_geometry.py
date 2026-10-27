"""Replay existing histories; add geometry inputs without changing any target."""
import json
import sys
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
from build_joint_splits import load
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.interface import PhysicalInterface
from rna2d.physics.geometric_contact_v2 import geometric_contacts


def main():
    torch.set_num_threads(1);out=ROOT/'data/simulated/representation_v2';out.mkdir(parents=True,exist_ok=True)
    public=ROOT/'data/manifests/representation_v2_geometry_data.json'
    if public.exists():raise ValueError('Version already frozen')
    # The user-requested static item must finish before starting this item.
    for seed in (42,43,44):assert (ROOT/'data/manifests'/f'representation_v2_static_seed{seed}.json').exists()
    engine=PhysicalInterface();splits={}
    for split in ('train','validation'):
        base=ROOT/'data/simulated/gate3_v1';meta=json.loads((base/(split+'_metadata.json')).read_text())
        raw=np.load(base/(split+'.npz'))['observations'];extra=np.zeros((*raw.shape[:2],2),dtype=np.float32)
        originals={r['id']:r for r in load(ROOT/'data/experimental/three_d_isolated_v1'/(split+'.jsonl'))}
        anchors=torch.load(base/(split+'_anchors.pt'),weights_only=True);cache={};max_obs_error=0.;max_anchor_error=0.
        for i,item in enumerate(meta):
            ai=item['anchor_index']
            if ai not in cache:
                row=originals[item['id']];seq=row['sequence'];n=len(seq);m=torch.ones(1,n);q=torch.tensor(row['q_nm']).unsqueeze(0)
                j=torch.tensor([[float(a+b in ('AU','UA','GC','CG','GU','UG') and abs(k-l)>=4) for l,b in enumerate(seq)] for k,a in enumerate(seq)]).unsqueeze(0)*.1
                g=torch.Generator().manual_seed(item['seed']);q[...,:3]+=torch.randn(q[...,:3].shape,generator=g)*.02
                state=engine.initial_state(q,m);features=[]
                with torch.no_grad():
                    for step in range(9):
                        if step:
                            obs=engine.observe(state,j,m)[0][0].numpy();difference=float(np.abs(obs-raw[i,step-1]).max());max_obs_error=max(max_obs_error,difference)
                            assert np.allclose(obs,raw[i,step-1],rtol=2e-6,atol=2e-6)
                            features.append(geometric_contacts(engine,state,j,m)[0].numpy())
                        if step<8:state,_=engine.transition(state,j,m)
                for key in ('q','p','xi'):
                    error=float((getattr(state,key)-anchors[ai][key]).abs().max());max_anchor_error=max(max_anchor_error,error);assert error<2e-6
                cache[ai]=np.stack(features)
            extra[i]=cache[ai]
        assert np.isfinite(extra).all()
        path=out/(split+'.npz');np.savez_compressed(path,geometry_contacts=extra)
        ranges=np.ptp(extra,axis=1)
        splits[split]=dict(cases=len(meta),unique_anchors=len(cache),shape=list(extra.shape),npz_sha256=digest_file(path),base_npz_sha256=digest_file(base/(split+'.npz')),base_metadata_sha256=digest_file(base/(split+'_metadata.json')),base_anchor_sha256=digest_file(base/(split+'_anchors.pt')),max_original_observation_difference=max_obs_error,max_replayed_anchor_difference=max_anchor_error,min=extra.min(axis=(0,1)).tolist(),max=extra.max(axis=(0,1)).tolist(),max_temporal_range=ranges.max(0).tolist(),fraction_cases_with_temporal_range_gt1e_8=(ranges>1e-8).mean(0).tolist())
        print(json.dumps(dict(split=split,**splits[split])),flush=True)
    report=dict(splits=splits,protocol_sha256=digest_file(ROOT/'data/manifests/representation_v2_protocol.json'),prediction_targets_changed=False,physics_changed=False,J_gate_enabled=False,script_sha256=digest_file(Path(__file__)),geometry_feature_source_sha256=digest_file(ROOT/'src/rna2d/physics/geometric_contact_v2.py'))
    write_json(out/'summary.json',report);write_json(public,report)


if __name__=='__main__':main()
