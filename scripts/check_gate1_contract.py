"""Gate 1: chronology, causal isolation, counterfactuals, train-only fitting."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT, write_json, digest_file
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.interface import PhysicalInterface, normalized_macro
from rna2d.physics.forecast import PhysicalForecast, window_batch


def setup_record():
    rows=[json.loads(l) for l in (ROOT/'data/experimental/three_d_isolated_v1/train.jsonl').read_text(encoding='utf-8').splitlines()]
    row=sorted(rows,key=lambda r:r['id'])[0]
    q=torch.tensor(row['q_nm']).unsqueeze(0);m=torch.ones(q.shape[:2]);seq=row['sequence']
    j=torch.tensor([[float(a+b in ('AU','UA','GC','CG','GU','UG') and abs(i-k)>=4) for k,b in enumerate(seq)] for i,a in enumerate(seq)]).unsqueeze(0)*.1
    return row,q,m,j


def contract_checks():
    torch.set_num_threads(1);engine=PhysicalInterface();row,q,m,j=setup_record()
    s=engine.initial_state(q,m);center=q[...,:3].mean(1,keepdim=True)
    radial=torch.nn.functional.normalize(q[...,:3]-center,dim=-1)
    action=torch.cat([radial*.4,torch.zeros_like(radial)],dim=-1)
    original=s.q.clone();p0=s.p.clone();x0=s.xi.clone()
    with torch.no_grad():
        zero,_=engine.transition(s,j,m)
        plus,_=engine.transition(s,j,m,action)
        minus,_=engine.transition(s,j,m,-action)
        assert torch.equal(s.q,original) and torch.equal(s.p,p0) and torch.equal(s.xi,x0)
        assert ((plus.q[...,:3]-minus.q[...,:3])*radial).sum().item()>0
        assert torch.allclose((plus.q[...,:3]+minus.q[...,:3])*.5,zero.q[...,:3],atol=2e-6,rtol=2e-6)
        o0=engine.observe(s,j,m)[0];o1=engine.observe(plus,j,m)[0]
        batch=torch.stack([o0[0],o1[0]])[None]
        inp,target=window_batch(batch,[0],[0],1)
        assert torch.equal(inp[0,0],o0[0]) and torch.equal(target[0,0],o1[0])
        assert not torch.equal(inp,target)
    # Verify existing v1 timestamps against a fresh replay of its first episode.
    meta=json.loads((ROOT/'data/experimental/physical_forecast_v1/train_metadata.json').read_text())
    records={r['id']:r for r in [json.loads(l) for l in (ROOT/'data/experimental/three_d_isolated_v1/train.jsonl').read_text(encoding='utf-8').splitlines()]}
    r=records[meta[0]['id']];seq=r['sequence'];n=len(seq)
    q=torch.tensor(r['q_nm']).unsqueeze(0);m=torch.ones(1,n)
    j=torch.tensor([[float(a+b in ('AU','UA','GC','CG','GU','UG') and abs(i-k)>=4) for k,b in enumerate(seq)] for i,a in enumerate(seq)]).unsqueeze(0)*.1
    g=torch.Generator().manual_seed(meta[0]['seed']);q[...,:3]+=torch.randn(q[...,:3].shape,generator=g)*.02
    state=engine.initial_state(q,m);saved=np.load(ROOT/'data/experimental/physical_forecast_v1/train.npz')['observations'][0]
    with torch.no_grad():
        for t in range(33):
            obs=engine.observe(state,j,m)[0][0].numpy()
            assert np.allclose(obs,saved[t],rtol=2e-6,atol=2e-6)
            if t<32:state,_=engine.transition(state,j,m)
    # Nonzero prediction head avoids a trivially constant untrained output.
    torch.manual_seed(101);model=PhysicalForecast().eval()
    torch.nn.init.normal_(model.ghost.cortex.phys_hallucination.weight,std=.01)
    x=torch.randn(2,8,18);changed=x.clone();changed[:,4:]+=100
    with torch.no_grad():
        a=model(x);b=model(changed)
        assert torch.equal(a[:,:4],b[:,:4])
        assert (a[:,4:]-b[:,4:]).abs().max()>0
    return dict(timing_and_action_alignment=True,input_state_not_mutated=True,radial_action_sign_and_symmetry=True,v1_33_states_replayed=True,future_isolation=True,source_training_id=row['id'])


def fit_small():
    device='cuda:0';torch.manual_seed(123);np.random.seed(123)
    episodes=torch.tensor(np.load(ROOT/'data/experimental/physical_forecast_v1/train.npz')['observations'][:1,:17],device=device)
    indices=[0]*12;starts=list(range(12));raw,nxt=window_batch(episodes,indices,starts,4)
    mean=raw.mean((0,1));scale=raw.std((0,1)).clamp_min(1e-5)
    target=normalized_macro(nxt);ym=target.mean((0,1));ys=target.std((0,1)).clamp_min(1e-5)
    x=(raw-mean)/scale;y=(target-ym)/ys;channels=[0,3,4]
    models={'original_ghost':PhysicalForecast(device=device),'mlp_positive_control':torch.nn.Sequential(torch.nn.Linear(18,128),torch.nn.SiLU(),torch.nn.Linear(128,5)).to(device)}
    results={}
    for name,model in models.items():
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
        start=time.time()
        def error():return (model(x)[...,channels]-y[...,channels]).square().mean()
        initial=float(error().detach())
        for step in range(500):
            loss=error();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        model.eval()
        with torch.no_grad():final=float(error())
        results[name]=dict(updates=500,initial_standardized_mse=initial,final_standardized_mse=final,pass_threshold=.01,passed=final<.01,elapsed_seconds=time.time()-start)
        print(json.dumps(dict(small_fit=name,**results[name])),flush=True)
    return results


def main():
    out=ROOT/'data/manifests/gate1_contract_v1.json'
    if out.exists():raise ValueError('Gate result already frozen')
    protocol=dict(seed=123,updates=500,lr=.001,weight_decay=0.,source='one train-only physical_forecast_v1 episode',window=4,windows=12,target='train-only standardized original macro targets',channels=[0,3,4],pass_threshold_standardized_mse=.01,models=['original_ghost','mlp_positive_control'],limitations=['In-sample implementation check, not generalization.','Normalization changes the small-fit diagnostic loss scale; the previous frozen experiment is untouched.'])
    write_json(ROOT/'data/manifests/gate1_small_fit_protocol_v1.json',protocol)
    checks=contract_checks();fits=fit_small()
    report=dict(checks=checks,small_fit=fits,all_checks_passed=all(v['passed'] for v in fits.values()),source_sha256=digest_file(Path(__file__)),protocol_sha256=digest_file(ROOT/'data/manifests/gate1_small_fit_protocol_v1.json'),experimental_trajectory_labels=False)
    write_json(out,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
