"""Build labeled synthetic counterfactuals from isolated experimental starts."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
from build_joint_splits import load
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.interface import PhysicalInterface,PhysicalState,normalized_macro
from rna2d.physics.causal_probe import radial_force,momentum_summary


def main():
    torch.set_num_threads(1);out=ROOT/'data/simulated/gate3_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Version already frozen')
    protocol=dict(seeds=[42,43,44],updates=500,batch_size=16,lr=.001,weight_decay=.0001,gradient_clip=1.,history_length=8,horizons=[1,4,8,16],actions=[-.4,0.,.4],episodes_per_sequence=2,anchor_step=8,temperature=0.,dt=.05,variants=['ghost_obs','ghost_momentum','ghost_action','ghost_full','mlp_obs','mlp_full'],target='change in original normalized macro observations',loss='MSE of change divided by train-only per-horizon channel std, floor1e-5; dynamic channels0,3,4',baselines=['persistence','causal_linear_trend','train_mean_change'],diagnostics=['history_shuffle_preserving_current','reset_every_step','momentum_drop','action_drop','matched_counterfactual_action_shuffle','action_effect_error'],mlp_capacity='match total Ghost parameter count within1%; width selected by parameters only',selection='All fixed500-update seeds reported; no validation selection',limits=['Simulator-generated auxiliary transitions, not experimental folding trajectories.','Current physical observations are input; no sequence-only folding claim.','Only35 train sequences and5 development validation; no fresh independent test.','Momentum is an8-dimensional summary, not full momenta; original GVP/policy/metric not restored.','Drop/shuffle/reset are distribution-shift diagnostics; independently trained variants are the principal controls.','Ghost total parameter count includes unused original bus/goal parameters; equal count does not equal active capacity or FLOPs.'])
    write_json(ROOT/'data/manifests/gate3_protocol_v1.json',protocol)
    engine=PhysicalInterface();summary={}
    for split in ('train','validation'):
        source=ROOT/'data/experimental/three_d_isolated_v1'/(split+'.jsonl');reps={}
        for row in sorted(load(source),key=lambda r:r['id']):reps.setdefault(row['sequence_sha256'],row)
        observations=[];momenta=[];actions=[];targets=[];meta=[];anchors=[]
        for h,row in sorted(reps.items()):
            seq=row['sequence'];n=len(seq);m=torch.ones(1,n);native=torch.tensor(row['q_nm']).unsqueeze(0)
            j=torch.tensor([[float(a+b in ('AU','UA','GC','CG','GU','UG') and abs(i-k)>=4) for k,b in enumerate(seq)] for i,a in enumerate(seq)]).unsqueeze(0)*.1
            for episode in range(2):
                seed=int(hashlib.sha256(f'gate3-v1:{h}:{episode}'.encode()).hexdigest()[:8],16)
                g=torch.Generator().manual_seed(seed);q=native.clone();q[...,:3]+=torch.randn(q[...,:3].shape,generator=g)*.02
                state=engine.initial_state(q,m);hist=[];mh=[]
                with torch.no_grad():
                    for t in range(9):
                        if t>0:hist.append(engine.observe(state,j,m)[0][0].numpy());mh.append(momentum_summary(state)[0].numpy())
                        if t<8:state,_=engine.transition(state,j,m)
                    current=normalized_macro(engine.observe(state,j,m)[0])[0].numpy()
                    anchor_index=len(anchors);anchors.append(dict(q=state.q,p=state.p,xi=state.xi,j=j,mask=m,id=row['id'],episode=episode))
                    for amplitude in protocol['actions']:
                        fork=PhysicalState(state.q.clone(),state.p.clone(),state.xi.clone());future=[]
                        for t in range(1,17):
                            fork,_=engine.transition(fork,j,m,radial_force(fork,amplitude))
                            if t in protocol['horizons']:future.append(normalized_macro(engine.observe(fork,j,m)[0])[0].numpy())
                        action=np.zeros((8,1),np.float32);action[-1]=amplitude
                        observations.append(np.stack(hist));momenta.append(np.stack(mh));actions.append(action);targets.append(np.stack(future)-current)
                        meta.append(dict(id=row['id'],sequence_sha256=h,split_group_id=row['split_group_id'],episode=episode,seed=seed,amplitude=amplitude,anchor_index=anchor_index,evidence='simulator_generated_auxiliary',static_source='isolated_official_experimental_coordinates'))
        arrays=dict(observations=np.stack(observations),momentum=np.stack(momenta),actions=np.stack(actions),delta=np.stack(targets))
        assert all(np.isfinite(v).all() for v in arrays.values())
        np.savez_compressed(out/(split+'.npz'),**arrays);write_json(out/(split+'_metadata.json'),meta);torch.save(anchors,out/(split+'_anchors.pt'))
        summary[split]=dict(sequences=len(reps),anchors=len(anchors),counterfactual_cases=len(meta),horizon_queries=len(meta)*4,source_sha256=digest_file(source),npz_sha256=digest_file(out/(split+'.npz')),metadata_sha256=digest_file(out/(split+'_metadata.json')),anchors_sha256=digest_file(out/(split+'_anchors.pt')))
        print(json.dumps(dict(split=split,**summary[split])),flush=True)
    report=dict(splits=summary,protocol_sha256=digest_file(ROOT/'data/manifests/gate3_protocol_v1.json'),script_sha256=digest_file(Path(__file__)),experimental_trajectory_labels=False)
    write_json(out/'summary.json',report);write_json(ROOT/'data/manifests/gate3_data_v1.json',report)


if __name__=='__main__':main()
