"""Freeze physical forecasting protocol and explicitly synthetic transitions."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
from build_joint_splits import load
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.interface import PhysicalInterface


def main():
    torch.set_num_threads(1)
    out=ROOT/'data/experimental/physical_forecast_v1';out.mkdir(exist_ok=True)
    protocol_path=ROOT/'data/manifests/physical_forecast_protocol_v1.json'
    if protocol_path.exists():raise ValueError('Protocol already frozen')
    protocol=dict(task='teacher_forced_next_physical_observation',source='three_d_isolated_v1',seeds=[42,43,44],training_update_checkpoints=[100,500],training_rollout_lengths=[4,8],evaluation_history_lengths=[4,8,16],batch_size=8,optimizer='AdamW',lr=.001,weight_decay=.0001,gradient_clip=1.,original_ghost_dims=[256,512,8],perception_adapter='new Linear(18,256)+LayerNorm',trajectory_length=32,initial_position_noise_nm=.02,episodes_per_sequence=4,physics_temperature=0.,physics_dt=.05,contact_prior='sequence_canonical_0.1_abs_sep_ge4',target='original log1p Rg/force, contact densities, normalized asinh energy',baselines=['persistence','same_trained_model_reset_each_step'],selection='No hyperparameter selection; report all frozen checkpoints/seeds, validation only',new_test_available=False,limits=['All transitions are synthetic auxiliary labels from restored analytical physics, not experimental trajectories.','Known current 3D state is input; not a sequence-to-fold benchmark.','Validation has five sequences/five groups; no independent test conclusion.','No 1D pretraining or old learned weights; no full original GVP/policy/metric restoration.','Teacher forcing checks state estimation/forecasting, not learned free-running folding.'])
    engine=PhysicalInterface();fingerprints={};counts={}
    for split in ('train','validation'):
        rows=load(ROOT/'data/experimental/three_d_isolated_v1'/(split+'.jsonl'))
        representatives={}
        for row in sorted(rows,key=lambda r:r['id']):representatives.setdefault(row['sequence_sha256'],row)
        episodes=[];meta=[]
        for h,row in sorted(representatives.items()):
            seq=row['sequence'];n=len(seq);m=torch.ones(1,n);q=torch.tensor(row['q_nm']).unsqueeze(0)
            j=torch.tensor([[float(a+b in ('AU','UA','GC','CG','GU','UG') and abs(i-k)>=4) for k,b in enumerate(seq)] for i,a in enumerate(seq)]).unsqueeze(0)*.1
            for episode in range(4):
                seed=int(hashlib.sha256(f'physical-v1:{h}:{episode}'.encode()).hexdigest()[:8],16)
                generator=torch.Generator().manual_seed(seed);start=q.clone()
                start[...,:3]+=torch.randn(start[...,:3].shape,generator=generator)*.02
                state=engine.initial_state(start,m);observations=[]
                with torch.no_grad():
                    for step in range(33):
                        obs,_,_,_=engine.observe(state,j,m);observations.append(obs[0])
                        if step<32:state,_=engine.transition(state,j,m)
                episodes.append(torch.stack(observations).numpy())
                meta.append(dict(id=row['id'],sequence_sha256=h,split_group_id=row['split_group_id'],episode=episode,seed=seed,transition_evidence='simulator_generated_auxiliary',static_initial_evidence='official_experimental_coordinates_plus_labeled_noise'))
        array=np.stack(episodes)
        assert array.shape[1:]==(33,18) and np.isfinite(array).all()
        np.savez_compressed(out/(split+'.npz'),observations=array)
        write_json(out/(split+'_metadata.json'),meta)
        fingerprints[split]=dict(source_sha256=digest_file(ROOT/'data/experimental/three_d_isolated_v1'/(split+'.jsonl')),array_sha256=hashlib.sha256(array.tobytes()).hexdigest(),file_sha256=digest_file(out/(split+'.npz')),metadata_sha256=digest_file(out/(split+'_metadata.json')))
        counts[split]=dict(sequences=len(representatives),episodes=len(episodes),transitions=len(episodes)*32)
    # Frozen before fitting: 250 updates x 8 steps matches the observation
    # exposure of 500 updates x 4 steps, in addition to fixed-update comparisons.
    protocol['training_update_checkpoints']=[100,250,500]
    protocol['common_terminal_evaluation_state']=16
    protocol['rollout_exposure_control']='r8@250 versus r4@500, batch8, 16000 observed states each'
    protocol.update(data_fingerprints=fingerprints,counts=counts)
    write_json(protocol_path,protocol);print(json.dumps(dict(counts=counts,protocol_sha256=digest_file(protocol_path)),indent=2))


if __name__=='__main__':main()
