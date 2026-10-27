"""Trace static J-density channels from raw physics to saved model inputs."""
import ast
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
from rna2d.physics.reference import compute_rna_force_field_fast


def channel_stats(a):
    c=a[...,1:3]
    return dict(shape=list(a.shape),min=c.min(axis=(0,1)).tolist(),max=c.max(axis=(0,1)).tolist(),max_temporal_range=np.ptp(c,axis=1).max(0).tolist(),between_case_std=c[:,0].astype(np.float64).std(0).tolist(),nonzero_values=np.count_nonzero(c,axis=(0,1)).tolist())


def main():
    torch.set_num_threads(1)
    output=ROOT/'data/manifests/contact_channel_audit_v1.json'
    if output.exists():raise ValueError('Audit version already frozen')
    datasets={};normalization={}
    specifications=[('physical_forecast_v1',ROOT/'data/experimental/physical_forecast_v1',ROOT/'runs/physical_forecast_v1_r4/seed42_step500.pt'),('gate3_v1',ROOT/'data/simulated/gate3_v1',ROOT/'runs/gate3_v1_seed42/ghost_full.pt')]
    for name,root,checkpoint in specifications:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        mean,scale=(ck['input_mean'],ck['input_scale']) if name=='physical_forecast_v1' else ck['stats'][:2]
        per_split={}
        for split in ('train','validation'):
            a=np.load(root/(split+'.npz'))['observations']
            x=torch.tensor(a);z=(x-mean)/scale;restored=z*scale+mean
            assert float((restored-x).abs().max())<1e-5
            datasets[name+':'+split]=dict(source_sha256=digest_file(root/(split+'.npz')),**channel_stats(a))
            per_split[split]=dict(normalized=channel_stats(z.numpy()),max_inverse_error=float((restored[...,1:3]-x[...,1:3]).abs().max()))
            assert np.ptp(a[...,1:3],axis=1).max()==0 and np.count_nonzero(a[...,1:3])==a.shape[0]*a.shape[1]*2
            if name=='gate3_v1':assert np.count_nonzero(np.load(root/(split+'.npz'))['delta'][...,[1,2]])==0
        normalization[name]=dict(checkpoint_sha256=digest_file(checkpoint),saved_mean_channels=mean[1:3].tolist(),saved_scale_channels=scale[1:3].tolist(),splits=per_split)
    # Freshly replay raw pre-model observations; log the distinct geometry-based
    # aux statistic too. It is not the feature called c_mean in the macro block.
    engine=PhysicalInterface();replays=[];local=ROOT/'data/reports/contact_channel_audit_v1';local.mkdir(parents=True,exist_ok=True)
    for split in ('train','validation'):
        root=ROOT/'data/experimental/physical_forecast_v1';meta=json.loads((root/(split+'_metadata.json')).read_text())[0]
        rows={r['id']:r for r in load(ROOT/'data/experimental/three_d_isolated_v1'/(split+'.jsonl'))};r=rows[meta['id']]
        q=torch.tensor(r['q_nm']).unsqueeze(0);seq=r['sequence'];n=len(seq);m=torch.ones(1,n)
        j=torch.tensor([[float(a+b in ('AU','UA','GC','CG','GU','UG') and abs(i-k)>=4) for k,b in enumerate(seq)] for i,a in enumerate(seq)]).unsqueeze(0)*.1
        generator=torch.Generator().manual_seed(meta['seed']);q[...,:3]+=torch.randn(q[...,:3].shape,generator=generator)*.02
        state=engine.initial_state(q,m);initial=state.q.clone();trace=[]
        saved=np.load(root/(split+'.npz'))['observations'][0]
        with torch.no_grad():
            for t in range(33):
                obs=engine.observe(state,j,m)[0][0]
                assert np.allclose(obs.numpy(),saved[t],rtol=2e-6,atol=2e-6)
                _,_,aux=compute_rna_force_field_fast(state.q,j,m,engine.ff_params,contact_rc=1.,return_aux=True)
                trace.append(dict(step=t,J_density_mean=float(obs[1]),J_density_max=float(obs[2]),geometry_contact_mean=float(aux['contact_mean'][0]),geometry_contact_std=float(aux['contact_std'][0])))
                if t<32:state,_=engine.transition(state,j,m)
        path=local/(split+'_raw_trace.json');write_json(path,trace)
        aux_range=float(np.ptp([t['geometry_contact_mean'] for t in trace]));assert aux_range>1e-7
        replays.append(dict(split=split,id=r['id'],states_replayed=33,J_fixed=True,mask_fixed=True,mask_valid_residues=n,macro_contact_temporal_range=[float(np.ptp([t[k] for t in trace])) for k in ('J_density_mean','J_density_max')],geometry_contact_mean_range=aux_range,first_geometry_contact_mean=trace[0]['geometry_contact_mean'],last_geometry_contact_mean=trace[-1]['geometry_contact_mean'],max_position_change_nm=float((state.q[...,:3]-initial[...,:3]).abs().max()),raw_trace_sha256=digest_file(path)))
    original=Path('D:/Nezha/Nezha-bio');source=(original/'rna_system.py').read_text(encoding='utf-8-sig');tree=ast.parse(source);bio=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BioSystem')
    config=ast.parse((original/'config.py').read_text(encoding='utf-8-sig'));cls=next(n for n in config.body if isinstance(n,ast.ClassDef) and n.name=='Config')
    gate={}
    for node in cls.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('RNA_MSA_GATE_ENABLE','RNA_MSA_GATE_USE_ENERGY'):
                    gate[target.id]=ast.literal_eval(node.value)
    original_lines={name:next(n.lineno for n in bio.body if getattr(n,'name',None)==name) for name in ('_fold_precompute_sums_and_contacts','_fold_build_obs_vec','simulate_fold')}
    report=dict(channel_numbering='one-based macro2/3; zero-based indices1/2',features=['mean masked row density of input J','maximum masked row density of input J'],source_is_nonzero_static_context=True,statement='当前设定下不携带动态信息',preprocessing_zeroing_detected=False,raw_datasets=datasets,normalization=normalization,raw_physics_replays=replays,original_source=dict(rna_system_sha256=digest_file(original/'rna_system.py'),method_lines=original_lines,msa_gate_config=gate,limitation='Full original BioSystem trajectories were not executed. Its optional energy-driven J gate can make these channels vary; that outer gate is not present in the current fixed-J interface.'),actual_observation_dimensions=18,static_channels_in_macro_fraction=.4,static_channels_in_observation_fraction=2/18,source_sha256=digest_file(Path(__file__)))
    write_json(output,report);print(json.dumps(dict(datasets=datasets,replays=replays,normalization={k:{a:b for a,b in v.items() if a!='splits'} for k,v in normalization.items()},original_gate=gate),indent=2))


if __name__=='__main__':main()
