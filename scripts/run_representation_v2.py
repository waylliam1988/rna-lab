"""Paired static-context ablation, followed by geometry additions; no gate."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
from run_gate3_probe import load_data,score,action_effect
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.forecast import PhysicalForecast
from rna2d.physics.representation_v2 import RepresentationProbe
from rna2d.physics.diagnostics import macro_numpy,group_mean,DYNAMIC_CHANNELS as C


def get_data(split,phase,device):
    data,meta=load_data(split,device)
    if phase=='static':extra=torch.zeros((*data['observations'].shape[:2],2),device=device)
    else:
        path=ROOT/'data/simulated/representation_v2'/(split+'.npz')
        extra=torch.tensor(np.load(path)['geometry_contacts'],device=device)
    data['observations']=torch.cat([data['observations'],extra],dim=-1)
    return data,meta


def predict(model,data,stats,device):
    n=len(data['observations']);result=np.zeros((n,4,5))
    with torch.no_grad():
        for hi,h in enumerate((1,4,8,16)):
            for start in range(0,n,32):
                stop=min(start+32,n);horizon=torch.full((stop-start,),h,device=device)
                y=model(data['observations'][start:stop],data['momentum'][start:stop],data['actions'][start:stop],horizon,*stats[:4])
                result[start:stop,hi]=(y*stats[4][hi]).cpu().numpy()
    result[:,:,[1,2]]=0
    return result


def weights_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['static','geometry'],required=True);parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--device',required=True);args=parser.parse_args()
    torch.set_num_threads(1);device=args.device
    protocol_path=ROOT/'data/manifests/representation_v2_protocol.json';p=json.loads(protocol_path.read_text())
    base_manifest=json.loads((ROOT/'data/manifests/gate3_data_v1.json').read_text())
    for split in ('train','validation'):
        assert digest_file(ROOT/'data/simulated/gate3_v1'/(split+'.npz'))==base_manifest['splits'][split]['npz_sha256']
    if args.phase=='geometry':
        geometry_manifest=json.loads((ROOT/'data/manifests/representation_v2_geometry_data.json').read_text())
        for split in ('train','validation'):assert digest_file(ROOT/'data/simulated/representation_v2'/(split+'.npz'))==geometry_manifest['splits'][split]['npz_sha256']
    out=ROOT/'runs'/f'representation_v2_{args.phase}_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise ValueError('Run frozen')
    train,tm=get_data('train',args.phase,device);val,vm=get_data('validation',args.phase,device)
    mean=train['observations'].mean((0,1));std=train['observations'].std((0,1));floor=torch.full((20,),.01,device=device);floor[18:]=1e-5;std=torch.maximum(std,floor)
    mm=train['momentum'].mean((0,1));ms=train['momentum'].std((0,1)).clamp_min(1e-5);ds=train['delta'].std(0).clamp_min(1e-5);stats=(mean,std,mm,ms,ds)
    actual=val['delta'].cpu().numpy();scale=ds.cpu().numpy();current=macro_numpy(val['observations'][:,-1].cpu().numpy());previous=macro_numpy(val['observations'][:,-2].cpu().numpy());hs=np.array([1,4,8,16])
    baselines={name:score(pred,actual,vm,scale,current) for name,pred in [('persistence',np.zeros_like(actual)),('linear',(current-previous)[:,None,:]*hs[None,:,None]),('train_mean_change',np.broadcast_to(group_mean(train['delta'].cpu().numpy(),tm),actual.shape))]}
    reference=PhysicalForecast(30,device=device);count=sum(v.numel() for v in reference.parameters());del reference
    profiles=['keep_static_no_geom','drop_static_no_geom'] if args.phase=='static' else ['keep_static_geom','drop_static_geom']
    results=[];start_time=time.time()
    for kind in ('ghost','mlp'):
        hashes=[]
        for profile in profiles:
            torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
            model=RepresentationProbe(kind,profile,device,count);initial=weights_digest(model);hashes.append(initial)
            optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);rng=np.random.default_rng(args.seed);losses=[]
            model.train()
            for _ in range(p['updates']):
                ix=rng.integers(0,len(train['observations']),size=p['batch_size']);hi=rng.integers(0,4,size=p['batch_size']);horizon=torch.tensor(hs[hi],device=device)
                pred=model(train['observations'][ix],train['momentum'][ix],train['actions'][ix],horizon,*stats[:4]);target=train['delta'][ix,hi]/ds[hi]
                loss=(pred[:,C]-target[:,C]).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            model.eval();pred=predict(model,val,stats,device);metrics=score(pred,actual,vm,scale,current)
            path=out/(kind+'_'+profile+'.pt');torch.save(dict(model=model.state_dict(),stats=[v.cpu() for v in stats],kind=kind,profile=profile,matched_parameters=count,protocol_sha256=digest_file(protocol_path)),path)
            ck=torch.load(path,map_location=device,weights_only=True);fresh=RepresentationProbe(kind,profile,device,count);fresh.load_state_dict(ck['model'],strict=True);fresh.eval();restored=predict(fresh,val,[v.to(device) for v in ck['stats']],device)
            assert np.allclose(pred,restored,atol=1e-7,rtol=1e-5)
            entry=dict(kind=kind,profile=profile,initial_weights_sha256=initial,parameters=sum(v.numel() for v in model.parameters()),last50_train_loss=float(np.mean(losses[-50:])),validation=metrics,action_effect=action_effect(pred,actual,vm,scale),checkpoint_sha256=digest_file(path),reload_verified=True)
            results.append(entry);write_json(out/'progress.json',dict(seed=args.seed,phase=args.phase,results=results,baselines=baselines))
            print(json.dumps(dict(seed=args.seed,kind=kind,profile=profile,mean_horizon_mse=float(np.mean([r['standardized_dynamic_mse'] for r in metrics])),elapsed_seconds=time.time()-start_time)),flush=True)
            del model,fresh,optimizer
        assert len(set(hashes))==1
    report=dict(seed=args.seed,phase=args.phase,protocol_sha256=digest_file(protocol_path),runner_sha256=digest_file(Path(__file__)),model_sha256=digest_file(ROOT/'src/rna2d/physics/representation_v2.py'),baselines=baselines,results=results,new_test_evaluated=False,elapsed_seconds=time.time()-start_time)
    write_json(out/'results.json',report);write_json(ROOT/'data/manifests'/f'representation_v2_{args.phase}_seed{args.seed}.json',report)


if __name__=='__main__':main()
