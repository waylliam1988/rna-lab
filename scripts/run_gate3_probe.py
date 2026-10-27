"""Frozen independently trained controls and causal perturbation diagnostics."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.causal_probe import ResidualProbe,VARIANTS
from rna2d.physics.forecast import PhysicalForecast
from rna2d.physics.diagnostics import macro_numpy,group_mean,skill,DYNAMIC_CHANNELS as C


def predict(model,data,stats,device,mode='ordered'):
    n=len(data['observations']);result=np.zeros((n,4,5))
    obs=data['observations'];mom=data['momentum'];actions=data['actions'].clone()
    if mode=='action_drop':actions.zero_()
    if mode=='action_shuffle':
        # Move scalar action within each identical-state counterfactual triplet.
        idx=torch.arange(n,device=device).reshape(-1,3)[:,[1,2,0]].reshape(-1)
        actions=actions[idx]
    if mode=='momentum_drop':mom=torch.zeros_like(mom)
    with torch.no_grad():
        for hidx,h in enumerate((1,4,8,16)):
            for start in range(0,n,32):
                stop=min(n,start+32);hs=torch.full((stop-start,),h,device=device)
                pred=model(obs[start:stop],mom[start:stop],actions[start:stop],hs,*stats[:4],shuffle=mode=='history_shuffle',reset=mode=='reset')
                result[start:stop,hidx]=(pred*stats[4][hidx]).cpu().numpy()
    result[:,:,[1,2]]=0
    return result


def score(pred,actual,metadata,delta_scale,current):
    result=[];groups=sorted({m['split_group_id'] for m in metadata})
    for hi,h in enumerate((1,4,8,16)):
        err=(pred[:,hi]-actual[:,hi])**2;persist=actual[:,hi]**2
        mse=group_mean(err,metadata);std_err=group_mean(err/delta_scale[hi]**2,metadata)
        baseline=group_mean(persist/delta_scale[hi]**2,metadata)
        threshold=32*np.finfo(np.float32).eps*np.maximum(1,np.abs(current))
        valid=np.abs(actual[:,hi])>threshold
        signs=[]
        for c in C:
            scores=[]
            for g in groups:
                ix=[i for i,m in enumerate(metadata) if m['split_group_id']==g and valid[i,c]]
                if ix:scores.append(float((np.sign(pred[ix,hi,c])==np.sign(actual[ix,hi,c])).mean()))
            signs.append(float(np.mean(scores)) if scores else None)
        pergroup=[]
        for g in groups:
            ix=[i for i,m in enumerate(metadata) if m['split_group_id']==g]
            a=float((err[ix]/delta_scale[hi]**2)[:,C].mean());b=float((persist[ix]/delta_scale[hi]**2)[:,C].mean())
            pergroup.append(dict(group=g,standardized_dynamic_mse=a,skill=skill(a,b)))
        result.append(dict(horizon=h,mse_by_channel=mse.tolist(),standardized_dynamic_mse=float(std_err[C].mean()),skill_vs_persistence=skill(float(std_err[C].mean()),float(baseline[C].mean())),sign_accuracy_by_dynamic_channel=signs,per_group=pergroup))
    return result


def action_effect(pred,actual,metadata,scale):
    assert len(pred)%3==0
    assert all([r['amplitude'] for r in metadata[i:i+3]]==[-.4,0.,.4] for i in range(0,len(metadata),3))
    truth=actual[2::3]-actual[0::3];effect=pred[2::3]-pred[0::3]
    meta=metadata[0::3];results=[]
    for hi,h in enumerate((1,4,8,16)):
        e=group_mean((effect[:,hi]-truth[:,hi])**2/scale[hi]**2,meta)[C].mean()
        b=group_mean(truth[:,hi]**2/scale[hi]**2,meta)[C].mean()
        results.append(dict(horizon=h,standardized_effect_mse=float(e),skill_vs_zero_action_effect=skill(float(e),float(b))))
    return results


def load_data(split,device):
    root=ROOT/'data/simulated/gate3_v1'
    arrays=np.load(root/(split+'.npz'))
    return {k:torch.tensor(arrays[k],device=device) for k in arrays.files},json.loads((root/(split+'_metadata.json')).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43,44],required=True);p.add_argument('--device',required=True);args=p.parse_args()
    torch.set_num_threads(1);device=args.device
    protocol_path=ROOT/'data/manifests/gate3_protocol_v1.json';protocol=json.loads(protocol_path.read_text());fingerprints=json.loads((ROOT/'data/manifests/gate3_data_v1.json').read_text())
    out=ROOT/'runs'/f'gate3_v1_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise ValueError('Completed seed already frozen')
    for split in ('train','validation'):
        assert digest_file(ROOT/'data/simulated/gate3_v1'/(split+'.npz'))==fingerprints['splits'][split]['npz_sha256']
    train,tm=load_data('train',device);val,vm=load_data('validation',device)
    mean=train['observations'].mean((0,1));std=train['observations'].std((0,1)).clamp_min(.01)
    mm=train['momentum'].mean((0,1));ms=train['momentum'].std((0,1)).clamp_min(1e-5)
    ds=train['delta'].std(0).clamp_min(1e-5);stats=(mean,std,mm,ms,ds)
    delta=val['delta'].cpu().numpy();scale=ds.cpu().numpy();current=macro_numpy(val['observations'][:,-1].cpu().numpy())
    previous=macro_numpy(val['observations'][:,-2].cpu().numpy());horizons=np.array([1,4,8,16])
    linear=(current-previous)[:,None,:]*horizons[None,:,None]
    baseline_mean=group_mean(train['delta'].cpu().numpy(),tm)[None].repeat(len(delta),axis=0)
    baselines={name:score(pred,delta,vm,scale,current) for name,pred in [('persistence',np.zeros_like(delta)),('linear',linear),('train_mean_change',baseline_mean)]}
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    prototype=PhysicalForecast(28,device=device)
    shared=copy.deepcopy(prototype.ghost.state_dict());del prototype
    results=[];start_time=time.time()
    for variant in protocol['variants']:
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        kind,momentum,action=VARIANTS[variant];d=19+8*momentum+action
        # Parameter-only reference, constructed before reading any validation score.
        reference=PhysicalForecast(d,device=device);count=sum(p.numel() for p in reference.parameters());del reference
        model=ResidualProbe(variant,device,matched_parameters=count)
        if kind=='ghost':model.network.ghost.load_state_dict(shared,strict=True)
        model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
        rng=np.random.default_rng(args.seed);losses=[];active=None
        for step in range(protocol['updates']):
            ix=rng.integers(0,len(train['observations']),size=protocol['batch_size']);hi=rng.integers(0,4,size=protocol['batch_size'])
            horizon=torch.tensor(horizons[hi],device=device)
            pred=model(train['observations'][ix],train['momentum'][ix],train['actions'][ix],horizon,*stats[:4])
            target=train['delta'][ix,hi]/ds[hi]
            loss=(pred[:,C]-target[:,C]).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if active is None:active=sum(p.numel() for p in model.parameters() if p.grad is not None)
            optimizer.step();losses.append(float(loss.detach()))
        model.eval();ordered=predict(model,val,stats,device)
        metrics={'ordered':score(ordered,delta,vm,scale,current)}
        modes=['history_shuffle','reset'] if kind=='ghost' else []
        if variant=='ghost_full':modes+=['momentum_drop','action_drop','action_shuffle']
        for mode in modes:metrics[mode]=score(predict(model,val,stats,device,mode),delta,vm,scale,current)
        train_pred=predict(model,train,stats,device)
        train_current=macro_numpy(train['observations'][:,-1].cpu().numpy())
        train_score=score(train_pred,train['delta'].cpu().numpy(),tm,scale,train_current)
        checkpoint=out/(variant+'.pt');torch.save(dict(variant=variant,model=model.state_dict(),stats=[s.cpu() for s in stats],matched_parameters=count,seed=args.seed,protocol_sha256=digest_file(protocol_path)),checkpoint)
        # Fresh module and strict checkpoint loading; recompute all ordered predictions.
        ck=torch.load(checkpoint,map_location=device,weights_only=True);reloaded=ResidualProbe(variant,device,matched_parameters=count);reloaded.load_state_dict(ck['model'],strict=True);reloaded.eval()
        replay=predict(reloaded,val,stats,device)
        if not np.allclose(replay,ordered,rtol=1e-5,atol=1e-7):raise ValueError('Reload mismatch')
        entry=dict(variant=variant,parameters=sum(p.numel() for p in model.parameters()),parameters_with_gradient_on_first_update=active,matched_reference_parameters=count,last50_train_loss=float(np.mean(losses[-50:])),training_score=train_score,validation=metrics,action_effect=action_effect(ordered,delta,vm,scale),checkpoint_sha256=digest_file(checkpoint),reload_verified=True)
        results.append(entry);write_json(out/'progress.json',dict(seed=args.seed,results=results,baselines=baselines))
        print(json.dumps(dict(seed=args.seed,variant=variant,skill_h16=metrics['ordered'][-1]['skill_vs_persistence'],elapsed_seconds=time.time()-start_time)),flush=True)
        del model,reloaded,optimizer
    report=dict(seed=args.seed,device=torch.cuda.get_device_name(torch.device(device)),protocol_sha256=digest_file(protocol_path),runner_sha256=digest_file(Path(__file__)),model_sha256=digest_file(ROOT/'src/rna2d/physics/causal_probe.py'),delta_scale=scale.tolist(),baselines=baselines,results=results,elapsed_seconds=time.time()-start_time,new_test_evaluated=False)
    write_json(out/'results.json',report);write_json(ROOT/'data/manifests'/f'gate3_seed{args.seed}_v1.json',report)


if __name__=='__main__':main()
