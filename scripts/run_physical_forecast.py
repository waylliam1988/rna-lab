"""Controlled duration/rollout probe. Never loads an exposed test set."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
import torch
import torch.nn.functional as F
from rna2d.physics.forecast import PhysicalForecast,window_batch
from rna2d.physics.interface import normalized_macro


def score(model, episodes, metadata, mean, scale, history, reset=False):
    predictions=[];targets=[];persistence=[]
    with torch.no_grad():
        for i in range(0,len(episodes),8):
            # All history lengths end at the same target state16, avoiding
            # confusing a later/easier physical state with longer memory.
            raw=episodes[i:i+8,16-history:16]
            nxt=episodes[i:i+8,17-history:17]
            pred=model((raw-mean)/scale,reset_each_step=reset)
            # Evaluate terminal state for all K histories, not average warmup error.
            predictions.append(pred[:,-1]);targets.append(normalized_macro(nxt[:,-1]));persistence.append(normalized_macro(raw[:,-1]))
    pred=torch.cat(predictions);target=torch.cat(targets);persist=torch.cat(persistence)
    err=(pred-target).square().cpu().numpy();base=(persist-target).square().cpu().numpy()
    groups=sorted({r['split_group_id'] for r in metadata})
    macro=[];baseline=[]
    for g in groups:
        idx=[i for i,r in enumerate(metadata) if r['split_group_id']==g]
        macro.append(err[idx].mean(0));baseline.append(base[idx].mean(0))
    result=dict(group_macro_mse_by_channel=np.mean(macro,axis=0).tolist(),persistence_mse_by_channel=np.mean(baseline,axis=0).tolist(),groups=len(groups))
    # Contact channels are constants of the fixed J and excluded from dynamic aggregate.
    result['dynamic_mse']=float(np.mean(np.array(macro)[:,[0,3,4]]))
    result['persistence_dynamic_mse']=float(np.mean(np.array(baseline)[:,[0,3,4]]))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rollout',type=int,choices=[4,8],required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    protocol_path=ROOT/'data/manifests/physical_forecast_protocol_v1.json';p=json.loads(protocol_path.read_text())
    out=ROOT/'runs'/f'physical_forecast_v1_r{args.rollout}';out.mkdir(exist_ok=True,parents=True)
    if (out/'results.json').exists():raise ValueError('Run already completed')
    data=ROOT/'data/experimental/physical_forecast_v1';device='cuda:0'
    for split in ('train','validation'):
        assert digest_file(data/(split+'.npz'))==p['data_fingerprints'][split]['file_sha256']
        assert digest_file(data/(split+'_metadata.json'))==p['data_fingerprints'][split]['metadata_sha256']
    train=torch.tensor(np.load(data/'train.npz')['observations'],device=device)
    val=torch.tensor(np.load(data/'validation.npz')['observations'],device=device)
    metadata=json.loads((data/'validation_metadata.json').read_text())
    mean=train.mean((0,1));scale=train.std((0,1)).clamp_min(.01)
    result=[];start=time.time()
    for seed in p['seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        model=PhysicalForecast(device=device);opt=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        rng=np.random.default_rng(seed);train_losses=[]
        for step in range(1,max(p['training_update_checkpoints'])+1):
            model.train();indices=rng.integers(0,len(train),size=p['batch_size']);starts=rng.integers(0,33-args.rollout,size=p['batch_size'])
            raw,nxt=window_batch(train,indices,starts,args.rollout)
            pred=model((raw-mean)/scale);target=normalized_macro(nxt)
            loss=F.smooth_l1_loss(pred,target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True);opt.step()
            train_losses.append(float(loss.detach()))
            if step in p['training_update_checkpoints']:
                model.eval()
                evals={str(h):{'carry':score(model,val,metadata,mean,scale,h),'reset':score(model,val,metadata,mean,scale,h,True)} for h in p['evaluation_history_lengths']}
                entry=dict(seed=seed,updates=step,training_rollout=args.rollout,parameter_count=sum(v.numel() for v in model.parameters()),last_50_train_loss=float(np.mean(train_losses[-50:])),evaluation=evals)
                result.append(entry)
                torch.save(dict(model=model.state_dict(),input_mean=mean.cpu(),input_scale=scale.cpu(),seed=seed,updates=step,rollout=args.rollout,protocol_sha256=digest_file(protocol_path)),out/f'seed{seed}_step{step}.pt')
                write_json(out/'progress.json',dict(results=result,elapsed_seconds=time.time()-start))
                print(json.dumps(dict(seed=seed,updates=step,rollout=args.rollout,validation_dynamic_mse=evals[str(args.rollout)]['carry']['dynamic_mse'],elapsed_seconds=time.time()-start)),flush=True)
    report=dict(protocol_sha256=digest_file(protocol_path),model_source_sha256=digest_file(ROOT/'src/rna2d/physics/forecast.py'),runner_source_sha256=digest_file(Path(__file__)),device=torch.cuda.get_device_name(0),elapsed_seconds=time.time()-start,results=result,new_test_evaluated=False)
    write_json(out/'results.json',report);write_json(ROOT/'data/manifests'/f'physical_forecast_results_r{args.rollout}_v1.json',report)


if __name__=='__main__':main()
