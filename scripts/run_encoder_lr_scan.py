"""Bounded train-only learning-rate scan. Never loads validation or test files."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
from run_controlled_experiment import balanced_loss
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader
from rna2d.training import Records,collate,evaluate,seed_all
from rna2d.sequence_encoders import SequencePairModel
from rna2d.evidence import pairing_evidence_issues


def learning_rate(base,step,warmup_steps):
    if step<1:raise ValueError('Optimizer step is one-based')
    return base*min(1.,step/warmup_steps) if warmup_steps else base


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--device',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    pp=ROOT/'data/manifests/encoder_lr_scan_protocol_v1.json';p=json.loads(pp.read_text());device=torch.device(args.device)
    out=ROOT/'runs'/f'encoder_lr_scan_v1_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise ValueError('Completed scan seed frozen')
    train_path=ROOT/p['training_file'];assert digest_file(train_path)==p['training_sha256'];assert digest_file(ROOT/'src/rna2d/sequence_encoders.py')==p['model_source_sha256']
    ds=Records(train_path);assert all(not pairing_evidence_issues(r) for r in ds.rows)
    evaluation=DataLoader(ds,batch_size=p['batch_size'],shuffle=False,collate_fn=collate)
    cnn_path=ROOT/'runs'/f'encoder_comparison_v1_seed{args.seed}'/'cnn.pt';cnn_ck=torch.load(cnn_path,map_location=device,weights_only=True);cnn=SequencePairModel('cnn',p['widths']['cnn']).to(device);cnn.load_state_dict(cnn_ck['model']);cnn_metrics,_=evaluate(cnn,evaluation,device);del cnn
    results=[];total_steps=math.ceil(len(ds)/p['batch_size'])*p['epochs']
    for kind in p['encoders']:
        for config in p['variants']:
            seed_all(args.seed);model=SequencePairModel(kind,p['widths'][kind]).to(device)
            loader=DataLoader(ds,batch_size=p['batch_size'],shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(args.seed))
            warmup_steps=math.ceil(total_steps*p['warmup_fraction']) if config['warmup'] else 0
            optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=p['weight_decay']);history=[];step=0;failure=None;start=time.perf_counter()
            try:
                for epoch in range(1,p['epochs']+1):
                    model.train();losses=[]
                    for tokens,labels,mask,_ in loader:
                        step+=1;lr=learning_rate(config['lr'],step,warmup_steps)
                        for group in optimizer.param_groups:group['lr']=lr
                        tokens,labels,mask=tokens.to(device),labels.to(device),mask.to(device);optimizer.zero_grad(set_to_none=True)
                        loss=balanced_loss(model(tokens),labels,mask)
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
                    if epoch==1 or epoch%5==0:
                        metrics,_=evaluate(model,evaluation,device)
                        history.append(dict(epoch=epoch,updates=step,lr=lr,training_loss=statistics.mean(losses),train_macro_f1=metrics['macro_f1']))
                        print(json.dumps(dict(seed=args.seed,encoder=kind,variant=config['name'],epoch=epoch,train_f1=metrics['macro_f1'])),flush=True)
                metrics,details=evaluate(model,evaluation,device)
            except (FloatingPointError,RuntimeError) as exc:
                failure=str(exc);metrics=None;details=[]
            entry=dict(encoder=kind,variant=config['name'],lr=config['lr'],warmup_steps=warmup_steps,seed=args.seed,status='completed' if failure is None else 'failed',failure=failure,updates=step,history=history,training=metrics,details=details,elapsed_seconds=time.perf_counter()-start,validation_loaded=False,test_loaded=False)
            if failure is None:
                path=out/(kind+'_'+config['name']+'.pt');torch.save(dict(model=model.state_dict(),encoder=kind,width=p['widths'][kind],variant=config,seed=args.seed,epoch=p['epochs'],protocol_sha256=digest_file(pp)),path)
                fresh=SequencePairModel(kind,p['widths'][kind]).to(device);fresh.load_state_dict(torch.load(path,map_location=device,weights_only=True)['model']);reloaded,rd=evaluate(fresh,evaluation,device)
                assert abs(reloaded['macro_f1']-metrics['macro_f1'])<1e-9 and all(a['predicted_pairs']==b['predicted_pairs'] for a,b in zip(details,rd))
                entry.update(checkpoint_sha256=digest_file(path),reload_verified=True)
                del fresh
            results.append(entry);write_json(out/'progress.json',dict(seed=args.seed,results=results,cnn_training=cnn_metrics))
            del model,optimizer;torch.cuda.empty_cache()
    report=dict(seed=args.seed,protocol_sha256=digest_file(pp),runner_sha256=digest_file(Path(__file__)),model_source_sha256=p['model_source_sha256'],cnn_training=cnn_metrics,cnn_checkpoint_sha256=digest_file(cnn_path),results=results,validation_loaded=False,test_loaded=False)
    write_json(out/'results.json',report);write_json(ROOT/'data/manifests'/f'encoder_lr_scan_seed{args.seed}_v1.json',report)


if __name__=='__main__':main()
