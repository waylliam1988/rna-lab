"""Fixed-budget sequence->2D comparison, last epoch only, no old test access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
import sys
import time
import statistics
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
from run_controlled_experiment import balanced_loss
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader
from rna2d.training import Records,collate,evaluate,seed_all
from rna2d.evidence import pairing_evidence_issues
from rna2d.sequence_encoders import SequencePairModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[42,43,44],required=True);parser.add_argument('--device',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    path=ROOT/'data/manifests/encoder_comparison_protocol_v1.json';p=json.loads(path.read_text());device=torch.device(args.device)
    out=ROOT/'runs'/f'encoder_comparison_v1_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise ValueError('Completed seed frozen')
    datasets={s:Records(ROOT/p['dataset']/(s+'.jsonl')) for s in ('train','validation')}
    for s,ds in datasets.items():
        assert digest_file(ROOT/p['dataset']/(s+'.jsonl'))==p['data_sha256'][s]
        assert all(not pairing_evidence_issues(r) for r in ds.rows)
    groups={r['id']:r['split_group_id'] for r in datasets['validation'].rows};results=[]
    for kind in p['encoders']:
        seed_all(args.seed);model=SequencePairModel(kind,p['widths'][kind],p['hidden'],p['depth']).to(device)
        loaders={s:DataLoader(ds,batch_size=p['batch_size'],shuffle=s=='train',collate_fn=collate,generator=torch.Generator().manual_seed(args.seed)) for s,ds in datasets.items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay']);history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device);updates=0
        for epoch in range(1,p['epochs']+1):
            model.train();losses=[]
            for tokens,labels,mask,_ in loaders['train']:
                tokens,labels,mask=tokens.to(device),labels.to(device),mask.to(device);optimizer.zero_grad(set_to_none=True)
                logits=model(tokens);loss=balanced_loss(logits,labels,mask)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1;losses.append(float(loss.detach()))
            if epoch%5==0 or epoch==1:
                metrics,_=evaluate(model,loaders['validation'],device)
                history.append(dict(epoch=epoch,training_loss=statistics.mean(losses),validation_macro_f1=metrics['macro_f1']))
                print(json.dumps(dict(seed=args.seed,encoder=kind,epoch=epoch,macro_f1=metrics['macro_f1'])),flush=True)
        torch.cuda.synchronize(device);elapsed=time.perf_counter()-start;training_peak=torch.cuda.max_memory_allocated(device)/1024**2
        metrics,details=evaluate(model,loaders['validation'],device)
        by_group={}
        for r in details:by_group.setdefault(groups[r['id']],[]).append(r['f1'])
        group_metrics={g:statistics.mean(v) for g,v in by_group.items()};metrics['group_macro_f1']=statistics.mean(group_metrics.values());metrics['per_group_f1']=group_metrics
        checkpoint=out/(kind+'.pt');torch.save(dict(model=model.state_dict(),kind=kind,width=p['widths'][kind],hidden=p['hidden'],depth=p['depth'],seed=args.seed,epoch=p['epochs'],protocol_sha256=digest_file(path)),checkpoint)
        fresh=SequencePairModel(kind,p['widths'][kind],p['hidden'],p['depth']).to(device);fresh.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model']);reloaded,rd=evaluate(fresh,loaders['validation'],device)
        assert abs(reloaded['macro_f1']-metrics['macro_f1'])<1e-9
        assert all(a['predicted_pairs']==b['predicted_pairs'] for a,b in zip(details,rd))
        row=dict(encoder=kind,seed=args.seed,updates=updates,parameters=sum(v.numel() for v in model.parameters()),validation=metrics,details=details,history=history,training_including_logged_validation_seconds=elapsed,training_peak_allocated_MiB=training_peak,checkpoint_sha256=digest_file(checkpoint),reload_verified=True)
        results.append(row);write_json(out/'progress.json',dict(seed=args.seed,results=results));del model,fresh,optimizer;torch.cuda.empty_cache()
    report=dict(seed=args.seed,device=torch.cuda.get_device_name(device),protocol_sha256=digest_file(path),model_code_sha256=digest_file(ROOT/'src/rna2d/sequence_encoders.py'),runner_sha256=digest_file(Path(__file__)),results=results,test_evaluated=False)
    write_json(out/'results.json',report);write_json(ROOT/'data/manifests'/f'encoder_comparison_seed{args.seed}_v1.json',report)


if __name__=='__main__':main()
