"""Fixed-budget multi-seed experiments; test labels are sealed in this runner."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from torch.utils.data import DataLoader
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
from rna2d.training import Records,collate,evaluate,seed_all
from rna2d.model import PairBaseline
from rna2d.evidence import pairing_evidence_issues


def make_model(mode,hidden,steps):
    if mode=='static':return PairBaseline(hidden)
    if mode in ('iterative','rwkv','rwkv_reset'):
        from rna2d.iterative import PairDynamics
        return PairDynamics(hidden,steps,mode)
    raise ValueError('Variant not implemented/validated yet: '+mode)


def balanced_loss(logits,labels,mask):
    losses=[]
    for b in range(len(logits)):
        pos,neg=mask[b]&labels[b].bool(),mask[b]&~labels[b].bool();terms=[]
        if pos.any():terms.append(torch.nn.functional.softplus(-logits[b][pos]).mean())
        if neg.any():terms.append(torch.nn.functional.softplus(logits[b][neg]).mean())
        if terms:losses.append(torch.stack(terms).mean())
    if not losses:raise ValueError('No usable supervision')
    return torch.stack(losses).mean()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',default='static');parser.add_argument('--hidden',type=int)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    protocol_path=ROOT/'data/manifests/experiment_protocol_v1.json';protocol=json.loads(protocol_path.read_text())
    directory=ROOT/protocol['dataset'];validation=json.loads((directory/'validation.json').read_text())
    assert validation['passed']
    paths={s:directory/(s+'.jsonl') for s in ('train','validation')}
    for s,p in paths.items():assert digest_file(p)==validation['files'][s]
    datasets={s:Records(p) for s,p in paths.items()}
    for dataset in datasets.values():
        for r in dataset.rows:assert not pairing_evidence_issues(r)
    args.output.mkdir(parents=True,exist_ok=False)
    hidden=args.hidden or protocol['hidden'];device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    results=[]
    for seed in protocol['seeds']:
        seed_all(seed);model=make_model(args.variant,hidden,protocol['ordinary_steps']).to(device)
        loaders={s:DataLoader(ds,batch_size=protocol['batch_size'],shuffle=s=='train',collate_fn=collate,
                             generator=torch.Generator().manual_seed(seed)) for s,ds in datasets.items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=0.01)
        output=args.output/str(seed);output.mkdir()
        config=dict(variant=args.variant,hidden=hidden,steps=protocol['ordinary_steps'],seed=seed,protocol=protocol,
                    protocol_sha256=digest_file(protocol_path),data_sha256={s:digest_file(p) for s,p in paths.items()},
                    parameters=sum(p.numel() for p in model.parameters()),device=str(device),torch_version=str(torch.__version__),
                    scope='Exploratory contextual-chain canonical-pair prediction; no test evaluation or 1D pretraining')
        write_json(output/'config.json',config);best=-1;start=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1,protocol['epochs']+1):
            model.train();losses=[]
            for tokens,labels,mask,_ in loaders['train']:
                tokens,labels,mask=tokens.to(device),labels.to(device),mask.to(device);optimizer.zero_grad(set_to_none=True)
                if hasattr(model,'forward_with_aux'):logits,aux=model.forward_with_aux(tokens)
                else:logits,aux=model(tokens),{}
                loss=balanced_loss(logits,labels,mask)
                if 'prediction_mse' in aux:loss=loss+protocol['state_prediction_aux_weight']*aux['prediction_mse']
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'],error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            metrics,_=evaluate(model,loaders['validation'],device)
            with (output/'history.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(epoch=epoch,loss=statistics.mean(losses),**metrics))+'\n')
            if metrics['macro_f1']>best:
                best=metrics['macro_f1'];torch.save(dict(model=model.state_dict(),config=config,epoch=epoch,validation=metrics),output/'best.pt')
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        checkpoint=torch.load(output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(checkpoint['model'])
        metrics,details=evaluate(model,loaders['validation'],device)
        context={r['id']:r['context'] for r in datasets['validation'].rows}
        metrics['contexts']={c:dict(samples=len(rs),macro_f1=statistics.mean(r['f1'] for r in rs)) for c in set(context.values()) if (rs:=[r for r in details if context[r['id']]==c])}
        row=dict(seed=seed,epoch=checkpoint['epoch'],validation=metrics,training_seconds=elapsed,parameters=config['parameters'],
                 peak_gpu_MiB=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None)
        write_json(output/'result.json',row);results.append(row);print(args.variant,seed,metrics['macro_f1'],flush=True)
    summary=dict(variant=args.variant,hidden=hidden,dataset=protocol['dataset'],seeds=results,
                 validation_macro_f1_mean=statistics.mean(r['validation']['macro_f1'] for r in results),
                 validation_macro_f1_stdev=statistics.stdev(r['validation']['macro_f1'] for r in results),
                 test_evaluated=False,model_code_sha256=digest_file(ROOT/'src/rna2d/model.py'),
                 protocol_sha256=digest_file(protocol_path),scope='Exploratory; small grouped data; validation used for checkpoint selection, no significance or generalization claim')
    write_json(args.output/'summary.json',summary)
    write_json(ROOT/'data/manifests'/(args.output.name+'.json'),summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='seeds'},indent=2))


if __name__=='__main__':main()
