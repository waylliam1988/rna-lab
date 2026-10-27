"""Same-GPU sequential latency/memory benchmark after prediction runs finish."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
from run_controlled_experiment import balanced_loss
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.sequence_encoders import SequencePairModel
from rna2d.training import collate
from rna2d.data import greedy_decode


def timed(fn,repeats,device):
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize(device);start=time.perf_counter();fn();torch.cuda.synchronize(device);values.append((time.perf_counter()-start)*1000)
    return dict(median_ms=statistics.median(values),mean_ms=statistics.mean(values),min_ms=min(values),max_ms=max(values))


def one(kind,n,mode,checkpoint,p,device):
    model=SequencePairModel(kind,p['widths'][kind],p['hidden'],p['depth']).to(device);model.load_state_dict(checkpoint['model'],strict=True)
    seq=('ACGU'*((n+3)//4))[:n];tokens,labels,mask,_=collate([dict(sequence=seq,pairs=[(0,n-1)],observed_mask=[True]*n)])
    tokens,labels,mask=tokens.to(device),labels.to(device),mask.to(device)
    if mode=='inference':
        model.eval()
        @torch.inference_mode()
        def step():model(tokens)
        @torch.inference_mode()
        def decode():greedy_decode(model(tokens).sigmoid().cpu().tolist()[0],seq,.5)
        for _ in range(p['speed_benchmark']['warmup']):step()
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        timing=timed(step,p['speed_benchmark']['repeats'],device)
        memory=dict(peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/1024**2,peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/1024**2)
        for _ in range(p['speed_benchmark']['warmup']):decode()
        end_to_end=timed(decode,p['speed_benchmark']['repeats'],device)
        row=dict(encoder=kind,length=n,batch=1,mode=mode,network=timing,network_plus_cpu_decode=end_to_end,**memory)
    else:
        model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=p['lr'],weight_decay=p['weight_decay'])
        def step():
            optimizer.zero_grad(set_to_none=True);loss=balanced_loss(model(tokens),labels,mask);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        for _ in range(p['speed_benchmark']['warmup']):step()
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        timing=timed(step,p['speed_benchmark']['repeats'],device)
        row=dict(encoder=kind,length=n,batch=1,mode=mode,step=timing,peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/1024**2,peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/1024**2)
    return row


def main():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;device=torch.device('cuda:0')
    protocol_path=ROOT/'data/manifests/encoder_comparison_protocol_v1.json';p=json.loads(protocol_path.read_text());output=ROOT/'data/manifests/encoder_resource_benchmark_v1.json'
    if output.exists():raise ValueError('Benchmark frozen')
    # Explicit barrier: no competing training jobs from this comparison.
    assert all((ROOT/'data/manifests'/f'encoder_comparison_seed{s}_v1.json').exists() for s in p['seeds'])
    rows=[];weights={}
    for kind in p['encoders']:
        path=ROOT/'runs/encoder_comparison_v1_seed42'/(kind+'.pt');weights[kind]=digest_file(path);checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        for n in p['speed_benchmark']['lengths']:
            for mode in ('inference','training'):
                gc.collect();torch.cuda.empty_cache()
                try:row=one(kind,n,mode,checkpoint,p,device);row['status']='ok'
                except torch.cuda.OutOfMemoryError:row=dict(encoder=kind,length=n,batch=1,mode=mode,status='OOM')
                rows.append(row);print(json.dumps(row),flush=True)
                gc.collect();torch.cuda.empty_cache()
    report=dict(device=torch.cuda.get_device_name(device),torch_version=str(torch.__version__),precision=p['precision'],protocol_sha256=digest_file(protocol_path),weights_sha256=weights,benchmark_source_sha256=digest_file(Path(__file__)),cases=rows,limits=['PyTorch allocation excludes CUDA driver/context memory.','Wall timings include dispatch and synchronization; median of20 warmed repeats.','Synthetic shapes for performance only; accuracy measured separately.','Same GPU sequential run, but external unrelated system activity is not controlled.'])
    write_json(output,report)


if __name__=='__main__':main()
