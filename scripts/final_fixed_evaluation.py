"""One final evaluation after every declared experiment finishes; no tuning."""
import json
import statistics
import time
import torch
from torch.utils.data import DataLoader
from run_controlled_experiment import ROOT,make_model,write_json,digest_file
from rna2d.training import Records,collate,evaluate


def main():
    out=ROOT/'runs/final_evaluation_v1'
    if (out/'summary.json').exists():raise ValueError('Final evaluation already frozen; do not repeatedly select on test')
    declared=ROOT/'data/manifests/final_evaluation_runs_v1.json'
    plan=json.loads(declared.read_text());protocol_path=ROOT/'data/manifests/experiment_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text());data=ROOT/protocol['dataset']
    validation=json.loads((data/'validation.json').read_text())
    for name in plan['runs']:
        summary=json.loads((ROOT/'runs'/name/'summary.json').read_text())
        assert len(summary['seeds'])==3 and summary['protocol_sha256']==digest_file(protocol_path)
        assert not summary['test_evaluated']
        for seed in protocol['seeds']:assert (ROOT/'runs'/name/str(seed)/'best.pt').exists()
    assert digest_file(data/'test.jsonl')==validation['files']['test']
    ds=Records(data/'test.jsonl');loader=DataLoader(ds,batch_size=8,collate_fn=collate)
    rows={r['id']:r for r in ds.rows};device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=True);results={}
    for name in plan['runs']:
        seeds=[]
        for seed in protocol['seeds']:
            path=ROOT/'runs'/name/str(seed)/'best.pt';ck=torch.load(path,map_location='cpu',weights_only=True);c=ck['config']
            assert c['data_sha256']=={s:validation['files'][s] for s in ('train','validation')}
            model=make_model(c['variant'],c['hidden'],plan['final_steps']).to(device);model.load_state_dict(ck['model']);model.eval()
            warm=collate([ds.rows[0]])[0].to(device)
            with torch.no_grad():model(warm);model(warm)
            if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();metrics,details=evaluate(model,loader,device,plan['threshold'])
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            group={};contexts={}
            for r in details:
                original=rows[r['id']];group.setdefault(original['split_group_id'],[]).append(r['f1']);contexts.setdefault(original['context'],[]).append(r['f1'])
            row=dict(seed=seed,selected_epoch=ck['epoch'],checkpoint_sha256=digest_file(path),test=metrics,
                     group_macro_f1=statistics.mean(statistics.mean(v) for v in group.values()),
                     contexts={k:dict(samples=len(v),macro_f1=statistics.mean(v)) for k,v in contexts.items()},
                     inference_seconds_per_RNA=elapsed/len(ds),peak_inference_gpu_MiB=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
                     parameters=c['parameters'])
            seeds.append(row);directory=out/name;directory.mkdir(exist_ok=True)
            (directory/(str(seed)+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in details),encoding='utf-8',newline='\n')
            del model
            if device.type=='cuda':torch.cuda.empty_cache()
        results[name]=dict(seeds=seeds,test_macro_f1_mean=statistics.mean(r['test']['macro_f1'] for r in seeds),
                          test_macro_f1_stdev=statistics.stdev(r['test']['macro_f1'] for r in seeds),
                          test_group_macro_f1_mean=statistics.mean(r['group_macro_f1'] for r in seeds),
                          mean_inference_seconds_per_RNA=statistics.mean(r['inference_seconds_per_RNA'] for r in seeds))
    report=dict(results=results,test_samples=len(ds),test_groups=len({r['split_group_id'] for r in ds.rows}),
                protocol_sha256=digest_file(protocol_path),declared_runs_sha256=digest_file(declared),test_data_sha256=digest_file(data/'test.jsonl'),
                test_evaluated_once=True,selection_policy='All predeclared models/seeds reported; no best-test-model or best-test-step selection',
                limitations=['8 samples / 4 declared groups, masks and complex contexts: exploratory only.',
                             'Validation selected epochs; no hyperparameters changed after this test evaluation.',
                             'Epoch/sample budgets match; static capacity controls approximate parameter count, not exact training FLOPs.',
                             'Seed standard deviation is not a confidence interval over RNA families.'])
    write_json(out/'summary.json',report);write_json(ROOT/'data/manifests/final_evaluation_v1.json',report)
    print(json.dumps({name:dict(mean=r['test_macro_f1_mean'],stdev=r['test_macro_f1_stdev']) for name,r in results.items()},indent=2))


if __name__=='__main__':main()
