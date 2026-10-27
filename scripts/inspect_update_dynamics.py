"""Fixed validation-only step diagnostics; no best-step oracle or test access."""
import argparse
import json
import statistics
import torch
from torch.utils.data import DataLoader
from run_controlled_experiment import ROOT,make_model,write_json
from rna2d.training import Records,collate,evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    directory=ROOT/args.run;protocol=json.loads((ROOT/'data/manifests/experiment_protocol_v1.json').read_text())
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2);results=[]
    loader=DataLoader(Records(ROOT/protocol['dataset']/'validation.jsonl'),batch_size=8,collate_fn=collate)
    for seed in protocol['seeds']:
        ck=torch.load(directory/str(seed)/'best.pt',map_location=device,weights_only=True);c=ck['config']
        model=make_model(c['variant'],c['hidden'],c['steps']).to(device);model.load_state_dict(ck['model']);model.eval()
        for steps in protocol['validation_step_diagnostics']:
            model.steps=steps;metrics,_=evaluate(model,loader,device);trace=[]
            with torch.no_grad():
                for x,_,_,_ in loader:model(x.to(device));trace.append(model.last_trace)
            results.append(dict(seed=seed,steps=steps,validation_macro_f1=metrics['macro_f1'],
                mean_update_by_step=[statistics.mean(t[i]['mean_abs_update'] for t in trace) for i in range(steps)],
                positive_update_fraction=[statistics.mean(t[i]['positive_fraction'] for t in trace) for i in range(steps)],
                negative_update_fraction=[statistics.mean(t[i]['negative_fraction'] for t in trace) for i in range(steps)]))
    report=dict(results=results,selected_test_steps=4,test_evaluated=False,
                note='Validation diagnostics only; final step=4 frozen before training, no per-target or best-step selection.')
    write_json(directory/'dynamics.json',report);write_json(ROOT/'data/manifests'/(directory.name+'_dynamics.json'),report)


if __name__=='__main__':main()
