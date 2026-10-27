"""Validate algorithmic feature-transition prediction against persistence."""
import argparse
import json
import torch
from torch.utils.data import DataLoader
from run_controlled_experiment import ROOT,make_model,write_json
from rna2d.training import Records,collate


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();directory=ROOT/a.run
    protocol=json.loads((ROOT/'data/manifests/experiment_protocol_v1.json').read_text());results=[]
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');torch.set_num_threads(2)
    loader=DataLoader(Records(ROOT/protocol['dataset']/'validation.jsonl'),batch_size=8,collate_fn=collate)
    for seed in protocol['seeds']:
        ck=torch.load(directory/str(seed)/'best.pt',map_location=device,weights_only=True);c=ck['config']
        m=make_model(c['variant'],c['hidden'],4).to(device);m.load_state_dict(ck['model']);m.eval();pred=persist=weight=0
        with torch.no_grad():
            for tokens,_,_,_ in loader:
                _,aux=m.forward_with_aux(tokens.to(device));w=int(tokens.ne(4).sum())*c['hidden']
                pred+=float(aux['prediction_mse'])*w;persist+=float(aux['persistence_mse'])*w;weight+=w
        results.append(dict(seed=seed,prediction_mse=pred/weight,persistence_mse=persist/weight,beats_persistence=pred<persist))
    report=dict(results=results,test_evaluated=False,physical_world_model_validated=False,
                target='Next algorithmic pair-update input feature under the fixed model; stop-gradient target, not experimental folding intermediates',
                note='Persistence is a mandatory reference; outperforming it would still not prove physical RNA dynamics.')
    write_json(directory/'prediction_validation.json',report);write_json(ROOT/'data/manifests'/(directory.name+'_prediction.json'),report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
