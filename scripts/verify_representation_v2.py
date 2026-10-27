"""Reload all weights and saved normalization for one representation phase."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
from run_representation_v2 import get_data,predict
from run_gate3_probe import score
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.representation_v2 import RepresentationProbe
from rna2d.physics.diagnostics import macro_numpy


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['static','geometry'],required=True);args=p.parse_args()
    torch.set_num_threads(1);device='cuda:0';val,meta=get_data('validation',args.phase,device);actual=val['delta'].cpu().numpy();current=macro_numpy(val['observations'][:,-1].cpu().numpy());checks=[]
    for seed in (42,43,44):
        report=json.loads((ROOT/'data/manifests'/f'representation_v2_{args.phase}_seed{seed}.json').read_text())
        for row in report['results']:
            path=ROOT/'runs'/f'representation_v2_{args.phase}_seed{seed}'/(row['kind']+'_'+row['profile']+'.pt')
            assert digest_file(path)==row['checkpoint_sha256'];ck=torch.load(path,map_location=device,weights_only=True)
            model=RepresentationProbe(ck['kind'],ck['profile'],device,ck['matched_parameters']);model.load_state_dict(ck['model'],strict=True);model.eval();stats=[v.to(device) for v in ck['stats']]
            pred=predict(model,val,stats,device);metrics=score(pred,actual,meta,stats[4].cpu().numpy(),current)
            assert all(np.allclose(a['mse_by_channel'],b['mse_by_channel'],rtol=2e-5,atol=1e-8) for a,b in zip(metrics,row['validation']))
            checks.append(dict(seed=seed,kind=row['kind'],profile=row['profile'],sha256=row['checkpoint_sha256']))
        print(json.dumps(dict(seed=seed,verified=len(report['results']))),flush=True)
    write_json(ROOT/'data/manifests'/f'representation_v2_{args.phase}_verification.json',dict(all_checks_passed=True,independently_reloaded_checkpoints=len(checks),checks=checks,source_sha256=digest_file(Path(__file__))))


if __name__=='__main__':main()
