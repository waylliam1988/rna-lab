"""Independently reload every checkpoint and recompute declared validation."""
import argparse
import json
import sys
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
from run_physical_forecast import score
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.forecast import PhysicalForecast


def main():
    p=argparse.ArgumentParser();p.add_argument('--rollout',type=int,choices=[4,8],required=True);args=p.parse_args()
    torch.set_num_threads(1);device='cuda:0'
    result_path=ROOT/'data/manifests'/f'physical_forecast_results_r{args.rollout}_v1.json'
    results=json.loads(result_path.read_text());data=ROOT/'data/experimental/physical_forecast_v1'
    val=torch.tensor(np.load(data/'validation.npz')['observations'],device=device)
    metadata=json.loads((data/'validation_metadata.json').read_text());checks=[]
    for row in results['results']:
        path=ROOT/'runs'/f'physical_forecast_v1_r{args.rollout}'/f"seed{row['seed']}_step{row['updates']}.pt"
        ckpt=torch.load(path,map_location=device,weights_only=True)
        assert ckpt['protocol_sha256']==results['protocol_sha256']
        model=PhysicalForecast(device=device);model.load_state_dict(ckpt['model'],strict=True);model.eval()
        for h,modes in row['evaluation'].items():
            for mode,previous in modes.items():
                actual=score(model,val,metadata,ckpt['input_mean'],ckpt['input_scale'],int(h),mode=='reset')
                assert np.allclose(actual['group_macro_mse_by_channel'],previous['group_macro_mse_by_channel'],rtol=1e-5,atol=1e-8)
        checks.append(dict(seed=row['seed'],updates=row['updates'],sha256=digest_file(path)))
    report=dict(rollout=args.rollout,reloaded_checkpoints=len(checks),all_checks_passed=True,checkpoints=checks,results_sha256=digest_file(result_path))
    write_json(ROOT/'data/manifests'/f'physical_forecast_verification_r{args.rollout}_v1.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checkpoints'},indent=2))


if __name__=='__main__':main()
