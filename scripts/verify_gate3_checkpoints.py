"""Independent checkpoint+normalization reload; no refitting or test access."""
import json
import sys
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
from run_gate3_probe import load_data,predict,score
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.causal_probe import ResidualProbe
from rna2d.physics.diagnostics import macro_numpy


def main():
    torch.set_num_threads(1);device='cuda:0';root=ROOT/'data/simulated/gate3_v1'
    manifest=json.loads((ROOT/'data/manifests/gate3_data_v1.json').read_text())
    for split in ('train','validation'):
        for suffix,key in [('.npz','npz_sha256'),('_metadata.json','metadata_sha256'),('_anchors.pt','anchors_sha256')]:
            assert digest_file(root/(split+suffix))==manifest['splits'][split][key]
    assert digest_file(ROOT/'data/manifests/gate3_data_generation_protocol_v1.json')==manifest['protocol_sha256']
    val,meta=load_data('validation',device);actual=val['delta'].cpu().numpy();current=macro_numpy(val['observations'][:,-1].cpu().numpy());checks=[]
    for seed in (42,43,44):
        report=json.loads((ROOT/'data/manifests'/f'gate3_seed{seed}_v1.json').read_text())
        for row in report['results']:
            variant=row['variant'];path=ROOT/'runs'/f'gate3_v1_seed{seed}'/(variant+'.pt')
            assert digest_file(path)==row['checkpoint_sha256']
            checkpoint=torch.load(path,map_location=device,weights_only=True)
            assert checkpoint['protocol_sha256']==report['protocol_sha256']
            model=ResidualProbe(variant,device,matched_parameters=checkpoint['matched_parameters']);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
            stats=[s.to(device) for s in checkpoint['stats']]
            pred=predict(model,val,stats,device)
            metrics=score(pred,actual,meta,stats[4].cpu().numpy(),current)
            for a,b in zip(metrics,row['validation']['ordered']):
                assert np.allclose(a['mse_by_channel'],b['mse_by_channel'],rtol=2e-5,atol=1e-8)
            checks.append(dict(seed=seed,variant=variant,sha256=row['checkpoint_sha256']))
            del model
        print(json.dumps(dict(seed=seed,reloads_verified=len(report['results']))),flush=True)
    result=dict(all_checks_passed=True,independently_reloaded_checkpoints=len(checks),checks=checks,data_hashes_passed=True,script_sha256=digest_file(Path(__file__)))
    write_json(ROOT/'data/manifests/gate3_checkpoint_verification_v1.json',result)


if __name__=='__main__':main()
