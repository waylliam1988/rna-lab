"""Independent final checkpoint reload on GPU0; test remains untouched."""
import json
import sys
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader
from rna2d.sequence_encoders import SequencePairModel
from rna2d.training import Records,collate,evaluate


def main():
    torch.set_num_threads(1);device='cuda:0';p=json.loads((ROOT/'data/manifests/encoder_comparison_protocol_v1.json').read_text());loader=DataLoader(Records(ROOT/p['dataset']/'validation.jsonl'),batch_size=4,collate_fn=collate);checks=[]
    for seed in p['seeds']:
        report=json.loads((ROOT/'data/manifests'/f'encoder_comparison_seed{seed}_v1.json').read_text())
        for row in report['results']:
            kind=row['encoder'];path=ROOT/'runs'/f'encoder_comparison_v1_seed{seed}'/(kind+'.pt');assert digest_file(path)==row['checkpoint_sha256']
            ck=torch.load(path,map_location=device,weights_only=True);model=SequencePairModel(kind,ck['width'],ck['hidden'],ck['depth']).to(device);model.load_state_dict(ck['model'],strict=True);metrics,details=evaluate(model,loader,device)
            assert abs(metrics['macro_f1']-row['validation']['macro_f1'])<1e-9
            assert all(a['predicted_pairs']==b['predicted_pairs'] for a,b in zip(details,row['details']))
            checks.append(dict(seed=seed,encoder=kind,sha256=row['checkpoint_sha256']));del model;torch.cuda.empty_cache()
    write_json(ROOT/'data/manifests/encoder_comparison_verification_v1.json',dict(all_checks_passed=True,independently_reloaded_checkpoints=len(checks),checks=checks,source_sha256=digest_file(Path(__file__)),test_evaluated=False))
    print('Verified',len(checks),'final checkpoints and decoded predictions')


if __name__=='__main__':main()
