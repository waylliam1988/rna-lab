"""Train-only independent reload of every completed lr-scan checkpoint."""
import json
import sys
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader
from rna2d.training import Records,collate,evaluate
from rna2d.sequence_encoders import SequencePairModel


def main():
    torch.set_num_threads(1);device='cuda:0';pp=ROOT/'data/manifests/encoder_lr_scan_protocol_v1.json';p=json.loads(pp.read_text());loader=DataLoader(Records(ROOT/p['training_file']),batch_size=4,collate_fn=collate);checks=[]
    assert digest_file(ROOT/'src/rna2d/sequence_encoders.py')==p['model_source_sha256']
    assert digest_file(ROOT/p['training_file'])==p['training_sha256']
    for seed in p['seeds']:
        report=json.loads((ROOT/'runs'/f'encoder_lr_scan_v1_seed{seed}'/'results.json').read_text())
        for row in report['results']:
            if row['status']!='completed':continue
            kind=row['encoder'];path=ROOT/'runs'/f'encoder_lr_scan_v1_seed{seed}'/(kind+'_'+row['variant']+'.pt')
            assert digest_file(path)==row['checkpoint_sha256'];ck=torch.load(path,map_location=device,weights_only=True);assert ck['protocol_sha256']==digest_file(pp)
            model=SequencePairModel(kind,p['widths'][kind]).to(device);model.load_state_dict(ck['model'],strict=True);metrics,details=evaluate(model,loader,device)
            assert abs(metrics['macro_f1']-row['training']['macro_f1'])<1e-9
            assert len(details)==len(row['details'])==len(loader.dataset)
            assert all(a['predicted_pairs']==b['predicted_pairs'] for a,b in zip(details,row['details']))
            checks.append(dict(seed=seed,encoder=kind,variant=row['variant'],sha256=row['checkpoint_sha256']));del model;torch.cuda.empty_cache()
        print(json.dumps(dict(seed=seed,verified=len(checks))),flush=True)
    write_json(ROOT/'data/manifests/encoder_lr_scan_verification_v1.json',dict(all_checks_passed=True,reloaded_checkpoints=len(checks),checks=checks,validation_loaded=False,test_loaded=False,source_sha256=digest_file(Path(__file__))))


if __name__=='__main__':main()
