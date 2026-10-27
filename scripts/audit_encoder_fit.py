"""Train-only diagnostics; never retune or overwrite comparison checkpoints."""
import json
import sys
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
from run_controlled_experiment import balanced_loss
sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader
from rna2d.training import Records,collate,evaluate,seed_all
from rna2d.sequence_encoders import SequencePairModel


def main():
    torch.set_num_threads(1);device='cuda:0'
    output=ROOT/'data/manifests/encoder_fit_audit_v1.json'
    if output.exists():raise ValueError('Audit frozen')
    p=json.loads((ROOT/'data/manifests/encoder_comparison_protocol_v1.json').read_text());ds=Records(ROOT/p['dataset']/'train.jsonl');record=min(ds.rows,key=lambda r:(len(r['sequence']),r['id']))
    protocol=dict(seed=987,updates=300,lr=.001,weight_decay=.01,gradient_clip=1.,selection='shortest training RNA, ID tie-break, no validation',id=record['id'],length=len(record['sequence']),pass_threshold_balanced_loss=.05,checkpoint_selection='fixed update300, no best-step selection',effect_on_main_comparison='none; fresh models only for one-RNA fit')
    write_json(ROOT/'data/manifests/encoder_fit_audit_protocol_v1.json',protocol)
    loader=DataLoader(ds,batch_size=4,shuffle=False,collate_fn=collate);single=DataLoader([record],batch_size=1,collate_fn=collate)
    tokens,labels,mask,_=collate([record]);tokens,labels,mask=tokens.to(device),labels.to(device),mask.to(device);rows=[]
    for kind in p['encoders']:
        model=SequencePairModel(kind,p['widths'][kind]).to(device);path=ROOT/'runs/encoder_comparison_v1_seed42'/(kind+'.pt');model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['model']);training_metrics,_=evaluate(model,loader,device);del model
        seed_all(protocol['seed']);model=SequencePairModel(kind,p['widths'][kind]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);model.train()
        initial=float(balanced_loss(model(tokens),labels,mask).detach())
        for _ in range(protocol['updates']):
            opt.zero_grad();loss=balanced_loss(model(tokens),labels,mask);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        model.eval()
        with torch.no_grad():final=float(balanced_loss(model(tokens),labels,mask))
        fitted,_=evaluate(model,single,device)
        row=dict(encoder=kind,main_seed42_training_metrics=training_metrics,single_RNA_initial_loss=initial,single_RNA_final_loss=final,single_RNA_f1=fitted['macro_f1'],single_RNA_loss_threshold_passed=final<.05)
        rows.append(row);print(json.dumps(dict(encoder=kind,train_f1=training_metrics['macro_f1'],single_loss=final,single_f1=fitted['macro_f1'])),flush=True);del model,opt;torch.cuda.empty_cache()
    write_json(output,dict(results=rows,main_checkpoints_modified=False,test_evaluated=False,source_sha256=digest_file(Path(__file__))))


if __name__=='__main__':main()
