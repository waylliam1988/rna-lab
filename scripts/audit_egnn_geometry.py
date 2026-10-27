"""Audit actual EGNN tensors, variable padding and coordinate-to-output path."""
import json
import sys
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.sequence_encoders import SequencePairModel
from rna2d.training import Records,collate,seed_all


def inspect(model,tokens):
    captures={};handles=[]
    for i,layer in enumerate(model.layers):
        def pre(module,args,i=i):
            h,x,mask=args;captures[i]=dict(h=h.detach().clone(),x=x.detach().clone(),mask=mask.detach().clone())
        def edge(module,args,out,i=i):captures[i]['edge_input']=args[0].detach().clone()
        def coeff(module,args,out,i=i):captures[i]['coordinate_coeff']=out.detach().clone()
        def post(module,args,out,i=i):captures[i]['h_next']=out[0].detach().clone();captures[i]['x_next']=out[1].detach().clone()
        handles += [layer.register_forward_pre_hook(pre),layer.edge.register_forward_hook(edge),layer.coordinate.register_forward_hook(coeff),layer.register_forward_hook(post)]
    logits=model(tokens)
    for handle in handles:handle.remove()
    return logits,captures


def run_audit():
    torch.set_num_threads(1);p=json.loads((ROOT/'data/manifests/encoder_comparison_protocol_v1.json').read_text());ds=Records(ROOT/p['dataset']/'train.jsonl')
    selected=[]
    for length in (33,64,120):
        selected.append(min(ds.rows,key=lambda r:(abs(len(r['sequence'])-length),r['id'])))
    tokens,labels,mask,rows=collate(selected);seed_all(987);model=SequencePairModel('egnn_seed',p['widths']['egnn_seed']).eval()
    checkpoint=ROOT/'runs/encoder_comparison_v1_seed42/egnn_seed.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    logits,captures=inspect(model,tokens);report=[]
    for i,c in captures.items():
        h,x,valid=c['h'],c['x'],c['mask'];n=x.shape[1]
        distance=(x[:,:,None,:]-x[:,None,:,:]).square().sum(-1,keepdim=True)
        edges=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,dtype=torch.bool)[None]
        assert torch.equal(c['edge_input'][...,128:129],distance)
        assert torch.count_nonzero(x[~valid])==0 and torch.count_nonzero(c['x_next'][~valid])==0
        assert torch.count_nonzero(c['h_next'][~valid])==0
        shift=(c['x_next']-x)[valid].abs().max().item();assert shift>1e-7
        row=dict(layer=i,node_features=list(h.shape),coordinates=list(x.shape),squared_distances=list(distance.shape),edge_input=list(c['edge_input'].shape),edge_mask=list(edges.shape),coordinate_coefficients=list(c['coordinate_coeff'].shape),valid_lengths=valid.sum(1).tolist(),valid_edges=edges.sum((1,2)).tolist(),expected_valid_edges=(valid.sum(1)*(valid.sum(1)-1)).tolist(),padding_coordinates_zero=True,padding_outputs_zero=True,distance_present_in_edge_input=True,max_coordinate_update=shift)
        assert row['valid_edges']==row['expected_valid_edges'];report.append(row)
    # Batch alignment: compare each valid block and each layer's output coordinates.
    max_error=0.
    with torch.no_grad():
        for b,r in enumerate(rows):
            length=len(r['sequence']);alone,cap=inspect(model,tokens[b:b+1,:length])
            error=float((logits[b,:length,:length]-alone[0]).abs().max());max_error=max(max_error,error)
            assert torch.allclose(logits[b,:length,:length],alone[0],atol=2e-5,rtol=2e-4)
            for i in cap:assert torch.allclose(captures[i]['x_next'][b,:length],cap[i]['x_next'][0],atol=2e-5,rtol=2e-4)
    # Original classification loss must reach coordinate-update parameters.
    model.zero_grad();loss=logits.square().mean();loss.backward();gradients={}
    for name,param in model.named_parameters():
        if '.coordinate.' in name:
            assert param.grad is not None and torch.isfinite(param.grad).all()
            gradients[name]=float(param.grad.norm())
    assert all(sum(v for k,v in gradients.items() if k.startswith(f'layers.{i}.'))>0 for i in range(2))
    # Intervene only on coordinate updates. Distances still enter edge messages.
    handles=[layer.coordinate.register_forward_hook(lambda module,args,out:torch.zeros_like(out)) for layer in model.layers]
    with torch.no_grad():bypassed=model(tokens)
    for handle in handles:handle.remove()
    influence=float((logits.detach()-bypassed).abs().max());assert influence>1e-7
    return dict(all_checks_passed=True,records=[dict(id=r['id'],length=len(r['sequence'])) for r in rows],layers=report,padding_vs_individual_max_logit_error=max_error,coordinate_gradients=gradients,zero_coordinate_update_max_logit_difference=influence,input_type='synthetic sequence-position coordinates(cos i,sin i,i/128), not RNA experimental geometry',physical_RNA_coordinates_used=False,model_source_sha256=digest_file(ROOT/'src/rna2d/sequence_encoders.py'),checkpoint_sha256=digest_file(checkpoint),repair_required=False)


def main():
    out=ROOT/'data/manifests/egnn_geometry_audit_v1.json'
    if out.exists():raise ValueError('Audit frozen')
    report=run_audit();write_json(out,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
