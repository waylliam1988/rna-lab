"""Step 2: one synthetic CUDA update and checkpoint reload; no data fitting."""
import io
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.sequence_encoders import SequencePairModel
from rna2d.training import seed_all
from rna2d.thermodynamics import PriorStore
from rna2d.thermo_fusion import ThermoResidualModel
from rna2d.losses import balanced_pair_bce
from rna2d.experiment_data import file_sha256


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as folder:
        store=PriorStore(folder)
        for sequence in ('GGGAAACCC','AAAAA'):store.get(sequence,create=True)
        tokens=torch.tensor([[2,2,2,0,0,0,1,1,1],[0,0,0,0,0,4,4,4,4]],device='cuda:0')
        labels=torch.zeros((2,9,9),device=tokens.device);labels[0,0,8]=1
        mask=torch.zeros_like(labels,dtype=torch.bool);mask[0,0,8]=mask[0,1,7]=True
        results=[]
        for alpha in (0,0.25,1.0):
            seed_all(42)
            base=SequencePairModel('cnn',113,64,2).to(tokens.device)
            model=ThermoResidualModel(base,store,alpha=alpha)
            optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
            logits=model(tokens)
            if alpha==0 and not torch.equal(logits,base(tokens)):
                raise AssertionError('alpha=0 does not exactly reproduce CNN')
            loss=balanced_pair_bce(logits,labels,mask)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            final=model(tokens).detach()
            checkpoint={'model':model.state_dict(),'alpha':alpha,'epsilon':model.epsilon}
            buffer=io.BytesIO();torch.save(checkpoint,buffer);buffer.seek(0)
            checkpoint=torch.load(buffer,map_location=tokens.device,weights_only=True)
            rebuilt=ThermoResidualModel(SequencePairModel('cnn',113,64,2).to(tokens.device),store,
                                        alpha=checkpoint['alpha'],epsilon=checkpoint['epsilon'])
            rebuilt.load_state_dict(checkpoint['model'],strict=True)
            assert torch.equal(final,rebuilt(tokens))
            results.append({'alpha':alpha,'parameters':sum(p.numel() for p in model.parameters()),
                            'synthetic_one_update_loss':loss.item(),'finite_gradient':True,'exact_reload':True})
    report={'stage':'thermo-fusion-interface-v1','passed':True,'checks':results,
            'data_scope':'two synthetic sequences, one optimizer update per alpha; no experimental training or validation',
            'test_accessed':False,'source_sha256':{f:file_sha256(ROOT/f) for f in ('src/rna2d/thermo_fusion.py','scripts/smoke_thermo_fusion.py')},
            'prior_role':'fixed input offset, detached; unknown labels masked by existing BCE',
            'formula':'CNN_logits + alpha * logit(clamp(P_thermo,1e-4,1-1e-4))',
            'alpha_zero':'exact CNN output and gradients; does not access prior store',
            'production_cli_changed':False,'J_gate_enabled':False}
    path=ROOT/'data/manifests/thermo_fusion_interface_v1.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
