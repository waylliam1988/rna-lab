"""Replay counterfactual labels and expose a concrete hidden-momentum ambiguity."""
import json
import sys
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
import torch
from rna2d.physics.interface import PhysicalInterface,PhysicalState,normalized_macro
from rna2d.physics.causal_probe import radial_force,momentum_summary


def main():
    torch.set_num_threads(1);root=ROOT/'data/simulated/gate3_v1';engine=PhysicalInterface();groups={};checks={}
    data_manifest=json.loads((ROOT/'data/manifests/gate3_data_v1.json').read_text())
    for split in ('train','validation'):
        assert digest_file(root/(split+'.npz'))==data_manifest['splits'][split]['npz_sha256']
        arrays=np.load(root/(split+'.npz'));meta=json.loads((root/(split+'_metadata.json')).read_text())
        anchors=torch.load(root/(split+'_anchors.pt'),weights_only=True);groups[split]={r['split_group_id'] for r in meta}
        # Validate every current-state interface and counterfactual pairing.
        for i,r in enumerate(meta):
            a=anchors[r['anchor_index']];s=PhysicalState(a['q'],a['p'],a['xi'])
            assert a['id']==r['id'] and a['episode']==r['episode']
            with torch.no_grad():obs=engine.observe(s,a['j'],a['mask'])[0][0].numpy()
            assert np.allclose(obs,arrays['observations'][i,-1],atol=2e-6,rtol=2e-6)
            assert np.allclose(momentum_summary(s)[0].numpy(),arrays['momentum'][i,-1])
            assert not arrays['actions'][i,:-1].any() and abs(float(arrays['actions'][i,-1,0])-r['amplitude'])<1e-6
        replayed=0
        # All validation forks, plus one complete training counterfactual triplet.
        ids=range(len(meta)) if split=='validation' else range(3)
        for i in ids:
            r=meta[i];a=anchors[r['anchor_index']];state=PhysicalState(a['q'].clone(),a['p'].clone(),a['xi'].clone())
            with torch.no_grad():
                current=normalized_macro(engine.observe(state,a['j'],a['mask'])[0])[0].numpy();future=[]
                for step in range(1,17):
                    state,_=engine.transition(state,a['j'],a['mask'],radial_force(state,r['amplitude']))
                    if step in (1,4,8,16):future.append(normalized_macro(engine.observe(state,a['j'],a['mask'])[0])[0].numpy()-current)
            assert np.allclose(np.stack(future),arrays['delta'][i],atol=2e-6,rtol=2e-6);replayed+=1
        checks[split]=dict(current_interfaces_checked=len(meta),full_forks_replayed=replayed)
    assert not groups['train']&groups['validation']
    # Same q/force/energy observation, different momentum: construct a valid
    # partial-observability counterexample, not a claim about all natural RNAs.
    a=torch.load(root/'train_anchors.pt',weights_only=True)[0]
    base=PhysicalState(a['q'],torch.zeros_like(a['p']),torch.zeros_like(a['xi']))
    p=radial_force(base,.2)
    plus=PhysicalState(base.q.clone(),p.clone(),base.xi.clone());minus=PhysicalState(base.q.clone(),-p.clone(),base.xi.clone())
    with torch.no_grad():
        oplus=engine.observe(plus,a['j'],a['mask'])[0];ominus=engine.observe(minus,a['j'],a['mask'])[0]
        assert torch.equal(oplus,ominus)
        summaries_differ=not torch.equal(momentum_summary(plus),momentum_summary(minus))
        for _ in range(8):plus,_=engine.transition(plus,a['j'],a['mask']);minus,_=engine.transition(minus,a['j'],a['mask'])
        difference=float((normalized_macro(engine.observe(plus,a['j'],a['mask'])[0])-normalized_macro(engine.observe(minus,a['j'],a['mask'])[0])).abs().max())
    assert summaries_differ and difference>1e-5
    report=dict(checks=checks,known_group_isolation_passed=True,momentum_ambiguity=dict(current_observations_identical=True,momentum_summaries_differ=True,next8_macro_max_abs_difference=difference),all_checks_passed=True,source_sha256=digest_file(__import__('pathlib').Path(__file__)))
    write_json(ROOT/'data/manifests/gate3_data_verification_v1.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
