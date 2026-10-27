import ast
import json
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.physics.interface import PhysicalInterface, normalized_macro
from rna2d.physics.reference import compute_rna_force_field_fast
from rna2d.physics.ghost import Ghost

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    torch.manual_seed(314)
    q=torch.randn(2,9,6)*.3
    q[:,:,:3]+=torch.arange(9)[None,:,None]*.5
    mask=torch.ones(2,9);mask[1,-2:]=0
    j=torch.rand(2,9,9);j=(j+j.transpose(1,2))*.5
    return q,j,mask


class PhysicsContractTests(unittest.TestCase):
    def test_selected_source_ast_is_unchanged(self):
        manifest=json.loads((ROOT/'data/manifests/nezha_physics_restoration_v1.json').read_text())
        old=Path('D:/Nezha/Nezha-bio')
        if not old.exists():self.skipTest('Original checkout unavailable; frozen generated hashes still available')
        for source,names in manifest['selected_top_level'].items():
            filename={'rna_system.py':'reference.py'}.get(source,source)
            a=ast.parse((old/source).read_text(encoding='utf-8-sig'))
            b=ast.parse((ROOT/'src/rna2d/physics'/filename).read_text(encoding='utf-8'))
            aa={n.name:n for n in a.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
            bb={n.name:n for n in b.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
            for name in names:self.assertEqual(ast.dump(aa[name]),ast.dump(bb[name]),name)
        a=next(n for n in a.body if isinstance(n,ast.ClassDef) and n.name=='BioSystem')
        b=next(n for n in b.body if isinstance(n,ast.ClassDef) and n.name=='BioSystem')
        for name in manifest['selected_biosystem_methods']:
            self.assertEqual(ast.dump(next(n for n in a.body if getattr(n,'name',None)==name)),ast.dump(next(n for n in b.body if getattr(n,'name',None)==name)))

    def test_force_is_negative_energy_gradient(self):
        q,j,m=fixture();engine=PhysicalInterface();q=engine.initial_state(q,m).q.detach().double().requires_grad_()
        f,e=compute_rna_force_field_fast(q,j.double(),m.double(),engine.ff_params)
        grad=torch.autograd.grad(e.sum(),q)[0]
        self.assertTrue(torch.allclose(f[...,:3],-grad[...,:3],atol=4e-5,rtol=4e-4))
        normal=q[...,3:];g=grad[...,3:];tangent=g-(g*normal).sum(-1,keepdim=True)*normal
        self.assertTrue(torch.allclose(f[...,3:],-tangent,atol=4e-5,rtol=4e-4))

    def test_observation_symmetry_mask_and_feedback(self):
        q,j,m=fixture();engine=PhysicalInterface();s=engine.initial_state(q,m)
        obs,_,_,_=engine.observe(s,j,m)
        q2=q.clone();q2[...,:3]+=10
        obs2,_,_,_=engine.observe(engine.initial_state(q2,m),j,m)
        self.assertTrue(torch.allclose(obs,obs2,atol=3e-5,rtol=3e-5))
        self.assertTrue(torch.equal(s.q[1,-2:],torch.zeros(2,6)))
        pred=normalized_macro(obs).requires_grad_();loss=engine.feedback_loss(pred,obs)
        self.assertEqual(loss.item(),0.);loss.backward();self.assertIsNotNone(pred.grad)

    def test_action_feedback_and_deterministic_replay(self):
        q,j,m=fixture();engine=PhysicalInterface();s=engine.initial_state(q,m)
        a,stats=engine.transition(s,j,m)
        b,_=engine.transition(s,j,m)
        self.assertTrue(torch.equal(a.q,b.q))
        c,_=engine.transition(s,j,m,port_force=torch.ones_like(q)*.2)
        self.assertGreater((a.q-c.q).abs().max().item(),0.)
        self.assertTrue(torch.equal(a.q[1,-2:],torch.zeros(2,6)))
        self.assertTrue(all(torch.isfinite(x).all() for x in (a.q,a.p,a.xi,*stats.values())))

    def test_original_ghost_config_and_multistep_gradients(self):
        torch.manual_seed(9);g=Ghost();state=g.init_state(2);x=torch.randn(2,256)
        self.assertEqual(g.cortex.hidden_dim,512);self.assertEqual(g.cortex.time_steps,8)
        for _ in range(3):rate,state,pred=g.step_forward_micro(x,state,dt=.05,use_grad=True)
        self.assertEqual(pred.shape,(2,5));pred.square().sum().backward()
        self.assertIsNotNone(g.cortex.phys_hallucination.weight.grad)
        self.assertTrue(torch.isfinite(rate).all())

    def test_thermostat_noise_scales_with_temperature_and_mass(self):
        from rna2d.physics.kernels import jit_integrate_trajectory
        n=20000;q=torch.zeros(n,6);z=torch.zeros(n,1)
        def sample(temp,ginv):
            torch.manual_seed(99)
            return jit_integrate_trajectory(q,q,z,q,q,q,torch.full((1,6),ginv),torch.tensor(.05),1,0.,0.,torch.full((n,1),temp),1e-8,torch.tensor(.35),2.,80.)[1]
        a=sample(1.,1.);b=sample(2.,1.);c=sample(1.,.5)
        self.assertTrue(torch.allclose(b,a*2**.5,atol=1e-6,rtol=1e-5))
        self.assertTrue(torch.allclose(c,a*2**.5,atol=1e-6,rtol=1e-5))


if __name__=='__main__':unittest.main()
