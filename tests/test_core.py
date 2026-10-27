import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
try:
    import torch
except ImportError:
    torch = None


def legacy_class(name, file):
    source = (ROOT / 'legacy_reference' / file).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    return '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno])


@unittest.skipIf(torch is None, 'PyTorch not installed')
class CoreMigrationTests(unittest.TestCase):
    def test_primitives_are_verbatim(self):
        for name, old_file, new_file in [('RWKVCellLite', 'kernels.py', 'memory.py'),
                                          ('ContinuousControlLayer', 'ghost.py', 'control.py')]:
            source = (ROOT / 'src/rna2d/core' / new_file).read_text(encoding='utf-8')
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef))
            self.assertEqual('\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno]), legacy_class(name, old_file))

    def test_old_cortex_numerical_parity_over_steps(self):
        from rna2d.core import GhostDynamics, GhostSettings, RWKVCellLite, ContinuousControlLayer
        settings = GhostSettings(input_dim=8, hidden_dim=12, inner_steps=4)
        # Execute only a class definition with explicit, isolated dependencies.
        namespace = {'torch': torch, 'nn': torch.nn, 'math': math, 'RWKVCellLite': RWKVCellLite,
                     'ContinuousControlLayer': ContinuousControlLayer,
                     'Config': SimpleNamespace(GHOST_CORTEX_TIME_STEPS=4)}
        exec(compile(legacy_class('NeuralCortex', 'ghost.py'), '<isolated-legacy-cortex>', 'exec'), namespace)
        old = namespace['NeuralCortex'](8, 12).eval()
        new = GhostDynamics(settings).eval()
        mapped = {key.replace('phys_hallucination.', 'prediction_head.'): value for key, value in old.state_dict().items()}
        new.load_state_dict(mapped, strict=True)
        old_state, new_state = None, None
        torch.manual_seed(42)
        for _ in range(5):
            x = torch.randn(2, 8)
            with torch.no_grad():
                a, old_state, ah = old(x, current_state=old_state, use_grad=True, record_state=False)
                b, new_state, bh = new(x, new_state)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(ah, bh)
            for key in old_state:
                torch.testing.assert_close(old_state[key], new_state[key], atol=1e-6, rtol=1e-5)

    def test_reset_does_not_leak_previous_sample(self):
        from rna2d.core import GhostDynamics, GhostSettings
        model = GhostDynamics(GhostSettings(input_dim=8, hidden_dim=12, inner_steps=4)).eval()
        a, b = torch.randn(1, 8), torch.randn(1, 8)
        expected = model(b, None)[0]
        model(a, None)
        actual = model(b, None)[0]
        torch.testing.assert_close(expected, actual)

    def test_unroll_gradient_and_explicit_detach(self):
        from rna2d.core import GhostDynamics, GhostSettings
        torch.manual_seed(3)
        model = GhostDynamics(GhostSettings(input_dim=8, hidden_dim=12, inner_steps=4))
        x = torch.randn(2, 8, requires_grad=True)
        state = None
        outputs = []
        for _ in range(12):
            rate, state, prediction = model(x, state)
            self.assertTrue(torch.isfinite(rate).all())
            self.assertTrue(all(torch.isfinite(v).all() for v in state.values()))
            outputs.append(rate.square().mean() + prediction.square().mean())
        sum(outputs).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(not v.requires_grad for v in model.detach_state(state).values()))

    def test_rwkv_zero_drive(self):
        from rna2d.core import RWKVCellLite
        model = RWKVCellLite(8, 12, key_mode='softplus_zero')
        y, state = model(torch.zeros(2, 8))
        self.assertEqual(y.count_nonzero().item(), 0)
        self.assertTrue(all(v.count_nonzero().item() == 0 for v in state))


if __name__ == '__main__':
    unittest.main()
