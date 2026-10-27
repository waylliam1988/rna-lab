from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch not installed')
class ModelContractTests(unittest.TestCase):
    def test_padding_invariance_and_symmetry(self):
        from rna2d.model import PairBaseline
        torch.manual_seed(3)
        model = PairBaseline(16).eval()
        tokens = torch.tensor([[0, 1, 2, 3, 0, 1]])
        padded = torch.nn.functional.pad(tokens, (0, 5), value=4)
        with torch.no_grad():
            a, b = model(tokens), model(padded)[:, :6, :6]
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(a, a.transpose(-1, -2))

    def test_finite_backward(self):
        from rna2d.model import PairBaseline
        model = PairBaseline(16)
        loss = model(torch.tensor([[0, 1, 2, 3, 0, 1]])).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
