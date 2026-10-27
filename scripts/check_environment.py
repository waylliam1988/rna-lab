"""Test real CUDA kernels and backprop on every visible GPU."""
import json
import platform
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.model import PairBaseline
from rna2d.core import GhostDynamics, GhostSettings


def main():
    report = {'python': platform.python_version(), 'executable': sys.executable,
              'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(),
              'gpus': []}
    if platform.python_version() != '3.12.8':
        raise RuntimeError('This project requires Python 3.12.8')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; environment is not ready')
    for index in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{index}')
        torch.cuda.set_device(device)
        # The allocator counters require an initialized device context in 2.7.
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        model = PairBaseline().to(device)
        optim = torch.optim.AdamW(model.parameters())
        tokens = torch.randint(0, 4, (2, 256), device=device)
        loss = model(tokens).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optim.step()
        core = GhostDynamics(GhostSettings(input_dim=8, hidden_dim=16, inner_steps=4)).to(device)
        core_input = torch.randn(2, 8, device=device, requires_grad=True)
        state = None
        core_loss = core_input.new_zeros(())
        for _ in range(4):
            rate, state, prediction = core(core_input, state)
            core_loss = core_loss + rate.square().mean() + prediction.square().mean()
        core_loss.backward()
        if not torch.isfinite(core_input.grad).all() or not all(torch.isfinite(v).all() for v in state.values()):
            raise FloatingPointError('Ghost/RWKV CUDA gradient or state nonfinite')
        torch.cuda.synchronize(device)
        props = torch.cuda.get_device_properties(device)
        report['gpus'].append({'index': index, 'name': props.name,
                               'total_MiB': props.total_memory / 1024**2,
                               'compute_capability': list(torch.cuda.get_device_capability(index)),
                               'forward_backward_optimizer': 'passed', 'ghost_rwkv_4step_backward': 'passed', 'loss': loss.item(),
                               'peak_allocated_MiB': torch.cuda.max_memory_allocated(device) / 1024**2})
        del model, optim, tokens, loss, core, core_input, state, core_loss, rate, prediction
    path = ROOT / 'data/reports/environment.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
