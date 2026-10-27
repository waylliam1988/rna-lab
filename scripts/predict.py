import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.data import greedy_decode
from rna2d.model_factory import model_from_checkpoint
from rna2d.training import select_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--device', default='auto')
    p.add_argument('--threshold', type=float,
                   help='Override the checkpoint validation/inference score threshold')
    args = p.parse_args()
    seq = args.sequence.upper().replace('T', 'U')
    if not seq or set(seq) - set('ACGU'):
        p.error('Sequence must contain A/C/G/U only (T is normalized to U)')
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    training_config = checkpoint.get('config', {})
    if not isinstance(training_config, dict):
        training_config = {}
    threshold = training_config.get('threshold', 0.5) if args.threshold is None else args.threshold
    if not 0 <= threshold <= 1:
        p.error('Threshold must be between 0 and 1')
    model = model_from_checkpoint(checkpoint).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    with torch.no_grad():
        scores = model(torch.tensor([['ACGU'.index(c) for c in seq]], device=device)).sigmoid()[0].cpu().tolist()
    print(json.dumps({'sequence': seq, 'pairs_0based': greedy_decode(scores, seq, threshold),
                      'threshold': threshold,
                      'training_smoke_only': training_config.get('smoke_only', False),
                      'note': 'Uncalibrated scores; greedy matching. Not a validated biological predictor.'}, indent=2))


if __name__ == '__main__':
    main()
