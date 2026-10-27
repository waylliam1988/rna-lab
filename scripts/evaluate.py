import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.model import PairBaseline
from rna2d.training import Records, collate, evaluate, fingerprint, select_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split', choices=['validation', 'test', 'test_external', 'archiveII'], default='validation')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--limit', type=int, help='Smoke test only')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/experimental/paired_v0_2')
    args = p.parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = PairBaseline(checkpoint['hidden']).to(device)
    model.load_state_dict(checkpoint['model'])
    path = args.data_dir / f'{args.split}.jsonl'
    if not path.exists():
        p.error('Strict experimental paired dataset is not ready; see docs/EXPERIMENTAL_DATA_AUDIT.md')
    summary, details = evaluate(model, DataLoader(Records(path, args.limit), batch_size=args.batch_size, collate_fn=collate), device)
    summary.update({'split': args.split, 'data_sha256': fingerprint(path),
                    'checkpoint_sha256': fingerprint(args.checkpoint), 'limit': args.limit,
                    'training_smoke_only': checkpoint['config'].get('smoke_only', False)})
    output = args.checkpoint.parent / f'eval_{args.split}'
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    output.with_suffix('.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in details), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
