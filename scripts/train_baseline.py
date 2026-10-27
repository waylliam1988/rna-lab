import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.model import PairBaseline
from rna2d.training import Records, collate, evaluate, fingerprint, seed_all, select_device
from rna2d.evidence import pairing_evidence_issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int, help='Smoke test only: first N samples per split')
    p.add_argument('--max-batches', type=int, help='Smoke test only: truncate each epoch')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/baseline')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/experimental/paired',
                   help='Strict experimental paired dataset with train.jsonl and validation.jsonl')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden < 1 or args.lr <= 0:
        p.error('Positive epochs, batch size, hidden and lr required')
    if (args.limit is not None and args.limit < 1) or (args.max_batches is not None and args.max_batches < 1):
        p.error('Smoke limits must be positive')
    paths = {s: args.data_dir / f'{s}.jsonl' for s in ['train', 'validation']}
    if not all(path.exists() for path in paths.values()):
        p.error('Strict experimental paired dataset is not ready. See docs/EXPERIMENTAL_DATA_AUDIT.md. '
                'Legacy benchmark files are not approved experimental training labels.')
    datasets = {s: Records(path, args.limit) for s, path in paths.items()}
    for split, dataset in datasets.items():
        for record in dataset.rows:
            issues = pairing_evidence_issues(record)
            if issues:
                p.error(f'Experimental evidence gate rejected {split}:{record.get("id")}: {issues}')
    # Do not create experiment outputs until the data pass the evidence gate.
    args.output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    device = select_device(args.device)
    model = PairBaseline(args.hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loaders = {s: DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=s == 'train', collate_fn=collate) for s, dataset in datasets.items()}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'device_used': str(device), 'torch_version': str(torch.__version__),
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'data_sha256': {s: fingerprint(path) for s, path in paths.items()},
                   'smoke_only': args.limit is not None or args.max_batches is not None})
    (args.output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, (tokens, labels, mask, _) in enumerate(loaders['train']):
            if args.max_batches is not None and step >= args.max_batches:
                break
            tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
            if not mask.any():
                continue
            optim.zero_grad(set_to_none=True)
            logits = model(tokens)
            # Balance positive/negative classes within each sample; average samples equally.
            sample_losses = []
            for b in range(len(tokens)):
                pos, neg = mask[b] & labels[b].bool(), mask[b] & ~labels[b].bool()
                terms = []
                if pos.any(): terms.append(torch.nn.functional.softplus(-logits[b][pos]).mean())
                if neg.any(): terms.append(torch.nn.functional.softplus(logits[b][neg]).mean())
                if terms: sample_losses.append(torch.stack(terms).mean())
            loss = torch.stack(sample_losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optim.step()
            losses.append(loss.item())
        summary, _ = evaluate(model, loaders['validation'], device)
        row = {'epoch': epoch, 'train_loss': sum(losses) / len(losses) if losses else None, **summary}
        print(json.dumps(row), flush=True)
        with (args.output / 'history.jsonl').open('a') as out:
            out.write(json.dumps(row) + '\n')
        if summary['macro_f1'] > best:
            best = summary['macro_f1']
            torch.save({'model': model.state_dict(), 'hidden': args.hidden, 'epoch': epoch,
                        'validation': summary, 'config': config}, args.output / 'best.pt')


if __name__ == '__main__':
    main()
