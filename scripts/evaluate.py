import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.training import Records, collate, evaluate, fingerprint, select_device
from rna2d.experiment_data import validate_split_manifest
from rna2d.model_factory import model_from_checkpoint
from rna2d.reporting import group_macro_f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--split', choices=['validation'], default='validation',
                   help='Only the development validation split is available through this entrypoint')
    p.add_argument('--device', default='auto')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--limit', type=int, help='Smoke test only')
    p.add_argument('--threshold', type=float,
                   help='Override the checkpoint threshold for an explicitly chosen validation diagnostic')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/experimental/joint_split_v2')
    p.add_argument('--data-manifest', type=Path, default=ROOT / 'data/manifests/joint_split_v2_validation.json')
    p.add_argument('--allow-legacy-checkpoint', action='store_true',
                   help='Explicitly evaluate a checkpoint that lacks a training-manifest fingerprint')
    args = p.parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    try:
        data_manifest_sha256 = validate_split_manifest(args.data_dir, args.data_manifest, ['validation'])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(f'Experimental split manifest rejected the selected data: {exc}')
    training_config = checkpoint.get('config', {})
    if not isinstance(training_config, dict):
        training_config = {}
    training_manifest_sha256 = training_config.get('data_manifest_sha256')
    if training_manifest_sha256 is None and not args.allow_legacy_checkpoint:
        p.error('Checkpoint has no training data-manifest fingerprint; pass --allow-legacy-checkpoint only for an explicit historical validation run.')
    if training_manifest_sha256 is not None and training_manifest_sha256 != data_manifest_sha256:
        p.error('Checkpoint and evaluation data use different split manifests.')
    current_split_hashes = {
        split: fingerprint(args.data_dir / f'{split}.jsonl')
        for split in ('train', 'validation')
    }
    training_split_hashes = training_config.get('data_sha256')
    if not isinstance(training_split_hashes, dict):
        declared_train = training_config.get('train_sha256')
        declared_validation = training_config.get('validation_sha256')
        if declared_train is not None or declared_validation is not None:
            training_split_hashes = {
                'train': declared_train,
                'validation': declared_validation,
            }
    if isinstance(training_split_hashes, dict):
        for split, expected_sha256 in training_split_hashes.items():
            if expected_sha256 is not None and split in current_split_hashes and current_split_hashes[split] != expected_sha256:
                p.error(f'Checkpoint and current {split} split fingerprints differ.')
        if any(training_split_hashes.get(split) is None for split in ('train', 'validation')) and not args.allow_legacy_checkpoint:
            p.error('Checkpoint lacks a complete train/validation data fingerprint; pass --allow-legacy-checkpoint only for an explicit historical validation run.')
    elif not args.allow_legacy_checkpoint:
        p.error('Checkpoint has no train/validation data fingerprints; pass --allow-legacy-checkpoint only for an explicit historical validation run.')
    model = model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    threshold = training_config.get('threshold', 0.5) if args.threshold is None else args.threshold
    if not 0 <= threshold <= 1:
        p.error('Threshold must be between 0 and 1')
    path = args.data_dir / f'{args.split}.jsonl'
    dataset = Records(path, args.limit)
    summary, details = evaluate(model, DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate), device, threshold)
    group_f1, per_group_f1 = group_macro_f1(details, dataset.rows)
    summary.update({'validation_group_macro_f1': group_f1,
                    'validation_per_group_f1': per_group_f1})
    summary.update({'split': args.split, 'data_sha256': fingerprint(path),
                    'data_manifest_sha256': data_manifest_sha256,
                    'checkpoint_sha256': fingerprint(args.checkpoint), 'limit': args.limit,
                    'threshold_override': args.threshold,
                    'training_smoke_only': training_config.get('smoke_only', False),
                    'legacy_checkpoint_override': args.allow_legacy_checkpoint})
    output = args.checkpoint.parent / f'eval_{args.split}'
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    output.with_suffix('.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in details), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
