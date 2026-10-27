"""Run an actual GPU training/checkpoint/evaluation/prediction round trip."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'runs/smoke_gpu')
    args = p.parse_args()
    report = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'interpreter': sys.executable, 'smoke_only': True, 'steps': []}
    report_dir = ROOT / 'data/reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        ['scripts/check_environment.py'],
        ['-m', 'pip', 'check'],
        ['-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        ['scripts/train_baseline.py', '--epochs', '2', '--batch-size', '4', '--limit', '32',
         '--max-batches', '8', '--device', 'cuda:0', '--output', str(args.output)],
        ['scripts/evaluate.py', '--checkpoint', str(args.output / 'best.pt'),
         '--split', 'validation', '--limit', '32', '--device', 'cuda:0'],
        ['scripts/predict.py', '--checkpoint', str(args.output / 'best.pt'),
         '--sequence', 'GGGAAACCC', '--device', 'cuda:0'],
    ]
    for command in commands:
        print('RUN:', ' '.join(command), flush=True)
        result = subprocess.run([sys.executable, *command], cwd=ROOT, text=True,
                                encoding='utf-8', errors='replace', capture_output=True)
        print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
        report['steps'].append({'command': command, 'returncode': result.returncode,
                                'stdout': result.stdout, 'stderr': result.stderr})
        report['success'] = result.returncode == 0
        report['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        (report_dir / 'smoke_test.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
