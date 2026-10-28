"""Verify pinned common dependencies are imported from system Python."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    expected_root = Path(sys.base_prefix).resolve() / 'Lib' / 'site-packages'
    rows = []
    for line in (ROOT / 'requirements-shared-common.txt').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        name, version = line.split('==', 1)
        dist = metadata.distribution(name)
        location = Path(dist.locate_file('')).resolve()
        rows.append({'name': name, 'expected': version, 'actual': dist.version,
                     'location': str(location),
                     'passed': dist.version == version and location == expected_root})
    report = {'python': platform.python_version(), 'executable': sys.executable,
              'packages': rows,
              'passed': platform.python_version() == '3.12.8' and all(r['passed'] for r in rows)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'checked': len(rows),
                      'failures': [r for r in rows if not r['passed']]}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
