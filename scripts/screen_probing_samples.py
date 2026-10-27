import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.evidence import screen_rdat


def main():
    rows = []
    for path in sorted((ROOT / 'data/raw/source_catalog').glob('*.rdat')):
        rows.extend({'source_file': path.name, **r} for r in screen_rdat(path.read_text(encoding='utf-8-sig')))
    output = ROOT / 'data/experimental/probing_samples_screened.jsonl'
    output.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
    summary = [{'file': r['source_file'], 'profile': r['profile_id'], 'measured_positions': sum(r['measured_mask']),
                'positions': len(r['values']), 'coverage': r['coverage'], 'reasons': r['reasons'],
                'screen_pass': r['preliminary_screen_pass']} for r in rows]
    (ROOT / 'data/reports/probing_sample_qc.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
