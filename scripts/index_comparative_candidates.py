"""Index comparative-source records without approving unknown validation labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, default=Path('D:/E backup/下载/bpseqFiles.zip'))
    args = p.parse_args()
    output = ROOT / 'data/comparative'
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with zipfile.ZipFile(args.archive) as z, (output / 'source_index_NOT_APPROVED.jsonl').open('w', encoding='utf-8') as f:
        for member in sorted(z.namelist()):
            if not member.endswith('.bpseq'): continue
            ident = Path(member).stem
            source = ident.split('_')[1]
            if source == 'PDB': continue
            counts[source] += 1
            record = {'id': ident, 'source_database': source, 'archive_member': member,
                      'evidence_class': 'reported_comparative_source_candidate',
                      'validation_technique': None, 'publication': None, 'alignment_mapping_verified': False,
                      'contains_model_predictions': None, 'quality_verified': False,
                      'auxiliary_training_approved': False,
                      'pending': ['original_entry_and_publication', 'validation_method_not_predicted',
                                  'sequence_alignment_mapping', 'homology_and_test_overlap_audit']}
            f.write(json.dumps(record) + '\n')
    report = {'candidate_records': sum(counts.values()), 'by_source': dict(counts),
              'approved_auxiliary_training_records': 0,
              'note': 'Source-level declaration only. Do not confuse absence of approval with proof that all records are predicted.'}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
