"""Count explicit RDAT records and declared warnings without inventing labels.

This is a container inventory, not per-profile quality approval. Structures in
RDAT are deliberately ignored; SEQPOS/mutations/conditions need separate mapping.
"""
from collections import Counter
import json
from acquire_experimental_corpus import ROOT, now, write_json


def main():
    report = json.loads((ROOT/'data/reports/experimental_acquisition_v1_all.json').read_text(encoding='utf-8'))
    rows = []
    for receipt in report['results']:
        if receipt.get('kind')!='rdat' or not receipt['status'].startswith('verified'): continue
        counts = Counter()
        path = ROOT/receipt['path']
        # Count ASCII field names as bytes; historical comments may use Latin-1.
        with path.open('rb') as stream:
            for line in stream:
                key = line.split(None,1)[0] if line.strip() else b''
                if key.startswith(b'ANNOTATION_DATA:'):
                    counts['annotation_data_rows'] += 1
                    if b'sequence:' in line: counts['rows_with_explicit_sequence_annotation'] += 1
                    if b'warning:badQuality' in line: counts['declared_bad_quality_rows'] += 1
                    if b'signal_to_noise:weak:' in line: counts['declared_weak_snr_rows'] += 1
                elif key.startswith(b'REACTIVITY:'): counts['reactivity_rows'] += 1
                elif key.startswith(b'REACTIVITY_ERROR:'): counts['reactivity_error_rows'] += 1
                elif key==b'SEQPOS': counts['seqpos_header_rows'] += 1
                elif key==b'SEQUENCE': counts['sequence_header_rows'] += 1
        rows.append(dict(id=receipt['id'], sha256=receipt['sha256'], counts=dict(counts)))
    total = Counter()
    for row in rows: total.update(row['counts'])
    result = dict(created_at_utc=now(), files=len(rows), totals=dict(total), files_audit=rows,
                  approved_profiles=0, limitations=[
                      'Rows are not unique RNA sequences or independent experiments.',
                      'Missing warnings do not imply good quality. Explicit badQuality/weak-SNR flags require exclusion or review.',
                      'No secondary-structure fields used as experimental truth.',
                      'Per-profile trial conditions, coverage, reagent specificity, negative controls and sequence mapping remain pending.'])
    write_json(ROOT/'data/reports/rdat_container_inventory_v1.json',result)
    print(json.dumps(dict(files=len(rows),totals=dict(total)),indent=2))


if __name__=='__main__': main()
