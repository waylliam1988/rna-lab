"""Summarize frozen source coverage; never promote downloaded data to gold."""
from collections import Counter
import json
from pathlib import Path
from acquire_experimental_corpus import ROOT, now, write_json


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding='utf-8'))


def main():
    plan = read('data/raw/experimental_acquisition_v1/plan.json')
    main_report = read('data/reports/experimental_acquisition_v1_all.json')
    extra = read('data/reports/additional_experimental_acquisition_v1.json')
    extra_plan = read('data/raw/additional_experimental_acquisition_v1/plan.json')
    index = read('data/experimental/internet_annotation_candidates_v1/summary.json')
    rdat = read('data/reports/rdat_container_inventory_v1.json')
    references = read('data/reports/rasp_reference_acquisition_v1.json')
    fallbacks = read('data/reports/rasp_bigwig_fallback_v1.json')
    verified_main = [r for r in main_report['results'] if r['status'].startswith('verified')]
    verified_extra = [r for r in extra['results'] if r['status'].startswith('verified')]
    verified_refs = [r for r in references['results'] if r['status'].startswith('verified')]
    verified_fallbacks = [r for r in fallbacks['results'] if r['status'].startswith('verified')]
    verified_ids = {r['id'] for r in verified_main}
    coverage = Counter(row['status'] for row in plan['pdb_coverage'])
    covered = sum(row['status'] != 'scheduled_authoritative_download' or
                  'pdb:'+row['pdb_id'] in verified_ids for row in plan['pdb_coverage'])
    raw_rows = [row for j in extra_plan['jobs'] if j['id'].startswith('rasp-scores:')
                for row in j.get('source_track_rows',[])]
    assert all(str(row['tracktype']).lower()=='raw' for row in raw_rows)
    report = dict(created_at_utc=now(),
                  acquisition_attempts_complete=main_report['complete'] and extra['complete'] and references['complete'] and fallbacks['complete'],
                  all_planned_artifacts_verified=(len(verified_main)==main_report['requested_jobs'] and
                                                 len(verified_extra)==extra['requested_jobs'] and len(verified_refs)==references['requested_jobs']),
                  current_experimental_pdb_entries=plan['pdb_entry_count'],
                  covered_current_experimental_pdb_entries=covered,
                  pdb_coverage_sources=dict(coverage),
                  main_verified_by_kind=dict(Counter(r['kind'] for r in verified_main)),
                  additional_verified_by_source=dict(Counter(r['id'].split(':')[0] for r in verified_extra)),
                  new_verified_download_bytes=sum(r['bytes'] for r in verified_main+verified_extra+verified_refs+verified_fallbacks),
                  references_verified=len(verified_refs), references_requested=references['requested_jobs'],
                  alternative_raw_bigwig_artifacts_verified=len(verified_fallbacks),
                  rasp_original_display_rows=len(raw_rows),
                  rasp_imputed_tracks_requested=0,
                  sequence_pairing_candidate_index=index,
                  rdat_container_inventory={'files':rdat['files'], 'totals':rdat['totals'], 'approved_profiles':0},
                  unavailable_original_artifacts=main_report['failed_or_quarantined']+extra['failures']+references['failures'],
                  new_training_approved_records=0,
                  limits=['Coverage is a frozen catalog acquisition boundary, not the global limit of experimental RNA data.',
                          'Original probing scores and proximity contacts are not exact base-pair labels.',
                          'Previously audited local ZIP is reused; its payloads were not all re-downloaded from the current official release.',
                          'Archive views overlap and contain chains, fragments, models and multi-chain objects; counts cannot be summed as unique training examples.',
                          'Residue-level quality, modified-base mapping, multi-chain context and homology-controlled splits remain required.'])
    write_json(ROOT/'data/reports/internet_acquisition_summary_v1.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sequence_pairing_candidate_index','limits')},indent=2))


if __name__=='__main__': main()
