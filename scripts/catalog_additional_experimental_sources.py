"""Discover official RNA-derived annotations and original probing tracks.

No inferred/imputed RASP tracks are scheduled. Geometry annotations remain
derived evidence until mapped back to experimental PDB structures and QC'd.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlencode

from acquire_experimental_corpus import request_json, write_json, session

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / 'data/raw/additional_experimental_catalog_v1'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'plan.json').exists():
        print('Frozen additional-source plan exists; use it rather than overwrite.')
        return
    solo = request_json('https://rnasolo.cs.put.poznan.pl/api/archives/details/')
    write_json(directory / 'rnasolo_archive_details.json', solo)
    jobs = []
    for repo, molecules in solo['results'].items():
        for molecule, tree in molecules.items():
            choices = tree['All']['All']['all']
            for fmt in ['fasta', 'bpseq', 'dbn', 'txt']:
                asset = choices.get(fmt)
                if not asset: continue
                path = asset['file']
                name = path.rsplit('/', 1)[-1]
                if not re.fullmatch(r'[A-Za-z0-9_.+-]+', name): raise ValueError('Unsafe archive name')
                jobs.append({'id': 'rnasolo:' + name, 'kind': 'zip' if name.endswith('.zip') else 'text',
                             'url': 'https://rnasolo.cs.put.poznan.pl' + path,
                             'relative_path': 'rnasolo/' + name, 'expected_bytes': int(asset['size']),
                             'expected_sha256': None, 'repository_view': repo, 'molecule_view': molecule,
                             'format': fmt, 'reported_structures': asset['structure_count'],
                             'evidence_class': 'experimental_coordinate_derived_annotation_or_sequence_candidate',
                             'note': 'Overlapping repository/molecule views; counts are not additive unique RNAs.'})
    # Documented FR3D bulk coordinate-derived interaction annotation.
    jobs.append({'id': 'fr3d:all_interactions', 'kind': 'interaction_csv_gz',
                 'url': 'https://rna.bgsu.edu/rna3dhub/data/interactions.gz',
                 'relative_path': 'fr3d/interactions.csv.gz', 'expected_bytes': None, 'expected_sha256': None,
                 'evidence_class': 'coordinate_derived_interaction_annotation',
                 'note': 'Independent geometric annotation, not unknown-sequence fold prediction. Must map unit IDs, experimental methods and quality before training.'})
    source = session().get('https://rasp2.zhanglab.net/static/js/RASP.js', timeout=60)
    source.raise_for_status()
    (directory / 'RASP_species_reference.js').write_text(source.text, encoding='utf-8')
    block = re.search(r'const humanName2dbName\s*=\s*\{(.*?)\};', source.text, re.S).group(1)
    species = dict(re.findall(r"'([^']+)'\s*:\s*'([^']+)'", block))
    species = {name: key for name, key in species.items() if name != 'Nucleotides'}

    def fetch_species(item):
        human, key = item
        tracks = request_json('https://rasp2.zhanglab.net/api/query_bwfiles/', {'species': key})
        contacts = request_json('https://rasp2.zhanglab.net/api/query_bedpefiles/', {'species': key})
        return human, key, tracks, contacts

    track_counts, excluded, track_records, contact_records = Counter(), [], [], []
    catalog_failures = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [(item, pool.submit(fetch_species, item)) for item in species.items()]
        for (human, key), future in futures:
            try: _, _, tracks, contacts = future.result()
            except Exception as exc:
                catalog_failures.append({'species': human, 'error': f'{type(exc).__name__}: {exc}'})
                continue
            write_json(directory / f'rasp_{key}_scores.json', tracks)
            write_json(directory / f'rasp_{key}_contacts.json', contacts)
            for track in tracks.get('bw_files', []):
                track_records.append(track)
                track_type = str(track.get('tracktype', '')).lower()
                track_counts[track_type] += 1
                if track_type != 'raw':
                    excluded.append({'species': key, 'label': track.get('label'), 'tracktype': track_type,
                                     'reason': 'not_explicitly_original_raw_signal'})
                    continue
                remote = str(track.get('bed_file_path') or '').strip()
                if not remote:
                    excluded.append({'species': key, 'label': track.get('label'), 'reason': 'no_original_BED_download'})
                    continue
                ident = hashlib.sha256(remote.encode()).hexdigest()[:24]
                jobs.append({'id': f'rasp-scores:{key}:{ident}', 'kind': 'tar',
                             'url': 'https://rasp2.zhanglab.net/api/downloadfile/?' + urlencode({'filetype': 'bed', 'filelist': remote}),
                             'relative_path': f'rasp_scores/{key}/{ident}.tar',
                             'expected_bytes': None, 'expected_sha256': None,
                             'evidence_class': 'original_experimental_probing_track_container',
                             'source_track': track, 'tracktype_verified': 'raw',
                             'note': 'Original measured signal; transcript/coordinate mapping and assay QC still required, not pair labels.'})
            for contact in contacts.get('bedpe_files', []):
                contact_records.append(contact)
                # Actual endpoint field is preserved verbatim; don't invent a path.
                remote = str(contact.get('file_path') or contact.get('bedpe_file_path') or '').strip()
                if not remote:
                    excluded.append({'species': key, 'reason': 'contact_file_path_unresolved', 'record': contact})
                    continue
                ident = hashlib.sha256(remote.encode()).hexdigest()[:24]
                jobs.append({'id': f'rasp-contacts:{key}:{ident}', 'kind': 'tar',
                             'url': 'https://rasp2.zhanglab.net/api/downloadfile/?' + urlencode({'filetype': 'bedpe', 'filelist': remote}),
                             'relative_path': f'rasp_contacts/{key}/{ident}.tar',
                             'expected_bytes': None, 'expected_sha256': None,
                             'evidence_class': 'experimental_proximity_ligation_contact_container',
                             'source_track': contact,
                             'note': 'Contact intervals are not exact Watson-Crick pairing truth.'})
    write_json(directory / 'rasp_excluded_or_review.json', excluded)
    # Identical file paths may be exposed under multiple display rows.
    unique_jobs = {job['url']: job for job in jobs}
    plan = {'protocol': 'additional-experimental-sources-v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'jobs': list(unique_jobs.values()),
            'rasp_species_queried': len(species), 'rasp_track_types': dict(track_counts),
            'rasp_score_rows': len(track_records), 'rasp_contact_rows': len(contact_records),
            'rasp_excluded_or_review_rows': len(excluded), 'catalog_failures': catalog_failures,
            'rnasolo_views_are_overlapping': True, 'training_approved': False}
    write_json(directory / 'plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'jobs'}, indent=2), flush=True)
    print('Download jobs by source:', Counter(j['id'].split(':')[0] for j in plan['jobs']), flush=True)


if __name__ == '__main__':
    main()
