"""Build a conservative first paired-data version from screened coordinates.

Uses authoritative fresh PDB downloads plus Barnaba geometry annotation. No
RNAfold/AI-generated structure labels. Separate unknown homology and conflicting
conformers instead of silently forcing a unique answer.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import rna2d  # registers optional app-local native runtime before binary imports
from Bio.Align import PairwiseAligner
import requests
import gemmi
from rna2d.annotation import annotate_chain
from rna2d.coordinate_qc import parse_coordinate_entry, values
from rna2d.data import crossing_pairs


def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows), encoding='utf-8')


def download_authoritative_cif(pdb_id, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (pdb_id.lower() + '.cif')
    receipt = path.with_suffix('.receipt.json')
    if path.exists():
        if not receipt.exists(): return pdb_id, 'Existing raw CIF lacks download receipt; preserve file and review cache'
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            return pdb_id, 'Existing raw CIF checksum mismatch; preserve file and quarantine'
        return pdb_id, None
    for attempt in range(3):
        try:
            response = requests.get(f'https://files.rcsb.org/download/{pdb_id}.cif', timeout=(15, 90))
            response.raise_for_status()
            tmp = path.with_suffix('.partial')
            tmp.write_bytes(response.content)
            tmp.replace(path)
            receipt.write_text(json.dumps({'url': response.url,
                                           'downloaded_at_utc': datetime.now(timezone.utc).isoformat(),
                                           'sha256': hashlib.sha256(response.content).hexdigest()}, indent=2), encoding='utf-8')
            return pdb_id, None
        except requests.RequestException as exc:
            if attempt == 2: return pdb_id, f'{type(exc).__name__}: {exc}'
            time.sleep(attempt + 1)


def grouping(records, identity_cutoff=0.8):
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    component_owner = {}
    for i, record in enumerate(records):
        for component in record['homology_components']:
            if component in component_owner:
                union(i, component_owner[component])
            component_owner[component] = i
    aligner = PairwiseAligner(mode='global', match_score=2, mismatch_score=-1,
                             open_gap_score=-3, extend_gap_score=-0.5)
    identity_edges = []
    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            if min(len(a['sequence']), len(b['sequence'])) / max(len(a['sequence']), len(b['sequence'])) < identity_cutoff:
                continue
            alignment = aligner.align(a['sequence'], b['sequence'])[0]
            counts = alignment.counts()
            identity = counts.identities / (counts.identities + counts.mismatches + counts.gaps)
            if identity >= identity_cutoff:
                union(i, j)
                identity_edges.append({'a': a['id'], 'b': b['id'], 'global_identity': identity})
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[find(i)].append(record)
    return list(groups.values()), identity_edges


def assign_groups(groups, seed=42):
    """Largest-first deficit assignment; whole groups never cross splits."""
    ordered = list(groups)
    ordered.sort(key=lambda group: sorted(r['id'] for r in group))
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    result = {'train': [], 'validation': [], 'test': []}
    sizes = dict.fromkeys(result, 0)
    total = sum(map(len, ordered))
    targets = {'train': total * 0.8, 'validation': total * 0.1, 'test': total * 0.1}
    if len(ordered) < 3:
        return result
    for group in ordered:
        split = max(result, key=lambda s: targets[s] - sizes[s])
        result[split].append(group)
        sizes[split] += len(group)
    # Protect against tiny pathological scopes in which one partition remained empty.
    for split in result:
        if not result[split]:
            donor = max(result, key=lambda s: len(result[s]))
            if len(result[donor]) <= 1: return {s: [] for s in result}
            moved = min(result[donor], key=len)
            result[donor].remove(moved)
            result[split].append(moved)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qc-dir', type=Path, default=ROOT / 'data/experimental/coordinate_qc_v1')
    p.add_argument('--output', type=Path, default=ROOT / 'data/experimental/paired_v0_2')
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--min-label-coverage', type=float, default=0.5)
    p.add_argument('--limit', type=int, help='Pilot only: cap candidate chains')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'summary.json').exists():
        p.error('Dataset version already finalized; choose a new --output directory')
    qc_summary = json.loads((args.qc_dir / 'summary.json').read_text(encoding='utf-8'))
    if qc_summary['config']['limit'] is not None:
        raise SystemExit('A full coordinate archive audit is required before dataset construction')
    rows = read_jsonl(args.qc_dir / 'screen_pass_chains.jsonl')
    scope_reasons = Counter()
    selected = []
    for row in rows:
        reasons = []
        if not 20 <= row['length'] <= args.max_length: reasons.append('outside_first_version_length_scope')
        if row['rna_chain_count_in_entry'] != 1: reasons.append('multiple_RNA_chains_need_interface_audit')
        if row['contains_protein']: reasons.append('protein_context_outside_first_version_scope')
        if row['contains_dna_or_hybrid']: reasons.append('DNA_or_hybrid_context_outside_scope')
        if row['complete_base_coverage'] != 1: reasons.append('partial_coordinates_outside_first_version_scope')
        scope_reasons.update(reasons)
        if not reasons: selected.append(row)
    selected.sort(key=lambda r: (r['resolution_A'], r['id']))
    if args.limit: selected = selected[:args.limit]
    metadata = read_jsonl(ROOT / 'data/experimental/rna3db_all_chain_metadata.jsonl')
    by_id = {r['id'].lower().split('_', 1)[0] + '_' + r['id'].split('_', 1)[1]: r for r in metadata}
    sequence_components = defaultdict(set)
    for record in metadata:
        component = record.get('homology_component')
        if component and component != 'component_0':
            sequence_components[record['sequence']].add(component)
    current_ids = set(json.loads((ROOT / 'data/reports/current_experimental_rna_pdb.json').read_text(encoding='utf-8'))['entry_ids'])
    cache = ROOT / 'data/raw/experimental_cif_v1'
    cache.mkdir(parents=True, exist_ok=True)
    download_errors = []

    ids = sorted({r['pdb_id'] for r in selected})
    print(f'First-version scope: {len(selected)} chains in {len(ids)} experimental entries; fetching authoritative CIFs', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for pdb_id, error in pool.map(lambda pdb_id: download_authoritative_cif(pdb_id, cache), ids):
            if error: download_errors.append({'pdb_id': pdb_id, 'error': error})
    annotated, rejected = [], []
    for index, old in enumerate(selected, 1):
        path = cache / (old['pdb_id'].lower() + '.cif')
        if not path.exists(): continue
        try:
            data = path.read_bytes()
            entry, fresh = parse_coordinate_entry(data, path.name, current_ids)
            record = next((r for r in fresh if r['label_asym_id'] == old['label_asym_id']), None)
            if record is None or not record['coordinate_screen_pass'] or record['complete_base_coverage'] != 1:
                raise ValueError('Authoritative current coordinates no longer pass screening')
            if not 20 <= record['length'] <= args.max_length:
                raise ValueError('Authoritative sequence outside first-version length scope')
            if record['rna_chain_count_in_entry'] != 1 or record['contains_protein'] or record['contains_dna_or_hybrid']:
                raise ValueError('Authoritative context differs from first-version scope')
            annotation = annotate_chain(data, record)
            if annotation['annotation_reasons']:
                raise ValueError('Annotation: ' + ','.join(annotation['annotation_reasons']))
            components = set(sequence_components[record['sequence']])
            source_key = record['pdb_id'].lower() + '_' + str(record['auth_asym_id'])
            source = by_id.get(source_key)
            if source and source['sequence'] == record['sequence'] and source.get('homology_component') not in (None, 'component_0'):
                components.add(source['homology_component'])
            block = gemmi.cif.read_string(data.decode('utf-8')).sole_block()
            record.update(annotation)
            record.update({'evidence_class': 'experimental_coordinate_annotation',
                           'source_url': f'https://files.rcsb.org/download/{record["pdb_id"]}.cif',
                           'provenance_verified': True, 'quality_verified': True,
                           'pair_labels_available': True, 'contains_model_predictions': False,
                           'homology_components': sorted(components),
                           'nonpolymer_components': values(block, '_pdbx_entity_nonpoly.comp_id'),
                           'experimental_condition_text': values(block, '_exptl_crystal_grow.details'),
                           'conditions': {'pH_raw': values(block, '_exptl_crystal_grow.pH'),
                                          'temperature_K_raw': values(block, '_exptl_crystal_grow.temp')},
                           'quality_protocol': 'coordinate-qc-v1 + complete unmodified bases + single-RNA/no-protein scope + strict canonical annotation'})
            entity_ids = values(block, '_entity.id')
            entity_sources = dict(zip(entity_ids, values(block, '_entity.src_method')))
            entity_descriptions = dict(zip(entity_ids, values(block, '_entity.pdbx_description')))
            record['construct_provenance'] = {'entity_src_method': entity_sources.get(record['entity_id']),
                                               'description': entity_descriptions.get(record['entity_id']),
                                               'designed_status': 'unknown; chemical synthesis is not proof of de novo design'}
            record['source_download_receipt'] = json.loads(path.with_suffix('.receipt.json').read_text(encoding='utf-8'))
            annotated.append(record)
        except Exception as exc:
            rejected.append({'id': old['id'], 'reason': f'{type(exc).__name__}: {exc}'})
        if index % 25 == 0 or index == len(selected):
            print(f'Annotated {index}/{len(selected)}; accepted records {len(annotated)}; rejected {len(rejected)}', flush=True)
    save_jsonl(args.output / 'all_annotated_chains.jsonl', annotated)
    save_jsonl(args.output / 'annotation_rejected.jsonl', rejected)
    by_sequence = defaultdict(list)
    for record in annotated: by_sequence[record['sequence']].append(record)
    deduped, conflicts, unknown = [], [], []
    for seq, variants in by_sequence.items():
        mask = [all(r['observed_mask'][i] for r in variants) for i in range(len(seq))]
        signatures = {tuple(tuple(p) for p in r['pairs'] if mask[p[0]] and mask[p[1]]) for r in variants}
        if len(signatures) != 1:
            conflicts.append({'sequence_sha256': variants[0]['sequence_sha256'], 'ids': [r['id'] for r in variants],
                              'reason': 'same_sequence_different_canonical_topology_on_shared_observed_positions'})
            continue
        record = dict(min(variants, key=lambda r: (r['resolution_A'], r['id'])))
        record['alternative_structure_ids'] = [r['id'] for r in variants]
        record['observed_mask'] = mask
        record['pairs'] = [list(p) for p in next(iter(signatures))]
        record['homology_components'] = sorted({c for r in variants for c in r['homology_components']})
        if not record['pairs'] or sum(mask) / len(mask) < args.min_label_coverage:
            conflicts.append({'sequence_sha256': record['sequence_sha256'], 'ids': record['alternative_structure_ids'],
                              'reason': 'insufficient_unambiguous_shared_labels'})
        elif not record['homology_components']:
            unknown.append(record)
        else:
            deduped.append(record)
    groups, edges = grouping(deduped)
    splits = {'train': [], 'validation': [], 'test': []}
    for split, allocated in assign_groups(groups).items():
        for group in allocated:
            group_hash = hashlib.sha256('\n'.join(sorted(r['id'] for r in group)).encode()).hexdigest()
            for record in group:
                record['split_group_id'] = group_hash
                record['split'] = split
                splits[split].append(record)
    for split, records in splits.items():
        save_jsonl(args.output / f'{split}.jsonl', records)
        (args.output / f'{split}_sequences.fasta').write_text(''.join(f'>{r["id"]}\n{r["sequence"]}\n' for r in records), encoding='utf-8')
    save_jsonl(args.output / 'unknown_homology_NOT_FOR_TRAINING.jsonl', unknown)
    save_jsonl(args.output / 'sequence_conflicts.jsonl', conflicts)
    save_jsonl(args.output / 'similarity_edges.jsonl', edges)
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'unmodified single RNA, no protein/DNA, full base coverage, length20-'+str(args.max_length),
              'pilot_limit': args.limit, 'source_qc_sha256': hashlib.sha256((args.qc_dir / 'summary.json').read_bytes()).hexdigest(),
              'min_label_coverage': args.min_label_coverage,
              'scope_exclusion_counts': dict(scope_reasons), 'candidate_chains': len(selected),
              'authoritative_download_errors': download_errors, 'annotated_chains': len(annotated),
              'annotation_rejected': len(rejected), 'conflicting_sequences': len(conflicts),
              'unknown_homology_sequences': len(unknown), 'deduped_known_homology_sequences': len(deduped),
              'connected_homology_similarity_groups': len(groups), 'identity_threshold': 0.8,
              'split_protocol': 'largest-group-first target-sample-deficit, 80/10/10, seed42',
              'splits': {s: {'samples': len(rs), 'groups': len({r['split_group_id'] for r in rs}),
                            'with_crossing_pairs': sum(bool(crossing_pairs(r['pairs'])) for r in rs),
                            'sha256': hashlib.sha256((args.output / f'{s}.jsonl').read_bytes()).hexdigest()} for s, rs in splits.items()},
              'training_ready': args.limit is None and all(splits.values()),
              'benchmark_ready': False,
              'limitations': ['Small conservative scope, not a universal RNA dataset or SOTA benchmark.',
                              'Single published geometry annotator; no independent multi-tool consensus yet.',
                              'Global-resolution and atom-completeness QC do not validate local electron density.',
                              'Canonical WCc/GUc target only; unsupported contact endpoints masked.',
                              'Known RNA3DB components plus >=80% global identity; not proof of absence of all remote structural homology.',
                              'Unknown homology and conflicting same-sequence structures not used for training.']}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
