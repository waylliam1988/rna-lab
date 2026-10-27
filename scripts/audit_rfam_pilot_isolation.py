"""Sequence/metadata relations only, before selecting any representatives."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from acquire_experimental_corpus import ROOT, digest_file, write_json
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.comparative_alignment import read_stockholm, ungap


def sequence_key(sequence):
    sequence = sequence.upper().replace('T', 'U')
    return hashlib.sha256(sequence.encode()).hexdigest() if sequence and not set(sequence)-set('ACGU') else None


def relation_decision(source_supported, holdout_evidence, evidence_complete=False):
    if holdout_evidence:
        return 'blocked_by_current_or_historical_evaluation_family_relation'
    if not source_supported:
        return 'source_provenance_unresolved'
    if not evidence_complete:
        return 'no_positive_holdout_witness_but_isolation_unresolved'
    return 'requires_label_scope_review'


def main():
    output = ROOT / 'data/manifests/rfam_family_pilot_isolation_v1.json'
    local = ROOT / 'data/reports/rfam_family_pilot_v1/isolation_witnesses.jsonl'
    if output.exists() or local.exists():
        raise FileExistsError('Isolation audit already frozen')
    sources_path = ROOT / 'data/manifests/rfam_family_pilot_sources_v1.json'
    source_review = json.loads(sources_path.read_text())
    family_members = {}
    for family in source_review['families']:
        path = ROOT / 'data/raw/rfam_family_pilot_v1' / (family['family']+'.sto')
        assert digest_file(path) == family['seed_sha256']
        d = read_stockholm(path)
        members = defaultdict(list)
        for accession, aligned in d['sequences'].items():
            key = sequence_key(ungap(aligned))
            if key:
                members[key].append(accession)
        family_members[family['family']] = members
    paths = {}
    for dataset in ['joint_split_v2', 'paired_v0_2']:
        for split in ['train', 'validation', 'test']:
            paths[dataset+':'+split] = ROOT / 'data/experimental' / dataset / (split+'.jsonl')
    for split in ['train', 'validation', 'test', 'test_external', 'archiveII']:
        path = ROOT / 'data/processed' / (split+'.jsonl')
        if path.exists():
            paths['legacy_processed:'+split] = path
    metadata_path = ROOT / 'data/experimental/rna3db_all_chain_metadata.jsonl'
    metadata = [json.loads(line) for line in metadata_path.open() if line.strip()]
    by_key = defaultdict(list)
    for row in metadata:
        key = sequence_key(row['sequence'])
        if key:
            by_key[key].append(row)
    dataset_rows = {name:[json.loads(line) for line in path.open() if line.strip()] for name,path in paths.items()}
    witnesses, summaries = [], []
    for family in source_review['families']:
        acc = family['family']
        members = family_members[acc]
        components = set()
        for key in members:
            for row in by_key.get(key, []):
                comp = row.get('homology_component')
                if comp and comp != 'component_0':
                    components.add(comp)
                    witnesses.append(dict(family=acc, type='seed_exact_to_rna3db_component', seed_accessions=members[key], rna3db_id=row['id'], component=comp, sequence_sha256=key))
        exact_counts, component_counts = {}, {}
        evaluation_witness = False
        for name,rows in dataset_rows.items():
            exact = connected = 0
            is_evaluation = name.split(':')[1] in ['validation','test','test_external','archiveII']
            for row in rows:
                key = sequence_key(row['sequence'])
                shared = components.intersection(row.get('homology_components', []))
                if key in members:
                    exact += 1
                    witnesses.append(dict(family=acc,type='seed_exact_to_frozen_or_historical_sequence',dataset=name,record_id=row['id'],sequence_sha256=key,seed_accessions=members[key]))
                if shared:
                    connected += 1
                    witnesses.append(dict(family=acc,type='shared_declared_rna3db_component',dataset=name,record_id=row['id'],components=sorted(shared)))
                if is_evaluation and (key in members or shared):
                    evaluation_witness = True
            exact_counts[name] = exact
            component_counts[name] = connected
        decision = relation_decision(family['review']['provenance_supported'], evaluation_witness)
        summaries.append(dict(family=acc,clan=family['clan'],canonical_seed_sequences=len(members),exact_sequence_overlap_counts=exact_counts,declared_component_overlap_counts=component_counts,positive_evaluation_relation_found=evaluation_witness,complete_isolation_certified=False,decision=decision,representative_selection_allowed=False,auxiliary_training_approved=False))
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(''.join(json.dumps(row)+'\n' for row in witnesses),encoding='utf-8',newline='\n')
    report = dict(stage='2_isolation_review',families=summaries,eligible_families_for_representatives=0,local_witness_rows=len(witnesses),local_witness_file_sha256=digest_file(local),source_review_sha256=digest_file(sources_path),metadata_sha256=digest_file(metadata_path),dataset_files_sha256={k:digest_file(v) for k,v in paths.items()},dataset_record_counts={k:len(v) for k,v in dataset_rows.items()},scope='Only existing sequences and declared homology metadata; no label comparison, model predictions, or test performance.',limitations=['An exact seed member in an evaluation dataset proves a family-level connection; absent exact matches do not prove independence.', 'Existing RNA3DB components are inherited relation screens, not proof of all remote homology or construct relationships.', 'Historical evaluation exposure is a conservative exclusion under this pilot; it is not a permanent claim that these families can never be used in a differently designed future study.', 'No representatives selected before isolation; missing or unresolved evidence cannot be promoted to independent.'],source_script_sha256=digest_file(Path(__file__)))
    write_json(output, report)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
