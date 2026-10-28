"""Versioned reference-role inventory, never a replacement training dataset."""
import json
import sys
from collections import Counter
from pathlib import Path
from acquire_experimental_corpus import ROOT, digest_file, write_json
sys.path.insert(0,str(ROOT/'src'))
from rna2d.study_roles import candidate_review, effective_reference_role


def main():
    policy_path=ROOT/'data/manifests/study_v2_role_policy_v1.json'
    output=ROOT/'data/manifests/study_v2_role_inventory_v1.json'
    ledger=ROOT/'data/reports/study_v2_roles_v1/reference_roles.jsonl'
    if output.exists() or ledger.exists():
        raise FileExistsError('Role inventory already exists')
    policy=json.loads(policy_path.read_text())
    validation_path=ROOT/'data/manifests/joint_split_v2_validation.json'
    validation=json.loads(validation_path.read_text())
    assert validation['passed'] is True
    current=ROOT/'data/experimental/joint_split_v2'
    for split,sha in validation['files'].items():
        assert digest_file(current/(split+'.jsonl'))==sha
    dev=[json.loads(line) for line in (current/'validation.jsonl').open() if line.strip()]
    active_keys={row['sequence_sha256'] for row in dev}
    paths={}
    for pool in ['joint_split_v2','paired_v0_2']:
        for split in ['train','validation','test']:
            paths[pool+':'+split]=ROOT/'data/experimental'/pool/(split+'.jsonl')
    for split in ['train','validation','test','test_external','archiveII']:
        path=ROOT/'data/processed'/(split+'.jsonl')
        if path.exists():
            paths['legacy_processed:'+split]=path
    local_rows=[];summary={}
    for name,path in paths.items():
        if name=='joint_split_v2:train': role='core_train_reference'
        elif name=='joint_split_v2:validation': role='protected_development'
        elif name in policy['data_roles']['retired_evaluation_reference']['sources']: role='retired_evaluation_reference'
        else: role='historical_source_candidate'
        effective_counts=Counter();rows=[json.loads(line) for line in path.open() if line.strip()]
        for row in rows:
            key=row['sequence_sha256'];alias=key in active_keys and name!='joint_split_v2:validation'
            effective=effective_reference_role(role,alias);effective_counts[effective]+=1
            local_rows.append(dict(source_pool=name,record_id=row['id'],sequence_sha256=key,reference_role=role,effective_role=effective,exact_active_development_alias=alias,next_study_training_approved=False,reason='Role inventory only; existing roles preserved in old files. Other active homology/study/construct relations still require review.'))
        summary[name]=dict(records=len(rows),reference_role=role,effective_roles=dict(effective_counts),source_sha256=digest_file(path))
    family_sources_path=ROOT/'data/manifests/rfam_family_pilot_sources_v1.json'
    prior_isolation_path=ROOT/'data/manifests/rfam_family_pilot_isolation_v2.json'
    family_sources=json.loads(family_sources_path.read_text())
    prior_isolation=json.loads(prior_isolation_path.read_text())
    family_reviews=[]
    for source in family_sources['families']:
        prior=next(row for row in prior_isolation['families'] if row['family']==source['family'])
        exact=sum(count for key,count in prior['exact_sequence_overlap_counts'].items() if key in ['joint_split_v2:validation'])
        component=sum(count for key,count in prior['declared_component_overlap_counts'].items() if key in ['joint_split_v2:validation'])
        # The old screen did not certify complete active clan/construct/study
        # isolation. A lack of positive hits must not silently become False.
        relation=True if exact or component else None
        review=candidate_review(provenance_verified=source['review']['provenance_supported'],quality_verified=None,mapping_verified=False,contains_model_predictions=None,supervision_scope='not_mapped',active_holdout_relation=relation,retired_evaluation_relation=True)
        family_reviews.append(dict(family=source['family'],old_history_exclusion_preserved=True,next_study_history_is_sole_exclusion=False,existing_exact_or_component_active_development_witnesses=exact+component,active_relation_status='positive' if relation else 'unresolved_not_certified',review=review))
    ledger.parent.mkdir(parents=True,exist_ok=True)
    ledger.write_text(''.join(json.dumps(row)+'\n' for row in local_rows),encoding='utf-8',newline='\n')
    report=dict(study_id=policy['study_id'],policy_sha256=digest_file(policy_path),source_validation_sha256=digest_file(validation_path),source_family_review_sha256=digest_file(family_sources_path),source_old_isolation_sha256=digest_file(prior_isolation_path),pools=summary,reference_rows=len(local_rows),unique_sequence_hashes=len({row['sequence_sha256'] for row in local_rows}),counts_are_source_rows_not_unique_rnas=True,protected_development_records=len(dev),protected_development_groups=len({row['split_group_id'] for row in dev}),new_independent_test_records=0,new_independent_test_groups=0,new_training_admissions=0,new_auxiliary_admissions=0,family_reassessment=family_reviews,local_ledger_sha256=digest_file(ledger),source_script_sha256=digest_file(Path(__file__)),legacy_cli_unchanged=True,legacy_policy_unchanged=True,training_performed=False,test_performance_read=False,limits=['Exact active aliases are protected here; complete family/clan/study/construct review is not certified by this inventory.','Retired references may be reconsidered only after quality, label-scope and active-holdout isolation gates.','No new test exists; claims remain developmental until an independent reviewed test is frozen.'])
    write_json(output,report)
    print(json.dumps(dict(reference_rows=len(local_rows),core_train_reference=88,protected_development_records=len(dev),new_test=0,new_admissions=0,active_alias_records=sum(row['exact_active_development_alias'] for row in local_rows)),indent=2))


if __name__=='__main__':
    main()
