"""Freeze deduplicated, evidence-screened counts and export a local release.

No train/test assignment here. No probing signals are converted to pair labels.
"""
from collections import Counter,defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
from acquire_experimental_corpus import ROOT,now,write_json,digest_file


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    source=ROOT/'data/experimental/verified_coordinate_v1'
    completed=read(source/'completed.json')
    probing=ROOT/'data/experimental/probing_clean_v2'
    probe_summary=read(probing/'summary.json')
    source_review=read(probing/'source_header_review.json')
    if source_review['potential_prediction_or_simulation_mentions']:
        raise ValueError('Probing source headers require manual classification before release')
    rasp=read(ROOT/'data/experimental/rasp_quality_v1/summary.json')
    out=ROOT/'data/experimental/clean_release_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists(): raise ValueError('Release already frozen; choose a new version')
    # Resolve official receipts, including decompressed-CIF provenance.
    receipts={}
    for r in read(ROOT/'data/reports/experimental_acquisition_v1_all.json')['results']:
        if r.get('kind')=='pdb_cif_gz' and r['status'].startswith('verified'): receipts[r['id'].split(':')[-1]]=r
    for path in (ROOT/'data/raw/experimental_cif_v1').glob('*.receipt.json'):
        r=read(path);receipts[path.name.split('.')[0].upper()]=r
    for r in read(source/'source_verification_report.json')['results']:
        if r['status'].startswith('verified'): receipts[r['id'].split(':')[-1]]=r
    one={};two=defaultdict(list);counts=Counter();errors=[];context_counts=Counter()
    with (source/'official_coordinate_audit.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['error']: errors.append(dict(source_path=row['path'],error=row['error'],stage='official_parse'))
    spool=source/'fine_entry_results.jsonl'
    with spool.open(encoding='utf-8') as f,(out/'coordinate_sequence_records.jsonl').open('w',encoding='utf-8') as seqout:
        for line in f:
            result=json.loads(line);pdb=result['pdb_id']
            if result['error']: errors.append(dict(pdb_id=pdb,error=result['error']))
            if result['one_d'] and pdb not in receipts: raise ValueError('No authoritative source receipt')
            for row in result['one_d']:
                if row['source_kind']!='official_cif': raise ValueError('Archive-only sequence not approved')
                row['source_download_receipt']=receipts[pdb]
                row['provenance_verified']=True;row['sequence_quality_verified']=True
                one.setdefault(row['sequence_sha256'],row)
                seqout.write(json.dumps(row,allow_nan=False)+'\n');counts['coordinate_1d_records']+=1
            for row in result['two_d']:
                if row['source_kind']!='official_cif': raise ValueError('Archive-only pair labels not approved')
                row['source_download_receipt']=receipts[pdb]
                two[row['sequence_sha256']].append(row)
            counts.update('coordinate:'+a['reason'] for a in result['audit'])
    accepted=[];conflicts=[]
    for h,variants in two.items():
        variants.sort(key=lambda row:(row['id'],row['cif_sha256']))
        shared=[all(v['observed_mask'][i] for v in variants) for i in range(len(variants[0]['sequence']))]
        signatures={tuple(tuple(p) for p in v['pairs'] if shared[p[0]] and shared[p[1]]) for v in variants}
        if len(signatures)!=1 or sum(shared)/len(shared)<0.5 or not next(iter(signatures)):
            conflicts.append(dict(sequence_sha256=h,ids=[v['id'] for v in variants],reason='conformer_or_context_conflict_or_shared_coverage'))
            continue
        # Prefer an observed isolated-RNA context when the same sequence and
        # consensus labels are also represented in a complex.
        record=dict(min(variants,key=lambda v:(v['context']!='single_RNA',v['resolution_A'],v['id'])))
        record.update(observed_mask=shared,pairs=[list(p) for p in next(iter(signatures))],
                      alternative_structure_ids=[v['id'] for v in variants],
                      context_classes=sorted({v['context'] for v in variants}),training_split_ready=False)
        accepted.append(record);context_counts[record['context']]+=1
    accepted.sort(key=lambda r:r['sequence_sha256'])
    pairs_path=out/'paired_deduplicated.jsonl'
    with pairs_path.open('w',encoding='utf-8') as f:
        for r in accepted: f.write(json.dumps(r,allow_nan=False)+'\n')
    write_json(out/'pair_sequence_conflicts.json',conflicts)
    database=sqlite3.connect(out/'sequences.sqlite3')
    database.execute('CREATE TABLE IF NOT EXISTS sequences(sha256 TEXT PRIMARY KEY,sequence TEXT,coordinate_evidence INTEGER,probing_evidence INTEGER)')
    database.execute('DELETE FROM sequences')
    database.execute('ATTACH DATABASE ? AS probing',(str(probing/'profiles.sqlite3'),))
    database.execute('INSERT INTO sequences SELECT sha256,sequence,0,1 FROM probing.sequences')
    for h,r in one.items():
        database.execute('INSERT INTO sequences VALUES (?,?,1,0) ON CONFLICT(sha256) DO UPDATE SET coordinate_evidence=1',(h,r['sequence']))
    database.commit()
    total=database.execute('SELECT COUNT(*) FROM sequences').fetchone()[0]
    overlap=database.execute('SELECT COUNT(*) FROM sequences WHERE coordinate_evidence=1 AND probing_evidence=1').fetchone()[0]
    with (out/'sequences.fasta').open('w',encoding='utf-8') as f:
        for h,seq in database.execute('SELECT sha256,sequence FROM sequences ORDER BY sha256'):
            f.write('>'+h+'\n'+seq+'\n')
    database.close()
    summary=dict(created_at_utc=now(),protocol='clean-release-v1',
                 initial_coordinate_entries_screened=read(ROOT/'data/experimental/fine_qc_v1/coordinate_audit_summary.json')['entries'],
                 official_candidate_entries_rechecked=completed['processed_official_entries'],
                 unique_1d_sequences=total,unique_2d_sequences=len(accepted),
                 coordinate_unique_1d_sequences=len(one),probing_unique_1d_sequences=probe_summary['unique_sequences'],
                 coordinate_probing_sequence_overlap=overlap,accepted_probing_profiles=probe_summary['measured_profiles'],
                 paired_context_distribution=dict(context_counts),paired_conflicting_sequences_quarantined=len(conflicts),
                 coordinate_counts=dict(counts),coordinate_errors=errors,
                 official_coordinate_download_failures=completed['download_failures'],
                 rasp_tracks_audited=rasp['tracks_audited'],rasp_training_approved_sequences=0,
                 comparative_training_approved_sequences=0,
                 training_split_ready=False,model_training_performed=False,
                 tool_versions={name:version(name) for name in ('gemmi','barnaba','mdtraj','numpy','scipy')},
                 accepted_probing_source_files=source_review['accepted_source_files'],
                 protocol_fingerprints={str(p.relative_to(ROOT)):digest_file(p) for p in
                     [ROOT/'src/rna2d/fine_qc.py', ROOT/'src/rna2d/probing_qc.py',
                      ROOT/'src/rna2d/annotation.py',ROOT/'src/rna2d/coordinate_qc.py',
                      ROOT/'scripts/clean_coordinate_pairs.py',ROOT/'scripts/clean_probing_corpus.py',
                      source/'source_verification_report.json',ROOT/'data/experimental/fr3d_index_v1/summary.json',
                      probing/'source_header_review.json']},
                 fingerprints={p.name:digest_file(p) for p in [pairs_path,out/'sequences.fasta',spool,probing/'config.json']},
                 limitations=['Unique means exact complete sequence strings, not independent families; library mutants and flanks are retained.',
                              '1D is experimentally associated sequence; measured reactivity is not an exact partner label.',
                              '2D is canonical AU/GC/GU consensus with unknown/interface labels masked, length20-512; not all noncanonical RNA interactions.',
                              'RNA complexes are explicitly contextual; they are not evidence of isolated-RNA folding under unspecified conditions.',
                              'Automated strict QC is completed under recorded thresholds; unresolved/unsupported evidence is quarantined, not silently approved.',
                              'NMR, modified bases, source/version gaps, RASP assay/replicate evidence and comparative alignments remain outside the approved pool.',
                              'Homology/study/construct-controlled split and leakage audit are required before training; no new training-ready split is claimed.'])
    write_json(out/'summary.json',summary)
    write_json(ROOT/'data/manifests/clean_release_v1.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
