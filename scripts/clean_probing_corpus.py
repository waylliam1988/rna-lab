"""Versioned full RMDB strict screening; preserve every exclusion reason."""
from collections import Counter
import gzip
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zlib
from acquire_experimental_corpus import ROOT, now, write_json, digest_file

sys.path.insert(0,str(ROOT/'src'))
from rna2d.probing_qc import sequence_mapping,quality_metadata,measured_profile


def lines(path):
    with path.open('rb') as stream:
        for line in stream:
            parts=line.decode('latin-1').split()
            if parts: yield parts[0],parts[1:]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'data/experimental/probing_clean_v2')
    out=parser.parse_args().output
    out.mkdir(parents=True,exist_ok=True)
    database=sqlite3.connect(out/'profiles.sqlite3')
    database.executescript('''PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS files(id TEXT PRIMARY KEY,sha256 TEXT,annotations TEXT,counts TEXT);
    CREATE TABLE IF NOT EXISTS sequences(sha256 TEXT PRIMARY KEY,sequence TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS profiles(file TEXT,profile TEXT,seqhash TEXT,seqpos TEXT,reactivity BLOB,uncertainty BLOB,mask BLOB,annotations TEXT,PRIMARY KEY(file,profile));''')
    config=dict(protocol='strict-rdat-v1',snr_min=3,reads_min_if_reported=1000,min_observed=20,
                min_reported_position_coverage=0.8,require_errors=True,no_pair_labels=True,
                module_sha256=digest_file(ROOT/'src/rna2d/probing_qc.py'))
    config['script_sha256']=digest_file(Path(__file__))
    config_path=out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config: raise ValueError('Protocol changed; choose new version')
    write_json(config_path,config)
    report=json.loads((ROOT/'data/reports/experimental_acquisition_v1_all.json').read_text(encoding='utf-8'))
    for receipt in sorted(report['results'],key=lambda r:r['id']):
        if receipt.get('kind')!='rdat' or not receipt['status'].startswith('verified'): continue
        fid=receipt['id']
        previous=database.execute('SELECT sha256 FROM files WHERE id=?',(fid,)).fetchone()
        if previous:
            if previous[0]!=receipt['sha256']: raise ValueError('Source changed')
            continue
        path=ROOT/receipt['path']
        if digest_file(path)!=receipt['sha256']: raise ValueError('Raw RDAT checksum differs')
        fields={}; annotations=[]; profiles={}; errors={}; duplicates=set(); counts=Counter(); value_ids=set()
        for key,parts in lines(path):
            if key in ('SEQUENCE','OFFSET','SEQPOS'): fields[key]=parts
            elif key=='ANNOTATION': annotations.extend(parts)
            elif key.startswith('ANNOTATION_DATA:'):
                ident=key.split(':',1)[1]
                if ident in profiles: duplicates.add(ident)
                profiles[ident]=parts
            elif key.startswith('REACTIVITY_ERROR:'):
                ident=key.split(':',1)[1]
                if ident in errors: duplicates.add(ident)
                errors[ident]=' '.join(parts)
            elif key.startswith('REACTIVITY:'):
                ident=key.split(':',1)[1]
                if ident in value_ids: duplicates.add(ident)
                value_ids.add(ident)
        try: offset=int(fields.get('OFFSET',['0'])[0])
        except ValueError: offset=None
        seen=set()
        rejected=out/(path.stem+'.rejected.jsonl.gz')
        with gzip.open(rejected,'wt',encoding='utf-8') as audit:
            for key,parts in lines(path):
                if not key.startswith('REACTIVITY:'): continue
                ident=key.split(':',1)[1]; counts['profiles_seen']+=1
                meta=profiles.get(ident,[])
                reasons,modifiers=quality_metadata(meta,annotations)
                if ident not in profiles: reasons.append('missing_profile_annotations')
                if ident in duplicates or ident in seen: reasons.append('duplicate_profile_identifier')
                seen.add(ident)
                if offset is None: reasons.append('invalid_offset')
                if not reasons:
                    try:
                        seq,indices=sequence_mapping(''.join(fields.get('SEQUENCE',[])),meta,offset,fields.get('SEQPOS',[]))
                        if not 20<=len(seq)<=10000: raise ValueError('sequence_length_outside20_10000')
                        values,uncertainty,mask=measured_profile(' '.join(parts),errors.get(ident,''),len(indices),modifiers)
                    except (ValueError,TypeError) as exc: reasons.append(str(exc))
                if reasons:
                    counts.update('exclude:'+r for r in set(reasons))
                    audit.write(json.dumps(dict(profile=ident,reasons=sorted(set(reasons))))+'\n')
                    continue
                h=hashlib.sha256(seq.encode()).hexdigest()
                database.execute('INSERT OR IGNORE INTO sequences VALUES (?,?)',(h,seq))
                database.execute('INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?)',
                                 (fid,ident,h,json.dumps(indices),zlib.compress(values.tobytes()),
                                  zlib.compress(uncertainty.tobytes()),zlib.compress(mask.tobytes()),json.dumps(meta)))
                counts['accepted_measured_profiles']+=1
        database.execute('INSERT INTO files VALUES (?,?,?,?)',(fid,receipt['sha256'],json.dumps(annotations),json.dumps(dict(counts))))
        database.commit()
        if database.execute('SELECT COUNT(*) FROM files').fetchone()[0]%50==0: print('Probing files screened',database.execute('SELECT COUNT(*) FROM files').fetchone()[0],flush=True)
    totals=Counter()
    for (text,) in database.execute('SELECT counts FROM files'): totals.update(json.loads(text))
    summary=dict(created_at_utc=now(),files=database.execute('SELECT COUNT(*) FROM files').fetchone()[0],
                 unique_sequences=database.execute('SELECT COUNT(*) FROM sequences').fetchone()[0],
                 measured_profiles=database.execute('SELECT COUNT(*) FROM profiles').fetchone()[0],counts=dict(totals),
                 pair_labels=0,training_split_ready=False,
                 note='Strict machine-screened experimental reactivity pool; no imputation, no structure-field truth. Biological replicates, study/family split and cross-source dedup remain required.')
    write_json(out/'summary.json',summary)
    database.close()
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
