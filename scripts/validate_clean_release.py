"""Independent checks on exported sequence/pair evidence and measured arrays."""
import hashlib
import json
import sqlite3
import sys
import zlib
import numpy as np
from acquire_experimental_corpus import ROOT,write_json,now,digest_file

sys.path.insert(0,str(ROOT/'src'))
from rna2d.data import CANONICAL
from rna2d.evidence import pairing_evidence_issues


def main():
    directory=ROOT/'data/experimental/clean_release_v1'
    summary=json.loads((directory/'summary.json').read_text())
    sequences=sqlite3.connect(directory/'sequences.sqlite3')
    count=0;seen=set()
    with (directory/'paired_deduplicated.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);seq=r['sequence'];h=hashlib.sha256(seq.encode()).hexdigest()
            assert h==r['sequence_sha256'] and h not in seen
            assert not pairing_evidence_issues(r)
            assert r['source_kind']=='official_cif' and r['source_download_receipt']['sha256']
            assert not r['modified_positions_0based'] and set(seq)<=set('ACGU')
            assert r['independent_annotation']['canonical_pair_jaccard']>=0.9
            assert len(r['observed_mask'])==len(seq) and sum(r['observed_mask'])>=0.5*len(seq)
            assert all(not r['observed_mask'][i] for i in r['interface_positions_masked'])
            assert sequences.execute('SELECT coordinate_evidence FROM sequences WHERE sha256=?',(h,)).fetchone()==(1,)
            partners=set()
            for i,j in r['pairs']:
                assert 0<=i<j<len(seq) and j-i>=4 and seq[i]+seq[j] in CANONICAL
                assert r['observed_mask'][i] and r['observed_mask'][j]
                assert i not in partners and j not in partners
                partners.update((i,j))
            seen.add(h);count+=1
    assert count==summary['unique_2d_sequences']
    n=0
    for h,seq in sequences.execute('SELECT sha256,sequence FROM sequences'):
        assert hashlib.sha256(seq.encode()).hexdigest()==h and set(seq)<=set('ACGU')
        n+=1
    assert n==summary['unique_1d_sequences']
    profile_db=sqlite3.connect(ROOT/'data/experimental/probing_clean_v2/profiles.sqlite3')
    profiles=0
    for file,pid,h,positions,values,errors,mask in profile_db.execute('SELECT file,profile,seqhash,seqpos,reactivity,uncertainty,mask FROM profiles'):
        seq=sequences.execute('SELECT sequence FROM sequences WHERE sha256=?',(h,)).fetchone()[0]
        positions=json.loads(positions)
        values=np.frombuffer(zlib.decompress(values),dtype=np.float32)
        errors=np.frombuffer(zlib.decompress(errors),dtype=np.float32)
        mask=np.frombuffer(zlib.decompress(mask),dtype=np.bool_)
        assert len(values)==len(errors)==len(mask)==len(positions)
        assert len(set(positions))==len(positions) and all(0<=i<len(seq) for i in positions)
        assert mask.sum()>=20 and mask.mean()>=0.8
        assert np.isfinite(values[mask]).all() and (values[mask]>-100).all()
        assert np.isfinite(errors[mask]).all() and (errors[mask]>=0).all()
        profiles+=1
    assert profiles==summary['accepted_probing_profiles']
    inputs={p.name:p for p in [directory/'paired_deduplicated.jsonl',directory/'sequences.fasta',
             ROOT/'data/experimental/verified_coordinate_v1/fine_entry_results.jsonl',
             ROOT/'data/experimental/probing_clean_v2/config.json']}
    for filename,digest in summary['fingerprints'].items():
        assert digest_file(inputs[filename])==digest
    for relative,digest in summary['protocol_fingerprints'].items():
        assert digest_file(ROOT/relative)==digest
    report=dict(created_at_utc=now(),passed=True,unique_sequences_checked=n,
                paired_sequences_checked=count,measured_profiles_checked=profiles,
                training_split_ready=False,checks=['sequence SHA256/alphabet/dedup',
                    'official coordinate source receipt and experimental evidence flags',
                    'canonical pairing bounds, masks, one-partner rule, interface exclusion',
                    'independent annotation agreement and modified-base exclusion',
                    'all measured-array dimensions, source positions, finite values/uncertainties and missing masks'],
                note='Evidence/format contracts, not accuracy measurement or proof of biological native-state correctness.')
    write_json(directory/'validation_report.json',report)
    write_json(ROOT/'data/manifests/clean_release_validation_v1.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
