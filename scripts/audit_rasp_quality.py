"""Check original RASP containers, references, measured fields and coordinates.

Passing format/mapping checks is not assay-quality approval. Lack of read-depth,
uncertainty and replicate controls is reported as a hold, never filled in.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import math
from pathlib import Path
import tarfile
from acquire_experimental_corpus import ROOT,now,digest_file,write_json

_LENGTHS=None


def reference_lengths():
    report=json.loads((ROOT/'data/reports/rasp_reference_acquisition_v1.json').read_text())
    lengths={}
    for r in report['results']:
        if not r['status'].startswith('verified') or not r['id'].endswith('.fa.gz'): continue
        path=ROOT/r['path'];species=path.name.removesuffix('.fa.gz');chrom=None;mapping={}
        with gzip.open(path,'rb') as stream:
            for line in stream:
                if line.startswith(b'>'):
                    chrom=line[1:].split()[0].decode('ascii')
                    if chrom in mapping: raise ValueError('Duplicate reference sequence identifier')
                    mapping[chrom]=0
                elif line.strip():
                    if chrom is None: raise ValueError('Reference sequence missing header')
                    mapping[chrom]+=len(line.strip())
        lengths[species]=mapping
    return lengths


def initialize(lengths):
    global _LENGTHS
    _LENGTHS=lengths


def process(item):
    job,receipt=item;counts=Counter();errors=[]
    path=ROOT/receipt['path']
    rows=job.get('source_track_rows') or [job['source_track']]
    species=rows[0]['species'];kind='contact' if job['id'].startswith('rasp-contacts:') else 'score'
    lengths=_LENGTHS.get(species,{})
    if digest_file(path)!=receipt['sha256']: raise ValueError('Raw artifact checksum mismatch')
    try:
        with tarfile.open(path) as archive:
            for member in archive:
                if not member.isfile(): continue
                stream=archive.extractfile(member)
                if member.name.endswith('.gz'): stream=gzip.GzipFile(fileobj=stream)
                for line in stream:
                    if not line.strip() or line.startswith((b'#',b'track',b'browser')): continue
                    counts['data_rows']+=1
                    parts=line.split()
                    try:
                        if len(parts)<(6 if kind=='contact' else 4): raise ValueError('too_few_fields')
                        intervals=[(parts[0],parts[1],parts[2])]
                        if kind=='contact': intervals.append((parts[3],parts[4],parts[5]))
                        for chrom,start,end in intervals:
                            chrom=chrom.decode('ascii');start=int(start);end=int(end)
                            if chrom not in lengths: raise ValueError('reference_contig_unresolved')
                            if not 0<=start<end<=lengths[chrom]: raise ValueError('reference_interval_out_of_bounds')
                        if kind=='score':
                            token=parts[4] if len(parts)>=6 else parts[3]
                            value=float(token)
                            if not math.isfinite(value) or value<=-100: raise ValueError('missing_or_nonfinite_score')
                            if value<0: counts['negative_scores_require_assay_interpretation']+=1
                        counts['format_and_reference_valid_rows']+=1
                    except (ValueError,UnicodeError) as exc:
                        counts['exclude:'+str(exc)]+=1
    except Exception as exc: errors.append(str(exc))
    return dict(id=job['id'],sha256=receipt['sha256'],species=species,kind=kind,counts=dict(counts),errors=errors,
                quality_status='hold_for_assay_depth_uncertainty_replicate_and_transcript_mapping_review',
                approved_training_sequences=0,exact_pair_labels=0)


def main():
    out=ROOT/'data/experimental/rasp_quality_v1';out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'data/raw/additional_experimental_acquisition_v1/plan.json').read_text())
    report=json.loads((ROOT/'data/reports/additional_experimental_acquisition_v1.json').read_text())
    byid={j['id']:j for j in plan['jobs']};items=[]
    for r in report['results']:
        if r['status'].startswith('verified') and r['id'].startswith(('rasp-scores:','rasp-contacts:')):
            items.append((byid[r['id']],r))
    lengths_path=out/'reference_contig_lengths.json'
    if lengths_path.exists(): lengths=json.loads(lengths_path.read_text())
    else: lengths=reference_lengths();write_json(lengths_path,lengths)
    spool=out/'track_audit.jsonl';done=set()
    if spool.exists():
        with spool.open() as f: done={json.loads(s)['id'] for s in f}
    with spool.open('a',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=2,initializer=initialize,initargs=(lengths,)) as pool:
        for i,row in enumerate(pool.map(process,[item for item in items if item[0]['id'] not in done]),1):
            f.write(json.dumps(row)+'\n');f.flush()
            if i%25==0: print('RASP tracks audited',i,flush=True)
    totals=Counter();errors=[];n=0
    with spool.open() as f:
        for line in f:
            r=json.loads(line);n+=1;totals.update(r['counts'])
            if r['errors']: errors.append(dict(id=r['id'],errors=r['errors']))
    write_json(out/'summary.json',dict(created_at_utc=now(),tracks_audited=n,counts=dict(totals),errors=errors,
                approved_training_sequences=0,exact_pair_labels=0,
                note='Original measured genomic tracks retained with source provenance; numeric intervals do not establish assay quality, transcript identity or exact base-pair labels. BigWig alternatives kept separately pending binary-track audit.'))
    print('RASP structural audit finished',n,flush=True)


if __name__=='__main__': main()
