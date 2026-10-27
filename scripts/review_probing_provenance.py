"""Check accepted RMDB source headers for declared simulated/imputed signals.

The curated source and actual assay/read/SNR evidence remain primary; keyword
absence is not a substitute for a full original-paper audit.
"""
import json
import re
import sqlite3
from acquire_experimental_corpus import ROOT,write_json,digest_file


def main():
    directory=ROOT/'data/experimental/probing_clean_v2'
    database=sqlite3.connect(directory/'profiles.sqlite3')
    files=sorted(row[0].split(':',1)[1] for row in database.execute('SELECT DISTINCT file FROM profiles'))
    suspects=[]
    for name in files:
        path=ROOT/'data/raw/experimental_acquisition_v1/rmdb'/name
        with path.open('rb') as stream:
            for line in stream:
                if line.startswith(b'ANNOTATION_DATA:'): break
                if line.startswith((b'COMMENT',b'ANNOTATION')) and re.search(rb'simulat|imput|in.silico|predict',line,re.I):
                    suspects.append(dict(file=name,header=line.decode('latin1').strip()))
    report=dict(accepted_source_files=len(files),potential_prediction_or_simulation_mentions=suspects,
                script_sha256=digest_file(ROOT/'scripts/review_probing_provenance.py'),
                scope='Construct COMMENT/ANNOTATION headers before per-profile data; metadata review, not original-paper audit.')
    write_json(directory/'source_header_review.json',report)
    write_json(ROOT/'data/manifests/probing_source_header_review_v1.json',report)
    print('Source files reviewed:',len(files),'mentions needing review:',len(suspects))
    if suspects: raise SystemExit('Manual evidence classification required before release')


if __name__=='__main__': main()
