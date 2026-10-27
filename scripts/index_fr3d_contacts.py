"""Index coordinate-derived base-pair annotations, including cross-chain context."""
import csv
import gzip
import json
import sqlite3
from acquire_experimental_corpus import ROOT,now,digest_file,write_json


def main():
    out=ROOT/'data/experimental/fr3d_index_v1'
    out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data/raw/additional_experimental_acquisition_v1/fr3d/interactions.csv.gz'
    receipt=json.loads(path.with_suffix(path.suffix+'.receipt.json').read_text())
    digest=digest_file(path)
    if digest!=receipt['sha256']: raise ValueError('FR3D download hash mismatch')
    database=sqlite3.connect(out/'contacts.sqlite3')
    database.execute('CREATE TABLE IF NOT EXISTS metadata(sha256 TEXT PRIMARY KEY)')
    previous=database.execute('SELECT sha256 FROM metadata').fetchone()
    if previous:
        if previous[0]!=digest: raise ValueError('Source changed')
        return
    database.execute('CREATE TABLE IF NOT EXISTS contacts(pdb TEXT,u1 TEXT,u2 TEXT,code TEXT)')
    database.execute('DELETE FROM contacts')
    current={r['pdb_id'] for r in json.loads((ROOT/'data/raw/experimental_acquisition_v1/plan.json').read_text())['pdb_coverage']}
    n=0;batch=[]
    with gzip.open(path,'rt',encoding='utf-8',newline='') as stream:
        reader=csv.reader(stream);next(reader)
        for row in reader:
            if len(row)<3 or not row[2]: continue
            pdb=row[0].split('|',1)[0].upper()
            if pdb not in current: continue
            batch.append((pdb,row[0],row[1],row[2]));n+=1
            if len(batch)>=10000:
                database.executemany('INSERT INTO contacts VALUES (?,?,?,?)',batch);batch=[]
    database.executemany('INSERT INTO contacts VALUES (?,?,?,?)',batch)
    database.execute('CREATE INDEX IF NOT EXISTS contacts_pdb ON contacts(pdb)')
    database.execute('INSERT INTO metadata VALUES (?)',(digest,));database.commit()
    count=database.execute('SELECT COUNT(DISTINCT pdb) FROM contacts').fetchone()[0]
    write_json(out/'summary.json',dict(created_at_utc=now(),contact_rows=n,pdb_entries=count,
                sha256=digest,source_last_modified=receipt['source_last_modified'],
                note='FR3D geometric annotations of experimental coordinates; no sequence-to-structure predictions. Historical upstream snapshot; not full current PDB coverage.'))
    database.close();print('FR3D contacts indexed',n,count,flush=True)


if __name__=='__main__': main()
