"""Validate a TLS-verified curl fallback download and record its provenance.

Run only after curl exits successfully. Does not extract archives or approve labels.
"""
import gzip
import json
from pathlib import Path
from acquire_experimental_corpus import ROOT, digest_file, now, write_json


def main():
    directory = ROOT/'data/raw/additional_experimental_acquisition_v1'
    plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    job = next(j for j in plan['jobs'] if j['id']=='fr3d:all_interactions')
    target = directory/job['relative_path']
    partial = target.with_suffix(target.suffix+'.curl.partial')
    headers = (target.parent/'curl_headers.txt').read_text(encoding='utf-8')
    blocks = [b for b in headers.split('\n\n') if b.startswith('HTTP/')]
    header = blocks[-1]
    if ' 200 ' not in header.splitlines()[0]:
        raise ValueError('Expected complete HTTP 200 response')
    fields = dict(line.split(':',1) for line in header.splitlines()[1:] if ':' in line)
    fields = {k.lower():v.strip() for k,v in fields.items()}
    if partial.stat().st_size != int(fields['content-length']):
        raise ValueError('Content-Length mismatch')
    with gzip.open(partial,'rb') as stream:
        prefix = stream.read(65536)
        if b'unit_id' not in prefix: raise ValueError('Unexpected interactions CSV header')
        while stream.read(4*1024*1024): pass
    receipt = dict(id=job['id'], kind=job['kind'], url=job['url'],
                   path=str(target.relative_to(ROOT)), bytes=partial.stat().st_size,
                   sha256=digest_file(partial), upstream_sha256_verified=False,
                   source_etag=fields.get('etag'), source_last_modified=fields.get('last-modified'),
                   downloaded_at_utc=now(), status='verified_download', quality_approved=False,
                   format_note='gzip CRC and interaction header verified; upstream snapshot date retained.',
                   transport='curl with default TLS certificate verification; requests TLS EOF fallback')
    if target.exists(): raise ValueError('Refusing to overwrite existing raw artifact')
    partial.replace(target)
    write_json(target.with_suffix(target.suffix+'.receipt.json'),receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
