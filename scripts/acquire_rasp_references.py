"""Acquire the reference FASTA/GFF linked by the frozen RASP download page.

Reference genomes are sequence/mapping resources, not experimental pair labels.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from acquire_experimental_corpus import ROOT, acquire, digest_file, now, write_json


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            if href.startswith('/api/downloadfile/?'): self.links.append(href)


def main():
    directory=ROOT/'data/raw/rasp_reference_acquisition_v1'
    directory.mkdir(parents=True,exist_ok=True)
    plan_path=directory/'plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
    else:
        page=ROOT/'data/raw/source_catalog/internet_expansion/rasp_download_full.html'
        parser=Links()
        parser.feed(page.read_text(encoding='utf-8'))
        jobs=[]
        for href in dict.fromkeys(parser.links):
            query=parse_qs(urlparse(href).query)
            typ=query.get('filetype',[''])[0]
            if typ not in ('fasta','gff'): continue
            name=query['filelist'][0]
            if Path(name).name!=name or '/' in name or '\\' in name:
                raise ValueError('Unsafe reference name')
            jobs.append(dict(id='rasp-reference:'+name,kind='reference_gzip',reference_type=typ,
                             relative_path=name,url='https://rasp2.zhanglab.net'+href,
                             evidence_class='sequence_or_genome_annotation_not_pair_truth'))
        plan=dict(created_at_utc=now(),source_page_sha256=digest_file(page),jobs=jobs)
        write_json(plan_path,plan)
    results=[]
    report_path=ROOT/'data/reports/rasp_reference_acquisition_v1.json'
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(acquire,j,directory):j for j in plan['jobs']}
        for future in as_completed(pending):
            job=pending[future]
            try: results.append(future.result())
            except Exception as exc: results.append(dict(id=job['id'],status='failed',reason=str(exc)))
            write_json(report_path,dict(updated_at_utc=now(),requested_jobs=len(plan['jobs']),
                                       complete=len(results)==len(plan['jobs']),
                                       verified_jobs=sum(r['status'].startswith('verified') for r in results),
                                       failures=[r for r in results if not r['status'].startswith('verified')],
                                       results=results))
    print('Reference acquisition finished:',len(results))


if __name__=='__main__': main()
