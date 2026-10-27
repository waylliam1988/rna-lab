"""Fresh process per resource case, avoiding cross-model CUDA workspace history."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from acquire_experimental_corpus import ROOT,write_json,digest_file
from benchmark_sequence_encoders import one
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',nargs=3,metavar=('KIND','LENGTH','MODE'));args=parser.parse_args()
    ppath=ROOT/'data/manifests/encoder_comparison_protocol_v1.json';p=json.loads(ppath.read_text());root=ROOT/'data/reports/encoder_resources_isolated_v2';root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if args.case:
        kind,n,mode=args.case;n=int(n);path=ROOT/'runs/encoder_comparison_v1_seed42'/(kind+'.pt');ck=torch.load(path,map_location='cpu',weights_only=True)
        try:row=one(kind,n,mode,ck,p,torch.device('cuda:0'));row['status']='ok'
        except torch.cuda.OutOfMemoryError:row=dict(encoder=kind,length=n,batch=1,mode=mode,status='OOM')
        row['checkpoint_sha256']=digest_file(path);row['protocol_sha256']=digest_file(ppath)
        write_json(root/f'{kind}_{n}_{mode}.json',row);return
    out=ROOT/'data/manifests/encoder_resource_benchmark_isolated_v2.json'
    if out.exists():raise ValueError('Isolated benchmark frozen')
    assert all((ROOT/'data/manifests'/f'encoder_comparison_seed{s}_v1.json').exists() for s in p['seeds'])
    cases=[]
    for kind in p['encoders']:
        for n in p['speed_benchmark']['lengths']:
            for mode in ('inference','training'):
                subprocess.run([sys.executable,'-X','utf8',str(Path(__file__)), '--case',kind,str(n),mode],cwd=ROOT,check=True,capture_output=True,text=True)
                row=json.loads((root/f'{kind}_{n}_{mode}.json').read_text());cases.append(row)
            print(json.dumps(dict(encoder=kind,length=n,completed=len(cases))),flush=True)
    report=dict(device=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),precision=p['precision'],protocol_sha256=digest_file(ppath),cases=cases,measurement='fresh subprocess for every encoder/length/mode; same timing fixture as v1',benchmark_source_sha256=digest_file(Path(__file__)),fixture_source_sha256=digest_file(ROOT/'scripts/benchmark_sequence_encoders.py'),limits=['PyTorch allocations include current-process library workspaces but exclude driver/context memory.','Fresh processes remove persistent workspace carryover from prior cases.','Synthetic shapes; external load and clock state not fully controlled.'])
    write_json(out,report)


if __name__=='__main__':main()
