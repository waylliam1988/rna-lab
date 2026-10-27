"""Select a static width using parameter counts only, before training scores."""
import argparse
import subprocess
import sys
from pathlib import Path
from run_controlled_experiment import ROOT,make_model,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    count=lambda m:sum(p.numel() for p in m.parameters())
    target=count(make_model(a.variant,32,4))
    hidden=min(range(8,97),key=lambda h:abs(count(make_model('static',h,4))-target))
    write_json(ROOT/'data/manifests'/(a.output.name+'_capacity_selection.json'),dict(variant=a.variant,target_parameters=target,
               static_hidden=hidden,static_parameters=count(make_model('static',hidden,4)),selection='Nearest parameter count over widths8-96, no model scores used'))
    subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/run_controlled_experiment.py'),'--variant','static',
                    '--hidden',str(hidden),'--output',str(a.output)],cwd=ROOT,check=True)


if __name__=='__main__':main()
