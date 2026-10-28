"""Check the shared Python/PyTorch runtime without loading RNA data."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import torch
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert platform.python_version()=='3.12.8'
    assert torch.__version__=='2.7.1+cu128' and torch.version.cuda=='12.8'
    assert np.__version__=='2.2.6'
    assert torch.cuda.is_available() and torch.cuda.device_count()>0
    torch.manual_seed(3128)
    checks=[]
    for index in range(torch.cuda.device_count()):
        device=torch.device(f'cuda:{index}')
        model=torch.nn.Conv1d(4,8,3,padding=1).to(device)
        before=model.weight.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        inputs=torch.randn(4,4,64,device=device)
        loss=model(inputs).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model.weight.grad).all()
        assert model.weight.grad.abs().sum()>0
        optimizer.step();torch.cuda.synchronize(device)
        assert not torch.equal(before,model.weight)
        checks.append(dict(index=index,name=torch.cuda.get_device_name(index),loss=float(loss.detach()),forward_backward_optimizer_passed=True))
        del optimizer,model,inputs,before,loss
        torch.cuda.empty_cache()
    report=dict(python=platform.python_version(),executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,torch=torch.__version__,cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=np.__version__,torch_path=torch.__file__,torch_from_global=Path(torch.__file__).resolve().is_relative_to(Path(sys.base_prefix).resolve()),packages={name:metadata.version(name) for name in ['torch','numpy','filelock','typing_extensions','sympy','networkx','jinja2','fsspec','mpmath','MarkupSafe','setuptools']},gpus=checks,all_checks_passed=True,rna_data_loaded=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
