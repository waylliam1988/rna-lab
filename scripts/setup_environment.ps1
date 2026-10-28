param([string]$BasePython = 'C:\Program Files\Python312\python.exe')
$ErrorActionPreference = 'Stop'
$rnaRoot = Split-Path $PSScriptRoot -Parent
$rnaBasePython = $BasePython
$rnaPython = Join-Path $rnaRoot '.venv3128\Scripts\python.exe'
if (!(Test-Path -LiteralPath $rnaBasePython)) {
    throw '请先安装系统级Python 3.12.8，或通过-BasePython指定完整路径；不再在.runtime重复安装。'
}
if ((& $rnaBasePython -I -c 'import platform; print(platform.python_version())') -ne '3.12.8') { throw 'Expected Python 3.12.8' }
& $rnaBasePython -I -c 'import torch; assert torch.__version__ == "2.7.1+cu128" and torch.version.cuda == "12.8"'
if ($LASTEXITCODE -ne 0) { throw '请先运行scripts/setup_shared_gpu.ps1配置全局PyTorch/CUDA。' }
if (!(Test-Path -LiteralPath $rnaPython)) {
    & $rnaBasePython -I -m venv --without-pip --system-site-packages (Join-Path $rnaRoot '.venv3128')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
} else {
    & $rnaBasePython -I -m venv --upgrade --system-site-packages --without-pip (Join-Path $rnaRoot '.venv3128')
    if ($LASTEXITCODE -ne 0) { throw 'venv base update failed' }
}
& $rnaBasePython -I (Join-Path $PSScriptRoot 'check_shared_common.py')
if ($LASTEXITCODE -ne 0) { throw '请先运行scripts/setup_shared_common.ps1安装全局公共依赖。' }
& $rnaPython -I -m pip install -r (Join-Path $rnaRoot 'requirements-data.txt')
if ($LASTEXITCODE -ne 0) { throw 'RNA-specific dependency installation failed' }
& $rnaPython -I -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Project dependency validation failed' }
& $rnaPython -I -c 'import pathlib,sys,torch; assert pathlib.Path(torch.__file__).resolve().is_relative_to(pathlib.Path(sys.base_prefix).resolve()), "项目内仍有同名torch副本，尚未切换到全局共享，请先核对其来源。"'
if ($LASTEXITCODE -ne 0) { throw 'Shared torch source verification failed' }
& $rnaPython (Join-Path $PSScriptRoot 'check_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime verification failed' }
