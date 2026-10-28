param([string]$BasePython = 'C:\Program Files\Python312\python.exe')
$ErrorActionPreference = 'Stop'
$rnaRoot = Split-Path $PSScriptRoot -Parent
if ((& $BasePython -I -c 'import platform; print(platform.python_version())') -ne '3.12.8') { throw 'Expected system Python 3.12.8' }
& $BasePython -I -m pip install -r (Join-Path $rnaRoot 'requirements-shared-common.txt')
if ($LASTEXITCODE -ne 0) { throw 'Shared common dependency installation failed' }
& $BasePython -I -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Shared dependency check failed' }
& $BasePython -I (Join-Path $PSScriptRoot 'check_shared_common.py')
if ($LASTEXITCODE -ne 0) { throw 'Shared dependency source/version validation failed' }
