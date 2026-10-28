param([string]$BasePython = 'C:\Program Files\Python312\python.exe')
$ErrorActionPreference = 'Stop'
$rnaRoot = Split-Path $PSScriptRoot -Parent
if (!(Test-Path -LiteralPath $BasePython)) { throw '请先安装系统级 Python 3.12.8，或通过 -BasePython 指定其完整路径。' }
$rnaVersion = & $BasePython -I -c 'import platform; print(platform.python_version())'
if ($LASTEXITCODE -ne 0 -or $rnaVersion -ne '3.12.8') { throw 'Expected working system Python 3.12.8' }
$rnaWheelhouse = Join-Path $env:ProgramData 'RNA-Lab\shared-python3128\wheelhouse'
$rnaInstallArgs = @('-I','-m','pip','install','-r',(Join-Path $rnaRoot 'requirements-shared-gpu.txt'))
if (Test-Path -LiteralPath $rnaWheelhouse) { $rnaInstallArgs += @('--find-links',$rnaWheelhouse) }
& $BasePython @rnaInstallArgs
if ($LASTEXITCODE -ne 0) { throw 'Shared PyTorch installation failed' }
& $BasePython -I -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Shared dependency check failed' }
$rnaReport = Join-Path $env:ProgramData 'RNA-Lab\shared-python3128\gpu_check.json'
& $BasePython -I (Join-Path $PSScriptRoot 'check_shared_gpu.py') --output $rnaReport
if ($LASTEXITCODE -ne 0) { throw 'Shared GPU verification failed' }
