param(
    [string]$BasePython = 'C:\Program Files\Python312\python.exe',
    [string]$WheelDirectory = 'D:\E backup\下载'
)
$ErrorActionPreference = 'Stop'
$rnaRoot = Split-Path $PSScriptRoot -Parent
if (!(Test-Path -LiteralPath $BasePython)) { throw '请先安装系统级 Python 3.12.8，或通过 -BasePython 指定其完整路径。' }
$rnaVersion = & $BasePython -I -c 'import platform; print(platform.python_version())'
if ($LASTEXITCODE -ne 0 -or $rnaVersion -ne '3.12.8') { throw 'Expected working system Python 3.12.8' }
$rnaLocalWheel = Join-Path $WheelDirectory 'torch-2.7.1+cu128-cp312-cp312-win_amd64.whl'
$rnaInstallArgs = @('-I','-m','pip','install','-r',(Join-Path $rnaRoot 'requirements-shared-gpu.txt'))
if (Test-Path -LiteralPath $rnaLocalWheel -PathType Leaf) {
    $rnaWheelHash = (Get-FileHash -LiteralPath $rnaLocalWheel -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($rnaWheelHash -ne '2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30') { throw 'Local PyTorch wheel hash mismatch' }
    $rnaInstallArgs += $rnaLocalWheel
}
& $BasePython @rnaInstallArgs
if ($LASTEXITCODE -ne 0) { throw 'Shared PyTorch installation failed' }
& $BasePython -I -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Shared dependency check failed' }
$rnaReport = Join-Path $env:ProgramData 'RNA-Lab\shared-python3128\gpu_check.json'
& $BasePython -I (Join-Path $PSScriptRoot 'check_shared_gpu.py') --output $rnaReport
if ($LASTEXITCODE -ne 0) { throw 'Shared GPU verification failed' }
