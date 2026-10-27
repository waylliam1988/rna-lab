$ErrorActionPreference = 'Stop'
$rnaRoot = Split-Path $PSScriptRoot -Parent
$rnaBasePython = Join-Path $rnaRoot '.runtime\python3128\python.exe'
$rnaPython = Join-Path $rnaRoot '.venv3128\Scripts\python.exe'
if (!(Test-Path -LiteralPath $rnaBasePython)) {
    $rnaInstallerPath = 'D:\E backup\下载\python-3.12.8-amd64.exe'
    $rnaSignature = Get-AuthenticodeSignature -LiteralPath $rnaInstallerPath
    if ($rnaSignature.Status -ne 'Valid' -or $rnaSignature.SignerCertificate.Subject -notmatch 'Python Software Foundation') { throw 'Python installer signature invalid' }
    $rnaTarget = Join-Path $rnaRoot '.runtime\python3128'
    $rnaArgs = @('/quiet','InstallAllUsers=0',('TargetDir="' + $rnaTarget + '"'),'Include_pip=1','Include_test=0','Include_launcher=0','PrependPath=0','Shortcuts=0')
    $rnaProcess = Start-Process -FilePath $rnaInstallerPath -ArgumentList $rnaArgs -WindowStyle Hidden -PassThru -Wait
    if ($rnaProcess.ExitCode -ne 0) { throw 'Python installation failed' }
}
if ((& $rnaBasePython -c 'import platform; print(platform.python_version())') -ne '3.12.8') { throw 'Expected Python 3.12.8' }
if (!(Test-Path -LiteralPath $rnaPython)) {
    & $rnaBasePython -m venv (Join-Path $rnaRoot '.venv3128')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
& $rnaPython -m pip install -r (Join-Path $rnaRoot 'requirements-cuda.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $rnaPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed' }
& $rnaPython (Join-Path $PSScriptRoot 'check_environment.py')
if ($LASTEXITCODE -ne 0) { throw 'CUDA runtime verification failed' }
