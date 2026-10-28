param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$rnaPython = Join-Path $PSScriptRoot '.venv3128\Scripts\python.exe'
if (!(Test-Path -LiteralPath $rnaPython)) { throw '请先配置系统Python3.12.8及全局PyTorch，再运行scripts/setup_environment.ps1。' }
Push-Location $PSScriptRoot
try { & $rnaPython @PythonArgs; $rnaExitCode = $LASTEXITCODE } finally { Pop-Location }
exit $rnaExitCode
