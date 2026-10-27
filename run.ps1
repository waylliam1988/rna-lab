param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$rnaPython = Join-Path $PSScriptRoot '.venv3128\Scripts\python.exe'
if (!(Test-Path -LiteralPath $rnaPython)) { throw '请安装 Python 3.10+，或修改 run.ps1 中的路径。' }
Push-Location $PSScriptRoot
try { & $rnaPython @PythonArgs; $rnaExitCode = $LASTEXITCODE } finally { Pop-Location }
exit $rnaExitCode
