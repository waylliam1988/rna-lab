param([string]$SourceDirectory = 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\libheif\libheif\bin')
$ErrorActionPreference = 'Stop'
$rnaRoot = Split-Path $PSScriptRoot -Parent
$rnaTarget = Join-Path $rnaRoot '.runtime\vc'
New-Item -ItemType Directory -Force -Path $rnaTarget | Out-Null
$rnaManifest = @()
Get-ChildItem -LiteralPath $SourceDirectory -File | Where-Object Name -Match '^(msvcp140|vcruntime140).*\.dll$' | ForEach-Object {
    $rnaSignature = Get-AuthenticodeSignature -LiteralPath $_.FullName
    if ($rnaSignature.Status -ne 'Valid' -or $rnaSignature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') { throw 'Untrusted runtime DLL' }
    $rnaDestination = Join-Path $rnaTarget $_.Name
    $rnaSourceHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
    if (Test-Path -LiteralPath $rnaDestination) {
        if ((Get-FileHash -LiteralPath $rnaDestination -Algorithm SHA256).Hash -ne $rnaSourceHash) { throw 'Existing project-local runtime differs; preserve it and review versions before replacing.' }
    } else { Copy-Item -LiteralPath $_.FullName -Destination $rnaTarget }
    $rnaManifest += @{ name=$_.Name; source=$_.FullName; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash; signer=$rnaSignature.SignerCertificate.Subject }
}
if ($rnaManifest.Count -eq 0) { throw 'No signed Microsoft runtime DLLs found; install the official VC++ runtime or supply a trusted source directory.' }
$rnaManifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $rnaTarget 'manifest.json') -Encoding utf8
Write-Output 'Project-local signed native runtime prepared; system files and PATH unchanged.'
