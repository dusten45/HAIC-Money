$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $RepoRoot
try {
    python -m haic_research.cli validate --root $RepoRoot
}
finally {
    Pop-Location
}
exit $LASTEXITCODE
