[CmdletBinding()]
param(
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8765,
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$ArtifactRoot = ".haic-artifacts",
    [string]$AgentsRoot = ".haic-artifacts\agents",
    [switch]$ForceExternalPython
)

$ErrorActionPreference = "Stop"
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$venvPython = Join-Path $resolvedProjectRoot ".venv\Scripts\python.exe"
$venvSitePackages = Join-Path $resolvedProjectRoot ".venv\Lib\site-packages"

function Test-Python311([string]$pythonPath) {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        return $false
    }
    try {
        & $pythonPath -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        # Windows application-control policies can reject the executable before
        # Python starts.  In that case continue to the external interpreter.
        return $false
    }
}

$selectedPython = $null
$usingProjectVenv = $false

# Use the project venv when the host policy allows it.  On the affected
# machines this probe is rejected, so it is skipped without changing the venv.
if (-not $ForceExternalPython -and (Test-Python311 $venvPython)) {
    $selectedPython = $venvPython
    $usingProjectVenv = $true
}

# The Python launcher normally resolves this to the uv-managed 3.11 runtime.
# It is outside the blocked project .venv path and can import the existing
# wheels through PYTHONPATH below.
if ($null -eq $selectedPython) {
    $pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        $resolved = & $pyLauncher.Source -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            $candidate = $resolved | Select-Object -Last 1
            $candidate = ([string]$candidate).Trim()
            if (Test-Python311 $candidate) {
                $selectedPython = $candidate
            }
        }
    }
}

if ($null -eq $selectedPython) {
    $externalCandidates = @()
    if ($env:APPDATA) {
        $externalCandidates += Get-ChildItem `
            -Path (Join-Path $env:APPDATA "uv\python\cpython-3.11*-windows-x86_64-none\python.exe") `
            -File -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    }
    $externalCandidates += @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        "C:\Python311\python.exe"
    )
    foreach ($candidate in ($externalCandidates | Select-Object -Unique)) {
        if (Test-Python311 $candidate) {
            $selectedPython = $candidate
            break
        }
    }
}

if ($null -eq $selectedPython) {
    throw "Python 3.11을 실행할 수 없습니다. py -3.11 또는 허용된 Python 3.11 설치를 확인하십시오."
}

if (-not $usingProjectVenv -and (Test-Path -LiteralPath $venvSitePackages)) {
    $oldPythonPath = $env:PYTHONPATH
    if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
        $env:PYTHONPATH = $venvSitePackages
    } else {
        $env:PYTHONPATH = "$venvSitePackages;$oldPythonPath"
    }
    Write-Host "프로젝트 .venv 실행이 차단되어 외부 Python 3.11 + 기존 site-packages를 사용합니다."
}

Write-Host "Python: $selectedPython"
Write-Host "Track Lab: http://${BindHost}:${Port}/"
& $selectedPython -m local_simulator.web `
    --host $BindHost `
    --port $Port `
    --project-root $resolvedProjectRoot `
    --artifact-root (Join-Path $resolvedProjectRoot $ArtifactRoot) `
    --agents-root (Join-Path $resolvedProjectRoot $AgentsRoot)
exit $LASTEXITCODE
