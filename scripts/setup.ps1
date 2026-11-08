param(
    [ValidateSet('python', 'native', 'cuda')][string[]]$Backends = @('python'),
    [string]$Python = '',
    [string]$OpenCVDir = '',
    [string]$CudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4',
    [string]$Architectures = '89',
    [switch]$Portable,
    [switch]$CompatibilityOverride,
    [switch]$SkipDependencies
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
Push-Location $repoRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($Python) { & $Python -m venv .venv } else { py -3.12 -m venv .venv }
        if ($LASTEXITCODE -ne 0) { throw 'Creating Python 3.12 environment failed' }
    }
    if (-not $SkipDependencies) {
        & $venvPython -m pip install -r requirements-lock.txt
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    }
    if ('native' -in $Backends) {
        if (-not $OpenCVDir) {
            & $venvPython implementations/native/tools/setup_opencv.py --version 5.0.0
            if ($LASTEXITCODE -ne 0) { throw 'OpenCV SDK setup failed' }
        }
        & "$repoRoot\implementations\native\tools\build.ps1" -OpenCVDir $OpenCVDir -Portable:$Portable
    }
    if ('cuda' -in $Backends) {
        & "$repoRoot\implementations\cuda\tools\build.ps1" -CudaRoot $CudaRoot -Architectures $Architectures -CompatibilityOverride:$CompatibilityOverride
    }
    & $venvPython scripts/check.py --backends @Backends
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    Write-Host 'Setup and requested tests passed.' -ForegroundColor Green
} finally { Pop-Location }
