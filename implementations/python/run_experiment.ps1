param(
    [string]$DatasetRoot = "..\flir-rgbt-yolo-experiments\data\FLIR_ADAS_v2",
    [string]$Output = "results\reproduction"
)
$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
if (-not [System.IO.Path]::IsPathRooted($DatasetRoot)) {
    $DatasetRoot = Join-Path $repoRoot $DatasetRoot
}
$datasetPath = (Resolve-Path -LiteralPath $DatasetRoot).Path
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Create .venv and install requirements-lock.txt first; see README.md."
}
Push-Location $repoRoot
try {
    & $pythonPath -m agc_benchmark.run --dataset-root $datasetPath --output $Output --config configs/default.json
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
