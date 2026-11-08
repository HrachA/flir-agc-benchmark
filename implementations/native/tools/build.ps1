param([string]$OpenCVDir = "", [switch]$Portable)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OpenCVDir) { $OpenCVDir = Join-Path $repoRoot "third_party\opencv\build" }
Push-Location $repoRoot
try {
    $arch = if ($Portable) { "OFF" } else { "ON" }
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 "-DOpenCV_DIR=$OpenCVDir" "-DAGC_AVX2=$arch"
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
    cmake --build build --config Release --parallel 8
    if ($LASTEXITCODE -ne 0) { throw "C++ build failed" }
} finally { Pop-Location }
