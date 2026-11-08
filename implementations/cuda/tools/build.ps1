param([string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4", [string]$Architectures = "89", [switch]$CompatibilityOverride)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$flags = ""
if ($CompatibilityOverride) {
    $flags = "--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
}
Push-Location $repoRoot
try {
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T "cuda=$CudaRoot" "-DCMAKE_CUDA_FLAGS=$flags" "-DAGC_CUDA_ARCHITECTURES=$Architectures"
    if ($LASTEXITCODE -ne 0) { throw "CUDA configure failed" }
    cmake --build build --config Release --parallel 4
    if ($LASTEXITCODE -ne 0) { throw "CUDA build failed" }
} finally { Pop-Location }
