# Put the ROCm 10.1 user-space libraries on PATH for the llama.cpp ROCm build.
# The b11669 "rocm-10.0" zip bundles amdhip64_7.dll but not hipBLAS/rocBLAS, so
# ggml-hip.dll refuses to load without these (verified: hipblas.dll is the only
# unresolved PE import of ggml-hip.dll). Dot-source this before running it.
$lab = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$rocmBin = Join-Path $lab '.rocm-venv\Lib\site-packages\_rocm_sdk_libraries\bin'

if (-not (Test-Path (Join-Path $rocmBin 'hipblas.dll'))) {
    throw "ROCm libraries not installed at $rocmBin (pip install rocm-sdk-core rocm-sdk-libraries rocm-sdk-device-gfx1201)"
}

if (($env:PATH -split ';') -notcontains $rocmBin) { $env:PATH = "$rocmBin;$env:PATH" }
# rocBLAS/hipBLASLt resolve their per-arch code objects from here
$env:ROCBLAS_TENSILE_LIBPATH   = Join-Path $rocmBin 'rocblas\library'
$env:HIPBLASLT_TENSILE_LIBPATH = Join-Path $rocmBin 'hipblaslt\library'
if (Test-Path (Join-Path $env:HIPBLASLT_TENSILE_LIBPATH 'gfx1201')) {
    $env:HIPBLASLT_TENSILE_LIBPATH = Join-Path $env:HIPBLASLT_TENSILE_LIBPATH 'gfx1201'
}
$env:HIP_PATH = $rocmBin

Write-Host "[rocm] $rocmBin"
