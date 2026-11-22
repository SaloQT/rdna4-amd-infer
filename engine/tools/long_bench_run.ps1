# long_bench_run.ps1 -- ie_attn_long timing in the standalone bench at long positions, one process per position and
# T (timing only; the bench's buffers are about 360 MB). Every run goes through ..\gpu-run.ps1.
#   pwsh -NoProfile -File tools\long_bench_run.ps1 -Pos 65536,102400,131000 -T 1,4 [-Q8] [-Hsaco build\x.hsaco] [-Extra '--g','8']
param([string[]]$Pos = @('65536', '102400', '131000'), [string[]]$T = @('1', '4'), [switch]$Q8, [string]$Hsaco = '', [string[]]$Extra = @(), [int]$Timeout = 120)
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$Pos = @($Pos | ForEach-Object { $_ -split ',' }); $T = @($T | ForEach-Object { $_ -split ',' }); $Extra = @($Extra | ForEach-Object { $_ -split ',' })
$py = 'python'
if (-not $Hsaco) { $Hsaco = if ($Q8) { 'build\ie_kernels_q8.hsaco' } else { 'build\ie_kernels.hsaco' } }
foreach ($p in $Pos) {
  foreach ($t in $T) {
    $a = @('tools\attn_long_bench.py', '--hsaco', $Hsaco, '--pos', $p, '--T', $t, '--time', '--kernels', 'long') + $Extra
    if ($Q8) { $a += '--q8' }
    pwsh -NoProfile -File ..\gpu-run.ps1 -TimeoutSec $Timeout -- $py @a 2>&1 | ForEach-Object { "$_" } | Select-String 'time:|long|rc|FAIL|error|Error' | ForEach-Object { $_.Line }
    if ($LASTEXITCODE -ne 0) { "run failed (rc $LASTEXITCODE) at pos $p T $t"; return }
  }
}
