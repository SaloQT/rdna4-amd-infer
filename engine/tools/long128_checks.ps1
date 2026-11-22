# long128_checks.ps1 -- the short model checks of the 128K context work (every GPU run goes through ..\gpu-run.ps1).
#   pwsh -NoProfile -File tools\long128_checks.ps1 [-Only hash,ids16k,ids128k,sub]
#   hash    logits hash of the 5-token prompt (must be 2BB3D246...)
#   ids16k  w_8000, ctx 16384, f16 KV: --draft 3 ids == --draft 0 ids
#   ids128k w_8000, ctx 131072, q8 KV: --draft 3 ids == --draft 0 ids, and draft 3 with IE_ATTN_LONG_T=2 (the verify run
#           of 4 tokens in two launches) the same
#   sub     w_3000, ctx 16384, f16: IE_ATTN_LONG_T=1 and 2 (draft 3) == draft 0
param([string[]]$Only = @('hash', 'ids16k', 'ids128k', 'sub'), [string]$Tag = 'new', [string[]]$Modes = @('f16', 'q8', 'q4'), [string]$PplFile = 'build\spec\t_license.txt')
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$Only = @($Only | ForEach-Object { $_ -split ',' })
$Modes = @($Modes | ForEach-Object { $_ -split ',' })
$m = '..\..\models\Underdog-Saluki-27B-1.0-IQ2-mix.gguf'
$mtp = '..\..\models\MTP\mtp-Saluki-aligned.gguf'
New-Item -ItemType Directory -Force build\long128 | Out-Null

function Run([string]$name, [string[]]$a, [hashtable]$ev = @{}, [int]$to = 300) {
  foreach ($k in $ev.Keys) { Set-Item "Env:$k" $ev[$k] }
  $o = pwsh -NoProfile -File ..\gpu-run.ps1 -TimeoutSec $to -- build\ie-run.exe $m --backend gpu --hsaco build\ie_kernels.hsaco @a 2>&1 | ForEach-Object { "$_" }
  $rc = $LASTEXITCODE
  foreach ($k in $ev.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
  Set-Content "build\long128\${Tag}_$name.txt" $o
  if ($rc -ne 0) { "RUN FAILED ($rc): $name"; $o | Select-Object -Last 8; exit 1 }
  $line = ($o | Select-String 'ms/token' | Select-Object -First 1)
  "{0,-26} {1}" -f $name, $(if ($line) { $line.Line -replace '^gpu: ', '' } else { '(no ms/token line)' })
}
function Ids([string]$name) { ((Get-Content "build\long128\${Tag}_$name.txt" | Where-Object { $_ -like 'ids:*' }) -replace '^ids:\s*', '') -split '\s+' }
function Same([string]$a, [string]$b) {
  $x = Ids $a; $y = Ids $b
  $n = [Math]::Min($x.Count, $y.Count); $d = -1
  for ($i = 0; $i -lt $n; $i++) { if ($x[$i] -ne $y[$i]) { $d = $i; break } }
  "{0,-24} == {1,-24}: {2} ({3} / {4} ids{5})" -f $a, $b, $(if ($d -lt 0 -and $x.Count -eq $y.Count -and $x.Count -gt 1) { 'IDENTICAL' } else { 'DIFFERENT' }), $x.Count, $y.Count, $(if ($d -ge 0) { ", first divergence at id $d" } else { '' })
}
$res = @()
if ($Only -contains 'hash') {
  Run 'hash' @('--tokens', '760,6511,314,9338,369', '--n', '1', '--dump-logits', "build\long128\${Tag}_h.logits") | Out-Host
  $h = (Get-FileHash "build\long128\${Tag}_h.logits" -Algorithm SHA256).Hash
  $res += "logits hash {0}  {1}" -f $h, $(if ($h -eq '2BB3D246D67A018B78B5010BF510BDF59F338F5A0FC5F04B1900E7FBD2C0486F') { 'UNCHANGED' } else { 'CHANGED' })
}
if ($Only -contains 'ids16k') {
  $p = @('--tokens-file', 'build\spec\w_8000.txt', '--n', '32', '--ctx', '16384')
  Run 'f16_8000_d0' ($p + @('--draft', '0')) | Out-Host
  Run 'f16_8000_d3' ($p + @('--mtp', $mtp, '--draft', '3')) | Out-Host
  $res += Same 'f16_8000_d3' 'f16_8000_d0'
}
if ($Only -contains 'ids128k') {
  $p = @('--tokens-file', 'build\spec\w_8000.txt', '--n', '32', '--ctx', '131072')
  $q = @{ IE_KV_Q8 = '1' }
  Run 'q8_128k_d0' ($p + @('--draft', '0')) $q | Out-Host
  Run 'q8_128k_d3' ($p + @('--mtp', $mtp, '--draft', '3')) $q | Out-Host
  Run 'q8_128k_d3_t2' ($p + @('--mtp', $mtp, '--draft', '3')) @{ IE_KV_Q8 = '1'; IE_ATTN_LONG_T = '2' } | Out-Host
  $res += Same 'q8_128k_d3' 'q8_128k_d0'
  $res += Same 'q8_128k_d3_t2' 'q8_128k_d0'
}
if ($Only -contains 'sub') {
  $p = @('--tokens-file', 'build\spec\w_3000.txt', '--n', '32', '--ctx', '16384')
  Run 'f16_3000_d0' ($p + @('--draft', '0')) | Out-Host
  Run 'f16_3000_d3_t1' ($p + @('--mtp', $mtp, '--draft', '3')) @{ IE_ATTN_LONG_T = '1' } | Out-Host
  Run 'f16_3000_d3_t2' ($p + @('--mtp', $mtp, '--draft', '3')) @{ IE_ATTN_LONG_T = '2' } | Out-Host
  $res += Same 'f16_3000_d3_t1' 'f16_3000_d0'
  $res += Same 'f16_3000_d3_t2' 'f16_3000_d0'
}
if ($Only -contains 'q4') {
  # greedy generation after a natural 3000-token prompt per KV format (the first divergence from f16 shows how much a format
  # perturbs the model; ids: the 3000 prompt ids come first), and spec ids == draft 0 with the 4-bit cache
  $pn = @('--tokens-file', 'build\spec\nat_3000.txt', '--n', '64', '--ctx', '16384', '--draft', '0')
  Run 'nat_f16' $pn | Out-Host
  Run 'nat_q8' $pn @{ IE_KV_Q8 = '1' } | Out-Host
  Run 'nat_q4' $pn @{ IE_KV_Q4 = '1' } | Out-Host
  $res += Same 'nat_q8' 'nat_f16'
  $res += Same 'nat_q4' 'nat_f16'
  $p8 = @('--tokens-file', 'build\spec\w_8000.txt', '--n', '32', '--ctx', '16384')
  Run 'q4_8000_d0' ($p8 + @('--draft', '0')) @{ IE_KV_Q4 = '1' } | Out-Host
  Run 'q4_8000_d3' ($p8 + @('--mtp', $mtp, '--draft', '3')) @{ IE_KV_Q4 = '1' } | Out-Host
  $res += Same 'q4_8000_d3' 'q4_8000_d0'
}
if ($Only -contains 'ppl') {
  # perplexity of an 8153-token natural text through the decode kernels (--chunk 4), per KV format
  if (-not (Test-Path build\spec\t_license.txt)) { Copy-Item ..\hanw-attn\build\spec\t_license.txt build\spec\t_license.txt }
  $pp = @('--tokens-file', $PplFile, '--ppl', '--ctx', '16384', '--chunk', '4')
  foreach ($mode in $Modes) {
    $ev = @{}
    if ($mode -eq 'q8') { $ev = @{ IE_KV_Q8 = '1' } } elseif ($mode -eq 'q4') { $ev = @{ IE_KV_Q4 = '1' } }
    Run "ppl_$mode" $pp $ev | Out-Host
    $res += "ppl {0,-4} {1}" -f $mode, ((Get-Content "build\long128\${Tag}_ppl_$mode.txt" | Select-String 'perplexity').Line)
  }
}
if ($Only -contains 'grow') {
  # the KV cache grows in steps of 1024 / 2048 positions (default 8192): the ids must not change (needs the baseline runs of sub / ids128k)
  $p3 = @('--tokens-file', 'build\spec\w_3000.txt', '--n', '32', '--ctx', '16384')
  $p8 = @('--tokens-file', 'build\spec\w_8000.txt', '--n', '32', '--ctx', '131072')
  Run 'f16_3000_d0_g1k' ($p3 + @('--draft', '0')) @{ IE_KV_GROW_STEP = '1024' } | Out-Host
  Run 'q8_128k_d3_g2k' ($p8 + @('--mtp', $mtp, '--draft', '3')) @{ IE_KV_Q8 = '1'; IE_KV_GROW_STEP = '2048' } | Out-Host
  $res += Same 'f16_3000_d0_g1k' 'f16_3000_d0'
  $res += Same 'q8_128k_d3_g2k' 'q8_128k_d0'
}
''
$res
