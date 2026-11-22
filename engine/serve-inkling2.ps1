# ie-serve (OpenAI-compatible, http://127.0.0.1:8090/v1 by default) with
# OpenUncensored/Inkling-2-Mini-Uncensored (IQ3_XXS): 40 SWA (win=1024) + 8 global
# layers across 48 logical visits, f16 KV cache (kv_grow=1 on demand; max ~1.31 GiB
# at 32K, avoids IE_KV_Q8 drift), 96K default context, thinking temp 1.0 (per HF
# generation_config.json; non-thinking 0.7 / top_p 0.8), top_k 20, loop penalty on
# (--dry-multiplier 0.8), direct tool temp 0, default reasoning effort high (0.9).
# Model: models\Inkling-2-Mini-Uncensored-IQ3_XXS-fast.gguf when it exists (the same weights with output.weight as
# Q4_K instead of Q8_0, 231 MB less VRAM; scripts\inkling2_requant.py writes it), else the original file; -Model PATH
# picks another. DSpark drafts are on with the f16 KV cache: 5 drafts, p-min 0.6 with zero-sync Q4_0 split head
# (28.9 MB P) measured 130.4 against 94.3 tokens/s plain on sampled chat and 148.4 against 97.6 at temperature 0
# (bench\inkling2_serve_sweep.py); -Draft 0 is plain decode, -DsSplit:$false disables split head.
# -DiskCache DIR keeps prompt checkpoints on disk (IE_DISK_CACHE), so a restarted server or a reopened chat continues
# instead of re-reading the prompt (9.8K tokens: 0.9 s against 8.2 s), with or without DSpark.
#   powershell -File engine\serve-inkling2.ps1 [-Kv f16|q8|q4] [-Ctx 98304] [-Port 8090] [-Draft 5] [-DryRun]
param(
  [int]$Port = 8090,
  [int]$Ctx = 98304,
  [ValidateSet('f16', 'q8', 'q4')][string]$Kv = 'f16',
  [switch]$KvRot = $true,
  [string]$Name = 'inkling2-uncensored',
  [ValidateSet('none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max')][string]$Effort = 'high',
  [int]$Chunk = 512,
  [int]$Draft = 5,
  [double]$DraftPmin = 0.6,
  [string]$Dspark,
  [switch]$DsSplit = $true,
  [double]$DryMultiplier = 0.8,
  [double]$ToolTemp = 0,
  [string]$Model,
  [string]$DiskCache,
  [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$lab = Split-Path -Parent $here

. (Join-Path $lab 'scripts\env-rocm.ps1') | Out-Null

$fastModel = Join-Path $lab 'models\Inkling-2-Mini-Uncensored-IQ3_XXS-fast.gguf'
$modelPath = if ($Model) { $Model } elseif (Test-Path -LiteralPath $fastModel -PathType Leaf) { $fastModel } else { Join-Path $lab 'models\Inkling-2-Mini-Uncensored-IQ3_XXS.gguf' }
$dsparkQ4k = Join-Path $lab 'models\DSpark\Inkling-2-Mini-DSpark-Q4_K.gguf'
$dsparkQ8 = Join-Path $lab 'models\DSpark\Inkling-2-Mini-DSpark-Q8_0.gguf'
$dsparkPath = if ($Dspark) { $Dspark } elseif (Test-Path -LiteralPath $dsparkQ4k -PathType Leaf) { $dsparkQ4k } else { $dsparkQ8 }
$hsaco = Join-Path $here 'build\ie_kernels.hsaco'
$exe = Join-Path $here 'build\ie-serve.exe'
foreach ($f in @($modelPath, $hsaco, $exe)) {
  if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { throw "missing $f" }
}
$useDspark = ($Draft -gt 0) -and ($Kv -eq 'f16') -and (Test-Path -LiteralPath $dsparkPath -PathType Leaf)
if ($PSBoundParameters.ContainsKey('DsSplit')) {
  $env:IE_DS_SPLIT = if ($DsSplit) { '1' } else { '0' }
} elseif ($useDspark -and $Draft -ge 4 -and [string]::IsNullOrEmpty($env:IE_DS_SPLIT)) {
  $env:IE_DS_SPLIT = '1'
}
if ($useDspark -and $null -eq $env:IE_DS_SHORTLIST) {
  $shortlistManifest = Join-Path $lab 'data\inkling2-shortlist\inkling2-frspec-61440.manifest.json'
  if (Test-Path -LiteralPath $shortlistManifest -PathType Leaf) {
    $env:IE_DS_SHORTLIST = $shortlistManifest
  }
}

if ($DiskCache) {
  New-Item -ItemType Directory -Force $DiskCache | Out-Null
  $env:IE_DISK_CACHE = (Resolve-Path -LiteralPath $DiskCache).Path
}

Remove-Item Env:IE_KV_Q8, Env:IE_KV_Q4, Env:IE_KV_ROT -ErrorAction SilentlyContinue
if ($Kv -eq 'q8') {
  $env:IE_KV_Q8 = '1'
} elseif ($Kv -eq 'q4') {
  $env:IE_KV_Q4 = '1'
  if ($KvRot) { $env:IE_KV_ROT = '1' }
}

$inv = [Globalization.CultureInfo]::InvariantCulture
$dry = $DryMultiplier.ToString($inv)
$tt = $ToolTemp.ToString($inv)
$pminStr = $DraftPmin.ToString($inv)
$pass = @(foreach ($x in $args) { if ($x -is [array]) { $x -join ',' } else { "$x" } })

$serveArgs = @(
  $modelPath,
  '--backend', 'gpu',
  '--hsaco', $hsaco,
  '--port', "$Port",
  '--ctx', "$Ctx",
  '--name', $Name,
  '--chunk', "$Chunk",
  '--think-temp', '1.0',
  '--top-k', '20',
  '--dry-multiplier', $dry,
  '--dry-base', '1.75',
  '--dry-allowed-length', '24',
  '--tool-temp', $tt,
  '--effort', $Effort
)
if ($useDspark) {
  $serveArgs += @('--dspark', $dsparkPath, '--draft', "$Draft", '--draft-pmin', $pminStr)
}
$serveArgs += $pass

if ($DryRun) {
  Write-Output ("KV=$Kv IE_KV_Q8=$($env:IE_KV_Q8) IE_KV_Q4=$($env:IE_KV_Q4) | $exe " + ($serveArgs -join ' '))
  return
}
& $exe @serveArgs
exit $LASTEXITCODE
