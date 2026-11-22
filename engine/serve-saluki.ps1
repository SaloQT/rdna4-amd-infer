# ie-serve (OpenAI-compatible, http://127.0.0.1:8090/v1 by default) with the
# lab's Saluki 27B IQ2 mix and its MTP head: speculative decoding (mtp8(7): 7
# drafts / 8 verify tokens per step, p-min 0.4), separate launches, 128K context with a q8 KV cache
# (IE_KV_Q8=1: 4.85 GB at a full 128K) that grows with the conversation; it
# spills to shared memory past about 110K tokens while the desktop holds ~4 GB
# of VRAM. -Kv q4: the rotated 4-bit cache (IE_KV_Q4=1 IE_KV_ROT=1, 2.85 GB at
# 128K, research\kv-rot\README.md); it was the default on 2026-11-04 until long
# agent sessions on it looped (4 of 66 answers against 1 of 71 and 0 of 64 on
# q8 with the same sampling and drafts). -KvRot:$false: 4-bit unrotated; -Kv
# f16 for short contexts. Other arguments go to ie-serve and win over these,
# e.g. -Port 8091, or after the switches: --draft 0 (off).
#   pwsh -File engine\serve-saluki.ps1
# The head file holds Qwen3.8's token embedding (Q8_0), which the head reads its
# input token from (IE_MTP_EMBED=1; 1.35 GB of host memory) instead of Saluki's
# IQ1_M one: greedy ie-run 18.5% faster than 3 drafts at p-min 0.6 with the old
# head, ids equal (research\wsp-lab\round2\README.md). -MtpEmbed:$false or an
# inherited IE_MTP_EMBED=0 reads Saluki's table again; -Draft and -DraftPmin
# set the drafts.
# -OpenCodeAdapter loads the published Qwen3.8-27B pi LoRA, defaults to no thinking,
# and advertises saluki-opencode on the same OpenAI-compatible endpoint.
# -OpenCodeThinking selects the agent profile with low-effort thinking and a
# 512-token reasoning budget. Add -OpenCodeAdapter to load the pi LoRA.
# The default agent q8 profile loads measured verify kernels from a separate code object.
# IE_SALUKI_KERNELS=0 disables that module.
# Disk prompt cache (as DeepSeek's): checkpoints and the KV rows before them go to cache\prompt in the lab (the
# 100 GB most recently used), so a prompt that starts like an earlier one continues from there after a restart
# or another chat. -DiskCache DIR: another directory, -DiskCache '': off; -DiskCacheGB N: the limit.
# Loop penalty on (--dry-multiplier 0.8: a token that would continue a 24+ token repeat of this answer is
# lowered; a request sets dry_multiplier 0 to turn it off). -DryMultiplier 0 disables it here; a later
# --dry-multiplier in the arguments wins too.
# Direct (non-thinking) tool calls that set no temperature run at 0 (--tool-temp 0, the model card's
# direct tool-call recipe; the loop penalty needs temperature > 0, so they have none). Thinking + tool
# requests (such as omp) keep thinking temperature 0.6 and the DRY loop penalty. -ToolTemp -1 restores
# the normal default for non-thinking tool calls too; a request's own temperature always wins.
param([int]$Port = 8090, [int]$Ctx = 131072, [ValidateSet('q4', 'q8', 'f16')][string]$Kv = 'q8', [switch]$KvRot = $true,
  [string]$Name = 'saluki', [int]$Draft = 7, [double]$DraftPmin = 0.4, [switch]$MtpEmbed = $true,
  [string]$DiskCache, [double]$DiskCacheGB = 100, [double]$DryMultiplier = 0.8, [double]$ToolTemp = 0, [switch]$DryRun,
  [string]$Model, [string]$Mmproj, [switch]$OpenCodeAdapter, [switch]$OpenCodeThinking)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$lab = Split-Path -Parent $here
$profileArgs = @()
if ($null -eq $env:IE_SALUKI_KERNELS -and -not $Model -and $Kv -eq 'q8' -and ($Ctx -eq 65536 -or $Ctx -eq 131072) -and $Draft -eq 7) {
  $verifyKernels = Join-Path $here 'build\ie_saluki_kernels.hsaco'
  if (Test-Path -LiteralPath $verifyKernels -PathType Leaf) { $env:IE_SALUKI_KERNELS = $verifyKernels }
}
if ($OpenCodeAdapter -or $OpenCodeThinking) {
  if (-not $PSBoundParameters.ContainsKey('Name')) { $Name = 'saluki-opencode' }
  if ($null -eq $env:IE_OPENCODE_PREFIX) { $env:IE_OPENCODE_PREFIX = '1' }
  if ($null -eq $env:IE_DISK_RAM_GB) { $env:IE_DISK_RAM_GB = '2' }
  if ($null -eq $env:IE_DISK_PINNED_GB) { $env:IE_DISK_PINNED_GB = '2' }
  if ($null -eq $env:IE_DISK_QUEUE_GB) { $env:IE_DISK_QUEUE_GB = '1' }
}
if ($OpenCodeAdapter) {
  $adapter = Join-Path $lab 'models\adapters\Qwen3.8-27B-pi-lora-F16.gguf'
  if (-not (Test-Path -LiteralPath $adapter -PathType Leaf)) { throw "Missing pi adapter: $adapter" }
  if ($null -eq $env:IE_LORA_WAVE32) { $env:IE_LORA_WAVE32 = '1' }
  # Keep the pi cache separate from plain and OpenCode adapter cache files.
  if (-not $PSBoundParameters.ContainsKey('DiskCache') -and $null -eq $env:IE_DISK_CACHE) {
    $DiskCache = Join-Path $lab 'cache\prompt-saluki-pi'
    $env:IE_DISK_CACHE = $DiskCache
  }
  $profileArgs += @('--lora', $adapter)
}
if ($OpenCodeThinking) {
  $profileArgs += @('--effort', 'low', '--top-p', '0.95', '--top-k', '20', '--reasoning-budget', '512')
} elseif ($OpenCodeAdapter) {
  $profileArgs += '--no-think'
}
if ($PSBoundParameters.ContainsKey('DiskCache')) { $env:IE_DISK_CACHE = $DiskCache }
elseif ($null -eq $env:IE_DISK_CACHE) { $env:IE_DISK_CACHE = Join-Path $lab 'cache\prompt' }
if ($env:IE_DISK_CACHE) { New-Item -ItemType Directory -Force $env:IE_DISK_CACHE | Out-Null }
if ($PSBoundParameters.ContainsKey('DiskCacheGB') -or $null -eq $env:IE_DISK_CACHE_GB) {
  $env:IE_DISK_CACHE_GB = $DiskCacheGB.ToString([Globalization.CultureInfo]::InvariantCulture)
}
. (Join-Path $lab 'scripts\env-rocm.ps1') | Out-Null
$customModel = [bool]$Model
# -Model serves another Qwen3.8-27B GGUF (same architecture as Saluki) with Saluki's MTP head and settings
$model = if ($Model) { $Model } else { Join-Path $lab 'models\Underdog-Saluki-27B-1.0-IQ2-mix.gguf' }
$mtp = Join-Path $lab 'models\MTP\mtp-Saluki-aligned-q38embd.gguf'
foreach ($f in $model, $mtp) { if (-not (Test-Path $f)) { throw "missing $f" } }
$mmprojArgs = @()
if ($PSBoundParameters.ContainsKey('Mmproj')) {
  if ($Mmproj) {
    if (-not (Test-Path -LiteralPath $Mmproj -PathType Leaf)) { throw "missing $Mmproj" }
    $mmprojArgs = @('--mmproj', $Mmproj)
  }
} elseif (-not $customModel) {
  $defaultMmproj = Join-Path $lab 'models\mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf'
  if (Test-Path -LiteralPath $defaultMmproj -PathType Leaf) { $mmprojArgs = @('--mmproj', $defaultMmproj) }
}
if ($PSBoundParameters.ContainsKey('MtpEmbed')) { $env:IE_MTP_EMBED = if ($MtpEmbed) { '1' } else { '0' } }
elseif ($null -eq $env:IE_MTP_EMBED) { $env:IE_MTP_EMBED = '1' }
# IE_DRAFT_VOCAB=64k (opt-in draft head shortlist) finds its map from any working directory
if (-not $env:IE_DRAFT_VOCAB_DIR) { $env:IE_DRAFT_VOCAB_DIR = Join-Path $lab 'data\mtp-vocab-pilot\maps\default' }
Remove-Item Env:IE_KV_Q8, Env:IE_KV_Q4, Env:IE_KV_ROT -ErrorAction SilentlyContinue
if ($Kv -eq 'q8') { $env:IE_KV_Q8 = '1' } elseif ($Kv -eq 'q4') { $env:IE_KV_Q4 = '1'; if ($KvRot) { $env:IE_KV_ROT = '1' } }
$pass = @(foreach ($x in $args) { if ($x -is [array]) { $x -join ',' } else { "$x" } })
$pmin = $DraftPmin.ToString([Globalization.CultureInfo]::InvariantCulture)
$dry = $DryMultiplier.ToString([Globalization.CultureInfo]::InvariantCulture)
$toolTemp = $ToolTemp.ToString([Globalization.CultureInfo]::InvariantCulture)
# thinking at temperature 0.6, as Saluki's model card recommends (its file says 1.0); a later --think-temp wins
# the loop penalty (the engine's DRY defaults: base 1.75, allowed length 24); a later --dry-multiplier wins
$exe = Join-Path $here 'build\ie-serve.exe'
$serveArgs = @($model, '--backend', 'gpu', '--hsaco', (Join-Path $here 'build\ie_kernels.hsaco'),
  '--mtp', $mtp, '--draft', $Draft, '--draft-pmin', $pmin, '--port', $Port, '--ctx', $Ctx, '--name', $Name,
  '--think-temp', '0.6', '--dry-multiplier', $dry, '--dry-base', '1.75', '--dry-allowed-length', '24',
  '--tool-temp', $toolTemp) + $mmprojArgs + $profileArgs + $pass
if ($DryRun) { Write-Output ("$exe " + ($serveArgs -join ' ')); return }
& $exe @serveArgs
exit $LASTEXITCODE
