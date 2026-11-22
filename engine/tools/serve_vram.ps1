# serve_vram.ps1 -- start ie-serve through ..\gpu-run.ps1 (timeout 900 s), read the process's dedicated and shared GPU memory
# from the performance counters, optionally send one short chat request and read them again, then stop the server by its
# own process id (never by image name; a gpu-run timeout kill would write ref\.gpu-fault).
#   pwsh -NoProfile -File tools\serve_vram.ps1 [-Ctx 131072] [-Hsaco build\ie_kernels.hsaco] [-Request] [-Prompt FILE]
# Prints one line per reading; the server's stderr lands in build\long128\serve_<tag>.err.
param([int]$Ctx = 131072, [int]$Port = 8097, [string]$Hsaco = 'build\ie_kernels.hsaco', [string]$Tag = 'serve', [switch]$Request, [string]$Prompt = '', [int]$WaitSec = 240, [string]$Kv = 'q8')
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
New-Item -ItemType Directory -Force build\long128 | Out-Null
$m = '..\..\models\Underdog-Saluki-27B-1.0-IQ2-mix.gguf'
$mtp = '..\..\models\MTP\mtp-Saluki-aligned.gguf'
if ($Kv -eq 'q4') { $env:IE_KV_Q4 = '1' } else { $env:IE_KV_Q8 = '1' }
$since = Get-Date
$args2 = @('-NoProfile', '-File', '..\gpu-run.ps1', '-TimeoutSec', '900', '--', 'build\ie-serve.exe', $m, '--backend', 'gpu', '--hsaco', $Hsaco, '--mtp', $mtp, '--port', "$Port", '--ctx', "$Ctx", '--name', 'saluki')
$gr = Start-Process -FilePath pwsh -ArgumentList $args2 -PassThru -WindowStyle Hidden
Remove-Item Env:IE_KV_Q8, Env:IE_KV_Q4 -ErrorAction SilentlyContinue
function Counters([int]$id, [string]$label) {
  $r = @{}
  foreach ($c in 'Dedicated Usage', 'Shared Usage') {
    $s = (Get-Counter "\GPU Process Memory(pid_${id}_*)\$c" -ErrorAction SilentlyContinue).CounterSamples
    $r[$c] = ($s | Measure-Object -Property CookedValue -Sum).Sum
  }
  "{0,-22} dedicated {1,7:N0} MB   shared {2,7:N0} MB" -f $label, ($r['Dedicated Usage'] / 1MB), ($r['Shared Usage'] / 1MB)
}
# the server's stderr: gpu-run keeps it in a temp file until the run ends
$err = $null; $srv = $null
for ($i = 0; $i -lt $WaitSec -and -not $srv; $i++) {
  Start-Sleep 1
  $srv = Get-CimInstance Win32_Process -Filter "Name='ie-serve.exe'" | Where-Object { $_.ParentProcessId -eq $gr.Id } | Select-Object -First 1
  if ($gr.HasExited) { 'gpu-run exited early (refused or failed)'; return }
}
if (-not $srv) { 'no ie-serve child found'; Stop-Process -Id $gr.Id -Force; return }
"ie-serve pid $($srv.ProcessId) (gpu-run pid $($gr.Id))"
$errf = Get-ChildItem $env:TEMP -Filter 'gpu-run-*.err' | Where-Object { $_.LastWriteTime -ge $since.AddSeconds(-5) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
function ReadErr {
  if (-not $errf) { return '' }
  $fs = [IO.File]::Open($errf.FullName, 'Open', 'Read', 'ReadWrite,Delete'); $sr = New-Object IO.StreamReader($fs)
  $t = $sr.ReadToEnd(); $sr.Close(); $fs.Close(); $t
}
$ready = $false
for ($i = 0; $i -lt $WaitSec; $i++) {
  if ($null -eq (Get-Process -Id $srv.ProcessId -ErrorAction SilentlyContinue)) { 'ie-serve exited'; break }
  if ((ReadErr) -match 'ie-serve: http://') { $ready = $true; break }
  Start-Sleep 1
}
try {
  if ($ready) {
    Counters $srv.ProcessId 'after load'
    if ($Request) {
      $body = if ($Prompt) { Get-Content $Prompt -Raw } else { '{"model":"saluki","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":16,"stream":false}' }
      $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 600
      "request done: " + ($r.usage | ConvertTo-Json -Compress)
      "answer: " + ($r.choices[0].message.content -replace '\s+', ' ')
      Counters $srv.ProcessId 'after request'
    }
  }
} finally {
  Set-Content "build\long128\${Tag}.err" (ReadErr)
  if (Get-Process -Id $srv.ProcessId -ErrorAction SilentlyContinue) { Stop-Process -Id $srv.ProcessId -Force }
  $gr.WaitForExit(60000) | Out-Null
  'stopped; gpu-run exit: ' + $gr.ExitCode
}
