#!/usr/bin/env python3
"""Unified 40-round Inkling-2-Mini benchmark harness, parity gate, and ledger recorder.

Measures TTFT, Prefill (4K & 16K), and Decode (Plain T=1 & DSpark 5x0.6 split-head)
at true trace-off wall clock in ~45-60 seconds per round on RX 9070 XT (gfx1201).

Usage examples:
  # Dry run (no GPU):
  python bench/inkling2_40r_harness.py --dry-run --round 0 --name "Phase 7 Baseline"

  # Record Round 0 baseline (run inside bench/run-memory-bounded.ps1 -BudgetGiB 20):
  python bench/inkling2_40r_harness.py --record-round --round 0 --wave 0 \\
      --name "Phase 7 Baseline" --target All --status baseline \\
      --source-title "Phase 7 Promoted Defaults" \\
      --hypothesis "Establish trace-off R00 wall-clock baseline across TTFT, Prefill, Decode"

  # Run a round with optional warm-cache + live serve + verify-bench checks:
  python bench/inkling2_40r_harness.py --record-round --round 1 --wave 1 \\
      --name "Prefix Cache Retention + Fast Sampling" --target TTFT --status promoted \\
      --with-serve --with-cache --source-title "vLLM Prefix Caching" \\
      --source-url "https://github.com/vllm-project/vllm" \\
      --hypothesis "Prevent tiny side requests from evicting long cached prefixes"
"""

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEDGER = os.path.join(LAB, "results", "inkling2", "40round-ledger.json")
DEFAULT_HTML = os.path.join(LAB, "results", "inkling2", "40round-dashboard.html")
DEFAULT_MODEL_FAST = os.path.join(LAB, "models", "Inkling-2-Mini-Uncensored-IQ3_XXS-fast.gguf")
DEFAULT_MODEL_ORIG = os.path.join(LAB, "models", "Inkling-2-Mini-Uncensored-IQ3_XXS.gguf")
DEFAULT_DSPARK = os.path.join(LAB, "models", "DSpark", "Inkling-2-Mini-DSpark-Q4_K.gguf")
DEFAULT_EXE = os.path.join(LAB, "ref", "tr8-team-merge", "build", "ie-run.exe")
DEFAULT_HSACO = os.path.join(LAB, "ref", "tr8-team-merge", "build", "ie_kernels.hsaco")
DEFAULT_SERVE_PS1 = os.path.join(LAB, "ref", "tr8-team-merge", "serve-inkling2.ps1")

PROMPT_233 = os.path.join(LAB, "research", "inkling2", "prompts", "real-code-233.txt")
PROMPT_4K = os.path.join(LAB, "research", "inkling2", "prompts", "real-code-4k.txt")
PROMPT_16K_TMP = os.path.join(LAB, "results", "inkling2", "tmp", "real-code-16k.txt")
SRC_TEXT_FILE = os.path.join(LAB, "results", "inkling2", "tmp", "real-code-src.txt")
KLD_REF_FILE = os.path.join(LAB, "results", "inkling2", "tmp", "r00-ref-4k.kld")

CORE_STAGES = ("t1_233", "ds5_233", "ds5_4k", "long_16k")
BUSY_PROCS = ("ie-serve.exe", "ie-run.exe", "ie-lab.exe", "llama-server.exe")
REFUSED_ENV = {
    "AMD_DIRECT_DISPATCH": "ie-run hangs at start on Windows (no output, killed after 600 s)",
}

SERVE_SAMPLED_PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string by hand (no datetime.fromisoformat), with a few unit tests.",
    "Explain how a binary heap works, then implement push and pop for an int min-heap in C.",
    "Write a three.js scene with an orbiting camera, a rotating cube and a ground plane with shadows. Code only.",
]


def lab_path(p):
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.join(LAB, p)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fnum(pat, text, group=1, last=False):
    m = re.findall(pat, text, re.M)
    if not m:
        return None
    v = m[-1] if last else m[0]
    try:
        return float(v[group - 1] if isinstance(v, tuple) else v)
    except (ValueError, TypeError):
        return None


def inum(pat, text, group=1, last=False):
    v = fnum(pat, text, group=group, last=last)
    return int(round(v)) if v is not None else None


def sstr(pat, text, group=1, last=False):
    m = re.findall(pat, text, re.M)
    if not m:
        return None
    v = m[-1] if last else m[0]
    return v[group - 1] if isinstance(v, tuple) else v


def gpu_other_mb():
    """Dedicated VRAM held by other processes (Windows PerfCounter), MB; -1 if unavailable."""
    ps = (
        "$s=(Get-Counter -Counter '\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples;"
        "[long](($s | Measure-Object CookedValue -Sum).Sum/1MB)"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        lines = [x.strip() for x in out.splitlines() if x.strip()]
        return int(lines[-1]) if lines else -1
    except Exception:
        return -1


def check_preflight_guards():
    fault = os.path.join(LAB, "ref", ".gpu-fault")
    if os.path.exists(fault):
        raise SystemExit("refused: ref/.gpu-fault exists (an earlier GPU run faulted)")
    rlock = os.path.join(LAB, "research", "q4-attention", "state", "lock.json")
    if os.path.exists(rlock):
        raise SystemExit("refused: research/q4-attention/state/lock.json exists")
    busy = subprocess.run(["tasklist"], capture_output=True, text=True).stdout
    for proc in BUSY_PROCS:
        if proc.lower() in busy.lower():
            raise SystemExit("refused: %s is already running (one GPU job at a time)" % proc)


def take_gpu_lock(tag):
    lock = os.path.join(LAB, "ref", ".gpu-lock")
    fault = os.path.join(LAB, "ref", ".gpu-fault")
    if os.path.exists(fault):
        raise SystemExit("refused: ref/.gpu-fault exists (an earlier GPU run faulted)")
    waited = 0
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getctime(lock) > 1200:
                    shutil.rmtree(lock, ignore_errors=True)
                    continue
            except OSError:
                pass
            if waited % 30 == 0:
                print("[lock] waiting for ref/.gpu-lock (%d s)" % waited, flush=True)
            time.sleep(2)
            waited += 2
    try:
        with open(os.path.join(lock, "owner.txt"), "w", encoding="utf-8") as f:
            f.write("pid %d inkling2_40r_harness %s\n" % (os.getpid(), tag))
    except OSError:
        pass
    return lock


def release_gpu_lock(lock):
    if not lock:
        return
    try:
        owner = os.path.join(lock, "owner.txt")
        if os.path.exists(owner):
            os.remove(owner)
    except OSError:
        pass
    try:
        os.rmdir(lock)
    except OSError:
        shutil.rmtree(lock, ignore_errors=True)


def write_gpu_fault(reason):
    fault = os.path.join(LAB, "ref", ".gpu-fault")
    with open(fault, "w", encoding="utf-8") as f:
        f.write(reason)


def read_token_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x for x in re.split(r"[,\s]+", f.read().strip()) if x]


def ensure_16k_prompt():
    """Create deterministic 16,384-token real-code prompt by repeating real-code-4k.txt 4x."""
    os.makedirs(os.path.dirname(PROMPT_16K_TMP), exist_ok=True)
    toks_4k = read_token_list(PROMPT_4K)
    if not toks_4k:
        raise SystemExit("empty prompt file: %s" % PROMPT_4K)
    reps = (16384 + len(toks_4k) - 1) // len(toks_4k)
    toks_16k = (toks_4k * reps)[:16384]
    payload = ",".join(toks_16k)
    need_write = True
    if os.path.exists(PROMPT_16K_TMP):
        try:
            with open(PROMPT_16K_TMP, "r", encoding="utf-8") as f:
                if f.read().strip() == payload:
                    need_write = False
        except OSError:
            pass
    if need_write:
        with open(PROMPT_16K_TMP, "w", encoding="utf-8") as f:
            f.write(payload)
    return PROMPT_16K_TMP, len(toks_16k)


def build_cache_text_prompt():
    """Build a deterministic ~4.2K-token code prompt for warm disk-cache & prefix-retention tests."""
    if os.path.exists(SRC_TEXT_FILE):
        with open(SRC_TEXT_FILE, "r", encoding="utf-8", errors="replace") as f:
            base = f.read().strip()
    else:
        base = "\n".join(
            [
                "/* Deterministic C kernel benchmark fixture block %d */" % i
                + "\nstatic inline float dot_block_%d(const float *a, const float *b, int n) {\n"
                "  float acc = 0.0f;\n  for (int k = 0; k < n; ++k) acc += a[k] * b[k];\n  return acc;\n}\n" % i
                for i in range(120)
            ]
        )
    # ~18,000 chars of C code tokenizes cleanly to ~4,200-4,800 tokens (>4096 checkpoint boundary).
    chunks = []
    idx = 0
    while len("\n".join(chunks)) < 18500:
        chunks.append("/* Section %d */\n%s" % (idx, base))
        idx += 1
    body = "\n".join(chunks)[:18500]
    return (
        "Review the following C source code carefully and summarize its memory allocation, "
        "UTF-8 validation, and JSON string escaping invariants in 3 concise bullets:\n\n```c\n"
        + body
        + "\n```\n"
    )


def build_clean_env(extra_env):
    for k in extra_env:
        if k in REFUSED_ENV:
            raise SystemExit("refused env %s: %s" % (k, REFUSED_ENV[k]))
    env = {k: v for k, v in os.environ.items() if not k.startswith("IE_")}
    # True wall-clock defaults: trace OFF, VRAM telemetry + spec timing ON
    env["IE_VRAM_TRACE"] = "1"
    env["IE_SPEC_TIME"] = "1"
    for k, v in extra_env.items():
        if v == "":
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def parse_ie_run_output(stage_name, stdout, stderr, n_prompt, draft, is_raw=False, is_kld=False):
    res = {"stage": stage_name, "n_prompt": n_prompt, "draft": draft}
    mask_lo = sstr(r"ie_inkling_kernels\.hsaco \(mask (0x[0-9a-fA-F]+)\)", stderr)
    mask_hi = sstr(r"ie_inkling_kernels\.hsaco \(mask 0x[0-9a-fA-F]+\) mask2 (0x[0-9a-fA-F]+)", stderr)
    if mask_lo and mask_hi and mask_hi.lower() not in ("0x0", "0x00"):
        res["ink_mask"] = "%s | mask2=%s" % (mask_lo, mask_hi)
    else:
        res["ink_mask"] = mask_lo
    res["arena_bytes"] = inum(r"arena (\d+) bytes", stderr)
    if res["arena_bytes"] is not None:
        res["arena_mb"] = round(res["arena_bytes"] / (1024.0 * 1024.0), 2)
    res["n_ops"] = inum(r"plan: \d+ buffers over (\d+) ops", stderr)
    res["vram_free_mb"] = fnum(r"^gpu: VRAM free after loading: ([0-9.]+) MB of ([0-9.]+) MB", stderr, group=1)
    res["vram_total_mb"] = fnum(r"^gpu: VRAM free after loading: ([0-9.]+) MB of ([0-9.]+) MB", stderr, group=2)

    vr = re.findall(r"^vram: .* -> VRAM ([0-9.]+) GiB, system ([0-9.]+) GiB, budget ([0-9.]+) GiB", stderr, re.M)
    if vr:
        res["vram_used_gib"] = round(max(float(x[0]) for x in vr), 2)
        res["sys_mem_gib"] = round(max(float(x[1]) for x in vr), 2)
        res["vram_budget_gib"] = round(min(float(x[2]) for x in vr), 2)

    res["prefill_ms"] = fnum(r"^prefill: \d+ tokens in ([0-9.]+) ms", stderr)
    res["prefill_tps"] = fnum(r"^prefill: \d+ tokens in [0-9.]+ ms, ([0-9.]+) tokens/s", stderr)

    # Host issue time & launches per token (always emitted by hip.c on T=1)
    hm = re.search(
        r"^gpu: host issue time ([0-9.]+) ms/token \((\d+) launches per token for (\d+) ops\)",
        stderr,
        re.M,
    )
    if hm:
        res["host_issue_ms"] = float(hm.group(1))
        res["launches_per_step"] = int(hm.group(2))
        res["host_ops"] = int(hm.group(3))

    if is_raw:
        vb = {}
        for tm in re.finditer(r"^verify bench: T\s+(\d+)\s+([0-9.]+) ms", stderr, re.M):
            vb[int(tm.group(1))] = float(tm.group(2))
        st = fnum(r"^verify bench: step ([0-9.]+) ms", stderr)
        if st is not None:
            res["vb_step_ms"] = st
        res["vb_table"] = vb
        return res

    if is_kld:
        res["ppl"] = fnum(r"^ppl: .* perplexity ([0-9.]+)", stdout)
        km = re.search(
            r"^kld: (\d+) positions, mean ([0-9.]+), median ([0-9.]+), p90 ([0-9.]+), p99 ([0-9.]+), "
            r"p99\.9 ([0-9.]+), max ([0-9.]+), same top token ([0-9.]+)%",
            stdout,
            re.M,
        )
        if km:
            res["kld_n"] = int(km.group(1))
            res["kld_mean"] = float(km.group(2))
            res["kld_p99"] = float(km.group(5))
            res["kld_max"] = float(km.group(7))
            res["kld_same_top"] = float(km.group(8))
        return res

    im = re.search(r"^ids:(.*)$", stdout, re.M)
    if im:
        ids = im.group(1).split()
        gen = ids[n_prompt:]
        res["n_gen"] = len(gen)
        res["gen_ids_list"] = gen
        res["gen_sha256"] = hashlib.sha256(" ".join(gen).encode("utf-8")).hexdigest()
        res["gen_sha16"] = res["gen_sha256"][:16]
    else:
        res["n_gen"] = 0
        res["gen_ids_list"] = []
        res["gen_sha256"] = None
        res["gen_sha16"] = None

    if draft > 0:
        res["ms_tok"] = fnum(r"^gpu: \d+ tokens in [0-9.]+ ms, ([0-9.]+) ms/token", stderr)
        res["tok_s"] = fnum(r"^gpu: \d+ tokens in [0-9.]+ ms, [0-9.]+ ms/token, ([0-9.]+) tokens/s", stderr)
        res["tps_step"] = fnum(r"^mtp: .* ([0-9.]+) tokens per step", stderr)
        res["accept_pct"] = fnum(r"^mtp: .*\(([0-9.]+)%\)", stderr)
        res["draft_ms"] = fnum(r"^spec time: drafts ([0-9.]+) ms/step", stderr)
        res["verify_ms"] = fnum(r"^spec time: drafts [0-9.]+ ms/step, verify ([0-9.]+) ms/step", stderr)
    else:
        gpu_ms = fnum(r"^gpu: \d+ steps, ([0-9.]+) ms/token", stderr)
        gpu_tps = fnum(r"^gpu: \d+ steps, [0-9.]+ ms/token, ([0-9.]+) tokens/s", stderr)
        res["gpu_ms_tok"] = gpu_ms
        res["gpu_tok_s"] = gpu_tps
        res["gbs"] = fnum(r"^gpu: \d+ steps, .* ([0-9.]+) GB/s", stderr)
        wall_ms = fnum(r"^wall: ([0-9.]+) ms/token", stderr)
        if wall_ms and wall_ms > 0:
            res["ms_tok"] = round(wall_ms, 3)
            res["tok_s"] = round(1000.0 / wall_ms, 2)
        else:
            res["ms_tok"] = gpu_ms
            res["tok_s"] = gpu_tps
    return res


def run_ie_run_stage(stage_name, cmd, env, n_prompt, draft, timeout, logdir, is_raw=False, is_kld=False, dry_run=False):
    if dry_run:
        ie_subset = {k: v for k, v in sorted(env.items()) if k.startswith("IE_")}
        print("[dry-run] %-12s env=%s" % (stage_name, json.dumps(ie_subset)))
        print("          cmd=%s" % " ".join(cmd))
        return {"stage": stage_name, "dry_run": True, "exit": 0, "wall_s": 0.0}

    os.makedirs(logdir, exist_ok=True)
    lock = take_gpu_lock(stage_name)
    t0 = time.time()
    try:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=LAB,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as e:
            said = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace")
            with open(os.path.join(logdir, stage_name + ".err.log"), "w", encoding="utf-8") as f:
                f.write(said)
            write_gpu_fault("inkling2_40r_harness timeout (%ds) in %s\n%s" % (timeout, stage_name, " ".join(cmd)))
            raise SystemExit("TIMEOUT: %s exceeded %ds; wrote ref/.gpu-fault" % (stage_name, timeout))
    finally:
        release_gpu_lock(lock)

    wall_s = round(time.time() - t0, 2)
    with open(os.path.join(logdir, stage_name + ".err.log"), "w", encoding="utf-8") as f:
        f.write(proc.stderr)

    if proc.returncode != 0:
        tail = proc.stderr[-2500:]
        if re.search(r"HIP error|illegal|access fault|timed out|poison", tail, re.I):
            write_gpu_fault("inkling2_40r_harness GPU fault in %s\n%s" % (stage_name, tail[-600:]))
        raise SystemExit("%s FAILED (exit %d):\n%s" % (stage_name, proc.returncode, tail))

    parsed = parse_ie_run_output(stage_name, proc.stdout, proc.stderr, n_prompt, draft, is_raw=is_raw, is_kld=is_kld)
    parsed["exit"] = proc.returncode
    parsed["wall_s"] = wall_s
    return parsed


def http_post_json(url, body, timeout=240):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def start_serve_process(serve_ps1, model, dspark, port, ctx, extra_args, env, log_path):
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        serve_ps1,
        "-Port",
        str(port),
        "-Ctx",
        str(ctx),
        "-Model",
        model,
        "-Dspark",
        dspark,
        "-Draft",
        "5",
        "-DraftPmin",
        "0.6",
    ] + extra_args
    lf = open(log_path, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=LAB)
    url = "http://127.0.0.1:%d" % port
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            lf.close()
            raise SystemExit("ie-serve exited early (%s); see %s" % (proc.returncode, log_path))
        try:
            urllib.request.urlopen(url + "/health", timeout=2).read()
            break
        except Exception:
            if time.time() - t0 > 180:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                proc.wait()
                lf.close()
                write_gpu_fault("ie-serve not ready after 180s (%s)" % log_path)
                raise SystemExit("ie-serve timeout waiting for /health")
            time.sleep(0.5)
    return proc, lf, url


def stop_serve_process(proc, lf):
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        proc.wait(timeout=15)
    except Exception:
        pass
    try:
        lf.close()
    except Exception:
        pass
    time.sleep(1.5)


def run_cache_stage(args, extra_env, logdir, dry_run=False):
    """Measure warm disk-cache 4K restart TTFT, checkpoint size, and exact SHA parity."""
    if dry_run:
        print("[dry-run] cache_4k     ie-serve seed + restart with -DiskCache (port %d)" % args.port)
        return {"dry_run": True}

    cache_dir = tempfile.mkdtemp(prefix="ink40r-dcache-", dir=os.path.join(LAB, "results", "inkling2", "tmp"))
    seed_log = os.path.join(logdir, "cache_seed.err.log")
    resume_log = os.path.join(logdir, "cache_resume.err.log")
    prompt_text = build_cache_text_prompt()

    env_seed = build_clean_env(extra_env)
    env_seed["IE_DS_SPLIT"] = "1"
    env_resume = dict(env_seed)
    if args.poison_cache:
        env_resume["IE_SNAP_POISON_SWA"] = "1"

    lock = take_gpu_lock("cache_4k")
    try:
        # 1. Seed cold pass -> write checkpoint to disk
        proc, lf, url = start_serve_process(
            args.serve_ps1,
            args.model,
            args.dspark,
            args.port,
            16384,
            ["-DiskCache", cache_dir],
            env_seed,
            seed_log,
        )
        try:
            body = {
                "model": "inkling2-uncensored",
                "max_tokens": 16,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt_text}],
            }
            t0 = time.time()
            r_cold = http_post_json(url + "/v1/chat/completions", body, timeout=240)
            cold_wall_ms = round((time.time() - t0) * 1000.0, 1)
            msg = r_cold["choices"][0]["message"]
            cold_text = (msg.get("reasoning_content") or "") + "|" + (msg.get("content") or "")
            cold_sha = hashlib.sha256(cold_text.encode("utf-8")).hexdigest()
            # Give background checkpoint writer up to 10s to finish flushing .iec files
            t_wait = time.time()
            while time.time() - t_wait < 10.0:
                if glob.glob(os.path.join(cache_dir, "*-e-*.iec")):
                    time.sleep(1.0)
                    break
                time.sleep(0.5)
        finally:
            stop_serve_process(proc, lf)

        e_files = glob.glob(os.path.join(cache_dir, "*-e-*.iec"))
        all_files = [
            os.path.join(root, fn)
            for root, _, fns in os.walk(cache_dir)
            for fn in fns
        ]
        ckpt_mb = round(max((os.path.getsize(p) for p in e_files), default=0) / (1024.0 * 1024.0), 2)
        ckpt_total_mb = round(sum(os.path.getsize(p) for p in all_files) / (1024.0 * 1024.0), 2)

        # 2. Warm restart pass -> reload checkpoint from disk
        proc, lf, url = start_serve_process(
            args.serve_ps1,
            args.model,
            args.dspark,
            args.port,
            16384,
            ["-DiskCache", cache_dir],
            env_resume,
            resume_log,
        )
        try:
            t0 = time.time()
            r_warm = http_post_json(url + "/v1/chat/completions", body, timeout=240)
            warm_wall_ms = round((time.time() - t0) * 1000.0, 1)
            msg = r_warm["choices"][0]["message"]
            warm_text = (msg.get("reasoning_content") or "") + "|" + (msg.get("content") or "")
            warm_sha = hashlib.sha256(warm_text.encode("utf-8")).hexdigest()
        finally:
            stop_serve_process(proc, lf)
    finally:
        release_gpu_lock(lock)
        if not args.keep_cache_dir:
            shutil.rmtree(cache_dir, ignore_errors=True)

    with open(resume_log, "r", encoding="utf-8", errors="replace") as f:
        rlog = f.read()

    disk_ms = fnum(r"disk cache: continued from \d+ tokens \([0-9.]+ MB\) after ([0-9.]+) ms", rlog)
    disk_mb = fnum(r"disk cache: continued from \d+ tokens \(([0-9.]+) MB\) after [0-9.]+ ms", rlog)
    hit_toks = inum(r"disk cache: continued from (\d+) tokens", rlog)
    prompt_ms = fnum(r"^\[\d+\] prompt \d+ tokens \(\d+ cached[^,]*, ([0-9.]+) ms;", rlog)

    return {
        "warm_ttft_4k_ms": prompt_ms if prompt_ms is not None else warm_wall_ms,
        "cache_disk_read_ms": disk_ms,
        "cache_read_mb": disk_mb,
        "cache_hit_tokens": hit_toks,
        "ckpt_mb": ckpt_mb,
        "ckpt_total_mb": ckpt_total_mb,
        "cold_cache_wall_ms": cold_wall_ms,
        "warm_cache_wall_ms": warm_wall_ms,
        "cache_exact": (cold_sha == warm_sha),
        "cache_sha16": warm_sha[:16],
    }


def run_serve_stage(args, extra_env, logdir, dry_run=False):
    """Measure live ie-serve sampled chat tok/s, CPU sampling ms/step, and multi-req prefix retention TTFT."""
    if dry_run:
        print(
            "[dry-run] serve_chat   ie-serve 3 sampled + 1 greedy + 3 multi-req prefix check (port %d, %d toks)"
            % (args.port, args.serve_tokens)
        )
        return {"dry_run": True}

    log_path = os.path.join(logdir, "serve_chat.err.log")
    env = build_clean_env(extra_env)
    env["IE_DS_SPLIT"] = "1"
    shared_prefix = build_cache_text_prompt()

    lock = take_gpu_lock("serve_chat")
    greedy_sha = None
    try:
        proc, lf, url = start_serve_process(
            args.serve_ps1,
            args.model,
            args.dspark,
            args.port,
            32768,
            [],
            env,
            log_path,
        )
        try:
            # 1-3: Canonical sampled chat requests (model defaults: thinking temp 1.0, top_k 20, DRY 0.8)
            for txt in SERVE_SAMPLED_PROMPTS:
                http_post_json(
                    url + "/v1/chat/completions",
                    {
                        "model": "inkling2-uncensored",
                        "max_tokens": args.serve_tokens,
                        "messages": [{"role": "user", "content": txt}],
                    },
                    timeout=240,
                )
            # 4: Greedy request
            res_g = http_post_json(
                url + "/v1/chat/completions",
                {
                    "model": "inkling2-uncensored",
                    "max_tokens": args.serve_tokens,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": SERVE_SAMPLED_PROMPTS[1]}],
                },
                timeout=240,
            )
            mg = res_g["choices"][0]["message"]
            gtxt = (mg.get("reasoning_content") or "") + "|" + (mg.get("content") or "")
            greedy_sha = hashlib.sha256(gtxt.encode("utf-8")).hexdigest()[:16]

            # 5-7: Multi-request prefix cache retention check:
            # Long shared prefix -> tiny 150-token side request -> same long shared prefix again
            http_post_json(
                url + "/v1/chat/completions",
                {
                    "model": "inkling2-uncensored",
                    "max_tokens": 16,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": shared_prefix}],
                },
                timeout=240,
            )
            http_post_json(
                url + "/v1/chat/completions",
                {
                    "model": "inkling2-uncensored",
                    "max_tokens": 16,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "user",
                            "content": "List five prime numbers greater than 100 and briefly state why each is prime.",
                        }
                    ],
                },
                timeout=120,
            )
            http_post_json(
                url + "/v1/chat/completions",
                {
                    "model": "inkling2-uncensored",
                    "max_tokens": 16,
                    "temperature": 0,
                    "messages": [
                        {"role": "user", "content": shared_prefix + "\nAlso list all function names."}
                    ],
                },
                timeout=240,
            )
        finally:
            stop_serve_process(proc, lf)
    finally:
        release_gpu_lock(lock)

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        slog = f.read()

    req_lines = re.findall(
        r"^\[(\d+)\] prompt (\d+) tokens \((\d+) cached[^,]*, ([0-9.]+) ms;.*output \d+ tokens in [0-9.]+ s \(([0-9.]+) tokens/s\)",
        slog,
        re.M,
    )
    sampled_tps = [float(x[4]) for x in req_lines[:3]] if len(req_lines) >= 3 else []
    greedy_tps = float(req_lines[3][4]) if len(req_lines) >= 4 else None
    multi_hit_ttft = float(req_lines[6][3]) if len(req_lines) >= 7 else None
    multi_hit_cached = int(req_lines[6][2]) if len(req_lines) >= 7 else None

    samp_ms = [float(x) for x in re.findall(r"^sampling: [0-9.]+ ms in all \(([0-9.]+) ms per step\)", slog, re.M)]
    sampling_ms_step = round(sum(samp_ms) / len(samp_ms), 3) if samp_ms else None

    return {
        "serve_sampled_tok_s": round(sum(sampled_tps) / len(sampled_tps), 2) if sampled_tps else None,
        "serve_sampled_runs": sampled_tps,
        "serve_greedy_tok_s": greedy_tps,
        "serve_greedy_sha16": greedy_sha,
        "sampling_ms_step": sampling_ms_step,
        "multi_req_hit_ttft_ms": multi_hit_ttft,
        "multi_req_cached_tokens": multi_hit_cached,
    }


def load_ledger(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "schema_version": 1,
        "campaign": {
            "title": "Inkling-2-Mini 40-Round SOTA Optimization Campaign",
            "model": "models/Inkling-2-Mini-Uncensored-IQ3_XXS-fast.gguf",
            "dspark": "models/DSpark/Inkling-2-Mini-DSpark-Q4_K.gguf",
            "hardware": "AMD Radeon RX 9070 XT (gfx1201, RDNA4, 64 CUs, 640 GB/s, 16 GiB WDDM)",
            "engine": "amd-infer (ref/tr8-team-merge)",
            "target_rounds": 40,
            "updated_utc": utc_now(),
        },
        "historical_milestones": [],
        "rounds": [],
    }


def save_json_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def find_r00_baseline(ledger):
    for r in ledger.get("rounds", []):
        if r.get("round") == 0:
            return r
    return None


def evaluate_parity(round_num, metrics, r00_entry, kld_res, kld_max, cache_res):
    shas = {
        "t1_sha": metrics.get("t1_sha"),
        "ds5_233_sha": metrics.get("ds5_233_sha"),
        "ds5_4k_sha": metrics.get("ds5_4k_sha"),
        "ds5_16k_sha": metrics.get("ds5_16k_sha"),
    }
    if round_num == 0:
        checks = {k: ("BASELINE" if v else "SKIPPED") for k, v in shas.items()}
        if cache_res and "cache_exact" in cache_res:
            checks["cache_4k"] = "MATCH" if cache_res["cache_exact"] else "MISMATCH"
        ok = all(v != "MISMATCH" for v in checks.values())
        return {
            "status": "BASELINE" if ok else "MISMATCH",
            "exact_all": ok,
            "checks": checks,
            "shas": shas,
            "kld_mean": kld_res.get("kld_mean") if kld_res else None,
            "kld_same_top": kld_res.get("kld_same_top") if kld_res else None,
        }

    if not r00_entry:
        return {
            "status": "NO-REF",
            "exact_all": False,
            "checks": {k: ("UNCHECKED" if v else "SKIPPED") for k, v in shas.items()},
            "shas": shas,
            "kld_mean": kld_res.get("kld_mean") if kld_res else None,
            "kld_same_top": kld_res.get("kld_same_top") if kld_res else None,
        }

    ref_shas = r00_entry.get("parity", {}).get("shas") or {}
    ref_m = r00_entry.get("metrics") or {}
    checks = {}
    any_mismatch = False
    any_checked = False
    for key in ("t1_sha", "ds5_233_sha", "ds5_4k_sha", "ds5_16k_sha"):
        cur = shas.get(key)
        ref = ref_shas.get(key) or ref_m.get(key)
        if not cur:
            checks[key] = "SKIPPED"
        elif not ref:
            checks[key] = "NO-REF"
        else:
            any_checked = True
            if cur == ref:
                checks[key] = "MATCH"
            else:
                checks[key] = "MISMATCH"
                any_mismatch = True

    if cache_res and "cache_exact" in cache_res:
        any_checked = True
        checks["cache_4k"] = "MATCH" if cache_res["cache_exact"] else "MISMATCH"
        if not cache_res["cache_exact"]:
            any_mismatch = True

    kld_mean = kld_res.get("kld_mean") if kld_res else None
    kld_same_top = kld_res.get("kld_same_top") if kld_res else None
    if not any_checked:
        status = "PENDING"
        exact_all = False
    elif not any_mismatch:
        status = "EXACT"
        exact_all = True
    elif kld_mean is not None and kld_mean <= kld_max:
        status = "KLD-OK"
        exact_all = False
    else:
        status = "MISMATCH"
        exact_all = False

    return {
        "status": status,
        "exact_all": exact_all,
        "checks": checks,
        "shas": shas,
        "kld_mean": kld_mean,
        "kld_same_top": kld_same_top,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Inkling-2-Mini 40-round benchmark harness & ledger recorder")
    ap.add_argument("--args-json", default="", help="Optional JSON file with CLI arguments array or dict")

    # Engine & paths
    default_model = DEFAULT_MODEL_FAST if os.path.exists(DEFAULT_MODEL_FAST) else DEFAULT_MODEL_ORIG
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--dspark", default=DEFAULT_DSPARK)
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--hsaco", default=DEFAULT_HSACO)
    ap.add_argument("--serve-ps1", default=DEFAULT_SERVE_PS1)
    ap.add_argument("--ledger", default=DEFAULT_LEDGER)
    ap.add_argument("--html", default=DEFAULT_HTML)

    # Benchmark stage control
    ap.add_argument("--stages", "--only", dest="stages", default=",".join(CORE_STAGES),
                    help="Comma-separated core stages: t1_233,ds5_233,ds5_4k,long_16k")
    ap.add_argument("--with-vbench", action="store_true", help="Also run IE_VERIFY_BENCH=1 (T=1..16)")
    ap.add_argument("--vbench-prompt", choices=("233", "4k", "16k"), default="16k")
    ap.add_argument("--with-cache", action="store_true", help="Run warm disk-cache 4K restart TTFT check via ie-serve")
    ap.add_argument("--poison-cache", action="store_true", help="Set IE_SNAP_POISON_SWA=1 on warm cache resume")
    ap.add_argument("--keep-cache-dir", action="store_true", help="Keep temporary disk-cache directory after test")
    ap.add_argument("--with-serve", action="store_true", help="Run live ie-serve sampled chat + prefix retention check")
    ap.add_argument("--serve-tokens", type=int, default=192, help="Output tokens per live ie-serve chat prompt")
    ap.add_argument("--with-kld", action="store_true", help="Save KLD ref on R00 or evaluate KLD vs R00 on R>0")
    ap.add_argument("--kld-max", type=float, default=0.001, help="Max mean KL divergence for KLD-OK gate")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--mask", default="", help="Override IE_INKLING_KERNEL_MASK (e.g. 0x67dfc0369c3ffff)")
    ap.add_argument("--mask2", default="", help="Override IE_INKLING_KERNEL_MASK2 (bits 64..127)")
    ap.add_argument("--env", action="append", default=[], help="Extra KEY=VAL env var (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands and exit without touching GPU")
    ap.add_argument("--skip-bench", action="store_true", help="Skip GPU runs (for metadata updates or --metrics-json)")
    ap.add_argument("--metrics-json", default="", help="Optional JSON metrics file when using --skip-bench")

    # Round recording flags
    ap.add_argument("--record-round", action="store_true", help="Upsert round into ledger and render HTML dashboard")
    ap.add_argument("--round", type=int, default=None, help="Round number (0..40)")
    ap.add_argument("--wave", default="", help="Wave label (e.g. 0, W1, 2)")
    ap.add_argument("--name", default="", help="Short experiment title")
    ap.add_argument("--target", choices=("TTFT", "Prefill", "Decode", "All"), default="All")
    ap.add_argument("--status", choices=("promoted", "rejected", "baseline"), default="promoted")
    ap.add_argument("--source-title", default="", help="Research paper / repo / engineering inspiration title")
    ap.add_argument("--source-url", default="", help="Clickable URL for the source")
    ap.add_argument("--hypothesis", default="", help="Concise optimization hypothesis")
    ap.add_argument("--notes", default="", help="Measured outcome notes / ablation details")
    ap.add_argument("--launches", type=int, default=None, help="Override launches/step if not auto-parsed")
    ap.add_argument("--allow-nonexact", action="store_true",
                    help="Allow recording promoted status when parity is KLD-OK (or explicitly documented non-exact)")
    ap.add_argument("--no-render", action="store_true", help="Skip auto-rendering HTML dashboard")

    # Support --args-json pre-expansion
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--args-json" in raw_argv:
        idx = raw_argv.index("--args-json")
        if idx + 1 >= len(raw_argv):
            raise SystemExit("--args-json requires a file path")
        with open(lab_path(raw_argv[idx + 1]), "r", encoding="utf-8") as f:
            jdata = json.load(f)
        expanded = []
        if isinstance(jdata, list):
            expanded = [str(x) for x in jdata]
        elif isinstance(jdata, dict):
            for k, v in jdata.items():
                flag = "--" + k.replace("_", "-")
                if isinstance(v, bool):
                    if v:
                        expanded.append(flag)
                elif isinstance(v, list):
                    for item in v:
                        expanded.extend([flag, str(item)])
                elif v is not None:
                    expanded.extend([flag, str(v)])
        else:
            raise SystemExit("--args-json must contain a JSON array or object")
        raw_argv = expanded + raw_argv[:idx] + raw_argv[idx + 2:]

    args = ap.parse_args(raw_argv)
    args.model = lab_path(args.model)
    args.dspark = lab_path(args.dspark)
    args.exe = lab_path(args.exe)
    args.hsaco = lab_path(args.hsaco)
    args.serve_ps1 = lab_path(args.serve_ps1)
    args.ledger = lab_path(args.ledger)
    args.html = lab_path(args.html)
    return args


def print_summary(round_num, metrics, parity):
    print("\n================ 40-ROUND BENCHMARK SUMMARY (R%02d) ================" % (round_num if round_num is not None else 0))
    print("  Parity       : %s  checks=%s" % (parity.get("status"), json.dumps(parity.get("checks", {}))))
    if metrics.get("t1_tok_s") is not None:
        print(
            "  Plain T=1    : %6.2f tok/s  (%6.3f ms/tok, GPU %s ms, %s GB/s, launches=%s)  sha=%s"
            % (
                metrics["t1_tok_s"],
                metrics.get("t1_ms_tok") or 0.0,
                metrics.get("t1_gpu_ms_tok"),
                metrics.get("t1_gbs"),
                metrics.get("launches_per_step"),
                (metrics.get("t1_sha") or "")[:16],
            )
        )
    if metrics.get("ds5_233_tok_s") is not None:
        print(
            "  DS5 233      : %6.2f tok/s  (%.2f tok/step, %.1f%% acc, draft=%s verify=%s ms)  sha=%s"
            % (
                metrics["ds5_233_tok_s"],
                metrics.get("ds5_233_tps_step") or 0.0,
                metrics.get("ds5_233_accept_pct") or 0.0,
                metrics.get("ds5_233_draft_ms"),
                metrics.get("ds5_233_verify_ms"),
                (metrics.get("ds5_233_sha") or "")[:16],
            )
        )
    if metrics.get("ds5_4k_tok_s") is not None or metrics.get("pf4k_tok_s") is not None:
        print(
            "  DS5 4K       : decode %6.2f tok/s (%.2f t/s) | prefill %7.1f tok/s (TTFT %6.1f ms) | arena %s MiB, VRAM free %s MB  sha=%s"
            % (
                metrics.get("ds5_4k_tok_s") or 0.0,
                metrics.get("ds5_4k_tps_step") or 0.0,
                metrics.get("pf4k_tok_s") or 0.0,
                metrics.get("ttft_4k_ms") or 0.0,
                metrics.get("arena_mb"),
                metrics.get("vram_free_mb"),
                (metrics.get("ds5_4k_sha") or "")[:16],
            )
        )
    if metrics.get("ds5_16k_tok_s") is not None or metrics.get("pf16k_tok_s") is not None:
        print(
            "  DS5 16K      : decode %6.2f tok/s (%.2f t/s) | prefill %7.1f tok/s (TTFT %7.1f ms)  sha=%s"
            % (
                metrics.get("ds5_16k_tok_s") or 0.0,
                metrics.get("ds5_16k_tps_step") or 0.0,
                metrics.get("pf16k_tok_s") or 0.0,
                metrics.get("ttft_16k_ms") or 0.0,
                (metrics.get("ds5_16k_sha") or "")[:16],
            )
        )
    if metrics.get("vb_t6_ms") is not None or metrics.get("vb_t8_ms") is not None:
        print("  Verify Bench : T=1 %s ms | T=6 %s ms | T=8 %s ms | T=16 %s ms" % (
            metrics.get("vb_t1_ms"), metrics.get("vb_t6_ms"), metrics.get("vb_t8_ms"), metrics.get("vb_t16_ms")))
    if metrics.get("warm_ttft_4k_ms") is not None:
        print("  Warm Cache 4K: TTFT %.1f ms (disk read %s ms, ckpt %s MiB, exact=%s)" % (
            metrics["warm_ttft_4k_ms"], metrics.get("cache_disk_read_ms"), metrics.get("ckpt_mb"), metrics.get("cache_exact")))
    if metrics.get("serve_sampled_tok_s") is not None:
        print("  Live ie-serve: sampled %.2f tok/s | greedy %s tok/s | sampling %s ms/step | multi-req hit TTFT %s ms (%s cached)" % (
            metrics["serve_sampled_tok_s"], metrics.get("serve_greedy_tok_s"), metrics.get("sampling_ms_step"),
            metrics.get("multi_req_hit_ttft_ms"), metrics.get("multi_req_cached_tokens")))
    print("  Total Wall   : %.1f s  (other VRAM: %s MB)" % (
        metrics.get("total_bench_wall_s") or 0.0, metrics.get("other_vram_mb")))
    print("==================================================================\n", flush=True)


def main(argv=None):
    args = parse_args(argv)

    extra_env = {}
    if args.mask:
        extra_env["IE_INKLING_KERNEL_MASK"] = args.mask
    if args.mask2:
        extra_env["IE_INKLING_KERNEL_MASK2"] = args.mask2
    for kv in args.env:
        if "=" not in kv:
            raise SystemExit("invalid --env '%s' (expected KEY=VAL)" % kv)
        k, v = kv.split("=", 1)
        extra_env[k.strip()] = v.strip()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in CORE_STAGES:
            raise SystemExit("unknown stage '%s' (valid: %s)" % (s, ",".join(CORE_STAGES)))

    round_tag = "r%02d" % args.round if args.round is not None else "adhoc"
    logdir = os.path.join(LAB, "logs", "inkling2", "40round", round_tag)
    metrics = {}
    kld_res = None
    cache_res = None
    serve_res = None
    ink_mask = (("%s | mask2=%s" % (args.mask, args.mask2)) if (args.mask and args.mask2) else (args.mask or None))
    t_all = time.time()

    if args.skip_bench:
        if args.metrics_json:
            with open(lab_path(args.metrics_json), "r", encoding="utf-8") as f:
                metrics.update(json.load(f))
        if args.launches is not None:
            metrics["launches_per_step"] = args.launches
    else:
        if not args.dry_run:
            check_preflight_guards()
            for req_file in (args.exe, args.hsaco, args.model, PROMPT_233, PROMPT_4K):
                if not os.path.exists(req_file):
                    raise SystemExit("missing required file: %s" % req_file)
            if any(s != "t1_233" for s in stages) or args.with_vbench or args.with_cache or args.with_serve:
                if not os.path.exists(args.dspark):
                    raise SystemExit("missing required DSpark file: %s" % args.dspark)
            metrics["other_vram_mb"] = gpu_other_mb()

        n_233 = len(read_token_list(PROMPT_233))
        n_4k = len(read_token_list(PROMPT_4K))
        prompt_16k_path, n_16k = ensure_16k_prompt()

        # 1. t1_233: Plain T=1 decode on real-code-233.txt (--n 64)
        if "t1_233" in stages:
            env = build_clean_env(extra_env)
            cmd = [
                args.exe, args.model, "--backend", "gpu", "--hsaco", args.hsaco,
                "--tokens-file", PROMPT_233, "--ctx", "32768", "--chunk", "512",
                "--n", "64", "--draft", "0",
            ]
            r = run_ie_run_stage("t1_233", cmd, env, n_233, 0, args.timeout, logdir, dry_run=args.dry_run)
            if not args.dry_run:
                metrics["t1_tok_s"] = r.get("tok_s")
                metrics["t1_ms_tok"] = r.get("ms_tok")
                metrics["t1_gpu_ms_tok"] = r.get("gpu_ms_tok")
                metrics["t1_gbs"] = r.get("gbs")
                metrics["t1_sha"] = r.get("gen_sha256")
                metrics["t1_pf_ms"] = r.get("prefill_ms")
                if r.get("launches_per_step") is not None:
                    metrics["launches_per_step"] = r["launches_per_step"]
                if r.get("host_issue_ms") is not None:
                    metrics["host_issue_ms"] = r["host_issue_ms"]
                if r.get("ink_mask"):
                    ink_mask = r["ink_mask"]
                if r.get("arena_mb") is not None and "arena_mb" not in metrics:
                    metrics["arena_mb"] = r["arena_mb"]
                    metrics["arena_bytes"] = r["arena_bytes"]
                if r.get("vram_free_mb") is not None and "vram_free_mb" not in metrics:
                    metrics["vram_free_mb"] = r["vram_free_mb"]
                    metrics["vram_used_gib"] = r.get("vram_used_gib")
                    metrics["vram_budget_gib"] = r.get("vram_budget_gib")
                print("[ok] t1_233   : %.2f tok/s (%.3f ms/tok, launches=%s, sha=%s) in %.1fs" % (
                    metrics["t1_tok_s"] or 0, metrics["t1_ms_tok"] or 0,
                    metrics.get("launches_per_step"), (metrics.get("t1_sha") or "")[:16], r["wall_s"]), flush=True)

        # 2. ds5_233: DSpark 5x0.6 split-head on real-code-233.txt (--n 128)
        if "ds5_233" in stages:
            env = build_clean_env(dict(extra_env, IE_DS_SPLIT="1"))
            cmd = [
                args.exe, args.model, "--backend", "gpu", "--hsaco", args.hsaco,
                "--tokens-file", PROMPT_233, "--ctx", "32768", "--chunk", "512",
                "--n", "128", "--draft", "5", "--dspark", args.dspark, "--draft-pmin", "0.6",
            ]
            r = run_ie_run_stage("ds5_233", cmd, env, n_233, 5, args.timeout, logdir, dry_run=args.dry_run)
            if not args.dry_run:
                metrics["ds5_233_tok_s"] = r.get("tok_s")
                metrics["ds5_233_ms_tok"] = r.get("ms_tok")
                metrics["ds5_233_tps_step"] = r.get("tps_step")
                metrics["ds5_233_accept_pct"] = r.get("accept_pct")
                metrics["ds5_233_draft_ms"] = r.get("draft_ms")
                metrics["ds5_233_verify_ms"] = r.get("verify_ms")
                metrics["ds5_233_sha"] = r.get("gen_sha256")
                if r.get("ink_mask"):
                    ink_mask = r["ink_mask"]
                print("[ok] ds5_233  : %.2f tok/s (%.2f tok/step, draft=%s verify=%s ms, sha=%s) in %.1fs" % (
                    metrics["ds5_233_tok_s"] or 0, metrics["ds5_233_tps_step"] or 0,
                    metrics.get("ds5_233_draft_ms"), metrics.get("ds5_233_verify_ms"),
                    (metrics.get("ds5_233_sha") or "")[:16], r["wall_s"]), flush=True)

        # 3. ds5_4k: DSpark 5x0.6 split-head on real-code-4k.txt (--n 64, --chunk 512)
        if "ds5_4k" in stages:
            env = build_clean_env(dict(extra_env, IE_DS_SPLIT="1"))
            cmd = [
                args.exe, args.model, "--backend", "gpu", "--hsaco", args.hsaco,
                "--tokens-file", PROMPT_4K, "--ctx", "32768", "--chunk", "512",
                "--n", "64", "--draft", "5", "--dspark", args.dspark, "--draft-pmin", "0.6",
            ]
            r = run_ie_run_stage("ds5_4k", cmd, env, n_4k, 5, args.timeout, logdir, dry_run=args.dry_run)
            if not args.dry_run:
                metrics["pf4k_tok_s"] = r.get("prefill_tps")
                metrics["ttft_4k_ms"] = r.get("prefill_ms")
                metrics["ds5_4k_tok_s"] = r.get("tok_s")
                metrics["ds5_4k_ms_tok"] = r.get("ms_tok")
                metrics["ds5_4k_tps_step"] = r.get("tps_step")
                metrics["ds5_4k_accept_pct"] = r.get("accept_pct")
                metrics["ds5_4k_draft_ms"] = r.get("draft_ms")
                metrics["ds5_4k_verify_ms"] = r.get("verify_ms")
                metrics["ds5_4k_sha"] = r.get("gen_sha256")
                metrics["arena_mb"] = r.get("arena_mb")
                metrics["arena_bytes"] = r.get("arena_bytes")
                metrics["vram_free_mb"] = r.get("vram_free_mb")
                metrics["vram_total_mb"] = r.get("vram_total_mb")
                metrics["vram_used_gib"] = r.get("vram_used_gib")
                metrics["vram_budget_gib"] = r.get("vram_budget_gib")
                metrics["sys_mem_gib"] = r.get("sys_mem_gib")
                if r.get("ink_mask"):
                    ink_mask = r["ink_mask"]
                print("[ok] ds5_4k   : pf %.1f tok/s (TTFT %.1f ms) | dec %.2f tok/s (%.2f t/s) | arena %s MiB free %s MB sha=%s in %.1fs" % (
                    metrics["pf4k_tok_s"] or 0, metrics["ttft_4k_ms"] or 0,
                    metrics["ds5_4k_tok_s"] or 0, metrics["ds5_4k_tps_step"] or 0,
                    metrics.get("arena_mb"), metrics.get("vram_free_mb"),
                    (metrics.get("ds5_4k_sha") or "")[:16], r["wall_s"]), flush=True)

        # 4. long_16k: 16K real-code prompt (--ctx 32768, --chunk 512, --n 32, DSpark 5x0.6)
        if "long_16k" in stages:
            env = build_clean_env(dict(extra_env, IE_DS_SPLIT="1"))
            cmd = [
                args.exe, args.model, "--backend", "gpu", "--hsaco", args.hsaco,
                "--tokens-file", prompt_16k_path, "--ctx", "32768", "--chunk", "512",
                "--n", "32", "--draft", "5", "--dspark", args.dspark, "--draft-pmin", "0.6",
            ]
            r = run_ie_run_stage("long_16k", cmd, env, n_16k, 5, args.timeout, logdir, dry_run=args.dry_run)
            if not args.dry_run:
                metrics["pf16k_tok_s"] = r.get("prefill_tps")
                metrics["ttft_16k_ms"] = r.get("prefill_ms")
                metrics["ds5_16k_tok_s"] = r.get("tok_s")
                metrics["ds5_16k_ms_tok"] = r.get("ms_tok")
                metrics["ds5_16k_tps_step"] = r.get("tps_step")
                metrics["ds5_16k_accept_pct"] = r.get("accept_pct")
                metrics["ds5_16k_draft_ms"] = r.get("draft_ms")
                metrics["ds5_16k_verify_ms"] = r.get("verify_ms")
                metrics["ds5_16k_sha"] = r.get("gen_sha256")
                if r.get("ink_mask"):
                    ink_mask = r["ink_mask"]
                print("[ok] long_16k : pf %.1f tok/s (TTFT %.1f ms) | dec %.2f tok/s (%.2f t/s) sha=%s in %.1fs" % (
                    metrics["pf16k_tok_s"] or 0, metrics["ttft_16k_ms"] or 0,
                    metrics["ds5_16k_tok_s"] or 0, metrics["ds5_16k_tps_step"] or 0,
                    (metrics.get("ds5_16k_sha") or "")[:16], r["wall_s"]), flush=True)

        # 4b. Optional verify-bench (IE_VERIFY_BENCH=1)
        if args.with_vbench:
            vb_prompt = {"233": PROMPT_233, "4k": PROMPT_4K, "16k": prompt_16k_path}[args.vbench_prompt]
            vb_n = {"233": n_233, "4k": n_4k, "16k": n_16k}[args.vbench_prompt]
            env = build_clean_env(dict(extra_env, IE_VERIFY_BENCH="1"))
            cmd = [
                args.exe, args.model, "--backend", "gpu", "--hsaco", args.hsaco,
                "--tokens-file", vb_prompt, "--ctx", "32768", "--chunk", "512",
                "--n", "16", "--draft", "7", "--dspark", args.dspark, "--draft-pmin", "0.5",
            ]
            r = run_ie_run_stage("vbench_" + args.vbench_prompt, cmd, env, vb_n, 7, args.timeout, logdir, is_raw=True, dry_run=args.dry_run)
            if not args.dry_run:
                tbl = r.get("vb_table", {})
                metrics["vb_prompt"] = args.vbench_prompt
                metrics["vb_t1_ms"] = tbl.get(1)
                metrics["vb_t6_ms"] = tbl.get(6)
                metrics["vb_t8_ms"] = tbl.get(8)
                metrics["vb_t16_ms"] = tbl.get(16)
                metrics["vb_step_ms"] = r.get("vb_step_ms")
                print("[ok] vbench   : T=1 %s | T=6 %s | T=8 %s | T=16 %s ms in %.1fs" % (
                    metrics["vb_t1_ms"], metrics["vb_t6_ms"], metrics["vb_t8_ms"], metrics["vb_t16_ms"], r["wall_s"]), flush=True)

        # 5a. Optional KLD gate
        if args.with_kld:
            os.makedirs(os.path.dirname(KLD_REF_FILE), exist_ok=True)
            is_save = (args.round == 0) or (not os.path.exists(KLD_REF_FILE))
            env = build_clean_env(extra_env)
            cmd = [
                args.exe, args.model, "--backend", "gpu", "--hsaco", args.hsaco,
                "--tokens-file", PROMPT_4K, "--ctx", "32768", "--chunk", "512",
                "--ppl", "--ppl-first", "64",
                "--kld-save" if is_save else "--kld", KLD_REF_FILE,
                "--draft", "0",
            ]
            kld_res = run_ie_run_stage("kld_4k", cmd, env, n_4k, 0, args.timeout, logdir, is_kld=True, dry_run=args.dry_run)
            if not args.dry_run:
                metrics["ppl_4k"] = kld_res.get("ppl")
                if kld_res.get("kld_mean") is not None:
                    metrics["kld_mean"] = kld_res["kld_mean"]
                    metrics["kld_p99"] = kld_res.get("kld_p99")
                    metrics["kld_same_top"] = kld_res.get("kld_same_top")
                    print("[ok] kld_4k   : mean=%.6f p99=%.6f same_top=%.2f%%" % (
                        kld_res["kld_mean"], kld_res.get("kld_p99") or 0, kld_res.get("kld_same_top") or 0), flush=True)
                else:
                    print("[ok] kld_4k   : saved reference logits to %s (ppl=%s)" % (
                        os.path.relpath(KLD_REF_FILE, LAB), kld_res.get("ppl")), flush=True)

        # 5b. Optional warm disk-cache 4K restart TTFT
        if args.with_cache:
            cache_res = run_cache_stage(args, extra_env, logdir, dry_run=args.dry_run)
            if not args.dry_run:
                metrics.update(cache_res)
                print("[ok] cache_4k : warm TTFT %.1f ms (disk %.1f ms, ckpt %.2f MiB, exact=%s)" % (
                    metrics.get("warm_ttft_4k_ms") or 0, metrics.get("cache_disk_read_ms") or 0,
                    metrics.get("ckpt_mb") or 0, metrics.get("cache_exact")), flush=True)

        # 5c. Optional live ie-serve sampled chat + prefix retention
        if args.with_serve:
            serve_res = run_serve_stage(args, extra_env, logdir, dry_run=args.dry_run)
            if not args.dry_run:
                metrics.update(serve_res)
                print("[ok] serve    : sampled %.2f tok/s | sampling %s ms/step | multi-req hit TTFT %s ms (%s cached)" % (
                    metrics.get("serve_sampled_tok_s") or 0, metrics.get("sampling_ms_step"),
                    metrics.get("multi_req_hit_ttft_ms"), metrics.get("multi_req_cached_tokens")), flush=True)

        if args.launches is not None:
            metrics["launches_per_step"] = args.launches
        metrics["total_bench_wall_s"] = round(time.time() - t_all, 1)

    if args.dry_run:
        print("[dry-run] complete (0 GPU commands executed).")
        return 0

    ledger = load_ledger(args.ledger)
    r00 = find_r00_baseline(ledger)
    round_num = args.round if args.round is not None else -1
    parity = evaluate_parity(round_num, metrics, r00, kld_res, args.kld_max, cache_res)
    print_summary(args.round, metrics, parity)

    if not args.record_round:
        return 1 if parity["status"] == "MISMATCH" else 0

    if args.round is None:
        raise SystemExit("--record-round requires --round N (0..40)")
    if not args.name:
        raise SystemExit("--record-round requires --name '...'")

    # Enforce parity gate on promoted rounds
    if args.status == "promoted" and parity["status"] == "MISMATCH" and not args.allow_nonexact:
        raise SystemExit(
            "PARITY GATE REFUSED PROMOTION for R%02d (%s): %s. "
            "Record as --status rejected, or re-run with --with-kld and --allow-nonexact if KLD <= %.4f."
            % (args.round, args.name, json.dumps(parity["checks"]), args.kld_max)
        )

    entry = {
        "round": int(args.round),
        "wave": str(args.wave) if args.wave != "" else ("0" if args.round == 0 else ""),
        "name": args.name,
        "target": args.target,
        "status": "baseline" if args.round == 0 else args.status,
        "timestamp_utc": utc_now(),
        "source": {
            "title": args.source_title or "",
            "url": args.source_url or "",
        },
        "hypothesis": args.hypothesis or "",
        "notes": args.notes or "",
        "ink_mask": ink_mask or "",
        "env": extra_env,
        "parity": parity,
        "metrics": metrics,
    }

    # Upsert round in sorted order
    rounds = [r for r in ledger.get("rounds", []) if r.get("round") != entry["round"]]
    rounds.append(entry)
    rounds.sort(key=lambda r: int(r.get("round", 0)))
    ledger["rounds"] = rounds
    ledger.setdefault("campaign", {})["updated_utc"] = utc_now()
    save_json_atomic(args.ledger, ledger)
    print("[saved] updated %s (round %d, total %d rounds)" % (
        os.path.relpath(args.ledger, LAB), entry["round"], len(rounds)), flush=True)

    if not args.no_render:
        sys.path.insert(0, BENCH_DIR)
        import render_40r_dashboard  # noqa: E402
        render_40r_dashboard.render_dashboard(args.ledger, args.html)
        print("[rendered] updated %s" % os.path.relpath(args.html, LAB), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
