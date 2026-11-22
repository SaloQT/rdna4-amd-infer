"""topk_bench.py -- the sampling top-K kernels alone: ie_topk_part + ie_topk_merge (K passes over a segment) against
ie_topk_max + ie_topk_sel (threshold from the segment maxima), same answers, and their time.

  python tools/topk_bench.py [build/ie_kernels.hsaco] [-k 20] [-r 4]
Run it through ref\\gpu-run.ps1 (one GPU job at a time):
  pwsh -File ref\\gpu-run.ps1 -- python.exe tools/topk_bench.py

Rows are 248320 random logits (normal, a few planted large values, ties, equal values, -inf runs). The check is
against numpy (value descending, equal values smaller index first) and between the two kernel pairs; a case where
a row has more than 1024 candidates (ie_topk_sel writes index 0xFFFFFFFF) is reported as 'slow path'.
Time: us per call back to back (200 calls), and one call alone (event to event, after a pause).
"""
import ctypes as C
import os
import sys
import time

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
core = os.path.join(here, "..", "..", "..", ".rocm-venv", "Lib", "site-packages", "_rocm_sdk_core", "bin")
for d in (core, os.path.join(core, "..", "..", "_rocm_sdk_libraries", "bin")):
    if os.path.isdir(d):
        os.add_dll_directory(os.path.abspath(d))
H = C.CDLL(os.path.join(os.path.abspath(core), "amdhip64_7.dll"))

TK_MAX, TK_CS, TK_ROWS, TK_NG, TF_CAP = 64, 4096, 16, 64, 1024
NV = 248320


def ck(e, what):
    if e:
        sys.exit("HIP error %d in %s" % (e, what))


def dmalloc(n):
    p = C.c_void_p()
    ck(H.hipMalloc(C.byref(p), C.c_size_t(n)), "hipMalloc")
    return p


def off(p, nbytes):
    return C.c_void_p(p.value + nbytes)


def main():
    a = sys.argv[1:]
    K, nrows = 20, 4
    if "-k" in a:
        i = a.index("-k"); K = int(a[i + 1]); del a[i:i + 2]
    if "-r" in a:
        i = a.index("-r"); nrows = int(a[i + 1]); del a[i:i + 2]
    hsaco = a[0] if a else os.path.join(here, "..", "build", "ie_kernels.hsaco")
    ck(H.hipInit(0), "hipInit")
    mod = C.c_void_p()
    ck(H.hipModuleLoad(C.byref(mod), os.path.abspath(hsaco).encode()), "hipModuleLoad")
    fn = {}
    for name in ("ie_topk_part", "ie_topk_merge", "ie_topk_max", "ie_topk_sel"):
        f = C.c_void_p()
        ck(H.hipModuleGetFunction(C.byref(f), mod, name.encode()), name)
        fn[name] = f

    xs = (NV * 4 + 255) // 256 * 64  # floats per row, as the engine's 256-byte aligned rows
    x = dmalloc(TK_ROWS * xs * 4)
    tk = dmalloc(TK_ROWS * (TK_NG + 1) * TK_MAX * 8)
    ng = (NV + TK_CS - 1) // TK_CS
    cs = (NV + ng - 1) // ng
    cv = tk
    ci = off(tk, TK_ROWS * TK_NG * TK_MAX * 4)
    ov = off(tk, TK_ROWS * TK_NG * TK_MAX * 8)
    oi = off(ov, TK_ROWS * TK_MAX * 4)
    smax = off(cv, TK_ROWS * TF_CAP * 4)
    cnt = off(ci, TK_ROWS * TF_CAP * 4)
    fin = off(cnt, TK_ROWS * 4)

    def args(*vals):
        return (C.c_void_p * len(vals))(*[C.cast(C.byref(v), C.c_void_p) for v in vals])

    u = C.c_uint32
    nv_, xs_, K_, cs_, ng_ = u(NV), u(xs), u(K), u(cs), u(ng)
    nc_ = u(ng * K)
    xp = C.c_void_p(x.value)
    cvp, cip, ovp, oip, smp, cnp, fnp = (C.c_void_p(p.value) for p in (cv, ci, ov, oi, smax, cnt, fin))
    a_part = args(xp, nv_, xs_, K_, cs_, ng_, cvp, cip)
    a_merge = args(cvp, cip, nc_, K_, ovp, oip)
    a_max = args(xp, nv_, xs_, cs_, smp, cnp, fnp)
    a_sel = args(xp, nv_, xs_, K_, cs_, ng_, smp, cvp, cip, cnp, fnp, ovp, oip)

    def launch(name, gx, gy, a):
        ck(H.hipModuleLaunchKernel(fn[name], gx, gy, 1, 256, 1, 1, 0, None, a, None), "launch " + name)

    def slow(n):
        launch("ie_topk_part", ng, n, a_part)
        launch("ie_topk_merge", n, 1, a_merge)

    def fast(n):
        launch("ie_topk_max", ng, n, a_max)
        launch("ie_topk_sel", ng, n, a_sel)

    def result(n):
        res = C.create_string_buffer(TK_ROWS * TK_MAX * 8)
        ck(H.hipMemcpy(res, ov, C.c_size_t(len(res)), 2), "d2h")
        v = np.frombuffer(res.raw, dtype=np.float32, count=TK_ROWS * TK_MAX)
        i = np.frombuffer(res.raw, dtype=np.uint32, count=TK_ROWS * TK_MAX, offset=TK_ROWS * TK_MAX * 4)
        return v[:n * K].reshape(n, K).copy(), i[:n * K].reshape(n, K).copy()

    rng = np.random.default_rng(5)

    def make(case, n):
        rows = np.empty((n, NV), dtype=np.float32)
        for r in range(n):
            row = rng.normal(0.0, 3.0, NV).astype(np.float32)
            if case == "peaked":  # a few large logits, clustered in id space like frequent tokens
                row[rng.integers(0, 2000, 30)] += rng.uniform(8, 25, 30).astype(np.float32)
            elif case == "flat":
                row *= 0.05
            elif case == "ties":  # few distinct values: many equal logits
                row = np.round(row * 2) / 2
            elif case == "equal":
                row[:] = 1.5
            elif case == "inf":
                row[rng.random(NV) < 0.97] = -np.inf
            elif case == "late":  # the best values are at the end of the row
                row[-300:] += 30.0
            rows[r] = row
        return rows

    def ref(rows):
        n = rows.shape[0]
        ov_ = np.empty((n, K), np.float32)
        oi_ = np.empty((n, K), np.uint32)
        for r in range(n):
            order = np.lexsort((np.arange(NV), -rows[r].astype(np.float64)))[:K]
            ov_[r], oi_[r] = rows[r][order], order
        return ov_, oi_

    ok = True
    for case in ("normal", "peaked", "flat", "ties", "equal", "inf", "late"):
        for n in (1, nrows, 8):
            rows = make(case, n)
            pad = np.zeros((n, xs), np.float32)
            pad[:, :NV] = rows
            ck(H.hipMemcpy(x, pad.ctypes.data_as(C.c_void_p), C.c_size_t(pad.nbytes), 1), "h2d")
            rv, ri = ref(rows)
            slow(n)
            sv, si = result(n)
            fast(n)
            fv, fi = result(n)
            slow_path = bool((fi[:, 0] == 0xFFFFFFFF).any())
            same_slow = np.array_equal(sv, rv) and np.array_equal(si, ri)
            same_fast = slow_path or (np.array_equal(fv, rv) and np.array_equal(fi, ri))
            ok &= same_slow and same_fast
            print("%-7s rows %d K %d: old==numpy %s, new==numpy %s%s" % (
                case, n, K, same_slow, same_fast, "  (fast kernels gave up: slow path)" if slow_path else ""))
    print("ALL SAME" if ok else "MISMATCH")

    # time on the 'peaked' case
    rows = make("peaked", nrows)
    pad = np.zeros((nrows, xs), np.float32)
    pad[:, :NV] = rows
    ck(H.hipMemcpy(x, pad.ctypes.data_as(C.c_void_p), C.c_size_t(pad.nbytes), 1), "h2d")
    e0, e1 = C.c_void_p(), C.c_void_p()
    ck(H.hipEventCreate(C.byref(e0)), "ev"), ck(H.hipEventCreate(C.byref(e1)), "ev")
    for label, f in (("old (part+merge)", slow), ("new (max+sel)   ", fast), ("old (part+merge)", slow), ("new (max+sel)   ", fast)):
        for _ in range(20):
            f(nrows)
        ck(H.hipDeviceSynchronize(), "sync")
        ck(H.hipEventRecord(e0, None), "rec")
        for _ in range(200):
            f(nrows)
        ck(H.hipEventRecord(e1, None), "rec")
        ck(H.hipEventSynchronize(e1), "sync")
        ms = C.c_float()
        ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
        alone = []
        for _ in range(10):
            time.sleep(0.05)
            ck(H.hipEventRecord(e0, None), "rec")
            f(nrows)
            ck(H.hipEventRecord(e1, None), "rec")
            ck(H.hipEventSynchronize(e1), "sync")
            t = C.c_float()
            ck(H.hipEventElapsedTime(C.byref(t), e0, e1), "elapsed")
            alone.append(t.value * 1000)
        alone.sort()
        print("%s rows %d: %.1f us per call back to back, %.1f us alone after a pause (median of 10, min %.1f)" % (
            label, nrows, ms.value * 1000 / 200, alone[5], alone[0]))


main()
