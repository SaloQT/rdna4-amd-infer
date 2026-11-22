"""raw_gemm_bench.py -- the matrix-core prompt-chunk kernels (ie_gemm_rq_*) against the multi-token GEMVs
(ie_gemv_raw_*_tr), on Saluki's shapes.

  python tools/raw_gemm_bench.py TR.hsaco GEMM.hsaco [-k iq2xxs,q2k] [-s 17408x5120,...] [-t 32,64,256] [-gt 128]
                                 [-gr 128] [-q] [-nocheck] [-notr]

TR.hsaco holds the ie_gemv_raw_*_tr kernels (the baseline, launched 8 tokens at a time as the engine does),
GEMM.hsaco the ie_gemm_rq_* kernels (it may be the same file). -gt/-gr: tokens and rows per workgroup of the
GEMM build (RQ_T, RQ_R of kernels/ie_gemm_raw.inc). -q: skip the timing (correctness only).

For each type, shape and token count T:
  correctness: one sanitised weight copy (valid fp16 scales), random Q8 activations; the GEMM output against
    the TR output: max |gemm - tr| / max |tr| (and the largest relative difference of the entries above 1% of
    the max), once without bias/residual and once with both;
  speed: the weights spread over a 512 MiB pool (each launch reads another copy, so the bytes come from DRAM
    as in a model); best of 3 runs of 5 timed launches; us per T-token chunk for the TR loop and for the
    GEMM, the ratio, and the GEMM's int8 TOPS.
"""
import ctypes as C
import math
import os
import random
import struct
import sys

import numpy as np

ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib",
                        "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")

TYPES = {"q2k": (10, 84), "iq2xxs": (16, 66), "iq2xs": (17, 74), "iq3xxs": (18, 98), "iq1s": (19, 50),
         "iq3s": (21, 110), "iq2s": (22, 82), "iq1m": (29, 56)}
# fp16 scale offsets inside a block (sanitised so every output is finite)
D_OFF = {"q2k": (80, 82), "iq2xxs": (0,), "iq2xs": (0,), "iq3xxs": (0,), "iq1s": (0,), "iq3s": (0,), "iq2s": (0,),
         "iq1m": ()}
SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144), (1024, 5120)]
POOL = 512 << 20
MAX_WBYTES = 128 << 20  # the largest weight copy the check keeps


def ck(e, what):
    if e:
        sys.exit("HIP error %d in %s" % (e, what))


def dmalloc(n):
    p = C.c_void_p()
    ck(H.hipMalloc(C.byref(p), C.c_size_t(n)), "hipMalloc")
    return p


def h2d(dst, data, off=0):
    ck(H.hipMemcpy(C.c_void_p(dst.value + off), data, C.c_size_t(len(data)), 1), "h2d")


def d2h(src, n):
    b = C.create_string_buffer(n)
    ck(H.hipMemcpy(b, src, C.c_size_t(n), 2), "d2h")
    return b.raw


class Mod:
    def __init__(self, path):
        self.m = C.c_void_p()
        ck(H.hipModuleLoad(C.byref(self.m), path.encode()), "hipModuleLoad " + path)
        self.f = {}

    def fn(self, name):
        if name not in self.f:
            f = C.c_void_p()
            ck(H.hipModuleGetFunction(C.byref(f), self.m, name.encode()), "get " + name)
            self.f[name] = f
        return self.f[name]


def groups(rows):  # the engine's GEMV grid (gemv_groups)
    g = (rows + 7) // 8
    return 257 if 253 <= g <= 256 else g


def sanitise(kind, blk, bs):
    for o in D_OFF[kind]:
        for b in range(o, len(blk), bs):
            blk[b], blk[b + 1] = 0x00, 0x2C  # 0.0625
    if kind == "iq1m":  # d is split over the high nibbles of the four scale words: 0x2C00
        for b in range(0, len(blk), bs):
            blk[b + 49] &= 0x0F                                   # nibble 0 of d = 0
            blk[b + 51] &= 0x0F                                   # nibble 1 = 0
            blk[b + 53] = (blk[b + 53] & 0x0F) | 0xC0             # nibble 2 = 0xC
            blk[b + 55] = (blk[b + 55] & 0x0F) | 0x20             # nibble 3 = 2


def main():
    a = sys.argv[1:]
    kinds, shapes, ts = list(TYPES), SHAPES, [32, 64, 256]
    gt, gr, timing, check = 128, 128, True, True
    if "-k" in a:
        i = a.index("-k"); kinds = a[i + 1].split(","); del a[i:i + 2]
    if "-s" in a:
        i = a.index("-s"); shapes = [tuple(int(v) for v in s.split("x")) for s in a[i + 1].split(",")]; del a[i:i + 2]
    if "-t" in a:
        i = a.index("-t"); ts = [int(v) for v in a[i + 1].split(",")]; del a[i:i + 2]
    if "-gt" in a:
        i = a.index("-gt"); gt = int(a[i + 1]); del a[i:i + 2]
    if "-gr" in a:
        i = a.index("-gr"); gr = int(a[i + 1]); del a[i:i + 2]
    if "-q" in a:
        a.remove("-q"); timing = False
    notr = "-notr" in a  # no TR timing
    if notr:
        a.remove("-notr")
    if "-nocheck" in a:
        a.remove("-nocheck"); check = False
    if len(a) not in (1, 2):
        sys.exit(__doc__)
    ck(H.hipInit(0), "hipInit")
    tr_mod = Mod(a[0])
    gm_mod = Mod(a[1]) if len(a) == 2 else tr_mod
    rnd = random.Random(7)
    for t in ts:
        if t < 17:
            sys.exit("T must be >= 17 (the engine's IE_GEMM_MIN)")
    maxT = max(ts)
    maxcols = max(c for _, c in shapes)
    maxrows = max(r for r, _ in shapes)
    pool = dmalloc(POOL)
    for off in range(0, POOL, 64 << 20):
        h2d(pool, rnd.randbytes(64 << 20), off)
    wchk = dmalloc(MAX_WBYTES)
    e0, e1 = C.c_void_p(), C.c_void_p()
    ck(H.hipEventCreate(C.byref(e0)), "ev"), ck(H.hipEventCreate(C.byref(e1)), "ev")
    nbm = maxcols // 32
    xs_max = nbm * 40
    xq = dmalloc(xs_max * maxT)
    y_tr = dmalloc(maxrows * 4 * maxT)
    y_gm = dmalloc(maxrows * 4 * maxT)
    bias = dmalloc(maxrows * 4)
    res = dmalloc(maxrows * 4 * maxT)
    bias_h = struct.pack("<%df" % maxrows, *[0.25 * ((i % 13) - 6) for i in range(maxrows)])
    h2d(bias, bias_h)
    res_h = struct.pack("<%df" % (maxrows * maxT), *[0.5 * ((i % 11) - 5) for i in range(maxrows * maxT)])
    h2d(res, res_h)
    print("%-7s %13s %4s %10s %10s %7s %9s %9s %9s" % ("type", "shape", "T", "TR us", "GEMM us", "x", "TOPS", "rel", "rel+b/r"))
    worst = 0.0
    for kind in kinds:
        tid, bs = TYPES[kind]
        nm_tr, nm_gm = "ie_gemv_raw_%s_tr" % kind, "ie_gemm_rq_%s" % kind
        for rows, cols in shapes:
            nb = cols // 32
            xs = nb * 40
            ys = rows * 4
            mbytes = rows * (nb // 8) * bs
            # activations: int8 in [-127, 127], da, int32 sums, for the largest T (smaller T use a prefix)
            nrng = np.random.default_rng(11)
            x8 = nrng.integers(-127, 128, size=(maxT, nb * 32), dtype=np.int8)
            da = (0.002 + 0.0007 * ((np.arange(maxT)[:, None] + np.arange(nb)[None, :]) % 7)).astype(np.float32)
            sums = x8.reshape(maxT, nb, 32).astype(np.int32).sum(axis=2).astype(np.int32)
            h2d(xq, b"".join(x8[k].tobytes() + da[k].tobytes() + sums[k].tobytes() for k in range(maxT)))
            ncopy = max(1, POOL // mbytes)

            def launch_tr(w, T, b=None, r=None, out=y_tr):
                t0 = 0
                while t0 < T:
                    tt = min(8, T - t0)
                    vals = [C.c_void_p(w), C.c_void_p(xq.value + t0 * xs), C.c_void_p(out.value + t0 * ys), C.c_uint32(rows),
                            C.c_uint32(nb), C.c_uint32(tt), C.c_void_p(b.value if b else 0),
                            C.c_void_p(r.value + t0 * ys if r else 0), C.c_void_p(), C.c_void_p(), C.c_void_p(),
                            C.c_float(0.0), C.c_void_p(), C.c_uint32(xs), C.c_uint32(ys), C.c_uint32(ys)]
                    args = (C.c_void_p * len(vals))(*[C.cast(C.byref(v), C.c_void_p) for v in vals])
                    ck(H.hipModuleLaunchKernel(tr_mod.fn(nm_tr), groups(rows), 1, 1, 256, 1, 1, 0, None, args, None), "launch tr")
                    t0 += tt

            def launch_gm(w, T, b=None, r=None, out=y_gm):
                vals = [C.c_void_p(w), C.c_void_p(xq.value), C.c_void_p(out.value), C.c_uint32(rows), C.c_uint32(nb),
                        C.c_uint32(T), C.c_void_p(b.value if b else 0), C.c_void_p(r.value if r else 0),
                        C.c_uint32(xs), C.c_uint32(ys), C.c_uint32(ys)]
                args = (C.c_void_p * len(vals))(*[C.cast(C.byref(v), C.c_void_p) for v in vals])
                nrt, ntt = (rows + gr - 1) // gr, (T + gt - 1) // gt
                ck(H.hipModuleLaunchKernel(gm_mod.fn(nm_gm), nrt * ntt, 1, 1, 256, 1, 1, 0, None, args, None), "launch gemm")

            # one sanitised weight copy for the correctness check (valid fp16 scales)
            if check and mbytes <= MAX_WBYTES:
                blk = bytearray(rnd.randbytes(mbytes))
                sanitise(kind, blk, bs)
                h2d(wchk, bytes(blk))
            for T in ts:
                rel = relb = float("nan")
                if check and mbytes <= MAX_WBYTES:
                    res_ = []
                    for with_br in (False, True):
                        b, r = (bias, res) if with_br else (None, None)
                        ck(H.hipMemset(y_tr, 0, C.c_size_t(maxrows * 4 * maxT)), "memset")
                        ck(H.hipMemset(y_gm, 0xFF, C.c_size_t(maxrows * 4 * maxT)), "memset")  # NaN fill: unwritten entries show
                        launch_tr(wchk.value, T, b, r)
                        launch_gm(wchk.value, T, b, r)
                        ck(H.hipDeviceSynchronize(), "sync")
                        ref = np.frombuffer(d2h(y_tr, ys * T), dtype=np.float32)
                        full = d2h(y_gm, maxrows * 4 * maxT)
                        got = np.frombuffer(full[:ys * T], dtype=np.float32)
                        mx = float(np.max(np.abs(ref))) or 1.0
                        bad = int(np.count_nonzero(~np.isfinite(got)))
                        if full[ys * T:].count(0xFF) != len(full) - ys * T:  # the 0xFF fill after the T outputs must be untouched
                            print("WRITE PAST THE OUTPUT: %s %dx%d T=%d" % (kind, rows, cols, T), flush=True)
                            bad += 1
                        d = float(np.max(np.abs(ref - got))) if not bad else 0.0
                        res_.append(float("inf") if bad else d / mx)
                    rel, relb = res_
                    worst = max(worst, rel, relb)
                tr_us = float("nan")
                gm_us = float("nan")
                if timing:
                    for name in (("gm",) if notr else ("tr", "gm")):
                        best = 1e30
                        for _rep in range(3):
                            for i in range(2):
                                (launch_tr if name == "tr" else launch_gm)(pool.value + (i % ncopy) * mbytes, T)
                            ck(H.hipEventRecord(e0, None), "rec")
                            for i in range(5):
                                (launch_tr if name == "tr" else launch_gm)(pool.value + (i % ncopy) * mbytes, T)
                            ck(H.hipEventRecord(e1, None), "rec"); ck(H.hipEventSynchronize(e1), "evs")
                            ms = C.c_float(); ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "el")
                            best = min(best, ms.value / 5 * 1e3)
                        if name == "tr":
                            tr_us = best
                        else:
                            gm_us = best
                tops = 2.0 * rows * cols * T / gm_us / 1e6 if timing else float("nan")
                print("%-7s %6dx%-6d %4d %10.1f %10.1f %7.2f %9.2f %9.2e %9.2e" % (
                    kind, rows, cols, T, tr_us, gm_us, tr_us / gm_us if timing and not notr else float("nan"), tops, rel, relb), flush=True)
    print("worst relative difference: %.3e" % worst)


main()
