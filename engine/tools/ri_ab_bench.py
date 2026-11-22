"""ri_ab_bench.py -- the per-256 prompt-chunk kernels ie_gemm_ri_<type> of two or more code objects, side by side.

  python tools/ri_ab_bench.py BASE.hsaco CAND.hsaco [CAND2.hsaco ...] [-k iq2xxs,q2k] [-s 17408x5120,...] [-t 256]
                              [-q] [-nocheck] [-reps N]

For each type, shape (rows x cols) and token count T:
  correctness: one sanitised weight copy (valid fp16 scales), random Q8 activations in the layout ie_requant_sb
    writes (int8 [32 nb], float da [nb] = the scale of the super-block repeated, int32 sums [nb]); every code object
    runs the same kernel; each candidate's output must be bit for bit equal to the base's, once without and once
    with bias and residual (and nothing past the T outputs may change).
  speed (unless -q): the weights spread over a 512 MiB pool (each launch reads another copy, so the bytes come from
    DRAM as in the model); the code objects alternate in every repetition; best of -reps (default 5) runs of 5
    launches; us per launch for each, each candidate's speed relative to the base, and the last one's int8 T MAC/s.
"""
import ctypes as C
import os
import random
import struct
import sys

import numpy as np

ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib", "site-packages",
                        "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")

TYPES = {"q2k": (10, 84), "iq2xxs": (16, 66), "iq2xs": (17, 74), "iq3xxs": (18, 98), "iq1s": (19, 50),
         "iq3s": (21, 110), "iq2s": (22, 82), "iq1m": (29, 56)}
D_OFF = {"q2k": (80, 82), "iq2xxs": (0,), "iq2xs": (0,), "iq3xxs": (0,), "iq1s": (0,), "iq3s": (0,), "iq2s": (0,), "iq1m": ()}
SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (6144, 5120), (5120, 6144), (1024, 5120)]
CHECK_SHAPES = [(1000, 2048), (1024, 5120)]
RT, TT = 128, 64  # rows and tokens per workgroup of ie_gemm_ri_*
POOL = 512 << 20
MAX_WBYTES = 128 << 20


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


def sanitise(kind, blk, bs):
    for o in D_OFF[kind]:
        for b in range(o, len(blk), bs):
            blk[b], blk[b + 1] = 0x00, 0x2C  # 0.0625
    if kind == "iq1m":  # d is split over the high nibbles of the four scale words: 0x2C00
        for b in range(0, len(blk), bs):
            blk[b + 49] &= 0x0F
            blk[b + 51] &= 0x0F
            blk[b + 53] = (blk[b + 53] & 0x0F) | 0xC0
            blk[b + 55] = (blk[b + 55] & 0x0F) | 0x20


def main():
    a = sys.argv[1:]
    kinds, shapes, ts = list(TYPES), SHAPES, [256]
    timing, check, reps = True, True, 5
    custom_shapes = False
    if "-k" in a:
        i = a.index("-k"); kinds = a[i + 1].split(","); del a[i:i + 2]
    if "-s" in a:
        i = a.index("-s"); shapes = [tuple(int(v) for v in s.split("x")) for s in a[i + 1].split(",")]; del a[i:i + 2]; custom_shapes = True
    if "-t" in a:
        i = a.index("-t"); ts = [int(v) for v in a[i + 1].split(",")]; del a[i:i + 2]
    if "-reps" in a:
        i = a.index("-reps"); reps = int(a[i + 1]); del a[i:i + 2]
    if "-q" in a:
        a.remove("-q"); timing = False
    if "-nocheck" in a:
        a.remove("-nocheck"); check = False
    if len(a) < 2:
        sys.exit(__doc__)
    ck(H.hipInit(0), "hipInit")
    mods = [Mod(p) for p in a]
    nm = len(mods)
    rnd = random.Random(7)
    maxT = max(ts)
    allshapes = sorted(set(shapes) | (set(CHECK_SHAPES) if check and not custom_shapes else set()))
    maxcols = max(c for _, c in allshapes)
    maxrows = max(r for r, _ in allshapes)
    pool = dmalloc(POOL)
    if timing:
        for off in range(0, POOL, 64 << 20):
            h2d(pool, rnd.randbytes(64 << 20), off)
    wchk = dmalloc(MAX_WBYTES)
    e0, e1 = C.c_void_p(), C.c_void_p()
    ck(H.hipEventCreate(C.byref(e0)), "ev"), ck(H.hipEventCreate(C.byref(e1)), "ev")
    nbm = maxcols // 32
    xq = dmalloc(nbm * 40 * maxT)
    ys_max = maxrows * 4
    outs = [dmalloc(ys_max * maxT) for _ in range(nm)]
    bias = dmalloc(maxrows * 4)
    res = dmalloc(maxrows * 4 * maxT)
    h2d(bias, struct.pack("<%df" % maxrows, *[0.25 * ((i % 13) - 6) for i in range(maxrows)]))
    h2d(res, struct.pack("<%df" % (maxrows * maxT), *[0.5 * ((i % 11) - 5) for i in range(maxrows * maxT)]))
    hdr = "%-7s %13s %4s" % ("type", "shape", "T") + "".join(" %9s" % ("us[%d]" % i) for i in range(nm)) + "".join(" %7s" % ("x[%d]" % i) for i in range(1, nm)) + " %9s  %s" % ("TMAC/s", "bitwise")
    print(hdr)
    nbad = 0
    for kind in kinds:
        tid, bs = TYPES[kind]
        name = "ie_gemm_ri_%s" % kind
        for rows, cols in allshapes:
            do_check = check and (rows, cols) in (CHECK_SHAPES if not custom_shapes else shapes)
            do_time = timing and (rows, cols) in shapes
            if not (do_check or do_time):
                continue
            nb = cols // 32
            xs = nb * 40
            ys = rows * 4
            mbytes = rows * (nb // 8) * bs
            nrng = np.random.default_rng(11)
            x8 = nrng.integers(-127, 128, size=(maxT, nb * 32), dtype=np.int8)
            dsb = (0.002 + 0.0007 * ((np.arange(maxT)[:, None] + np.arange(nb // 8)[None, :]) % 7)).astype(np.float32)
            da = np.repeat(dsb, 8, axis=1)
            sums = x8.reshape(maxT, nb, 32).astype(np.int32).sum(axis=2).astype(np.int32)
            h2d(xq, b"".join(x8[k].tobytes() + da[k].tobytes() + sums[k].tobytes() for k in range(maxT)))
            ncopy = max(1, POOL // mbytes)

            def launch(which, w, T, b=None, r=None):
                vals = [C.c_void_p(w), C.c_void_p(xq.value), C.c_void_p(outs[which].value), C.c_uint32(rows), C.c_uint32(nb),
                        C.c_uint32(T), C.c_void_p(b.value if b else 0), C.c_void_p(r.value if r else 0),
                        C.c_uint32(xs), C.c_uint32(ys), C.c_uint32(ys)]
                args = (C.c_void_p * len(vals))(*[C.cast(C.byref(v), C.c_void_p) for v in vals])
                nrt, ntt = (rows + RT - 1) // RT, (T + TT - 1) // TT
                ck(H.hipModuleLaunchKernel(mods[which].fn(name), nrt * ntt, 1, 1, 256, 1, 1, 0, None, args, None), "launch " + name)

            if do_check and mbytes <= MAX_WBYTES:
                blk = bytearray(rnd.randbytes(mbytes))
                sanitise(kind, blk, bs)
                h2d(wchk, bytes(blk))
            for T in ts:
                verdict = ""
                if do_check and mbytes <= MAX_WBYTES:
                    verdict = "same"
                    for T2 in sorted({T, min(T, 100)}):
                        for with_br in (False, True):
                            b, r = (bias, res) if with_br else (None, None)
                            got = []
                            for which in range(nm):
                                ck(H.hipMemset(outs[which], 0xFF, C.c_size_t(ys_max * maxT)), "memset")
                                launch(which, wchk.value, T2, b, r)
                                ck(H.hipDeviceSynchronize(), "sync")
                                got.append(d2h(outs[which], ys_max * maxT))
                            for which in range(1, nm):
                                if got[0] != got[which]:
                                    n = sum(1 for i in range(0, ys * T2, 4) if got[0][i:i + 4] != got[which][i:i + 4])
                                    tail = got[0][ys * T2:] != got[which][ys * T2:]
                                    verdict = "DIFFERENT [%d] (T=%d%s: %d of %d outputs%s)" % (which, T2, " +bias/res" if with_br else "", n, ys * T2 // 4, ", past the end too" if tail else "")
                                    nbad += 1
                            if got[0][:ys * T2].count(0xFF) > ys * T2 // 2:
                                verdict = "ALL UNWRITTEN? T=%d" % T2
                                nbad += 1
                us = [float("nan")] * nm
                if do_time:
                    best = [1e30] * nm
                    for rep in range(reps):
                        order = list(range(nm)) if rep % 2 == 0 else list(range(nm))[::-1]
                        for which in order:
                            for i in range(2):
                                launch(which, pool.value + (i % ncopy) * mbytes, T)
                            ck(H.hipEventRecord(e0, None), "rec")
                            for i in range(5):
                                launch(which, pool.value + (i % ncopy) * mbytes, T)
                            ck(H.hipEventRecord(e1, None), "rec"); ck(H.hipEventSynchronize(e1), "evs")
                            ms = C.c_float(); ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "el")
                            best[which] = min(best[which], ms.value / 5 * 1e3)
                    us = best
                tmacs = rows * cols * T / us[-1] / 1e6 if do_time else float("nan")
                print("%-7s %6dx%-6d %4d" % (kind, rows, cols, T) + "".join(" %9.1f" % u for u in us) +
                      "".join(" %7.3f" % (us[0] / u if do_time else float("nan")) for u in us[1:]) + " %9.2f  %s" % (tmacs, verdict), flush=True)
    print("bitwise differences: %d" % nbad)
    sys.exit(1 if nbad else 0)


main()
