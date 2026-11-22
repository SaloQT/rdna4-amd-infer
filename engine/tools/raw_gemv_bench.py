"""raw_gemv_bench.py -- time the raw-block GEMV kernels (ie_gemv_raw_*) alone.

  python tools/raw_gemv_bench.py BASE.hsaco [CAND.hsaco] [-k iq2xxs,q2k] [-s 17408x5120,...] [-lds KiB] [-t T] [-tr4 | -tr8]

A hsaco argument FILE@R launches R-rows-per-wave kernels (rows / 8R groups).

For each type and shape (rows x cols, Saluki's decode shapes by default):
the weights are random blocks spread over a 512 MiB pool (each launch reads
another copy, so the bytes come from DRAM, as in a decode step); 100 warm-up
launches, then the best of 3 runs of 200 timed launches. With CAND, the same kernel of the
candidate hsaco is timed too. -lds pads every launch with dynamic LDS
(KiB per workgroup) to cap the workgroups per WGP, e.g. at the megakernel's.
The and its output on one sanitised copy (valid
fp16 scales) is compared bitwise with BASE's. -t T (1 to 8) times the
multi-token kernels (ie_gemv_raw_*_tr) with T tokens of different
activations instead; GB/s counts the weight bytes once. -tr4 (with -t 2..4 and FILE@2): CAND runs its
ie_gemv_raw_*_tr4 kernels (two rows per wave) against BASE's ie_gemv_raw_*_tr. -tr8 (with -t 5..8 and
FILE@R, R the rows per wave of that T): the same for the ie_gemv_raw_*_tr8 kernels. -tr8all: every file's
ie_gemv_raw_*_tr8 kernels, the first file's as the reference (variants of the _tr8 kernels compared bitwise).
-tr9 (with -t 5..9): the first file's _tr8 kernels against the other files' ie_gemv_raw_*_tr9 kernels (a kernel file
built with -DIE_TR9=1, given as FILE@4: 32 rows per workgroup). -xs0 (debug): every token reads token 0's
activations (their cache footprint shrinks; the outputs then differ).
"""
import ctypes as C
import os
import random
import struct
import sys

ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib",
                        "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")

# ggml type ids (src/common.h numbering) and bytes per 256-weight block
TYPES = {"q2k": (10, 84), "iq2xxs": (16, 66), "iq2xs": (17, 74), "iq3xxs": (18, 98), "iq1s": (19, 50),
         "iq3s": (21, 110), "iq2s": (22, 82), "iq1m": (29, 56)}
# fp16 scale offsets inside a block (sanitised for the bitwise check)
D_OFF = {"q2k": (80, 82), "iq2xxs": (0,), "iq2xs": (0,), "iq3xxs": (0,), "iq1s": (0,), "iq3s": (0,), "iq2s": (0,),
         "iq1m": ()}
SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (12288, 5120), (6144, 5120), (5120, 6144), (1024, 5120)]
POOL = 512 << 20
NWARM, NTIME, NREP = 100, 200, 5


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
        path, _, rpw = path.partition("@")  # FILE@R: R rows per wave (fewer workgroups)
        self.rpw = int(rpw or 1)
        self.suffix = ''  # appended to the kernel name (-tr4: the 2-4 token kernels ie_gemv_raw_*_tr4)
        self.m = C.c_void_p()
        ck(H.hipModuleLoad(C.byref(self.m), path.encode()), "hipModuleLoad " + path)
        self.f = {}

    def fn(self, name):
        name += self.suffix
        if name not in self.f:
            f = C.c_void_p()
            ck(H.hipModuleGetFunction(C.byref(f), self.m, name.encode()), "get " + name)
            self.f[name] = f
        return self.f[name]


def groups(rows, rpw=1):
    g = (rows + 8 * rpw - 1) // (8 * rpw)
    return 257 if 253 <= g <= 256 else g


def main():
    a = sys.argv[1:]
    kinds, shapes = list(TYPES), SHAPES
    if "-k" in a:
        i = a.index("-k"); kinds = a[i + 1].split(","); del a[i:i + 2]
    lds = 0
    if "-lds" in a:
        i = a.index("-lds"); lds = int(a[i + 1]) << 10; del a[i:i + 2]
    tr4 = "-tr4" in a  # the second and later files are timed with their ie_gemv_raw_*_tr4 kernels (with -t 2..4)
    if tr4:
        a.remove("-tr4")
    tr8 = "-tr8" in a  # ... or with their ie_gemv_raw_*_tr8 kernels (with -t 5..8)
    if tr8:
        a.remove("-tr8")
    tr8all = "-tr8all" in a  # every file (the first, the reference, too) with its _tr8 kernels: tr8 variants compared
    if tr8all:
        a.remove("-tr8all")
    tr9 = "-tr9" in a  # the first file's _tr8 kernels (the reference) against the other files' _tr9 kernels (FILE@4)
    if tr9:
        a.remove("-tr9")
        tr8all = True
    xs0 = "-xs0" in a  # debug: every token reads token 0's activations (stride 0; outputs then differ from real ones)
    if xs0:
        a.remove("-xs0")
    T = 0
    if "-t" in a:
        i = a.index("-t"); T = int(a[i + 1]); del a[i:i + 2]
    if "-s" in a:
        i = a.index("-s"); shapes = [tuple(int(v) for v in s.split("x")) for s in a[i + 1].split(",")]; del a[i:i + 2]
    ck(H.hipInit(0), "hipInit")
    mods = [Mod(p) for p in a]
    if tr4 or tr8 or tr8all:
        for m in (mods if tr8all else mods[1:]):
            m.suffix = "4" if tr4 else "8"
        if tr9:
            for m in mods[1:]:
                m.suffix = "9"
    rnd = random.Random(1)
    pool = dmalloc(POOL)
    for off in range(0, POOL, 64 << 20):
        h2d(pool, rnd.randbytes(64 << 20), off)
    e0, e1 = C.c_void_p(), C.c_void_p()
    ck(H.hipEventCreate(C.byref(e0)), "ev"), ck(H.hipEventCreate(C.byref(e1)), "ev")
    maxcols = max(c for _, c in shapes)
    nt = max(T, 1)
    xq = dmalloc(maxcols // 32 * 40 * nt)
    nbm = maxcols // 32
    x8s = [bytes(rnd.randrange(256) for _ in range(nbm * 32)) for _ in range(nt)]
    x8 = x8s[0]
    da = struct.pack("<%df" % nbm, *[0.01 + 0.001 * (i % 7) for i in range(nbm)])
    sums = struct.pack("<%di" % nbm, *[sum(struct.unpack("32b", x8[32 * i:32 * i + 32])) for i in range(nbm)])
    maxrows = max(r for r, _ in shapes)
    y = dmalloc(maxrows * 4 * nt)
    chk = dmalloc(64 << 20)
    print("%-7s %13s %9s" % ("type", "shape", "MB") + "".join("  %9s %7s" % ("us", "GB/s") for _ in mods) +
          ("  bitwise" if len(mods) > 1 else ""))
    for kind in kinds:
        tid, bs = TYPES[kind]
        name = "ie_gemv_raw_" + kind + ("_tr" if T else "")
        for rows, cols in shapes:
            nb = cols // 32
            # the Q8 activations of this width: int8 [nb*32], da [nb], sums [nb]
            for k in range(nt):
                xk = x8s[k][:nb * 32]
                sk = struct.pack("<%di" % nb, *[sum(struct.unpack("32b", xk[32 * i:32 * i + 32])) for i in range(nb)])
                h2d(xq, xk, k * nb * 40); h2d(xq, da[:nb * 4], k * nb * 40 + nb * 32); h2d(xq, sk, k * nb * 40 + nb * 36)
            mbytes = rows * (nb // 8) * bs
            ncopy = max(1, POOL // mbytes)
            vals = [C.c_void_p(), xq, y, C.c_uint32(rows), C.c_uint32(nb), C.c_uint32(T if T else tid), C.c_void_p(),
                    C.c_void_p(), C.c_void_p(), C.c_void_p(), C.c_void_p(), C.c_float(0.0), C.c_void_p(),
                    C.c_uint32(nb * 40 if T and not xs0 else 0), C.c_uint32(rows * 4 if T else 0), C.c_uint32(0)]
            args = (C.c_void_p * len(vals))(*[C.cast(C.byref(v), C.c_void_p) for v in vals])
            row = "%-7s %6dx%-6d %9.1f" % (kind, rows, cols, mbytes / 1e6)
            outs = []

            def launch(m, i, pad=lds):
                vals[0].value = pool.value + (i % ncopy) * mbytes
                ck(H.hipModuleLaunchKernel(m.fn(name), groups(rows, m.rpw), 1, 1, 256, 1, 1, pad, None, args, None),
                   "launch")
            for i in range(NWARM):
                launch(mods[0], i)
            # the candidates take turns within each repetition: clock drift
            # between repetitions hits every candidate alike
            best = [1e30] * len(mods)
            for _ in range(NREP):
                for mi, m in enumerate(mods):
                    for i in range(NWARM // 4):
                        launch(m, i)
                    ck(H.hipEventRecord(e0, None), "rec")
                    for i in range(NTIME):
                        launch(m, i)
                    ck(H.hipEventRecord(e1, None), "rec"); ck(H.hipEventSynchronize(e1), "evs")
                    ms = C.c_float(); ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "el")
                    best[mi] = min(best[mi], ms.value / NTIME * 1e3)
            for mi, m in enumerate(mods):
                f = m.fn(name)
                us = best[mi]
                row += "  %9.1f %7.1f" % (us, mbytes / us / 1e3)
                if len(mods) > 1:
                    # one sanitised copy: valid fp16 scales, so the output is finite
                    if mbytes <= (64 << 20):
                        blk = bytearray(rnd.randbytes(mbytes)) if not outs else blk
                        if not outs:
                            for o in D_OFF[kind]:
                                for b in range(o, mbytes, bs):
                                    blk[b], blk[b + 1] = 0x00, 0x2C  # 0.0625
                            if kind == "iq1m":  # its fp16 scale sits in the top nibble of the four u16 at 48..55: 0x2C00
                                for b in range(0, mbytes, bs):
                                    blk[b + 49] &= 0x0F
                                    blk[b + 51] &= 0x0F
                                    blk[b + 53] = (blk[b + 53] & 0x0F) | 0xC0
                                    blk[b + 55] = (blk[b + 55] & 0x0F) | 0x20
                            h2d(chk, bytes(blk))
                        vals[0].value = chk.value
                        ck(H.hipModuleLaunchKernel(f, groups(rows, m.rpw), 1, 1, 256, 1, 1, 0, None, args, None), "launch")
                        ck(H.hipDeviceSynchronize(), "sync")
                        outs.append(d2h(y, rows * 4 * nt))
            if len(outs) > 1:
                row += "  " + ("same" if all(o == outs[0] for o in outs[1:]) else "DIFFERENT")
            print(row, flush=True)


main()
