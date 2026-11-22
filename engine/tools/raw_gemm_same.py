"""raw_gemm_same.py -- are the prompt-chunk kernels (ie_gemm_ri_* and ie_gemm_rq_*) bitwise the same in two builds?

  python tools/raw_gemm_same.py BASE.hsaco NEW.hsaco [-k iq2s,iq3s] [-s 6144x5120,5120x6144] [-t 256]

Both code objects run the same random sanitised weights and random Q8 activations (the per-256 kernels get the
activations through ie_requant_sb first, as the engine does), once without and once with bias and residual.
Prints "same" or "DIFFERENT" per kernel, type, shape and token count. Written to check a change of the shared
sign decode of kernels/ie_iqraw.inc (IQ2_S, IQ3_S), which the prefill GEMMs use too.
"""
import array
import ctypes as C
import math
import os
import random
import struct
import sys

ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib",
                        "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")

TYPES = {"q2k": (10, 84), "iq2xxs": (16, 66), "iq2xs": (17, 74), "iq3xxs": (18, 98), "iq1s": (19, 50),
         "iq3s": (21, 110), "iq2s": (22, 82), "iq1m": (29, 56)}
D_OFF = {"q2k": (80, 82), "iq2xxs": (0,), "iq2xs": (0,), "iq3xxs": (0,), "iq1s": (0,), "iq3s": (0,), "iq2s": (0,), "iq1m": ()}
RQ_R, RQ_T, RI_T = 128, 128, 64  # rows and tokens per workgroup (kernels/ie_gemm_raw.inc, src/hip.c)


def ck(e, what):
    if e:
        sys.exit("HIP error %d in %s" % (e, what))


def dmalloc(n):
    p = C.c_void_p()
    ck(H.hipMalloc(C.byref(p), C.c_size_t(n)), "hipMalloc")
    return p


def h2d(dst, data):
    ck(H.hipMemcpy(dst, data, C.c_size_t(len(data)), 1), "h2d")


def d2h(src, n):
    b = C.create_string_buffer(n)
    ck(H.hipMemcpy(b, src, C.c_size_t(n), 2), "d2h")
    return b.raw


class Mod:
    def __init__(self, path):
        self.m = C.c_void_p()
        ck(H.hipModuleLoad(C.byref(self.m), path.encode()), "hipModuleLoad " + path)

    def fn(self, name):
        f = C.c_void_p()
        ck(H.hipModuleGetFunction(C.byref(f), self.m, name.encode()), "get " + name)
        return f


def main():
    a = sys.argv[1:]
    kinds, shapes, ts = ["iq2s", "iq3s"], [(6144, 5120), (5120, 6144)], [256]
    if "-k" in a:
        i = a.index("-k"); kinds = a[i + 1].split(","); del a[i:i + 2]
    if "-s" in a:
        i = a.index("-s"); shapes = [tuple(int(v) for v in s.split("x")) for s in a[i + 1].split(",")]; del a[i:i + 2]
    if "-t" in a:
        i = a.index("-t"); ts = [int(v) for v in a[i + 1].split(",")]; del a[i:i + 2]
    if len(a) != 2:
        sys.exit(__doc__)
    ck(H.hipInit(0), "hipInit")
    mods = [Mod(p) for p in a]
    rnd = random.Random(5)
    maxT, maxrows, maxcols = max(ts), max(r for r, _ in shapes), max(c for _, c in shapes)
    xq = dmalloc(maxcols // 32 * 40 * maxT)
    qq = dmalloc(maxcols // 32 * 40 * maxT)
    out = dmalloc(maxrows * 4 * maxT)
    wbuf = dmalloc(maxrows * (maxcols // 256) * 110)
    bias = dmalloc(maxrows * 4)
    res = dmalloc(maxrows * 4 * maxT)
    h2d(bias, struct.pack("<%df" % maxrows, *[0.25 * ((i % 13) - 6) for i in range(maxrows)]))
    h2d(res, struct.pack("<%df" % (maxrows * maxT), *[0.5 * ((i % 11) - 5) for i in range(maxrows * maxT)]))
    allsame = True
    for kind in kinds:
        tid, bs = TYPES[kind]
        for rows, cols in shapes:
            nb = cols // 32
            xs, ys = nb * 40, rows * 4
            fix = bytes(range(128)) + b"\x81" + bytes(range(129, 256))  # int8 -128 -> -127: activations in [-127, 127]
            parts = []
            for k in range(maxT):
                x8 = rnd.randbytes(nb * 32).translate(fix)
                da = struct.pack("<%df" % nb, *[0.002 + 0.0007 * ((k + i) % 7) for i in range(nb)])
                sums = struct.pack("<%di" % nb, *[sum(struct.unpack("32b", x8[32 * i:32 * i + 32])) for i in range(nb)])
                parts.append(x8 + da + sums)
            h2d(xq, b"".join(parts))
            blk = bytearray(rnd.randbytes(rows * (nb // 8) * bs))
            for o in D_OFF[kind]:
                for b in range(o, len(blk), bs):
                    blk[b], blk[b + 1] = 0x00, 0x2C
            if kind == "iq1m":  # IQ1_M keeps its fp16 super-scale in the top nibbles of bytes 49-55: a finite one
                for b in range(0, len(blk), bs):
                    blk[b + 49] &= 0x0F
                    blk[b + 51] &= 0x0F
                    blk[b + 53] = (blk[b + 53] & 0x0F) | 0xC0
                    blk[b + 55] = (blk[b + 55] & 0x0F) | 0x20
            h2d(wbuf, bytes(blk))
            for T in ts:
                for which in ("ri", "rq"):
                    outs = []
                    for m in mods:
                        for with_br in (False, True):
                            ck(H.hipMemset(out, 0xFF, C.c_size_t(maxrows * 4 * maxT)), "memset")
                            src = xq
                            if which == "ri":  # the engine's default prefill: requantize to one scale per 256 first
                                v = [C.c_void_p(xq.value), C.c_void_p(qq.value), C.c_uint32(nb), C.c_uint32(xs), C.c_uint32(xs)]
                                ra = (C.c_void_p * len(v))(*[C.cast(C.byref(x), C.c_void_p) for x in v])
                                ck(H.hipModuleLaunchKernel(m.fn("ie_requant_sb"), (nb // 8 + 7) // 8, T, 1, 256, 1, 1, 0, None, ra, None), "requant")
                                src = qq
                            v = [C.c_void_p(wbuf.value), C.c_void_p(src.value), C.c_void_p(out.value), C.c_uint32(rows),
                                 C.c_uint32(nb), C.c_uint32(T), C.c_void_p(bias.value if with_br else 0),
                                 C.c_void_p(res.value if with_br else 0), C.c_uint32(xs), C.c_uint32(ys), C.c_uint32(ys)]
                            args = (C.c_void_p * len(v))(*[C.cast(C.byref(x), C.c_void_p) for x in v])
                            nrt = (rows + RQ_R - 1) // RQ_R
                            ntt = (T + (RI_T if which == "ri" else RQ_T) - 1) // (RI_T if which == "ri" else RQ_T)
                            ck(H.hipModuleLaunchKernel(m.fn("ie_gemm_%s_%s" % (which, kind)), nrt * ntt, 1, 1, 256, 1, 1, 0, None, args, None), "launch")
                            ck(H.hipDeviceSynchronize(), "sync")
                            outs.append(d2h(out, ys * T))
                    same = outs[0] == outs[2] and outs[1] == outs[3]
                    f32 = array.array("f")
                    f32.frombytes(outs[0])
                    finite = all(math.isfinite(v) for v in f32)
                    allsame &= same
                    print("%-4s %-6s %6dx%-6d T=%-4d %s%s" % ("gemm_" + which, kind, rows, cols, T, "same" if same else "DIFFERENT",
                                                          "" if finite else " (non-finite outputs)"), flush=True)
    print("all same" if allsame else "SOME DIFFERENT")


main()
