"""ie_gdn1p (the linear attention of a verify run) alone: random inputs at Saluki's shapes (16 key heads, 48 value
heads, 128 x 128 states), a verify with records (only token 0's state written, records of tokens 1 .. T - 1), from
one or more kernel files: every output (out, its Q8 copy, the state slots, the records, the ring) compared bitwise
with the first file's, and the time per launch (best of 5 x 200 launches, inputs restored before each check run).
usage: python tools/gdn_bench.py A.hsaco [B.hsaco ...] [-t T] [--nk 16] [--nv 48] [--pos 5000]"""
import argparse
import ctypes as C
import os
import random
import struct

ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib",
                        "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")


def ck(e, what):
    if e:
        raise SystemExit("%s failed: %d" % (what, e))


def dmalloc(n):
    p = C.c_void_p()
    ck(H.hipMalloc(C.byref(p), C.c_size_t(n)), "malloc")
    return p


def h2d(dst, data):
    ck(H.hipMemcpy(dst, data, C.c_size_t(len(data)), 1), "h2d")


def d2h(src, n):
    b = C.create_string_buffer(n)
    ck(H.hipMemcpy(b, src, C.c_size_t(n), 2), "d2h")
    return b.raw


ap = argparse.ArgumentParser()
ap.add_argument("hsaco", nargs="+")
ap.add_argument("-t", type=int, default=6)
ap.add_argument("--nk", type=int, default=16)
ap.add_argument("--nv", type=int, default=48)
ap.add_argument("--pos", type=int, default=5000)
a = ap.parse_args()
T, nk, nv, pos = a.t, a.nk, a.nv, a.pos
cd, inner = 2 * nk * 128 + nv * 128, nv * 128
isf = cd + inner + 2 * nv  # floats per input token: q k v | z | beta | alpha
osf = inner
nbq = inner // 32
qs = nbq * 40
ns, c = 4, 1
RECF = 260
rnd = random.Random(7)


def floats(n, f):
    return struct.pack("<%df" % n, *[f() for _ in range(n)])


inp = floats(T * isf, lambda: rnd.gauss(0.0, 1.0))
ring = floats(32 * cd, lambda: rnd.gauss(0.0, 1.0))
S = floats(ns * nv * 16384, lambda: rnd.gauss(0.0, 0.05))
cw = floats(cd * 4, lambda: rnd.gauss(0.0, 0.5))
dtb = floats(nv, lambda: rnd.gauss(0.0, 0.5))
sa = floats(nv, lambda: -rnd.uniform(0.2, 2.0))
nw = floats(128, lambda: rnd.uniform(0.5, 1.5))
sizes = {"in": len(inp), "out": T * osf * 4, "oq": T * qs, "ring": len(ring), "S": len(S), "cw": len(cw), "dtb": len(dtb),
         "sa": len(sa), "nw": len(nw), "rec": max(1, T - 1) * nv * RECF * 4}
buf = {k: dmalloc(v) for k, v in sizes.items()}


def reset():
    h2d(buf["in"], inp), h2d(buf["ring"], ring), h2d(buf["S"], S), h2d(buf["cw"], cw), h2d(buf["dtb"], dtb)
    h2d(buf["sa"], sa), h2d(buf["nw"], nw)
    for k in ("out", "oq", "rec"):
        h2d(buf[k], b"\0" * sizes[k])


vals = [buf["in"], buf["out"], buf["oq"], buf["ring"], buf["S"], buf["cw"], buf["dtb"], buf["sa"], buf["nw"],
        C.c_uint32(pos), C.c_uint32(cd), C.c_uint32(nk), C.c_uint32(nv), C.c_float(1e-6), C.c_uint32(T),
        C.c_uint32(isf), C.c_uint32(osf), C.c_uint32(qs), C.c_uint32(c), C.c_uint32(0), C.c_uint32(1), C.c_uint32(ns),
        buf["rec"], C.c_void_p()]
args = (C.c_void_p * len(vals))(*[C.cast(C.byref(v), C.c_void_p) for v in vals])
ck(H.hipInit(0), "hipInit")
e0, e1 = C.c_void_p(), C.c_void_p()
ck(H.hipEventCreate(C.byref(e0)), "ev"), ck(H.hipEventCreate(C.byref(e1)), "ev")
ref = None
for path in a.hsaco:
    m, f = C.c_void_p(), C.c_void_p()
    ck(H.hipModuleLoad(C.byref(m), path.encode()), "load " + path)
    ck(H.hipModuleGetFunction(C.byref(f), m, b"ie_gdn1p"), "get ie_gdn1p")
    reset()
    ck(H.hipModuleLaunchKernel(f, nv, 1, 1, 256, 1, 1, 0, None, args, None), "launch")
    ck(H.hipDeviceSynchronize(), "sync")
    outs = {k: d2h(buf[k], sizes[k]) for k in ("out", "oq", "S", "rec", "ring")}
    best = 1e30
    for _ in range(5):
        ck(H.hipEventRecord(e0, None), "rec")
        for _ in range(200):
            ck(H.hipModuleLaunchKernel(f, nv, 1, 1, 256, 1, 1, 0, None, args, None), "launch")
        ck(H.hipEventRecord(e1, None), "rec")
        ck(H.hipEventSynchronize(e1), "sync")
        ms = C.c_float()
        ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
        best = min(best, ms.value / 200)
    same = "reference" if ref is None else ", ".join("%s %s" % (k, "same" if outs[k] == ref[k] else "DIFFERENT") for k in outs)
    print("%-60s T %d: %7.2f us  %s" % (path, T, best * 1e3, same))
    if ref is None:
        ref = outs
