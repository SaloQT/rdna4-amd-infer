"""swiglu_bench.py -- time ie_swiglu alone (with the Q8 output) at Saluki's FFN width, T tokens per launch.
usage: python tools/swiglu_bench.py HSACO [HSACO2 ...] [-n 17408] [-t 6]"""
import ctypes as C
import os
import struct
import sys

ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib",
                        "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")


def ck(e, what):
    if e:
        sys.exit("HIP error %d in %s" % (e, what))


a = sys.argv[1:]
n, T = 17408, 6
if "-n" in a:
    i = a.index("-n"); n = int(a[i + 1]); del a[i:i + 2]
if "-t" in a:
    i = a.index("-t"); T = int(a[i + 1]); del a[i:i + 2]
ck(H.hipInit(0), "init")


def dmalloc(sz):
    p = C.c_void_p()
    ck(H.hipMalloc(C.byref(p), C.c_size_t(sz)), "malloc")
    return p


st = n * 4
A, B, Q = dmalloc(st * T), dmalloc(st * T), dmalloc(n // 32 * 40 * T)
vals = struct.pack("<%df" % (n * T), *[((i * 37) % 200 - 100) / 50.0 for i in range(n * T)])
ck(H.hipMemcpy(B, vals, C.c_size_t(len(vals)), 1), "h2d")
e0, e1 = C.c_void_p(), C.c_void_p()
ck(H.hipEventCreate(C.byref(e0)), "ev"), ck(H.hipEventCreate(C.byref(e1)), "ev")
print("%-60s %9s" % ("hsaco", "us/launch"))
for path in a:
    m = C.c_void_p()
    ck(H.hipModuleLoad(C.byref(m), path.encode()), "load")
    f = C.c_void_p()
    ck(H.hipModuleGetFunction(C.byref(f), m, b"ie_swiglu"), "get")
    vs = [A, B, C.c_uint32(n), Q, C.c_uint32(st), C.c_uint32(n // 32 * 40)]
    args = (C.c_void_p * len(vs))(*[C.cast(C.byref(v), C.c_void_p) for v in vs])
    best = 1e30
    for rep in range(5):
        ck(H.hipMemcpy(A, vals, C.c_size_t(len(vals)), 1), "h2d")
        for _ in range(20):
            ck(H.hipModuleLaunchKernel(f, (n + 255) // 256, T, 1, 256, 1, 1, 0, None, args, None), "launch")
        ck(H.hipEventRecord(e0, None), "rec")
        for _ in range(200):
            ck(H.hipModuleLaunchKernel(f, (n + 255) // 256, T, 1, 256, 1, 1, 0, None, args, None), "launch")
        ck(H.hipEventRecord(e1, None), "rec"), ck(H.hipEventSynchronize(e1), "sync")
        ms = C.c_float()
        ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "el")
        best = min(best, ms.value / 200 * 1e3)
    print("%-60s %9.2f" % (path, best))
