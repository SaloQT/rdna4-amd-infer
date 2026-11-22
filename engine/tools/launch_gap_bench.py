"""The GPU's cost per kernel launch on one stream: N launches of a kernel that returns at once (ie_gemv_raw_iq2xxs_tr8
with 0 rows: one workgroup fills its LDS tables and returns), timed with events, against N launches of the same
kernel with real work between them is not needed: the empty launches alone give the gap between kernels.
usage: python tools/launch_gap_bench.py [HSACO] [N] [WORKGROUPS]"""
import ctypes as C
import os
import sys

ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib",
                        "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")


def ck(e, what):
    if e:
        raise SystemExit("%s failed: %d" % (what, e))


hs = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "ie_kernels.hsaco")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
wgs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ck(H.hipInit(0), "hipInit")
m = C.c_void_p()
ck(H.hipModuleLoad(C.byref(m), hs.encode()), "load")
f = C.c_void_p()
ck(H.hipModuleGetFunction(C.byref(f), m, b"ie_gemv_raw_iq2xxs_tr8"), "get")
buf = C.c_void_p()
ck(H.hipMalloc(C.byref(buf), C.c_size_t(1 << 20)), "malloc")
vals = [buf, buf, buf, C.c_uint32(0), C.c_uint32(160), C.c_uint32(6), C.c_void_p(), C.c_void_p(), C.c_void_p(),
        C.c_void_p(), C.c_void_p(), C.c_float(0.0), C.c_void_p(), C.c_uint32(0), C.c_uint32(0), C.c_uint32(0)]
args = (C.c_void_p * len(vals))(*[C.cast(C.byref(v), C.c_void_p) for v in vals])
e0, e1 = C.c_void_p(), C.c_void_p()
ck(H.hipEventCreate(C.byref(e0)), "ev"), ck(H.hipEventCreate(C.byref(e1)), "ev")
for _ in range(200):
    ck(H.hipModuleLaunchKernel(f, wgs, 1, 1, 256, 1, 1, 0, None, args, None), "launch")
ck(H.hipDeviceSynchronize(), "sync")
best = 1e30
for _ in range(5):
    ck(H.hipEventRecord(e0, None), "rec")
    for _ in range(n):
        ck(H.hipModuleLaunchKernel(f, wgs, 1, 1, 256, 1, 1, 0, None, args, None), "launch")
    ck(H.hipEventRecord(e1, None), "rec")
    ck(H.hipEventSynchronize(e1), "sync")
    ms = C.c_float()
    ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
    best = min(best, ms.value)
print("%d launches of %d workgroups: %.2f us per launch" % (n, wgs, best * 1e3 / n))
