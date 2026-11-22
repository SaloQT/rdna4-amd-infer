"""bw_bench.py -- attainable DRAM read bandwidth (tools/bw_bench_kernel.c).

  python tools/bw_bench.py build/bw_bench.hsaco [MiB]

Reads a buffer (default 2048 MiB, far past the 64 MB cache) with 256-thread
workgroups at several grid sizes; 3 warm-up and 10 timed launches each.
"""
import ctypes as C
import os
import sys

ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib",
                        "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")


def ck(e, what):
    if e:
        sys.exit("HIP error %d in %s" % (e, what))


hsaco = sys.argv[1]
mib = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
nbytes = mib << 20
ck(H.hipInit(0), "hipInit")
buf, out = C.c_void_p(), C.c_void_p()
ck(H.hipMalloc(C.byref(buf), C.c_size_t(nbytes)), "hipMalloc")
ck(H.hipMalloc(C.byref(out), C.c_size_t(64)), "hipMalloc")
ck(H.hipMemset(buf, 1, C.c_size_t(nbytes)), "hipMemset")
mod, fn = C.c_void_p(), C.c_void_p()
ck(H.hipModuleLoad(C.byref(mod), hsaco.encode()), "hipModuleLoad")
ck(H.hipModuleGetFunction(C.byref(fn), mod, b"bw_read"), "hipModuleGetFunction")
e0, e1 = C.c_void_p(), C.c_void_p()
ck(H.hipEventCreate(C.byref(e0)), "ev")
ck(H.hipEventCreate(C.byref(e1)), "ev")
n16 = C.c_uint64(nbytes // 16)
args = (C.c_void_p * 3)(C.cast(C.byref(buf), C.c_void_p), C.cast(C.byref(n16), C.c_void_p),
                        C.cast(C.byref(out), C.c_void_p))


def launch(groups):
    ck(H.hipModuleLaunchKernel(fn, groups, 1, 1, 256, 1, 1, 0, None, args, None), "launch")


print("buffer %d MiB" % mib)
for groups in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
    for _ in range(3):
        launch(groups)
    ck(H.hipDeviceSynchronize(), "sync")
    ck(H.hipEventRecord(e0, None), "rec")
    for _ in range(10):
        launch(groups)
    ck(H.hipEventRecord(e1, None), "rec")
    ck(H.hipEventSynchronize(e1), "evsync")
    ms = C.c_float()
    ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
    t = ms.value / 10
    print("groups %5d  %.3f ms  %.1f GB/s" % (groups, t, nbytes / t / 1e6))
