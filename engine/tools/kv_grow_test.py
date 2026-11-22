"""kv_grow_test.py -- does growing the KV arrays by copy (malloc new, copy, free old) return the old VRAM?
34 arrays (17 layers x K, V) of 1088 B per position grow from 8192 to --max positions in steps of 8192; after every
step the process's dedicated and shared GPU memory (performance counters) are printed next to the bytes in use.
Run through ref\\gpu-run.ps1:  pwsh -File ..\\gpu-run.ps1 -TimeoutSec 120 -- python tools\\kv_grow_test.py
--sync: hipDeviceSynchronize after every free. --mode copy|realloc (realloc: free first, no copy)."""
import argparse
import ctypes as C
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROCM_BIN = os.path.join(HERE, "..", "..", "..", ".rocm-venv", "Lib", "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")


def ck(e, what):
    if e:
        sys.exit("HIP error %d in %s" % (e, what))


def counters():
    pid = os.getpid()
    cmd = ["pwsh", "-NoProfile", "-Command",
           "$r=@(); foreach($c in 'Dedicated Usage','Shared Usage'){ $s=(Get-Counter \"\\GPU Process Memory(pid_%d_*)\\$c\").CounterSamples; "
           "$r += [math]::Round((($s|Measure-Object CookedValue -Sum).Sum)/1MB) }; $r -join ' '" % pid]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.split()
    return int(out[0]), int(out[1])


class Prop(C.Structure):  # hipMemAllocationProp
    _fields_ = [("type", C.c_int), ("handleType", C.c_int), ("locType", C.c_int), ("locId", C.c_int),
                ("win32", C.c_void_p), ("compression", C.c_ubyte), ("rdma", C.c_ubyte), ("usage", C.c_ushort)]


class Access(C.Structure):  # hipMemAccessDesc
    _fields_ = [("locType", C.c_int), ("locId", C.c_int), ("flags", C.c_int)]


def vmm(a):
    """Virtual memory: reserve the whole address range of each array, map physical chunks as the context grows (no copy)."""
    prop = Prop(type=1, handleType=0, locType=1, locId=0)
    gran = C.c_size_t()
    ck(H.hipMemGetAllocationGranularity(C.byref(gran), C.byref(prop), 0), "granularity")
    g = gran.value
    chunk = (a.chunk_mb << 20) // g * g
    print("granularity %d B, chunk %d B" % (g, chunk))
    total = (a.max * a.rowb + 4 + chunk - 1) // chunk * chunk
    base = []
    for i in range(a.n):
        p = C.c_void_p()
        ck(H.hipMemAddressReserve(C.byref(p), C.c_size_t(total), C.c_size_t(0), C.c_void_p(0), C.c_ulonglong(0)), "reserve")
        base.append(p)
    mapped = 0
    acc = Access(locType=1, locId=0, flags=3)
    handles = []
    cap = 0
    print("step cap   in-use-MiB  dedicated  shared")
    while cap < a.max:
        cap = min(a.max, cap + a.step)
        need = (cap * a.rowb + 4 + chunk - 1) // chunk * chunk
        while mapped < need:
            for i in range(a.n):
                h = C.c_void_p()
                ck(H.hipMemCreate(C.byref(h), C.c_size_t(chunk), C.byref(prop), C.c_ulonglong(0)), "create")
                dst = C.c_void_p(base[i].value + mapped)
                ck(H.hipMemMap(dst, C.c_size_t(chunk), C.c_size_t(0), h, C.c_ulonglong(0)), "map")
                ck(H.hipMemSetAccess(dst, C.c_size_t(chunk), C.byref(acc), C.c_size_t(1)), "setaccess")
                handles.append(h)
            mapped += chunk
        # touch the last byte of every array
        for i in range(a.n):
            ck(H.hipMemset(C.c_void_p(base[i].value + cap * a.rowb - 16), 0x5A, C.c_size_t(16)), "memset")
        ck(H.hipDeviceSynchronize(), "sync")
        d, s = counters()
        print("%5d %6d %9.0f %10d %7d" % (cap // a.step, cap, a.n * (cap * a.rowb) / 1048576.0, d, s), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=106496)
    ap.add_argument("--step", type=int, default=8192)
    ap.add_argument("--rowb", type=int, default=1088)
    ap.add_argument("--n", type=int, default=34)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--trim", action="store_true", help="hipMemPoolTrimTo(default pool, 0) after every step")
    ap.add_argument("--chunk-mb", type=int, default=8)
    ap.add_argument("--mode", default="copy", choices=["copy", "realloc", "vmm"])
    a = ap.parse_args()
    if a.mode == "vmm":
        return vmm(a)
    pool = C.c_void_p()
    if a.trim:
        ck(H.hipDeviceGetDefaultMemPool(C.byref(pool), 0), "default pool")
    ptrs = [None] * a.n
    cap = 0
    print("step cap   in-use-MiB  dedicated  shared")
    while cap < a.max:
        new = min(a.max, cap + a.step)
        for i in range(a.n):
            nw = C.c_void_p()
            if a.mode == "realloc" and ptrs[i]:
                ck(H.hipFree(ptrs[i]), "free")
                ptrs[i] = None
            ck(H.hipMalloc(C.byref(nw), C.c_size_t(new * a.rowb + 4)), "hipMalloc")
            if a.mode == "copy" and ptrs[i]:
                ck(H.hipMemcpy(nw, ptrs[i], C.c_size_t(cap * a.rowb), 3), "copy")
                ck(H.hipFree(ptrs[i]), "free")
            ptrs[i] = nw
            if a.sync:
                ck(H.hipDeviceSynchronize(), "sync")
        cap = new
        ck(H.hipDeviceSynchronize(), "sync")
        if a.trim:
            ck(H.hipMemPoolTrimTo(pool, C.c_size_t(0)), "trim")
        d, s = counters()
        print("%5d %6d %9.0f %10d %7d" % (cap // a.step, cap, a.n * (cap * a.rowb) / 1048576.0, d, s), flush=True)
    for p in ptrs:
        H.hipFree(p)


if __name__ == "__main__":
    main()
