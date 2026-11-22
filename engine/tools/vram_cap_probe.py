"""How much VRAM one process gets before Windows puts its new GPU allocations in system memory: allocates 256 MB pieces
(hipMalloc + hipMemset, or --vmm: hipMemCreate chunks as the engine's KV cache) up to --max GiB and stops at the first
piece that lands outside VRAM (DXGI QueryVideoMemoryInfo of this process: local and non-local usage). Frees all at exit.
usage: python tools/vram_cap_probe.py [--max 12] [--vmm]"""
import argparse
import ctypes as C
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROCM_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".rocm-venv", "Lib",
                        "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")


class GUID(C.Structure):
    _fields_ = [("a", C.c_uint32), ("b", C.c_uint16), ("c", C.c_uint16), ("d", C.c_ubyte * 8)]


def guid(s):
    p = s.split("-")
    return GUID(int(p[0], 16), int(p[1], 16), int(p[2], 16), (C.c_ubyte * 8)(*bytes.fromhex(p[3] + p[4])))


class INFO(C.Structure):
    _fields_ = [("budget", C.c_uint64), ("usage", C.c_uint64), ("avail_res", C.c_uint64), ("cur_res", C.c_uint64)]


def method(obj, index, restype, *argtypes):
    vt = C.cast(C.cast(obj, C.POINTER(C.c_void_p))[0], C.POINTER(C.c_void_p))
    return C.WINFUNCTYPE(restype, C.c_void_p, *argtypes)(vt[index])


class _Prop(C.Structure):
    _fields_ = [("type", C.c_int), ("handleType", C.c_int), ("locType", C.c_int), ("locId", C.c_int),
                ("win32", C.c_void_p), ("compression", C.c_ubyte), ("rdma", C.c_ubyte), ("usage", C.c_ushort)]


ap = argparse.ArgumentParser()
ap.add_argument("--max", type=float, default=12.0)
ap.add_argument("--vmm", action="store_true")
ap.add_argument("--piece", type=int, default=256, help="MiB per allocation (the engine's KV chunks: 4)")
ap.add_argument("--pin", type=float, default=0.0, help="GiB of pinned host memory (hipHostMalloc) first, as the engine's token embedding")
ap.add_argument("--small", type=int, default=0, help="number of 1 MiB device allocations made first")
ap.add_argument("--hostram", type=float, default=0.0, help="GiB of ordinary host memory written first (the engine holds 8-12 GiB while loading)")
ap.add_argument("--copy", action="store_true", help="fill each piece with hipMemcpy from pageable host memory (as the engine's load), not hipMemset")
ap.add_argument("--pin-piece", type=int, default=256, help="MiB per pinned allocation (the engine: one per embedding, 1288)")
ap.add_argument("--hold", type=int, default=0, help="then keep everything N seconds, printing local, non-local and budget every 2 s")
a = ap.parse_args()
PIECE = a.piece << 20
dxgi = C.WinDLL("dxgi")
fac = C.c_void_p()
dxgi.CreateDXGIFactory1(C.byref(guid("770aae78-f26f-4dba-a829-253c83d1b387")), C.byref(fac))
ad, a3 = C.c_void_p(), C.c_void_p()
method(fac, 12, C.c_long, C.c_uint, C.POINTER(C.c_void_p))(fac, 0, C.byref(ad))
method(ad, 0, C.c_long, C.POINTER(GUID), C.POINTER(C.c_void_p))(ad, C.byref(guid("645967a4-1392-4310-a798-8053ce3e93fd")), C.byref(a3))
query = method(a3, 14, C.c_long, C.c_uint, C.c_int, C.POINTER(INFO))
G, P = 1 << 30, 256 << 20


def usage():
    lo, nl = INFO(), INFO()
    query(a3, 0, 0, C.byref(lo)), query(a3, 0, 1, C.byref(nl))
    return lo.usage, nl.usage, lo.budget


if H.hipInit(0):
    raise SystemExit("hipInit failed")
ram = [bytearray(b"\1") * (256 << 20) for _ in range(int(a.hostram * 4))]
small = []
for _ in range(a.small):  # 1 MiB device allocations first (the engine's buffers)
    sp = C.c_void_p()
    H.hipMalloc(C.byref(sp), C.c_size_t(1 << 20))
    small.append(sp)
pins = []
PP = a.pin_piece << 20
for _ in range(int(a.pin * G) // PP):
    hp = C.c_void_p()
    if H.hipHostMalloc(C.byref(hp), C.c_size_t(PP), C.c_uint(0)):
        raise SystemExit("hipHostMalloc failed")
    C.memset(hp, 0, PP)
    pins.append(hp)
lo0, nl0, bud = usage()
print("start: local %.2f GiB, non-local %.2f GiB, local budget %.2f GiB%s" % (lo0 / G, nl0 / G, bud / G,
      ", PAL_ALWAYS_RESIDENT=" + os.environ["PAL_ALWAYS_RESIDENT"] if "PAL_ALWAYS_RESIDENT" in os.environ else ""))
held, va = [], C.c_void_p()
if a.vmm and H.hipMemAddressReserve(C.byref(va), C.c_size_t(int(a.max * G)), C.c_size_t(0), C.c_void_p(0), C.c_ulonglong(0)):
    raise SystemExit("reserve failed")
prop = _Prop(type=1, handleType=0, locType=1, locId=0)
P2 = P
P = PIECE
for i in range(int(a.max * G) // P):
    p = C.c_void_p()
    if a.vmm:
        h = C.c_void_p()
        e = H.hipMemCreate(C.byref(h), C.c_size_t(P), C.byref(prop), C.c_ulonglong(0))
        if e:
            print("hipMemCreate failed (%d) at %.2f GiB" % (e, i * P / G))
            break
        p = C.c_void_p(va.value + i * P)
        H.hipMemMap(p, C.c_size_t(P), C.c_size_t(0), h, C.c_ulonglong(0))

        class _Acc(C.Structure):
            _fields_ = [("locType", C.c_int), ("locId", C.c_int), ("flags", C.c_int)]
        H.hipMemSetAccess(p, C.c_size_t(P), C.byref(_Acc(1, 0, 3)), C.c_size_t(1))
    elif H.hipMalloc(C.byref(p), C.c_size_t(P)):
        print("hipMalloc failed at %.2f GiB" % (i * P / G))
        break
    if a.copy:  # as the engine's load: hipMemcpy from pageable host memory
        if 'src' not in globals():
            src = C.create_string_buffer(P)
        H.hipMemcpy(p, src, C.c_size_t(P), C.c_int(1))
    else:
        H.hipMemset(p, 0, C.c_size_t(P))
    H.hipDeviceSynchronize()
    held.append(p)
    lo, nl, bud = usage()
    if nl - nl0 >= max(P // 2, 64 << 20):
        print("piece %d: %.2f GiB allocated, local %.2f GiB (+%.2f), non-local +%.2f GiB: this piece went to system memory"
              % (i + 1, (i + 1) * P / G, lo / G, (lo - lo0) / G, (nl - nl0) / G))
        break
else:
    lo, nl, bud = usage()
    print("all %.2f GiB in VRAM: local %.2f GiB" % (len(held) * P / G, lo / G))
print("local budget now %.2f GiB" % (bud / G))
for s in range(0, a.hold, 2):
    import time
    time.sleep(2)
    lo, nl, bud = usage()
    print("  %3d s: local %.2f GiB, non-local %.2f GiB, local budget %.2f GiB" % (s + 2, lo / G, nl / G, bud / G))
