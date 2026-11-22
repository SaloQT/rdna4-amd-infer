"""The Windows video memory budget of this process on each adapter (DXGI QueryVideoMemoryInfo): what the OS lets one
process keep in VRAM (local) before new allocations go to system memory, and how much it could reserve.
usage: python tools/vram_budget.py"""
import ctypes as C
import ctypes.wintypes as W


class GUID(C.Structure):
    _fields_ = [("a", C.c_uint32), ("b", C.c_uint16), ("c", C.c_uint16), ("d", C.c_ubyte * 8)]


def guid(s):
    p = s.split("-")
    d = bytes.fromhex(p[3] + p[4])
    return GUID(int(p[0], 16), int(p[1], 16), int(p[2], 16), (C.c_ubyte * 8)(*d))


class INFO(C.Structure):
    _fields_ = [("budget", C.c_uint64), ("usage", C.c_uint64), ("avail_res", C.c_uint64), ("cur_res", C.c_uint64)]


class DESC1(C.Structure):
    _fields_ = [("desc", W.WCHAR * 128), ("vendor", C.c_uint), ("device", C.c_uint), ("subsys", C.c_uint),
                ("rev", C.c_uint), ("dedvid", C.c_size_t), ("dedsys", C.c_size_t), ("shared", C.c_size_t),
                ("luid", C.c_int64), ("flags", C.c_uint)]


def method(obj, index, restype, *argtypes):
    vt = C.cast(C.cast(obj, C.POINTER(C.c_void_p))[0], C.POINTER(C.c_void_p))
    return C.WINFUNCTYPE(restype, C.c_void_p, *argtypes)(vt[index])


dxgi = C.WinDLL("dxgi")
fac = C.c_void_p()
if dxgi.CreateDXGIFactory1(C.byref(guid("770aae78-f26f-4dba-a829-253c83d1b387")), C.byref(fac)):
    raise SystemExit("CreateDXGIFactory1 failed")
G = 1 << 30
for i in range(8):
    ad = C.c_void_p()
    if method(fac, 12, C.c_long, C.c_uint, C.POINTER(C.c_void_p))(fac, i, C.byref(ad)):  # EnumAdapters1
        break
    d = DESC1()
    method(ad, 10, C.c_long, C.POINTER(DESC1))(ad, C.byref(d))  # GetDesc1
    a3 = C.c_void_p()
    if method(ad, 0, C.c_long, C.POINTER(GUID), C.POINTER(C.c_void_p))(ad, C.byref(guid("645967a4-1392-4310-a798-8053ce3e93fd")), C.byref(a3)):
        print("%d %s: no IDXGIAdapter3" % (i, d.desc))
        continue
    out = []
    for grp, name in ((0, "local"), (1, "non-local")):
        m = INFO()
        method(a3, 14, C.c_long, C.c_uint, C.c_int, C.POINTER(INFO))(a3, 0, grp, C.byref(m))  # QueryVideoMemoryInfo
        out.append("%s budget %.2f GiB, in use %.2f, reservable %.2f" % (name, m.budget / G, m.usage / G, m.avail_res / G))
    print("%d %s (VRAM %.2f GiB): %s" % (i, d.desc, d.dedvid / G, "; ".join(out)))
