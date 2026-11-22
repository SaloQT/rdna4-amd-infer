"""attn_long_bench.py -- standalone check and timing of the decode attention kernels (ie_attn_split,
ie_attn_kv, ie_attn_long) on random KV caches. Always run through ref\\gpu-run.ps1:

  pwsh -NoProfile -File ..\\gpu-run.ps1 -TimeoutSec 120 -- <python.exe> tools\\attn_long_bench.py [options]

  --hsaco FILE   kernel object (default build/ie_kernels.hsaco; the q8_0 KV build: build/ie_kernels_q8.hsaco, --q8)
  --pos N        position of the first token (keys 0 .. N + T - 1 exist), default 3000
  --T N          tokens in the chunk (positions N .. N + T - 1), default 1
  --check        compare ie_attn_long with ie_attn_split (per token) and a float64 reference, and the T = 1 runs
  --time         time the kernels (a graph of launches over several layers of KV, so the KV comes from DRAM)

Every buffer is allocated with padding; the byte range each kernel can touch is computed here and checked
against the allocation before the launch (a wrong stride would otherwise fault the GPU).
"""
import argparse
import ctypes as C
import gc
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROCM_BIN = os.path.join(HERE, "..", "..", "..", ".rocm-venv", "Lib", "site-packages", "_rocm_sdk_libraries", "bin")
os.add_dll_directory(os.path.abspath(ROCM_BIN))
H = C.CDLL("amdhip64_7.dll")

PAD = 1 << 20  # bytes of padding after every buffer (filled with a NaN pattern)


class _Prop(C.Structure):  # hipMemAllocationProp
    _fields_ = [("type", C.c_int), ("handleType", C.c_int), ("locType", C.c_int), ("locId", C.c_int),
                ("win32", C.c_void_p), ("compression", C.c_ubyte), ("rdma", C.c_ubyte), ("usage", C.c_ushort)]


class _Access(C.Structure):  # hipMemAccessDesc
    _fields_ = [("locType", C.c_int), ("locId", C.c_int), ("flags", C.c_int)]


def vmm_alloc(ptr, nbytes, chunk=8 << 20):
    """Reserve nbytes of addresses, back them with physical chunks of 8 MiB, give the device read/write access."""
    prop = _Prop(type=1, handleType=0, locType=1, locId=0)
    total = (nbytes + chunk - 1) // chunk * chunk
    if H.hipMemAddressReserve(C.byref(ptr), C.c_size_t(total), C.c_size_t(0), C.c_void_p(0), C.c_ulonglong(0)):
        sys.exit("hipMemAddressReserve failed")
    acc = _Access(locType=1, locId=0, flags=3)
    for off in range(0, total, chunk):
        h = C.c_void_p()
        dst = C.c_void_p(ptr.value + off)
        ck(H.hipMemCreate(C.byref(h), C.c_size_t(chunk), C.byref(prop), C.c_ulonglong(0)), "hipMemCreate")
        ck(H.hipMemMap(dst, C.c_size_t(chunk), C.c_size_t(0), h, C.c_ulonglong(0)), "hipMemMap")
        ck(H.hipMemSetAccess(dst, C.c_size_t(chunk), C.byref(acc), C.c_size_t(1)), "hipMemSetAccess")


def ck(e, what):
    if e:
        sys.exit("HIP error %d in %s" % (e, what))


class Buf:
    """A device allocation of size bytes (+ padding)."""

    def __init__(self, nbytes, fill=0xFF, vmm=False):
        self.size = int(nbytes)
        self.ptr = C.c_void_p()
        if vmm:  # virtual memory: reserved addresses with 8 MiB physical chunks mapped (what the engine's growing KV cache does)
            vmm_alloc(self.ptr, self.size + PAD)
        else:
            ck(H.hipMalloc(C.byref(self.ptr), C.c_size_t(self.size + PAD)), "hipMalloc")
        ck(H.hipMemset(self.ptr, fill, C.c_size_t(self.size + PAD)), "hipMemset")
        self.addr = self.ptr.value
        self.vmm = vmm

    def __del__(self):
        if getattr(self, "ptr", None) is not None and self.ptr.value and not getattr(self, "vmm", False):
            H.hipFree(self.ptr)
            self.ptr = None

    def put(self, a):
        a = np.ascontiguousarray(a)
        if a.nbytes > self.size:
            sys.exit("Buf.put: %d bytes into %d" % (a.nbytes, self.size))
        ck(H.hipMemcpy(self.ptr, a.ctypes.data_as(C.c_void_p), C.c_size_t(a.nbytes), 1), "hipMemcpy H2D")

    def get(self, dtype, count, offset=0):
        out = np.empty(count, dtype=dtype)
        if offset + out.nbytes > self.size:
            sys.exit("Buf.get: %d bytes at %d from %d" % (out.nbytes, offset, self.size))
        ck(H.hipMemcpy(out.ctypes.data_as(C.c_void_p), C.c_void_p(self.addr + offset), C.c_size_t(out.nbytes), 2),
           "hipMemcpy D2H")
        return out

    def zero(self):
        ck(H.hipMemset(self.ptr, 0, C.c_size_t(self.size)), "hipMemset")


class EngBuf(Buf):
    """One array of the engine's growing KV cache (src/hip.c kv_ensure): addresses for ctx rows reserved, physical
    chunks mapped as it grows."""

    def __init__(self, va_bytes):  # pylint: disable=super-init-not-called
        self.ptr = C.c_void_p()
        if H.hipMemAddressReserve(C.byref(self.ptr), C.c_size_t(va_bytes), C.c_size_t(0), C.c_void_p(0), C.c_ulonglong(0)):
            sys.exit("hipMemAddressReserve failed")
        self.addr, self.size, self.va, self.vmm = self.ptr.value, 0, va_bytes, True

    def grow(self, nbytes, chunk):
        prop, acc = _Prop(type=1, handleType=0, locType=1, locId=0), _Access(locType=1, locId=0, flags=3)
        nbytes = min(self.va, (nbytes + chunk - 1) // chunk * chunk)
        old = self.size
        for off in range(old, nbytes, chunk):
            h = C.c_void_p()
            ck(H.hipMemCreate(C.byref(h), C.c_size_t(chunk), C.byref(prop), C.c_ulonglong(0)), "hipMemCreate")
            ck(H.hipMemMap(C.c_void_p(self.addr + off), C.c_size_t(chunk), C.c_size_t(0), h, C.c_ulonglong(0)), "hipMemMap")
        if nbytes > old:
            ck(H.hipMemSetAccess(C.c_void_p(self.addr + old), C.c_size_t(nbytes - old), C.byref(acc), C.c_size_t(1)), "access")
            self.size = nbytes


def eng_kv(layers, rows, rowb, ctx, chunk_mb, step=4096, slack=64):
    """K and V arrays of `layers` layers as the engine maps them: every array reserved for ctx + slack rows, then
    step positions at a time, all arrays in turn, until rows are covered."""
    chunk = chunk_mb << 20
    va = ((ctx + slack) * rowb + 4 + chunk - 1) // chunk * chunk
    arrs = [EngBuf(va) for _ in range(2 * layers)]
    for want in range(step, rows + step, step):
        for b in arrs:
            b.grow((min(want, ctx) + slack) * rowb + 4, chunk)
    return arrs[0::2], arrs[1::2]


def need(buf, end, what):
    """The kernel touches bytes [0, end) of buf (relative to its pointer): must lie inside the allocation."""
    if end > buf.size:
        sys.exit("bounds: %s would touch %d bytes of a %d byte buffer" % (what, end, buf.size))


class Module:
    def __init__(self, path):
        self.mod = C.c_void_p()
        ck(H.hipModuleLoad(C.byref(self.mod), os.path.abspath(path).encode()), "hipModuleLoad")
        self.fn = {}

    def get(self, name):
        if name not in self.fn:
            f = C.c_void_p()
            ck(H.hipModuleGetFunction(C.byref(f), self.mod, name.encode()), "hipModuleGetFunction " + name)
            self.fn[name] = f
        return self.fn[name]


class Args:
    def __init__(self, *vals):
        self.keep = []
        for v in vals:
            if isinstance(v, Buf):
                self.keep.append(C.c_void_p(v.addr))
            elif isinstance(v, tuple):  # (Buf, byte offset)
                self.keep.append(C.c_void_p(v[0].addr + v[1]))
            elif v is None:
                self.keep.append(C.c_void_p(0))
            else:
                self.keep.append(C.c_uint32(int(v)))
        self.arr = (C.c_void_p * len(self.keep))(*[C.cast(C.byref(k), C.c_void_p) for k in self.keep])


def launch(mod, name, gx, gy, args, stream=None):
    ck(H.hipModuleLaunchKernel(mod.get(name), gx, gy, 1, 256, 1, 1, 0, stream, args.arr, None), "launch " + name)


def sync():
    ck(H.hipDeviceSynchronize(), "sync")


# --------------------------------------------------------------------------------------------------- data

def quant_q8(x):
    """x: [n, hd] float32 -> records [n, hd + hd/16] uint8 (int8 values, then fp16 scales), dequantized float32."""
    n, hd = x.shape
    xb = x.reshape(n, hd // 32, 32)
    amax = np.abs(xb).max(axis=2)
    sh = (amax / 127.0).astype(np.float16)
    s = sh.astype(np.float32)
    s_safe = np.where(s == 0, 1.0, s)
    qi = np.clip(np.rint(xb / s_safe[:, :, None]), -127, 127).astype(np.int8)
    qi = np.where(s[:, :, None] == 0, 0, qi).astype(np.int8)
    rec = np.empty((n, hd + hd // 16), dtype=np.uint8)
    rec[:, :hd] = qi.reshape(n, hd).view(np.uint8)
    rec[:, hd:] = sh.view(np.uint8).reshape(n, hd // 16)
    deq = (qi.astype(np.float32) * s[:, :, None]).reshape(n, hd)
    return rec, deq


def quant_q4(x):
    """x: [n, hd] float32 -> records [n, hd/2 + hd/8] uint8 (value i in byte i/2, even i low nibble; then per block of 32
    an fp16 d and an fp16 m), dequantized float32 = fmaf(q, d, m)."""
    n, hd = x.shape
    xb = x.reshape(n, hd // 32, 32)
    mx = xb.max(axis=2)
    mn = xb.min(axis=2)
    dh = ((mx - mn) / np.float32(15.0)).astype(np.float16)
    mh = mn.astype(np.float16)
    d = dh.astype(np.float32)
    m = mh.astype(np.float32)
    d_safe = np.where(d == 0, 1.0, d)
    qi = np.clip(np.rint((xb - m[:, :, None]) / d_safe[:, :, None]), 0, 15)
    qi = np.where(d[:, :, None] == 0, 0, qi).astype(np.uint8)
    q = qi.reshape(n, hd)
    rec = np.empty((n, hd // 2 + hd // 8), dtype=np.uint8)
    rec[:, : hd // 2] = q[:, 0::2] | (q[:, 1::2] << 4)
    dm = np.empty((n, hd // 32, 2), dtype=np.float16)
    dm[:, :, 0] = dh
    dm[:, :, 1] = mh
    rec[:, hd // 2 :] = dm.view(np.uint8).reshape(n, hd // 8)
    deq = (qi.astype(np.float32) * d[:, :, None] + m[:, :, None]).reshape(n, hd)  # exact product, one rounding: fmaf
    return rec, deq


KSCALE = 1.0  # --kscale: K multiplied by this (wide score ranges: most softmax weights underflow)


def make_kv(rng, nrows, n_kv, hd, q8, dist):
    """K, V as device-format arrays and float64 copies of what the kernels read: [nrows, n_kv, hd]."""
    out = []
    for _ in range(2):
        x = rng.standard_normal((nrows * n_kv, hd)).astype(np.float32)
        if dist == "peaked":
            x *= np.exp(rng.standard_normal((nrows * n_kv, 1)).astype(np.float32) * 0.7)
            x[:, :4] *= 8.0
        if not out:
            x *= KSCALE
        if q8 == "q4":
            rec, deq = quant_q4(x)
            out.append((rec, deq.astype(np.float64).reshape(nrows, n_kv, hd)))
        elif q8:
            rec, deq = quant_q8(x)
            out.append((rec, deq.astype(np.float64).reshape(nrows, n_kv, hd)))
        else:
            h = x.astype(np.float16)
            out.append((h, h.astype(np.float64).reshape(nrows, n_kv, hd)))
    return out


def reference(q, K, V, pos, T, n_head, n_kv, hd, gate, gstride):
    """float64 attention of tokens pos .. pos + T - 1: [T, n_head, hd]."""
    grp = n_head // n_kv
    out = np.zeros((T, n_head, hd))
    for j in range(T):
        pj = pos + j
        for h in range(n_head):
            kh = h // grp
            s = K[: pj + 1, kh, :] @ q[j, h].astype(np.float64) / np.sqrt(hd)
            e = np.exp(s - s.max())
            o = (e @ V[: pj + 1, kh, :]) / e.sum()
            if gate is not None:
                o = o / (1.0 + np.exp(-gate[j, h * gstride : h * gstride + hd].astype(np.float64)))
            out[j, h] = o
    return out


def relerr(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


# --------------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", default=os.path.join(HERE, "..", "build", "ie_kernels.hsaco"))
    ap.add_argument("--q8", action="store_true")
    ap.add_argument("--q4", action="store_true", help="the 4-bit KV cache (build/ie_kernels_q4.hsaco)")
    ap.add_argument("--vmm", action="store_true", help="K and V in virtual memory mapped in 8 MiB chunks (as the engine's growing KV cache)")
    ap.add_argument("--pos", default="3000", help="comma list of positions")
    ap.add_argument("--T", default="1", help="comma list of chunk sizes")
    ap.add_argument("--nkv", type=int, default=4)
    ap.add_argument("--grp", type=int, default=6)
    ap.add_argument("--hd", type=int, default=256)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--gate", action="store_true", help="fuse the output gate")
    ap.add_argument("--oq", action="store_true", help="fuse the Q8 copy of the output")
    ap.add_argument("--dist", default="normal", choices=["normal", "peaked"])
    ap.add_argument("--kernels", default="split,kv,long", help="which kernels to time/check")
    ap.add_argument("--layers", type=int, default=0, help="KV layers to cycle through when timing (0: auto)")
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--g", type=int, default=8, help="splits per merge group (AL_G of the kernel build)")
    ap.add_argument("--pgloop", action="store_true", help="ie_attn_long: every pair group in one workgroup per split")
    ap.add_argument("--early", action="store_true", help="time ie_attn_long_e (the V loads first) instead of ie_attn_long")
    ap.add_argument("--engkv", type=int, default=0, help="time with K/V arrays mapped as the engine's growing cache "
                    "for a context of this many positions (4096-position steps over all arrays)")
    ap.add_argument("--chunk", type=int, default=4, help="--engkv: MiB per physical chunk (IE_KV_CHUNK_MB)")
    ap.add_argument("--kscale", type=float, default=1.0, help="K times this (wide score ranges)")
    ap.add_argument("--hostpin", type=float, default=0.0, help="GB of pinned host memory (hipHostMalloc, 256 MB pieces) first")
    ap.add_argument("--hog", type=float, default=0.0, help="GB of other device memory allocated (256 MB pieces, set) before the KV")
    a = ap.parse_args()
    global KSCALE
    KSCALE = a.kscale
    if a.hd != 256:
        sys.exit("hd 256 only")
    ck(H.hipInit(0), "hipInit")
    mod = Module(a.hsaco)
    pins = []
    for _ in range(int(a.hostpin * 1e9) // (256 << 20)):
        hp = C.c_void_p()
        ck(H.hipHostMalloc(C.byref(hp), C.c_size_t(256 << 20), C.c_uint(0)), "hipHostMalloc")
        C.memset(hp, 0, 256 << 20)
        pins.append(hp)
    if pins:
        print("hostpin: %.2f GB of pinned host memory" % (len(pins) * (256 << 20) / 1e9))
    hog = [Buf(256 << 20, fill=0) for _ in range(int(a.hog * 1e9) // (256 << 20))]  # noqa: F841 (kept alive)
    if hog:
        print("hog: %.2f GB allocated before the KV" % (len(hog) * (256 << 20) / 1e9))
    for nm in ("ie_attn_long", "ie_attn_kv"):
        nb = C.c_int()
        if H.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(C.byref(nb), mod.get(nm), 256, C.c_size_t(0)) == 0:
            print("occupancy %s: %d workgroups per multiprocessor (WGP)" % (nm, nb.value))
    for pos in [int(x) for x in a.pos.split(",")]:
        for T in [int(x) for x in a.T.split(",")]:
            run_case(a, mod, pos, T)
            gc.collect()
    print("done")


def run_case(a, mod, pos, T):
    n_kv, grp, hd = a.nkv, a.grp, a.hd
    n_head = n_kv * grp
    kend = pos + T
    if T < 1 or T > 16 or grp > 8:
        sys.exit("T 1..16, grp <= 8")
    rng = np.random.default_rng(a.seed)
    kernels = a.kernels.split(",")
    if a.q4:
        a.q8 = "q4"
    krow = (hd // 2 + hd // 8) if a.q4 else (hd + hd // 16) if a.q8 else hd * 2  # bytes of a (position, head) record
    kvbytes = lambda rows: rows * n_kv * krow
    # ---- one layer of KV for the checks
    (Kd, Kf), (Vd, Vf) = make_kv(rng, kend, n_kv, hd, a.q8, a.dist)
    kc, vc = Buf(kvbytes(kend), vmm=a.vmm), Buf(kvbytes(kend), vmm=a.vmm)
    kc.put(Kd)
    vc.put(Vd)
    # ---- q, gate, outputs (strides in bytes, 256-aligned as the graph's token strides)
    qst = (n_head * hd * 4 + 255) // 256 * 256
    ost = qst
    gst = (n_head * 2 * hd * 4 + 255) // 256 * 256
    nbq = n_head * hd // 32
    oqst = (40 * nbq + 255) // 256 * 256
    gstride = 2 * hd
    q = (rng.standard_normal((T, n_head, hd)) * (3.0 if a.dist == "peaked" else 1.0)).astype(np.float32)
    qb = Buf(T * qst)
    qpad = np.zeros((T, qst // 4), dtype=np.float32)
    qpad[:, : n_head * hd] = q.reshape(T, -1)
    qb.put(qpad)
    gb = Buf(T * gst)
    gdata = np.zeros((T, gst // 4), dtype=np.float32)
    gdata[:, : n_head * 2 * hd] = rng.standard_normal((T, n_head * 2 * hd)).astype(np.float32)
    gb.put(gdata)
    gtoff = hd * 4
    gate_arg = (gb, gtoff) if a.gate else None
    gate_ref = gdata[:, hd:] if a.gate else None  # element h * gstride + i of the gate view
    if a.gate:
        need(gb, (T - 1) * gst + gtoff + ((n_head - 1) * gstride + hd) * 4, "gate")
    ob = {k: Buf(T * ost) for k in ("split", "kv", "long")}
    oqb = {k: Buf(T * oqst) for k in ("split", "kv", "long")}
    need(qb, (T - 1) * qst + n_head * hd * 4, "q")
    need(ob["long"], (T - 1) * ost + n_head * hd * 4, "out")
    need(oqb["long"], (T - 1) * oqst + 40 * nbq, "oq")
    # ---- part areas and counters
    CH = 32
    max_ns_old = (kend + CH) // CH
    part_old = Buf(T * n_head * max_ns_old * (hd + 2) * 4)
    pst_old = n_head * max_ns_old * (hd + 2)  # floats per token (as at_pstride)
    cnt = Buf(16 * n_head * 4)
    cnt.zero()
    AL_CH, AL_PSZ, AL_G = 128, 260, a.g
    smax_l = (kend - 1) // AL_CH + 1
    gmax = (smax_l + AL_G - 1) // AL_G
    pst_new = (n_head * (smax_l + gmax + 1) * AL_PSZ + 3) // 4 * 4
    part_new = Buf(T * pst_new * 4)
    gcnt = Buf(16 * n_head * gmax * 4)
    gcnt.zero()
    npair = grp * T
    npg = (npair + 15) // 16
    res = {}

    def run_split():
        for j in range(T):
            pj = pos + j
            ns = (pj + CH) // CH
            need(kc, kvbytes(pj + 1), "split K")
            need(vc, kvbytes(pj + 1), "split V")
            need(part_old, (n_head * ns * (hd + 2)) * 4, "split part")
            need(cnt, n_head * 4, "split count")
            # per-token pointers: q, out, gate move by j strides
            ar = Args((qb, j * qst), kc, vc, part_old, (ob["split"], j * ost), pj, hd, n_head, n_kv, ns, CH, cnt,
                      (oqb["split"], j * oqst) if a.oq else None, (gb, gtoff + j * gst) if a.gate else None, gstride)
            launch(mod, "ie_attn_split", n_head * ns, 1, ar)

    def run_kv():
        smax = (kend - 1 + CH) // CH
        ngs_unused = 0
        nb = (grp * T + 7) // 8
        spw = max(1, min(16, smax * n_kv * nb // 128))
        ngs = (smax + spw - 1) // spw
        pst = 0 if T == 1 else pst_old
        need(kc, kvbytes(kend), "kv K")
        need(vc, kvbytes(kend), "kv V")
        need(part_old, (T - 1) * pst * 4 + n_head * smax * (hd + 2) * 4, "kv part")
        ar = Args(qb, kc, vc, part_old, ob["kv"], pos, hd, n_head, n_kv, smax, CH, cnt,
                  oqb["kv"] if a.oq else None, gate_arg, gstride, qst, ost, oqst, gst, pst, T, nb, spw)
        launch(mod, "ie_attn_kv", n_kv * ngs * nb, 1, ar)

    def run_long(tok0=0, T_=None, kcb=None, vcb=None, outb=None, oqbuf=None, stream=None):
        """tokens tok0 .. tok0 + T_ - 1 of the chunk (the pointers shifted as the host does)."""
        T_ = T if T_ is None else T_
        kcb, vcb, outb = kcb or kc, vcb or vc, outb or ob["long"]
        p0 = pos + tok0
        ke = p0 + T_
        smx = (ke - 1) // AL_CH + 1
        npr = grp * T_
        npg_ = (npr + 15) // 16
        need(kcb, kvbytes(ke), "long K")
        need(vcb, kvbytes(ke), "long V")
        need(qb, (tok0 + T_ - 1) * qst + n_head * hd * 4, "long q")
        need(outb, (tok0 + T_ - 1) * ost + n_head * hd * 4, "long out")
        need(part_new, (tok0 + T_ - 1) * pst_new * 4 + n_head * (smx + (smx + AL_G - 1) // AL_G) * AL_PSZ * 4, "long part")
        need(cnt, (tok0 + T_) * n_head * 4, "long count")
        need(gcnt, (tok0 + T_) * n_head * gmax * 4, "long gcount")
        if a.oq:
            need(oqb["long"], (tok0 + T_ - 1) * oqst + 40 * nbq, "long oq")
        # --pgloop: one workgroup per split and KV head does every pair group (npg with bit 15 set)
        npga, nwgp = (npg_ | 0x8000, 1) if a.pgloop and npg_ > 1 else (npg_, npg_)
        ar = Args((qb, tok0 * qst), kcb, vcb, (part_new, tok0 * pst_new * 4), (outb, tok0 * ost), p0, n_head, n_kv, T_, npga,
                  (cnt, tok0 * n_head * 4), (gcnt, tok0 * n_head * gmax * 4), gmax, (oqb["long"], tok0 * oqst) if a.oq else None,
                  (gb, gtoff + tok0 * gst) if a.gate else None, gstride, qst, ost, oqst, gst, pst_new, None)  # tr: no tree
        launch(mod, "ie_attn_long_e" if a.early else "ie_attn_long", n_kv * smx * nwgp, 1, ar, stream)

    def fetch(name):
        return ob[name].get(np.float32, T * ost // 4).reshape(T, ost // 4)[:, : n_head * hd].reshape(T, n_head, hd)

    if a.check:
        print("check: pos %d T %d (keys %d), %s KV, dist %s, gate %d, oq %d" % (pos, T, kend, "q4" if a.q4 else "q8" if a.q8 else "f16", a.dist,
                                                                                a.gate, a.oq))
        cnt.zero()
        ref = reference(q, Kf, Vf, pos, T, n_head, n_kv, hd, gate_ref, gstride)
        outs = {}
        if "split" in kernels:
            run_split()
            sync()
            outs["split"] = fetch("split")
        if "kv" in kernels:
            run_kv()
            sync()
            outs["kv"] = fetch("kv")
        if "long" in kernels:
            run_long()
            sync()
            outs["long"] = fetch("long")
            cnt_after = cnt.get(np.uint32, 16 * n_head)
            gc_after = gcnt.get(np.uint32, 16 * n_head * gmax)
            print("  counters back to 0: %s" % (not cnt_after.any() and not gc_after.any()))
        for k, o in outs.items():
            print("  %-5s vs float64 reference: rel %.3e   finite %s" % (k, relerr(o, ref), bool(np.isfinite(o).all())))
        if "long" in outs and "split" in outs:
            print("  long vs split: rel %.3e   kv vs split bitwise %s" % (
                relerr(outs["long"], outs["split"]), bool((outs["kv"] == outs["split"]).all()) if "kv" in outs else "n/a"))
        if "long" in outs:
            # a hash of the output bits, to compare two kernel objects (the inputs are seeded: same bits in, same hash out)
            import hashlib
            hh = hashlib.sha256(np.ascontiguousarray(outs["long"]).tobytes())
            if a.oq:
                hh.update(np.ascontiguousarray(oqb["long"].get(np.int8, T * oqst)).tobytes())
            print("  long output sha256: %s" % hh.hexdigest()[:24])
            # each token alone (T = 1 launches) must give the same bits
            full = outs["long"].copy()
            same = True
            for j in range(T):
                ob["long"].zero()
                ob["long"].put(np.full(T * ost // 4, np.nan, dtype=np.float32))
                run_long(tok0=j, T_=1)
                sync()
                one = fetch("long")[j]
                eq = bool((one.view(np.uint32) == full[j].view(np.uint32)).all())
                same &= eq
                if not eq:
                    print("  token %d alone differs from the chunk: max abs %.3e" % (j, float(np.abs(one - full[j]).max())))
            print("  T=1 per token bitwise equal to the T=%d chunk: %s" % (T, same))
            # a split chunk (first half, then second half) must equal the whole as well
            if T > 1:
                ob["long"].put(np.full(T * ost // 4, np.nan, dtype=np.float32))
                h = T // 2
                run_long(tok0=0, T_=h)
                run_long(tok0=h, T_=T - h)
                sync()
                two = fetch("long")
                print("  chunk split in %d + %d bitwise equal: %s" % (h, T - h, bool((two.view(np.uint32) == full.view(np.uint32)).all())))
            if a.oq and "split" in outs:
                oqn = oqb["long"].get(np.int8, T * oqst)
                oqo = oqb["split"].get(np.int8, T * oqst)
                d = 0
                for j in range(T):
                    x = oqn[j * oqst : j * oqst + 32 * nbq].astype(np.int32)
                    y = oqo[j * oqst : j * oqst + 32 * nbq].astype(np.int32)
                    d = max(d, int(np.abs(x - y).max()))
                print("  oq (int8 values) max |new - old| = %d" % d)

    if a.time:
        bytes_kv = 2 * kend * n_kv * krow
        layers = a.layers or max(2, min(32, (384 << 20) // max(bytes_kv, 1)))
        if a.engkv:  # every layer in arrays mapped as the engine maps them
            kcs, vcs = eng_kv(layers, kend, n_kv * krow, a.engkv, a.chunk)
            kcs[0].put(Kd), vcs[0].put(Vd)
        else:
            kcs = [kc] + [Buf(kvbytes(kend), vmm=a.vmm) for _ in range(layers - 1)]
            vcs = [vc] + [Buf(kvbytes(kend), vmm=a.vmm) for _ in range(layers - 1)]
        for b in kcs[1:]:
            b.put(Kd)
        for b in vcs[1:]:
            b.put(Vd)
        print("time: pos %d T %d (keys %d), %s KV, %d layers cycled, KV %.1f MB per layer" % (
            pos, T, kend, "q4" if a.q4 else "q8" if a.q8 else "f16", layers, bytes_kv / 1e6))
        stream = C.c_void_p()
        ck(H.hipStreamCreate(C.byref(stream)), "hipStreamCreate")
        e0, e1 = C.c_void_p(), C.c_void_p()
        ck(H.hipEventCreate(C.byref(e0)), "ev")
        ck(H.hipEventCreate(C.byref(e1)), "ev")

        def timeit(name, fn):
            reps = a.reps
            # warm up, then capture reps launches over the layers into a graph and replay it
            for i in range(min(layers, 4)):
                fn(i, None)
            sync()
            g, ge = C.c_void_p(), C.c_void_p()
            ck(H.hipStreamBeginCapture(stream, 2), "capture")
            for i in range(reps):
                fn(i % layers, stream)
            ck(H.hipStreamEndCapture(stream, C.byref(g)), "end capture")
            ck(H.hipGraphInstantiate(C.byref(ge), g, None, None, 0), "instantiate")
            ck(H.hipGraphLaunch(ge, stream), "graph launch")
            ck(H.hipStreamSynchronize(stream), "sync")
            best = 1e30
            for _ in range(5):
                ck(H.hipEventRecord(e0, stream), "rec")
                ck(H.hipGraphLaunch(ge, stream), "graph launch")
                ck(H.hipEventRecord(e1, stream), "rec")
                ck(H.hipEventSynchronize(e1), "evsync")
                ms = C.c_float()
                ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
                best = min(best, ms.value / reps)
            print("  %-6s %8.1f us   %6.1f GB/s of KV" % (name, best * 1000, bytes_kv / best / 1e6))
            # one launch at a time (events around it, waited for): its latency, as the model sees it between other kernels
            one = []
            for i in range(2 * layers):
                ck(H.hipEventRecord(e0, stream), "rec")
                fn(i % layers, stream)
                ck(H.hipEventRecord(e1, stream), "rec")
                ck(H.hipEventSynchronize(e1), "evsync")
                ms = C.c_float()
                ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
                one.append(ms.value)
            one.sort()
            print("  %-6s %8.1f us   one launch at a time (median; min %.1f)" % (name + "1", one[len(one) // 2] * 1000, one[0] * 1000))

        def f_long(i, st):
            run_long(kcb=kcs[i], vcb=vcs[i], stream=st)

        def f_kv(i, st):
            smax = (kend - 1 + CH) // CH
            nb = (grp * T + 7) // 8
            spw = max(1, min(16, smax * n_kv * nb // 128))
            ngs = (smax + spw - 1) // spw
            ar = Args(qb, kcs[i], vcs[i], part_old, ob["kv"], pos, hd, n_head, n_kv, smax, CH, cnt,
                      oqb["kv"] if a.oq else None, gate_arg, gstride, qst, ost, oqst, gst, 0 if T == 1 else pst_old, T, nb, spw)
            launch(mod, "ie_attn_kv", n_kv * ngs * nb, 1, ar, st)

        def f_split(i, st):
            ns = (pos + CH) // CH
            ar = Args(qb, kcs[i], vcs[i], part_old, ob["split"], pos, hd, n_head, n_kv, ns, CH, cnt,
                      oqb["split"] if a.oq else None, gate_arg, gstride)
            launch(mod, "ie_attn_split", n_head * ns, 1, ar, st)

        cnt.zero()
        if "split" in kernels and T == 1:
            timeit("split", f_split)
        if "kv" in kernels:
            timeit("kv", f_kv)
        if "long" in kernels:
            timeit("long", f_long)


if __name__ == "__main__":
    main()
