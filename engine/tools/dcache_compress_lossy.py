"""Lossy storage of the state / ring for reference: int16 (one f32 scale per 128 values), int8 likewise, bf16, fp16.

usage: python dcache_compress_lossy.py FILE.iec
Error is measured against the original f32: per-element relative error |x'-x|/|x| (elements with x == 0 skipped),
error relative to the group maximum, and the relative RMS error ||x'-x|| / ||x|| overall and for the worst head matrix.
Sizes: raw stored size and zstd -1 of the stored bytes (byte-plane split for 16-bit types).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import zstandard as zstd
import dcache_compress_common as C

CC1 = zstd.ZstdCompressor(level=1)


def z1(a):
    return len(CC1.compress(np.ascontiguousarray(a).tobytes()))


def stats(label, x, y, stored_bytes, zbytes, group_max=None, mat=None):
    n = x.size
    err = np.abs(y.astype(np.float32) - x)
    nz = x != 0
    rel = err[nz] / np.abs(x[nz])
    line = (f"  {label:30s} stored {stored_bytes/1e6:7.2f} MB (x{x.nbytes/stored_bytes:5.3f})  +zstd1 {zbytes/1e6:7.2f} MB (x{x.nbytes/zbytes:5.3f}) | "
            f"rel err/elem: median {np.median(rel):.2e} mean {rel.mean():.2e} p99 {np.percentile(rel,99):.2e} p99.9 {np.percentile(rel,99.9):.2e} max {rel.max():.2e}")
    print(line)
    rms = float(np.sqrt((err.astype(np.float64) ** 2).sum() / (x.astype(np.float64) ** 2).sum()))
    extra = f"    overall relative RMS {rms:.3e}"
    if mat:
        e2 = (err.astype(np.float64) ** 2).reshape(-1, mat).sum(1)
        x2 = (x.astype(np.float64) ** 2).reshape(-1, mat).sum(1)
        r = np.sqrt(e2 / np.maximum(x2, 1e-300))
        extra += f"; per head matrix: median {np.median(r):.3e}, worst {r.max():.3e}"
    if group_max is not None:
        extra += f"; max |err| / group max {float((err.reshape(-1, 128).max(1) / group_max).max()):.3e}"
    print(extra)


def run(name, x, mat):
    print(f"\n=== {name}: {x.size} f32 = {x.nbytes/1e6:.2f} MB; mean|x| {np.abs(x).mean():.3e}, max|x| {np.abs(x).max():.3g}; raw zstd1 {z1(x)/1e6:.2f} MB")
    g = x.reshape(-1, 128)
    gm = np.abs(g).max(1)
    for bits, qmax, dtype in ((16, 32767, np.int16), (8, 127, np.int8)):
        scale = np.where(gm > 0, gm / qmax, 1.0).astype(np.float32)
        q = np.rint(g / scale[:, None]).astype(dtype)
        y = (q.astype(np.float32) * scale[:, None]).reshape(-1)
        stored = q.nbytes + scale.nbytes
        qb = q.view(np.uint8).reshape(-1, bits // 8).T if bits == 16 else q
        zb = z1(qb) + z1(scale)
        stats(f"int{bits} + f32 scale per 128", x, y, stored, zb, gm, mat)
    # bf16 round-to-nearest-even
    u = x.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    y = (r.astype(np.uint32) << 16).view(np.float32)
    stats("bf16 (RTNE)", x, y, r.nbytes, z1(r.view(np.uint8).reshape(-1, 2).T), None, mat)
    # fp16
    h = x.astype(np.float16)
    y = h.astype(np.float32)
    under = float(((np.abs(x) < 6.1e-5) & (x != 0)).mean())
    stats(f"fp16 (flushes {under*100:.1f}% below 6.1e-5)", x, y, h.nbytes, z1(h.view(np.uint8).reshape(-1, 2).T), None, mat)


if __name__ == "__main__":
    path = sys.argv[1]
    h = C.read_head(path)
    s = C.e_sections(h)
    st = np.fromfile(path, dtype=np.float32, count=C.STATE_BYTES // 4, offset=s["state"])
    run("linear-attention state", st, C.SD * C.SD)
    del st
    rg = np.fromfile(path, dtype=np.float32, count=C.RING_BYTES // 4, offset=s["ring"])
    want = [(h["pos"] - 1 - i) & 31 for i in range(3)]
    r3 = np.ascontiguousarray(rg.reshape(C.N_REC, C.RING_SLOTS, C.CONV_DIM)[:, want, :]).reshape(-1)
    run(f"conv ring, 3 needed slots {want}", r3, C.CONV_DIM)
