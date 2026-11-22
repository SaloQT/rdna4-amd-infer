"""Lossless compressibility of the f32 parts of a checkpoint (linear-attention state, conv ring).

usage: python -I dcache_compress_f32.py FILE.iec [state|ring|both]
Read-only. Prints tables: field entropy, per-plane sizes (order-0 bound, zstd), whole-buffer schemes, per-layer spread.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import zstandard as zstd
import lz4.block, lz4.frame
import dcache_compress_common as C


def z(buf, level):
    return len(zstd.ZstdCompressor(level=level).compress(bytes(memoryview(np.ascontiguousarray(buf)).cast("B"))))


def l4(buf, hc=False):
    b = bytes(memoryview(np.ascontiguousarray(buf)).cast("B"))
    if hc:
        return len(lz4.block.compress(b, mode="high_compression", compression=9, store_size=False))
    return len(lz4.block.compress(b, mode="default", store_size=False))


def mb(x):
    return f"{x/1e6:9.2f}"


def H_hist(cnt):
    return C.entropy_bits(cnt)


def planes_lo_to_hi(u32):
    v = np.ascontiguousarray(u32).view(np.uint8).reshape(-1, 4)
    return [np.ascontiguousarray(v[:, k]) for k in range(4)]  # p0 = low mantissa byte ... p3 = sign+exp[7:1]


def split_concat(u32):
    return np.ascontiguousarray(np.ascontiguousarray(u32).view(np.uint8).reshape(-1, 4).T).reshape(-1)  # 4 contiguous planes, low byte first


def xor_prev(u32, row):
    x = u32.reshape(-1, row)
    y = x.copy()
    y[:, 1:] ^= x[:, :-1]
    return y.reshape(-1)


def xor_prev_axis(u32, rows, cols):
    """XOR with the element one row up (stride cols) inside each rows x cols matrix"""
    x = u32.reshape(-1, rows, cols)
    y = x.copy()
    y[:, 1:, :] ^= x[:, :-1, :]
    return y.reshape(-1)


def analyze(name, u32, row, matrix=None, unit_floats=None, per_unit_label="layer", units=None):
    n = u32.size
    raw = n * 4
    print(f"\n=== {name}: {n} f32 = {raw/1e6:.2f} MB ===")
    # value facts
    f = u32.view(np.float32)
    zero = int((u32 & 0x7FFFFFFF == 0).sum())
    exp = ((u32 >> 23) & 0xFF)
    den = int(((exp == 0) & (u32 & 0x7FFFFFFF != 0)).sum())
    nan = int((exp == 255).sum())
    print(f"exact zeros {zero/n*100:.3f}%  denormals {den/n*100:.4f}%  inf/nan {nan}  |x| max {np.abs(f).max():.4g} "
          f"mean|x| {np.abs(f).mean():.4g}")
    # field entropies
    sign_exp9 = (u32 >> 23).astype(np.uint16)  # 9 bit
    mant_hi = ((u32 >> 15) & 0xFF).astype(np.uint8)  # top 8 mantissa bits
    mant_mid = ((u32 >> 7) & 0xFF).astype(np.uint8)  # next 8
    mant_lo = (u32 & 0x7F).astype(np.uint8)  # low 7
    rows = []
    for nm, arr, bits in (("sign+exp (9b)", sign_exp9, 9), ("mantissa[22:15] (8b)", mant_hi, 8),
                          ("mantissa[14:7] (8b)", mant_mid, 8), ("mantissa[6:0] (7b)", mant_lo, 7)):
        cnt = np.bincount(arr, minlength=1 << 9)
        h = H_hist(cnt)
        rows.append((nm, bits, h, h * n / 8))
    print("field entropy (global order-0):")
    print(f"  {'field':24s} {'bits':>5s} {'H bits/sym':>11s} {'bound MB':>10s}")
    tot = 0
    for nm, bits, h, b in rows:
        print(f"  {nm:24s} {bits:5d} {h:11.3f} {b/1e6:10.2f}")
        tot += b
    print(f"  fields total bound: {tot/1e6:.2f} MB = ratio {raw/tot:.3f}  (saves {(1-tot/raw)*100:.1f}%)")
    # conditional: exponent given left neighbour's exponent
    ex = exp.astype(np.uint8).reshape(-1, row)
    joint = ex[:, 1:].astype(np.uint32) * 256 + ex[:, :-1]
    jc = np.bincount(joint.reshape(-1), minlength=65536)
    hj = H_hist(jc)
    hl = H_hist(np.bincount(ex[:, :-1].reshape(-1), minlength=256))
    hexp = H_hist(np.bincount(ex.reshape(-1), minlength=256))
    print(f"  H(exp)={hexp:.3f} b  H(exp | left neighbour exp)={hj-hl:.3f} b  (context modelling gain on the 8 exp bits)")
    # mant_hi given exponent
    j2 = exp.astype(np.uint32) * 256 + mant_hi
    h2 = H_hist(np.bincount(j2, minlength=65536)) - hexp
    print(f"  H(mantissa top 8 | exp)={h2:.3f} b of 8")

    # per byte plane
    pl = planes_lo_to_hi(u32)
    names = ["byte0 (mant[7:0])", "byte1 (mant[15:8])", "byte2 (exp0+mant[22:16])", "byte3 (sign+exp[7:1])"]
    print("per byte plane (little-endian byte k of each f32):")
    print(f"  {'plane':26s} {'H b/B':>6s} {'bound':>8s} {'zstd1':>8s} {'zstd3':>8s} {'lz4':>8s}   (MB; raw plane = {n/1e6:.2f})")
    plane_sz = {}
    for k in range(4):
        h = C.entropy_bits(np.bincount(pl[k], minlength=256))
        a, b, c = z(pl[k], 1), z(pl[k], 3), l4(pl[k])
        plane_sz[k] = (h * n / 8, a, b, c)
        print(f"  {names[k]:26s} {h:6.3f} {h*n/8/1e6:8.2f} {a/1e6:8.2f} {b/1e6:8.2f} {c/1e6:8.2f}")
    # with xor
    xr = xor_prev(u32, row)
    plx = planes_lo_to_hi(xr)
    print("same planes after XOR with the left neighbour (u32 xor):")
    for k in range(4):
        h = C.entropy_bits(np.bincount(plx[k], minlength=256))
        a, b = z(plx[k], 1), z(plx[k], 3)
        print(f"  {names[k]:26s} {h:6.3f} {h*n/8/1e6:8.2f} {a/1e6:8.2f} {b/1e6:8.2f}")
    if matrix:
        xm = xor_prev_axis(u32, *matrix)
        plm = planes_lo_to_hi(xm)
        print(f"same planes after XOR with the element one row up ({matrix[1]} floats earlier):")
        for k in range(4):
            h = C.entropy_bits(np.bincount(plm[k], minlength=256))
            print(f"  {names[k]:26s} {h:6.3f} {h*n/8/1e6:8.2f} {z(plm[k],1)/1e6:8.2f} {z(plm[k],3)/1e6:8.2f}")

    # whole-buffer schemes
    print("whole-buffer schemes (MB, ratio = raw/size):")
    res = []

    def add(label, size, t=None):
        res.append((label, size))
        print(f"  {label:58s} {size/1e6:9.2f}  x{raw/size:6.3f}  saves {(1-size/raw)*100:5.1f}%")

    t0 = time.time()
    add("raw zstd -1", z(u32, 1))
    add("raw zstd -3", z(u32, 3))
    add("raw zstd -9", z(u32, 9))
    add("raw lz4 (default)", l4(u32))
    sc = split_concat(u32)
    add("byte-plane split + zstd -1", z(sc, 1))
    add("byte-plane split + zstd -3", z(sc, 3))
    add("byte-plane split + zstd -9", z(sc, 9))
    add("byte-plane split + lz4", l4(sc))
    add("byte-plane split, planes 0-1 stored raw, planes 2-3 zstd -3",
        2 * n + z(sc[2 * n:], 3))
    add("byte-plane split, planes 0-1 raw, planes 2-3 zstd -1", 2 * n + z(sc[2 * n:], 1))
    add("byte-plane split, plane 0 raw, planes 1-3 zstd -3", n + z(sc[n:], 3))
    scx = split_concat(xr)
    add("XOR-left + byte-plane split + zstd -1", z(scx, 1))
    add("XOR-left + byte-plane split + zstd -3", z(scx, 3))
    add("XOR-left + byte-plane split + lz4", l4(scx))
    bnd = sum(plane_sz[k][0] for k in range(4))
    add("entropy bound: 4 byte planes, global order-0 (Huffman/ANS ideal)", bnd)
    # chunked adaptive order-0 bound: table per unit (a head matrix or a layer)
    if units:
        per = u32.reshape(units, -1)
        tot = 0.0
        for u in range(units):
            for k in range(4):
                tot += C.entropy_bits(np.bincount(planes_lo_to_hi(per[u])[k], minlength=256)) * per.shape[1] / 8
        add(f"entropy bound: 4 byte planes, order-0 table per {per_unit_label} ({units} units)", tot)
    # fields-split bound
    tot_fields = sum(r[3] for r in rows)
    add("entropy bound: sign+exp 9b, mant 8b/8b/7b, global order-0", tot_fields)
    # exponent coded with the left neighbour's exponent as context (sign stays inside the 9b symbol: subtract the gain)
    gain = (hexp - (hj - hl)) * n / 8
    add("entropy bound: as above, exponent given left neighbour's exponent", tot_fields - gain)
    print(f"  [{time.time()-t0:.0f}s]")
    return res


def per_unit_spread(name, u32, units, label):
    per = u32.reshape(units, -1)
    n = per.shape[1]
    rat_bound, rat_z = [], []
    for u in range(units):
        sc = split_concat(per[u])
        b = sum(C.entropy_bits(np.bincount(p, minlength=256)) * n / 8 for p in planes_lo_to_hi(per[u]))
        rat_bound.append(b / (n * 4))
        rat_z.append(z(sc, 3) / (n * 4))
    rb, rz = np.array(rat_bound), np.array(rat_z)
    print(f"\nper-{label} spread of compressed fraction ({name}, {units} units): ")
    print(f"  plane-split+zstd3: min {rz.min():.3f} mean {rz.mean():.3f} max {rz.max():.3f}  | order-0 bound: "
          f"min {rb.min():.3f} mean {rb.mean():.3f} max {rb.max():.3f}")
    idx = np.argsort(rz)
    print("  lowest 3:", [(int(i), round(float(rz[i]), 3)) for i in idx[:3]], " highest 3:",
          [(int(i), round(float(rz[i]), 3)) for i in idx[-3:]])
    print("  per layer fractions (zstd3 split): " + " ".join(f"{x:.2f}" for x in rz))
    return rz


if __name__ == "__main__":
    path = sys.argv[1]
    what = sys.argv[2] if len(sys.argv) > 2 else "both"
    h = C.read_head(path)
    s = C.e_sections(h)
    print(f"file {os.path.basename(path)} pos={h['pos']} kind={h['kind']}")
    if what in ("state", "both"):
        st = np.fromfile(path, dtype=np.uint32, count=C.STATE_BYTES // 4, offset=s["state"])
        analyze("linear-attention state", st, row=C.SD, matrix=(C.SD, C.SD), units=C.N_REC * C.N_VH, per_unit_label="head matrix")
        per_unit_spread("state", st, C.N_REC, "layer")
        # per head within layer 0 / last for spread
        del st
    if what in ("ring", "both"):
        rg = np.fromfile(path, dtype=np.uint32, count=C.RING_BYTES // 4, offset=s["ring"])
        analyze("conv ring (all 32 slots)", rg, row=C.CONV_DIM, units=C.N_REC * C.RING_SLOTS, per_unit_label="slot row")
        per_unit_spread("ring", rg, C.N_REC, "layer")
        pos = h["pos"]
        want = [(pos - 1 - i) & 31 for i in range(3)]
        r3 = rg.reshape(C.N_REC, C.RING_SLOTS, C.CONV_DIM)[:, want, :].copy().reshape(-1)
        analyze(f"conv ring, only the 3 needed slots {want}", r3, row=C.CONV_DIM, units=C.N_REC * 3, per_unit_label="slot row")
