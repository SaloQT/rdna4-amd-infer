"""Delta between consecutive checkpoints of one conversation (state only; ring/h noted separately).

usage: python dcache_compress_delta.py
Pairs of conversation A (the one that reaches 26736 tokens): 4096->8192->12288->16384->20480->24576->26731 and 26731->26736.
For every pair: standalone size of the later state (byte-plane split + zstd -1), size of XOR(a,b) and of u32 difference
(b - a mod 2^32) after byte-plane split + zstd -1, order-0 bound of the planes, fraction of bytes equal.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import zstandard as zstd
import dcache_compress_common as C

POS = [4096, 8192, 12288, 16384, 20480, 24576, 26731, 26736]


def z1(a):
    return len(zstd.ZstdCompressor(level=1).compress(a.tobytes()))


def planes(u32):
    return np.ascontiguousarray(np.ascontiguousarray(u32).view(np.uint8).reshape(-1, 4).T)  # (4, n) low byte first


def report(label, u32, n):
    p = planes(u32)
    h = [C.entropy_bits(np.bincount(p[k], minlength=256)) for k in range(4)]
    sz = z1(p.reshape(-1))
    bound = sum(h) * n / 8
    zero = float((u32 == 0).mean())
    print(f"    {label:22s} zstd1(split) {sz/1e6:8.2f} MB ({sz/(4*n)*100:5.1f}% of 151)  order-0 bound {bound/1e6:8.2f} MB  "
          f"H/plane(lo..hi) {h[0]:.2f} {h[1]:.2f} {h[2]:.2f} {h[3]:.2f}  zero words {zero*100:5.1f}%")
    return sz


def state(path):
    h = C.read_head(path)
    return np.fromfile(path, dtype=np.uint32, count=C.STATE_BYTES // 4, offset=C.e_sections(h)["state"])


if __name__ == "__main__":
    paths = {}
    for p in POS:
        paths[p] = C.e_path_by_pos(p, kind={26731: 0, 26736: 1}.get(p))
    prev = None
    n = C.STATE_BYTES // 4
    for i, p in enumerate(POS):
        cur = state(paths[p])
        print(f"\ncheckpoint pos={p} ({os.path.basename(paths[p])[-20:-4]})")
        report("standalone", cur, n)
        if prev is not None:
            pp = POS[i - 1]
            print(f"  delta from pos={pp} ({p-pp} tokens earlier):")
            report("XOR", cur ^ prev, n)
            report("u32 difference", cur - prev, n)
            # float relative change
            a, b = prev.view(np.float32), cur.view(np.float32)
            rel = np.abs(b - a) / (np.abs(a) + 1e-30)
            print(f"    relative change |b-a|/|a|: median {np.median(rel):.3g}, p10 {np.percentile(rel,10):.3g}, "
                  f"p90 {np.percentile(rel,90):.3g}; identical words {(a == b).mean()*100:.2f}%")
        prev = cur
    # the ring, h row and token / KV-tail duplication between 26731 and 26736
    a, b = paths[26731], paths[26736]
    ha, hb = C.read_head(a), C.read_head(b)
    sa, sb = C.e_sections(ha), C.e_sections(hb)
    ra = np.fromfile(a, dtype=np.uint8, count=C.RING_BYTES, offset=sa["ring"])
    rb = np.fromfile(b, dtype=np.uint8, count=C.RING_BYTES, offset=sb["ring"])
    ta = np.fromfile(a, dtype=np.uint8, count=sa["tail"] * ha["rowb"], offset=sa["kv"])
    tb = np.fromfile(b, dtype=np.uint8, count=sb["tail"] * hb["rowb"], offset=sb["kv"])
    # tails: layer-major [layer][K|V][pos]; compare the first min(tail) positions of each (layer,kv)
    na, nb_ = sa["tail"], sb["tail"]
    ra_ = ta.reshape(-1)
    same = True
    off_a = off_b = 0
    eq = tot = 0
    for _ in range(C.N_KVL * 2):
        xa = ta[off_a:off_a + na * C.ROW_HALF]
        xb = tb[off_b:off_b + nb_ * C.ROW_HALF]
        eq += int((xa == xb[:na * C.ROW_HALF]).sum())
        tot += xa.size
        off_a += na * C.ROW_HALF
        off_b += nb_ * C.ROW_HALF
    print(f"\n26731 vs 26736: KV tail rows of the first {na} positions identical in {eq/tot*100:.2f}% of bytes")
    print(f"ring bytes identical: {(ra == rb).mean()*100:.2f}%; tokens of 26731 are a prefix of 26736: "
          f"{(np.fromfile(a, dtype=np.uint32, count=ha['pos'], offset=56) == np.fromfile(b, dtype=np.uint32, count=ha['pos'], offset=56)).all()}")
