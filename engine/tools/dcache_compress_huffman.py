"""Actual Huffman code lengths (not just entropy) for the high byte plane (sign + 7 exponent bits) of the state.

usage: python dcache_compress_huffman.py FILE.iec
Chunk = symbols per code table.  Table cost: 128 B (256 x 4-bit lengths, canonical code).  Lengths are unconstrained;
the maximum length seen is printed (a decoder wants <= 11-12; package-merge costs a hair more).
"""
import sys, os, heapq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dcache_compress_common as C


def huff_lengths(cnt):
    items = [(int(c), i, [i]) for i, c in enumerate(cnt) if c]
    if len(items) == 1:
        return {items[0][2][0]: 1}
    heapq.heapify(items)
    ln = {i: 0 for _, i, _ in items}
    uid = 256
    while len(items) > 1:
        a = heapq.heappop(items)
        b = heapq.heappop(items)
        for s in a[2] + b[2]:
            ln[s] += 1
        heapq.heappush(items, (a[0] + b[0], uid, a[2] + b[2]))
        uid += 1
    return ln


def cost(hi, chunk):
    n = hi.size
    tot_bits = 0
    mx = 0
    tables = 0
    for off in range(0, n, chunk):
        part = hi[off:off + chunk]
        cnt = np.bincount(part, minlength=256)
        ln = huff_lengths(cnt)
        tot_bits += sum(cnt[s] * l for s, l in ln.items())
        mx = max(mx, max(ln.values()))
        tables += 1
    ent = 0.0
    for off in range(0, n, chunk):
        ent += C.entropy_bits(np.bincount(hi[off:off + chunk], minlength=256)) * min(chunk, n - off) / 8
    return tot_bits / 8, tables * 128, mx, ent


if __name__ == "__main__":
    path = sys.argv[1]
    h = C.read_head(path)
    s = C.e_sections(h)
    st = np.fromfile(path, dtype=np.uint32, count=C.STATE_BYTES // 4, offset=s["state"])
    hi = (st >> 24).astype(np.uint8)
    n = hi.size
    print(f"hi-byte plane {n/1e6:.2f} MB; zstd -1 of it was 15.41 MB, order-0 global bound 16.54 MB")
    print(f"{'chunk (symbols)':>18s} {'payload MB':>11s} {'tables MB':>10s} {'total MB':>9s} {'order-0 bound MB':>17s} {'max len':>8s}  state total (3 raw bytes + this)")
    for chunk in (n, 786432, 65536, 16384, 4096):
        b, t, mx, ent = cost(hi, chunk)
        print(f"{chunk:18d} {b/1e6:11.2f} {t/1e6:10.3f} {(b+t)/1e6:9.2f} {ent/1e6:17.2f} {mx:8d}  {(3*n+b+t)/1e6:8.2f} MB (saves {(1-(3*n+b+t)/(4*n))*100:4.1f}%)")
