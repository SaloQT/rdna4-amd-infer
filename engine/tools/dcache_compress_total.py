"""Measured compressed sizes of every file of conversation A (the 26.7K-token prompt: 8 checkpoints + 26 KV blocks).

usage: python dcache_compress_total.py
Schemes per file part (sizes in bytes, all measured with python-zstandard 1.5.7 on the real files):
  state : raw | zstd1(raw) | hi-byte plane zstd1 (other 3 bytes raw) | all planes zstd1 (same size, tested)
  ring  : raw 62.9 MB | planes+zstd1 on the whole ring | planes+zstd1 on the 3 needed slots only (the other agent's trim)
  kv    : raw | zstd1(raw) | channel-major nibbles + planes of fp16 scales, zstd1 each
  rest  : tokens, h row, header: stored raw (tiny)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import zstandard as zstd
import dcache_compress_common as C

POS = [(4096, None), (8192, None), (12288, None), (16384, None), (20480, None), (24576, None), (26731, 0), (26736, 1)]
CC1 = zstd.ZstdCompressor(level=1)


def z1(a):
    return len(CC1.compress(np.ascontiguousarray(a).tobytes()))


def planes(u32):
    return np.ascontiguousarray(np.ascontiguousarray(u32).view(np.uint8).reshape(-1, 4).T).reshape(-1)


def hi_only(u32):
    v = np.ascontiguousarray(u32).view(np.uint8).reshape(-1, 4)
    return v.shape[0] * 3 + z1(v[:, 3])


def kv_scheme(rows, n):
    """rows: uint8 array of 17*2*n*640 bytes in file order. returns zstd1 size of the channel-major scheme"""
    if n == 0:
        return 0
    a = rows.reshape(C.N_KVL, 2, n, C.KV_HEADS, C.REC)
    nb = a[..., :128]
    nib = np.empty(nb.shape[:-1] + (128, 2), dtype=np.uint8)
    nib[..., 0] = nb & 15
    nib[..., 1] = nb >> 4
    nib = nib.reshape(nb.shape[:-1] + (256,))
    cm = np.ascontiguousarray(nib.transpose(0, 1, 3, 4, 2)).reshape(-1)  # layer, kv, head, chan, pos
    if cm.size % 2:
        cm = np.append(cm, 0)
    packed = (cm[0::2] | (cm[1::2] << 4)).astype(np.uint8)
    sc = np.ascontiguousarray(a[..., 128:]).reshape(-1, 2).T.reshape(-1)
    return z1(packed) + z1(sc)


def main():
    t0 = time.time()
    tot = {}

    def add(k, v):
        tot[k] = tot.get(k, 0) + v

    print(f"{'ckpt':>6s} {'kind':>4s} {'file MB':>8s} | state: {'raw':>7s} {'zstd1':>7s} {'hi-byte':>7s} | ring(full): {'raw':>6s} {'planes+z1':>9s} | ring(3 slots): {'raw':>5s} {'z1':>6s} | kv tail raw/scheme")
    for pos, kind in POS:
        p = C.e_path_by_pos(pos, kind)
        h = C.read_head(p)
        s = C.e_sections(h)
        size = os.path.getsize(p)
        st = np.fromfile(p, dtype=np.uint32, count=C.STATE_BYTES // 4, offset=s["state"])
        rg = np.fromfile(p, dtype=np.uint32, count=C.RING_BYTES // 4, offset=s["ring"])
        want = [(pos - 1 - i) & 31 for i in range(3)]
        r3 = np.ascontiguousarray(rg.reshape(C.N_REC, C.RING_SLOTS, C.CONV_DIM)[:, want, :]).reshape(-1)
        tail = np.fromfile(p, dtype=np.uint8, count=s["tail"] * h["rowb"], offset=s["kv"])
        # tail file order: [layer][K|V][tail positions] with rows of 640 B: same layout as a block
        a_zraw = z1(st)
        a_hi = hi_only(st)
        r_full = z1(planes(rg))
        r_3 = z1(planes(r3))
        kv_t = kv_scheme(tail, s["tail"])
        # the whole file, zstd1, untransformed (naive "compress the file")
        whole = len(CC1.compress(open(p, "rb").read()))
        other = 56 + 4 * pos + 20484
        add("e_file_raw", size), add("e_whole_zstd1", whole)
        add("state_raw", C.STATE_BYTES), add("state_zstd1", a_zraw), add("state_hi", a_hi)
        add("ring_raw", C.RING_BYTES), add("ring_full_planes", r_full)
        add("ring3_raw", r3.nbytes), add("ring3_z1", r_3)
        add("tail_raw", tail.nbytes), add("tail_scheme", kv_t), add("other", other)
        print(f"{pos:6d} {kind if kind is not None else 3:4d} {size/1e6:8.2f} | {C.STATE_BYTES/1e6:13.2f} {a_zraw/1e6:7.2f} {a_hi/1e6:7.2f} | {C.RING_BYTES/1e6:18.2f} {r_full/1e6:9.2f} | {r3.nbytes/1e6:19.2f} {r_3/1e6:6.2f} | {tail.nbytes/1e6:.2f}/{kv_t/1e6:.2f}  whole-file zstd1 {whole/1e6:.2f}  [{time.time()-t0:.0f}s]")
        sys.stdout.flush()
    ks = C.k_paths_of(C.e_path_by_pos(26731, 0))
    kraw = kz = kzraw = 0
    for p in ks:
        raw = np.fromfile(p, dtype=np.uint8, offset=56 + 4096)
        kraw += os.path.getsize(p)
        kzraw += z1(raw)
        kz += kv_scheme(raw, 1024)
        add("k_other", 56 + 4096)
    print(f"\n26 KV blocks: raw files {kraw/1e6:.1f} MB; zstd1 of body {kzraw/1e6:.1f} MB; channel-major scheme {kz/1e6:.1f} MB (+ header/tokens {26*(56+4096)/1e6:.2f} MB raw)")
    tot["k_raw"], tot["k_zraw"], tot["k_scheme"] = kraw, kzraw, kz

    print("\n=== totals for a fresh 26.7K-token prompt (8 checkpoints + 26 blocks), bytes ===")
    cur = tot["e_file_raw"] + tot["k_raw"]
    print(f"current files: e {tot['e_file_raw']/1e9:.3f} GB + k {tot['k_raw']/1e9:.3f} GB = {cur/1e9:.3f} GB")
    # with the ring trimmed to the 3 needed slots
    trimmed = cur - tot["ring_raw"] + tot["ring3_raw"]
    print(f"with the ring trimmed to 3 slots: {trimmed/1e9:.3f} GB (saves {(cur-trimmed)/1e9:.3f} GB by itself)")

    def line(label, base, new):
        print(f"  {label:70s} {new/1e9:7.3f} GB  saves {(base-new)/1e6:8.1f} MB ({(1-new/base)*100:4.1f}% of base)")

    print("-- on the CURRENT format (full ring) --")
    line("naive: zstd -1 each whole file (e files only; k files zstd1 of body)", cur, tot["e_whole_zstd1"] + tot["k_zraw"] + tot["k_other"])
    base_scheme_state_hi = tot["state_hi"]
    new_e = tot["state_hi"] + tot["ring_full_planes"] + tot["tail_scheme"] + tot["other"]
    new_k = tot["k_scheme"] + tot["k_other"]
    line("state hi-byte plane zstd1 + ring planes zstd1 + KV channel-major", cur, new_e + new_k)
    new_e2 = tot["state_hi"] + tot["ring_raw"] + tot["tail_scheme"] + tot["other"]
    line("state hi-byte only (ring raw, KV raw)", cur, new_e2 + tot["k_raw"])
    line("KV blocks only (channel-major scheme)", cur, cur - tot["k_raw"] + new_k)
    print("-- on the TRIMMED format (3-slot ring) --")
    line("naive: nothing", trimmed, trimmed)
    new_e3 = tot["state_hi"] + tot["ring3_z1"] + tot["tail_scheme"] + tot["other"]
    line("state hi-byte + ring3 planes zstd1 + KV channel-major", trimmed, new_e3 + new_k)
    new_e4 = tot["state_hi"] + tot["ring3_raw"] + tot["tail_raw"] + tot["other"]
    line("state hi-byte only", trimmed, new_e4 + tot["k_raw"])
    print("\nper-checkpoint averages (8 files): state raw 151.0 MB -> hi-byte %.2f MB; ring3 %.2f -> %.2f MB; ring full %.2f -> %.2f MB" % (
        tot["state_hi"] / 8e6, tot["ring3_raw"] / 8e6, tot["ring3_z1"] / 8e6, tot["ring_raw"] / 8e6, tot["ring_full_planes"] / 8e6))
    print("per KV block (avg of 26): raw %.2f MB -> %.2f MB scheme" % (kraw / 26e6, (kz + 26 * (56 + 4096)) / 26e6))


if __name__ == "__main__":
    main()
