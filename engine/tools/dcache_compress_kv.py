"""Lossless compressibility of 4-bit KV blocks (the 'k' files; the same rows sit at the end of 'e' files).

usage: python dcache_compress_kv.py FILE_k.iec [more files ...]
Layout (kernels/ie_kernels.c IE_KV_Q4, src/hip.c gpu_kv_io): per layer (17 incl. MTP): K rows of the block, then V rows;
a row = 4 heads x 160 B; a head record = 128 B of nibbles (value i in byte i/2, even i = low nibble) + 8 pairs fp16 (d, m)
for the 8 blocks of 32 values.   value = q * d + m.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import zstandard as zstd
import lz4.block
import dcache_compress_common as C

BLOCK = 1024


def z(a, level):
    return len(zstd.ZstdCompressor(level=level).compress(np.ascontiguousarray(a).tobytes()))


def l4(a):
    return len(lz4.block.compress(np.ascontiguousarray(a).tobytes(), store_size=False))


def H(cnt):
    return C.entropy_bits(cnt)


def load(path):
    raw = np.fromfile(path, dtype=np.uint8, offset=56 + 4 * BLOCK)
    assert raw.size == C.N_KVL * 2 * BLOCK * C.ROW_HALF, raw.size
    return raw.reshape(C.N_KVL, 2, BLOCK, C.KV_HEADS, C.REC)  # layer, K/V, pos, head, byte


def nibbles(rec):
    nib = rec[..., :128]
    out = np.empty(nib.shape[:-1] + (128, 2), dtype=np.uint8)
    out[..., 0] = nib & 15
    out[..., 1] = nib >> 4
    return out.reshape(nib.shape[:-1] + (256,))  # value index i


def analyze(path):
    a = load(path)
    total = a.size
    print(f"\n=== {os.path.basename(path)}  KV body {total/1e6:.2f} MB ({C.N_KVL} layers x K/V x {BLOCK} pos x 4 heads x 160 B)")
    nib = nibbles(a)  # (17,2,1024,4,256)
    sc = np.ascontiguousarray(a[..., 128:]).view(np.uint16)  # (17,2,1024,4,16)
    d_, m_ = sc[..., 0::2], sc[..., 1::2]
    print(f"bytes: nibbles {nib.size//2/1e6:.2f} MB (80%), scales {sc.size*2/1e6:.2f} MB (20%)")

    # --- nibble entropies, K vs V
    print("nibble entropy (bits of 4):  order-0 | given neighbour value i-1 | given channel i (256 ctx, per layer & K/V)")
    nb_o0 = {0: 0.0, 1: 0.0}
    nb_o1 = {0: 0.0, 1: 0.0}
    nb_ch = {0: 0.0, 1: 0.0}
    nb_chh = {0: 0.0, 1: 0.0}
    nsym = nib.shape[2] * nib.shape[3] * 256
    per_layer = []
    for kv in (0, 1):
        for l in range(C.N_KVL):
            x = nib[l, kv]  # (1024,4,256)
            h0 = H(np.bincount(x.reshape(-1), minlength=16))
            j1 = x[..., 1:].astype(np.uint16) * 16 + x[..., :-1]
            h1 = H(np.bincount(j1.reshape(-1), minlength=256)) - H(np.bincount(x[..., :-1].reshape(-1), minlength=16))
            ch = (np.arange(256, dtype=np.uint16)[None, None, :] * 16 + x).reshape(-1)
            hch = H(np.bincount(ch, minlength=4096)) - H(np.bincount(np.broadcast_to(np.arange(256), x.shape).reshape(-1), minlength=256))
            cid = (np.arange(4, dtype=np.uint32)[None, :, None] * 256 + np.arange(256, dtype=np.uint32)[None, None, :])  # head, channel
            chh = (cid * 16 + x).reshape(-1)
            hchh = H(np.bincount(chh, minlength=16384)) - H(np.bincount(np.broadcast_to(cid, x.shape).reshape(-1), minlength=1024))
            nb_o0[kv] += h0 / C.N_KVL
            nb_o1[kv] += h1 / C.N_KVL
            nb_ch[kv] += hch / C.N_KVL
            nb_chh[kv] += hchh / C.N_KVL
            per_layer.append((kv, l, h0, h1, hch, hchh))
    for kv, nm in ((0, "K"), (1, "V")):
        print(f"  {nm}: order-0 {nb_o0[kv]:.3f} | neighbour {nb_o1[kv]:.3f} | channel {nb_ch[kv]:.3f} | head+channel {nb_chh[kv]:.3f}")
    print("  per layer (K order-0 / K channel-ctx / V order-0 / V channel-ctx):")
    pl = {(kv, l): (h0, hch) for kv, l, h0, h1, hch, hchh in per_layer}
    print("   " + " ".join(f"L{l}:{pl[(0,l)][0]:.2f}/{pl[(0,l)][1]:.2f}/{pl[(1,l)][0]:.2f}/{pl[(1,l)][1]:.2f}" for l in range(C.N_KVL)))

    # --- scales
    print("fp16 scale entropy (bits of 16, per layer averaged):")
    res = {}
    for nm, arr in (("d (step)", d_), ("m (min)", m_)):
        h16 = h_hi = h_lo = h_ctx = 0.0
        for kv in (0, 1):
            for l in range(C.N_KVL):
                x = arr[l, kv].reshape(-1, C.KV_HEADS, 8)  # (pos, head, blk)
                h16 += H(np.bincount(x.reshape(-1), minlength=65536)) / (2 * C.N_KVL)
                h_hi += H(np.bincount((x >> 8).reshape(-1), minlength=256)) / (2 * C.N_KVL)
                h_lo += H(np.bincount((x & 255).reshape(-1), minlength=256)) / (2 * C.N_KVL)
                # hi byte given (head, block j): 32 contexts
                ctx = (np.arange(32).reshape(1, 4, 8) * 256 + (x >> 8)).reshape(-1)
                cc = np.bincount(ctx, minlength=32 * 256)
                h_ctx += (H(cc) - H(cc.reshape(32, 256).sum(1))) / (2 * C.N_KVL)
        res[nm] = (h16, h_hi, h_lo, h_ctx)
        print(f"  {nm}: whole u16 {h16:.2f} | hi byte {h_hi:.2f} | lo byte {h_lo:.2f} | hi byte given (head,blk) {h_ctx:.2f}")

    # --- entropy-bound totals
    nib_bytes_o0 = (nb_o0[0] + nb_o0[1]) / 2 * nib.size / 8
    nib_bytes_ch = (nb_ch[0] + nb_ch[1]) / 2 * nib.size / 8
    sc_n = sc.size / 2  # d count == m count
    sc_hi_lo = (res["d (step)"][1] + res["d (step)"][2]) / 2 + (res["m (min)"][1] + res["m (min)"][2]) / 2
    sc_bytes_split = sc_n * (res["d (step)"][1] + res["d (step)"][2] + res["m (min)"][1] + res["m (min)"][2]) / 8
    sc_bytes_ctx = sc_n * (res["d (step)"][3] + res["d (step)"][2] + res["m (min)"][3] + res["m (min)"][2]) / 8
    print("entropy bounds (MB):")
    print(f"  nibbles order-0                {nib_bytes_o0/1e6:8.2f}  (raw {nib.size/2/1e6:.2f})")
    print(f"  nibbles channel ctx            {nib_bytes_ch/1e6:8.2f}")
    print(f"  scales hi+lo byte order-0      {sc_bytes_split/1e6:8.2f}  (raw {sc.size*2/1e6:.2f})")
    print(f"  scales hi byte|(head,blk)+lo   {sc_bytes_ctx/1e6:8.2f}")
    b0 = nib_bytes_o0 + sc_bytes_split
    b1 = nib_bytes_ch + sc_bytes_ctx
    print(f"  total order-0   {b0/1e6:8.2f}  ratio x{total/b0:.3f} saves {(1-b0/total)*100:.1f}%")
    print(f"  total ctx       {b1/1e6:8.2f}  ratio x{total/b1:.3f} saves {(1-b1/total)*100:.1f}%")

    # --- codec sizes
    flat = a.reshape(-1)
    print("codec sizes (MB):")
    rows = []

    def add(label, size):
        rows.append((label, size))
        print(f"  {label:60s} {size/1e6:8.2f}  x{total/size:6.3f} saves {(1-size/total)*100:5.1f}%")

    add("raw zstd -1", z(flat, 1))
    add("raw zstd -3", z(flat, 3))
    add("raw lz4", l4(flat))
    nbytes = np.ascontiguousarray(a[..., :128])
    sbytes = np.ascontiguousarray(a[..., 128:])
    scbytes_planes = np.ascontiguousarray(sbytes.reshape(-1, 2).T).reshape(-1)  # all lo bytes, then all hi bytes
    add("streams split (nibbles | scales), zstd -1 each", z(nbytes, 1) + z(sbytes, 1))
    add("streams split, zstd -3 each", z(nbytes, 3) + z(sbytes, 3))
    add("nibbles raw + scales zstd -3", nbytes.size + z(sbytes, 3))
    add("nibbles zstd -3 + scales byte-planes (lo|hi) zstd -3", z(nbytes, 3) + z(scbytes_planes, 3))
    add("nibbles zstd -3 + scales byte-planes zstd -1", z(nbytes, 3) + z(scbytes_planes, 1))
    # channel-major nibble stream (per layer, K/V, head: channel x position), re-packed two per byte
    cm = np.ascontiguousarray(nib.transpose(0, 1, 3, 4, 2))  # l,kv,head,chan,pos
    cm = cm.reshape(-1, 2)
    packed = (cm[:, 0] | (cm[:, 1] << 4)).astype(np.uint8)
    add("nibbles channel-major (pos fastest) zstd -3 + scales planes zstd -3", z(packed, 3) + z(scbytes_planes, 3))
    # scales d/m planes ordered pos-fastest per (head, blk)
    sct = np.ascontiguousarray(sc.reshape(C.N_KVL, 2, BLOCK, C.KV_HEADS, 8, 2).transpose(0, 1, 3, 4, 5, 2))  # ...,(d/m),pos
    sct8 = sct.view(np.uint8).reshape(-1, 2).T.reshape(-1)  # hmm: lo/hi planes of a pos-fastest ordering
    add("nibbles ch-major zstd -3 + scales pos-fastest planes zstd -3", z(packed, 3) + z(sct8, 3))
    add("nibbles channel-major zstd -1 + scales planes zstd -1", z(packed, 1) + z(scbytes_planes, 1))
    add("nibbles channel-major lz4 + scales planes lz4", l4(packed) + l4(scbytes_planes))
    add("bound: nibble head+channel ctx (in-sample) + scales", (nb_chh[0] + nb_chh[1]) / 2 * nib.size / 8 + sc_bytes_ctx)
    add("bound: nibble channel ctx + scale hi|ctx + lo (arith/ANS ideal)", b1)
    add("bound: nibble order-0 + scale hi/lo order-0 (Huffman-class ideal)", b0)
    return rows, b0, b1


if __name__ == "__main__":
    allrows = []
    for p in sys.argv[1:]:
        allrows.append(analyze(p))
