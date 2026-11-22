"""Shared helpers for the disk-cache compression measurements (read-only on the cache files)."""
import os, struct, glob
import numpy as np

CACHE_DIR = r"cache/prompt"
HEAD = struct.Struct("<IIQQIIIIQQ")  # magic, version, sig, hash, pos, kind, has_h, block, rowb, snapb
assert HEAD.size == 56

N_REC, N_VH, SD = 48, 48, 128
STATE_PER_LAYER = N_VH * SD * SD * 4
STATE_BYTES = N_REC * STATE_PER_LAYER            # 150,994,944
RING_SLOTS, CONV_DIM = 32, 10240
RING_BYTES = N_REC * RING_SLOTS * CONV_DIM * 4   # 62,914,560
N_KVL = 17
KV_HEADS, HD = 4, 256
REC = HD // 2 + HD // 8                          # 160 bytes per head record
ROW_HALF = KV_HEADS * REC                        # 640: one K (or V) row of one layer


def read_head(path):
    with open(path, "rb") as f:
        h = HEAD.unpack(f.read(56))
    keys = "magic version sig hash pos kind has_h block rowb snapb".split()
    return dict(zip(keys, h))


def list_files():
    out = []
    for p in glob.glob(os.path.join(CACHE_DIR, "*.iec")):
        h = read_head(p)
        t = os.path.basename(p).split("-")[1]
        out.append(dict(path=p, type=t, size=os.path.getsize(p), **h))
    return out


def e_sections(h):
    """byte offsets of the sections of an 'e' file (h from read_head)"""
    pos, snapb, rowb, block = h["pos"], h["snapb"], h["rowb"], h["block"]
    tok = 56
    snap = tok + 4 * pos
    state = snap
    ring = snap + STATE_BYTES
    hrow = ring + RING_BYTES
    kv = snap + snapb
    tail = pos % block
    return dict(tok=tok, state=state, ring=ring, hrow=hrow, kv=kv, tail=tail, end=kv + tail * rowb)


def mm(path, off, n, dtype=np.uint8):
    return np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(n,)).view(dtype) if dtype != np.uint8 else \
        np.memmap(path, dtype=np.uint8, mode="r", offset=off, shape=(n,))


_M = (1 << 64) - 1


def _fnv64(h, data):
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & _M
    return h


def k_paths_of(e_path):
    """paths of the 'k' block files an 'e' checkpoint needs (chain of FNV-64 hashes over its tokens, as dc_chain)"""
    h = read_head(e_path)
    t = np.fromfile(e_path, dtype=np.uint32, count=h["pos"], offset=56)
    x, out = h["sig"], []
    for i in range(h["pos"] // h["block"]):
        x = _fnv64(x, t[i * h["block"]:(i + 1) * h["block"]].tobytes())
        out.append(os.path.join(CACHE_DIR, f"{h['sig']:016x}-k-{x:016x}.iec"))
    return out


def e_path_by_pos(pos, kind=None, conv_a=True):
    """the e file of conversation A (the one that reaches 26736 tokens) at this position"""
    longest = max((f for f in list_files() if f["type"] == "e"), key=lambda f: f["pos"])
    lt = np.fromfile(longest["path"], dtype=np.uint32, count=longest["pos"], offset=56)
    for f in list_files():
        if f["type"] == "e" and f["pos"] == pos and (kind is None or f["kind"] == kind):
            t = np.fromfile(f["path"], dtype=np.uint32, count=pos, offset=56)
            if not conv_a or (pos <= len(lt) and (t == lt[:pos]).all()):
                return f["path"]
    raise KeyError(pos)


# ---- entropy helpers -------------------------------------------------------
def entropy_bits(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def order0_bytes(u8):
    """order-0 entropy bound in bytes of a uint8 array (bytes)"""
    cnt = np.bincount(np.asarray(u8).reshape(-1), minlength=256)
    return entropy_bits(cnt) * cnt.sum() / 8.0, entropy_bits(cnt)


def order0_u16(u16):
    cnt = np.bincount(np.asarray(u16).reshape(-1), minlength=65536)
    return entropy_bits(cnt) * cnt.sum() / 8.0, entropy_bits(cnt)


def planes_f32(a_u32):
    """4 byte planes of a uint32 array (plane 3 = sign + 7 exponent bits, plane 2 = 1 exp bit + 7 mantissa bits, ...)"""
    a = np.asarray(a_u32)
    return [((a >> (8 * k)) & 0xFF).astype(np.uint8) for k in (3, 2, 1, 0)]  # [hi .. lo]


def f32_fields(a_u32):
    """sign, exponent (8 bit), mantissa(23 bit) split"""
    a = np.asarray(a_u32)
    return (a >> 31).astype(np.uint8), ((a >> 23) & 0xFF).astype(np.uint8), a & 0x7FFFFF
