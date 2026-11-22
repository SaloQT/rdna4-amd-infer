#!/usr/bin/env python3
"""slice_iqraw.py -- real raw IQ/K blocks out of a GGUF, for tests/iqraw.

For each of the seven raw block types the engine reads through
kernels/ie_iqraw.inc (Q2_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ2_S, IQ1_M)
write N blocks sampled evenly over *all* blocks of that type in the file (so a
sample spans many tensors and rows), plus one manifest line per type:

    <out>/iqraw_<type>.bin      N blocks, raw GGUF bytes, back to back
    <out>/iqraw_manifest.txt    "type block_bytes n_blocks file # tensor@block ..."

Usage: python slice_iqraw.py [--model M.gguf] [--out DIR] [--n 8]

The default model is the multimodel under test,
models/Underdog-Saluki-27B-1.0-IQ2-mix.gguf (relative to the repo root, four
directories above this script).
"""

import argparse
import os
import struct
import sys

# ggml types handled by kernels/ie_iqraw.inc and their 256-weight block size
RAW_TYPES = {
    10: ("Q2_K", 84),
    16: ("IQ2_XXS", 66),
    17: ("IQ2_XS", 74),
    18: ("IQ3_XXS", 98),
    21: ("IQ3_S", 110),
    19: ("IQ1_S", 50),
    22: ("IQ2_S", 82),
    29: ("IQ1_M", 56),
}

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL = range(8)
T_STR, T_ARR, T_U64, T_I64, T_F64 = 8, 9, 10, 11, 12
SCALAR_SIZE = {T_U8: 1, T_I8: 1, T_U16: 2, T_I16: 2, T_U32: 4, T_I32: 4,
               T_F32: 4, T_BOOL: 1, T_U64: 8, T_I64: 8, T_F64: 8}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DEFAULT_MODEL = os.path.join(ROOT, "models", "Underdog-Saluki-27B-1.0-IQ2-mix.gguf")


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise SystemExit("truncated GGUF: wanted %d bytes, got %d" % (n, len(b)))
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def skip(self, n):
        if n:
            self.raw(n)


def parse_gguf(path):
    """-> (tensors [(name, dims, type, offset)], data_offset, version)"""
    with open(path, "rb") as f:
        r = Reader(f)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise SystemExit("%s: not a GGUF (magic %r)" % (path, magic))
        version = r.u32()
        n_tensors, n_kv = r.u64(), r.u64()
        align = 32
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            if vtype == T_STR:
                r.string()
            elif vtype == T_ARR:
                etype, n = r.u32(), r.u64()
                if etype == T_STR:
                    for _ in range(n):
                        r.string()
                else:
                    r.skip(SCALAR_SIZE[etype] * n)
            elif key == "general.alignment" and vtype == T_U32:
                align = struct.unpack("<I", r.raw(4))[0]
            else:
                r.skip(SCALAR_SIZE[vtype])
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()  # relative to the data section
            tensors.append((name, dims, ttype, offset))
        pos = f.tell()
        data = (pos + align - 1) // align * align
    return tensors, data, version


def sample(f, path, tensors, data, n_per_type):
    """-> {type: [(tensor name, global block, bytes)]} evenly over all blocks of
    that type (the sample thereby spans many tensors and rows)."""
    by_type = {}
    for name, dims, ttype, offset in tensors:
        if ttype not in RAW_TYPES:
            continue
        bs = RAW_TYPES[ttype][1]
        cols = dims[0]
        if cols % 256:
            raise SystemExit("%s: %s has %d columns, not a multiple of 256" % (path, name, cols))
        rows = 1
        for d in dims[1:]:
            rows *= d
        per_row = cols // 256
        by_type.setdefault(ttype, []).append((name, offset, rows * per_row, per_row, bs))

    out = {}
    for ttype, entries in sorted(by_type.items()):
        total = sum(e[2] for e in entries)
        picks = []
        for k in range(n_per_type):
            g = (k * total) // n_per_type  # evenly spaced over every block of this type
            for name, offset, nblk, per_row, bs in entries:
                if g < nblk:
                    row, blk = divmod(g, per_row)
                    picks.append((name, g, data + offset + g * bs, bs))
                    break
                g -= nblk
            else:
                raise SystemExit("internal: block %d out of range for type %d" % (g, ttype))
        blocks = []
        for name, g, pos, bs in picks:
            f.seek(pos)
            blocks.append((name, g, f.read(bs)))
        out[ttype] = blocks
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=os.path.join(HERE, "build"))
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit("no such model: %s" % args.model)
    os.makedirs(args.out, exist_ok=True)

    tensors, data, version = parse_gguf(args.model)
    print("%s: GGUF v%d, %d tensors, data at %d" % (os.path.basename(args.model), version, len(tensors), data))
    counts = {}
    for name, dims, ttype, offset in tensors:
        counts[ttype] = counts.get(ttype, 0) + 1
    print("raw block types present: " + ", ".join(
        "%s(%d): %d tensors" % (RAW_TYPES[t][0], t, counts[t]) for t in sorted(RAW_TYPES) if t in counts))

    with open(args.model, "rb") as f:
        slices = sample(f, args.model, tensors, data, args.n)

    lines = []
    for ttype in sorted(RAW_TYPES):
        if ttype not in slices:
            print("WARNING: no tensor of type %s(%d) in the model" % (RAW_TYPES[ttype][0], ttype))
            continue
        blocks = slices[ttype]
        bs = RAW_TYPES[ttype][1]
        fn = "iqraw_%d.bin" % ttype
        with open(os.path.join(args.out, fn), "wb") as o:
            for _, _, b in blocks:
                assert len(b) == bs, (ttype, len(b), bs)
                o.write(b)
        src = ",".join("%s@%d" % (name, g) for name, g, _ in blocks)
        lines.append("%d %d %d %s # %s" % (ttype, bs, len(blocks), fn, src))
        print("%-8s %2d B blocks: %d" % (RAW_TYPES[ttype][0], bs, len(blocks)))

    with open(os.path.join(args.out, "iqraw_manifest.txt"), "w") as o:
        o.write("\n".join(lines) + "\n")
    print("wrote %d slice files + iqraw_manifest.txt to %s" % (len(lines), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
