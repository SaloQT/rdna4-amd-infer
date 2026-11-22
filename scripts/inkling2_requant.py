#!/usr/bin/env python3
"""Requantize Q8_0 tensors of a GGUF to Q4_K or Q6_K on the CPU (ggml_quantize_chunk from ref/llama-bin/ggml-base.dll).

usage: python bench/inkling2_requant.py SRC.gguf --plan REGEX=TYPE [--plan REGEX=TYPE ...]
           (--overlay DIR | --out OUT.gguf) [--dry-run] [--threads N]

--plan     REGEX is matched against a tensor's whole name; the first plan that matches decides its type
           (Q4_K or Q6_K). Only Q8_0 tensors with a multiple of 256 columns can be requantized.
--overlay  one raw file per requantized tensor and DIR/overlay.txt, the list ie-run reads from
           IE_TENSOR_OVERLAY (name<TAB>ggml type<TAB>path). SRC stays the model file.
--out      a complete GGUF: SRC's metadata and tensors with the planned ones requantized (ie-serve reads no
           overlay). Written to OUT.part and renamed; the copied tensors are compared with SRC afterwards.
Each requantized tensor's relative RMS error against its Q8_0 values is printed (decoded by the same library).

Inkling-2-Mini (dim 2304, vocab 201024): output.weight is 492 MB as Q8_0, 261 MB as Q4_K, 380 MB as Q6_K; the
dense attention projections (attn_q, attn_k, attn_v, attn_gate, attn_output) are 662 MB as Q8_0.
"""
import argparse
import ctypes
import hashlib
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(LAB, "tools"))
import gguf_lite  # noqa: E402

GGML_DIR = os.path.join(LAB, "ref", "llama-bin")
Q8_0 = 8
TYPES = {"Q4_K": (12, 144), "Q6_K": (14, 210)}  # ggml type id, bytes per block of 256 weights


class Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


def load_ggml():
    os.add_dll_directory(GGML_DIR)
    dll = ctypes.CDLL(os.path.join(GGML_DIR, "ggml-base.dll"))
    q = dll.ggml_quantize_chunk
    q.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                  ctypes.c_void_p]
    q.restype = ctypes.c_size_t
    tr = dll.ggml_get_type_traits
    tr.argtypes = [ctypes.c_int]
    tr.restype = ctypes.POINTER(Traits)
    dec = {}
    for name, (ty, bs) in TYPES.items():
        t = tr(ty).contents
        if t.blck_size != 256 or t.type_size != bs or not t.to_float:
            raise SystemExit("ggml-base.dll: %s is %d weights in %d bytes, expected 256 in %d" % (name, t.blck_size, t.type_size, bs))
        dec[ty] = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(t.to_float)
    return q, dec


def q8_rows(raw, r0, nr, cols):
    """Rows r0 .. r0 + nr of a Q8_0 tensor (blocks of an f16 scale and 32 int8) as float32 [nr, cols]."""
    nb = cols // 32
    blk = np.frombuffer(raw, np.uint8, nr * nb * 34, r0 * nb * 34).reshape(nr, nb, 34)
    d = blk[:, :, :2].copy().view(np.float16).astype(np.float32)
    return np.ascontiguousarray((d * blk[:, :, 2:].view(np.int8).astype(np.float32)).reshape(nr, cols))


def requant(q, dec, raw, rows, cols, tname, threads):
    """The tensor's bytes in the new type and (sum e^2, sum w^2) against the Q8_0 values."""
    ty, bs = TYPES[tname]
    rb = cols // 256 * bs
    out = bytearray(rows * rb)
    view = (ctypes.c_uint8 * len(out)).from_buffer(out)
    base = ctypes.addressof(view)
    step = max(1, (16 << 20) // (cols * 4))

    def part(r0):
        nr = min(step, rows - r0)
        w = q8_rows(raw, r0, nr, cols)
        n = q(ty, w.ctypes.data, base + r0 * rb, 0, nr, cols, None)
        if n != nr * rb:
            raise SystemExit("ggml_quantize_chunk wrote %d bytes, expected %d" % (n, nr * rb))
        back = np.empty_like(w)
        dec[ty](base + r0 * rb, back.ctypes.data, nr * cols)
        e = (back - w).astype(np.float64)
        return float((e * e).sum()), float((w.astype(np.float64) ** 2).sum())

    with ThreadPoolExecutor(threads) as ex:
        errs = list(ex.map(part, range(0, rows, step)))
    del view
    return bytes(out), sum(e[0] for e in errs), sum(e[1] for e in errs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--plan", action="append", required=True, metavar="REGEX=TYPE")
    ap.add_argument("--overlay", metavar="DIR")
    ap.add_argument("--out", metavar="OUT.gguf")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 4))
    a = ap.parse_args()
    if bool(a.overlay) == bool(a.out) and not a.dry_run:
        raise SystemExit("one of --overlay DIR and --out OUT.gguf")
    plan = []
    for p in a.plan:
        rx, _, tname = p.rpartition("=")
        if tname not in TYPES or not rx:
            raise SystemExit("--plan %s: REGEX=Q4_K or REGEX=Q6_K" % p)
        plan.append((re.compile(rx), tname))
    g = gguf_lite.GGUF(a.src)
    jobs = []  # (tensor, target type name or None, bytes in the source, bytes written)
    for t in g.tensors:
        n = g.size(t)
        tname = next((ty for rx, ty in plan if rx.fullmatch(t.name)), None)
        if tname is None:
            jobs.append((t, None, n, n))
            continue
        if t.ggml_type != Q8_0 or t.cols % 256:
            raise SystemExit("%s: type %d with %d columns cannot be requantized (Q8_0, a multiple of 256 columns)"
                             % (t.name, t.ggml_type, t.cols))
        jobs.append((t, tname, n, t.rows * (t.cols // 256) * TYPES[tname][1]))
    todo = [j for j in jobs if j[1]]
    print("plan: %d of %d tensors requantized, %.1f MB -> %.1f MB (file %.2f GB -> %.2f GB)" % (
        len(todo), len(jobs), sum(j[2] for j in todo) / 1e6, sum(j[3] for j in todo) / 1e6,
        sum(j[2] for j in jobs) / 1e9, sum(j[3] for j in jobs) / 1e9))
    if not todo:
        raise SystemExit("no tensor matches the plan")
    if a.dry_run:
        for t, tname, n, m in todo:
            print("  %-36s %s x %-7d Q8_0 %7.1f MB -> %s %7.1f MB" % (t.name, t.cols, t.rows, n / 1e6, tname, m / 1e6))
        return
    q, dec = load_ggml()
    t0 = time.time()
    wr = part = None
    lines = []
    if a.out:
        part = a.out + ".part"
        blob = gguf_lite.build_kv_blob(gguf_lite.kv_entries_raw(a.src))
        wr = gguf_lite.StreamWriter(part, blob, [(t.name, t.dims, TYPES[tn][0] if tn else t.ggml_type, m)
                                                 for t, tn, _, m in jobs], version=g.version, align=g.align)
    else:
        os.makedirs(a.overlay, exist_ok=True)
    kept = {}
    for t, tname, n, m in jobs:
        if tname is None:
            if wr:
                b = g.read(t, n)
                kept[t.name] = hashlib.sha256(b).hexdigest()
                wr.put(b)
            continue
        raw = g.read(t, t.rows * (t.cols // 32) * 34)
        tq = time.time()
        b, e2, w2 = requant(q, dec, raw, t.rows, t.cols, tname, a.threads)
        print("  %-36s %s x %-7d %7.1f MB -> %s %7.1f MB  relRMSE %.5f  %.1f s" % (
            t.name, t.cols, t.rows, n / 1e6, tname, len(b) / 1e6, (e2 / max(w2, 1e-30)) ** 0.5, time.time() - tq), flush=True)
        if wr:
            wr.put(b + b"\0" * (m - len(b)))
        else:
            p = os.path.abspath(os.path.join(a.overlay, "%s.%s.bin" % (t.name, tname)))
            with open(p, "wb") as f:
                f.write(b)
            lines.append("%s\t%d\t%s\n" % (t.name, TYPES[tname][0], p))
    if wr:
        wr.close()
        os.replace(part, a.out)
        o = gguf_lite.GGUF(a.out)
        if [x.name for x in o.tensors] != [j[0].name for j in jobs]:
            raise SystemExit("%s: tensor table differs from the source" % a.out)
        for (t, tname, _, m), x in zip(jobs, o.tensors):
            if x.dims != t.dims or o.size(x) < m:
                raise SystemExit("%s: %s has dims %r, %d bytes" % (a.out, x.name, x.dims, o.size(x)))
            if tname is None and hashlib.sha256(o.read(x, m)).hexdigest() != kept[t.name]:
                raise SystemExit("%s: %s differs from the source" % (a.out, x.name))
        o.close()
        print("wrote %s (%.2f GB), copied tensors equal to the source, in %.0f s" % (a.out, os.path.getsize(a.out) / 1e9, time.time() - t0))
    else:
        mp = os.path.join(a.overlay, "overlay.txt")
        with open(mp, "w", newline="\n") as f:
            f.writelines(lines)
        print("wrote %s (%d tensors) in %.0f s; run ie-run with IE_TENSOR_OVERLAY=%s" % (mp, len(lines), time.time() - t0, os.path.abspath(mp)))
    g.close()


if __name__ == "__main__":
    main()
