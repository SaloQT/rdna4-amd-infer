"""attn_pf_bench.py -- standalone check and timing of the prompt-chunk attention kernels (ie_attn_fa2 and the
long-position kernels) on random KV caches. Always run through ref\\gpu-run.ps1:

  pwsh -NoProfile -File ..\\gpu-run.ps1 -TimeoutSec 300 -- <python.exe> tools\\attn_pf_bench.py [options]

  --hsaco FILE   kernel object (default build/ie_kernels.hsaco; the q8_0 KV build: build/ie_kernels_q8.hsaco, --q8)
  --pos LIST     position of the first token of the chunk (keys 0 .. pos + T - 1 exist), default 32768
  --T N          tokens in the chunk, default 256
  --kernels LIST fa2 (old), pl (the long-position kernel), default fa2
  --check        compare each kernel with a float64 reference on a few tokens (all heads) and with fa2
  --time         time the kernels (graph replay of several launches)
  --dist         normal | peaked (scores with a few dominant keys)

Every buffer is allocated with padding (NaN pattern); the byte range each kernel can touch is computed here and
checked against the allocation before the launch (a wrong stride would otherwise fault the GPU).
"""
import argparse
import ctypes as C
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

from attn_long_bench import Args, Buf, H, Module, ck, launch, need, quant_q8, relerr, sync  # noqa: E402


def make_kv_big(rng, nrows, n_kv, hd, q8, dist):
    """K, V device-format arrays [nrows * n_kv, ...] built in slices (limits the float32 temporaries) and the
    dequantized float32 copies: [nrows, n_kv, hd]."""
    outs = []
    for _ in range(2):
        dev, deq = [], []
        sl = 8192
        for r0 in range(0, nrows, sl):
            n = min(sl, nrows - r0) * n_kv
            x = rng.standard_normal((n, hd)).astype(np.float32)
            if dist == "peaked":
                x *= np.exp(rng.standard_normal((n, 1)).astype(np.float32) * 0.7)
                x[:, :4] *= 8.0
            if q8:
                rec, d = quant_q8(x)
                dev.append(rec)
                deq.append(d.astype(np.float32))
            else:
                h = x.astype(np.float16)
                dev.append(h)
                deq.append(h.astype(np.float32))
        outs.append((np.concatenate(dev), np.concatenate(deq).reshape(nrows, n_kv, hd)))
    return outs


def ref_tokens(q, K, V, pos, toks, n_head, n_kv, hd, gate, gstride):
    """float64 attention of the chunk tokens in toks: [len(toks), n_head, hd]."""
    grp = n_head // n_kv
    out = np.zeros((len(toks), n_head, hd))
    for a, j in enumerate(toks):
        pj = pos + j
        for h in range(n_head):
            kh = h // grp
            s = K[: pj + 1, kh, :].astype(np.float64) @ q[j, h].astype(np.float64) / np.sqrt(hd)
            e = np.exp(s - s.max())
            o = (e @ V[: pj + 1, kh, :].astype(np.float64)) / e.sum()
            if gate is not None:
                o = o / (1.0 + np.exp(-gate[j, h * gstride : h * gstride + hd].astype(np.float64)))
            out[a, h] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hsaco", default=os.path.join(HERE, "..", "build", "ie_kernels.hsaco"))
    ap.add_argument("--q8", action="store_true")
    ap.add_argument("--pos", default="32768", help="comma list of positions")
    ap.add_argument("--T", default="256", help="comma list of chunk sizes")
    ap.add_argument("--nkv", type=int, default=4)
    ap.add_argument("--grp", type=int, default=6)
    ap.add_argument("--hd", type=int, default=256)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--nogate", action="store_true", help="no output gate")
    ap.add_argument("--dist", default="normal", choices=["normal", "peaked"])
    ap.add_argument("--kernels", default="fa2")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--nsplit", type=int, default=0, help="pl: key splits per tile (0: the kernel's rule)")
    ap.add_argument("--cold", type=int, default=0, help="MB written before every timed call (caches and TLB cold, as after the GEMMs); its own time is subtracted")
    a = ap.parse_args()
    if a.hd != 256:
        sys.exit("hd 256 only")
    ck(H.hipInit(0), "hipInit")
    mod = Module(a.hsaco)
    for pos in [int(x) for x in a.pos.split(",")]:
        for T in [int(x) for x in a.T.split(",")]:
            run_case(a, mod, pos, T)
            gc.collect()
    print("done")


def run_case(a, mod, pos, T):
    n_kv, grp, hd = a.nkv, a.grp, a.hd
    n_head = n_kv * grp
    kend = pos + T
    if T < 1 or T > 512 or grp > 6:
        sys.exit("T 1..512, grp <= 6")
    rng = np.random.default_rng(a.seed)
    kernels = a.kernels.split(",")
    krow = (hd + hd // 16) if a.q8 else hd * 2  # bytes of a (position, head) record
    kvbytes = lambda rows: rows * n_kv * krow
    (Kd, Kf), (Vd, Vf) = make_kv_big(rng, kend, n_kv, hd, a.q8, a.dist)
    kc, vc = Buf(kvbytes(kend)), Buf(kvbytes(kend))
    kc.put(Kd)
    vc.put(Vd)
    del Kd, Vd
    # q: the strides of the graph's token rows (bytes, 256-aligned); gate in the same row layout as the host's (offset hd floats)
    qst = (n_head * hd * 4 + 255) // 256 * 256
    ost = qst
    gst = (n_head * 2 * hd * 4 + 255) // 256 * 256
    gstride = 2 * hd
    gtoff = hd * 4
    q = (rng.standard_normal((T, n_head, hd)) * (3.0 if a.dist == "peaked" else 1.0)).astype(np.float32)
    qb = Buf(T * qst)
    qpad = np.zeros((T, qst // 4), dtype=np.float32)
    qpad[:, : n_head * hd] = q.reshape(T, -1)
    qb.put(qpad)
    gb = Buf(T * gst)
    gdata = np.zeros((T, gst // 4), dtype=np.float32)
    gdata[:, : n_head * 2 * hd] = rng.standard_normal((T, n_head * 2 * hd)).astype(np.float32)
    gb.put(gdata)
    gate_arg = None if a.nogate else (gb, gtoff)
    gate_ref = None if a.nogate else gdata[:, hd:]
    if not a.nogate:
        need(gb, (T - 1) * gst + gtoff + ((n_head - 1) * gstride + hd) * 4, "gate")
    need(qb, (T - 1) * qst + n_head * hd * 4, "q")
    ntile = (T + 15) // 16
    obs = {k: Buf(T * ost) for k in kernels}
    for k in kernels:
        need(obs[k], (T - 1) * ost + n_head * hd * 4, "out " + k)
    need(kc, kvbytes(kend), "K")
    need(vc, kvbytes(kend), "V")

    # ---- the launchers (same argument order as the host's a1[] in src/hip.c)
    def fa2_args(out, kcb, vcb):
        return Args(qb, kcb, vcb, out, pos, hd, n_head, n_kv, gate_arg, gstride, qst, ost, gst, T)

    parts = {}

    def run(name, out, kcb, vcb, stream=None):
        if name == "fa2":
            launch(mod, "ie_attn_fa2", n_kv * ntile, 1, fa2_args(out, kcb, vcb), stream)
        elif name == "pl":  # the same arguments and grid
            launch(mod, "ie_attn_pl", n_kv * ntile, 1, fa2_args(out, kcb, vcb), stream)
        else:
            sys.exit("unknown kernel " + name)

    def fetch(name):
        return obs[name].get(np.float32, T * ost // 4).reshape(T, ost // 4)[:, : n_head * hd].reshape(T, n_head, hd)

    if a.check:
        toks = sorted({0, 1, min(15, T - 1), min(16, T - 1), T // 2, T - 1})
        print("check: pos %d T %d (keys %d), %s KV, dist %s, gate %d" % (pos, T, kend, "q8" if a.q8 else "f16", a.dist,
                                                                        not a.nogate))
        ref = ref_tokens(q, Kf, Vf, pos, toks, n_head, n_kv, hd, gate_ref, gstride)
        outs = {}
        for k in kernels:
            obs[k].put(np.full(T * ost // 4, np.nan, dtype=np.float32))
            run(k, obs[k], kc, vc)
            sync()
            outs[k] = fetch(k)
            fin = bool(np.isfinite(outs[k]).all())
            print("  %-5s vs float64 reference (tokens %s): rel %.3e   finite %s" % (k, toks, relerr(outs[k][toks], ref), fin))
        if "fa2" in outs:
            for k in kernels:
                if k != "fa2":
                    print("  %-5s vs fa2 (all tokens): rel %.3e" % (k, relerr(outs[k], outs["fa2"])))

    if a.time:
        bytes_kv = 2 * kend * n_kv * krow
        print("time: pos %d T %d (keys %d), %s KV %.1f MB" % (pos, T, kend, "q8" if a.q8 else "f16", bytes_kv / 1e6))
        stream = C.c_void_p()
        ck(H.hipStreamCreate(C.byref(stream)), "hipStreamCreate")
        e0, e1 = C.c_void_p(), C.c_void_p()
        ck(H.hipEventCreate(C.byref(e0)), "ev")
        ck(H.hipEventCreate(C.byref(e1)), "ev")
        scratch = Buf(a.cold << 20) if a.cold else None

        def timed(fn, reps):
            g, ge = C.c_void_p(), C.c_void_p()
            ck(H.hipStreamBeginCapture(stream, 2), "capture")
            for _ in range(reps):
                fn()
            ck(H.hipStreamEndCapture(stream, C.byref(g)), "end capture")
            ck(H.hipGraphInstantiate(C.byref(ge), g, None, None, 0), "instantiate")
            ck(H.hipGraphLaunch(ge, stream), "graph launch")
            ck(H.hipStreamSynchronize(stream), "sync")
            best = 1e30
            for _ in range(3):
                ck(H.hipEventRecord(e0, stream), "rec")
                ck(H.hipGraphLaunch(ge, stream), "graph launch")
                ck(H.hipEventRecord(e1, stream), "rec")
                ck(H.hipEventSynchronize(e1), "evsync")
                ms = C.c_float()
                ck(H.hipEventElapsedTime(C.byref(ms), e0, e1), "elapsed")
                best = min(best, ms.value / reps)
            return best

        def wipe():
            ck(H.hipMemsetAsync(scratch.ptr, 0x5A, C.c_size_t(scratch.size), stream), "memset")

        for k in kernels:
            reps = a.reps
            run(k, obs[k], kc, vc)
            sync()
            if scratch:
                tw = timed(wipe, reps)
                best = timed(lambda: (wipe(), run(k, obs[k], kc, vc, stream)), reps) - tw
            else:
                best = timed(lambda: run(k, obs[k], kc, vc, stream), reps)
            flops = 4.0 * T * n_head * hd * (pos + T / 2.0)  # QK and PV, causal
            print("  %-5s %9.1f us   %6.1f GB/s of KV (once)   %6.1f TFLOPS" % (k, best * 1000, bytes_kv / best / 1e6,
                                                                              flops / best / 1e9))


if __name__ == "__main__":
    main()
