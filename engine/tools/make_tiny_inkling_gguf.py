#!/usr/bin/env python3
"""Write a synthetic tiny inkling2 (inkling2_loop) GGUF v3 file (numpy only).

Default geometry exercises every architectural path of Inkling-2-Mini-Uncensored
in milliseconds:
  - 8 physical layers -> 16 logical visits (core [2, 6) x 3 passes, block_size=2)
  - 256 hidden dim, 6 Q heads / 2 KV heads of dim 64 (qd=384 != dim, kvd=128)
  - Full (global) attention at physical layers 1, 3, 5, 7 with dynamic sconv & log tau
  - SWA attention (win=6) at physical layers 0, 2, 4, 6 with static sconv
  - Relative-position bias (d_rel=16, rel_extent=16), Q/R L2 clips, sinks, attn gate
  - 8 MoE experts, top-2 routing, inter=128 (cols % 32 == 0, NOT % 256!)
    * ffn_gate_up_exps: IQ3_XXS (type 18, cols=256, de-interleaved [gate; up])
    * ffn_down_exps:    IQ4_NL  (type 20, cols=128) by default (or q4_0 / q8_0)
  - vocab_size=320 padded vs vocab_size_unpadded=300 (eos=286), pre='inkling2'

usage: make_tiny_inkling_gguf.py [OUT_PATH_OR_DIR] [--w2-type iq4_nl|q4_0|q8_0] [--seed N]
"""
import argparse
import os
import struct
import sys

import numpy as np

GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0, GGML_IQ3_XXS, GGML_IQ4_NL = 0, 1, 2, 8, 18, 20
T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR = 4, 5, 6, 7, 8, 9
ALIGN = 32


def s_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def s_val(t, v):
    if t == T_U32:
        return struct.pack("<I", v)
    if t == T_I32:
        return struct.pack("<i", v)
    if t == T_F32:
        return struct.pack("<f", v)
    if t == T_BOOL:
        return struct.pack("<B", 1 if v else 0)
    if t == T_STR:
        return s_str(v)
    raise ValueError(t)


def write_gguf(path, kvs, tensors):
    """Write GGUF v3 file with 32-byte tensor data alignment."""
    out = bytearray()
    out += b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for k, t, v in kvs:
        out += s_str(k) + struct.pack("<I", t)
        if t == T_ARR:
            et, items = v
            out += struct.pack("<IQ", et, len(items))
            for it in items:
                out += s_val(et, it)
        else:
            out += s_val(t, v)
    off = 0
    offs = []
    for name, ne, typ, raw in tensors:
        offs.append(off)
        out += s_str(name) + struct.pack("<I", len(ne))
        for d in ne:
            out += struct.pack("<Q", d)
        out += struct.pack("<IQ", typ, off)
        off += (len(raw) + ALIGN - 1) // ALIGN * ALIGN
    out += b"\0" * ((-len(out)) % ALIGN)
    for (_, _, _, raw), _ in zip(tensors, offs):
        out += raw + b"\0" * ((-len(raw)) % ALIGN)
    with open(path, "wb") as f:
        f.write(out)


def f32(a):
    return np.asarray(a, dtype=np.float32).tobytes()


def q4_0(rng, rows, cols, scale):
    """Random Q4_0 matrix (cols % 32 == 0)."""
    assert cols % 32 == 0
    nb = cols // 32
    d = (rng.uniform(0.5, 1.5, size=(rows, nb)) * scale / 4.0).astype(np.float16)
    q = rng.integers(0, 16, size=(rows, nb, 32), dtype=np.uint8)
    blk = np.zeros((rows, nb, 18), dtype=np.uint8)
    blk[:, :, 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
    blk[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)
    return blk.tobytes()


def iq4_nl(rng, rows, cols, scale):
    """Random IQ4_NL matrix (type=20, 18 B/32 weights, cols % 32 == 0)."""
    assert cols % 32 == 0
    nb = cols // 32
    d = (rng.uniform(0.5, 1.5, size=(rows, nb)) * (scale / 55.0)).astype(np.float16)
    q = rng.integers(0, 16, size=(rows, nb, 32), dtype=np.uint8)
    blk = np.zeros((rows, nb, 18), dtype=np.uint8)
    blk[:, :, 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
    blk[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)
    return blk.tobytes()


def q8_0(rng, rows, cols, scale):
    """Random Q8_0 matrix (cols % 32 == 0)."""
    assert cols % 32 == 0
    nb = cols // 32
    d = (rng.uniform(0.5, 1.5, size=(rows, nb)) * scale / 64.0).astype(np.float16)
    q = rng.integers(-127, 128, size=(rows, nb, 32)).astype(np.int8)
    blk = np.zeros((rows, nb, 34), dtype=np.uint8)
    blk[:, :, 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
    blk[:, :, 2:] = q.view(np.uint8)
    return blk.tobytes()


def iq3_xxs(rng, rows, cols, scale):
    """Random IQ3_XXS matrix (type=18, 98 B/256 weights, cols % 256 == 0)."""
    assert cols % 256 == 0
    nsb = (rows * cols) // 256
    blk = np.zeros((nsb, 98), dtype=np.uint8)
    d = (rng.uniform(0.5, 1.5, size=nsb) * (scale / 75.0)).astype(np.float16)
    blk[:, 0:2] = d.view(np.uint8).reshape(nsb, 2)
    blk[:, 2:66] = rng.integers(0, 256, size=(nsb, 64), dtype=np.uint8)
    # 8 subblock u32 aux words: low 28 bits = 4x 7-bit ksigns indices, top 4 bits = subblock scale 0..15
    aux = rng.integers(0, 1 << 32, size=(nsb, 8), dtype=np.uint32)
    blk[:, 66:98] = aux.view(np.uint8)
    return blk.tobytes()


def gpt2_bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    k = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + k)
            k += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def inkling_tiny_vocab(n_unpadded):
    """Build unpadded token table (256 byte tokens + common tokens + Inkling specials)."""
    b2u = gpt2_bytes_to_unicode()
    toks = [b2u[b] for b in range(256)]
    ttype = [1] * 256
    specials = [
        "<|unused|>",
        "<|endoftext|>",                 # pad
        "<|message_user|>",
        "<|message_model|>",
        "<|message_system|>",
        "<|message_tool|>",
        "<|content_text|>",
        "<|content_model_end_sampling|>",  # eos / eot
        "<|content_thinking|>",
        "<|end_message|>",
        "<|content_tool_error|>",
        "<|content_xml|>",
        "<|begin_of_text|>",
        "<|content_invoke_tool_json|>",
        "<|unused_200054|>",
        "<|content_invoke_tool_text|>",
    ]
    n_words = n_unpadded - len(specials)
    assert n_words >= 256
    words = [" the", " and", "ing", " of", "er", " is", " hello", " world", "\n\n", " AMD"]
    i = 0
    while len(toks) < n_words:
        w = words[i % len(words)] + ("" if i < len(words) else str(i))
        toks.append("".join(b2u[b] for b in w.encode("utf-8")))
        ttype.append(1)
        i += 1
    base_sp = len(toks)
    toks.extend(specials)
    ttype.extend([3] * len(specials))
    sp_ids = {s: base_sp + j for j, s in enumerate(specials)}
    return toks, ttype, sp_ids


def make_inkling2(
    path,
    seed=42,
    w2_type="iq4_nl",
    n_phys=8,
    block_size=2,
    core_start=2,
    core_end=6,
    num_loops=3,
    gate_rank=32,
    dim=256,
    n_head=6,
    n_head_kv=2,
    hd=64,
    inter=128,
    n_exp=8,
    n_used=2,
    d_rel=16,
    rel_extent=16,
    swa_win=6,
    l_cache=4,
    vocab_padded=320,
    vocab_unpadded=300,
    ctx=64,
):
    """Create a tiny PR #32695 (`general.architecture=inkling2`) GGUF."""
    assert n_phys % block_size == 0
    assert 0 < core_start < core_end <= n_phys
    assert core_start % block_size == 0 and (core_end - core_start) % block_size == 0
    assert n_head % n_head_kv == 0
    assert dim % 32 == 0 and (n_head * hd) % 32 == 0 and (n_head_kv * hd) % 32 == 0
    assert (2 * inter) % 256 == 0 and dim % 256 == 0 and inter % 32 == 0
    assert vocab_unpadded <= vocab_padded

    rng = np.random.default_rng(seed)
    visits = list(range(core_start)) + list(range(core_start, core_end)) * num_loops + list(range(core_end, n_phys))
    n_visits = len(visits)
    is_swa_phys = [(ip + 1) % block_size != 0 for ip in range(n_phys)]
    swp = [is_swa_phys[p] for p in visits]

    toks, ttype, sp_ids = inkling_tiny_vocab(vocab_unpadded)
    eos_id = sp_ids["<|content_model_end_sampling|>"]
    pad_id = sp_ids["<|endoftext|>"]

    qk_clip = 2.0
    rel_clip = 1.5
    log_floor = 4
    log_alpha = 0.25
    route_scale = 8.0
    situ_beta = 4.0
    situ_lin_beta = 25.0
    final_softcap = 10.0

    a = "inkling2"
    kvs = [
        ("general.architecture", T_STR, a),
        ("general.name", T_STR, "tiny-inkling2"),
        ("general.alignment", T_U32, ALIGN),
        (a + ".block_count", T_U32, n_visits),
        (a + ".context_length", T_U32, ctx),
        (a + ".embedding_length", T_U32, dim),
        (a + ".attention.head_count", T_ARR, (T_U32, [n_head] * n_visits)),
        (a + ".attention.head_count_kv", T_ARR, (T_U32, [n_head_kv] * n_visits)),
        (a + ".attention.key_length", T_U32, hd),
        (a + ".attention.value_length", T_U32, hd),
        (a + ".attention.key_length_swa", T_U32, hd),
        (a + ".attention.value_length_swa", T_U32, hd),
        (a + ".attention.layer_norm_rms_epsilon", T_F32, 1e-6),
        (a + ".attention.sliding_window", T_U32, swa_win),
        (a + ".attention.sliding_window_pattern", T_ARR, (T_BOOL, swp)),
        (a + ".attention.relative_length", T_U32, d_rel),
        (a + ".attention.relative_extent", T_U32, rel_extent),
        (a + ".attention.qk_clip", T_F32, qk_clip),
        (a + ".attention.relative_clip", T_F32, rel_clip),
        (a + ".attention.log_scale_floor", T_U32, log_floor),
        (a + ".attention.log_scale_alpha", T_F32, log_alpha),
        (a + ".rope.dimension_count", T_U32, 0),
        (a + ".shortconv.l_cache", T_U32, l_cache),
        (a + ".attn_res.block_size", T_U32, block_size),
        (a + ".expert_count", T_U32, n_exp),
        (a + ".expert_used_count", T_U32, n_used),
        (a + ".expert_feed_forward_length", T_U32, inter),
        (a + ".expert_weights_scale", T_F32, route_scale),
        (a + ".activation.situ_beta", T_F32, situ_beta),
        (a + ".activation.situ_linear_beta", T_F32, situ_lin_beta),
        (a + ".final_logit_softcapping", T_F32, final_softcap),
        (a + ".vocab_size", T_U32, vocab_padded),
        (a + ".vocab_size_unpadded", T_U32, vocab_unpadded),
        (a + ".num_loops", T_U32, num_loops),
        (a + ".loop.core_start", T_U32, core_start),
        (a + ".loop.core_end", T_U32, core_end),
        (a + ".loop.gate_rank", T_U32, gate_rank),
        ("tokenizer.ggml.model", T_STR, "gpt2"),
        ("tokenizer.ggml.pre", T_STR, "inkling2"),
        ("tokenizer.ggml.tokens", T_ARR, (T_STR, toks)),
        ("tokenizer.ggml.token_type", T_ARR, (T_I32, ttype)),
        ("tokenizer.ggml.merges", T_ARR, (T_STR, ["Ġ t", "h e"])),
        ("tokenizer.ggml.eos_token_id", T_U32, eos_id),
        ("tokenizer.ggml.eot_token_id", T_U32, eos_id),
        ("tokenizer.ggml.padding_token_id", T_U32, pad_id),
        ("tokenizer.ggml.add_bos_token", T_BOOL, False),
    ]

    w2_fn, w2_ggml = {
        "iq4_nl": (iq4_nl, GGML_IQ4_NL),
        "q4_0": (q4_0, GGML_Q4_0),
        "q8_0": (q8_0, GGML_Q8_0),
    }[w2_type]

    qd = n_head * hd
    kvd = n_head_kv * hd
    sc = lambda k: 0.5 / np.sqrt(k)
    n_loop_core = core_end - core_start

    T = [
        ("token_embd.weight", [dim, vocab_padded], GGML_Q8_0, q8_0(rng, vocab_padded, dim, 1.5)),
        ("token_embd_norm.weight", [dim], GGML_F32, f32(1.0 + 0.1 * rng.standard_normal(dim))),
    ]

    for ip in range(n_phys):
        p = f"blk.{ip}."
        swa = is_swa_phys[ip]

        if ip > 0:
            if swa:
                T.append((p + "attn_res_query.weight", [dim], GGML_F32, f32(0.2 * rng.standard_normal(dim))))
            else:
                T.append((p + "attn_res_query.weight", [dim, 3], GGML_F32, f32(0.2 * rng.standard_normal((3, dim)))))
        T.append((p + "ffn_res_query.weight", [dim], GGML_F32, f32(0.2 * rng.standard_normal(dim))))

        if swa:
            T.append((p + "attn_norm.weight", [dim], GGML_F32, f32(1.0 + 0.1 * rng.standard_normal(dim))))
        else:
            for nm in ("attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"):
                T.append((p + nm + ".weight", [dim], GGML_F32, f32(1.0 + 0.1 * rng.standard_normal(dim))))
            T.append((p + "attn_k_conv_proj.weight", [dim, l_cache * n_head_kv], GGML_Q8_0,
                      q8_0(rng, l_cache * n_head_kv, dim, sc(dim))))
            T.append((p + "attn_v_conv_proj.weight", [dim, l_cache * n_head_kv], GGML_Q8_0,
                      q8_0(rng, l_cache * n_head_kv, dim, sc(dim))))

        # Note: attn_q / attn_gate scale chosen slightly higher so L2 clip triggers
        T.append((p + "attn_q.weight", [dim, qd], GGML_Q8_0, q8_0(rng, qd, dim, 4.0 * sc(dim))))
        T.append((p + "attn_k.weight", [dim, kvd], GGML_Q8_0, q8_0(rng, kvd, dim, 4.0 * sc(dim))))
        T.append((p + "attn_v.weight", [dim, kvd], GGML_Q8_0, q8_0(rng, kvd, dim, sc(dim))))
        T.append((p + "attn_output.weight", [qd, dim], GGML_Q8_0, q8_0(rng, dim, qd, sc(qd))))
        T.append((p + "attn_gate.weight", [dim, qd], GGML_Q8_0, q8_0(rng, qd, dim, sc(dim))))
        T.append((p + "attn_rel.weight", [dim, d_rel * n_head], GGML_Q8_0, q8_0(rng, d_rel * n_head, dim, 4.0 * sc(dim))))
        T.append((p + "attn_sink_proj.weight", [dim, n_head], GGML_Q8_0, q8_0(rng, n_head, dim, sc(dim))))

        T.append((p + "attn_q_clip.weight", [hd, n_head], GGML_F32, f32(1.0 + 0.15 * rng.standard_normal((n_head, hd)))))
        T.append((p + "attn_rel_clip.weight", [d_rel, n_head], GGML_F32, f32(1.0 + 0.15 * rng.standard_normal((n_head, d_rel)))))
        T.append((p + "attn_rel_bias.weight", [d_rel, rel_extent], GGML_F32, f32(0.15 * rng.standard_normal((rel_extent, d_rel)))))

        T.append((p + "attn_k_conv.weight", [l_cache, kvd], GGML_F32, f32(0.25 * rng.standard_normal((kvd, l_cache)))))
        T.append((p + "attn_v_conv.weight", [l_cache, kvd], GGML_F32, f32(0.25 * rng.standard_normal((kvd, l_cache)))))
        T.append((p + "attn_post_conv.weight", [l_cache, dim], GGML_F32, f32(0.25 * rng.standard_normal((dim, l_cache)))))
        T.append((p + "ffn_post_conv.weight", [l_cache, dim], GGML_F32, f32(0.25 * rng.standard_normal((dim, l_cache)))))

        if core_start <= ip < core_end:
            for mod in ("attn", "k", "v", "mlp"):
                T.append((p + f"loop_{mod}_down.weight", [2 * dim, gate_rank], GGML_Q8_0,
                          q8_0(rng, gate_rank, 2 * dim, sc(2 * dim))))
                T.append((p + f"loop_{mod}_gain.weight", [gate_rank, dim], GGML_Q8_0,
                          q8_0(rng, dim, gate_rank, sc(gate_rank))))
                T.append((p + f"loop_{mod}_forget.weight", [gate_rank, dim], GGML_Q8_0,
                          q8_0(rng, dim, gate_rank, sc(gate_rank))))

        T.append((p + "ffn_norm.weight", [dim], GGML_F32, f32(1.0 + 0.1 * rng.standard_normal(dim))))
        # Separate gate_inp weights cleanly so top-k routing is insensitive to tiny rounding/q8 noise
        gate_w = rng.standard_normal((n_exp, dim)) * (2.0 / np.sqrt(dim))
        T.append((p + "ffn_gate_inp.weight", [dim, n_exp], GGML_F32, f32(gate_w)))
        T.append((p + "exp_probs_b.bias", [n_exp], GGML_F32, f32(0.05 * rng.standard_normal(n_exp))))
        T.append((p + "ffn_gate_up_exps.weight", [dim, 2 * inter, n_exp], GGML_IQ3_XXS,
                  iq3_xxs(rng, n_exp * 2 * inter, dim, 8.0 * sc(dim))))
        T.append((p + "ffn_down_exps.weight", [inter, dim, n_exp], w2_ggml,
                  w2_fn(rng, n_exp * dim, inter, sc(inter))))

    T.append(("loop_res_query.weight", [dim, 2 * n_loop_core], GGML_F32,
              f32(0.2 * rng.standard_normal((2 * n_loop_core, dim)))))
    T.append(("loop_kv_res_query.weight", [dim, 2 * n_loop_core], GGML_F32,
              f32(0.2 * rng.standard_normal((2 * n_loop_core, dim)))))
    T.append(("output_res_query.weight", [dim], GGML_F32, f32(0.2 * rng.standard_normal(dim))))
    T.append(("output_norm.weight", [dim], GGML_F32, f32(1.0 + 0.1 * rng.standard_normal(dim))))

    # Make padded rows [vocab_unpadded..vocab_padded) distinctly larger so missing unpadded-vocab slicing fails loudly
    out_raw = bytearray(q8_0(rng, vocab_unpadded, dim, 6.0 * sc(dim)))
    out_raw += q8_0(rng, vocab_padded - vocab_unpadded, dim, 40.0 * sc(dim))
    T.append(("output.weight", [dim, vocab_padded], GGML_Q8_0, bytes(out_raw)))

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_gguf(path, kvs, T)
    return path


def main():
    ap = argparse.ArgumentParser(description="Generate tiny synthetic inkling2_loop GGUF v3 test model.")
    ap.add_argument("out", nargs="?", default="build/tiny-inkling2.gguf",
                    help="Output .gguf path or directory (writes tiny-inkling2.gguf inside directory)")
    ap.add_argument("--w2-type", choices=["iq4_nl", "q4_0", "q8_0"], default="iq4_nl")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = os.path.join(a.out, "tiny-inkling2.gguf") if os.path.isdir(a.out) or not a.out.endswith(".gguf") else a.out
    make_inkling2(out, seed=a.seed, w2_type=a.w2_type)
    print(f"wrote {out} (w2_type={a.w2_type}, seed={a.seed})")


if __name__ == "__main__":
    main()
