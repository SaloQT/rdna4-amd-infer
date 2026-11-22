#!/usr/bin/env python3
"""Independent numpy float64 reference forward pass for Inkling-2-Mini (inkling2 / inkling2_loop).

Reads the GGUF (via `gguf.GGUFReader` or built-in pure-stdlib GGUF v3 fallback),
dequantizes weights directly from GGUF layout (including IQ3_XXS and IQ4_NL), and
runs the exact 48-visit looped forward pass in float64.

Modes:
  float : float64 activations everywhere with dequantized weights.
  q8    : before quantized matrices (Q8_0, Q4_0, IQ4_NL, IQ3_XXS, etc. -- not embeddings
          and not F32 tensors such as router/biases/conv/attn_res queries), quantizes
          the input activation per block of 32 as amd-infer does (d = amax/127,
          q = round half away from zero) and forms block dot products in float64.

usage:
  inkling_ref.py MODEL.gguf DUMP --mode q8|float [--tol REL] [--tol-median REL]
  inkling_ref.py --ppl MODEL.gguf IDS --mode q8|float [--first F] [--chunk C]
  inkling_ref.py --tokens MODEL.gguf 1,2,3 --mode float|q8
"""
import argparse
import os
import struct
import sys

import numpy as np

F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ3_XXS, IQ4_NL, IQ3_S, IQ4_XS, BF16 = (
    0, 1, 2, 8, 12, 13, 14, 18, 20, 21, 23, 30
)

_KVALUES_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.int64,
)

# Exact IQ3_XXS ksigns (128 bytes) & 256x4 grid (matches ggml / gguf.quants / kernels/ie_iqraw.inc)
_IQ3XXS_KSIGNS = np.array(
    [i | (128 if int(i).bit_count() & 1 else 0) for i in range(128)],
    dtype=np.uint8,
)
_IQ3XXS_GRID_MAP = np.array([0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3E], dtype=np.int64)
_IQ3XXS_GRID_HEX = (
    b"0000020004001100130017002000220031004200730075000101030110011201"
    b"2101250130013201410154017001000202020402110220022202310233023702"
    b"5102570275020103070310031203250370031304370444045704730475040105"
    b"0705320552053506640610071407160743076107011003101010121021102310"
    b"3010321034104710501000110211111120112211011203121012121221123012"
    b"7212001302132013311346136613011405145014201524154615711505162217"
    b"4017002002201120132020202220262031204220012103210521102112212121"
    b"3021632167217021002202221122172220222222372240225522012310231423"
    b"7023742335245324032527254125742501270327162745270130103012302130"
    b"2330503065307230003102312031313144314631013203321032253252327232"
    b"1133333330344734723400350635223555351436363663363337603704401740"
    b"3540374053405740744120423742404260426642074345430444514464442545"
    b"4345704505471047124730471250415070500051065126515551145232527252"
    b"0253535310542354275472540255315550562457425724604460466064602161"
    b"6161176264623063366344640565526533660367216703700570077010703270"
    b"5270267140711272457252720073157333736073217441740075027524753076"
)


def _build_iq3xxs_grid():
    raw = bytes.fromhex(_IQ3XXS_GRID_HEX.decode("ascii"))
    # Each 2-byte pair stores 4 octal digits across little-endian uint16? Let's match gguf:
    # nibble_i = (bytes_i >> {0,4}), 4 digits per 2 bytes: byte0=(d1<<4|d0), byte1=(d3<<4|d2)
    b = np.frombuffer(raw, dtype=np.uint8).reshape(256, 2)
    idx = np.stack([b[:, 0] & 0xF, b[:, 0] >> 4, b[:, 1] & 0xF, b[:, 1] >> 4], axis=-1)
    return _IQ3XXS_GRID_MAP[idx]  # (256, 4) int64


_IQ3XXS_GRID = _build_iq3xxs_grid()

_TYPE_BYTES = {F32: 4, F16: 2, BF16: 2, Q4_0: 18, Q8_0: 34, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ3_XXS: 98, IQ4_NL: 18, IQ3_S: 110, IQ4_XS: 136}
_TYPE_BLOCK = {F32: 1, F16: 1, BF16: 1, Q4_0: 32, Q8_0: 32, Q4_K: 256, Q5_K: 256, Q6_K: 256, IQ3_XXS: 256, IQ4_NL: 32, IQ3_S: 256, IQ4_XS: 256}


class _MiniTensor:
    def __init__(self, name, shape, tensor_type, data):
        self.name = name
        self.shape = shape
        self.tensor_type = tensor_type
        self.data = data


class _MiniGGUFReader:
    """Minimal fallback GGUF v3 reader when the `gguf` pip package is absent."""

    def __init__(self, path):
        self.fields = {}
        self.tensors = []
        with open(path, "rb") as f:
            magic, ver, n_ten, n_kv = struct.unpack("<4sIQQ", f.read(24))
            assert magic == b"GGUF" and ver == 3
            for _ in range(n_kv):
                k = self._str(f)
                t = struct.unpack("<I", f.read(4))[0]
                self.fields[k] = self._val(f, t)
            meta = []
            for _ in range(n_ten):
                name = self._str(f)
                nd = struct.unpack("<I", f.read(4))[0]
                ne = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
                tt, off = struct.unpack("<IQ", f.read(12))
                meta.append((name, ne, tt, off))
            align = int(self.fields.get("general.alignment", 32))
            pos = f.tell()
            data_start = (pos + align - 1) // align * align
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        for name, ne, tt, off in meta:
            nel = int(np.prod(ne))
            bs, tb = _TYPE_BLOCK[tt], _TYPE_BYTES[tt]
            nbytes = (nel // bs) * tb
            s = data_start + off
            self.tensors.append(_MiniTensor(name, ne, tt, mm[s:s + nbytes]))

    @staticmethod
    def _str(f):
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8")

    def _val(self, f, t):
        if t == 4:
            return struct.unpack("<I", f.read(4))[0]
        if t == 5:
            return struct.unpack("<i", f.read(4))[0]
        if t == 6:
            return struct.unpack("<f", f.read(4))[0]
        if t == 7:
            return bool(struct.unpack("<B", f.read(1))[0])
        if t == 8:
            return self._str(f)
        if t == 9:
            et, cnt = struct.unpack("<IQ", f.read(12))
            return [self._val(f, et) for _ in range(cnt)]
        if t == 10:
            return struct.unpack("<Q", f.read(8))[0]
        if t == 12:
            return struct.unpack("<d", f.read(8))[0]
        raise ValueError(f"Unsupported GGUF metadata type {t}")


def open_gguf(path):
    try:
        from gguf import GGUFReader
        return GGUFReader(path), True
    except ImportError:
        return _MiniGGUFReader(path), False


def field(r, is_pkg, key, default=None):
    if not is_pkg:
        return r.fields.get(key, default)
    f = r.get_field(key)
    if f is None:
        return default
    if len(f.types) >= 2 and f.types[0] == 9:  # array
        et = f.types[1]
        if et == 8:
            return [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        return [f.parts[i][0].item() for i in f.data]
    v = f.parts[f.data[0]]
    if f.types[0] == 8:
        return bytes(v).decode("utf-8")
    return v[0].item()


def _decode_raw(raw, tt, rows, cols, name=""):
    """Decode raw GGUF byte slice of shape (rows, cols) -> (w_f64, qint, d_blk, q_blk)."""
    if tt == F32:
        w = raw.view(np.float32).astype(np.float64).reshape(rows, cols)
        return w, False, None, None
    if tt == F16:
        w = raw.view(np.float16).astype(np.float64).reshape(rows, cols)
        return w, False, None, None
    if tt == BF16:
        w = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(rows, cols)
        return w, False, None, None
    if tt == Q8_0:
        nb = cols // 32
        b = raw.reshape(rows, nb, 34)
        d = b[:, :, :2].copy().view(np.float16).astype(np.float64).reshape(rows, nb)
        q = b[:, :, 2:].copy().view(np.int8).astype(np.int64)
        w = (d[:, :, None] * q.astype(np.float64)).reshape(rows, cols)
        return w, True, d, q
    if tt == Q4_0:
        nb = cols // 32
        b = raw.reshape(rows, nb, 18)
        d = b[:, :, :2].copy().view(np.float16).astype(np.float64).reshape(rows, nb)
        q = np.concatenate([b[:, :, 2:] & 15, b[:, :, 2:] >> 4], axis=2).astype(np.int64) - 8
        w = (d[:, :, None] * q.astype(np.float64)).reshape(rows, cols)
        return w, True, d, q
    if tt == IQ4_NL:
        nb = cols // 32
        b = raw.reshape(rows, nb, 18)
        d = b[:, :, :2].copy().view(np.float16).astype(np.float64).reshape(rows, nb)
        idx = np.concatenate([b[:, :, 2:] & 15, b[:, :, 2:] >> 4], axis=2)
        q = _KVALUES_IQ4NL[idx]
        w = (d[:, :, None] * q.astype(np.float64)).reshape(rows, cols)
        return w, True, d, q
    if tt == IQ3_XXS:
        assert cols % 256 == 0
        nsb = cols // 256
        nb = cols // 32
        b = raw.reshape(rows, nsb, 98)
        d = b[:, :, 0:2].copy().view(np.float16).astype(np.float64).reshape(rows, nsb)
        qs = b[:, :, 2:66].reshape(rows, nsb, 8, 8)  # 8 subblocks of 8 grid bytes (32 weights)
        aux = b[:, :, 66:98].copy().view(np.uint32).reshape(rows, nsb, 8)
        s4 = (aux >> np.uint32(28)).astype(np.float64)
        d_sub = (d[:, :, None] * (0.5 + s4) * 0.5).reshape(rows, nb)
        # 4 sign bytes per subblock -> 32 bits per subblock
        sidx = (aux[:, :, :, None] >> np.array([0, 7, 14, 21], dtype=np.uint32)) & np.uint32(0x7F)
        sbytes = _IQ3XXS_KSIGNS[sidx]  # (rows, nsb, 8, 4) u8
        sbits = (sbytes[:, :, :, :, None] >> np.arange(8, dtype=np.uint8)) & np.uint8(1)
        signs = np.where(sbits == 0, 1, -1).astype(np.int64).reshape(rows, nsb, 8, 32)
        grid = _IQ3XXS_GRID[qs].reshape(rows, nsb, 8, 32)
        q = (grid * signs).reshape(rows, nb, 32)
        w = (d_sub[:, :, None] * q.astype(np.float64)).reshape(rows, cols)
        return w, True, d_sub, q
    if tt in (Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS):
        from gguf import GGMLQuantizationType
        from gguf.quants import dequantize
        w = dequantize(raw, GGMLQuantizationType(tt)).astype(np.float64).reshape(rows, cols)
        nb = cols // 32
        return w, True, np.ones((rows, nb), dtype=np.float64), w.reshape(rows, nb, 32)
    raise SystemExit(f"unsupported tensor type {tt} in {name}")


class Mat:
    def __init__(self, t, slice_rows=None):
        ne = [int(x) for x in t.shape]
        cols = ne[0]
        rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
        tt = int(t.tensor_type)
        raw = np.asarray(t.data).view(np.uint8).reshape(-1)
        if slice_rows is not None and slice_rows < rows:
            rows = slice_rows
            rbytes = (cols // _TYPE_BLOCK[tt]) * _TYPE_BYTES[tt]
            raw = raw[:rows * rbytes]
        self.tt = tt
        self.rows = rows
        self.cols = cols
        self.w, self.qint, self.d, self.q = _decode_raw(raw, tt, rows, cols, t.name)


class ExpertBank:
    """Lazy per-expert decoder so running on a full 128-expert GGUF stays light."""

    def __init__(self, t, n_exp):
        ne = [int(x) for x in t.shape]
        self.cols = ne[0]
        self.rows_exp = ne[1]
        assert ne[2] == n_exp
        self.n_exp = n_exp
        self.tt = int(t.tensor_type)
        self.raw = np.asarray(t.data).view(np.uint8).reshape(-1)
        self.exp_bytes = self.rows_exp * (self.cols // _TYPE_BLOCK[self.tt]) * _TYPE_BYTES[self.tt]
        self.name = t.name
        self._cache = {}

    def get(self, e):
        m = self._cache.get(e)
        if m is None:
            s = e * self.exp_bytes
            w, qint, d, q = _decode_raw(self.raw[s:s + self.exp_bytes], self.tt, self.rows_exp, self.cols, self.name)
            m = _RawMat(w, qint, d, q)
            self._cache[e] = m
        return m


class _RawMat:
    def __init__(self, w, qint, d, q):
        self.w = w
        self.qint = qint
        self.d = d
        self.q = q


def quant_q8(x):
    xb = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(xb).max(axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    safe = np.where(d == 0, np.float32(1), d)
    r = (xb / safe[:, None]).astype(np.float32)
    q = (np.sign(r) * np.floor(np.abs(r) + np.float32(0.5))).astype(np.int64)
    q[d == 0] = 0
    return d.astype(np.float64), q


def matvec(m, x, mode):
    if mode == "q8" and m.qint:
        da, qa = quant_q8(x)
        dots = np.einsum("rbj,bj->rb", m.q, qa).astype(np.float64)
        return (m.d * da[None, :] * dots).sum(axis=1)
    return m.w @ x


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def rmsnorm_unweighted(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_l2(v, scale, threshold, eps=1e-6):
    if scale is not None:
        v = v * scale
    norm = np.sqrt(np.sum(v * v, axis=-1, keepdims=True))
    factor = np.minimum(1.0, threshold / (norm + eps))
    return v * factor


def attn_res_mix(blocks, cur, queries, eps):
    """Depth-attention residual mixing. queries shape: (dim,) or (n_q, dim)."""
    stacked = np.stack(blocks + [cur], axis=0)  # (M, dim)
    normed = rmsnorm_unweighted(stacked, eps)   # (M, dim)
    q = queries.reshape(-1, cur.shape[0])       # (n_q, dim)
    scores = (q @ normed.T) / np.sqrt(cur.shape[0])  # (n_q, M)
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=1, keepdims=True)
    out = probs @ normed                        # (n_q, dim)
    return out[0] if queries.ndim == 1 else out


class InklingModel:
    def __init__(self, path):
        r, is_pkg = open_gguf(path)
        a = field(r, is_pkg, "general.architecture")
        assert a == "inkling2", f"Expected architecture 'inkling2', got {a!r}"
        self.a = a
        self.n_visits = int(field(r, is_pkg, a + ".block_count"))
        self.dim = int(field(r, is_pkg, a + ".embedding_length"))
        self.eps = float(field(r, is_pkg, a + ".attention.layer_norm_rms_epsilon", 1e-6))
        self.swa_win = int(field(r, is_pkg, a + ".attention.sliding_window"))
        self.swp = [bool(v) for v in field(r, is_pkg, a + ".attention.sliding_window_pattern")]
        assert len(self.swp) == self.n_visits

        hc = field(r, is_pkg, a + ".attention.head_count")
        hkv = field(r, is_pkg, a + ".attention.head_count_kv")
        self.nh = int(hc[0] if isinstance(hc, list) else hc)
        self.nkv = int(hkv[0] if isinstance(hkv, list) else hkv)
        self.hd = int(field(r, is_pkg, a + ".attention.key_length"))
        self.d_rel = int(field(r, is_pkg, a + ".attention.relative_length"))
        self.rel_extent = int(field(r, is_pkg, a + ".attention.relative_extent"))
        self.qk_clip = float(field(r, is_pkg, a + ".attention.qk_clip"))
        self.rel_clip = float(field(r, is_pkg, a + ".attention.relative_clip"))
        self.log_floor = int(field(r, is_pkg, a + ".attention.log_scale_floor"))
        self.log_alpha = float(field(r, is_pkg, a + ".attention.log_scale_alpha"))
        self.l_cache = int(field(r, is_pkg, a + ".shortconv.l_cache"))
        self.block_size = int(field(r, is_pkg, a + ".attn_res.block_size"))

        self.n_exp = int(field(r, is_pkg, a + ".expert_count"))
        self.n_used = int(field(r, is_pkg, a + ".expert_used_count"))
        self.inter = int(field(r, is_pkg, a + ".expert_feed_forward_length"))
        self.route_scale = float(field(r, is_pkg, a + ".expert_weights_scale", 1.0))
        self.situ_beta = float(field(r, is_pkg, a + ".activation.situ_beta"))
        self.situ_lin_beta = float(field(r, is_pkg, a + ".activation.situ_linear_beta"))
        self.final_softcap = float(field(r, is_pkg, a + ".final_logit_softcapping", 0.0))

        self.vocab_padded = int(field(r, is_pkg, a + ".vocab_size"))
        self.vocab_unpadded = int(field(r, is_pkg, a + ".vocab_size_unpadded"))
        self.num_loops = int(field(r, is_pkg, a + ".num_loops", 1))
        self.core_start = int(field(r, is_pkg, a + ".loop.core_start", 0))
        self.core_end = int(field(r, is_pkg, a + ".loop.core_end", 0))
        self.gate_rank = int(field(r, is_pkg, a + ".loop.gate_rank", 0))
        self.n_loop_core = (self.core_end - self.core_start) if self.num_loops > 1 else 0
        self.n_phys = self.n_visits - (self.num_loops - 1) * self.n_loop_core

        T = {t.name: t for t in r.tensors}
        self.tok = Mat(T["token_embd.weight"])
        self.tok_norm = Mat(T["token_embd_norm.weight"]).w[0]
        self.out_norm = Mat(T["output_norm.weight"]).w[0]
        self.out_res_q = Mat(T["output_res_query.weight"]).w[0]
        self.out = Mat(T["output.weight"], slice_rows=self.vocab_unpadded)

        if self.num_loops > 1:
            self.loop_res_q = Mat(T["loop_res_query.weight"]).w
            self.loop_kv_res_q = Mat(T["loop_kv_res_query.weight"]).w

        self.phys = []
        for ip in range(self.n_phys):
            p = f"blk.{ip}."
            L = {}
            is_swa = self.swp[ip] if ip < self.core_end else self.swp[ip + (self.num_loops - 1) * self.n_loop_core]
            L["is_swa"] = is_swa
            if (p + "attn_res_query.weight") in T:
                q_mat = Mat(T[p + "attn_res_query.weight"]).w
                L["attn_res_query"] = q_mat[0] if is_swa else q_mat
            else:
                L["attn_res_query"] = None
            L["ffn_res_query"] = Mat(T[p + "ffn_res_query.weight"]).w[0]

            if is_swa:
                L["attn_norm"] = Mat(T[p + "attn_norm.weight"]).w[0]
            else:
                L["q_inp_norm"] = Mat(T[p + "attn_q_inp_norm.weight"]).w[0]
                L["k_inp_norm"] = Mat(T[p + "attn_k_inp_norm.weight"]).w[0]
                L["v_inp_norm"] = Mat(T[p + "attn_v_inp_norm.weight"]).w[0]
                L["k_conv_proj"] = Mat(T[p + "attn_k_conv_proj.weight"])
                L["v_conv_proj"] = Mat(T[p + "attn_v_conv_proj.weight"])

            for k in ("attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "attn_rel", "attn_sink_proj"):
                L[k] = Mat(T[p + k + ".weight"])

            L["q_clip"] = Mat(T[p + "attn_q_clip.weight"]).w
            L["rel_clip"] = Mat(T[p + "attn_rel_clip.weight"]).w
            L["rel_bias"] = Mat(T[p + "attn_rel_bias.weight"]).w  # (rel_extent, d_rel)

            for k in ("attn_k_conv", "attn_v_conv", "attn_post_conv", "ffn_post_conv"):
                L[k] = Mat(T[p + k + ".weight"]).w  # (channels, l_cache), [t-3, t-2, t-1, t]

            if self.num_loops > 1 and self.core_start <= ip < self.core_end:
                L["loop"] = {}
                for mod in ("attn", "k", "v", "mlp"):
                    L["loop"][mod] = (
                        Mat(T[p + f"loop_{mod}_down.weight"]),
                        Mat(T[p + f"loop_{mod}_gain.weight"]),
                        Mat(T[p + f"loop_{mod}_forget.weight"]),
                    )

            L["ffn_norm"] = Mat(T[p + "ffn_norm.weight"]).w[0]
            L["gate_inp"] = Mat(T[p + "ffn_gate_inp.weight"]).w  # F32
            L["exp_bias"] = Mat(T[p + "exp_probs_b.bias"]).w[0] if (p + "exp_probs_b.bias") in T else np.zeros(self.n_exp)
            L["gate_up_exps"] = ExpertBank(T[p + "ffn_gate_up_exps.weight"], self.n_exp)
            L["down_exps"] = ExpertBank(T[p + "ffn_down_exps.weight"], self.n_exp)
            self.phys.append(L)

        self.reset()

    def reset(self):
        kvd = self.nkv * self.hd
        l_prev = self.l_cache - 1
        self.kc = [[] for _ in range(self.n_visits)]
        self.vc = [[] for _ in range(self.n_visits)]
        self.k_pos = [[] for _ in range(self.n_visits)]
        # Conv state per logical visit: last (L-1) inputs [t-(L-1), ..., t-1]
        self.conv_k = [np.zeros((l_prev, kvd), dtype=np.float64) for _ in range(self.n_visits)]
        self.conv_v = [np.zeros((l_prev, kvd), dtype=np.float64) for _ in range(self.n_visits)]
        self.conv_pa = [np.zeros((l_prev, self.dim), dtype=np.float64) for _ in range(self.n_visits)]
        self.conv_pm = [np.zeros((l_prev, self.dim), dtype=np.float64) for _ in range(self.n_visits)]
        self.min_gap = 1e9

    def _visit_phys_pass(self, il):
        if self.num_loops <= 1 or il < self.core_start:
            return il, -1
        core_tot = self.num_loops * self.n_loop_core
        if il < self.core_start + core_tot:
            rel = il - self.core_start
            return self.core_start + (rel % self.n_loop_core), rel // self.n_loop_core
        return il - (self.num_loops - 1) * self.n_loop_core, -1

    def _sconv(self, u, static_w, state, dyn_taps=None):
        """Causal short conv (L = l_cache).
        static_w: (ch, L) ordered [t-(L-1), ..., t].
        dyn_taps: optional (n_kv, L) ordered [lag=0 (t), ..., lag=L-1].
        """
        win = np.concatenate([state, u[None, :]], axis=0)  # (L, ch)
        out = np.sum(win.T * static_w, axis=1)
        if dyn_taps is not None:
            # lag=0 weights win[L-1] (u), lag=L-1 weights win[0]
            hd = u.shape[0] // self.nkv
            win_h = win.reshape(self.l_cache, self.nkv, hd)
            dyn_rev = dyn_taps[:, ::-1]  # (n_kv, L) aligned with win axis 0
            out = out + np.sum(win_h * dyn_rev.T[:, :, None], axis=0).reshape(-1)
            scale = 1.0 / np.sqrt(2.0 * self.l_cache + 1.0)
        else:
            scale = 1.0 / np.sqrt(self.l_cache + 1.0)
        if state.shape[0] > 0:
            state[:] = win[1:]
        return (u + out) * scale

    def _control(self, cur, msg, ctrl, mode):
        down, gain, forget = ctrl
        h_in = np.concatenate([rmsnorm_unweighted(cur, self.eps), rmsnorm_unweighted(msg, self.eps)], axis=0)
        h = matvec(down, h_in, mode)
        g = np.tanh(matvec(gain, h, mode))
        f = np.tanh(matvec(forget, h, mode))
        return (1.0 - f) * cur + g * msg

    def step(self, tok, pos, mode="q8"):
        x = rmsnorm(self.tok.w[tok], self.tok_norm, self.eps)
        blocks = []
        pre_blocks = None
        pre_cur = None
        bank = None
        bank_last = None
        grp = self.nh // self.nkv

        for il in range(self.n_visits):
            ip, p_idx = self._visit_phys_pass(il)
            L = self.phys[ip]
            is_swa = L["is_swa"]

            messages = None
            if p_idx >= 0 and ip == self.core_start:
                if p_idx == 0:
                    pre_blocks = [b.copy() for b in blocks]
                    pre_cur = x.copy()
                else:
                    bank = [b.copy() for b in blocks[len(pre_blocks) + 1:]]
                    bank_last = x.copy()
                    blocks = [b.copy() for b in pre_blocks]
                    x = pre_cur.copy()

            if p_idx > 0:
                row = 2 * (ip - self.core_start)
                m_attn = attn_res_mix(bank, bank_last, self.loop_res_q[row:row + 2], self.eps)
                m_kv = attn_res_mix(bank, bank_last, self.loop_kv_res_q[row:row + 2], self.eps)
                messages = (m_attn[0], m_kv[0], m_kv[1], m_attn[1])

            block_res = x.copy()
            if L["attn_res_query"] is not None:
                mixed = attn_res_mix(blocks, x, L["attn_res_query"], self.eps)
                inputs = [mixed, mixed, mixed] if is_swa else [mixed[0], mixed[1], mixed[2]]
            else:
                inputs = [x, x, x]

            if messages is not None:
                mods = ("attn", "k", "v")
                inputs = [self._control(inputs[j], messages[j], L["loop"][mods[j]], mode) for j in range(3)]

            if is_swa:
                if messages is not None:
                    q_h, k_h, v_h = [rmsnorm(inputs[j], L["attn_norm"], self.eps) for j in range(3)]
                else:
                    q_h = rmsnorm(inputs[0], L["attn_norm"], self.eps)
                    k_h = v_h = q_h
            else:
                q_h = rmsnorm(inputs[0], L["q_inp_norm"], self.eps)
                k_h = rmsnorm(inputs[1], L["k_inp_norm"], self.eps)
                v_h = rmsnorm(inputs[2], L["v_inp_norm"], self.eps)

            Q = matvec(L["attn_q"], q_h, mode).reshape(self.nh, self.hd)
            Q = clip_l2(Q, L["q_clip"], self.qk_clip, self.eps)

            k_dyn = None if is_swa else matvec(L["k_conv_proj"], k_h, mode).reshape(self.nkv, self.l_cache)
            v_dyn = None if is_swa else matvec(L["v_conv_proj"], v_h, mode).reshape(self.nkv, self.l_cache)
            K = self._sconv(matvec(L["attn_k"], k_h, mode), L["attn_k_conv"], self.conv_k[il], k_dyn).reshape(self.nkv, self.hd)
            K = clip_l2(K, None, self.qk_clip, self.eps)
            V = self._sconv(matvec(L["attn_v"], v_h, mode), L["attn_v_conv"], self.conv_v[il], v_dyn).reshape(self.nkv, self.hd)
            V = rmsnorm_unweighted(V, self.eps)

            R = matvec(L["attn_rel"], q_h, mode).reshape(self.nh, self.d_rel)
            R = clip_l2(R, L["rel_clip"], self.rel_clip, self.eps)

            if not is_swa:
                tau = 1.0 + self.log_alpha * np.log(max((pos + 1.0) / self.log_floor, 1.0))
                Q = Q * tau
                R = R * tau

            self.kc[il].append(K)
            self.vc[il].append(V)
            self.k_pos[il].append(pos)
            if is_swa and len(self.k_pos[il]) > self.swa_win:
                self.kc[il].pop(0)
                self.vc[il].pop(0)
                self.k_pos[il].pop(0)

            k_arr = np.stack(self.kc[il], axis=0)  # (kv_len, nkv, hd)
            v_arr = np.stack(self.vc[il], axis=0)
            dists = np.array([pos - kp for kp in self.k_pos[il]], dtype=np.int64)
            valid = (dists >= 0) & (dists < self.swa_win if is_swa else True)

            rel_table = R @ L["rel_bias"].T  # (nh, rel_extent)
            sinks = matvec(L["attn_sink_proj"], q_h, mode)
            extent_lim = self.swa_win if is_swa else self.rel_extent

            att = np.empty((self.nh, self.hd), dtype=np.float64)
            in_rel = (dists >= 0) & (dists < extent_lim)
            safe_d = np.clip(dists, 0, extent_lim - 1)
            for hh in range(self.nh):
                scores = (k_arr[:, hh // grp, :] @ Q[hh]) / np.sqrt(self.hd)
                bias = np.where(in_rel, rel_table[hh, safe_d], 0.0)
                logits_k = np.where(valid, scores + bias, -np.inf)
                comb = np.append(logits_k, sinks[hh])
                mx = comb.max()
                ex = np.exp(comb - mx)
                probs = ex[:-1] / ex.sum()
                att[hh] = probs @ v_arr[:, hh // grp, :]

            g_attn = sigmoid(matvec(L["attn_gate"], q_h, mode))
            attn_proj = matvec(L["attn_output"], att.reshape(-1) * g_attn, mode)
            attn_out = self._sconv(attn_proj, L["attn_post_conv"], self.conv_pa[il])

            if ip % self.block_size == 0:
                blocks.append(block_res)
                x = attn_out
            else:
                x = block_res + attn_out

            ffn_inp = x.copy()
            m_ffn = attn_res_mix(blocks, x, L["ffn_res_query"], self.eps)
            if messages is not None:
                m_ffn = self._control(m_ffn, messages[3], L["loop"]["mlp"], mode)
            h_ffn = rmsnorm(m_ffn, L["ffn_norm"], self.eps)

            r_scores = sigmoid(L["gate_inp"] @ h_ffn)
            sel_scores = r_scores + L["exp_bias"]
            order = np.argsort(-sel_scores, kind="mergesort")
            sel = order[:self.n_used]
            self.min_gap = min(self.min_gap, float(sel_scores[order[0 if self.n_used == 0 else self.n_used - 1]] - sel_scores[order[self.n_used]])) if self.n_used < self.n_exp else self.min_gap
            w_exp = r_scores[sel] / (r_scores[sel].sum() + 1e-20) * self.route_scale

            y_moe = np.zeros(self.dim, dtype=np.float64)
            for idx_e, w_e in zip(sel, w_exp):
                gu = matvec(L["gate_up_exps"].get(int(idx_e)), h_ffn, mode)
                g, u = gu[:self.inter], gu[self.inter:]
                act = (self.situ_beta * np.tanh(g / self.situ_beta)) * sigmoid(g) * (self.situ_lin_beta * np.tanh(u / self.situ_lin_beta))
                y_moe += w_e * matvec(L["down_exps"].get(int(idx_e)), act, mode)

            mlp_out = self._sconv(y_moe, L["ffn_post_conv"], self.conv_pm[il])
            x = ffn_inp + mlp_out

        x = attn_res_mix(blocks, x, self.out_res_q, self.eps)
        x = rmsnorm(x, self.out_norm, self.eps)
        logits = matvec(self.out, x, mode)
        if self.final_softcap > 0.0:
            logits = self.final_softcap * np.tanh(logits / self.final_softcap)
        return logits


def ppl_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("ids")
    ap.add_argument("--mode", choices=["q8", "float"], default="float")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    a = ap.parse_args(argv)
    ids = [int(t) for t in open(a.ids).read().replace(",", " ").split()]
    chunks = [ids[i:i + a.chunk] for i in range(0, len(ids), a.chunk)] if a.chunk else [ids]
    m = InklingModel(a.model)
    nll, n = 0.0, 0
    for ch in chunks:
        m.reset()
        for pos in range(len(ch) - 1):
            ref = m.step(ch[pos], pos, a.mode)
            if pos >= a.first:
                mx = ref.max()
                nll += mx + np.log(np.exp(ref - mx).sum()) - ref[ch[pos + 1]]
                n += 1
    print(f"ppl: {n} tokens scored, mean nll {nll / n:.6f}, perplexity {np.exp(nll / n):.4f} (reference, mode={a.mode})")


def tokens_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tokens")
    ap.add_argument("--mode", choices=["q8", "float"], default="float")
    a = ap.parse_args(argv)
    toks = [int(t) for t in a.tokens.replace(",", " ").split() if t.strip()]
    m = InklingModel(a.model)
    for pos, tok in enumerate(toks):
        logits = m.step(tok, pos, a.mode)
        top = int(np.argmax(logits))
        print(f"pos={pos} tok={tok} top={top} logit_max={logits[top]:.6f} min_gap={m.min_gap:.3e}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--ppl":
        return ppl_main(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "--tokens":
        return tokens_main(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dump")
    ap.add_argument("--mode", choices=["q8", "float"], default="q8")
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--tol-median", type=float, default=None)
    a = ap.parse_args()
    rows = [l.split() for l in open(a.dump) if l.strip()]
    m = InklingModel(a.model)
    worst_abs = worst_rel = 0.0
    rels = []
    tok_ok = True
    for i, row in enumerate(rows):
        pos, tok = int(row[0]), int(row[1])
        c = np.array([float(v) for v in row[2:]])
        ref = m.step(tok, pos, a.mode)
        err = np.abs(c - ref).max()
        rel = err / np.abs(ref).max()
        worst_abs, worst_rel = max(worst_abs, err), max(worst_rel, rel)
        rels.append(rel)
        if i + 1 < len(rows):
            nxt = int(rows[i + 1][1])
            is_gen = nxt == int(np.argmax(c))
            if is_gen and int(np.argmax(ref)) != nxt:
                srt = np.sort(ref)
                print(f"  pos {pos}: greedy token differs (ref {int(np.argmax(ref))}, engine {nxt}; ref top-2 gap {srt[-1] - srt[-2]:.3g})")
                tok_ok = False
    med = float(np.median(rels))
    ok = tok_ok and (a.tol is None or worst_rel <= a.tol) and (a.tol_median is None or med <= a.tol_median)
    print("%s mode=%s: %d positions, max abs err %.3e, rel err max %.3e (tol %s) median %.3e (tol %s), "
          "greedy tokens %s -> %s" % (os.path.basename(a.model), a.mode, len(rows), worst_abs, worst_rel, a.tol, med,
                                     a.tol_median, "identical" if tok_ok else "DIFFER", "PASS" if ok else "FAIL"))
    print("  per position rel err: " + " ".join("%.1e" % r for r in rels))
    print("  routing: smallest top-k boundary gap %.3g (a near-tie can choose another expert)" % m.min_gap)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
