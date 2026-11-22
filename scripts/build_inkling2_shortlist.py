#!/usr/bin/env python3
"""Offline FR-Spec + MicroSpec 60K+4K certified shortlist builder for Inkling-2-Mini (R30).

Reads Inkling-2-Mini's GGUF tokenizer metadata directly (pure Python stdlib, 0 GPU calls),
implements an exact 1:1 port of ref/tr8-team-merge/src/tok.c (PRE_INKLING2 / GPT-4o
pre-tokenizer + GPT-2 byte-level BPE with ignore_merges=True), counts empirical token
frequencies over an independent calibration corpus, and emits:

  1. inkling2-frspec-<core_size>.ids.txt        (ascending decimal IDs, LF-terminated)
  2. inkling2-frspec-<core_size>.bin            (32-byte header + u32[core_size] LE)
  3. inkling2-frspec-<core_size>.manifest.json  (rdna4.mtp.vocab-shortlist.v1 schema,
                                                 directly loadable by src/model.c
                                                 ie_load_vocab_shortlist()!)

Strict data hygiene: --heldout and --probe-prompts are used ONLY for read-only coverage
certification (static core vs core + 4K MicroSpec dynamic prompt union) and NEVER enter
the calibration frequency table.
"""

from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import os
import struct
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---- Minimal GGUF v3 Metadata Reader (metadata-only; skips tensor data) ----

_GGUF_MAGIC = b"GGUF"
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_SCALAR_FMT = {
    _U8: "<B",
    _I8: "<b",
    _U16: "<H",
    _I16: "<h",
    _U32: "<I",
    _I32: "<i",
    _F32: "<f",
    _BOOL: "<B",
    _U64: "<Q",
    _I64: "<q",
    _F64: "<d",
}
_SCALAR_SIZE = {k: struct.calcsize(v) for k, v in _SCALAR_FMT.items()}


def _read_gguf_kv(path: Path, wanted_keys: Optional[set] = None) -> Dict[str, object]:
    """Read GGUF KV metadata (fast: skips large unwanted arrays without decoding)."""
    kv: Dict[str, object] = {}
    with path.open("rb") as f:
        if f.read(4) != _GGUF_MAGIC:
            raise ValueError(f"{path}: not a GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        _n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def read_str(keep: bool) -> Optional[str]:
            n = struct.unpack("<Q", f.read(8))[0]
            if not keep:
                f.seek(n, os.SEEK_CUR)
                return None
            return f.read(n).decode("utf-8", "replace")

        def read_val(vtype: int, keep: bool):
            if vtype == _STR:
                return read_str(keep)
            if vtype == _ARR:
                ec = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                if ec == _STR:
                    if not keep:
                        for _ in range(n):
                            slen = struct.unpack("<Q", f.read(8))[0]
                            f.seek(slen, os.SEEK_CUR)
                        return None
                    return [f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace") for _ in range(n)]
                sz = _SCALAR_SIZE[ec]
                if not keep:
                    f.seek(n * sz, os.SEEK_CUR)
                    return None
                fmt = _SCALAR_FMT[ec]
                return list(struct.unpack(f"<{n}{fmt[1]}", f.read(n * sz)))
            sz = _SCALAR_SIZE[vtype]
            if not keep:
                f.seek(sz, os.SEEK_CUR)
                return None
            return struct.unpack(_SCALAR_FMT[vtype], f.read(sz))[0]

        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            key = f.read(klen).decode("utf-8")
            vtype = struct.unpack("<I", f.read(4))[0]
            keep = wanted_keys is None or key in wanted_keys
            val = read_val(vtype, keep)
            if keep:
                kv[key] = val
    return kv


# ---- Exact 1:1 Port of ref/tr8-team-merge/src/tok.c (PRE_INKLING2) ----

_UC_L = 1
_UC_M = 2
_UC_N = 4
_UC_SPACE = 8
_UC_CRLF = 16
_UC_UP = 32
_UC_LO = 64
_UC_LMO = 128


def _uc_space(cp: int) -> bool:
    return (
        (0x09 <= cp <= 0x0D)
        or cp == 0x20
        or cp == 0x85
        or cp == 0xA0
        or cp == 0x1680
        or (0x2000 <= cp <= 0x200A)
        or cp in (0x2028, 0x2029, 0x202F, 0x205F, 0x3000)
    )


@functools.lru_cache(maxsize=65536)
def _char_class(ch: str) -> int:
    cp = ord(ch)
    cat = unicodedata.category(ch)
    cl = 0
    if cat[0] == "L":
        cl |= _UC_L
        if cat in ("Lu", "Lt"):
            cl |= _UC_UP
        elif cat == "Ll":
            cl |= _UC_LO
        else:
            cl |= _UC_LMO
    elif cat[0] == "M":
        cl |= _UC_M
    elif cat[0] == "N":
        cl |= _UC_N
    if _uc_space(cp):
        cl |= _UC_SPACE
    if cp == 10 or cp == 13:
        cl |= _UC_CRLF
    return cl


def _gpt2_bytes_to_unicode() -> List[str]:
    out = [""] * 256
    n = 0
    for b in range(256):
        keep = (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255)
        cp = b if keep else 256 + n
        if not keep:
            n += 1
        out[b] = chr(cp)
    return out


_BCHAR = _gpt2_bytes_to_unicode()
_BCHAR_MAP = {ch: b for b, ch in enumerate(_BCHAR)}

_FU_MASK = _UC_UP | _UC_LMO | _UC_M
_FW_MASK = _UC_LO | _UC_LMO | _UC_M
_FPRE_FORBID = _UC_CRLF | _UC_L | _UC_N
_OTHER_FORBID = _UC_SPACE | _UC_L | _UC_N


def _contraction(text: str, j: int, n: int) -> int:
    if j + 1 < n and text[j] == "'":
        a = text[j + 1].lower()
        if a in ("s", "t", "m", "d"):
            return j + 2
        if j + 2 < n:
            ab = a + text[j + 2].lower()
            if ab in ("re", "ve", "ll"):
                return j + 3
    return j


def _match_case_word(text: str, cl: List[int], i: int, n: int) -> int:
    for pf in (1, 0):
        if pf and (cl[i] & _FPRE_FORBID):
            continue
        j = i + pf
        if j >= n:
            continue
        ue = j
        while ue < n and (cl[ue] & _FU_MASK):
            ue += 1
        k = ue
        while True:
            we = k
            while we < n and (cl[we] & _FW_MASK):
                we += 1
            if we > k:
                return _contraction(text, we, n)
            if k == j:
                break
            k -= 1
    for pf in (1, 0):
        if pf and (cl[i] & _FPRE_FORBID):
            continue
        j = i + pf
        if j >= n:
            continue
        ue = j
        while ue < n and (cl[ue] & _FU_MASK):
            ue += 1
        if ue > j:
            while ue < n and (cl[ue] & _FW_MASK):
                ue += 1
            return _contraction(text, ue, n)
    return i


def _match_ws(cl: List[int], i: int, n: int) -> int:
    j = i
    while j < n and (cl[j] & _UC_SPACE):
        j += 1
    for k in range(j, i, -1):
        if cl[k - 1] & _UC_CRLF:
            return k
    if j == n:
        return j
    if j - 1 > i:
        return j - 1
    return j


def _match_gpt4o(text: str, cl: List[int], i: int, n: int) -> int:
    w = _match_case_word(text, cl, i, n)
    if w > i:
        return w
    if cl[i] & _UC_N:
        j = i + 1
        lim = min(n, i + 3)
        while j < lim and (cl[j] & _UC_N):
            j += 1
        return j
    j = i + (1 if text[i] == " " else 0)
    k = j
    while k < n and not (cl[k] & _OTHER_FORBID):
        k += 1
    if k > j:
        while k < n and ((cl[k] & _UC_CRLF) or text[k] == "/"):
            k += 1
        return k
    if cl[i] & _UC_SPACE:
        return _match_ws(cl, i, n)
    return i + 1


class InklingTokenizer:
    """Exact pure-Python mirror of ref/tr8-team-merge/src/tok.c for PRE_INKLING2."""

    def __init__(self, gguf_path: Path):
        wanted = {
            "general.architecture",
            "general.name",
            "inkling2.vocab_size",
            "inkling2.vocab_size_unpadded",
            "tokenizer.ggml.model",
            "tokenizer.ggml.pre",
            "tokenizer.ggml.ignore_merges",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.padding_token_id",
        }
        kv = _read_gguf_kv(gguf_path, wanted)
        self.kv = kv
        self.tokens: List[str] = kv["tokenizer.ggml.tokens"]  # type: ignore[assignment]
        self.token_types: List[int] = kv.get("tokenizer.ggml.token_type") or [1] * len(self.tokens)  # type: ignore[assignment]
        self.merges_list: List[str] = kv["tokenizer.ggml.merges"]  # type: ignore[assignment]
        self.pre: str = str(kv.get("tokenizer.ggml.pre", "inkling2"))
        self.vocab_padded: int = int(kv.get("inkling2.vocab_size", len(self.tokens)))
        self.vocab_unpadded: int = int(kv.get("inkling2.vocab_size_unpadded", len(self.tokens)))
        self.ignore_merges: bool = bool(kv.get("tokenizer.ggml.ignore_merges", self.pre == "inkling2"))
        self.eos_id: int = int(kv.get("tokenizer.ggml.eos_token_id", 200006))
        self.pad_id: int = int(kv.get("tokenizer.ggml.padding_token_id", 199999))
        self.eot_id: int = int(kv.get("tokenizer.ggml.eot_token_id", self.eos_id))
        self.bos_id: Optional[int] = int(kv["tokenizer.ggml.bos_token_id"]) if "tokenizer.ggml.bos_token_id" in kv else None

        self.vocab: Dict[str, int] = {}
        for idx, tok in enumerate(self.tokens):
            if tok not in self.vocab:
                self.vocab[tok] = idx

        self.merges: Dict[Tuple[str, str], int] = {}
        for rank, line in enumerate(self.merges_list):
            sp = line.find(" ")
            if sp > 0:
                pair = (line[:sp], line[sp + 1 :])
                if pair not in self.merges:
                    self.merges[pair] = rank

        self.ctrl: bytearray = bytearray(len(self.tokens))
        specials: List[Tuple[str, int, bool]] = []
        for idx, ty in enumerate(self.token_types[: len(self.tokens)]):
            if ty == 3:
                self.ctrl[idx] = 1
            if ty in (3, 4) and self.tokens[idx]:
                specials.append((self.tokens[idx], idx, ty == 3))
        specials.sort(key=lambda x: (-len(x[0]), x[1]))
        self.specials = specials
        self.sp_by_first: Dict[str, List[Tuple[str, int, bool]]] = collections.defaultdict(list)
        for s_tok, s_id, s_ctrl in specials:
            self.sp_by_first[s_tok[0]].append((s_tok, s_id, s_ctrl))

        self.byte_ids: List[int] = [self.vocab.get(_BCHAR[b], -1) for b in range(256)]

        # Compute deterministic tokenizer fingerprint matching pilot provenance conventions
        h = hashlib.sha256()
        h.update(f"pre={self.pre}|vocab={self.vocab_padded}|unpadded={self.vocab_unpadded}\n".encode("utf-8"))
        for idx, tok in enumerate(self.tokens):
            ty = self.token_types[idx] if idx < len(self.token_types) else 1
            h.update(f"{idx}\t{ty}\t{tok}\n".encode("utf-8"))
        h.update(f"merges={len(self.merges_list)}\n".encode("utf-8"))
        for m in self.merges_list:
            h.update(m.encode("utf-8"))
            h.update(b"\n")
        self.fingerprint = h.hexdigest()

        self._bpe_cached = functools.lru_cache(maxsize=131072)(self._bpe_piece_uncached)

    def _bpe_piece_uncached(self, piece: str) -> Tuple[int, ...]:
        raw = piece.encode("utf-8")
        gstr = "".join(_BCHAR[b] for b in raw)
        if self.ignore_merges:
            tid = self.vocab.get(gstr)
            if tid is not None and not self.ctrl[tid]:
                return (tid,)
        syms = [_BCHAR[b] for b in raw]
        ns = len(syms)
        if ns > 1:
            merges = self.merges
            INF = 0x7FFFFFFF
            ranks = [merges.get((syms[k], syms[k + 1]), INF) for k in range(ns - 1)]
            while ns > 1:
                best = INF
                bk = -1
                for k, r in enumerate(ranks):
                    if r < best:
                        best = r
                        bk = k
                if best == INF:
                    break
                syms[bk] = syms[bk] + syms[bk + 1]
                del syms[bk + 1]
                del ranks[bk]
                ns -= 1
                if bk < ns - 1:
                    ranks[bk] = merges.get((syms[bk], syms[bk + 1]), INF)
                if bk > 0:
                    ranks[bk - 1] = merges.get((syms[bk - 1], syms[bk]), INF)
        out: List[int] = []
        vocab = self.vocab
        byte_ids = self.byte_ids
        for sym in syms:
            tid = vocab.get(sym)
            if tid is not None:
                out.append(tid)
            else:
                for ch in sym:
                    b = _BCHAR_MAP.get(ch)
                    if b is not None and byte_ids[b] >= 0:
                        out.append(byte_ids[b])
        return tuple(out)

    def _encode_plain(self, text: str, out: List[int]) -> None:
        n = len(text)
        if n == 0:
            return
        cl = [_char_class(ch) for ch in text]
        i = 0
        bpe = self._bpe_cached
        while i < n:
            j = _match_gpt4o(text, cl, i, n)
            out.extend(bpe(text[i:j]))
            i = j

    def encode(self, text: str, parse_control: bool = True) -> List[int]:
        out: List[int] = []
        n = len(text)
        a = 0
        i = 0
        sp_by_first = self.sp_by_first
        while i < n:
            cands = sp_by_first.get(text[i])
            hit = None
            if cands:
                for s_tok, s_id, s_ctrl in cands:
                    if (parse_control or not s_ctrl) and text.startswith(s_tok, i):
                        hit = (s_tok, s_id)
                        break
            if hit is None:
                i += 1
                continue
            if i > a:
                self._encode_plain(text[a:i], out)
            out.append(hit[1])
            i += len(hit[0])
            a = i
        if n > a:
            self._encode_plain(text[a:n], out)
        return out

    def tok_is_ascii(self, tid: int) -> bool:
        """True iff token decodes exclusively to 7-bit ASCII bytes (mirrors src/model.c:2192)."""
        if tid >= len(self.tokens):
            return False
        s = self.tokens[tid]
        if not s:
            return False
        for ch in s:
            b = _BCHAR_MAP.get(ch)
            if b is None or b >= 0x80:
                return False
        return True


# ---- Corpus Readers ----

def _iter_corpus_texts(paths: Sequence[Path], max_docs: int = 0, max_bytes: int = 0) -> Iterable[Tuple[str, str]]:
    """Yield (source_label, text) from .jsonl, .txt, or source directories."""
    docs = 0
    total_bytes = 0
    for p in paths:
        if p.is_dir():
            files = sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in {
                ".py", ".c", ".h", ".inc", ".cc", ".cpp", ".rs", ".ts", ".js", ".md", ".txt", ".json", ".ps1", ".sh"
            })
        else:
            files = [p]
        for fp in files:
            if fp.suffix.lower() == ".jsonl":
                with fp.open("r", encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        txt = obj.get("text") or obj.get("content") or ""
                        if not isinstance(txt, str) or not txt:
                            continue
                        docs += 1
                        total_bytes += len(txt.encode("utf-8", "ignore"))
                        yield (f"{fp.name}:{line_no}", txt)
                        if max_docs and docs >= max_docs:
                            return
                        if max_bytes and total_bytes >= max_bytes:
                            return
            else:
                txt = fp.read_text(encoding="utf-8", errors="replace")
                if not txt:
                    continue
                docs += 1
                total_bytes += len(txt.encode("utf-8", "ignore"))
                yield (str(fp), txt)
                if max_docs and docs >= max_docs:
                    return
                if max_bytes and total_bytes >= max_bytes:
                    return


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---- Core Shortlist Construction & MicroSpec Coverage Evaluator ----

def build_shortlist(
    tok: InklingTokenizer,
    counts: collections.Counter,
    core_size: int,
) -> Tuple[List[int], Dict[str, int], List[int]]:
    """Build strictly ascending certified core_size shortlist (< tok.vocab_unpadded)."""
    unpad = tok.vocab_unpadded
    if core_size < 512 or core_size >= unpad:
        raise ValueError(f"core_size={core_size} must be in [512, {unpad - 1}]")

    selected: set[int] = set()
    tier_counts = {
        "tier1_mandatory": 0,
        "tier2_corpus_freq": 0,
        "tier3a_space_ascii_bpe": 0,
        "tier3b_nonspace_ascii_bpe": 0,
        "tier3c_id_fill": 0,
    }
    required_eog: set[int] = set()

    def add(tid: int, tier: str) -> bool:
        if 0 <= tid < unpad and tid not in selected and len(selected) < core_size:
            selected.add(tid)
            tier_counts[tier] += 1
            return True
        return False

    # Tier 1: Mandatory EOG, control/special tokens, and all 256 byte-fallback tokens.
    for sid in (tok.eos_id, tok.pad_id, tok.eot_id, tok.bos_id):
        if sid is not None and 0 <= sid < unpad:
            required_eog.add(sid)
            add(sid, "tier1_mandatory")
    for tid, ty in enumerate(tok.token_types[:unpad]):
        if ty != 1:
            add(tid, "tier1_mandatory")
            # Include key structural Inkling-2 turn/end markers in required_eog_ids
            s = tok.tokens[tid]
            if s in (
                "<|endoftext|>",
                "<|content_model_end_sampling|>",
                "<|end_message|>",
                "<|message_user|>",
                "<|message_model|>",
                "<|message_system|>",
                "<|message_tool|>",
                "<|content_text|>",
                "<|content_thinking|>",
                "<|content_invoke_tool_json|>",
                "<|content_invoke_tool_text|>",
            ):
                required_eog.add(tid)
    for bid in tok.byte_ids:
        if 0 <= bid < unpad:
            add(bid, "tier1_mandatory")

    # Tier 2: Observed corpus tokens ranked by (-frequency, token_id).
    ranked_obs = sorted(
        ((tid, cnt) for tid, cnt in counts.items() if 0 <= tid < unpad and cnt > 0),
        key=lambda x: (-x[1], x[0]),
    )
    for tid, _cnt in ranked_obs:
        if len(selected) >= core_size:
            break
        add(tid, "tier2_corpus_freq")

    # Tier 3a: Inkling-2-Mini allocates leading-space ('Ġ') ASCII/code tokens in a separate upper
    # BPE band (~149,860..199,997; 32,339 tokens total). Include all remaining 'Ġ'-ASCII tokens in
    # ascending BPE merge-rank order before rare non-space substrings so space-prefixed words/code
    # identifiers are never starved by low-band non-space tokens.
    if len(selected) < core_size:
        for tid in range(unpad):
            if len(selected) >= core_size:
                break
            if tid not in selected and tok.tokens[tid].startswith("\u0120") and tok.tok_is_ascii(tid):
                add(tid, "tier3a_space_ascii_bpe")

    # Tier 3b: Zero-frequency non-space pure-ASCII / code tokens in BPE merge-rank (ID) order.
    if len(selected) < core_size:
        for tid in range(unpad):
            if len(selected) >= core_size:
                break
            if tid not in selected and tok.tok_is_ascii(tid):
                add(tid, "tier3b_nonspace_ascii_bpe")

    # Tier 3c: Remaining lowest BPE token IDs up to exact core_size.
    if len(selected) < core_size:
        for tid in range(unpad):
            if len(selected) >= core_size:
                break
            add(tid, "tier3c_id_fill")

    ids_sorted = sorted(selected)
    assert len(ids_sorted) == core_size, f"Expected {core_size} ids, got {len(ids_sorted)}"
    return ids_sorted, tier_counts, sorted(required_eog)


def _parse_id_file_or_encode(tok: InklingTokenizer, txt: str) -> List[int]:
    """If txt is a comma/whitespace-separated list of integer token IDs (e.g. real-code-4k.txt),
    parse the IDs directly; otherwise encode with InklingTokenizer."""
    stripped = txt.strip()
    if stripped and all(c in "0123456789, \t\r\n" for c in stripped[:512]):
        parts = [p for p in stripped.replace("\n", ",").replace("\r", ",").split(",") if p.strip()]
        if parts and all(p.strip().isdigit() for p in parts[:64]):
            try:
                ids = [int(p.strip()) for p in parts]
                if all(0 <= x < tok.vocab_unpadded for x in ids):
                    return ids
            except ValueError:
                pass
    return tok.encode(txt, parse_control=True)


def evaluate_microspec_coverage(
    tok: InklingTokenizer,
    core_set: set[int],
    docs: Sequence[Tuple[str, str]],
    max_dynamic: int = 4096,
    split_ratio: float = 0.5,
) -> Dict[str, object]:
    """Measure static 60K vs MicroSpec 60K+4K coverage.

    For each document (or prompt), splits tokens at split_ratio into prefix ('prompt')
    and suffix ('continuation') to measure:
      - full_static_cov: fraction of ALL tokens covered by static core_set
      - cont_static_cov: fraction of continuation tokens covered by static core_set
      - cont_microspec_cov: fraction of continuation tokens covered by
        (core_set U first <= max_dynamic novel unique tokens from prefix)
      - avg_dyn_slots: mean dynamic tail slots consumed per document
      - max_dyn_slots: max dynamic tail slots consumed across documents
    """
    total_toks = 0
    static_hits = 0
    cont_toks = 0
    cont_static_hits = 0
    cont_ms_hits = 0
    dyn_counts: List[int] = []
    per_doc: List[Dict[str, object]] = []

    for label, txt in docs:
        ids = _parse_id_file_or_encode(tok, txt)
        n = len(ids)
        if n == 0:
            continue
        s_hits = sum(1 for t in ids if t in core_set)
        total_toks += n
        static_hits += s_hits

        # Collect MicroSpec dynamic tail on the prefix (or whole prompt for short docs)
        cut = max(1, int(n * split_ratio)) if n >= 8 else n
        dyn: set[int] = set()
        for t in ids[:cut]:
            if t not in core_set and t not in dyn and len(dyn) < max_dynamic:
                dyn.add(t)
        dyn_counts.append(len(dyn))

        c_slice = ids[cut:] if cut < n else ids
        cn = len(c_slice)
        cs_hits = sum(1 for t in c_slice if t in core_set)
        cm_hits = sum(1 for t in c_slice if (t in core_set or t in dyn))
        cont_toks += cn
        cont_static_hits += cs_hits
        cont_ms_hits += cm_hits

        if len(docs) <= 16:
            per_doc.append({
                "label": label,
                "n_tokens": n,
                "static_coverage": round(s_hits / n, 6),
                "continuation_tokens": cn,
                "cont_static_coverage": round(cs_hits / cn, 6) if cn else 1.0,
                "cont_microspec_coverage": round(cm_hits / cn, 6) if cn else 1.0,
                "dyn_slots_used": len(dyn),
            })

    return {
        "n_docs": len(dyn_counts),
        "total_tokens": total_toks,
        "full_static_coverage": round(static_hits / total_toks, 6) if total_toks else 0.0,
        "cont_tokens": cont_toks,
        "cont_static_coverage": round(cont_static_hits / cont_toks, 6) if cont_toks else 0.0,
        "cont_microspec_coverage": round(cont_ms_hits / cont_toks, 6) if cont_toks else 0.0,
        "avg_dyn_slots_used": round(sum(dyn_counts) / len(dyn_counts), 2) if dyn_counts else 0.0,
        "max_dyn_slots_used": max(dyn_counts) if dyn_counts else 0,
        "per_doc": per_doc,
    }


def verify_manifest_c_contract(manifest_path: Path, expected_vocab: int) -> None:
    """Mirror src/model.c:1983 ie_load_vocab_shortlist() validation checks strictly."""
    raw = manifest_path.read_bytes()
    if len(raw) > (1 << 20):
        raise AssertionError("Manifest exceeds 1 MiB C loader cap")
    m = json.loads(raw.decode("utf-8"))
    assert m["schema"] == "rdna4.mtp.vocab-shortlist.v1", f"Bad schema {m['schema']}"
    assert int(m["vocabulary"]) == expected_vocab, f"Vocab mismatch {m['vocabulary']} != {expected_vocab}"
    size = int(m["size"])
    assert 1 <= size <= expected_vocab
    ids_path = Path(m["ids_text"])
    if not ids_path.exists():
        ids_path = manifest_path.parent / ids_path.name
    ids_bytes = ids_path.read_bytes()
    got_sha = hashlib.sha256(ids_bytes).hexdigest()
    assert got_sha == m["ids_text_sha256"].lower(), f"SHA mismatch {got_sha} != {m['ids_text_sha256']}"
    ids = [int(x) for x in ids_bytes.decode("ascii").split()]
    assert len(ids) == size, f"ID count {len(ids)} != size {size}"
    for i in range(1, len(ids)):
        assert ids[i] > ids[i - 1], f"IDs not strictly ascending at {i}: {ids[i-1]} >= {ids[i]}"
    assert ids[-1] < expected_vocab
    id_set = set(ids)
    for eog in m["required_eog_ids"]:
        assert int(eog) in id_set, f"Missing required_eog_id {eog}"


def write_artifacts(
    out_dir: Path,
    tok: InklingTokenizer,
    model_path: Path,
    ids_sorted: List[int],
    tier_counts: Dict[str, int],
    required_eog: List[int],
    max_dynamic: int,
    cal_stats: Dict[str, object],
    heldout_stats: Optional[Dict[str, object]],
    probe_stats: Optional[Dict[str, object]],
    corpus_paths: Sequence[Path],
) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    core_size = len(ids_sorted)
    stem = f"inkling2-frspec-{core_size}"
    ids_txt_path = out_dir / f"{stem}.ids.txt"
    ids_bin_path = out_dir / f"{stem}.bin"
    manifest_path = out_dir / f"{stem}.manifest.json"

    # 1. ids.txt (LF-terminated, exact C loader format)
    txt_payload = ("\n".join(str(x) for x in ids_sorted) + "\n").encode("ascii")
    tmp_txt = ids_txt_path.with_suffix(".txt.tmp")
    tmp_txt.write_bytes(txt_payload)
    os.replace(tmp_txt, ids_txt_path)
    txt_sha = hashlib.sha256(txt_payload).hexdigest()

    # 2. Compact binary (32-byte header + u32[core_size] LE)
    # Header: magic(4B)="IKSL", version(u32)=1, vocab_padded(u32), vocab_unpadded(u32),
    #         core_size(u32), max_dynamic(u32), flags(u32)=1(ascending), reserved(u32)=0
    header = struct.pack(
        "<4s7I",
        b"IKSL",
        1,
        tok.vocab_padded,
        tok.vocab_unpadded,
        core_size,
        max_dynamic,
        1,
        0,
    )
    bin_payload = header + struct.pack(f"<{core_size}I", *ids_sorted)
    tmp_bin = ids_bin_path.with_suffix(".bin.tmp")
    tmp_bin.write_bytes(bin_payload)
    os.replace(tmp_bin, ids_bin_path)
    bin_sha = hashlib.sha256(bin_payload).hexdigest()

    # 3. Manifest (rdna4.mtp.vocab-shortlist.v1 compatible with src/model.c:1983!)
    corpus_hashes = [
        {"path": str(p.resolve()), "sha256": _sha256_file(p) if p.is_file() else "directory"}
        for p in corpus_paths
    ]
    manifest = {
        "schema": "rdna4.mtp.vocab-shortlist.v1",
        "vocabulary": tok.vocab_padded,
        "vocab_unpadded": tok.vocab_unpadded,
        "size": core_size,
        "max_dynamic_slots": max_dynamic,
        "total_id_map_slots": core_size + max_dynamic,
        "ids_text": str(ids_txt_path.resolve()),
        "ids_text_sha256": txt_sha,
        "ids_bin": str(ids_bin_path.resolve()),
        "ids_bin_sha256": bin_sha,
        "required_eog_ids": required_eog,
        "target_padding_id": tok.pad_id,
        "tokenizer_pre": tok.pre,
        "tokenizer_fingerprint": tok.fingerprint,
        "target_model_file": str(model_path.resolve()),
        "source_policy": {
            "scope": "independent_calibration_corpus_only",
            "no_benchmark_or_probe_prompts_in_counts": True,
            "heldout_strictly_evaluation_only": True,
        },
        "calibration_sources": corpus_hashes,
        "tier_breakdown": tier_counts,
        "calibration_stats": cal_stats,
        "heldout_eval": heldout_stats,
        "probe_eval": probe_stats,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    tmp_man = manifest_path.with_suffix(".json.tmp")
    tmp_man.write_bytes(manifest_bytes)
    os.replace(tmp_man, manifest_path)

    verify_manifest_c_contract(manifest_path, tok.vocab_padded)
    return ids_txt_path, ids_bin_path, manifest_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build certified 60K+4K FR-Spec/MicroSpec shortlist for Inkling-2-Mini (CPU-only).")
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("models/Inkling-2-Mini-Uncensored-IQ3_XXS-fast.gguf"),
        help="Path to Inkling-2-Mini GGUF (metadata-only read).",
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=None,
        help="Calibration corpus file (.jsonl/.txt) or directory (repeatable; default: data/mtp-vocab-pilot/corpus.train.jsonl).",
    )
    ap.add_argument(
        "--heldout",
        type=Path,
        default=Path("data/mtp-vocab-pilot/corpus.heldout.jsonl"),
        help="Independent held-out corpus for static vs MicroSpec coverage evaluation (never counted).",
    )
    ap.add_argument(
        "--probe-prompts",
        type=Path,
        nargs="*",
        default=[
            Path("research/inkling2/prompts/real-code-233.txt"),
            Path("research/inkling2/prompts/real-code-4k.txt"),
        ],
        help="Read-only benchmark prompts for tokenizer count verification and coverage reporting (never counted).",
    )
    ap.add_argument("--core-size", type=int, default=61440, help="Static FR-Spec core shortlist size (default: 61440 = 60K).")
    ap.add_argument("--max-dynamic", type=int, default=4096, help="MicroSpec dynamic prompt-union tail slots (default: 4096 = 4K).")
    ap.add_argument("--out-dir", type=Path, default=Path("data/inkling2-shortlist"), help="Output directory for .ids.txt, .bin, .manifest.json.")
    ap.add_argument("--max-docs", type=int, default=0, help="Optional cap on calibration documents (0 = all).")
    ap.add_argument("--max-bytes", type=int, default=0, help="Optional cap on calibration UTF-8 bytes (0 = all).")
    args = ap.parse_args(argv)

    corpus_paths = args.corpus if args.corpus else [Path("data/mtp-vocab-pilot/corpus.train.jsonl")]
    t0 = time.perf_counter()
    print(f"[shortlist] loading GGUF tokenizer metadata from {args.model} ...")
    tok = InklingTokenizer(args.model)
    print(
        f"[shortlist] pre={tok.pre} vocab={tok.vocab_padded} unpadded={tok.vocab_unpadded} "
        f"merges={len(tok.merges_list)} eos={tok.eos_id} pad={tok.pad_id} ({time.perf_counter() - t0:.2f}s)"
    )

    t1 = time.perf_counter()
    counts: collections.Counter = collections.Counter()
    n_docs = 0
    total_occ = 0
    for _lbl, txt in _iter_corpus_texts(corpus_paths, max_docs=args.max_docs, max_bytes=args.max_bytes):
        ids = tok.encode(txt, parse_control=True)
        counts.update(ids)
        total_occ += len(ids)
        n_docs += 1
    dt_cal = time.perf_counter() - t1
    print(
        f"[shortlist] calibrated on {n_docs} docs, {total_occ:,} tokens, "
        f"{len(counts):,} unique IDs in {dt_cal:.2f}s ({total_occ / max(dt_cal, 1e-6):,.0f} tok/s)"
    )

    ids_sorted, tier_counts, required_eog = build_shortlist(tok, counts, args.core_size)
    core_set = set(ids_sorted)
    cal_hits = sum(cnt for tid, cnt in counts.items() if tid in core_set)
    cal_stats = {
        "docs": n_docs,
        "token_occurrences": total_occ,
        "observed_unique_support": len(counts),
        "calibration_occurrence_coverage": round(cal_hits / total_occ, 6) if total_occ else 0.0,
        "high_id_tokens_gt_32768_in_core": sum(1 for tid in ids_sorted if tid >= 32768),
        "max_id_in_core": ids_sorted[-1],
    }

    heldout_stats = None
    if args.heldout and args.heldout.exists():
        t2 = time.perf_counter()
        h_docs = list(_iter_corpus_texts([args.heldout]))
        heldout_stats = evaluate_microspec_coverage(tok, core_set, h_docs, max_dynamic=args.max_dynamic)
        print(
            f"[shortlist] heldout ({heldout_stats['n_docs']} docs, {heldout_stats['total_tokens']:,} tok, {time.perf_counter()-t2:.2f}s): "
            f"full_static={heldout_stats['full_static_coverage']*100:.3f}%, "
            f"cont_static={heldout_stats['cont_static_coverage']*100:.3f}% -> "
            f"cont_microspec(60K+4K)={heldout_stats['cont_microspec_coverage']*100:.3f}% "
            f"(avg_dyn={heldout_stats['avg_dyn_slots_used']}, max_dyn={heldout_stats['max_dyn_slots_used']})"
        )

    probe_stats = None
    valid_probes = [p for p in (args.probe_prompts or []) if p.exists()]
    if valid_probes:
        p_docs = [(p.name, p.read_text(encoding="utf-8", errors="replace")) for p in valid_probes]
        probe_stats = evaluate_microspec_coverage(tok, core_set, p_docs, max_dynamic=args.max_dynamic)
        for d in probe_stats["per_doc"]:  # type: ignore[index]
            print(
                f"[shortlist] probe {d['label']}: n_tok={d['n_tokens']}, "
                f"static={d['static_coverage']*100:.3f}%, "
                f"cont_static={d['cont_static_coverage']*100:.3f}% -> "
                f"cont_microspec={d['cont_microspec_coverage']*100:.3f}% (dyn={d['dyn_slots_used']})"
            )

    ids_txt, ids_bin, man = write_artifacts(
        args.out_dir,
        tok,
        args.model,
        ids_sorted,
        tier_counts,
        required_eog,
        args.max_dynamic,
        cal_stats,
        heldout_stats,
        probe_stats,
        corpus_paths,
    )
    print(f"[shortlist] VERIFIED C-contract & wrote:")
    print(f"  ids.txt  : {ids_txt} ({ids_txt.stat().st_size:,} B)")
    print(f"  ids.bin  : {ids_bin} ({ids_bin.stat().st_size:,} B)")
    print(f"  manifest : {man} ({man.stat().st_size:,} B)")
    print(f"[shortlist] tier_breakdown: {tier_counts}")
    print(f"[shortlist] high_id (>32768) tokens rescued into 60K core: {cal_stats['high_id_tokens_gt_32768_in_core']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
