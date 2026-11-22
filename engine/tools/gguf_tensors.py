"""List tensors of a GGUF file (header only; reads a few MB, no GPU).

  python -I tools/gguf_tensors.py FILE [--sha] [substring ...]
--sha: also the sha256 of each listed tensor's data (types known to BLK only)
"""
import struct
import sys

TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K',
         13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS', 19: 'IQ1_S', 20: 'IQ4_NL',
         21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32', 27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16'}
SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def rd(f, fmt):
    n = struct.calcsize('<' + fmt)
    return struct.unpack('<' + fmt, f.read(n))[0]


def rstr(f):
    return f.read(rd(f, 'Q')).decode('utf-8', 'replace')


def rval(f, t):
    if t in SCALAR:
        return rd(f, SCALAR[t])
    if t == 8:
        return rstr(f)
    if t == 9:
        et, n = rd(f, 'I'), rd(f, 'Q')
        if et in SCALAR:
            f.seek(n * struct.calcsize('<' + SCALAR[et]), 1)
            return '[%d x type %d]' % (n, et)
        return [rval(f, et) for _ in range(n)]
    raise ValueError(t)


path, pats = sys.argv[1], [a for a in sys.argv[2:] if a != '--sha']
with open(path, 'rb') as f:
    assert f.read(4) == b'GGUF'
    ver, nt, nkv = rd(f, 'I'), rd(f, 'Q'), rd(f, 'Q')
    kv = {}
    for _ in range(nkv):
        k, t = rstr(f), rd(f, 'I')
        kv[k] = rval(f, t)
    for k, v in kv.items():
        if k.startswith('general.') or 'block_count' in k or 'nextn' in k or 'embedding_length' in k:
            print('kv', k, v)
    tens = []
    for _ in range(nt):
        name, nd = rstr(f), rd(f, 'I')
        dims = [rd(f, 'Q') for _ in range(nd)]
        ty, off = rd(f, 'I'), rd(f, 'Q')
        tens.append((name, ty, dims, off))
    align = kv.get('general.alignment', 32)
    data0 = (f.tell() + align - 1) // align * align
    # (block elements, block bytes) of the types a draft head can have
    BLK = {'F32': (1, 4), 'F16': (1, 2), 'Q4_0': (32, 18), 'Q8_0': (32, 34), 'IQ3_S': (256, 110), 'IQ1_M': (256, 56),
           'Q6_K': (256, 210), 'Q4_K': (256, 144)}
    for name, ty, dims, off in tens:
        if not pats or any(p in name for p in pats):
            tn = TYPES.get(ty, ty)
            n = 1
            for d in dims:
                n *= d
            size = n // BLK[tn][0] * BLK[tn][1] if tn in BLK else 0
            line = '%-48s %-8s %s  offset %d  %d bytes' % (name, tn, dims, data0 + off, size)
            if '--sha' in sys.argv and size:
                import hashlib
                h = hashlib.sha256()
                f.seek(data0 + off)
                left = size
                while left:
                    b = f.read(min(left, 1 << 24))
                    h.update(b)
                    left -= len(b)
                line += '  sha256 ' + h.hexdigest()
            print(line)
