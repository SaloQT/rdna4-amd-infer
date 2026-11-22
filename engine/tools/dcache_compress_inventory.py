"""Inventory: which checkpoints share token prefixes (same conversation), which k blocks each uses."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dcache_compress_common as C

M = (1 << 64) - 1


def fnv64(h, data):
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & M
    return h


fs = C.list_files()
es = sorted([f for f in fs if f["type"] == "e"], key=lambda f: (f["pos"], f["path"]))
ks = {os.path.basename(f["path"]).split("-")[2][:16]: f for f in fs if f["type"] == "k"}
print(f"{len([f for f in fs if f['type']=='e'])} e files, {len(ks)} k files")
sig = es[0]["sig"]
toks = {}
for f in es:
    toks[f["path"]] = np.fromfile(f["path"], dtype=np.uint32, count=f["pos"], offset=56)
base = max(es, key=lambda f: f["pos"])
print("longest:", os.path.basename(base["path"]), base["pos"])
for f in es:
    t = toks[f["path"]]
    n = min(len(t), len(toks[base["path"]]))
    eq = np.nonzero(t[:n] != toks[base["path"]][:n])[0]
    common = int(eq[0]) if len(eq) else n
    nb = f["pos"] // 1024
    h = sig
    chain = []
    for i in range(nb):
        h = fnv64(h, t[i * 1024:(i + 1) * 1024].tobytes())
        chain.append(f"{h:016x}")
    have = sum(c in ks for c in chain)
    print(f"{os.path.basename(f['path'])[-20:-4]} pos={f['pos']:6d} kind={f['kind']} common-prefix-with-longest={common:6d} blocks={nb} on-disk={have}")
used = set()
for f in es:
    t = toks[f["path"]]
    h = sig
    for i in range(f["pos"] // 1024):
        h = fnv64(h, t[i * 1024:(i + 1) * 1024].tobytes())
        used.add(f"{h:016x}")
print("k files referenced by some e file:", len(used & set(ks)), "of", len(ks))
