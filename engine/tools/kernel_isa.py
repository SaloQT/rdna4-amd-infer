"""The loops of one kernel in a hsaco (llvm-objdump): each backward branch's body with its instruction count by kind
(dot4, other VALU, LDS, global loads, scalar, waits). Lists the loops, longest first.
usage: python tools/kernel_isa.py FILE.hsaco KERNEL [--dump N] (--dump: print loop N's instructions)"""
import collections
import os
import re
import subprocess
import sys

llvm = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.rocm-venv', 'Lib', 'site-packages',
                    '_rocm_sdk_core', 'lib', 'llvm', 'bin', 'llvm-objdump.exe')
args = sys.argv[1:]
dump = None
if '--dump' in args:
    i = args.index('--dump')
    dump = int(args[i + 1])
    del args[i:i + 2]
hs, kern = args[0], args[1]
t = subprocess.run([llvm, '-d', '--mcpu=gfx1201', '--disassemble-symbols=' + kern, hs], capture_output=True, text=True,
                   errors='replace').stdout
ins = []  # (addr, mnemonic, text)
for line in t.splitlines():
    m = re.match(r'\s+([a-z_0-9]+)\b(.*?)//\s*([0-9A-Fa-f]+):', line)
    if m:
        ins.append((int(m.group(3), 16), m.group(1), line.strip()))
addr_i = {a: i for i, (a, _, _) in enumerate(ins)}


def kind(mn):
    if 'dot4' in mn or 'dot8' in mn:
        return 'dot'
    if mn.startswith('ds_'):
        return 'lds'
    if mn.startswith(('global_load', 'buffer_load', 'flat_load')):
        return 'gload'
    if mn.startswith(('global_store', 'buffer_store')):
        return 'gstore'
    if mn.startswith('s_wait'):
        return 'wait'
    if mn.startswith('s_'):
        return 'salu'
    if mn.startswith('v_'):
        return 'valu'
    return 'other'


loops = []
for i, (a, mn, txt) in enumerate(ins):
    if mn.startswith('s_cbranch') or mn == 's_branch':
        m = re.search(r'(?:<[^>]*>|\b)0x([0-9a-f]+)|\+?(-?\d+)\s*$', txt)
        tm = re.search(r'//\s*[0-9A-Fa-f]+:.*', txt)
        tgt = re.search(r'\s([0-9a-fx]+)\s*<', txt) or re.search(r'(-?\d+)\s*//', txt)
        # objdump prints "s_cbranch_scc1 65494 // 000000001234: ..." (signed word offset)
        mo = re.match(r'\S+\s+(-?\d+)', txt)
        if not mo:
            continue
        off = int(mo.group(1))
        if off > 32767:
            off -= 65536
        dst = a + 4 + 4 * off
        if dst <= a and dst in addr_i:
            loops.append((addr_i[dst], i))
res = []
for n, (s, e) in enumerate(loops):
    c = collections.Counter(kind(mn) for _, mn, _ in ins[s:e + 1])
    res.append((e - s + 1, n, s, e, c))
res.sort(reverse=True)
for ln, n, s, e, c in res:
    print('loop %d: %d instructions at 0x%x  %s' % (n, ln, ins[s][0], ' '.join('%s %d' % kv for kv in sorted(c.items()))))
if dump is not None:
    s, e = loops[dump]
    vc = collections.Counter(mn for _, mn, _ in ins[s:e + 1])
    for _, _, txt in ins[s:e + 1]:
        print(txt.split('//')[0].rstrip())
    print('--- by mnemonic:', ', '.join('%s %d' % kv for kv in vc.most_common()))
