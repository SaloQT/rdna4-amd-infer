"""Registers, LDS and scratch of the kernels in a hsaco (llvm-readelf --notes), and the waves per SIMD that the
registers allow on gfx1201 (1536 VGPRs per SIMD lane in wave32, granule 8... the usual 16 waves at <= 96).
usage: python tools/kernel_regs.py FILE.hsaco [name-substring ...]
The LLVM tools come from $LLVM_BIN, or from the .rocm-venv next to engine/ (as in the build scripts)."""
import os
import re
import subprocess
import sys

llvm_bin = os.environ.get('LLVM_BIN') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '.rocm-venv',
                                                      'Lib', 'site-packages', '_rocm_sdk_core', 'lib', 'llvm', 'bin')
llvm = os.path.join(llvm_bin, 'llvm-readelf.exe')
t = subprocess.run([llvm, '--notes', sys.argv[1]], capture_output=True, text=True, errors='replace').stdout
pats = sys.argv[2:]
rows = []
for blk in re.split(r'\n  - \.args:', t):
    m = re.search(r'\n    \.name:\s+(\S+)', blk)
    if not m:
        continue
    name = m.group(1)
    if pats and not any(p in name for p in pats):
        continue

    def g(key):
        v = re.search(r'\n    \.' + key + r':\s+(\d+)', blk)
        return int(v.group(1)) if v else -1
    v = g('vgpr_count')
    waves = min(16, 1536 // (((v + 7) // 8) * 8)) if v > 0 else 0  # wave32: 1536 VGPRs per lane slot... approximate
    rows.append((name, v, g('sgpr_count'), g('group_segment_fixed_size'), g('private_segment_fixed_size'), g('vgpr_spill_count'), waves))
for r in sorted(rows):
    print('%-34s vgpr %4d sgpr %3d lds %6d scratch %4d spill %3d  ~waves/SIMD %d' % r)
