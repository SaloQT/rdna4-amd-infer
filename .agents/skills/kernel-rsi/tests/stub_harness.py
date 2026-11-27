"""Stand-in for the benchmark harness: prints the summary lines rsi.mjs parses.
Mode (file gate.mode next to this script): match | patchbad (mismatch when the test patch is applied) | allbad | missing."""
import os
import sys

here = os.environ['RSI_TEST_DIR']  # the test work directory (run_tests.mjs sets it): mode files, call logs
mode = open(os.path.join(here, 'gate.mode')).read().strip() if os.path.exists(os.path.join(here, 'gate.mode')) else 'match'
with open(os.path.join(here, 'gate.calls'), 'a') as f:
    f.write(mode + '\n')
exe = sys.argv[sys.argv.index('--exe') + 1]
inc = os.path.join(os.path.dirname(os.path.dirname(exe)), 'kernels', 'ie_inkling_r4_attn.inc')
patched = 'RSI-TEST-MARKER' in open(inc, encoding='utf-8').read()
calls = open(os.path.join(here, 'gate.calls')).read().count('\n')
bad = mode == 'allbad' or (mode == 'patchbad' and patched) or (mode == 'flaky-once' and calls == 1)
t1 = 'bead35841aaeebcc'
ds4 = 'ffffffffffffffff' if bad else '0043be349ae4d4a8'
speed = 100.0 if mode == 'slow' else 129.0
print('  Parity       : X')
print('  Plain T=1    : %.2f tok/s  (7.700 ms/tok)  sha=%s' % (speed, t1))
print('  DS5 233      : 320.00 tok/s  (6.68 tok/step)  sha=9183fc7067515952')
print('  DS5 4K       : decode 170.00 tok/s (3.71 t/s) | prefill 6000.0 tok/s  sha=%s' % ds4)
if mode != 'missing':
    print('  DS5 16K      : decode 170.00 tok/s (4.43 t/s) | prefill 5600.0 tok/s  sha=97545643da8d725a')
