#!/bin/sh
# Stand-in for build-win.sh (run from engine/). Mode file build.mode next to this script:
# ok | fail (always) | failpatch (only when the test patch is applied). Logs each call to build.calls.
here="${RSI_TEST_DIR:?set by run_tests.mjs}"  # the test work directory: mode files, call logs
mode=ok
[ -f "$here/build.mode" ] && mode=$(cat "$here/build.mode")
patched=no
grep -q RSI-TEST-MARKER kernels/ie_inkling_r4_attn.inc && patched=yes
echo "$mode patched=$patched" >> "$here/build.calls"
[ "$mode" = fail ] && exit 1
[ "$mode" = failpatch ] && [ "$patched" = yes ] && exit 1
touch build/ie-run.exe build/ie_kernels.hsaco build/ie_inkling_kernels.hsaco
exit 0
