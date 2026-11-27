#!/bin/sh
# End-to-end check of kernel-rsi on the real repository and GPU: claim (preflight, discovery when the build changed, loop
# setup), a comment-only eval (expected: no_gain, reverted), an eval that edits core/ (expected: out_of_scope), status,
# release. It uses the GPU (one engine or kbench job at a time) and may rebuild engine/build. It keeps its loops,
# captures and queue in its own runs directory (%TEMP%/kernel-rsi-e2e-runs, discovered on the first run), so it never
# adds attempts to the real loops; its GPU jobs wait while a worker's kbench or engine is running.
#   sh .agents/skills/kernel-rsi/tests/e2e.sh [SCENARIO]      (default vb16k_t6)
set -u
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$ROOT"
TMPD=$(node -e "console.log(require('os').tmpdir().replace(/\\\\/g, '/'))")
export RSI_CONFIG="$TMPD/kernel-rsi-e2e.json"
node -e "require('fs').writeFileSync(process.env.RSI_CONFIG, JSON.stringify({ runsDir: '$TMPD/kernel-rsi-e2e-runs' }))"
RSI="node .agents/skills/kernel-rsi/scripts/rsi.mjs"
SC=${1:-vb16k_t6}
W=e2e-test
echo "== claim $(date +%T)"
OUT=$($RSI claim "$SC" --worker $W 2>&1); echo "$OUT"
K=$(echo "$OUT" | sed -n 's/^claimed \([^ ]*\) for .*/\1/p')
WK=$(echo "$OUT" | sed -n 's/^work copy: \(.*\)\/kernels$/\1/p')
[ -z "$K" ] && { echo "nothing claimed"; exit 1; }
echo "claimed $K, work copy $WK"
# the file that defines the kernel (an edit there rebuilds only what includes it); else any of its unit's .inc files
F=$(cd "$WK/kernels" && grep -l "KERNEL $K(" *.inc 2>/dev/null | head -1)
[ -z "$F" ] && F=$(ls "$WK/kernels" | grep -m1 '^ie_inkling_.*\.inc$')
echo "== eval 1: a comment appended to $F (expect NO_GAIN) $(date +%T)"
printf '/* e2e comment */\n' >> "$WK/kernels/$F"
$RSI eval "$K" --idea "e2e test: a comment only" --tag schedule --worker $W; echo "exit $?"
echo "== eval 2: an edit to core/ (expect OUT_OF_SCOPE) $(date +%T)"
printf '/* e2e */\n' >> "$WK/core/ie_core.h"
$RSI eval "$K" --idea "e2e test: edit core" --tag alu --worker $W; echo "exit $?"
echo "== status $(date +%T)"
$RSI status "$K" | head -20
echo "== release"
$RSI release "$K" --worker $W; echo "exit $?"
echo "== done $(date +%T)"
