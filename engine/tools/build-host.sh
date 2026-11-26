#!/bin/sh
# Rebuild only the host programs (no GPU code objects): build/ie-run.exe, build/ie-serve.exe, or both.
#   sh tools/build-host.sh [run|serve|all]     (default: all)
# The compile lines are those of build-win.sh; use that for the kernels and the tests.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
B=build
w() { cygpath -w "$1" 2>/dev/null || echo "$1"; }
CC_WIN=${CC:-$(w "$(command -v gcc)")}
HOST_CFLAGS="-std=c11 -O3 -march=native -Wall -Wextra -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -Wno-unused-function -I$B"
COMMON="src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c prof/sm_profiler.c"
what=${1:-all}
if [ "$what" = run ] || [ "$what" = all ]; then
  echo "== $B/ie-run.exe"
  "$CC_WIN" $HOST_CFLAGS -o "$B/ie-run.exe" $COMMON src/main.c -lm
fi
if [ "$what" = serve ] || [ "$what" = all ]; then
  echo "== $B/ie-serve.exe"
  "$CC_WIN" $HOST_CFLAGS -o "$B/ie-serve.exe" $COMMON src/tok.c src/json.c src/chat.c src/vision.c src/serve.c -lm -lws2_32 -lpthread -lole32 -lwindowscodecs -luuid
fi
