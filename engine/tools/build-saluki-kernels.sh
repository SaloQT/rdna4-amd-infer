#!/bin/sh
# Build the optional Saluki verify kernels without replacing the base KV module.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
LLVM_BIN=${LLVM_BIN:-$ROOT/../.rocm-venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin}
CLANG=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/amdclang.exe" || echo "$LLVM_BIN/amdclang.exe")
LINK=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/ld.lld.exe" || echo "$LLVM_BIN/ld.lld.exe")
OUT=${1:-build/ie_saluki_kernels.hsaco}
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "$(dirname "$OUT")"
"$CLANG" -x c -std=c11 --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O3 -Ibuild "$@" \
  -S -emit-llvm -o "$OUT.ll.tmp" kernels/ie_saluki_kernels.c
sed -E '/^attributes #[0-9]+ = \{.*"target-cpu"/ s/"target-cpu"/"amdgpu-flat-work-group-size"="256,256" "target-cpu"/' \
  "$OUT.ll.tmp" > "$OUT.ll"
"$CLANG" -nogpulib --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -O3 -c -o "$OUT.o" "$OUT.ll"
"$LINK" -shared -o "$OUT" "$OUT.o"
rm -f "$OUT.ll.tmp" "$OUT.ll" "$OUT.o"
