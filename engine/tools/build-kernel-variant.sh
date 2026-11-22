#!/bin/sh
# Build only the separate-launch kernels (ie_kernels.hsaco, the q8_0-KV ie_kernels_q8.hsaco and the 4-bit-KV
# ie_kernels_q4.hsaco) with extra -D
# settings into build/var/NAME/, for timing kernel variants with --hsaco build/var/NAME/ie_kernels.hsaco without
# rebuilding the engine. Run build-win.sh first (the kernels include build/ie_mk_gen.h).
#   sh tools/build-kernel-variant.sh NAME [-DIE_TR8_GRP9=5 ...]
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
NAME=$1
shift
B=build
O=$B/var/$NAME
ARCH=gfx1201
LLVM_BIN=${LLVM_BIN:-$ROOT/../../.rocm-venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin}
CLANG=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/amdclang.exe" || echo "$LLVM_BIN/amdclang.exe")
LINK=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/ld.lld.exe" || echo "$LLVM_BIN/ld.lld.exe")
GPU_CFLAGS="-x c -std=c11 -Wall -Wextra --target=amdgcn-amd-amdhsa -mcpu=$ARCH -nogpulib -O3"
SED_ATTR='/^attributes #[0-9]+ = \{.*"target-cpu"/ s/"target-cpu"/"amdgpu-flat-work-group-size"="256,256" "target-cpu"/'
mkdir -p "$O"
for v in base q8 q4; do
  KV=""
  SFX=""
  if [ "$v" = q8 ]; then KV="-DIE_KV_Q8"; SFX="_q8"; fi
  if [ "$v" = q4 ]; then KV="-DIE_KV_Q4"; SFX="_q4"; fi
  "$CLANG" $GPU_CFLAGS $KV "$@" -I$B -S -emit-llvm -o "$O/k$SFX.ll.tmp" kernels/ie_kernels.c
  sed -E -e "$SED_ATTR" "$O/k$SFX.ll.tmp" > "$O/k$SFX.ll"
  "$CLANG" -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$ARCH -O3 -c -o "$O/k$SFX.o" "$O/k$SFX.ll"
  "$LINK" -shared -o "$O/ie_kernels$SFX.hsaco" "$O/k$SFX.o"
  rm -f "$O/k$SFX.ll.tmp" "$O/k$SFX.ll" "$O/k$SFX.o"
done
echo "$O/ie_kernels.hsaco"
