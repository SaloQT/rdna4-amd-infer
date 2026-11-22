#!/bin/sh
# Build the optional Inkling-2 f16-KV kernels without replacing the base module.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
LLVM_BIN=${LLVM_BIN:-$ROOT/../.rocm-venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin}
CLANG=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/amdclang.exe" || echo "$LLVM_BIN/amdclang.exe")
LINK=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/ld.lld.exe" || echo "$LLVM_BIN/ld.lld.exe")
READELF=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/llvm-readelf.exe" || echo "$LLVM_BIN/llvm-readelf.exe")
OUT=${1:-build/ie_inkling_kernels.hsaco}
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "$(dirname "$OUT")"
"$CLANG" -x c -std=c11 -Wall -Wextra --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O3 -Ibuild "$@" \
  -S -emit-llvm -o "$OUT.ll.tmp" kernels/ie_inkling_kernels.c
sed -E '/^attributes #[0-9]+ = \{.*"target-cpu"/ s/"target-cpu"/"amdgpu-flat-work-group-size"="256,256" "target-cpu"/' \
  "$OUT.ll.tmp" > "$OUT.ll"
"$CLANG" -nogpulib --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -O3 -c -o "$OUT.o" "$OUT.ll"
"$LINK" -shared -o "$OUT" "$OUT.o"
rm -f "$OUT.ll.tmp" "$OUT.ll" "$OUT.o"
if [ -x "$LLVM_BIN/llvm-readelf.exe" ]; then
  echo "== $OUT kernel resources (VGPR / SGPR / spill / LDS)"
  "$READELF" --notes "$OUT" | awk '
    /\.name: +ie_inkling_/ { name=$2 }
    /\.group_segment_fixed_size:/ { lds=$2 }
    /\.private_segment_fixed_size:/ { priv=$2 }
    /\.sgpr_count:/ { sgpr=$2 }
    /\.vgpr_count:/ { vgpr=$2 }
    /\.vgpr_spill_count:/ { vsp=$2 }
    /\.sgpr_spill_count:/ { ssp=$2 }
    /\.wavefront_size:/ {
      if (name != "") {
        printf "   %-36s VGPR=%3d SGPR=%3d spill=%d/%d priv=%d LDS=%d\n", name, vgpr, sgpr, vsp, ssp, priv, lds
        name=""
      }
    }'
fi
