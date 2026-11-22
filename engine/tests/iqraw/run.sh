#!/bin/sh
# run.sh -- the ie_iqraw.inc harness: slice real blocks, build, run.
#
#   sh tests/iqraw/run.sh [MODEL.gguf] [LLAMA.CPP dir]
#
# The oracle is ggml's ggml-quants.c, compiled on its own with contraction off
# (so the float results do not depend on whether the target fuses a*b+c), and
# linked against tests/iqraw/iqraw_test.c + src/common.c (the engine's exact
# half -> float). Prints the dequant / dot pass counts and the max abs/rel
# error; exits non-zero on any failure.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../../.." && pwd)   # repo root: ref/hanw-amd.c/tests/iqraw
model=${1:-$root/models/Underdog-Saluki-27B-1.0-IQ2-mix.gguf}
llama=${2:-$root/llama.cpp}
out=$here/build
cc=${CC:-clang}

mkdir -p "$out"
${PYTHON:-python} "$here/slice_iqraw.py" --model "$model" --out "$out"

"$cc" -std=c11 -O1 -ffp-contract=off -I"$llama/ggml/src" -I"$llama/ggml/include" \
      -c "$llama/ggml/src/ggml-quants.c" -o "$out/ggml-quants.o"
# the engine's exact half -> float (src/common.c), built on its own so that its
# own include-only noise does not mask warnings of the harness
"$cc" -std=c11 -O2 -c "$here/../../src/common.c" -o "$out/common.o"
"$cc" -std=c11 -O2 -Wall -Wextra -ffp-contract=off \
      -o "$out/iqraw_test.exe" "$here/iqraw_test.c" "$out/ggml-quants.o" "$out/common.o" -lm
"$out/iqraw_test.exe" "$out"

# the amdgcn build of the .inc on its own (the ticket's command, on a TU that
# defines IE_RQP/IE_RQF/IE_RQU as address space 1 pointers): no error, and the
# kernel's local (private segment) size must stay 0
amdclang=${AMDCLANG:-$root/.rocm-venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin/amdclang.exe}
if [ -x "$amdclang" ]; then
  "$amdclang" -x c -std=c11 -Wall -Wextra --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O3 \
      -S -emit-llvm "$here/device_tu.c" -o "$out/ie_iqraw_tu.ll"
  "$amdclang" -x c -std=c11 -Wall -Wextra --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O3 \
      -S "$here/device_tu.c" -o "$out/ie_iqraw_tu.s"
  echo "== (c) amdgcn build of the .inc (gfx1201): see $out/ie_iqraw_tu.ll / .s"
  sed -n '/\.amdhsa_kernel ie_rq_test/,/amdhsa_next_free_sgpr/p' "$out/ie_iqraw_tu.s" |
      grep -E "private_segment_fixed_size|group_segment_fixed_size|next_free_vgpr|next_free_sgpr" |
      sed 's/^[[:space:]]*/   /'
  priv=$(sed -n '/\.amdhsa_kernel ie_rq_test/,/amdhsa_next_free_sgpr/p' "$out/ie_iqraw_tu.s" |
      awk '/amdhsa_private_segment_fixed_size/{print $2}')
  if [ "$priv" != "0" ]; then
    echo "   FAIL: the dot kernel uses $priv bytes of local memory"
    exit 1
  fi
else
  echo "note: $amdclang not found, skipping the amdgcn build"
fi
