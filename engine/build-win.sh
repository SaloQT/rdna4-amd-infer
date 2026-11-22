#!/bin/sh
# build-win.sh -- build amd-infer on Windows with no ROCm installed.
#
#   sh build-win.sh          all: build/ie-run.exe, build/ie-serve.exe and the five gfx1201 code
#                            objects (ie_kernels, ie_mk, ie_mk_mtp,
#                            ie_mk_flint, ie_mk_flintl) + the objdump check
#   sh build-win.sh test     all, then the CPU-only half of `make test`
#                            (tiny GGUFs, CPU backend vs numpy reference,
#                            kernel_test, ext_test, vocab_shortlist_test, cache_check, laws_test)
#
# Why not the toolchain on PATH: the llvm-mingw clang 23 there is built with
# only the X86 target.  It happily emits IR for amdgcn (`-S -emit-llvm`
# needs no code generator), but the object step dies with
#   unable to create target: 'No available targets are compatible with
#   triple "amdgcn-amd-amdhsa"'
# and llvm-objdump of the same package cannot disassemble the result
# ('cannot find target').  The ROCm wheel that setup.ps1 already installs
# for the HIP runtime ships a full AMD LLVM (amdclang 24, AMDGPU + lld +
# objdump): that is a user-space pip package, not a ROCm install, and
# -nogpulib keeps it from looking for device bitcode.  Override LLVM_BIN to
# use a system ROCm/LLVM instead.
#
# The only Windows specifics vs the Makefile:
#   * absolute Windows paths for every .exe (git-bash here refuses to exec
#     a PE given a POSIX or relative path: "command not found"), hence w();
#   * build/ie_mk_gen.h is generated from the Makefile's MK_* lists if absent.
#   * the megakernel is compiled with -DIE_MK_VGPR=184 (override IE_MK_VGPR=0
#     to let the compiler choose): RDNA4 gives 192 KiB of VGPR per SIMD, and a
#     wave of 32 lanes costs V*128 B, so 184 keeps 8 waves/SIMD (two
#     workgroups per CU) where amdclang 24's unconstrained 248 keeps 6.
set -eu

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

B=build
ARCH=gfx1201
LLVM_BIN=${LLVM_BIN:-$ROOT/../.rocm-venv/Lib/site-packages/_rocm_sdk_core/lib/llvm/bin}
CLANG=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/amdclang.exe" || echo "$LLVM_BIN/amdclang.exe")
LINK=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/ld.lld.exe" || echo "$LLVM_BIN/ld.lld.exe")
OBJDUMP=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/llvm-objdump.exe" || echo "$LLVM_BIN/llvm-objdump.exe")
READELF=$(command -v cygpath >/dev/null && cygpath -m "$LLVM_BIN/llvm-readelf.exe" || echo "$LLVM_BIN/llvm-readelf.exe")
# git-bash: exes are only exec'd with an absolute Windows path; CC is an
# absolute Windows path too (the PATH gcc is llvm-mingw's clang shim).
w() { cygpath -w "$1" 2>/dev/null || echo "$1"; }
CC_WIN=${CC:-$(w "$(command -v gcc)")}

GPU_CFLAGS="-x c -std=c11 -Wall -Wextra --target=amdgcn-amd-amdhsa -mcpu=$ARCH -nogpulib -O3"
# The attribute amdgpu_flat_work_group_size cannot be written in C, so the IR
# gets it (Makefile): every kernel runs with 256 threads.
SED_ATTR='/^attributes #[0-9]+ = \{.*"target-cpu"/ s/"target-cpu"/"amdgpu-flat-work-group-size"="256,256" "target-cpu"/'
HOST_CFLAGS="-std=c11 -O2 -g -Wall -Wextra -D_GNU_SOURCE -D_FILE_OFFSET_BITS=64 -Wno-unused-function -I$B"
HOST_SRC="src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c prof/sm_profiler.c src/main.c"

[ -x "$LLVM_BIN/amdclang.exe" ] || {
  echo "build-win.sh: no amdclang at $LLVM_BIN" >&2
  echo "  set LLVM_BIN to a directory holding amdclang/ld.lld/llvm-objdump" >&2
  exit 2
}
mkdir -p "$B"

echo "== amdclang: $("$CLANG" --version | head -1)"

# --- the megakernel's kernel table (tools/mk_gen.py, the Makefile's lists) ---
MK_DECODE="ie_embed_q8,ie_embed_q4,ie_embed_f32,ie_rope_kv,ie_gemv_q4q8,ie_rmsnorm,ie_gemv_q4kq8,ie_gemv_q6kq8,ie_gemv_q8q8,ie_gemv_f32,ie_swiglu,ie_gdn1,ie_qkn_rope_kv,ie_attn_split,ie_attn_kv,ie_argmax,ie_copy_rows,ie_gemv_raw_q2k,ie_gemv_raw_iq2xxs,ie_gemv_raw_iq2xs,ie_gemv_raw_iq3xxs,ie_gemv_raw_iq1s,ie_gemv_raw_iq2s,ie_gemv_raw_iq1m,ie_gemv_raw_iq3s,ie_gemv_raw_q2k_tr,ie_gemv_raw_iq2xxs_tr,ie_gemv_raw_iq2xs_tr,ie_gemv_raw_iq3xxs_tr,ie_gemv_raw_iq1s_tr,ie_gemv_raw_iq2s_tr,ie_gemv_raw_iq1m_tr,ie_gemv_raw_iq3s_tr,ie_gemv_iq4xsq8"
MK_FLINT="ie_embed_q8,ie_embed_q4,ie_embed_f32,ie_rmsnorm,ie_gemv_q4kq8,ie_gemv_q6kq8,ie_gemv_q8q8,ie_gemv_f32,ie_swiglu,ie_argmax,ie_copy_rows,ie_gemv_q5kq8,ie_gemv_iq4xsq8,ie_gexp,ie_moe_route,ie_moe_sum,ie_qkn_rope_kv2,ie_attn_sw"
MK_FLINTL="$MK_FLINT,ie_attn_gq"
MK_MTP="ie_embed_q8,ie_rmsnorm,ie_rmsnorm_t,ie_quant_q8,ie_swiglu,ie_gdn1,ie_qkn_rope_kv,ie_attn_split,ie_attn_split_t,ie_attn_kv,ie_argmax,ie_gemv_q4q8,ie_gemv_q4kq8,ie_gemv_q6kq8,ie_gemv_q8q8,ie_gemv_f32,ie_gemv_f32_t,ie_gemv_q4kq8_tr,ie_gemv_q6kq8_tr,ie_gemv_q8q8_tr,ie_copy_rows,ie_gemv_raw_q2k,ie_gemv_raw_iq2xxs,ie_gemv_raw_iq2xs,ie_gemv_raw_iq3xxs,ie_gemv_raw_iq1s,ie_gemv_raw_iq2s,ie_gemv_raw_iq1m,ie_gemv_raw_iq3s,ie_gemv_raw_q2k_tr,ie_gemv_raw_iq2xxs_tr,ie_gemv_raw_iq2xs_tr,ie_gemv_raw_iq3xxs_tr,ie_gemv_raw_iq1s_tr,ie_gemv_raw_iq2s_tr,ie_gemv_raw_iq1m_tr,ie_gemv_raw_iq3s_tr,ie_gemv_iq4xsq8"
MK_INKLING="ie_embed_q8,ie_quant_q8,ie_gemv_q8q8,ie_gemv_q4kq8,ie_gemv_q6kq8,ie_gemv_f32,ie_add,ie_copy_rows,ie_kv_store,ie_gate,ie_moe_route,ie_logit_cap,ie_argmax,ie_inkling_rmsnorm_mt_r1,ie_inkling_attn_res_r1,ie_inkling_vclip_r1,ie_inkling_sconv_r1,ie_inkling_loop_gate_r1,ie_inkling_gemv_q8_small_r1,ie_inkling_gemv_q8_nb2_r1,ie_inkling_rel_attn_dec_r1,ie_inkling_gexp_raw_iq3xxs_r1,ie_inkling_gexp_raw_iq3xxs_r3,@ie_inkling_gexp_iq4nl_r1,ie_inkling_gexp_iq4nl_r3,ie_inkling_situ_r1,ie_inkling_moe_sum_r1,ie_inkling_lc_prenorm_q8_r2,ie_inkling_lc_wfg_gate_r2,ie_inkling_sconv_add_r2,ie_inkling_gate_q8_r2"
MK_INKLING_VER="ie_gemv_q8q8_tr,@ie_gemv_q8q8_ts,@ie_gemv_q8q8_t,ie_gemv_q4kq8_tr,ie_gemv_q6kq8_tr,ie_kv_store_t,ie_sconv_ring_store,ie_moe_route,ie_copy_rows,ie_add_t,ie_logit_cap,ie_argmax,ie_inkling_embed_q8_batch_r2,ie_inkling_rmsnorm_mt_r1,ie_inkling_attn_res_r1,ie_inkling_vclip_r1,ie_inkling_sconv_r1,ie_inkling_sconv_add_r2,ie_inkling_sconv_hist_r4,ie_inkling_lc_prenorm_q8_r2,ie_inkling_lc_wfg_gate_r2,ie_inkling_rel_attn_ver_r4,ie_inkling_gate_q8_r2,ie_inkling_gemv_f32_router_ver_r4,ie_inkling_moe_sort_r1,ie_inkling_gexp_raw_iq3xxs_ver_r4,ie_inkling_gexp_iq4nl_ver_r4,ie_inkling_situ_r1,ie_inkling_moe_sum_r1"
if true; then # always regenerate: a stale table left kernels (ie_gemv_raw_iq3s) out of the megakernel
  echo "== $B/ie_mk_gen.h (tools/mk_gen.py)"
  python tools/mk_gen.py kernels/ie_mk.c "$B/ie_mk_gen.h" \
    DECODE=$MK_DECODE MTP=$MK_MTP FLINT=$MK_FLINT FLINTL=$MK_FLINTL INKLING=$MK_INKLING INKLING_VER=$MK_INKLING_VER
fi

# --- the kernels used as separate launches --------------------------------
echo "== $B/ie_kernels.ll"
"$CLANG" $GPU_CFLAGS -I$B -S -emit-llvm -o "$B/ie_kernels.ll.tmp" kernels/ie_kernels.c
sed -E -e "$SED_ATTR" "$B/ie_kernels.ll.tmp" > "$B/ie_kernels.ll"
rm -f "$B/ie_kernels.ll.tmp"
echo "== $B/ie_kernels.o $B/ie_kernels.hsaco"
"$CLANG" -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$ARCH -O3 -c -o "$B/ie_kernels.o" "$B/ie_kernels.ll"
"$LINK" -shared -o "$B/ie_kernels.hsaco" "$B/ie_kernels.o"

# --- the same kernels with the q8_0 KV cache (IE_KV_Q8=1 picks this file; the
# fp16 build above is unchanged) -------------------------------------------
echo "== $B/ie_kernels_q8.ll"
"$CLANG" $GPU_CFLAGS -DIE_KV_Q8 -I$B -S -emit-llvm -o "$B/ie_kernels_q8.ll.tmp" kernels/ie_kernels.c
sed -E -e "$SED_ATTR" "$B/ie_kernels_q8.ll.tmp" > "$B/ie_kernels_q8.ll"
rm -f "$B/ie_kernels_q8.ll.tmp"
echo "== $B/ie_kernels_q8.o $B/ie_kernels_q8.hsaco"
"$CLANG" -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$ARCH -O3 -c -o "$B/ie_kernels_q8.o" "$B/ie_kernels_q8.ll"
"$LINK" -shared -o "$B/ie_kernels_q8.hsaco" "$B/ie_kernels_q8.o"

# Adapted Saluki LoRA and IQ2_XXS verify kernels are loaded separately.
sh tools/build-saluki-kernels.sh "$B/ie_saluki_kernels.hsaco"
sh tools/build-inkling-kernels.sh "$B/ie_inkling_kernels.hsaco"
sh tools/build-vision-kernels.sh "$B/ie_vision_kernels.hsaco"

# --- the same kernels with the 4-bit KV cache (IE_KV_Q4=1 picks this file) ---
echo "== $B/ie_kernels_q4.ll"
"$CLANG" $GPU_CFLAGS -DIE_KV_Q4 -I$B -S -emit-llvm -o "$B/ie_kernels_q4.ll.tmp" kernels/ie_kernels.c
sed -E -e "$SED_ATTR" "$B/ie_kernels_q4.ll.tmp" > "$B/ie_kernels_q4.ll"
rm -f "$B/ie_kernels_q4.ll.tmp"
echo "== $B/ie_kernels_q4.o $B/ie_kernels_q4.hsaco"
"$CLANG" -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$ARCH -O3 -c -o "$B/ie_kernels_q4.o" "$B/ie_kernels_q4.ll"
"$LINK" -shared -o "$B/ie_kernels_q4.hsaco" "$B/ie_kernels_q4.o"

# --- the megakernel, one build per call table (MKS_*) ----------------------
build_mk() { # $1 = -D flag, $2 = output stem
  echo "== $B/$2.ll"
  "$CLANG" $GPU_CFLAGS -I$B -DMKS_$1 -DIE_MK_VGPR=${IE_MK_VGPR:-184} -S -emit-llvm -o "$B/$2.ll.tmp" kernels/ie_mk.c
  # amdclang 24 ignores amdgpu-num-vgpr; amdgpu-waves-per-eu is what caps the frame (192 VGPR, 4 workgroups/CU-pair)
  sed -E -e "$SED_ATTR" -e "s/\"amdgpu-num-vgpr\"=\"[0-9]+\"/\"amdgpu-waves-per-eu\"=\"${IE_MK_WAVES:-8},${IE_MK_WAVES:-8}\"/" "$B/$2.ll.tmp" > "$B/$2.ll"
  rm -f "$B/$2.ll.tmp"
  echo "== $B/$2.o $B/$2.hsaco"
  "$CLANG" -nogpulib --target=amdgcn-amd-amdhsa -mcpu=$ARCH -O3 -c -o "$B/$2.o" "$B/$2.ll"
  "$LINK" -shared -o "$B/$2.hsaco" "$B/$2.o"
}
build_mk DECODE ie_mk
build_mk MTP ie_mk_mtp
build_mk FLINT ie_mk_flint
build_mk FLINTL ie_mk_flintl
build_mk INKLING ie_mk_inkling
build_mk INKLING_VER ie_mk_inkling_ver

# --- the host: CPU backend, GPU host (-ldl is in the Makefile's LDLIBS; the
# Windows build takes hip.c's LoadLibrary path; -lm is all it needs here) ----
# A running engine keeps build/ie-run.exe as its image, and Windows will not
# let the linker replace it ("failed to write output ... Permission denied").
# Retry rather than fail the build while somebody is benching the old one.
echo "== $B/ie-run.exe"
n=0
until "$CC_WIN" $HOST_CFLAGS -o "$B/ie-run.exe" $HOST_SRC -lm; do
  n=$((n + 1))
  if [ $n -ge 12 ]; then
    echo "build-win.sh: $B/ie-run.exe stayed busy for 60 s; stop the running engine" >&2
    exit 1
  fi
  echo "   ($B/ie-run.exe is in use by a running engine; retry $n/12 in 5 s)"
  sleep 5
done

# ie-serve: the OpenAI-compatible server (SERVE.md); Winsock via src/win_net.h
echo "== $B/ie-serve.exe"
SERVE_SRC="src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c prof/sm_profiler.c src/tok.c src/json.c src/chat.c src/vision.c src/serve.c"
n=0
until "$CC_WIN" $HOST_CFLAGS -o "$B/ie-serve.exe" $SERVE_SRC -lm -lws2_32 -lpthread -lole32 -lwindowscodecs -luuid; do
  n=$((n + 1))
  if [ $n -ge 12 ]; then
    echo "build-win.sh: $B/ie-serve.exe stayed busy for 60 s; stop the running server" >&2
    exit 1
  fi
  echo "   ($B/ie-serve.exe is in use by a running server; retry $n/12 in 5 s)"
  sleep 5
done

# ie-lab: the model resident for weight-editing experiments (src/lab.c); stop a running one to rebuild
echo "== $B/ie-lab.exe"
LAB_SRC="src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c prof/sm_profiler.c src/json.c src/lab.c"
"$CC_WIN" $HOST_CFLAGS -o "$B/ie-lab.exe" $LAB_SRC -lm -lws2_32 || { echo "build-win.sh: ie-lab.exe failed (is it running?)" >&2; exit 1; }

# --- evidence: every .hsaco is a gfx1201 code object with real code -------
echo "== code objects"
for h in ie_kernels ie_kernels_q8 ie_mk ie_mk_mtp ie_mk_flint ie_mk_flintl ie_mk_inkling ie_mk_inkling_ver; do
  f=$B/$h.hsaco
  bytes=$(wc -c < "$f" | tr -d ' ')
  hits=$("$OBJDUMP" -d --mcpu=$ARCH "$f" | grep -c 'gfx1201\|v_dot' || true)
  insns=$("$OBJDUMP" -d --mcpu=$ARCH "$f" | grep -cE '^[[:space:]]+[sv]_' || true)
  names=$("$READELF" --notes "$f" | grep -E '^ +\.name: +ie_' | sed 's/.*\.name: *//' | tr '\n' ' ')
  n=$("$READELF" --notes "$f" | grep -cE '^ +\.name: +ie_' || true)
  echo "   $h.hsaco  $bytes bytes  kernels $n  insns $insns  'gfx1201|v_dot' lines $hits"
  echo "      $names" | cut -c1-200
done
OBJDUMP="$OBJDUMP" sh tools/check_objdump.sh "$B/ie_kernels.hsaco"

[ "${1:-}" = test ] || exit 0

# --- CPU-only half of `make test` (no GPU, no ROCm) ----------------------
E=$(w "$ROOT/$B/ie-run.exe")
GCC=$CC_WIN
echo "== tiny GGUFs (tools/make_tiny_gguf.py)"
python tools/make_tiny_gguf.py "$B"

echo "== kernel_test (CPU GEMV vs plain dot products, repack vs GGUF)"
"$GCC" $HOST_CFLAGS -o "$B/kernel_test.exe" tests/kernel_test.c src/common.c src/gguf.c src/model.c src/cpu.c -lm
"$(w "$ROOT/$B/kernel_test.exe")"

echo "== ext_test (Q5_K, IQ4_XS: core/ie_ext.h)"
"$GCC" $HOST_CFLAGS -o "$B/ext_test.exe" tests/ext_test.c src/common.c src/gguf.c src/model.c src/cpu.c -lm
"$(w "$ROOT/$B/ext_test.exe")"

echo "== vocab_shortlist_test (IE_DRAFT_VOCAB=64k loader: SHA-256, the manifest checks, the real 64k map if present)"
"$GCC" $HOST_CFLAGS -o "$B/vocab_shortlist_test.exe" tests/vocab_shortlist_test.c src/common.c src/gguf.c src/model.c src/cpu.c -lm
"$(w "$ROOT/$B/vocab_shortlist_test.exe")"

echo "== dry_test (the loop penalty: --dry-multiplier, made-up histories)"
"$GCC" $HOST_CFLAGS -o "$B/dry_test.exe" tests/dry_test.c src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c prof/sm_profiler.c -lm
"$(w "$ROOT/$B/dry_test.exe")"

for m in qwen2 llama llama-k flint; do
  case $m in
    qwen2)   toks=5,200,17,42,299,256; n=10; med=1e-5; rel=2e-3 ;;
    llama)   toks=1,300,77,259,3;      n=10; med=1e-5; rel=2e-3 ;;
    llama-k) toks=1,300,77,259,3;      n=10; med=1e-5; rel=2e-3 ;;
    flint)   toks=5,200,17,42,299,256; n=30; med=1e-4; rel=5e-3 ;;
  esac
  log=$B/$(echo $m | tr -d -).logits
  echo "== tiny $m: CPU backend vs numpy reference"
  "$E" "$B/tiny-$m.gguf" --tokens "$toks" --n "$n" --dump-logits "$log"
  PYTHONDONTWRITEBYTECODE=1 python tools/ref_forward.py "$B/tiny-$m.gguf" "$log" --mode q8 --tol-median $med --tol $rel
  PYTHONDONTWRITEBYTECODE=1 python tools/ref_forward.py "$B/tiny-$m.gguf" "$log" --mode float --tol 5e-2
done

echo "== prompt cache (gen_run): with and without the cache, the same answers (CPU)"
"$GCC" $HOST_CFLAGS -o "$B/cache_check.exe" tools/cache_check.c src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c prof/sm_profiler.c -lm
for m in tiny-qwen2 tiny-llama-k tiny-flint; do "$(w "$ROOT/$B/cache_check.exe")" "$B/$m.gguf"; done

echo "== disk prompt cache: which checkpoints go to disk, and when; the files, the RAM tier, eviction (CPU)"
"$GCC" $HOST_CFLAGS -o "$B/dcache_policy_check.exe" tools/dcache_policy_check.c src/common.c src/gguf.c src/model.c src/plan.c src/cpu.c src/hip.c src/gen.c prof/sm_profiler.c -lm
"$GCC" $HOST_CFLAGS -o "$B/dcache_io_check.exe" tools/dcache_io_check.c src/common.c src/gguf.c src/model.c src/plan.c prof/sm_profiler.c -lm -lpthread
mkdir -p "$B/dc-test"
for m in tiny-qwen2 tiny-llama-k; do "$(w "$ROOT/$B/dcache_policy_check.exe")" "$B/$m.gguf" "$B/dc-test"; done
"$(w "$ROOT/$B/dcache_io_check.exe")" "$B/dc-test"

echo "== laws_test (laws/ie_laws.cpp, random inputs)"
CXX_WIN=${CXX:-$(w "$(command -v g++)")}
"$CXX_WIN" -std=c++17 -O2 -o "$B/laws_test.exe" tests/laws_test.cpp
"$(w "$ROOT/$B/laws_test.exe")"

echo "== build-win.sh test: all passed"
