/* bw_bench_kernel.c -- streaming read of a large buffer (tools/bw_bench.py):
 * the attainable DRAM read bandwidth of the card, against which the decode
 * kernels are measured. Grid-stride loop, 16-byte loads; the result only
 * guards the loads from being dropped. */
#define G __attribute__((address_space(1)))
typedef unsigned int u32;
typedef u32 u32x4 __attribute__((ext_vector_type(4)));

__attribute__((amdgpu_kernel, visibility("protected"))) void bw_read(const G u32x4 *p, unsigned long n16, G u32 *out) {
  const unsigned long nt = (unsigned long)__builtin_amdgcn_grid_size_x();
  u32x4 a0 = 0, a1 = 0;
  unsigned long i = (unsigned long)__builtin_amdgcn_workgroup_id_x() * 256ul + __builtin_amdgcn_workitem_id_x();
  for (; i + nt < n16; i += 2ul * nt) a0 ^= p[i], a1 ^= p[i + nt];
  if (i < n16) a0 ^= p[i];
  a0 ^= a1;
  const u32 v = a0.x ^ a0.y ^ a0.z ^ a0.w;
  if (v == 0x9e3779b9u) out[0] = v;
}
