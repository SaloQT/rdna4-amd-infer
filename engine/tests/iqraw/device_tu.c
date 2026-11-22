/* device_tu.c -- the amdgcn build of kernels/ie_iqraw.inc on its own, with the
 * pointer macros kernels/ie_kernels.c passes (address space 1), and one kernel
 * that calls ie_rq_dot for each of the seven handled types, so every case is
 * compiled and inlined. run.sh runs (the ticket's command):
 *
 *   amdclang -x c -std=c11 --target=amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib \
 *            -O3 -S -emit-llvm device_tu.c -o -
 *
 * and the same without -emit-llvm for the ISA, where the kernel's local
 * (private segment / scratch) size must stay 0.
 */
typedef unsigned char u8;
typedef unsigned int u32;
#define G __attribute__((address_space(1)))
#define IE_RQP const G u8
#define IE_RQF const G float
#define IE_RQU const G u32
#include "../../kernels/ie_iqraw.inc"

__attribute__((amdgpu_kernel, visibility("protected"))) void ie_rq_test(const G u8 *p, const G u8 *xq, G float *y,
                                                                      u32 nb, u32 qb) {
  y[0] = ie_rq_dot(10u, p, xq, nb, qb)  /* Q2_K */
         + ie_rq_dot(16u, p, xq, nb, qb) /* IQ2_XXS */
         + ie_rq_dot(17u, p, xq, nb, qb) /* IQ2_XS */
         + ie_rq_dot(18u, p, xq, nb, qb) /* IQ3_XXS */
         + ie_rq_dot(19u, p, xq, nb, qb) /* IQ1_S */
         + ie_rq_dot(22u, p, xq, nb, qb) /* IQ2_S */
         + ie_rq_dot(29u, p, xq, nb, qb) /* IQ1_M */
         + (float)ie_rq_bs(10u) + (float)ie_rq_bs(16u) + (float)ie_rq_bs(17u) + (float)ie_rq_bs(18u) +
         (float)ie_rq_bs(19u) + (float)ie_rq_bs(22u) + (float)ie_rq_bs(29u);
}
