/* plan.c -- static plan of the activation arena.
 *
 * Greedy: take the buffers from the largest to the smallest; put each at the
 * lowest 256-aligned offset where it does not overlap any already placed
 * buffer whose lifetime overlaps its own. The planner is NOT verified; its
 * output is checked by the verified ie_plan_check (core/ie_core.h), and the
 * engine stops if the check fails. The laws plan_fits, plan_aligned and
 * plan_no_clobber say what an accepted plan guarantees. */
#include <string.h>

#include "../core/ie_core.h"
#include "common.h"
#include "model.h"

static const graph *sort_g;
static int plan_cmp_mode = 0;
static int by_size(const void *a, const void *b) {
  const int ia = *(const int *)a, ib = *(const int *)b;
  const buf *x = &sort_g->bufs[ia], *y = &sort_g->bufs[ib];
  if (plan_cmp_mode == 3) {
    /* Lifetime-span descending, then size descending */
    const uint32_t sx = x->last >= x->first ? x->last - x->first : 0u;
    const uint32_t sy = y->last >= y->first ? y->last - y->first : 0u;
    if (sx != sy) return sx < sy ? 1 : -1;
    if (x->size != y->size) return x->size < y->size ? 1 : -1;
    return ia - ib;
  }
  if (x->size != y->size) return x->size < y->size ? 1 : -1;
  if (plan_cmp_mode == 1) {
    const uint32_t sx = x->last >= x->first ? x->last - x->first : 0u;
    const uint32_t sy = y->last >= y->first ? y->last - y->first : 0u;
    if (sx != sy) return sx < sy ? 1 : -1;
  } else if (plan_cmp_mode == 2) {
    if (x->first != y->first) return x->first < y->first ? -1 : 1;
  }
  return ia - ib;
}

static int live_overlap(const buf *a, const buf *b) { return a->first <= b->last && b->first <= a->last; }

/* Pack buffers in given order using either first-fit (best_fit=0) or tightest-hole best-fit (best_fit=1). */
static uint64_t pack_order(const graph *g, const int *order, uint32_t n, int best_fit, uint32_t *out_off) {
  char *placed = calloc(n, 1);
  uint32_t *cands = best_fit ? malloc((size_t)(n + 1u) * sizeof(uint32_t)) : NULL;
  uint64_t top = 0;
  for (uint32_t k = 0; k < n; k++) {
    const int idx = order[k];
    const buf *b = &g->bufs[idx];
    uint64_t chosen = 0;
    if (!best_fit) {
      uint64_t off = 0;
      for (int moved = 1; moved;) {
        moved = 0;
        for (uint32_t j = 0; j < n; j++) {
          if (!placed[j] || !live_overlap(b, &g->bufs[j])) continue;
          const uint64_t co = out_off[j], ce = co + g->bufs[j].size;
          if (off < ce && co < off + b->size) {
            off = (ce + 255u) & ~(uint64_t)255u;
            moved = 1;
          }
        }
      }
      chosen = off;
    } else {
      /* Candidate offsets: 0 and aligned ends of all overlapping placed buffers; pick tightest gap (smallest slack). */
      uint32_t nc = 0u;
      cands[nc++] = 0u;
      for (uint32_t j = 0; j < n; j++) {
        if (!placed[j] || !live_overlap(b, &g->bufs[j])) continue;
        const uint64_t ce = ((uint64_t)out_off[j] + g->bufs[j].size + 255u) & ~(uint64_t)255u;
        if (ce < 0xFFFFFF00u) cands[nc++] = (uint32_t)ce;
      }
      uint64_t best_off = UINT64_MAX, best_gap = UINT64_MAX;
      for (uint32_t ci = 0; ci < nc; ci++) {
        const uint64_t off = cands[ci];
        int ok = 1;
        uint64_t next_start = UINT64_MAX;
        for (uint32_t j = 0; j < n; j++) {
          if (!placed[j] || !live_overlap(b, &g->bufs[j])) continue;
          const uint64_t co = out_off[j], ce = co + g->bufs[j].size;
          if (off < ce && co < off + b->size) { ok = 0; break; }
          if (co >= off + b->size && co < next_start) next_start = co;
        }
        if (!ok) continue;
        const uint64_t gap = next_start == UINT64_MAX ? (UINT64_MAX - 1u) : (next_start - (off + b->size));
        if (gap < best_gap || (gap == best_gap && off < best_off)) {
          best_gap = gap;
          best_off = off;
        }
      }
      chosen = best_off;
    }
    if (chosen >= 0xFFFFFF00u || chosen + b->size > 0xFFFFFF00u) ie_die("activation arena larger than 4 GiB");
    out_off[idx] = (uint32_t)chosen;
    placed[idx] = 1;
    if (chosen + b->size > top) top = chosen + b->size;
  }
  free(cands), free(placed);
  return (top + 255u) & ~(uint64_t)255u;
}

void plan_arena(graph *g, int verbose) {
  uint32_t n = g->n_bufs;
  int *order = malloc(n * sizeof(int));
  int *best_order = malloc(n * sizeof(int));
  uint32_t *cur_off = malloc(n * sizeof(uint32_t));
  uint32_t *best_off = malloc(n * sizeof(uint32_t));
  const char *pe = getenv("IE_PLAN_LARGEST");
  const int multi_on = !(pe && pe[0] == '0');
  sort_g = g;

  /* Strategy 0 (exact legacy baseline): size-desc + index-asc, first-fit; guarantees R41 <= R37 arena always. */
  uint64_t best_arena = UINT64_MAX;
  const int n_modes = multi_on ? 4 : 1;
  for (int m = 0; m < n_modes; m++) {
    for (uint32_t i = 0; i < n; i++) order[i] = (int)i;
    plan_cmp_mode = m;
    qsort(order, n, sizeof(int), by_size);
    for (int bf = 0; bf <= (multi_on ? 1 : 0); bf++) {
      const uint64_t ar = pack_order(g, order, n, bf, cur_off);
      if (ar < best_arena) {
        best_arena = ar;
        memcpy(best_off, cur_off, n * sizeof(uint32_t));
        memcpy(best_order, order, n * sizeof(int));
      }
    }
  }
  for (uint32_t i = 0; i < n; i++) g->bufs[i].off = best_off[i];
  g->arena = (uint32_t)best_arena;

  /* The verified check. */
  u32 *off = malloc(n * 4), *size = malloc(n * 4), *first = malloc(n * 4), *last = malloc(n * 4);
  for (uint32_t i = 0; i < n; i++) off[i] = g->bufs[i].off, size[i] = g->bufs[i].size, first[i] = g->bufs[i].first, last[i] = g->bufs[i].last;
  Plan p = {{off}, {size}, {first}, {last}, n, g->arena};
  int ok = ie_plan_check(p);
  uint64_t sum = 0;
  for (uint32_t i = 0; i < n; i++) sum += g->bufs[i].size;
  if (verbose)
    fprintf(stderr, "plan: %u buffers over %u ops, arena %u bytes (sum of sizes %llu), ie_plan_check: %s\n", n,
            g->n_ops, g->arena, (unsigned long long)sum, ok ? "ok" : "FAILED");
  if (getenv("IE_PLAN_DUMP")) { /* debug: the buffers by size (the order of the planner), with their place in the arena */
    for (uint32_t k = 0; k < n && k < 16; k++) {
      const buf *b = &g->bufs[best_order[k]];
      fprintf(stderr, "plan: %-14s size %10u stride %8u off %10u ops %u .. %u\n", b->name, b->size, b->stride, b->off, b->first, b->last);
    }
  }
  free(off), free(size), free(first), free(last), free(order), free(best_order), free(cur_off), free(best_off);
  if (!ok) ie_die("ie_plan_check rejected the activation plan");
}
