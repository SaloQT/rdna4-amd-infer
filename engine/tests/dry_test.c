/* dry_test.c -- the loop penalty (gen.h sampler.dry_*, gen.c dry_apply) on
 * made-up histories, CPU only: off by default; a repeat of at least
 * dry_allowed tokens lowers the continuing candidate; a shorter one does
 * not; </think> starts a new span set; a verify row's drafts count; the
 * penalty grows with the match. Built by build-win.sh test. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/gen.h"

static int fails;
#define CHECK(c, msg)                                   \
  do {                                                  \
    if (!(c)) fprintf(stderr, "FAIL %s\n", msg), fails++; \
    else printf("ok   %s\n", msg);                      \
  } while (0)

static sampler mk(float mult) {
  sampler sp;
  memset(&sp, 0, sizeof sp);
  sp.temp = 1e-4f, sp.top_p = 1.0f, sp.top_k = 4, sp.rng = 12345;
  sp.pr = malloc(64 * 4), sp.idx = malloc(64 * 4);
  sp.dry_mult = mult, sp.dry_base = 1.75f, sp.dry_allowed = 24, sp.dry_reset = 999;
  dry_begin(&sp);
  return sp;
}
/* candidates: cont (would continue the repeat) highest, then 3 others */
static uint32_t pick(sampler *sp, uint32_t cont, float *v_out) {
  float v[4] = {5.0f, 4.0f, 3.0f, 2.0f};
  uint32_t id[4] = {cont, 501, 502, 503};
  const uint32_t t = sample_cands(sp, v, id, 4);
  if (v_out) {
    for (int c = 0; c < 4; c++)
      if (id[c] == cont) *v_out = v[c];
  }
  return t;
}
static void commit_range(sampler *sp, uint32_t a, uint32_t n) { /* tokens a, a+1, .., a+n-1 */
  for (uint32_t i = 0; i < n; i++) {
    const uint32_t t = a + i;
    dry_commit(sp, &t, 1);
  }
}

int main(void) {
  { /* off: the order is kept */
    sampler sp = mk(0.0f);
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 30);
    CHECK(pick(&sp, 130, NULL) == 130, "off: the repeat continues");
  }
  { /* a repeat of 30 (>= 24): the continuation is lowered and another token wins */
    sampler sp = mk(0.8f);
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 30);
    float v = 0;
    const uint32_t t = pick(&sp, 130, &v);
    CHECK(fabsf(v - (5.0f - 0.8f * powf(1.75f, 6.0f))) < 1e-4f, "repeat 30: penalty 0.8 * 1.75^6");
    CHECK(t == 501, "repeat 30: the next candidate is chosen");
    CHECK(sp.dry_hits == 1, "repeat 30: one candidate counted");
  }
  { /* a repeat of 20 (< 24): no penalty */
    sampler sp = mk(0.8f);
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 20);
    CHECK(pick(&sp, 120, NULL) == 120 && sp.dry_hits == 0, "repeat 20: no penalty");
  }
  { /* exactly 24: penalty 0.8 (base^0) is not enough to drop a 1.0 lead */
    sampler sp = mk(0.8f);
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 24);
    float v = 0;
    CHECK(pick(&sp, 124, &v) == 124 && fabsf(v - 4.2f) < 1e-4f, "repeat 24: penalty 0.8, still first");
  }
  { /* </think> (999) between: a new span set, no penalty */
    sampler sp = mk(0.8f);
    commit_range(&sp, 100, 40);
    const uint32_t r = 999;
    dry_commit(&sp, &r, 1);
    commit_range(&sp, 100, 30);
    CHECK(pick(&sp, 130, NULL) == 130 && sp.dry_hits == 0, "after </think>: the reasoning's spans do not count");
  }
  { /* a verify row: 22 committed + 8 drafts = a repeat of 30 */
    sampler sp = mk(0.8f);
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 22);
    uint32_t d[8];
    for (int i = 0; i < 8; i++) d[i] = 122 + i;
    sp.dry_x = d, sp.dry_xn = 8;
    CHECK(pick(&sp, 130, NULL) == 501, "verify row: drafts extend the match");
    sp.dry_xn = 0;
    CHECK(pick(&sp, 122, NULL) == 122, "same history without the drafts: repeat 22, no penalty");
  }
  { /* a loop: block of 30 tokens three times; the 4th time is penalized hard */
    sampler sp = mk(0.8f);
    for (int k = 0; k < 3; k++) commit_range(&sp, 100, 30);
    commit_range(&sp, 100, 29);
    float v = 0;
    pick(&sp, 129, &v);
    CHECK(v < -1000.0f, "loop: a long match makes the continuation all but impossible");
  }
  { /* the prompt does not count: dry_begin forgets the output */
    sampler sp = mk(0.8f);
    commit_range(&sp, 100, 40);
    dry_begin(&sp);
    commit_range(&sp, 100, 30);
    CHECK(pick(&sp, 130, NULL) == 130, "new request: the earlier output is forgotten");
  }
  { /* invalid / NaN / OOB candidates never segfault or return OOB ids */
    sampler sp = mk(0.8f);
    sp.temp = 0.6f, sp.top_p = 0.95f, sp.vocab = 1000;
    float v_nan[4] = {-INFINITY, -INFINITY, NAN, -INFINITY};
    uint32_t id_nan[4] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 1500u};
    CHECK(sample_cands(&sp, v_nan, id_nan, 4) == 0, "all NaN/-inf/OOB candidates fall back to 0");
    CHECK(sample_cands(&sp, v_nan, id_nan, 0) == 0, "empty candidate set falls back to 0");
    float v_mix[4] = {8.0f, -INFINITY, -INFINITY, -INFINITY};
    uint32_t id_mix[4] = {42u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
    CHECK(sample_cands(&sp, v_mix, id_mix, 4) == 42u, "valid candidate survives -inf/0xFFFFFFFF tail");
  }
  { /* pause inside <tool_call>..</tool_call>, resume after </tool_call>, re-pause on second tool call */
    const uint32_t tc_open = 880, tc_close = 881;
    sampler sp = mk(0.8f);
    sp.dry_pause_open_plus_one = tc_open + 1u;
    sp.dry_pause_close_plus_one = tc_close + 1u;
    /* Before <tool_call>: prose repeat IS penalized */
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 30);
    CHECK(pick(&sp, 130, NULL) == 501 && sp.dry_hits == 1, "before <tool_call>: repeat penalized");
    /* Enter <tool_call>: repeated CSS/JS blocks get zero penalty */
    dry_commit(&sp, &tc_open, 1);
    for (int k = 0; k < 3; k++) commit_range(&sp, 200, 30);
    commit_range(&sp, 200, 29);
    CHECK(pick(&sp, 229, NULL) == 229 && sp.dry_hits == 1, "inside <tool_call>: long repeat unpenalized");
    /* Exit </tool_call>: tool-call code does not leak into post-tool prose */
    dry_commit(&sp, &tc_close, 1);
    commit_range(&sp, 200, 30);
    CHECK(pick(&sp, 230, NULL) == 230 && sp.dry_hits == 1, "after </tool_call>: tool tokens do not trigger DRY");
    /* A new prose loop after </tool_call> IS penalized again (commit 230..239 so 230 follows 200..229) */
    commit_range(&sp, 230, 10), commit_range(&sp, 900, 3), commit_range(&sp, 200, 30);
    CHECK(pick(&sp, 230, NULL) == 501 && sp.dry_hits == 2, "after </tool_call>: new prose repeat penalized");
    /* Second <tool_call> pauses DRY again */
    dry_commit(&sp, &tc_open, 1);
    commit_range(&sp, 300, 40), commit_range(&sp, 900, 3), commit_range(&sp, 300, 30);
    CHECK(pick(&sp, 330, NULL) == 330 && sp.dry_hits == 2, "second <tool_call>: repeat unpenalized");
  }
  { /* dry_pause_after_reset: <tool_call> inside <think> does NOT pause; after </think> it DOES */
    const uint32_t tc_open = 880, tc_close = 881, think_end = 999;
    sampler sp = mk(0.8f);
    sp.dry_pause_open_plus_one = tc_open + 1u;
    sp.dry_pause_close_plus_one = tc_close + 1u;
    sp.dry_pause_after_reset = 1;
    dry_begin(&sp);
    dry_commit(&sp, &tc_open, 1); /* literal mention inside <think> */
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 30);
    CHECK(pick(&sp, 130, NULL) == 501 && sp.dry_hits == 1, "in <think>: <tool_call> does not pause DRY");
    /* Close </think>, then real <tool_call> pauses DRY */
    dry_commit(&sp, &think_end, 1);
    dry_commit(&sp, &tc_open, 1);
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 30);
    CHECK(pick(&sp, 130, NULL) == 130 && sp.dry_hits == 1, "after </think>: <tool_call> pauses DRY");
    /* dry_begin restores wait_reset state for next request */
    dry_begin(&sp);
    dry_commit(&sp, &tc_open, 1);
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 30);
    CHECK(pick(&sp, 130, NULL) == 501 && sp.dry_hits == 1, "dry_begin resets pause-after-reset state");
  }
  { /* dry_x verify drafts crossing <tool_call> and </tool_call> */
    const uint32_t tc_open = 880, tc_close = 881;
    sampler sp = mk(0.8f);
    sp.dry_pause_open_plus_one = tc_open + 1u;
    sp.dry_pause_close_plus_one = tc_close + 1u;
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 22);
    uint32_t d1[9] = {tc_open, 122, 123, 124, 125, 126, 127, 128, 129};
    sp.dry_x = d1, sp.dry_xn = 9;
    CHECK(pick(&sp, 130, NULL) == 130 && sp.dry_hits == 0, "verify row: draft <tool_call> pauses DRY");
    /* Already paused, draft crosses </tool_call>: no stale match before close */
    sp.dry_xn = 0;
    dry_commit(&sp, &tc_open, 1);
    commit_range(&sp, 100, 30);
    uint32_t d2[4] = {tc_close, 100, 101, 102};
    sp.dry_x = d2, sp.dry_xn = 4;
    CHECK(pick(&sp, 103, NULL) == 103 && sp.dry_hits == 0, "verify row: draft </tool_call> resets span cleanly");
  }
  { /* token id 0 works cleanly via plus-one encoding */
    const uint32_t tc_open = 0, tc_close = 881;
    sampler sp = mk(0.8f);
    sp.dry_pause_open_plus_one = tc_open + 1u;
    sp.dry_pause_close_plus_one = tc_close + 1u;
    dry_commit(&sp, &tc_open, 1);
    commit_range(&sp, 100, 40), commit_range(&sp, 900, 3), commit_range(&sp, 100, 30);
    CHECK(pick(&sp, 130, NULL) == 130 && sp.dry_hits == 0, "pause open id 0 works via plus-one encoding");
  }
  printf(fails ? "dry_test: %d FAILED\n" : "dry_test: all passed\n", fails);
  return fails != 0;
}
