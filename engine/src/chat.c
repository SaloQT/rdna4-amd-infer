/* chat.c -- see chat.h. The structure follows the template's Jinja code
 * (ie-serve --show-template); the checks are in SERVE.md section 6. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat.h"

/* The literal texts of the template's tools system message. */
static const char TOOLS_HEAD[] =
"# Tools\n"
"\n"
"You have access to the following functions:\n"
"\n"
"<tools>"
;
static const char TOOLS_INSTR[] =
"\n"
"\n"
"If you choose to call a function ONLY reply in the following format with NO suffix:\n"
"\n"
"<tool_call>\n"
"<function=example_function_name>\n"
"<parameter=example_parameter_1>\n"
"value_1\n"
"</parameter>\n"
"<parameter=example_parameter_2>\n"
"This is the value for the second parameter\n"
"that can span\n"
"multiple lines\n"
"</parameter>\n"
"</function>\n"
"</tool_call>\n"
"\n"
"<IMPORTANT>\n"
"Reminder:\n"
"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
"- Required parameters MUST be specified\n"
"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
"</IMPORTANT>"
;

/* ---- helpers ---- */

/* Python's str.isspace() for one code point */
static int py_space(unsigned cp) {
  return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
/* the code point at s (n > 0 bytes left) and its length; an invalid byte is itself */
static unsigned cp_at(const unsigned char *s, size_t n, size_t *len) {
  unsigned c = s[0], k = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
  if (k > n) k = 1;
  if (k == 1) { *len = 1; return c; }
  unsigned cp = c & (0x3F >> (k - 1));
  for (unsigned i = 1; i < k; i++) {
    if ((s[i] & 0xC0) != 0x80) { *len = 1; return c; }
    cp = (cp << 6) | (s[i] & 0x3F);
  }
  *len = k;
  return cp;
}
/* Jinja's |trim (Python's str.strip()) */
static void py_trim(const char **s, size_t *n) {
  const unsigned char *p = (const unsigned char *)*s;
  size_t a = 0, e = *n, len;
  while (a < e && py_space(cp_at(p + a, e - a, &len))) a += len;
  while (e > a) { /* back up to the start of the last code point */
    size_t b = e - 1;
    while (b > a && (p[b] & 0xC0) == 0x80 && e - b < 4) b--;
    if (!py_space(cp_at(p + b, e - b, &len)) || b + len != e) break;
    e = b;
  }
  *s += a, *n = e - a;
}

static void seg_push(chat_prompt *p, int kind, size_t off, size_t n) {
  if (kind == SEG_TEXT && !n) return;
  if (kind == SEG_TEXT && p->n_seg && p->seg[p->n_seg - 1].kind == SEG_TEXT &&
      p->seg[p->n_seg - 1].off + p->seg[p->n_seg - 1].n == off) { /* join adjacent text */
    p->seg[p->n_seg - 1].n += n;
    return;
  }
  if (p->n_seg == p->cap) {
    p->cap = p->cap ? 2 * p->cap : 32;
    p->seg = realloc(p->seg, p->cap * sizeof(chat_seg));
    if (!p->seg) abort();
  }
  p->seg[p->n_seg++] = (chat_seg){kind, off, n};
}
static void T(chat_prompt *p, const char *s, size_t n) {
  const size_t off = p->text.n;
  sb_add(&p->text, s, n);
  seg_push(p, SEG_TEXT, off, n);
}
static void TS(chat_prompt *p, const char *s) { T(p, s, strlen(s)); }
static void START(chat_prompt *p) {
  sb_puts(&p->text, "<|im_start|>");
  seg_push(p, SEG_IM_START, 0, 0);
}
static void END(chat_prompt *p) {
  sb_puts(&p->text, "<|im_end|>");
  seg_push(p, SEG_IM_END, 0, 0);
}
static void IMG(chat_prompt *p, const char *url, size_t len) {
  if (p->n_imgs == p->cap_imgs) {
    p->cap_imgs = p->cap_imgs ? 2 * p->cap_imgs : 4;
    p->imgs = realloc(p->imgs, p->cap_imgs * sizeof(chat_img));
    if (!p->imgs) abort();
  }
  const size_t idx = p->n_imgs++;
  p->imgs[idx] = (chat_img){url, len};
  sb_puts(&p->text, "<|vision_start|><|image_pad|><|vision_end|>");
  seg_push(p, SEG_IMG, idx, 0);
}
static void INK_IMG(chat_prompt *p, const char *url, size_t len) {
  if (p->n_imgs == p->cap_imgs) {
    p->cap_imgs = p->cap_imgs ? 2 * p->cap_imgs : 4;
    p->imgs = realloc(p->imgs, p->cap_imgs * sizeof(chat_img));
    if (!p->imgs) abort();
  }
  const size_t idx = p->n_imgs++;
  p->imgs[idx] = (chat_img){url, len};
  seg_push(p, SEG_IMG, p->text.n, idx);
}

void chat_prompt_free(chat_prompt *p) {
  sb_free(&p->text);
  free(p->seg);
  free(p->imgs);
  memset(p, 0, sizeof *p);
}

/* ---- Python json.dumps ---- */
static void py_str(sbuf *b, const char *s, size_t n) {
  sb_add(b, "\"", 1);
  for (size_t i = 0; i < n; i++) {
    const unsigned char c = (unsigned char)s[i];
    if (c == '"') sb_add(b, "\\\"", 2);
    else if (c == '\\') sb_add(b, "\\\\", 2);
    else if (c == '\n') sb_add(b, "\\n", 2);
    else if (c == '\r') sb_add(b, "\\r", 2);
    else if (c == '\t') sb_add(b, "\\t", 2);
    else if (c == '\b') sb_add(b, "\\b", 2);
    else if (c == '\f') sb_add(b, "\\f", 2);
    else if (c < 0x20) sb_printf(b, "\\u%04x", c);
    else sb_add(b, (const char *)s + i, 1);
  }
  sb_add(b, "\"", 1);
}
static int cmp_jkey(const void *a, const void *b) {
  const unsigned char *x = (const unsigned char *)(*(const jval *const *)a)->key;
  const unsigned char *y = (const unsigned char *)(*(const jval *const *)b)->key;
  while (*x && *x == *y) x++, y++;
  return *x < *y ? -1 : *x > *y;
}
static void py_json_opts(sbuf *b, const jval *v, int compact, int sort_keys) {
  const char *csep = compact ? "," : ", ", *ksep = compact ? ":" : ": ";
  switch (v->t) {
  case J_NULL: sb_puts(b, "null"); break;
  case J_TRUE: sb_puts(b, "true"); break;
  case J_FALSE: sb_puts(b, "false"); break;
  case J_NUM: sb_puts(b, v->raw ? v->raw : "0"); break;
  case J_STR: py_str(b, v->str, v->slen); break;
  case J_ARR:
    sb_add(b, "[", 1);
    for (const jval *k = v->kid; k; k = k->next) {
      if (k != v->kid) sb_puts(b, csep);
      py_json_opts(b, k, compact, sort_keys);
    }
    sb_add(b, "]", 1);
    break;
  case J_OBJ:
    sb_add(b, "{", 1);
    if (!sort_keys || !v->kid || !v->kid->next) {
      for (const jval *k = v->kid; k; k = k->next) {
        if (k != v->kid) sb_puts(b, csep);
        py_str(b, k->key, strlen(k->key)), sb_puts(b, ksep);
        py_json_opts(b, k, compact, sort_keys);
      }
    } else {
      size_t nk = 0;
      for (const jval *k = v->kid; k; k = k->next) nk++;
      const jval *stk[64], **ks = nk <= 64 ? stk : malloc(nk * sizeof *ks);
      if (!ks) abort();
      size_t i = 0;
      for (const jval *k = v->kid; k; k = k->next) ks[i++] = k;
      qsort(ks, nk, sizeof *ks, cmp_jkey);
      for (i = 0; i < nk; i++) {
        if (i) sb_puts(b, csep);
        py_str(b, ks[i]->key, strlen(ks[i]->key)), sb_puts(b, ksep);
        py_json_opts(b, ks[i], compact, sort_keys);
      }
      if (ks != stk) free(ks);
    }
    sb_add(b, "}", 1);
    break;
  }
}
void py_json(sbuf *b, const jval *v) { py_json_opts(b, v, 0, 0); }

/* ---- the template ---- */

typedef struct {
  size_t off;
  const char *url;
  size_t len;
} msg_img;
typedef struct {
  msg_img *v;
  size_t n, cap;
} msg_imgs;

static void msg_img_push(msg_imgs *a, size_t off, const char *url, size_t len) {
  if (a->n == a->cap) {
    a->cap = a->cap ? 2 * a->cap : 4;
    a->v = realloc(a->v, a->cap * sizeof(msg_img));
    if (!a->v) abort();
  }
  a->v[a->n++] = (msg_img){off, url, len};
}

static int extract_image_url(const jval *p, const char **url, size_t *len) {
  const jval *iu = json_get(p, "image_url");
  if (iu) {
    if (iu->t == J_STR) { *url = iu->str, *len = iu->slen; return 1; }
    if (iu->t == J_OBJ) {
      const jval *u = json_get(iu, "url");
      if (u && u->t == J_STR) { *url = u->str, *len = u->slen; return 1; }
    }
    return -1;
  }
  const jval *im = json_get(p, "image");
  if (im && im->t == J_STR) { *url = im->str, *len = im->slen; return 1; }
  return 0;
}

/* render_content: a string, null, or parts with "text" (and optional "image_url"/"image" when mi != NULL) */
static int content_of_vis(const jval *c, sbuf *out, msg_imgs *mi) {
  if (!c || c->t == J_NULL) return 1;
  if (c->t == J_STR) { sb_add(out, c->str, c->slen); return 1; }
  if (c->t != J_ARR) return 0;
  for (const jval *p = c->kid; p; p = p->next) {
    if (json_get(p, "video")) return 0;
    const char *url = NULL;
    size_t ulen = 0;
    const int is_img = extract_image_url(p, &url, &ulen);
    if (is_img < 0) return 0;
    if (is_img > 0) {
      if (!mi) return 0;
      msg_img_push(mi, out->n, url, ulen);
      continue;
    }
    const jval *tx = json_get(p, "text");
    if (!tx || tx->t != J_STR) return 0;
    sb_add(out, tx->str, tx->slen);
  }
  return 1;
}
static int content_of(const jval *c, sbuf *out) { return content_of_vis(c, out, NULL); }

static void emit_with_imgs(chat_prompt *p, const char *s, size_t n, const msg_imgs *mi) {
  if (!mi || !mi->n) { T(p, s, n); return; }
  size_t pos = 0;
  for (size_t k = 0; k < mi->n; k++) {
    size_t at = mi->v[k].off <= n ? mi->v[k].off : n;
    if (at > pos) T(p, s + pos, at - pos);
    IMG(p, mi->v[k].url, mi->v[k].len);
    pos = at;
  }
  if (pos < n) T(p, s + pos, n - pos);
}

static const char *effort_text(const char *e) {
  if (!strcmp(e, "low") || !strcmp(e, "minimal"))
    return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
           "unnecessary elaboration.";
  if (!strcmp(e, "medium")) return "";
  return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
         "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
}

static const char *role_of(const jval *m) {
  const jval *r = json_get(m, "role");
  if (!r || r->t != J_STR) return NULL;
  return !strcmp(r->str, "developer") ? "system" : r->str;
}

/* An assistant tool call: <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n...</function>\n</tool_call> */
static int render_call(chat_prompt *p, const jval *tc, const char *lead, char *err, size_t errn) {
  const jval *f = json_get(tc, "function");
  if (f) tc = f;
  const jval *name = json_get(tc, "name"), *args = json_get(tc, "arguments");
  if (!name || name->t != J_STR) { snprintf(err, errn, "a tool call has no function name"); return 0; }
  TS(p, lead), TS(p, "<tool_call>\n<function="), T(p, name->str, name->slen), TS(p, ">\n");
  jval *parsed = NULL;
  if (args && args->t == J_STR && args->slen) { /* OpenAI sends the arguments as a JSON string */
    char e2[128];
    parsed = json_parse(args->str, args->slen, e2, sizeof e2);
    if (!parsed || parsed->t != J_OBJ) {
      json_free(parsed);
      snprintf(err, errn, "the arguments of a tool call are not a JSON object");
      return 0;
    }
    args = parsed;
  }
  if (args && args->t == J_OBJ)
    for (const jval *a = args->kid; a; a = a->next) {
      TS(p, "<parameter="), TS(p, a->key), TS(p, ">\n");
      if (a->t == J_STR) T(p, a->str, a->slen);
      else {
        sbuf j = {0};
        py_json(&j, a);
        T(p, j.p, j.n);
        sb_free(&j);
      }
      TS(p, "\n</parameter>\n");
    }
  TS(p, "</function>\n</tool_call>");
  json_free(parsed);
  return 1;
}

int chat_render_vis(const jval *msgs, const jval *tools, int think, const char *effort, int allow_images, chat_prompt *p, char *err, size_t errn) {
  memset(p, 0, sizeof *p);
  if (!msgs || msgs->t != J_ARR || !msgs->kid) { snprintf(err, errn, "\"messages\" must be a non-empty array"); return 0; }
  if (tools && (tools->t != J_ARR || !tools->kid)) tools = NULL;
  const char *ri = think ? effort_text(effort) : "";
  size_t n_msg = 0;
  for (const jval *m = msgs->kid; m; m = m->next) n_msg++;
  const jval **ms = calloc(n_msg, sizeof *ms);
  sbuf *cs = calloc(n_msg, sizeof *cs); /* the trimmed contents */
  msg_imgs *mis = calloc(n_msg, sizeof *mis);
  const char **rs = calloc(n_msg, sizeof *rs);
  size_t i = 0, *cn = calloc(n_msg, sizeof *cn);
  int ok = 0, user_query = 0;
  for (const jval *m = msgs->kid; m; m = m->next, i++) {
    ms[i] = m;
    const char *r = rs[i] = role_of(m);
    if (!r || (strcmp(r, "system") && strcmp(r, "user") && strcmp(r, "assistant") && strcmp(r, "tool"))) {
      snprintf(err, errn, "a message's role must be system, user, assistant or tool");
      goto done;
    }
    if (!strcmp(r, "system") && i) { snprintf(err, errn, "a system message must be the first message"); goto done; }
    const int msg_img_ok = allow_images && strcmp(r, "system") != 0 && strcmp(r, "assistant") != 0;
    if (!content_of_vis(json_get(m, "content"), &cs[i], msg_img_ok ? &mis[i] : NULL)) {
      snprintf(err, errn, "%s", allow_images ? "unsupported multimodal content (images are allowed in user/tool messages only)"
                                             : "only text content is supported (pass --mmproj to enable images)");
      goto done;
    }
    const char *orig = cs[i].p ? cs[i].p : "";
    const char *c = orig;
    size_t n = cs[i].n;
    if (!mis[i].n) {
      py_trim(&c, &n);
      if (cs[i].p) { /* keep the trimmed text at the start */
        memmove(cs[i].p, c, n);
        cs[i].p[n] = 0;
      }
    } else {
      /* Match Jinja |trim around <|vision_start|>..<|vision_end|>: only trim leading spaces before first image
       * and trailing spaces after last image, adjusting stored image byte offsets accordingly. */
      const size_t first_off = mis[i].v[0].off;
      const size_t last_off = mis[i].v[mis[i].n - 1].off;
      size_t a = 0, e = n, len;
      while (a < first_off && py_space(cp_at((const unsigned char *)orig + a, first_off - a, &len))) a += len;
      while (e > last_off) {
        size_t b = e - 1;
        while (b > last_off && (((unsigned char)orig[b]) & 0xC0) == 0x80 && e - b < 4) b--;
        if (!py_space(cp_at((const unsigned char *)orig + b, e - b, &len)) || b + len != e) break;
        e = b;
      }
      n = e - a;
      if (cs[i].p) {
        memmove(cs[i].p, orig + a, n);
        cs[i].p[n] = 0;
      }
      for (size_t k = 0; k < mis[i].n; k++) mis[i].v[k].off -= a;
    }
    cn[i] = n;
    if (!strcmp(r, "user")) { /* the template needs a user message that is not only a tool response */
      const char *t = cs[i].p ? cs[i].p : "";
      if (mis[i].n || !(n >= 15 + 16 && !memcmp(t, "<tool_response>", 15) && !memcmp(t + n - 16, "</tool_response>", 16))) user_query = 1;
    }
  }
  if (!user_query) { snprintf(err, errn, "no user message"); goto done; }

  /* the system message */
  const int sys = !strcmp(rs[0], "system");
  if (tools) {
    START(p), TS(p, "system\n");
    if (ri[0]) TS(p, ri), TS(p, "\n\n");
    TS(p, TOOLS_HEAD);
    for (const jval *t = tools->kid; t; t = t->next) {
      sbuf j = {0};
      py_json(&j, t);
      TS(p, "\n"), T(p, j.p, j.n);
      sb_free(&j);
    }
    p->tools_end = p->text.n;
    TS(p, "\n</tools>"), TS(p, TOOLS_INSTR);
    if (sys && cn[0]) TS(p, "\n\n"), T(p, cs[0].p, cn[0]);
    END(p), TS(p, "\n");
  } else if ((sys && cn[0]) || ri[0]) {
    START(p), TS(p, "system\n");
    if (ri[0]) TS(p, ri), TS(p, sys && cn[0] ? "\n\n" : "");
    if (sys && cn[0]) T(p, cs[0].p, cn[0]);
    END(p), TS(p, "\n");
  }

  for (i = 0; i < n_msg; i++) {
    const char *r = rs[i], *c = cs[i].p ? cs[i].p : "";
    size_t n = cn[i];
    if (!strcmp(r, "user")) {
      START(p), TS(p, "user\n"), emit_with_imgs(p, c, n, &mis[i]), END(p), TS(p, "\n");
    } else if (!strcmp(r, "assistant")) {
      /* reasoning: reasoning_content, or (Open WebUI) the text before </think> */
      const char *rt = "";
      size_t rn = 0;
      const jval *rc = json_get(ms[i], "reasoning_content");
      const char *e = NULL;
      for (const char *q = c; (q = strstr(q, "</think>")) && q < c + n; q++) e = q;
      if (e) {
        rt = c, rn = (size_t)(e - c);
        py_trim(&rt, &rn);
        if (rn >= 7 && !memcmp(rt, "<think>", 7)) rt += 7, rn -= 7;
        n -= (size_t)(e + 8 - c), c = e + 8;
        py_trim(&c, &n);
      } else if (rc && rc->t == J_STR) rt = rc->str, rn = rc->slen;
      py_trim(&rt, &rn);
      START(p), TS(p, "assistant\n<think>\n"), T(p, rt, rn), TS(p, "\n</think>\n\n"), T(p, c, n);
      const jval *tcs = json_get(ms[i], "tool_calls");
      if (tcs && tcs->t == J_ARR)
        for (const jval *tc = tcs->kid; tc; tc = tc->next)
          if (!render_call(p, tc, tc != tcs->kid ? "\n" : n ? "\n\n" : "", err, errn)) goto done;
      END(p), TS(p, "\n");
    } else if (!strcmp(r, "tool")) {
      if (i && strcmp(rs[i - 1], "tool")) START(p), TS(p, "user");
      TS(p, "\n<tool_response>\n"), emit_with_imgs(p, c, n, &mis[i]), TS(p, "\n</tool_response>");
      if (i + 1 == n_msg || strcmp(rs[i + 1], "tool")) END(p), TS(p, "\n");
    }
  }
  START(p), TS(p, think ? "assistant\n<think>\n" : "assistant\n<think>\n\n</think>\n\n");
  ok = 1;
done:
  for (i = 0; i < n_msg; i++) sb_free(&cs[i]), free(mis[i].v);
  free(mis), free(cs), free(cn), free(rs), free(ms);
  if (!ok) chat_prompt_free(p);
  return ok;
}

int chat_render(const jval *msgs, const jval *tools, int think, const char *effort, chat_prompt *p, char *err, size_t errn) {
  return chat_render_vis(msgs, tools, think, effort, 0, p, err, errn);
}

/* ---- MiMo v2.6 (XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B: its own template on qwen35) ----
 * The template's macros, as written: no message separator after
 * <|im_end|>, contents as they are (not trimmed), an assistant turn is
 * <think>REASONING</think>CONTENT then its tool calls with no newlines, and
 * the generation prompt is "<|im_start|>assistant\n" (thinking: the model
 * writes <think> itself) or "...\n<think></think>" (enable_thinking false). */
static void mimo_tools(chat_prompt *p, const jval *tools) {
  TS(p, "You are provided with the following tools:\n\n<tools>");
  for (const jval *t = tools->kid; t; t = t->next) {
    sbuf j = {0};
    py_json(&j, t);
    TS(p, "\n"), T(p, j.p, j.n);
    sb_free(&j);
  }
  TS(p, "\n</tools>");
}
/* render_tool_calls: <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>.
 * OpenAI clients send the arguments as a JSON string; a JSON object is
 * rendered as its parameters (as the template renders a mapping), any
 * other string as it is (as the template does). */
static int mimo_call(chat_prompt *p, const jval *tc, char *err, size_t errn) {
  const jval *f = json_get(tc, "function");
  if (f) tc = f;
  else if ((f = json_get(tc, "custom"))) tc = f;
  const jval *name = json_get(tc, "name"), *input = json_get(tc, "input"), *args = json_get(tc, "arguments");
  if (!name || name->t != J_STR) { snprintf(err, errn, "a tool call has no function name"); return 0; }
  TS(p, "<tool_call><function="), T(p, name->str, name->slen), TS(p, ">");
  jval *parsed = NULL;
  if (input && input->t == J_STR) T(p, input->str, input->slen);
  else if (args && args->t == J_STR && args->slen) {
    char e2[128];
    parsed = json_parse(args->str, args->slen, e2, sizeof e2);
    if (parsed && parsed->t == J_OBJ) args = parsed;
    else T(p, args->str, args->slen);
  }
  if (args && args->t == J_OBJ && !(input && input->t == J_STR))
    for (const jval *a = args->kid; a; a = a->next) {
      TS(p, "<parameter="), TS(p, a->key), TS(p, ">");
      if (a->t == J_STR) T(p, a->str, a->slen);
      else {
        sbuf j = {0};
        py_json(&j, a);
        T(p, j.p, j.n);
        sb_free(&j);
      }
      TS(p, "</parameter>");
    }
  TS(p, "</function></tool_call>");
  json_free(parsed);
  return 1;
}

int chat_render_mimo(const jval *msgs, const jval *tools, int think, chat_prompt *p, char *err, size_t errn) {
  memset(p, 0, sizeof *p);
  if (!msgs || msgs->t != J_ARR || !msgs->kid) { snprintf(err, errn, "\"messages\" must be a non-empty array"); return 0; }
  if (tools && (tools->t != J_ARR || !tools->kid)) tools = NULL;
  int ok = 0;
  if (tools) START(p), TS(p, "system\n"), mimo_tools(p, tools), END(p);
  for (const jval *m = msgs->kid; m; m = m->next) {
    const jval *r = json_get(m, "role");
    if (!r || r->t != J_STR) { snprintf(err, errn, "a message has no role"); goto done; }
    sbuf c = {0};
    if (!content_of(json_get(m, "content"), &c)) { sb_free(&c); snprintf(err, errn, "only text content is supported"); goto done; }
    const char *cs = c.p ? c.p : "";
    size_t cn = c.n;
    if (!strcmp(r->str, "assistant")) {
      const jval *rc = json_get(m, "reasoning_content");
      const char *rt = rc && rc->t == J_STR ? rc->str : "";
      size_t rn = rc && rc->t == J_STR ? rc->slen : 0;
      if (!(rc && rc->t == J_STR)) { /* Open WebUI: the reasoning in the content, before </think> */
        const char *e = NULL;
        for (const char *q = cs; (q = strstr(q, "</think>")) && q < cs + cn; q++) e = q;
        if (e) {
          rt = cs, rn = (size_t)(e - cs);
          py_trim(&rt, &rn);
          if (rn >= 7 && !memcmp(rt, "<think>", 7)) rt += 7, rn -= 7;
          py_trim(&rt, &rn);
          cn -= (size_t)(e + 8 - cs), cs = e + 8;
          py_trim(&cs, &cn);
        }
      }
      START(p), TS(p, "assistant\n<think>"), T(p, rt, rn), TS(p, "</think>"), T(p, cs, cn);
      const jval *tcs = json_get(m, "tool_calls");
      if (tcs && tcs->t == J_ARR)
        for (const jval *tc = tcs->kid; tc; tc = tc->next)
          if (!mimo_call(p, tc, err, errn)) { sb_free(&c); goto done; }
      END(p);
    } else {
      START(p), T(p, r->str, r->slen), TS(p, "\n"), T(p, cs, cn);
      const jval *mt = json_get(m, "tools");
      if (mt && mt->t == J_ARR && mt->kid) {
        if (cn) TS(p, "\n\n");
        mimo_tools(p, mt);
      }
      END(p);
    }
    sb_free(&c);
  }
  START(p), TS(p, think ? "assistant\n" : "assistant\n<think></think>");
  ok = 1;
done:
  if (!ok) chat_prompt_free(p);
  return ok;
}

/* ---- the model's tool calls ---- */

static const char *find(const char *s, const char *e, const char *pat) {
  const size_t k = strlen(pat);
  for (const char *q = s; q + k <= e; q++)
    if (!memcmp(q, pat, k)) return q;
  return NULL;
}

/* the schema type of parameter key of tool name ("" if not known) */
static const char *param_type(const jval *tools, const char *name, size_t nn, const char *key, size_t kn) {
  if (!tools) return "";
  for (const jval *t = tools->kid; t; t = t->next) {
    const jval *f = json_get(t, "function");
    if (!f) f = t;
    const jval *fn = json_get(f, "name");
    if (!fn || fn->t != J_STR || fn->slen != nn || memcmp(fn->str, name, nn)) continue;
    const jval *props = json_get(json_get(f, "parameters"), "properties");
    for (const jval *pr = props ? props->kid : NULL; pr; pr = pr->next)
      if (strlen(pr->key) == kn && !memcmp(pr->key, key, kn)) {
        const jval *ty = json_get(pr, "type");
        if (ty && ty->t == J_STR) return ty->str;
        if (ty && ty->t == J_ARR && ty->kid && ty->kid->t == J_STR) return ty->kid->str; /* ["integer","null"] */
        return "";
      }
  }
  return "";
}

const char *chat_param_type(const jval *tools, const char *name, size_t nn, const char *key, size_t kn) {
  return param_type(tools, name, nn, key, kn);
}

int chat_tool_calls(const char *s, size_t n, const jval *tools, const char *id_prefix, sbuf *out) {
  const char *e = s + n, *q = s;
  sbuf arr = {0};
  int k = 0;
  sb_add(&arr, "[", 1);
  while ((q = find(q, e, "<tool_call>"))) {
    const char *f = find(q, e, "<function="), *fe;
    if (!f || !(fe = find(f, e, ">"))) goto bad;
    const char *name = f + 10;
    size_t nn = (size_t)(fe - name);
    const char *end = find(fe, e, "</function>");
    if (!end) goto bad;
    if (k) sb_add(&arr, ",", 1);
    sb_printf(&arr, "{\"id\":\"%s%d\",\"type\":\"function\",\"function\":{\"name\":", id_prefix, k);
    sb_json_str(&arr, name, nn);
    sbuf args = {0};
    sb_add(&args, "{", 1);
    int np = 0;
    for (const char *pp = fe + 1; (pp = find(pp, end, "<parameter=")) != NULL;) {
      const char *ke = find(pp, end, ">");
      if (!ke) { sb_free(&args); goto bad; }
      const char *key = pp + 11, *v = ke + 1, *ve = find(v, end, "</parameter>");
      if (!ve) ve = end; /* the model may leave out the last </parameter> */
      const size_t kn = (size_t)(ke - key);
      const char *vs = v;
      size_t vn = (size_t)(ve - v);
      if (vn && vs[0] == '\n') vs++, vn--; /* the template writes <parameter=K>\nV\n</parameter> */
      if (vn && vs[vn - 1] == '\n') vn--;
      if (np++) sb_add(&args, ", ", 2);
      sb_json_str(&args, key, kn);
      sb_add(&args, ": ", 2);
      const char *ty = param_type(tools, name, nn, key, kn);
      jval *jv = NULL;
      if (strcmp(ty, "string") && *ty) { /* a number, boolean, object or array: the value as JSON if it is valid */
        char e2[64];
        jv = json_parse(vs, vn, e2, sizeof e2);
      }
      if (jv) py_json(&args, jv);
      else sb_json_str(&args, vs, vn);
      json_free(jv);
      pp = ve == end ? end : ve + 12;
    }
    sb_add(&args, "}", 1);
    sb_puts(&arr, ",\"arguments\":");
    sb_json_str(&arr, args.p, args.n);
    sb_puts(&arr, "}}");
    sb_free(&args);
    k++;
    q = end + 11;
  }
  sb_add(&arr, "]", 1);
  if (!k) goto bad;
  sb_add(out, arr.p, arr.n);
  sb_free(&arr);
  return k;
bad:
  sb_free(&arr);
  return 0;
}

/* ---- flint ---- */

/* the template's flint_text macro on a string: without NUL, <think> and
 * </think>; the control tokens escaped (&lt;|user|&gt;) */
static void flint_text(sbuf *b, const char *s, size_t n) {
  static const char *const ctl[] = {"<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|developer|>", "<|user|>", "<|assistant|>",
                                    "<|observation|>"};
  for (size_t i = 0; i < n;) {
    if (!s[i]) { i++; continue; }
    if (s[i] == '<') {
      if (n - i >= 7 && !memcmp(s + i, "<think>", 7)) { i += 7; continue; }
      if (n - i >= 8 && !memcmp(s + i, "</think>", 8)) { i += 8; continue; }
      size_t k = 0;
      for (; k < sizeof ctl / sizeof ctl[0]; k++) {
        const size_t L = strlen(ctl[k]);
        if (n - i >= L && !memcmp(s + i, ctl[k], L)) {
          sb_puts(b, "&lt;"), sb_add(b, ctl[k] + 1, L - 2), sb_puts(b, "&gt;");
          i += L;
          break;
        }
      }
      if (k < sizeof ctl / sizeof ctl[0]) continue;
    }
    sb_add(b, s + i, 1), i++;
  }
}

/* a flint tool call: <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>
 * (V: a string as it is, any other value as JSON) */
static int flint_call(sbuf *b, const jval *tc, char *err, size_t errn) {
  const jval *f = json_get(tc, "function");
  if (f) tc = f;
  const jval *name = json_get(tc, "name"), *args = json_get(tc, "arguments");
  if (!name || name->t != J_STR) { snprintf(err, errn, "a tool call has no function name"); return 0; }
  sb_puts(b, "<tool_call>"), sb_add(b, name->str, name->slen);
  jval *parsed = NULL;
  if (args && args->t == J_STR && args->slen) { /* OpenAI sends the arguments as a JSON string */
    char e2[128];
    parsed = json_parse(args->str, args->slen, e2, sizeof e2);
    if (!parsed || parsed->t != J_OBJ) {
      json_free(parsed);
      snprintf(err, errn, "the arguments of a tool call are not a JSON object");
      return 0;
    }
    args = parsed;
  }
  if (args && args->t == J_OBJ)
    for (const jval *a = args->kid; a; a = a->next) {
      sb_puts(b, "<arg_key>"), sb_puts(b, a->key), sb_puts(b, "</arg_key><arg_value>");
      if (a->t == J_STR) sb_add(b, a->str, a->slen);
      else py_json(b, a);
      sb_puts(b, "</arg_value>");
    }
  sb_puts(b, "</tool_call>");
  json_free(parsed);
  return 1;
}

/* Jinja's truth of a message field: present, not null, not "" */
static int truthy(const jval *v) { return v && v->t != J_NULL && v->t != J_FALSE && !(v->t == J_STR && !v->slen) && !(v->t == J_ARR && !v->kid); }

int chat_render_flint(const jval *msgs, const jval *tools, chat_prompt *p, char *err, size_t errn) {
  memset(p, 0, sizeof *p);
  if (!msgs || msgs->t != J_ARR || !msgs->kid) { snprintf(err, errn, "\"messages\" must be a non-empty array"); return 0; }
  if (tools && (tools->t != J_ARR || !tools->kid)) tools = NULL;
  sbuf *b = &p->text, pre = {0};
  if (tools) { /* the tool preamble */
    sb_puts(&pre, "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with "
                  "function signatures within <tools></tools> XML tags:\n<tools>");
    for (const jval *t = tools->kid; t; t = t->next) {
      const jval *fn = json_get(t, "function");
      py_json(&pre, fn ? fn : t); /* back to back: trim_blocks (as transformers renders the template) */
    }
    sb_puts(&pre, "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n"
                  "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n");
  }
  int pending = 1, ok = 0;
  for (const jval *m = msgs->kid; m; m = m->next) {
    const jval *r = json_get(m, "role");
    if (!r || r->t != J_STR) { snprintf(err, errn, "a message has no role"); goto done; }
    const char *role = r->str;
    if (pending && strcmp(role, "system")) sb_add(b, pre.p ? pre.p : "", pre.n), pending = 0;
    sbuf c = {0};
    if (!content_of(json_get(m, "content"), &c)) { sb_free(&c); snprintf(err, errn, "only text content is supported"); goto done; }
    if (!strcmp(role, "system") || !strcmp(role, "developer") || !strcmp(role, "user")) {
      sb_puts(b, !strcmp(role, "system") ? "<|system|>" : !strcmp(role, "developer") ? "<|developer|>" : "<|user|>");
      flint_text(b, c.p ? c.p : "", c.n);
    } else if (!strcmp(role, "assistant")) {
      sb_puts(b, "<|assistant|>");
      const jval *rc = json_get(m, "reasoning_content");
      if (truthy(rc) && rc->t == J_STR) sb_puts(b, "<think>"), sb_add(b, rc->str, rc->slen), sb_puts(b, "</think>");
      if (c.n) sb_add(b, c.p, c.n);
      const jval *tcs = json_get(m, "tool_calls");
      if (tcs && tcs->t == J_ARR)
        for (const jval *tc = tcs->kid; tc; tc = tc->next)
          if (!flint_call(b, tc, err, errn)) { sb_free(&c); goto done; }
    } else if (!strcmp(role, "tool")) {
      sb_puts(b, "<|observation|><tool_response>"), flint_text(b, c.p ? c.p : "", c.n), sb_puts(b, "</tool_response>");
    }
    sb_free(&c);
  }
  if (pending) sb_add(b, pre.p ? pre.p : "", pre.n);
  sb_puts(b, "<|assistant|>");
  seg_push(p, SEG_TEXT, 0, b->n);
  ok = 1;
done:
  sb_free(&pre);
  if (!ok) chat_prompt_free(p);
  return ok;
}

int chat_tool_calls_flint(const char *s, size_t n, const jval *tools, const char *id_prefix, sbuf *out) {
  const char *e = s + n, *q = s;
  sbuf arr = {0};
  int k = 0;
  sb_add(&arr, "[", 1);
  while ((q = find(q, e, "<tool_call>"))) {
    const char *name = q + 11, *end = find(name, e, "</tool_call>");
    if (!end) end = e; /* the model may stop before </tool_call> */
    const char *ne = find(name, end, "<arg_key>");
    if (!ne) ne = end;
    const char *nm = name;
    size_t nn = (size_t)(ne - name);
    py_trim(&nm, &nn);
    if (!nn) goto bad;
    if (k) sb_add(&arr, ",", 1);
    sb_printf(&arr, "{\"id\":\"%s%d\",\"type\":\"function\",\"function\":{\"name\":", id_prefix, k);
    sb_json_str(&arr, nm, nn);
    sbuf args = {0};
    sb_add(&args, "{", 1);
    int np = 0;
    for (const char *pp = ne; (pp = find(pp, end, "<arg_key>")) != NULL;) {
      const char *key = pp + 9, *ke = find(key, end, "</arg_key>");
      if (!ke) { sb_free(&args); goto bad; }
      const char *v = find(ke, end, "<arg_value>");
      if (!v) { sb_free(&args); goto bad; }
      v += 11;
      const char *ve = find(v, end, "</arg_value>");
      if (!ve) ve = end;
      const size_t kn = (size_t)(ke - key), vn = (size_t)(ve - v);
      if (np++) sb_add(&args, ", ", 2);
      sb_json_str(&args, key, kn);
      sb_add(&args, ": ", 2);
      const char *ty = param_type(tools, nm, nn, key, kn);
      jval *jv = NULL;
      if (strcmp(ty, "string")) { /* not a string by the schema (or unknown): the value as JSON if it is valid */
        char e2[64];
        jv = json_parse(v, vn, e2, sizeof e2);
        if (jv && !*ty && jv->t == J_STR) json_free(jv), jv = NULL; /* unknown type: a JSON string stays the text itself */
      }
      if (jv) py_json(&args, jv);
      else sb_json_str(&args, v, vn);
      json_free(jv);
      pp = ve == end ? end : ve + 12;
    }
    sb_add(&args, "}", 1);
    sb_puts(&arr, ",\"arguments\":");
    sb_json_str(&arr, args.p, args.n);
    sb_puts(&arr, "}}");
    sb_free(&args);
    k++;
    q = end == e ? e : end + 12;
  }
  sb_add(&arr, "]", 1);
  if (!k) goto bad;
  sb_add(out, arr.p, arr.n);
  sb_free(&arr);
  return k;
bad:
  sb_free(&arr);
  return 0;
}

/* ---- Inkling-2 (inkling2 / inkling2_loop: chat_template.jinja) ---- */

static void CTRL(chat_prompt *p, const char *s) { sb_puts(&p->text, s); }

static int inkling2_effort(const jval *ef_val, const char *ef_str, char *out, size_t outn, char *err, size_t errn) {
  static const struct { const char *k, *lit; double v; } map[] = {
      {"none", "0", 0.0}, {"minimal", "0.1", 0.1}, {"low", "0.2", 0.2}, {"medium", "0.7", 0.7},
      {"high", "0.9", 0.9}, {"xhigh", "0.99", 0.99}, {"max", "0.99", 0.99},
  };
  double num = 0.9;
  const char *lit = "0.9";
  if (ef_val && ef_val->t != J_NULL) {
    if (ef_val->t == J_STR) {
      const char *k = ef_val->str;
      size_t kn = ef_val->slen;
      py_trim(&k, &kn);
      int hit = 0;
      for (size_t i = 0; i < sizeof map / sizeof map[0]; i++)
        if (strlen(map[i].k) == kn && !memcmp(map[i].k, k, kn)) { num = map[i].v, lit = map[i].lit, hit = 1; break; }
      if (!hit) { snprintf(err, errn, "Unknown reasoning_effort: %s", ef_val->str); return 0; }
    } else if (ef_val->t == J_NUM) {
      num = ef_val->num, lit = NULL;
    } else {
      snprintf(err, errn, "reasoning_effort must be a string or number");
      return 0;
    }
  } else if (ef_str) {
    const char *k = ef_str;
    size_t kn = strlen(ef_str);
    py_trim(&k, &kn);
    int hit = 0;
    for (size_t i = 0; i < sizeof map / sizeof map[0]; i++)
      if (strlen(map[i].k) == kn && !memcmp(map[i].k, k, kn)) { num = map[i].v, lit = map[i].lit, hit = 1; break; }
    if (!hit) {
      char buf[64], *end = NULL;
      if (kn && kn < sizeof buf) {
        memcpy(buf, k, kn), buf[kn] = 0;
        num = strtod(buf, &end);
        if (end == buf + kn) lit = NULL, hit = 1;
      }
      if (!hit) { snprintf(err, errn, "Unknown reasoning_effort: %s", ef_str); return 0; }
    }
  }
  if (!isfinite(num) || num < 0.0 || num > 0.99) {
    snprintf(err, errn, "reasoning_effort must be in [0.0, 0.99]");
    return 0;
  }
  if (num == 0.0) snprintf(out, outn, "0");
  else if (lit) snprintf(out, outn, "%s", lit);
  else snprintf(out, outn, "%.15g", num);
  return 1;
}

static void emit_inkling2_effort(chat_prompt *p, const char *eff_txt) {
  CTRL(p, "<|message_system|>");
  CTRL(p, "<|content_text|>");
  TS(p, "Thinking effort level: ");
  TS(p, eff_txt);
  CTRL(p, "<|end_message|>");
}

static int emit_inkling2_tools(chat_prompt *p, const jval *tools, char *err, size_t errn) {
  CTRL(p, "<|message_system|>");
  TS(p, "tool_declare");
  CTRL(p, "<|content_xml|>");
  sbuf jb = {0};
  sb_add(&jb, "[", 1);
  for (const jval *t = tools->kid; t; t = t->next) {
    if (t != tools->kid) sb_add(&jb, ",", 1);
    const jval *fn = json_get(t, "function");
    if (!fn) fn = t;
    if (fn->t != J_OBJ) { sb_free(&jb); snprintf(err, errn, "tool entry must be an object"); return 0; }
    const jval *name = json_get(fn, "name");
    if (!name || name->t != J_STR) { sb_free(&jb); snprintf(err, errn, "tool function name must be a string"); return 0; }
    const jval *desc = json_get(fn, "description");
    const jval *params = json_get(fn, "parameters");
    const jval *ttype = json_get(t, "type");
    sb_puts(&jb, "{\"description\":");
    if (truthy(desc)) py_json_opts(&jb, desc, 1, 1);
    else sb_puts(&jb, "\"\"");
    sb_puts(&jb, ",\"name\":");
    py_str(&jb, name->str, name->slen);
    sb_puts(&jb, ",\"parameters\":");
    if (truthy(params)) py_json_opts(&jb, params, 1, 1);
    else sb_puts(&jb, "{}");
    sb_puts(&jb, ",\"type\":");
    if (truthy(ttype)) py_json_opts(&jb, ttype, 1, 1);
    else sb_puts(&jb, "\"function\"");
    sb_add(&jb, "}", 1);
  }
  sb_add(&jb, "]", 1);
  T(p, jb.p, jb.n);
  sb_free(&jb);
  CTRL(p, "<|end_message|>");
  return 1;
}

static int emit_inkling2_call(chat_prompt *p, const jval *tc, const jval *msg_name, char *err, size_t errn) {
  const jval *fn = json_get(tc, "function");
  if (!fn || fn->t != J_OBJ) { snprintf(err, errn, "tool call missing function object"); return 0; }
  const jval *fnn = json_get(fn, "name");
  if (!fnn || fnn->t != J_STR) { snprintf(err, errn, "tool call function name must be a string"); return 0; }
  if (truthy(msg_name) && msg_name->t == J_STR &&
      (msg_name->slen != fnn->slen || memcmp(msg_name->str, fnn->str, fnn->slen))) {
    snprintf(err, errn, "assistant name %s does not match its tool call %s", msg_name->str, fnn->str);
    return 0;
  }
  const jval *args = json_get(fn, "arguments");
  jval *parsed = NULL;
  if (!args || args->t == J_NULL || (args->t == J_STR && args->slen == 0)) {
    args = NULL;
  } else if (args->t == J_STR) {
    char e2[128];
    parsed = json_parse(args->str, args->slen, e2, sizeof e2);
    if (!parsed || parsed->t != J_OBJ) {
      json_free(parsed);
      snprintf(err, errn, "tool call arguments are not a JSON object");
      return 0;
    }
    args = parsed;
  } else if (args->t != J_OBJ) {
    snprintf(err, errn, "tool call arguments must be an object");
    return 0;
  }
  CTRL(p, "<|message_model|>");
  T(p, fnn->str, fnn->slen);
  CTRL(p, "<|content_invoke_tool_json|>");
  sbuf jb = {0};
  sb_puts(&jb, "{\"name\":");
  py_str(&jb, fnn->str, fnn->slen);
  sb_puts(&jb, ",\"args\":{");
  if (args && args->kid) {
    size_t nk = 0;
    for (const jval *a = args->kid; a; a = a->next) nk++;
    const jval *stk[64], **ks = nk <= 64 ? stk : malloc(nk * sizeof *ks);
    if (!ks) abort();
    size_t i = 0;
    for (const jval *a = args->kid; a; a = a->next) ks[i++] = a;
    if (nk > 1) qsort(ks, nk, sizeof *ks, cmp_jkey);
    for (i = 0; i < nk; i++) {
      if (i) sb_add(&jb, ",", 1);
      py_str(&jb, ks[i]->key, strlen(ks[i]->key));
      sb_add(&jb, ":", 1);
      py_json_opts(&jb, ks[i], 1, 0);
    }
    if (ks != stk) free(ks);
  }
  sb_puts(&jb, "}}");
  T(p, jb.p, jb.n);
  sb_free(&jb);
  CTRL(p, "<|end_message|>");
  json_free(parsed);
  return 1;
}

/* Render messages + tools according to Inkling-2-Mini's chat_template.jinja.
 * Control tokens are written to p->text outside SEG_TEXT segments so serve.c
 * encodes user/assistant text with parse_control=0 while template tags use parse_control=1. */
int chat_render_inkling2(const jval *msgs, const jval *tools, const jval *effort_jv, const char *effort_str,
                         int allow_images, chat_prompt *p, char *err, size_t errn) {
  memset(p, 0, sizeof *p);
  if (!msgs || msgs->t != J_ARR || !msgs->kid) { snprintf(err, errn, "\"messages\" must be a non-empty array"); return 0; }
  if (tools && (tools->t != J_ARR || !tools->kid)) tools = NULL;
  char eff_txt[64];
  if (!inkling2_effort(effort_jv, effort_str, eff_txt, sizeof eff_txt, err, errn)) return 0;
  int ok = 0, effort_emitted = 0;
  if (tools && !emit_inkling2_tools(p, tools, err, errn)) goto done;
  for (const jval *m = msgs->kid; m; m = m->next) {
    if (m->t != J_OBJ) { snprintf(err, errn, "each message must be an object"); goto done; }
    const jval *r = json_get(m, "role");
    if (!r || r->t != J_STR) { snprintf(err, errn, "a message has no role"); goto done; }
    const char *role = r->str, *rtok = NULL;
    if (!strcmp(role, "user")) rtok = "<|message_user|>";
    else if (!strcmp(role, "assistant")) rtok = "<|message_model|>";
    else if (!strcmp(role, "system")) rtok = "<|message_system|>";
    else if (!strcmp(role, "tool")) rtok = "<|message_tool|>";
    else { snprintf(err, errn, "Unknown message role: %s", role); goto done; }
    if (!effort_emitted && strcmp(role, "system") != 0) {
      emit_inkling2_effort(p, eff_txt);
      effort_emitted = 1;
    }
    sbuf rname = {0};
    const jval *mnm = json_get(m, "name");
    if (!strcmp(role, "tool")) {
      if (truthy(mnm) && mnm->t == J_STR) sb_add(&rname, mnm->str, mnm->slen);
      else {
        const jval *tcid = json_get(m, "tool_call_id");
        if (truthy(tcid) && tcid->t == J_STR) {
          for (const jval *prev = msgs->kid; prev; prev = prev->next) {
            const jval *pr = json_get(prev, "role"), *ptcs = json_get(prev, "tool_calls");
            if (!pr || pr->t != J_STR || strcmp(pr->str, "assistant") || !ptcs || ptcs->t != J_ARR) continue;
            for (const jval *tc = ptcs->kid; tc; tc = tc->next) {
              const jval *id = json_get(tc, "id"), *fn = json_get(tc, "function");
              const jval *fnn = fn ? json_get(fn, "name") : NULL;
              if (id && id->t == J_STR && id->slen == tcid->slen && !memcmp(id->str, tcid->str, id->slen) &&
                  fnn && fnn->t == J_STR)
                sb_add(&rname, fnn->str, fnn->slen);
            }
          }
        }
      }
    } else if (!strcmp(role, "assistant") && truthy(mnm) && mnm->t == J_STR) {
      sb_add(&rname, mnm->str, mnm->slen);
    }
#define INK_RTOK() do { CTRL(p, rtok); if (rname.n) T(p, rname.p, rname.n); } while (0)
    const int is_asst = !strcmp(role, "assistant");
    const jval *rc = json_get(m, "reasoning_content"), *tcs = json_get(m, "tool_calls"), *c = json_get(m, "content");
    const int has_rc = truthy(rc) && rc->t == J_STR, has_tc = tcs && tcs->t == J_ARR && tcs->kid != NULL;
    if (is_asst && has_rc) {
      INK_RTOK();
      CTRL(p, "<|content_thinking|>");
      T(p, rc->str, rc->slen);
      CTRL(p, "<|end_message|>");
    }
    if (c && c->t == J_STR) {
      if (c->slen > 0 || !(is_asst && (has_tc || has_rc))) {
        INK_RTOK();
        CTRL(p, "<|content_text|>");
        T(p, c->str, c->slen);
        CTRL(p, "<|end_message|>");
      }
    } else if (c && c->t == J_ARR && c->kid) {
      for (const jval *part = c->kid; part; part = part->next) {
        if (part->t == J_STR) {
          INK_RTOK();
          CTRL(p, "<|content_text|>");
          T(p, part->str, part->slen);
          CTRL(p, "<|end_message|>");
        } else if (part->t == J_OBJ) {
          const jval *pt = json_get(part, "type");
          if (!pt || pt->t == J_NULL || (pt->t == J_STR && (!strcmp(pt->str, "text") || !strcmp(pt->str, "input_text")))) {
            const jval *tx = json_get(part, "text");
            INK_RTOK();
            CTRL(p, "<|content_text|>");
            if (tx && tx->t == J_STR) T(p, tx->str, tx->slen);
            CTRL(p, "<|end_message|>");
          } else if (pt->t == J_STR &&
                     (!strcmp(pt->str, "image") || !strcmp(pt->str, "input_image") || !strcmp(pt->str, "image_url"))) {
            if (strcmp(role, "user") != 0 && strcmp(role, "tool") != 0) {
              sb_free(&rname);
              snprintf(err, errn, "Image content is only supported for user and tool messages");
              goto done;
            }
            if (!allow_images) {
              sb_free(&rname);
              snprintf(err, errn, "only text content is supported (pass --mmproj to enable images)");
              goto done;
            }
            const char *url = NULL;
            size_t ulen = 0;
            int ir = extract_image_url(part, &url, &ulen);
            if (ir == 0) {
              const jval *u = json_get(part, "url");
              if (u && u->t == J_STR) { url = u->str; ulen = u->slen; ir = 1; }
            }
            if (ir <= 0 || !ulen) {
              sb_free(&rname);
              snprintf(err, errn, "image content part missing valid image_url");
              goto done;
            }
            INK_RTOK();
            CTRL(p, "<|content_image|>");
            INK_IMG(p, url, ulen);
            CTRL(p, "<|end_message|>");
          } else if (pt->t == J_STR &&
                     (!strcmp(pt->str, "audio") || !strcmp(pt->str, "input_audio") || !strcmp(pt->str, "audio_url"))) {
            sb_free(&rname);
            snprintf(err, errn, "Inkling 2 does not accept audio input");
            goto done;
          } else {
            sb_free(&rname);
            snprintf(err, errn, "Unsupported content part type: %s", pt && pt->t == J_STR ? pt->str : "(invalid)");
            goto done;
          }
        } else {
          sb_free(&rname);
          snprintf(err, errn, "unsupported content part");
          goto done;
        }
      }
    } else if (c && c->t != J_NULL && c->t != J_ARR) {
      sb_free(&rname);
      snprintf(err, errn, "unsupported message content");
      goto done;
    }
    if (is_asst && has_tc)
      for (const jval *tc = tcs->kid; tc; tc = tc->next)
        if (!emit_inkling2_call(p, tc, mnm, err, errn)) { sb_free(&rname); goto done; }
    if (is_asst) CTRL(p, "<|content_model_end_sampling|>");
    sb_free(&rname);
#undef INK_RTOK
  }
  if (!effort_emitted) emit_inkling2_effort(p, eff_txt);
  ok = 1;
done:
  if (!ok) chat_prompt_free(p);
  return ok;
}

/* Parse one Inkling-2 <|content_invoke_tool_json|> payload s[0..n) into
 * function name and serialized JSON arguments string. Returns 1 on success, 0 on error. */
int chat_parse_inkling2_tool(const char *s, size_t n, sbuf *name_out, sbuf *args_out) {
  char err[64];
  jval *v = json_parse(s, n, err, sizeof err);
  if (!v || v->t != J_OBJ) { json_free(v); return 0; }
  const jval *nm = json_get(v, "name");
  if (!nm || nm->t != J_STR || !nm->slen) { json_free(v); return 0; }
  const jval *args = json_get(v, "args");
  if (!args) args = json_get(v, "arguments");
  sb_add(name_out, nm->str, nm->slen);
  if (!args || args->t == J_NULL) sb_puts(args_out, "{}");
  else if (args->t == J_STR) sb_add(args_out, args->str, args->slen);
  else py_json_opts(args_out, args, 1, 0);
  json_free(v);
  return 1;
}
