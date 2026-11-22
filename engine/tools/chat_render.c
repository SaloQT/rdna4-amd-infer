/* chat_render.c -- print the prompt that ie-serve makes for a request.
 *   chat_render [--flint | --mimo] REQUEST.json   (a /v1/chat/completions body)
 * For checks against the model's Jinja template (tools/chat_check.py). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/chat.h"
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main(int argc, char **argv) {
  const int fl = argc == 3 && !strcmp(argv[1], "--flint"), mi = argc == 3 && !strcmp(argv[1], "--mimo");
  if (argc != 2 && !fl && !mi) { fprintf(stderr, "usage: chat_render [--flint | --mimo] REQUEST.json\n"); return 2; }
  const char *path = argv[argc - 1];
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); return 1; }
  sbuf b = {0};
  char buf[65536];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, f))) sb_add(&b, buf, k);
  fclose(f);
  char err[256];
  jval *req = json_parse(b.p, b.n, err, sizeof err);
  if (!req) { fprintf(stderr, "%s\n", err); return 1; }
  int think = 1;
  const jval *kw = json_get(req, "chat_template_kwargs"), *et = json_get(kw, "enable_thinking");
  if (et && (et->t == J_TRUE || et->t == J_FALSE)) think = et->t == J_TRUE;
  const char *effort = "xhigh";
  const jval *ef = json_get(req, "reasoning_effort");
  if (ef && ef->t == J_STR) effort = ef->str;
  chat_prompt p;
  const jval *msgs = json_get(req, "messages"), *tools = json_get(req, "tools");
  if (fl ? !chat_render_flint(msgs, tools, &p, err, sizeof err)
         : mi ? !chat_render_mimo(msgs, tools, think, &p, err, sizeof err)
              : !chat_render(msgs, tools, think, effort, &p, err, sizeof err)) {
    fprintf(stderr, "error: %s\n", err);
    return 1;
  }
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY); /* the prompt bytes as they are (no CRLF) */
#endif
  fwrite(p.text.p, 1, p.text.n, stdout);
  chat_prompt_free(&p);
  json_free(req);
  sb_free(&b);
  return 0;
}
