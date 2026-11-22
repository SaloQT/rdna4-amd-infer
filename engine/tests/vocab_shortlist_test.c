/* vocab_shortlist_test.c -- host-only check of the draft vocabulary shortlist loader (model.c): the SHA-256 against
 * Python's hashlib, the manifest checks (each refusal), unsorted ids, and the real 64k map when its files are present.
 *   build/vocab_shortlist_test [MANIFEST]   (run from the repo root; MANIFEST: the real map-65536.manifest.json) */
#include <string.h>

#include "../src/common.h"
#include "../src/model.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL line %d: ", __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static void put(const char *path, const char *text) {
  FILE *f = fopen(path, "wb");
  if (!f || fwrite(text, 1, strlen(text), f) != strlen(text)) ie_die("cannot write %s", path);
  fclose(f);
}

/* a manifest (build/vs_test.manifest.json, for a model of 1100 tokens) for the ids text in build/vs_ids.txt; its vocabulary
 * entry may differ from the model's. sha: the sha256 written in the manifest (NULL: the text's own) */
static uint32_t *try_load(const char *ids_text, const char *ids_path, uint32_t vocab, uint32_t size, const char *eog, const char *sha, uint32_t *n, char *err) {
  char hex[65], man[1600];
  put("build/vs_ids.txt", ids_text);
  ie_sha256_hex(ids_text, strlen(ids_text), hex);
  snprintf(man, sizeof man,
           "{\n \"schema\": \"rdna4.mtp.vocab-shortlist.v1\",\n \"vocabulary\": %u,\n \"size\": %u,\n \"ids_text\": \"%s\",\n"
           " \"ids_text_sha256\": \"%s\",\n \"required_eog_ids\": [%s],\n \"integration_allowed\": false\n}\n",
           vocab, size, ids_path, sha ? sha : hex, eog);
  put("build/vs_test.manifest.json", man);
  err[0] = 0;
  return ie_load_vocab_shortlist("build/vs_test.manifest.json", 1100, n, err, 600);
}

int main(int argc, char **argv) {
  /* SHA-256 */
  static const struct { size_t n; const char *hex; } V[] = {
      {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
      {56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
      {63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
      {64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
      {65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
      {119, "da18797ed7c3a777f0847f429724a2d8cd5138e6ed2895c3fa1a6d39d18f7ec6"},
      {120, "f52b23db1fbb6ded89ef42a23ce0c8922c45f25c50b568a93bf1c075420bbb7c"},
      {1000, "4e4c294b331f7a2099a379bec34b9f9fc03dc46ab465d998f4d683da53487e6d"}};
  uint8_t buf[1000];
  for (int i = 0; i < 1000; i++) buf[i] = (uint8_t)(i % 251);
  char hex[65];
  for (unsigned i = 0; i < sizeof V / sizeof V[0]; i++) {
    ie_sha256_hex(buf, V[i].n, hex);
    CHECK(!strcmp(hex, V[i].hex), "sha256 of %zu bytes: %s", V[i].n, hex);
  }
  ie_sha256_hex("abc", 3, hex);
  CHECK(!strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "sha256 abc");

  uint32_t n = 0, *ids;
  char err[600];
  /* a good small shortlist, unsorted, CRLF: loads ascending */
  ids = try_load("5\r\n3\r\n9\r\n1\r\n", "build/vs_ids.txt", 1100, 4, "9", NULL, &n, err);
  CHECK(ids && n == 4 && ids[0] == 1 && ids[1] == 3 && ids[2] == 5 && ids[3] == 9, "unsorted ids: %s", ids ? "wrong ids" : err);
  free(ids);
  /* the ids path as a Windows path that does not exist: the file next to the manifest is used */
  ids = try_load("5\n3\n9\n1\n", "C:\\\\no\\\\such\\\\vs_ids.txt", 1100, 4, "1, 9", NULL, &n, err);
  CHECK(ids && n == 4, "fallback next to the manifest: %s", err);
  free(ids);
  /* refusals */
  ids = try_load("5\n3\n9\n1\n", "build/vs_ids.txt", 1100, 4, "9", "0000000000000000000000000000000000000000000000000000000000000000", &n, err);
  CHECK(!ids && strstr(err, "sha256"), "wrong sha256 must be refused: %s", err);
  ids = try_load("5\n3\n9\n1\n", "build/vs_ids.txt", 1100, 4, "9", "6C3133CE436EEE504564CE80DEFF6C5C45BE999E1A759643532BC10A46050C65", &n, err);
  CHECK(!ids && strstr(err, "sha256"), "the sha256 of other ids must be refused: %s", err);
  ids = try_load("5\n3\n9\n1\n", "build/vs_ids.txt", 1099, 4, "9", NULL, &n, err);
  CHECK(!ids && strstr(err, "vocabulary"), "wrong vocabulary must be refused: %s", err);
  ids = try_load("5\n3\n9\n1\n", "build/vs_ids.txt", 1100, 5, "9", NULL, &n, err);
  CHECK(!ids && strstr(err, "holds 4 ids"), "wrong size must be refused: %s", err);
  ids = try_load("5\n3\n9\n1\n", "build/vs_ids.txt", 1100, 3, "9", NULL, &n, err);
  CHECK(!ids && strstr(err, "more than size"), "ids past size must be refused: %s", err);
  ids = try_load("5\n5\n9\n1\n", "build/vs_ids.txt", 1100, 4, "9", NULL, &n, err);
  CHECK(!ids && strstr(err, "twice"), "a repeated id must be refused: %s", err);
  ids = try_load("5\n3\n9\n1000\n", "build/vs_ids.txt", 1100, 4, "9", NULL, &n, err);
  CHECK(ids && ids[3] == 1000, "id 1000 < vocabulary 1100 loads: %s", err);
  free(ids);
  ids = try_load("5\n3\n9\n1100\n", "build/vs_ids.txt", 1100, 4, "9", NULL, &n, err);
  CHECK(!ids && strstr(err, ">= vocabulary"), "an id >= vocabulary must be refused: %s", err);
  ids = try_load("5\n3\n9\n-1\n", "build/vs_ids.txt", 1100, 4, "9", NULL, &n, err);
  CHECK(!ids && strstr(err, "unexpected character"), "a bad character must be refused: %s", err);
  ids = try_load("5\n3\n9\n1\n", "build/vs_ids.txt", 1100, 4, "9, 1050", NULL, &n, err);
  CHECK(!ids && strstr(err, "1050"), "a missing end-of-generation id must be refused: %s", err);
  ids = try_load("5\n3\n9\n1\n", "build/vs_ids.txt", 1100, 4, "", NULL, &n, err);
  CHECK(ids != NULL, "an empty required_eog_ids list loads: %s", err);
  free(ids);
  CHECK(!ie_load_vocab_shortlist("build/no-such.manifest.json", 1100, &n, err, sizeof err) && strstr(err, "cannot read"), "missing manifest: %s", err);

  /* the real 64k map */
  const char *real = argc > 1 ? argv[1] : "../../data/mtp-vocab-pilot/maps/default/map-65536.manifest.json";
  FILE *f = fopen(real, "rb");
  if (f) {
    fclose(f);
    ids = ie_load_vocab_shortlist(real, 248320, &n, err, sizeof err);
    CHECK(ids && n == 65536, "real map: %s", ids ? "wrong size" : err);
    if (ids) {
      int asc = 1;
      for (uint32_t i = 1; i < n; i++) asc &= ids[i] > ids[i - 1];
      CHECK(asc && ids[0] == 0 && ids[n - 1] == 248065, "real map ascending, first %u last %u", ids[0], ids[n - 1]);
      printf("real map: %u ids, first %u, last %u\n", n, ids[0], ids[n - 1]);
      free(ids);
    }
    ids = ie_load_vocab_shortlist(real, 248319, &n, err, sizeof err);
    CHECK(!ids && strstr(err, "vocabulary"), "real map with another vocabulary must be refused: %s", err);
  } else {
    printf("real map: %s not found, skipped\n", real);
  }
  printf(fails ? "vocab_shortlist_test: %d FAILED\n" : "vocab_shortlist_test: all passed\n", fails);
  return fails != 0;
}
