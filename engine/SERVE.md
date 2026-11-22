# ie-serve: an OpenAI-compatible HTTP server

`ie-serve` turns the engine into an HTTP server. Chat front ends (for example Open WebUI) can connect to it directly.

## Terms

- **OpenAI-compatible API**: the paths and JSON format are the same as OpenAI's `/v1/chat/completions`.
- **SSE** (server-sent events): when the request has `"stream": true`, the server sends each piece of text as soon as it is generated.
- **ChatML**: Qwen's chat template. Each message is `<|im_start|>role\ncontent<|im_end|>\n`.
- **Engine thread**: the only thread that uses the model and the GPU. It runs one request at a time, in order.

## 1. Build

```
make build/ie-serve build/ie_kernels.hsaco
```

## 2. Start

```
./build/ie-serve ~/models/q4/Qwen3.8-27B-Q4_K_M.gguf --host 127.0.0.1 --port 8000 --ctx 65536 --draft 7 \
    --dflash ~/models/dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

Drafter file: `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` from `incoai/Qwen3.8-27B-DFlash2-GGUF` (1.1 GB).
Note: with `--dflash`, the context is at most 65536 in 32 GB of VRAM. For a 131072 context, remove `--dflash`.

Lab scripts on Windows: `pwsh -File serve-saluki.ps1` (Saluki 27B, port 8090: `--draft 5 --draft-pmin 0.4`, q8 KV,
and a head file whose own token embedding the head reads, `IE_MTP_EMBED=1`) and `pwsh -File serve-3d-coder.ps1`
(Qwen3.5-4B-3D-Coder, port 8091: the Qwen3.5-4B base MTP head, `--draft 7 --draft-pmin 0.35`, `IE_DRAFT_VOCAB=32768`,
f16 KV, model-card sampling, thinking on; `-NoThink` for no reasoning block). The lab's
`start-3d-coder-chat.ps1` starts it with Open WebUI (`webui-3d-coder.ps1`, port 3001).

The server binds the port first, then loads the model. When loading finishes, the log has the line `ie-serve: http://127.0.0.1:8000/v1`.

Note: if `--host` is not 127.0.0.1, every device on the network that can reach this address can use the model. In that case set `--api-key`.

| Option | Default | Description |
|---|---|---|
| `--backend gpu\|cpu` | gpu | cpu is only for small-model tests |
| `--host ADDR`, `--port N` | 127.0.0.1, 8000 | Listen address and port |
| `--api-key KEY` | none | Requests must carry `Authorization: Bearer KEY`. The environment variable `IE_API_KEY` can also be used |
| `--name ID` | the model file's `general.name` | Model name returned by `/v1/models`. Whitespace is replaced with `-` (Open WebUI does not accept model names with spaces), for example `Qwen3.8-27B` |
| `--ctx N` | 8192 | Maximum number of tokens for prompt plus output. The KV cache is allocated at this size (see "Context length" below) |
| `--draft D` | 0 | Number of MTP drafts (1 to 7, GPU only). 7 is recommended (context drafting and DFlash also use this verify space) |
| `--dflash FILE` | none | Use together with `--draft`: also loads the DFlash 2 drafter; at each step, DFlash or MTP is chosen by measured speed (see README "DFlash 2 drafter") |
| `--draft-pmin P`, `--mtp FILE` | 0.6, none | Same as `ie-run` |
| `--chunk N` | 256 | Tokens per prefill chunk (1 to 512) |
| `--temp T`, `--top-k K`, `--top-p P` | by thinking mode | Used when the request does not give these parameters. Without them: a thinking request uses the model file's `general.sampling.*` (Qwen3.8 27B: 1.0, 20, 0.95), or 0.6, 20, 0.95 if the file has none; a request without thinking uses 0.7, 20, 0.8. `--think-temp T` then sets the thinking temperature alone (`serve-saluki.ps1` passes 0.6, as Saluki's model card recommends). The values in use are in the log line `ie-serve: default sampling` and in each `queued` line |
| `--dry-multiplier M`, `--dry-base B`, `--dry-allowed-length N` | 0 (off), 1.75, 24 | The loop penalty, sampling only (`temp > 0`): a candidate that would continue a span of at least N tokens already in this answer loses `M * B^(match - N)` from its logit. Only the output counts, not the prompt; `</think>` starts new spans. A request may set `dry_multiplier`, `dry_base`, `dry_allowed_length`. `serve-3d-coder.ps1` and `serve-saluki.ps1` pass 0.8 (`-DryMultiplier 0` off) |
| `--tool-temp T` | -1 (off) | A request that carries `tools` and sets no `temperature` uses T. `serve-saluki.ps1` passes 0, the Saluki card's direct tool-call recipe; sampling only, so 0 (greedy) has no loop penalty. A request's own temperature wins |
| `--no-think` | off | Appends an empty `<think></think>` to the end of the prompt; the model does not output its reasoning |
| `--effort E` | xhigh | Reasoning effort when thinking: xhigh, medium, low (the template's reasoning_effort) |
| `--reasoning-budget N` | 0 (unlimited) | For Qwen chat with thinking enabled, close the reasoning block after N generated tokens and continue the answer within the request's completion limit |
| `--queue N` | 8 | Maximum number of queued requests. Beyond that, HTTP 503 is returned |
| `--no-cache` | off | Turns off the prompt cache; every request is computed from position 0 |
| `--think-in-content` | off | Writes the reasoning into `content`, wrapped in `<think>…</think>` (old format). Default: the reasoning is in the `reasoning_content` field and `content` holds only the answer |
| `--show-template` | - | Prints the chat template in the model file, then exits |

## 3. API

| Method and path | Description |
|---|---|
| `GET /health` | Returns `{"status":"ok"}`. The key is not checked |
| `GET /v1/models` | Returns one model |
| `POST /v1/chat/completions` | Chat. Supports `messages`, `stream`, `stream_options.include_usage`, `max_tokens` (or `max_completion_tokens`), `temperature`, `top_p`, `top_k`, `seed`, `reasoning_effort`, `reasoning_budget` (request override, 0 for unlimited), `chat_template_kwargs.enable_thinking`, `tools`, `tool_choice` (only `"none"` is recognized) |

The role in `messages` can be system, user, assistant or tool (developer is treated as system). The content can only be text.

The server builds the prompt from the template in the model file (`--show-template`), without handling tools and images:

- When thinking is on, the prompt starts with a system message holding the reasoning effort instruction (there is none for medium).
- The prompt ends with `<|im_start|>assistant\n<think>\n`. So the server adds `<think>\n` at the start of the answer, and the client sees a complete `<think>…</think>` block.
- Earlier answers are written as `<think>\nreasoning\n</think>\n\nanswer`. The reasoning comes from the `reasoning_content` field, or from the text before `</think>` in the content (Open WebUI's format).

The template code is in `src/chat.c`. `build/chat_render request.json` prints the prompt the server builds for a request.

### Tool calls

Term: a **tool call** is when the model does not answer directly but asks the client to run a function and send the result back to the model.

1. When the request has `tools`, the server writes the tool definitions into the leading system message according to the template. Each tool is written as one line of JSON, in the same format as Python's `json.dumps(ensure_ascii=False)`.
2. The model calls a tool in its own format: `<tool_call>\n<function=name>\n<parameter=param>\nvalue\n</parameter>\n</function>\n</tool_call>`.
3. The server looks for `<tool_call>` only after the reasoning (`</think>`). Once it finds one, the text after it is no longer sent as content; at the end it is converted into OpenAI `tool_calls`, and `finish_reason` is `tool_calls`. When streaming, `tool_calls` is sent at once in the last chunk.
4. Parameter values are converted according to the tool's JSON Schema: `string` type stays a string; for `integer`, `number`, `boolean`, `object` and `array` types, a value that is valid JSON is converted to the matching JSON value.
5. The assistant `tool_calls` in the history (`arguments` is a JSON string) and the results with the `tool` role are written back into the prompt according to the template.
6. If a `<tool_call>` block is incomplete (for example `max_tokens` was reached), the server returns it as plain text.

To use tools in Open WebUI: Admin Panel → Settings → Models → edit Qwen3.8 27B → Advanced Params → set Function Calling to "Native".
Guess: this menu location applies to the current version; I have not checked it on your instance.

Example:

```
curl -sN http://127.0.0.1:8000/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"用三段话介绍上海的历史"}],"stream":true}'
```

The prompt in the example is `用三段话介绍上海的历史` ("Introduce the history of Shanghai in three paragraphs").

### Prompt cache

Term: the **prompt cache** means that when a new request's prompt starts the same as the previous request's, the server skips the identical part and computes only the new tokens.

1. The KV cache keeps the tokens of the last request at their positions. The linear attention (GDN) state cannot be rolled back to an earlier position, so the server keeps **checkpoints**: copies of the linear attention state, the convolution ring and the MTP h input after the first N tokens (about 160 MB of host memory each for Saluki 27B; the KV cache is not copied).
2. A new request continues from the latest checkpoint inside its first tokens that equal the cached ones, and computes only from there. The checkpoints past its first different token are dropped.
3. A request adds checkpoints:
   - at the start of the line where its prompt first differs from the cached tokens (after the last newline token, at most 512 tokens back), so that the next prompt that changes in the same place (an agent's system prompt with a recalled-memories block, a date or a session id near its end) continues from there, even when its first different token comes a few tokens earlier in that line;
   - every `IE_CACHE_EVERY` positions (default 4096; spaced wider if more would fill half the cache);
   - before the last token of the prompt (the same prompt again);
   - at the end of the answer (the next turn of the conversation).
4. At most `IE_CACHE_CKPT` checkpoints are kept (default 16, 2 to 64; 2 for models with sliding window layers). A full cache first drops periodic ones, then the least recently used; the places where a prompt changed go last.
5. In a multi-turn conversation, the previous turn's prompt and answer (including the reasoning) are not computed again; only the new user message is computed. A request with another start (an agent's subagent, Open WebUI's title request) writes over the positions past the first different token; the conversation's next request continues from the latest checkpoint before that token.
6. With the cache, the prompt is split into chunks differently than without it, so the numerical results are close but not bitwise identical. With greedy decoding, the answers in the checks were identical (see section 6).
7. With MTP, a stop token inside the drafts is no longer accepted, so the model does not compute past the stop token. The output does not change.
8. The log line of each request: `cached N` with where it continued from (`with the last answer`, `with the last prompt`, `from where a prompt changed`, `from a checkpoint`, `from the disk cache`, `from where a prompt is likely to change`), the tokens in common with the cached ones and the checkpoints kept. With `IE_SERVE_DEBUG=1`, the log prints the tokens around the first different position. Every `IE_HIT_LOG_N` requests (default 10) and when idle, a `cache:` line gives the hit rate: prompt tokens, cached tokens and their share, how many came from the disk cache, and the totals since the start.
9. **Checkpoints where a prompt is likely to change** (`IE_CACHE_MARKS`, default `256,1024,4096`; `0`: none): agents put what changes per session (recalled memories, date, working directory) at the end of the system prompt. A request that computes a long system prompt adds checkpoints at the starts of the lines about 256, 1024 and 4096 tokens before its end, and one at the start of the last user message (a tool result counts). The next session continues within a few hundred tokens of its change, before any earlier session showed where that is. With the disk cache, the distances to the places where system prompts did differ are kept (`policy-splits.txt`, the last 16) and get checkpoints too. `IE_CACHE_MARK_GAP` (128): the least distance to another checkpoint.
10. **Disk cache** (`IE_DISK_CACHE=DIR`; off unless set, `serve-saluki.ps1` sets the lab's `cache\prompt`), as DeepSeek's context caching on disk. The memory cache holds one conversation's tokens: another conversation, or a restart, loses them. With the disk cache, checkpoints go to files in DIR, each with the K/V rows of the positions before it. The rows are in blocks of 1024 positions, each named by a hash of every token up to its end, so a conversation's checkpoints share them. A prompt continues from the furthest checkpoint on disk whose tokens it starts with, when that is further than the checkpoints in memory; when that one is still being written (a quick return), the engine finishes writing it first.
    - **Agent prefixes:** `omp/` and `opencode/` requests get persistent checkpoints at tool and system boundaries. The engine learns additional positions from exact token matches when tool schemas or system text change. It queues new prefix copies after generation. `IE_OMP_PREFIX=0` disables OMP checkpoints. `IE_OPENCODE_PREFIX=1` enables OpenCode checkpoints and is the adapted Saluki launcher's default. The adapted launcher keeps its own disk cache and defaults to 2 GB each for its RAM tier and pinned pool, with backpressure on disk writes starting at `IE_DISK_QUEUE_GB=1`.
    - **What is written:** the places where a prompt differed or is likely to, the latest prompt end and answer end, and the deepest periodic checkpoint below them; under 1024 tokens nothing (`IE_DISK_MIN`). `IE_DISK_ADMIT=all`: every checkpoint.
    - **When:** not after each request. When a request leaves the conversation in memory (its checkpoints would be dropped), when no request came for `IE_DISK_IDLE_S` seconds (default 10; 0: after each request), and before a full memory cache drops one. The engine copies them to host memory (pinned, up to `IE_DISK_PINNED_GB`, default 4); a disk thread writes the files while the next request runs. An agent's session writes nothing between its requests; a server stopped within 10 s of a request loses that request's checkpoints.
    - **Files** (format 2): a 64-byte header with the times the checkpoint was read, sections at 4096-byte offsets, read with unbuffered overlapped reads in 8 MB pieces, 4 blocks ahead of the copies to the GPU (`IE_DISK_DIRECT=0`: through the system cache; `IE_DISK_CHUNK_MB`, `IE_DISK_AHEAD`). A snapshot holds the linear attention state, the 3 conv ring slots before its position (the GPU's ring has 32; a resume reads only those) and the MTP h input: 157 MB on Saluki 27B.
    - **RAM tier** (`IE_DISK_RAM_GB`, default 8): the files written or read last stay in host memory too; a resume from there skips the disk.
    - **Limit** (`IE_DISK_CACHE_GB`, default 100): over it, files of another setup go first (oldest first), then checkpoints never read, then the rest (least recently used first); checkpoints placed at a split or a mark count as read. Blocks go when no checkpoint needs them.
    - The files are tied to the model, MTP head and hsaco files (size and time) and the cache format (`IE_KV_Q4`, `IE_KV_Q8`, `IE_KV_ROT`, `IE_MTP_EMBED`); files of another setup are not read. Not with the CPU backend's sliding window layers or DFlash (`tools/dcache_*_check.c` run it on the CPU with the tiny models).
    - `IE_DISK_STATE16=1` (off): snapshots stored as int16 with a scale per 128 values, files 42% smaller; the first-token logits after a resume moved by up to 0.12 (`bench/disk-cache-check-task.ps1`, phase 3), so it is not the default. Lossless compression saves about 14% and makes no load faster (`research/dcache-compress/README.md`).
    - Measured on Saluki 27B (`bench/disk-cache-check-task.ps1`, 26.7K-token omp-shaped prompt, 22.9 s to compute): back to it after another prompt 0.85 s (first-token logits identical to a resume from memory); after a restart 0.61 s (739 MB from the disk in 0.39 s); a new session with other memories 0.96 s (continued from 26,413).

### Context length

The KV cache is 16-bit floating point (fp16). Qwen3.8-27B uses 69,632 bytes per token (16 attention layers plus the MTP layer).
It used to be 32-bit floating point, at 139,264 bytes per token. Building the kernels with `-DIE_KV_F32` goes back to 32-bit; the host program chooses automatically depending on whether the hsaco has the `ie_kv_f16` kernel.
The attention kernel reads fp16 as f32 and then computes, so the only rounding happens once, when writing to the KV cache.

Measured on the R9700 (32 GB) (Q4_K_M, 3 drafts):

| `--ctx` | KV cache | VRAM use | Result |
|---|---|---|---|
| 65536 (32-bit, old) | 9.1 GB | 31.75 GB | OK |
| 131072 (16-bit) | 9.1 GB | 30.39 GB, 30.79 GB with a 100,000-token request | OK (amd-gpu-host uses this value now) |

The model was trained with a context of 262,144 (`qwen35.context_length`). 128K needs 4,096 position splits, so the split limit `IE_ATT_MAX_SPLIT` was raised from 2048 to 4096.
Note: if the number of splits exceeds the limit, split attention is turned off and decoding falls back to the old single-workgroup kernel. At 100,000 tokens that gives only 1.8 tokens/s.

Perplexity with the fp16 KV cache (wikitext, `ie-run --ppl --ppl-first 256`, 2026-10-09):

| Length | 32-bit | 16-bit |
|---|---|---|
| 4,096 | 5.7318 | 5.7357 |
| 32,768 | 5.3265 | 5.3272 |
| 61,440 | 5.7147 | 5.7148 |
| 126,976 | does not fit | 5.3751 |

The output of MTP speculative decoding is still identical to plain decoding.

Measured at 128K context (101,301 tokens of wikitext plus a question): the first turn prefills in 231 s (about 439 tokens/s; with `--chunk 512`, 193 s, about 526 tokens/s; after `ie_attn_fa2`, a 127K prefill takes 159 s) and outputs 23.3 tokens/s;
the second turn skips 101,300 tokens, prefills in 78 ms and outputs 21.8 tokens/s.

Measured at 64K context (56,248 tokens of code plus a question, no thinking; these are the numbers from before `ie_attn_fa`, after which a 63K prefill takes about 82 s): the first turn prefills in 212 s (about 265 tokens/s) and outputs 26.9 tokens/s;
the second turn skips 56,336 tokens, prefills in 0.3 s and outputs 22.4 tokens/s.

### Speed information

Besides OpenAI's token counts, `usage` carries speeds:

| Field | Meaning |
|---|---|
| `prompt_tokens_details.cached_tokens` | Number of tokens skipped thanks to the prompt cache |
| `response_token/s` | Output speed (after the first token) |
| `prompt_token/s` | Prefill speed (counts only the tokens actually computed) |
| `prompt_ms`, `response_ms` | Prefill and output times (ms) |
| `mtp_tokens_per_step` | With MTP, the average number of tokens obtained per step |

When streaming, the last chunk, the one with `finish_reason`, also carries `usage` (llama.cpp does this too), so the client sees it even if it does not request `include_usage`.
Open WebUI shows the whole `usage` in the tooltip of the ⓘ icon under the answer.
Each output token is its own chunk, and a sender thread spreads the tokens of one MTP step (several come out of one verify pass at once) evenly over the time a step takes, so text arrives one token at a time at the decoding rate; a token waits at most about one step, and what is left goes at once when the next step's tokens come. The first content chunk after the reasoning also carries the held-back `</think>` tokens. `IE_STREAM_PACE=0`: one chunk per step, sent as soon as the step ends. Check: `python tools/serve_stream_check.py --port PORT` against a running server.
Every streamed chunk also has a top-level `n_tokens`: the output tokens so far. Other clients ignore it; the lab's Open WebUI Speed filter (`bench/openwebui_speed_filter.py`) turns it into a live tokens-per-second line above the answer.

## 4. Connecting Open WebUI

Guess: the command below applies to the current Open WebUI version. I have not run it on amd-gpu-host.

```
docker run -d --network host --restart always --name open-webui \
  -v open-webui:/app/backend/data \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1 \
  -e OPENAI_API_KEY=none \
  ghcr.io/open-webui/open-webui:main
```

1. You start the container with `--network host`. This lets the container reach the host's 127.0.0.1:8000.
2. You open `http://<host's Tailscale name>:8080` in a browser and create the first account (the administrator).
3. You open the same address in your phone's browser, then "Add to Home Screen".

If ie-serve has `--api-key` set, change `OPENAI_API_KEY` to the same key.

The container on amd-gpu-host additionally has `-e ENABLE_OLLAMA_API=false -e ENABLE_TAGS_GENERATION=false -e ENABLE_FOLLOW_UP_GENERATION=false -e ENABLE_AUTOCOMPLETE_GENERATION=false`.
Reason: ie-serve runs only one request at a time, and these background tasks would queue up the chat. Title generation is still on.

Note: on first start Open WebUI downloads a text embedding model; if the download is slow, port 8080 opens only after 5 to 10 minutes.

## 4b. Connecting omp (the oh-my-pi coding assistant)

Confirmed (2026-10-09): Tailscale Serve on the host, at `https://amd-gpu-host.tail1aaa84.ts.net/`, now forwards to ie-serve (127.0.0.1:8000). Open WebUI has been stopped.

1. You edit `~/.omp/agent/models.yml` on your own computer and write in the content below.
2. You run `omp --model amd/Qwen3.8-27B`, or pick it in omp with `/model`.

```yaml
providers:
  amd:
    baseUrl: https://amd-gpu-host.tail1aaa84.ts.net/v1
    api: openai-completions
    auth: none
    compat:
      maxTokensField: max_tokens
      supportsDeveloperRole: false
      supportsReasoningEffort: true
      supportsStore: false
    models:
      - id: Qwen3.8-27B
        name: Qwen3.8 27B (amd-gpu-host)
        reasoning: true
        input: [text]
        contextWindow: 131072
        maxTokens: 16384
        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
```

- The reasoning is returned in `reasoning_content` and tool calls are returned in `tool_calls`.
- `reasoning_effort`: low or minimal makes the reasoning shorter; medium adds no instruction; other values (high, xhigh) make the reasoning longer.
- The coding assistant's system prompt and tool descriptions are long (often more than 10,000 tokens). The first request's prefill takes a few seconds; later requests hit the prompt cache.

## 5. Start at boot (systemd)

**Without sudo (amd-gpu-host uses this now): a user-level service.**

1. You write the `[Service]` and `[Install]` sections of `tools/ie-serve.service` to `~/.config/systemd/user/ie-serve.service`, delete the `User=` line, and change `WantedBy` to `default.target`.
2. You run `systemctl --user daemon-reload && systemctl --user enable --now ie-serve`.
3. You run `loginctl enable-linger $USER`. This keeps the service running when nobody is logged in, and starts it automatically at boot.
4. You read the log with `journalctl --user -u ie-serve -f`.

**With sudo: a system-level service.**

1. You change the user and the paths in `tools/ie-serve.service`.
2. You run `sudo cp tools/ie-serve.service /etc/systemd/system/`.
3. You run `sudo systemctl daemon-reload && sudo systemctl enable --now ie-serve`.
4. You read the log with `journalctl -u ie-serve -f`.

Each request has one line in the log: the number of prompt tokens, the prefill speed, the output speed and the MTP tokens per step.

## 6. Check results on the GPU (2026-10-08, amd-gpu-host)

1. `ie-run` output is unchanged: 917db85 and the new code give identical `ids:` in three settings (no drafts; 7 drafts; 7 drafts plus sampling).
2. The prompt matches the official template: 4 conversations (thinking on, thinking off, multi-turn plus low, medium) were rendered with the Jinja template,
   then tokenized with `llama-tokenize` from llama.cpp b11222. The result is identical, ID for ID, to ie-serve's prompt IDs.
3. Generation: Chinese chat (3 drafts) runs at 51 tokens/s, writing code at 78 tokens/s. Generation stops when the client disconnects.
4. Tool-call prompts: `tools/chat_check.py` renders 5 conversations with Jinja (the same settings as transformers) (with tools, with calls and results, thinking off, without tools),
   and the output is identical, character for character, to `build/chat_render`'s. After tokenizing with `llama-tokenize --no-escape`, it is also identical, ID for ID, to ie-serve's prompt IDs.
   Note: `llama-tokenize` by default converts `\n` in the text to a newline, so `--no-escape` is required for the comparison.
5. Tool-call generation: the model called two tools in one turn (Shanghai, Beijing, `days` is the integer 2); `tool_calls` was returned when streaming;
   after the tool results were sent back, the model gave the correct final answer.

6. Prompt cache: `build/cache_check` first runs prompt A, then "A plus the answer plus 7 tokens" (continues from the end of the answer), then the same prompt again (from the end of the prompt), then the same prompt without the cache; then three prompts with the same first 100 tokens, a line that starts at 91 and other tails (the third differs earlier in that line, at 95, and continues from the checkpoint the second left at 91) and the third without the cache. On the CPU the answers must be the same. `bench/saluki-cache-check-task.ps1` does the same with Saluki 27B through ie-serve.
   - CPU (computed token by token): the sampled answers are identical for Qwen3.5-0.8B (which has linear attention) and the small model. `make test` includes this check on the small model.
   - GPU (27B): with greedy decoding, the answers are identical with or without the cache and with or without MTP. With temperature-1 full-vocabulary sampling, the answers differ, because the prompt chunking differs.
   - Real conversation: a 5,385-token document plus a question; the second turn skips 5,469 tokens, and prefill drops from 6.7 s to 0.13 s.
   - `ie-run` output is unchanged (no drafts, 7 drafts, 7 drafts plus sampling).

To repeat check 4:

```
./build/ie-serve MODEL --show-template > /tmp/tpl.txt
make build/chat_render && python3 tools/chat_check.py /tmp/tpl.txt build/chat_render
```

When changing the template code later, you can set `IE_SERVE_DEBUG=1`; the server then prints each request's prompt IDs, which is used to repeat check 2.

## 7. Tokenizer

`src/tok.c` implements GPT2 byte-level BPE and Qwen's pre-tokenization regex (two variants, `qwen2` and `qwen35`).
The Unicode category table `src/tok_unicode.h` is generated by `tools/gen_unicode.py`.

Confirmed: `make tokcheck LLAMA_CPP=../llama.cpp` checks the tokenization against llama.cpp's test vocabularies.
All 46 qwen2 tests and 50 qwen35 tests give identical results.

## 8. Testing on a machine without a GPU

```
pip install gguf numpy
python3 tools/make_tiny_chat_gguf.py ../llama.cpp/models/ggml-vocab-qwen35.gguf build/tiny-chat.gguf
./build/ie-serve build/tiny-chat.gguf --backend cpu --port 18000 --ctx 4096
```

The small model's weights are random, so the output text is meaningless. It is only used to check HTTP, the template, tokenization and streaming.
Environment variables for testing: `IE_EXTRA_STOP=ID` adds a stop token; `IE_SERVE_DEBUG=1` prints the prompt IDs.

## 9. Limits

- Only one request runs at a time. Other requests queue.
- There is one KV cache: a request with another start writes over the positions past its first different token (the checkpoints before that token stay).
- Images, `stop` strings, `logprobs` and `n > 1` are not supported. `tool_choice` only recognizes `"none"`; forcing a call to a specific tool is not supported.
- Only GPT2 byte-level vocabularies and the ChatML template are supported (Qwen2, Qwen3.5, Qwen3.8).
- It listens on IPv4 addresses only.
