"""ie-serve streaming check on the CPU (a tiny model, no GPU): with IE_STREAM_PACE on (default) and off, the streamed
text (reasoning + content) must equal the non-streamed answer, the stream must end with usage and [DONE], n_tokens
must count the output tokens, and with pacing every token chunk must carry one token. A client that closes the
connection mid-stream must not stop the server.
usage: python tools/serve_stream_check.py [IE_SERVE_EXE] [MODEL.gguf] [PORT]
       python tools/serve_stream_check.py --port PORT   (a server that is running; its IE_STREAM_PACE, assumed on.
       Its prompt cache may resume the second request from a checkpoint, so a text difference is reported, not failed)
The CPU form needs a model whose tokenizer ie-serve reads (build/tiny-chat.gguf: SERVE.md, section on the tiny model)."""
import http.client
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = len(sys.argv) == 3 and sys.argv[1] == '--port'
EXE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 and not LIVE else os.path.join(HERE, 'build', 'ie-serve.exe'))
MODEL = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 and not LIVE else os.path.join(HERE, 'build', 'tiny-chat.gguf'))
PORT = int(sys.argv[2]) if LIVE else int(sys.argv[3]) if len(sys.argv) > 3 else 8097
REQ = {'messages': [{'role': 'user', 'content': 'Write a short poem about the sea.'}], 'max_tokens': 48,
       'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}


def post(body, stream):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=120)
    c.request('POST', '/v1/chat/completions', json.dumps({**body, 'stream': stream}), {'Content-Type': 'application/json'})
    r = c.getresponse()
    if not stream:
        return json.loads(r.read())
    events, t = [], []
    for line in r:
        line = line.decode().strip()
        if line.startswith('data: '):
            events.append(line[6:])
            t.append(time.perf_counter())
    return events, t


def run(pace):
    env = dict(os.environ, IE_STREAM_PACE='1' if pace else '0')
    log = open(os.path.join(HERE, 'build', f'serve_stream_check_{pace}.log'), 'w')
    p = None if LIVE else subprocess.Popen([EXE, MODEL, '--backend', 'cpu', '--port', str(PORT), '--ctx', '512'],
                                           stderr=log, stdout=log, env=env, cwd=HERE)
    try:
        for _ in range(120):
            try:
                c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=2)
                c.request('GET', '/health')
                if c.getresponse().status == 200:
                    break
            except OSError:
                time.sleep(0.25)
        else:
            raise SystemExit('server did not start')
        fails = []
        for name, body in (('plain', REQ), ('thinking', {**REQ, 'chat_template_kwargs': {'enable_thinking': True}})):
            ref = post(body, False)
            msg = ref['choices'][0]['message']
            events, _ = post(body, True)
            if events[-1] != '[DONE]':
                fails.append(f'{name}: last event {events[-1][:40]!r}, not [DONE]')
            chunks = [json.loads(e) for e in events[:-1]]
            text = {'content': '', 'reasoning_content': ''}
            counts, per_chunk = [], []
            for ch in chunks:
                d = (ch.get('choices') or [{}])[0].get('delta', {}) if ch.get('choices') else {}
                for k in text:
                    if d.get(k):
                        # the first content chunk after reasoning also counts the held-back </think> tokens
                        switch = k == 'content' and not text['content'] and text['reasoning_content']
                        text[k] += d[k]
                        if not switch:
                            per_chunk.append(ch['n_tokens'] - (counts[-1] if counts else 0))
                counts.append(ch['n_tokens'])
            usage = chunks[-1].get('usage') or {}
            if text['content'] != (msg.get('content') or ''):
                fails.append(f'{name}: streamed content differs')
            if text['reasoning_content'].strip() != (msg.get('reasoning_content') or '').strip():
                fails.append(f'{name}: streamed reasoning differs')
            if counts != sorted(counts) or counts[-1] != usage.get('completion_tokens'):
                fails.append(f'{name}: n_tokens {counts[-3:]} vs completion_tokens {usage.get("completion_tokens")}')
            if pace and any(k > 1 for k in per_chunk):
                fails.append(f'{name}: paced chunks with several tokens: {per_chunk}')
            print(f'  pace {pace} {name}: {len(chunks)} chunks, {usage.get("completion_tokens")} tokens, '
                  f'tokens per text chunk max {max(per_chunk) if per_chunk else 0}')
        # a client that goes away mid-stream: the server must still answer the next request
        s = socket.create_connection(('127.0.0.1', PORT))
        b = json.dumps({**REQ, 'max_tokens': 200, 'stream': True}).encode()
        s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: '
                  + str(len(b)).encode() + b'\r\n\r\n' + b)
        s.recv(300)
        s.close()
        ref2 = post(REQ, False)
        if not ref2['choices'][0]['message'].get('content'):
            fails.append('after a client left mid-stream: no answer')
        return fails
    finally:
        if p:
            p.terminate()
            p.wait(10)
        log.close()


fails = []
for pace in (1,) if LIVE else (1, 0):
    fails += [f'pace {pace}: {f}' for f in run(pace)]
print('\n'.join(fails) if fails else 'all passed')
sys.exit(1 if fails else 0)
