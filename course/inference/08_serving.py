"""
INFERENCE MODULE 8: THE SERVING STACK — HTTP API, Streaming, Metrics
======================================================================

All the optimizations we've built (KV cache, batching, quantization,
FlashAttention, speculative decoding) are useless if users can't send
requests and receive responses. The serving stack wraps our inference
engine in an HTTP API, streams tokens back to users, and collects
metrics to monitor performance.

This module builds a minimal but functional inference API server.
Not production-grade, but production-SHAPED — with the same structure
as vLLM, TGI, and Ollama.

WHAT YOU'LL LEARN:
   1. The OpenAI-compatible API format (the industry standard)
   2. How streaming works (Server-Sent Events / chunked transfer)
   3. Building a minimal HTTP server for our inference engine
   4. Key metrics: TTFT, TPOT, throughput, latency percentiles
   5. What production servers add beyond this

AFTER THIS MODULE:
   You'll have a working inference API you can `curl` — and you'll
   understand the serving architecture behind every LLM API provider.
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
from course._model import MiniGPT, softmax


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND: WHY THE OPENAI FORMAT WON
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: THE OPENAI API FORMAT — The Industry Standard")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 1.1  WHY EVERYONE USES THE OPENAI FORMAT                       │
└─────────────────────────────────────────────────────────────────┘

When OpenAI released their API in 2020, they defined a JSON schema
for chat completions. Because thousands of applications were built
against it, EVERY new inference engine implements the same format:

  vLLM → OpenAI-compatible
  TGI (HuggingFace) → OpenAI-compatible
  Ollama → OpenAI-compatible
  llama.cpp server → OpenAI-compatible
  Together API, Anyscale, Fireworks → OpenAI-compatible

This means: write code once against the OpenAI spec, and it works
with ANY inference engine. No client changes needed when switching
from OpenAI to an open-source model.

Core endpoints:
  POST /v1/chat/completions  — Chat-style (messages array)
  POST /v1/completions       — Raw completion (prompt string)
  GET  /v1/models            — List available models
  GET  /health               — Health check for load balancers
  GET  /metrics              — Prometheus-format metrics

┌─────────────────────────────────────────────────────────────────┐
│ 1.2  STREAMING — Why sending tokens one-by-one matters         │
└─────────────────────────────────────────────────────────────────┘

Non-streaming: user waits 5 seconds, gets full response.
Streaming:     user sees "The" → "The capital" → "The capital of" →
               "The capital of France" → ... in real time.

Perceived latency = Time-To-First-Token, not Time-To-Full-Response.

TTFT of 200ms → feels instant.
TTFT of 2000ms → feels broken, even if the full response is fast.

Streaming uses Server-Sent Events (SSE):
  Content-Type: text/event-stream
  Each token:   data: {"choices":[{"delta":{"content":" new"}}]}\\n\\n
  End signal:   data: [DONE]\\n\\n

The client opens a connection, keeps it alive, and processes each
token as it arrives. This is the STANDARD for chat applications.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE (simplified wrapper around MiniGPT)
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceEngine:
    """Simple inference engine for demonstration.

    In production, this uses KV cache, continuous batching, etc.
    Here we use the basic generate() for simplicity — the API layer
    is the focus of this module.
    """
    def __init__(self, model):
        self.model = model
        self.total_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0

    def generate(self, prompt_ids, max_tokens=50, temperature=0.8,
                 stream_callback=None):
        """Generate tokens with optional streaming."""
        self.total_requests += 1
        generated = list(prompt_ids)
        t0 = time.perf_counter()
        t_first_token = None

        for step in range(max_tokens):
            context = generated[-self.model.max_seq_len:]
            batch = np.array([context])
            logits = self.model.forward(batch)
            next_logits = logits[0, -1, :] / max(temperature, 1e-8)
            probs = softmax(next_logits)
            next_token = int(np.random.choice(self.model.vocab_size, p=probs))
            generated.append(next_token)

            if t_first_token is None:
                t_first_token = time.perf_counter()
            if stream_callback:
                stream_callback(next_token, step)

        elapsed = time.perf_counter() - t0
        new_tokens = len(generated) - len(prompt_ids)
        self.total_tokens += new_tokens
        self.total_time += elapsed

        return generated[len(prompt_ids):], {
            'ttft_ms': (t_first_token - t0) * 1000 if t_first_token else 0,
            'tpot_ms': (elapsed - (t_first_token - t0)) / max(new_tokens - 1, 1) * 1000
            if t_first_token and new_tokens > 1 else 0,
            'total_time_ms': elapsed * 1000,
            'tokens': new_tokens,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP API HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceAPIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the inference API — OpenAI-compatible endpoints.

    In production, this is built with FastAPI/Starlette for async I/O,
    proper error handling, and middleware. This http.server version
    is educational: single-threaded, blocking, but structurally correct.
    """

    engine: "InferenceEngine | None" = None

    def _eng(self):
        eng = self.engine
        assert eng is not None, "engine not set on InferenceAPIHandler"
        return eng

    def log_message(self, format, *args):
        pass  # suppress noisy logs during course execution

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/v1/chat/completions':
            self._handle_chat_completion()
        elif parsed.path == '/v1/completions':
            self._handle_completion()
        else:
            self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/health':
            self._handle_health()
        elif parsed.path == '/metrics':
            self._handle_metrics()
        else:
            self.send_error(404)

    def _read_body(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        return json.loads(body)

    def _send_json(self, data, status=200):
        response = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _handle_health(self):
        self._send_json({
            'status': 'healthy',
            'model': f'MiniGPT-vocab{self._eng().model.vocab_size}',
        })

    def _handle_metrics(self):
        engine = self._eng()
        avg_tps = (engine.total_tokens / engine.total_time
                   if engine.total_time > 0 else 0)
        self._send_json({
            'requests_total': engine.total_requests,
            'tokens_total': engine.total_tokens,
            'avg_tokens_per_second': round(avg_tps, 1),
            'uptime_seconds': round(engine.total_time, 1),
        })

    def _handle_chat_completion(self):
        try:
            body = self._read_body()
        except (json.JSONDecodeError, ValueError):
            self._send_json({'error': 'Invalid JSON'}, 400)
            return

        messages = body.get('messages', [])
        max_tokens = body.get('max_tokens', 50)
        temperature = body.get('temperature', 0.8)
        do_stream = body.get('stream', False)

        if not messages:
            self._send_json({'error': 'No messages'}, 400)
            return

        prompt_text = ""
        for msg in messages:
            prompt_text += f"[{msg.get('role', 'user')}] {msg.get('content', '')}\n"
        prompt_text += "[assistant]"

        prompt_ids = [ord(c) % self._eng().model.vocab_size for c in prompt_text]

        if do_stream:
            self._stream_response(prompt_ids, max_tokens, temperature)
        else:
            self._complete_response(prompt_ids, max_tokens, temperature)

    def _handle_completion(self):
        try:
            body = self._read_body()
        except (json.JSONDecodeError, ValueError):
            self._send_json({'error': 'Invalid JSON'}, 400)
            return

        prompt_ids = [ord(c) % self._eng().model.vocab_size
                      for c in body.get('prompt', '')]
        max_tokens = body.get('max_tokens', 50)
        temperature = body.get('temperature', 0.8)

        if body.get('stream', False):
            self._stream_response(prompt_ids, max_tokens, temperature, chat=False)
        else:
            self._complete_response(prompt_ids, max_tokens, temperature, chat=False)

    def _complete_response(self, prompt_ids, max_tokens, temp, chat=True):
        new_tokens, _stats = self._eng().generate(prompt_ids, max_tokens, temp)
        output_text = ''.join(chr(t % 128) for t in new_tokens
                              if 32 <= t % 128 < 127)

        if chat:
            response = {
                'id': f'chatcmpl-{int(time.time())}',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': 'minigpt',
                'choices': [{'index': 0, 'message': {
                    'role': 'assistant', 'content': output_text.strip()
                }, 'finish_reason': 'length'}],
                'usage': {'prompt_tokens': len(prompt_ids),
                          'completion_tokens': len(new_tokens),
                          'total_tokens': len(prompt_ids) + len(new_tokens)},
            }
        else:
            response = {
                'id': f'cmpl-{int(time.time())}',
                'object': 'text_completion',
                'created': int(time.time()),
                'model': 'minigpt',
                'choices': [{'index': 0, 'text': output_text, 'finish_reason': 'length'}],
                'usage': {'prompt_tokens': len(prompt_ids),
                          'completion_tokens': len(new_tokens),
                          'total_tokens': len(prompt_ids) + len(new_tokens)},
            }
        self._send_json(response)

    def _stream_response(self, prompt_ids, max_tokens, temp, chat=True):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        stream_id = f'chatcmpl-{int(time.time())}' if chat else f'cmpl-{int(time.time())}'
        completed = []

        def on_token(token_id, step):
            completed.append(token_id)
            token_text = chr(token_id % 128) if 32 <= token_id % 128 < 127 else ' '
            chunk = {
                'id': stream_id, 'object': 'chat.completion.chunk',
                'created': int(time.time()), 'model': 'minigpt',
                'choices': [{'index': 0, 'delta': {'content': token_text},
                             'finish_reason': None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
            self.wfile.flush()

        self._eng().generate(prompt_ids, max_tokens, temp, stream_callback=on_token)

        finish = {
            'id': stream_id, 'object': 'chat.completion.chunk',
            'created': int(time.time()), 'model': 'minigpt',
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}],
        }
        self.wfile.write(f"data: {json.dumps(finish)}\n\n".encode('utf-8'))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: RUNNING THE SERVER
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 2: RUNNING THE INFERENCE API SERVER")
print("=" * 70)

np.random.seed(42)
model = MiniGPT(vocab_size=100, d_model=32, num_heads=4,
                num_layers=2, max_seq_len=64)
engine = InferenceEngine(model)

print(f"\nStarting server on http://localhost:8765")
print(f"Model: MiniGPT (vocab={model.vocab_size}, d_model={model.d_model})")

InferenceAPIHandler.engine = engine
server = HTTPServer(('127.0.0.1', 8765), InferenceAPIHandler)
server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()

print("\nTesting endpoints:\n")

# Health check
status, result = None, None
try:
    with urlopen(Request("http://127.0.0.1:8765/health"), timeout=5) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        status = resp.status
except Exception as e:
    result = str(e)
print(f"1. GET /health → {status}: {result}")

# Non-streaming chat
print(f"2. POST /v1/chat/completions (non-streaming):")
try:
    body = json.dumps({
        'model': 'minigpt', 'messages': [{'role': 'user', 'content': 'Hi!'}],
        'max_tokens': 15, 'temperature': 0.8, 'stream': False
    }).encode('utf-8')
    req = Request("http://127.0.0.1:8765/v1/chat/completions", data=body)
    req.add_header('Content-Type', 'application/json')
    with urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read().decode('utf-8'))
    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
    print(f"   Response: \"{content[:60]}...\"")
    print(f"   Tokens: {result.get('usage', {})}")
except Exception as e:
    print(f"   Error: {e}")

# Streaming test
print(f"3. POST /v1/chat/completions (streaming):")
try:
    body = json.dumps({
        'model': 'minigpt', 'messages': [{'role': 'user', 'content': 'Tell me.'}],
        'max_tokens': 8, 'temperature': 0.8, 'stream': True
    }).encode('utf-8')
    req = Request("http://127.0.0.1:8765/v1/chat/completions", data=body)
    req.add_header('Content-Type', 'application/json')
    tokens = []
    with urlopen(req, timeout=10) as resp:
        buf = ""
        while True:
            chunk = resp.read(1)
            if not chunk: break
            buf += chunk.decode('utf-8')
            while '\n\n' in buf:
                event, buf = buf.split('\n\n', 1)
                if event.startswith('data: '):
                    d = event[6:]
                    if d == '[DONE]': break
                    try:
                        ev = json.loads(d)
                        t = ev['choices'][0].get('delta', {}).get('content', '')
                        if t: tokens.append(t)
                    except (json.JSONDecodeError, KeyError): pass
    print(f"   Streamed tokens: {tokens}")
except Exception as e:
    print(f"   Error: {e}")

# Metrics
print(f"4. GET /metrics:")
try:
    with urlopen(Request("http://127.0.0.1:8765/metrics"), timeout=5) as resp:
        result = json.loads(resp.read().decode('utf-8'))
    print(f"   Requests: {result.get('requests_total')}, Tokens: {result.get('tokens_total')}")
except Exception as e:
    print(f"   Error: {e}")

server.shutdown()
print("\nServer stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# PART 3: WHAT PRODUCTION SERVERS ADD
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: PRODUCTION SERVERS — Beyond This Toy")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────┐
│ 3.1  WHAT PRODUCTION SERVERS DO DIFFERENTLY                     │
└─────────────────────────────────────────────────────────────────┘

Our toy server: single-threaded, blocking, one request at a time.
Production servers are a completely different beast:

ASYNC I/O (FastAPI/Starlette):
  Non-blocking request handling. While one request waits for GPU
  output, the server accepts new connections. CRITICAL for streaming
  — multiple concurrent SSE connections.

REQUEST QUEUE:
  Incoming requests go to a priority queue. Scheduler picks N
  requests for the next batch based on priority, age, and predicted
  output length. Fair scheduling prevents starvation.

MIDDLEWARE STACK:
  - Authentication: API keys, OAuth tokens
  - Rate limiting: tokens per minute per key, tiered pricing
  - Request ID tracing: every request gets a UUID for logging
  - CORS headers for browser clients
  - Compression: gzip/brotli for large non-streaming responses
  - Timeouts: kill requests exceeding max_tokens or time

GRACEFUL DEGRADATION:
  - Circuit breaker: if GPU OOM, pause new requests
  - Backpressure: reject requests above capacity threshold
  - Graceful shutdown: drain active requests before restart
  - Health-aware load balancing: route around unhealthy instances

METRICS (Prometheus format):
  - request_duration_seconds  (histogram: p50, p90, p99 buckets)
  - time_to_first_token_seconds  (TTFT)
  - time_per_output_token_seconds  (TPOT)
  - gpu_memory_used_bytes
  - kv_cache_usage_percent
  - requests_total, tokens_total
  - http_errors_total (4xx, 5xx breakdown)

ALERTING THRESHOLDS:
  - P95 TTFT > 2s → investigate slowdown
  - GPU memory > 90% → risk of OOM, scale up
  - KV cache > 85% → increase max_seq_len or add GPUs
  - Error rate > 1% → check model bugs or capacity issues
""")


print("=" * 70)
print("SUMMARY: What you need from this module")
print("=" * 70)
print("""
┌─────────────────────────────────────────────────────────────────┐
│ THE SERVING STACK — Bridge between engine and users            │
│                                                                 │
│ 1. OpenAI API format is the industry standard. Every engine    │
│    implements it for maximum compatibility.                    │
│                                                                 │
│ 2. Streaming via SSE sends tokens as generated → low perceived │
│    latency (TTFT matters more than total time).                 │
│                                                                 │
│ 3. A production server adds: async I/O, request queue, auth,   │
│    rate limiting, middleware, graceful degradation, monitoring. │
│                                                                 │
│ 4. Key metrics: TTFT (first token), TPOT (per token speed),    │
│    throughput (total output), P95/P99 latency.                  │
│                                                                 │
│ 5. vLLM, TGI, Ollama all follow this pattern — our toy server  │
│    is architecturally identical, just simpler.                  │
└─────────────────────────────────────────────────────────────────┘

Next: Module 9 — Production Engines: vLLM, TGI, SGLang & The State of the Art
""")

if __name__ == "__main__":
    print("\nModule 8 complete! Next: i09_production.py")
    print("Run with: uv run python course/inference/i09_production.py")
