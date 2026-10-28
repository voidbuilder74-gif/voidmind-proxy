"""VoidMind Proxy Server (v0.1).

A small HTTP service that:
  - Holds VoidMind's Groq API keys (never shipped in the APK)
  - Rotates keys on 429
  - Rate-limits per IP (default: 30/day)
  - Returns clean JSON responses

Local dev:      python3 proxy.py
Production:     deploy to Cloudflare Workers / Vercel (same logic)

Run:
  GROQ_KEY_1=gsk_... GROQ_KEY_2=gsk_... python3 proxy.py
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# --- Configuration ------------------------------------------------------
# Local dev (Termux):  no PORT env -> binds to 127.0.0.1:8001
# Render / cloud:      PORT env set -> binds to 0.0.0.0:$PORT (required)
_port_env_set = "PORT" in os.environ
HOST = os.environ.get("HOST", "0.0.0.0" if _port_env_set else "127.0.0.1")
PORT = int(os.environ.get("PORT", "8001"))

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_USER_AGENT = "VoidMind-Proxy/1.0"

_DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

_MAX_PROMPT_LEN = 10_000
_MAX_SYSTEM_LEN = 5_000
_MAX_TOKENS_CAP = 4_000
_DEFAULT_MAX_TOKENS = 1_000

# Rate limit: 30 requests per IP per 24h
_RATE_LIMIT = 9999  # bypass for testing
_RATE_WINDOW = 86400


# --- Exceptions ---------------------------------------------------------
class RateLimitError(Exception):
    """Raised when a Groq key hits its rate limit."""


class AllKeysExhausted(Exception):
    """Raised when every key in the pool returns 429."""


# --- Key loading --------------------------------------------------------
_ENV_KEY_RE = re.compile(r"^GROQ_KEY_(\d+)$")


def _load_keys_from_env(env):
    """Return list of keys from GROQ_KEY_1, GROQ_KEY_2, ... sorted by index."""
    numbered = []
    for k, v in env.items():
        m = _ENV_KEY_RE.match(k)
        if m and isinstance(v, str) and v.strip():
            numbered.append((int(m.group(1)), v.strip()))
    numbered.sort(key=lambda t: t[0])
    return [v for _, v in numbered]


def _load_keys():
    """Load keys from current environment."""
    return _load_keys_from_env(dict(os.environ))


# --- Request validation -------------------------------------------------
def _validate_llm_request(data):
    """Validate/normalize a /api/llm body. Returns dict or raises ValueError."""
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if len(prompt) > _MAX_PROMPT_LEN:
        raise ValueError(f"prompt too long (max {_MAX_PROMPT_LEN} chars)")

    system = data.get("system")
    if system is not None:
        if not isinstance(system, str):
            raise ValueError("system must be a string")
        if len(system) > _MAX_SYSTEM_LEN:
            raise ValueError(f"system too long (max {_MAX_SYSTEM_LEN} chars)")

    model = data.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("model must be a string")

    max_tokens = data.get("max_tokens", _DEFAULT_MAX_TOKENS)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError("max_tokens must be an integer")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    max_tokens = min(max_tokens, _MAX_TOKENS_CAP)

    return {
        "prompt": prompt,
        "system": system,
        "model": model,
        "max_tokens": max_tokens,
    }


# --- Groq caller --------------------------------------------------------
def _default_groq_caller(key, prompt, system, model, max_tokens, use_search=False):
    """Call Groq. Returns (text, model_used, meta) or raises RateLimitError."""
    if use_search:
        # Strictly use the search model. Do not fall back to dumb models!
        models = ["groq/compound-mini"]
    else:
        models = [model] if model else list(_DEFAULT_MODELS)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_exc = None
    for m in models:
        payload = {
            "model": m,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        if use_search:
            payload["tools"] = [{"type": "browser_search"}]

        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            _GROQ_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {}) or {}
            return (text, m, {"total_tokens": usage.get("total_tokens", 0)})
        except urllib.error.HTTPError as e:
            if e.code == 404:
                last_exc = e
                continue  # try next model
            if e.code == 429:
                raise RateLimitError(f"{m}: rate limited")
            
            # Read the error body from Groq
            error_body = e.read().decode('utf-8')
            raise RuntimeError(f"Groq Error {e.code}: {error_body}")
    if last_exc:
        raise RateLimitError("all models exhausted")
    raise RuntimeError("unreachable")


# --- Key rotation -------------------------------------------------------
def _call_groq_with_rotation(prompt, system, model, max_tokens, keys, groq_caller, use_search=False):
    """Try each key in order. Return (text, model, meta).

    Raises:
      AllKeysExhausted  - every key returned 429
      RuntimeError/etc  - non-rate-limit errors propagate
    """
    if not keys:
        raise AllKeysExhausted("no keys configured")

    for key in keys:
        try:
            return groq_caller(key, prompt, system, model, max_tokens, use_search)
        except RateLimitError:
            continue
    raise AllKeysExhausted("all keys rate-limited")


# --- Rate limiting ------------------------------------------------------
def _check_rate_limit(ip, store, now_func, limit, window):
    """Return True if allowed, False if rate-limited.

    store: dict {ip: [timestamp, timestamp, ...]}
    """
    now = now_func()
    cutoff = now - window
    history = store.get(ip, [])
    # Drop old entries
    history = [t for t in history if t > cutoff]
    if len(history) >= limit:
        store[ip] = history
        return False
    history.append(now)
    store[ip] = history
    return True


# --- Handlers -----------------------------------------------------------
def _handle_llm(data, ip, keys, rate_store, now_func,
                rate_limit, rate_window, groq_caller):
    """Handle /api/llm. Returns (status_code, response_dict)."""
    # 1. Rate limit check
    if not _check_rate_limit(ip, rate_store, now_func,
                              rate_limit, rate_window):
        return 429, {
            "ok": False,
            "error": "shared_limit_reached",
            "message": (
                "Free AI limit reached. "
                "Get your own free Groq key in Settings."
            ),
        }

    # 2. Validate request
    try:
        req = _validate_llm_request(data)
    except ValueError as e:
        return 400, {"ok": False, "error": "invalid_request",
                     "message": str(e)}

    # 3. Call Groq with rotation
    try:
        text, model_used, meta = _call_groq_with_rotation(
            prompt=req["prompt"],
            system=req["system"],
            model=req["model"],
            max_tokens=req["max_tokens"],
            keys=keys,
            groq_caller=groq_caller,
            use_search=data.get("use_search", False),
        )
    except AllKeysExhausted as e:
        return 429, {
            "ok": False,
            "error": "shared_limit_reached",
            "message": f"Groq Limit Exhausted: {str(e)}. Wait 1 hour or use BYOK.",
        }
    except Exception as e:
        return 500, {"ok": False, "error": "internal_error",
                     "message": str(e)}

    # 4. Success
    return 200, {
        "ok": True,
        "text": text,
        "model": model_used,
        "tokens": meta.get("total_tokens", 0),
    }


def _handle_health(keys):
    """Handle /health."""
    return 200, {
        "ok": True,
        "service": "VoidMind Proxy",
        "keys_available": len(keys),
    }


# --- HTTP server --------------------------------------------------------
_RATE_STORE = {}


class _ProxyHandler(BaseHTTPRequestHandler):

    def log_message(self, *a, **k):
        pass  # silent

    def _send(self, status, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                return None
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _client_ip(self):
        # Read the real user IP from proxy headers (Render uses X-Forwarded-For)
        xff = self.headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
        return self.client_address[0] if self.client_address else "unknown"

    def do_GET(self):
        if self.path == "/health":
            status, body = _handle_health(keys=_load_keys())
            self._send(status, body)
            return
        self._send(404, {"ok": False, "error": "not_found"})

    def do_POST(self):
        if self.path == "/api/llm":
            data = self._read_json()
            if data is None:
                self._send(400, {"ok": False,
                                 "error": "invalid_request",
                                 "message": "Send a valid JSON object"})
                return
            status, body = _handle_llm(
                data=data,
                ip=self._client_ip(),
                keys=_load_keys(),
                rate_store=_RATE_STORE,
                now_func=time.time,
                rate_limit=_RATE_LIMIT,
                rate_window=_RATE_WINDOW,
                groq_caller=_default_groq_caller,
            )
            self._send(status, body)
            return
        self._send(404, {"ok": False, "error": "not_found"})


def main():
    keys = _load_keys()
    if not keys:
        print("WARNING: no GROQ_KEY_N in environment. Set at least one.")
        print("Example: GROQ_KEY_1=gsk_... python3 proxy.py")
    print(f"VoidMind Proxy starting on http://{HOST}:{PORT}")
    print(f"Keys loaded: {len(keys)}")
    print("Press Ctrl+C to stop.")
    server = ThreadingHTTPServer((HOST, PORT), _ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nProxy stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
