"""ClipStudio LLM brain — pluggable PRIMARY provider with automatic FALLBACK.

All of ClipStudio's reasoning (script analysis, per-beat visual enrichment, the AI vision verifier)
runs on the selected provider. If it's unavailable (no key) OR a call errors, it falls back to the
next provider automatically, so the pipeline never hard-fails on the LLM.

Switch with VIDLORE_CLIPSTUDIO_LLM_PROVIDER = deepseek | anthropic | gemini  (DEFAULT: deepseek).
  deepseek  → deepseek-v4-pro PRIMARY → deepseek-v4-flash → Gemini → Claude (LAST resort).
  anthropic → Claude primary (explicit choice) → DeepSeek → Gemini.
  gemini    → Gemini/Vertex primary → DeepSeek → Claude (last).
DeepSeek does ALL the reasoning by default; Claude is only the final safety net. NOTE: DeepSeek is
TEXT-ONLY — VISION calls (the image verifier) automatically skip BOTH DeepSeek models and use the
first vision-capable provider (Gemini, else Claude), since DeepSeek physically cannot see images.

Supports BOTH text and VISION: messages use the Anthropic shape
  [{"role":"user","content":[{"type":"text","text":...},
                             {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":<b64>}}]}]
and are translated to Gemini Parts (text + inline image bytes) under the hood — so the same call
works on either backend with no change at the call-site. Returns the plain response text.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

# isolated deps (google-genai installed here to keep the engine venv pristine)
_LIBS = Path(__file__).resolve().parents[2] / ".clipstudio_libs"
if _LIBS.exists() and str(_LIBS) not in sys.path:
    sys.path.append(str(_LIBS))

# every knob reads at CALL time, not import time — the engine loads .env after this module
# may already have been imported


def _provider() -> str:
    # DeepSeek is the default brain (Claude is now only the last-resort fallback).
    return os.environ.get("VIDLORE_CLIPSTUDIO_LLM_PROVIDER", "deepseek").strip().lower()


def _gemini_model() -> str:
    return os.environ.get("VIDLORE_CLIPSTUDIO_GEMINI_MODEL", "").strip() or "gemini-2.5-flash"


def _gemini_location() -> str:
    return os.environ.get("VIDLORE_CLIPSTUDIO_GEMINI_LOCATION", "").strip() or "us-central1"


def _claude_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "").strip() or "claude-sonnet-4-6"


def _claude_key(eng_cfg=None) -> str:
    return ((getattr(eng_cfg, "anthropic_api_key", "") if eng_cfg else "")
            or os.environ.get("ANTHROPIC_API_KEY", ""))


# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible chat API at api.deepseek.com). TEXT-ONLY: the standard DeepSeek
# chat model has no vision, so image (verifier) calls return '' here and fall through to a
# vision-capable provider (Claude/Gemini). Used via stdlib urllib — no extra SDK dependency.
# ---------------------------------------------------------------------------
def _deepseek_key(eng_cfg=None) -> str:
    return ((getattr(eng_cfg, "deepseek_api_key", "") if eng_cfg else "")
            or os.environ.get("DEEPSEEK_API_KEY", "")).strip()


def _deepseek_model() -> str:
    # PRIMARY model. deepseek-v4-pro = the reasoning model (best quality, the tool's default brain).
    return os.environ.get("VIDLORE_CLIPSTUDIO_DEEPSEEK_MODEL", "").strip() or "deepseek-v4-pro"


def beat_model() -> str:
    """Model id for the HIGH-VOLUME per-beat enrichment calls. Defaults to the MAIN DeepSeek model
    (so deepseek-v4-pro gives full pro quality on per-beat scene queries too). The per-beat pass is
    then slower (pro reasons per chunk) — set VIDLORE_CLIPSTUDIO_DEEPSEEK_BEAT_MODEL=deepseek-v4-flash
    to trade a little per-beat quality for much faster analysis. (This endpoint serves ONLY
    deepseek-v4-pro / deepseek-v4-flash — not 'deepseek-chat'.) '' for non-DeepSeek primaries."""
    if _provider() in ("deepseek", "ds"):
        return os.environ.get("VIDLORE_CLIPSTUDIO_DEEPSEEK_BEAT_MODEL", "").strip() or _deepseek_model()
    return ""


def fast_deepseek_model() -> str:
    """The fast 'flash' DeepSeek model id — the automatic fallback when deepseek-v4-pro is
    unavailable or returns unparseable/empty data, so the brain stays on DeepSeek (and analysis
    never silently degrades to the heuristic) before Claude is ever considered. NOTE: this key's
    endpoint only serves deepseek-v4-pro / deepseek-v4-flash (no 'deepseek-chat')."""
    return os.environ.get("VIDLORE_CLIPSTUDIO_DEEPSEEK_FAST_MODEL", "").strip() or "deepseek-v4-flash"


def _deepseek_base() -> str:
    return (os.environ.get("DEEPSEEK_BASE_URL", "").strip()
            or "https://api.deepseek.com").rstrip("/")


def deepseek_available(eng_cfg=None) -> bool:
    return bool(_deepseek_key(eng_cfg))


def _msgs_have_image(messages) -> bool:
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "image" for b in c):
            return True
    return False


def _to_openai_messages(system, messages):
    """Anthropic-shape messages → OpenAI chat messages (text only). Image blocks are dropped to
    text placeholders (callers with images should not reach DeepSeek — see _msgs_have_image)."""
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages or []:
        role = m.get("role", "user")
        c = m.get("content")
        if isinstance(c, str):
            text = c
        elif isinstance(c, list):
            text = "\n".join(b.get("text", "") for b in c
                             if isinstance(b, dict) and b.get("type") == "text")
        else:
            text = str(c or "")
        out.append({"role": role, "content": text})
    return out


def _deepseek_complete(system, messages, max_tokens, model) -> str:
    """OpenAI-compatible chat completion against DeepSeek. '' on any failure or on a vision call
    (so the fallback provider serves it)."""
    import urllib.request
    key = _deepseek_key()
    if not key or _msgs_have_image(messages):
        return ""                                      # keyless, or a vision call DeepSeek can't do
    mdl = (model or "").strip()
    if not mdl.lower().startswith("deepseek"):         # a Claude/Gemini id forwarded by a caller
        mdl = _deepseek_model()
    # REASONING models (deepseek-v4-pro / deepseek-reasoner) spend tokens THINKING before the final
    # answer, and max_tokens caps the TOTAL completion (reasoning + content). A small caller budget
    # (e.g. ~600 for a tiny per-beat call) would be eaten entirely by reasoning, leaving content
    # EMPTY → a wrong fallback. Give such models a reasoning headroom so the final answer survives.
    mt = int(max_tokens)
    if any(k in mdl.lower() for k in ("pro", "reason", "think")):
        try:
            mt += int(os.environ.get("VIDLORE_CLIPSTUDIO_DEEPSEEK_REASON_HEADROOM", "6000") or 6000)
        except (TypeError, ValueError):
            mt += 6000
    body = json.dumps({
        "model": mdl,
        "messages": _to_openai_messages(system, messages),
        "max_tokens": mt,
        "stream": False,
        "temperature": float(os.environ.get("VIDLORE_CLIPSTUDIO_DEEPSEEK_TEMP", "0.3") or 0.3),
    }).encode("utf-8")
    req = urllib.request.Request(
        _deepseek_base() + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    # reasoning models (pro) think for thousands of tokens → a single call can run 60-180s; give a
    # generous timeout so a slow-but-valid response isn't cut off (env-tunable).
    try:
        _to = int(os.environ.get("VIDLORE_CLIPSTUDIO_DEEPSEEK_TIMEOUT", "300") or 300)
    except (TypeError, ValueError):
        _to = 300
    resp = urllib.request.urlopen(req, timeout=_to)
    d = json.loads(resp.read())
    return (d.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


_CLIENT = None  # cached Gemini client


def _gcp_cred() -> str:
    """Resolve the GCP service-account JSON (Vertex AI auth). '' if none found."""
    cands = [
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip(),
        str(_LIBS.parent / ".secrets" / "gcp_vidlore.json"),                 # clone-local copy
        os.path.expanduser("~/Desktop/vidrush-clone/.secrets/gcp_vidlore.json"),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return ""


def _gemini_api_key() -> str:
    """Google AI Studio (Gemini Developer API) key — the simple, CHEAP vision path. When set it is
    preferred over Vertex, so the image verifier runs on gemini-2.5-flash via a plain API key with no
    GCP service account needed (~10x cheaper than the Claude vision fallback)."""
    return (os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip())


def gemini_available() -> bool:
    """Is Gemini USABLE here (creds + SDK)? — independent of which provider is PRIMARY, so Gemini can
    serve as a fallback. An AI-Studio API key OR a Vertex service account counts. Order lives in
    complete()."""
    if not (_gemini_api_key() or _gcp_cred()):
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except Exception:
        return False


def _gemini_http_options():
    """A hard per-request timeout: the SDK's default is NO deadline, so one hung TLS
    connection froze a 157-call verify pass for 2 hours. Timed-out calls raise, the
    retry wrapper already classes 'timeout' as retryable, and the brain fallback chain
    (gemini → claude) covers persistent failure. Env VIDLORE_GEMINI_TIMEOUT_SEC (default
    120 — vision verdicts normally return in <15s, essays' long analyze calls in <90s)."""
    from google.genai import types
    try:
        sec = float(os.environ.get("VIDLORE_GEMINI_TIMEOUT_SEC", "").strip() or 120.0)
        ms = int(sec * 1000)                                 # SDK expects milliseconds
        if ms <= 0:                                          # 0/negative = no deadline → footgun
            ms = 120000
    except (TypeError, ValueError, OverflowError):           # 'nan' fails at int(), 'inf' overflows
        ms = 120000
    return types.HttpOptions(timeout=ms)


def _gemini_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        key = _gemini_api_key()
        if key:                                 # AI Studio (Developer API) — the cheap, simple path
            _CLIENT = genai.Client(api_key=key, http_options=_gemini_http_options())
            return _CLIENT
        cred = _gcp_cred()                       # else fall back to Vertex (service-account JSON)
        # cred is verified to exist on disk; a stale env var pointing at a missing file must
        # not win over it (setdefault would keep the broken value)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred
        try:
            proj = json.load(open(cred)).get("project_id", "")
        except Exception:
            proj = os.environ.get("VIDLORE_GCP_PROJECT", "")
        if not proj:
            raise RuntimeError("no GCP project_id resolved for Gemini/Vertex")
        _CLIENT = genai.Client(vertexai=True, project=proj, location=_gemini_location(),
                               http_options=_gemini_http_options())
    return _CLIENT


def _gemini_parts(content):
    """Anthropic-style content (str | list of text/image blocks) → Gemini Parts (text + image)."""
    from google.genai import types
    if isinstance(content, str):
        return [types.Part(text=content)]
    parts = []
    for b in (content if isinstance(content, list) else [content]):
        if not isinstance(b, dict):
            parts.append(types.Part(text=str(b)))
        elif b.get("type") == "text":
            parts.append(types.Part(text=b.get("text", "")))
        elif b.get("type") == "image":
            src = b.get("source", {}) or {}
            try:
                data = base64.b64decode(src.get("data", ""))
                parts.append(types.Part.from_bytes(
                    data=data, mime_type=src.get("media_type", "image/jpeg")))
            except Exception:
                pass
    return parts or [types.Part(text="")]


def _gemini_complete(system, messages, max_tokens, model) -> str:
    from google.genai import types
    cl = _gemini_client()
    contents = []
    for m in messages:
        role = "model" if m.get("role") == "assistant" else "user"
        contents.append(types.Content(role=role, parts=_gemini_parts(m.get("content"))))
    gcfg = dict(max_output_tokens=int(max_tokens))
    if system:
        gcfg["system_instruction"] = system
    think = os.environ.get("VIDLORE_GEMINI_THINKING", "0").strip()
    try:                                              # thinking OFF by default (else empty output)
        gcfg["thinking_config"] = types.ThinkingConfig(thinking_budget=int(think))
    except Exception:
        pass
    # callers forward eng_cfg.anthropic_model verbatim — a Claude id must not kill the fallback
    mdl = (model or "").strip()
    if not mdl.lower().startswith("gemini"):
        mdl = _gemini_model()
    resp = cl.models.generate_content(
        model=mdl, contents=contents,
        config=types.GenerateContentConfig(**gcfg))
    return getattr(resp, "text", None) or ""


def _claude_complete(system, messages, max_tokens, eng_cfg, model) -> str:
    import anthropic
    key = _claude_key(eng_cfg)
    if not key:
        return ""
    cl = anthropic.Anthropic(api_key=key)
    mdl = model or (getattr(eng_cfg, "anthropic_model", "") if eng_cfg else "") or _claude_model()
    if not mdl.lower().startswith("claude"):     # a Gemini id forwarded by a caller
        mdl = ((getattr(eng_cfg, "anthropic_model", "") if eng_cfg else "") or _claude_model())
        if not mdl.lower().startswith("claude"):   # eng_cfg itself misconfigured with a non-Claude id
            mdl = "claude-sonnet-4-6"
    resp = cl.messages.create(model=mdl, max_tokens=max_tokens,
                              system=(system or ""), messages=messages)
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _retries() -> int:
    try:
        return max(1, int(os.environ.get("VIDLORE_CLIPSTUDIO_LLM_RETRIES", "4")))
    except (TypeError, ValueError):
        return 4


def _is_transient(e: Exception) -> bool:
    """Rate-limit / overload / timeout / 5xx — worth retrying. Auth/quota-exhausted are not, but we
    still let the provider FALLBACK handle those, so over-retrying briefly is harmless."""
    s = (str(e) or "").lower() + " " + type(e).__name__.lower()
    return any(k in s for k in ("429", "rate", "overload", "529", "timeout", "timed out",
                                "connection", "503", "502", "500", "unavailable", "resourceexhausted"))


def complete(*, system: str = "", messages, max_tokens: int = 1024,
             eng_cfg=None, model: str = "") -> str:
    """Back-compat wrapper over complete_ex(): text only ('' on total failure)."""
    return complete_ex(system=system, messages=messages, max_tokens=max_tokens,
                       eng_cfg=eng_cfg, model=model)[0]


def complete_ex(*, system: str = "", messages, max_tokens: int = 1024,
                eng_cfg=None, model: str = "") -> tuple:
    """Run an LLM completion and report WHO ACTUALLY SERVED it.

    Returns (text, meta) where meta = {"provider", "model", "transport", "served"} describes
    the provider branch that produced the text — not the configured/predicted one. "served" is
    the canonical identity string in exactly vision_config()'s format, so cache keys derived
    from a prediction and keys derived from the actual server are directly comparable. On
    total failure -> ("", {"served": "none", ...}). The provider ladder, retries, backoff and
    fallback order are byte-identical to the historical complete()."""
    import time as _time

    def _with_retry(call):
        n = _retries()
        for i in range(n):
            try:
                out = call()
                if out and out.strip():
                    return out
                # empty (non-exception) — brief retry, could be a transient truncation
                if i < n - 1:
                    _time.sleep(min(8.0, 0.8 * (2 ** i)))
            except Exception as e:                          # noqa: BLE001
                if i < n - 1 and _is_transient(e):
                    _time.sleep(min(12.0, 1.0 * (2 ** i)) + 0.3 * i)
                    continue
                return None
        return None

    def _try_claude():
        if not _claude_key(eng_cfg):     # keyless: skip instantly — retrying '' burns ~5.6s/call
            return None
        return _with_retry(lambda: _claude_complete(system, messages, max_tokens, eng_cfg, model))

    def _try_gemini():
        if not gemini_available():
            return None
        return _with_retry(lambda: _gemini_complete(system, messages, max_tokens, model))

    def _try_deepseek(model_id: str = ""):
        # keyless or a vision call (DeepSeek is text-only) → skip so a vision-capable provider serves
        if not _deepseek_key(eng_cfg) or _msgs_have_image(messages):
            return None
        # explicit model_id (e.g. the flash fallback) wins; else the caller's model (pro by default)
        return _with_retry(lambda: _deepseek_complete(system, messages, max_tokens, model_id or model))

    def _gemini_id():
        return {"provider": "gemini", "model": _gemini_model(),
                "transport": "apikey" if _gemini_api_key() else "vertex",
                "served": f"gemini:{_gemini_model()}:{'apikey' if _gemini_api_key() else 'vertex'}"}

    def _claude_id():
        return {"provider": "anthropic", "model": _claude_model(), "transport": "sdk",
                "served": f"anthropic:{_claude_model()}"}

    def _deepseek_id(model_id: str = ""):
        _m = model_id or model or _deepseek_model()
        return {"provider": "deepseek", "model": _m, "transport": "http",
                "served": f"deepseek:{_m}"}

    # provider ORDER: the selected primary first, then automatic fallbacks. DeepSeek is text-only,
    # so when it's primary, vision calls (image messages) skip BOTH its models and Gemini/Claude
    # serve them. Claude is the LAST resort everywhere except when explicitly chosen as primary.
    prov = _provider()
    if prov in ("deepseek", "ds"):
        # deepseek-v4-pro → deepseek-v4-flash → Gemini → Claude (last)
        seq = ((_try_deepseek, _deepseek_id),
               (lambda: _try_deepseek(fast_deepseek_model()),
                lambda: _deepseek_id(fast_deepseek_model())),
               (_try_gemini, _gemini_id),
               (_try_claude, _claude_id))
    elif prov in ("gemini", "google", "vertex"):
        seq = ((_try_gemini, _gemini_id), (_try_deepseek, _deepseek_id),
               (_try_claude, _claude_id))
    else:                                              # anthropic (Claude) primary — explicit choice
        seq = ((_try_claude, _claude_id), (_try_deepseek, _deepseek_id),
               (_try_gemini, _gemini_id))
    for fn, ident in seq:
        out = fn()
        if out:
            return out, ident()
    return "", {"provider": "", "model": "", "transport": "", "served": "none"}


def has_llm(eng_cfg=None) -> bool:
    """Is ANY brain available (DeepSeek, Gemini, or Claude)?"""
    return bool(_deepseek_key(eng_cfg)) or gemini_available() or bool(_claude_key(eng_cfg))


def vision_config(eng_cfg=None) -> str:
    """Identity of the provider+model that will actually serve a VISION call.

    NOT the same as `active_provider`, which answers for TEXT. A vision call skips both DeepSeek
    models (they cannot see images, `_msgs_have_image`), so the brain selected in config is
    routinely NOT the brain that judges a frame: with the default deepseek primary, footage QC is
    really served by Gemini, and by Claude only if Gemini is unavailable.

    This exists to be hashed into the verifier's cache key. Keying on `eng_cfg.anthropic_model`
    described a model that may never have run, so a Gemini verdict and a Claude verdict for the same
    frame collided on one key — and swapping GEMINI_API_KEY in or out silently reused the other
    provider's judgment. The string must therefore change whenever the answer could:
    provider, model id, and (for Gemini) the API-key vs Vertex transport."""
    if gemini_available() and _provider() not in ("anthropic", "claude"):
        return f"gemini:{_gemini_model()}:{'apikey' if _gemini_api_key() else 'vertex'}"
    if _claude_key(eng_cfg):
        return f"anthropic:{_claude_model()}"
    if gemini_available():
        return f"gemini:{_gemini_model()}:{'apikey' if _gemini_api_key() else 'vertex'}"
    return "none"


# a 1×1 black JPEG — the smallest valid image to probe the vision transport with (base64)
_PROBE_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAAB"
    "AAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q==")


def classify_vision_error(exc_or_text) -> str:
    """Bucket a vision-call failure: 'billing' (out of credits / payment), 'auth' (bad/expired key /
    unauthorized / permission), or 'transient' (rate-limit / quota-per-minute / overload / timeout /
    network — worth continuing, the per-beat breaker + retries handle it).

    Billing and auth are HARD-DOWN (retrying inside one render can't fix them → abort fast with an
    actionable message). 'transient' must NOT abort — it recovers.

    CRITICAL distinction (a real bug this fixes): a 429 RESOURCE_EXHAUSTED is EITHER credit
    depletion ('prepayment credits are depleted' → billing, hard) OR a RATE limit ('Quota exceeded
    for quota metric ... requests per minute' → transient, retry). Keying 'billing' on the words
    'quota'/'exhaust'/'resource_exhausted' misclassified an ordinary rate-limit blip as
    out-of-credits and hard-failed a render whose API was perfectly funded. So billing is keyed ONLY
    on MONEY words; anything about rate/quota-per-time is transient."""
    s = (str(exc_or_text) or "").lower()
    # MONEY words only — a real credit/payment problem. NOT 'quota'/'exhaust' (those are rate limits).
    _billing = ("credit", "billing", "prepayment", "insufficient fund", "balance is too low",
                "balance too low", "payment", "purchase", "top up", "top-up", "past due",
                "account is not active", "free tier")
    if any(k in s for k in _billing):
        return "billing"
    if any(k in s for k in ("unauthorized", "invalid api key", "invalid_api_key", "permission",
                            "forbidden", "authentication", "401", "403", "api key not valid",
                            "api_key_invalid", "api key expired")):
        return "auth"
    # everything else — rate limits ('requests per minute', 'quota exceeded', RESOURCE_EXHAUSTED
    # without money words), overload, timeout, network — is TRANSIENT and must not hard-fail.
    return "transient"


def vision_probe(eng_cfg=None, *, timeout_sec: float = 20.0, attempts: int = 3) -> tuple[bool, str]:
    """Health check of the ACTUAL vision chain (the same fallback order verify uses).
    Returns (ok, reason): ok=True → a provider answered; ok=False with reason in
    {'billing','auth','down'} → no vision provider is usable RIGHT NOW.

    RETRIES transient failures (rate-limit / overload / network) up to `attempts` times with
    backoff before concluding — a preflight that hard-fails a render must never trip on a single
    rate-limit blip on a perfectly funded API. A CONFIRMED billing/auth error returns immediately
    (no point retrying a money/key problem). 'down' = every provider only failed transiently.
    Never raises."""
    import time as _t
    cfg_id = vision_config(eng_cfg)
    if cfg_id == "none":
        return False, "auth"                       # no vision provider configured at all
    img = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                       "data": _PROBE_JPEG_B64}}
    msg = [{"role": "user", "content": [img, {"type": "text", "text": "Reply with the word OK."}]}]
    # probe each vision-capable provider the SAME order complete() would try them
    order = []
    if gemini_available() and _provider() not in ("anthropic", "claude"):
        order.append(("gemini", lambda: _gemini_complete("Reply OK.", msg, 8, _gemini_model())))
    if _claude_key(eng_cfg):
        order.append(("claude", lambda: _claude_complete("Reply OK.", msg, 8, eng_cfg, "")))
    if gemini_available() and _provider() in ("anthropic", "claude"):
        order.append(("gemini", lambda: _gemini_complete("Reply OK.", msg, 8, _gemini_model())))
    if not order:
        return False, "auth"
    worst = "down"
    for _attempt in range(max(1, attempts)):
        for _name, call in order:
            try:
                out = call()
                if out and out.strip():
                    return True, "ok"
                # empty non-error reply — transient blip, try the next provider
            except Exception as e:                        # noqa: BLE001
                kind = classify_vision_error(e)
                if kind in ("billing", "auth"):
                    return False, kind                    # confirmed money/key problem — retrying won't help
                worst = "down"                            # transient — may recover on retry
        if _attempt < attempts - 1:
            _t.sleep(min(8.0, 1.5 * (2 ** _attempt)))     # backoff before re-probing
    return False, worst


def active_provider(eng_cfg=None) -> str:
    """The provider that will actually serve a (text) call — the selected primary if available,
    else the first available fallback in order."""
    avail = {
        "deepseek": (bool(_deepseek_key(eng_cfg)), f"deepseek ({_deepseek_model()})"),
        "anthropic": (bool(_claude_key(eng_cfg)), f"anthropic ({_claude_model()})"),
        "gemini": (gemini_available(),
                   f"gemini ({_gemini_model()}, {'API key' if _gemini_api_key() else 'Vertex'})"),
    }
    prov = _provider()
    if prov in ("deepseek", "ds"):
        order = ("deepseek", "gemini", "anthropic")     # Claude last
    elif prov in ("gemini", "google", "vertex"):
        order = ("gemini", "deepseek", "anthropic")     # Claude last
    else:
        order = ("anthropic", "deepseek", "gemini")     # Claude chosen primary
    for k in order:
        ok, label = avail[k]
        if ok:
            return label
    return "none"
