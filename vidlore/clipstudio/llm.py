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

# ---------------------------------------------------------------------------
# TOKEN / COST ACCOUNTING
#
# A render makes hundreds of vision calls — 512 fresh on job 69d80e9dd4_v4 with a warm verdict
# cache, ~2000+ cold — and none of it was recorded anywhere, so "what did this render cost" had no
# answer and decisions like "should verify run a second round" were guesses. Every provider branch
# now reports its usage here; accounting NEVER raises, so a missing usage field can't fail a render.
#
# Prices are USD per 1M tokens and are configuration, not fact — override per model with
# VIDLORE_CLIPSTUDIO_PRICE_<MODEL>_IN / _OUT (dots and dashes become underscores) when the
# provider's rate card changes.
_USAGE: dict = {}
_CLAUDE_VISION_CALLS = [0]        # last-resort vision fallback counter (see _claude_vision_budget_ok)

_PRICE_DEFAULTS = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),   # no row → a lite call books $0 and hides its own spend
    "gemini-2.5-pro": (1.25, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "deepseek-v4-pro": (0.28, 0.42),
    "deepseek-v4-flash": (0.07, 0.14),
}

# WHICH PIPELINE STAGE is spending. record_usage has always accepted stage=, but no call site ever
# passed it, so every cost_report shipped `stages: {}` and "what did verify cost vs the still layer"
# could only be inferred by reading code. The provider functions are far below the stages that call
# them, so the label rides a context var: a stage wraps its work in `with llm.usage_stage("verify")`
# and every call underneath books against it. Thread-safe and task-safe by construction (ContextVar),
# which matters because verify/still/selfheal all fan out to worker threads.
import contextvars as _contextvars
_STAGE: "_contextvars.ContextVar[str]" = _contextvars.ContextVar("clipstudio_usage_stage",
                                                                default="")
# ContextVars are NOT inherited by plain worker threads, and the expensive stages (verify prefetch,
# still layer, self-heal) all fan out to ThreadPoolExecutors — their calls would book against no
# stage at all. So the label is also kept in a module global that any thread can read. One render
# per process is the production shape (the portal runs jobs serially), so the global is accurate;
# were two renders ever to share a process the ContextVar still gives the calling task the right
# label and only unlabelled worker calls could blur. Attribution is observability, never a decision.
_STAGE_GLOBAL = [""]


def set_stage(name: str) -> None:
    """Label everything booked from now on (until the next set_stage). Never raises."""
    try:
        n = (name or "").strip()[:48]
        _STAGE_GLOBAL[0] = n
        _STAGE.set(n)
    except Exception:                                  # noqa: BLE001 — accounting is never fatal
        pass


class usage_stage:
    """Scoped variant of set_stage for a bounded block of work."""

    def __init__(self, name: str):
        self.name = (name or "").strip()[:48]
        self._prev = ""

    def __enter__(self):
        self._prev = current_stage()
        set_stage(self.name)
        return self

    def __exit__(self, *exc):
        set_stage(self._prev)
        return False


def current_stage() -> str:
    try:
        v = _STAGE.get()
    except Exception:                                  # noqa: BLE001
        v = ""
    return v or _STAGE_GLOBAL[0]


def _price_for(model: str) -> tuple:
    key = (model or "").strip().lower()
    env = key.replace(".", "_").replace("-", "_").upper()
    lo, hi = _PRICE_DEFAULTS.get(key, (0.0, 0.0))
    try:
        lo = float(os.environ.get(f"VIDLORE_CLIPSTUDIO_PRICE_{env}_IN", "") or lo)
        hi = float(os.environ.get(f"VIDLORE_CLIPSTUDIO_PRICE_{env}_OUT", "") or hi)
    except (TypeError, ValueError):
        pass
    return (lo, hi)


def record_usage(model: str, *, prompt: int = 0, completion: int = 0, stage: str = "") -> None:
    """Book one call's token usage. Safe to call with zeros or junk.

    `stage` defaults to the ambient `usage_stage(...)` label, so provider functions book against
    whatever pipeline stage is running without every call site having to pass it. Tokens are
    booked per stage too — call counts alone can't tell a cheap stage from an expensive one."""
    try:
        k = (model or "unknown").strip().lower()
        e = _USAGE.setdefault(k, {"calls": 0, "prompt": 0, "completion": 0, "stages": {},
                                  "stage_tokens": {}})
        _p, _c = max(0, int(prompt or 0)), max(0, int(completion or 0))
        e["calls"] += 1
        e["prompt"] += _p
        e["completion"] += _c
        st = (stage or current_stage() or "").strip()
        if st:
            e["stages"][st] = e["stages"].get(st, 0) + 1
            _t = e.setdefault("stage_tokens", {}).setdefault(st, {"prompt": 0, "completion": 0})
            _t["prompt"] += _p
            _t["completion"] += _c
    except Exception:                                  # noqa: BLE001 — never break a render
        pass


def _usage_from(resp) -> tuple:
    """(prompt, completion) from a Gemini or Claude response object; (0, 0) when absent.

    THINKING tokens count as completion. Gemini reports them separately (thoughts_token_count) and
    they are billed at the output rate — a silently re-enabled thinking config would otherwise
    multiply the real bill several times over while the recorded cost stayed flat."""
    try:
        u = getattr(resp, "usage_metadata", None)      # Gemini
        if u is not None:
            return (int(getattr(u, "prompt_token_count", 0) or 0),
                    int(getattr(u, "candidates_token_count", 0) or 0)
                    + int(getattr(u, "thoughts_token_count", 0) or 0))
        u = getattr(resp, "usage", None)               # Claude
        if u is not None:
            return (int(getattr(u, "input_tokens", 0) or 0),
                    int(getattr(u, "output_tokens", 0) or 0))
    except Exception:                                  # noqa: BLE001
        pass
    return (0, 0)


def usage_summary() -> dict:
    """Everything booked so far: per-model tokens/calls plus a USD estimate."""
    out = {"models": {}, "calls": 0, "prompt": 0, "completion": 0, "usd": 0.0}
    for model, e in _USAGE.items():
        pin, pout = _price_for(model)
        usd = e["prompt"] / 1e6 * pin + e["completion"] / 1e6 * pout
        out["models"][model] = {**e, "usd": round(usd, 4),
                                "price_per_1m": {"in": pin, "out": pout}}
        out["calls"] += e["calls"]
        out["prompt"] += e["prompt"]
        out["completion"] += e["completion"]
        out["usd"] += usd
    out["usd"] = round(out["usd"], 4)
    return out


def reset_usage() -> None:
    """Start a fresh accounting scope. The portal is a long-lived process, so without this every
    job's cost_report accumulated every PREVIOUS job's spend too."""
    _USAGE.clear()
    _CLAUDE_VISION_CALLS[0] = 0
    set_stage("")                    # BOTH the global and the ContextVar — clearing only one
                                     # leaves the previous job's stage labelling the next one's


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
    _u = d.get("usage") or {}                          # OpenAI-shaped usage block
    record_usage(d.get("model") or model or "deepseek",
                 prompt=_u.get("prompt_tokens", 0), completion=_u.get("completion_tokens", 0))
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
    # A configured proxy is enough on its own: it speaks plain HTTP, so it works even where the
    # google-genai SDK is not installed. The SDK path stays the fallback when creds exist.
    if all(_gemini_proxy()):
        return True
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


# --- cheap Gemini proxy (OpenAI-compatible) ------------------------------------------------
# ~95% of a render's API spend is gemini-flash VISION, so the per-token price of that one model
# sets the cost of the whole product. A cheaper OpenAI-compatible reseller in front of the same
# model is therefore worth having — but only if it can never become a NEW way for a render to die.
# So it is strictly a fast path: the proxy is tried first, and ANY failure falls through to the
# official Google SDK path below, loudly, once per process.
def _gemini_proxy() -> tuple:
    """(base_url, api_key) when a proxy is configured, else ("", "")."""
    base = os.environ.get("VIDLORE_GEMINI_PROXY_BASE", "").strip().rstrip("/")
    key = os.environ.get("VIDLORE_GEMINI_PROXY_KEY", "").strip()
    return (base, key) if (base and key) else ("", "")


def _openai_content(content):
    """Anthropic-style content (str | text/image blocks) → OpenAI chat content."""
    if isinstance(content, str):
        return content
    out = []
    for b in (content if isinstance(content, list) else [content]):
        if not isinstance(b, dict):
            out.append({"type": "text", "text": str(b)})
        elif b.get("type") == "text":
            out.append({"type": "text", "text": b.get("text", "")})
        elif b.get("type") == "image":
            src = b.get("source", {}) or {}
            data, mime = src.get("data", ""), src.get("media_type", "image/jpeg")
            if data:
                out.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"}})
    return out or ""


def _gemini_proxy_complete(system, messages, max_tokens, model) -> str:
    """One call through the OpenAI-compatible proxy. Raises on any problem — the caller falls back.

    Deliberately strict about what counts as success: a 200 that carries no text is a FAILURE here,
    not an empty answer handed to a verifier. A verifier that receives "" reads it as "no verdict",
    which is how a whole render once got downgraded by an outage nobody noticed.
    """
    import json as _j
    import urllib.request as _u
    base, key = _gemini_proxy()
    mdl = (model or "").strip()
    if not mdl.lower().startswith("gemini"):
        mdl = _gemini_model()
    msgs = ([{"role": "system", "content": system}] if system else [])
    for m in messages:
        role = "assistant" if m.get("role") == "assistant" else "user"
        msgs.append({"role": role, "content": _openai_content(m.get("content"))})
    body = _j.dumps({"model": mdl, "messages": msgs,
                     "max_tokens": int(max_tokens)}).encode("utf-8")
    req = _u.Request(f"{base}/chat/completions", data=body,
                     headers={"Authorization": f"Bearer {key}",
                              "Content-Type": "application/json"})
    try:
        sec = float(os.environ.get("VIDLORE_GEMINI_TIMEOUT_SEC", "").strip() or 120.0)
    except (TypeError, ValueError):
        sec = 120.0
    with _u.urlopen(req, timeout=max(5.0, sec)) as r:
        d = _j.loads(r.read())
    txt = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not str(txt).strip():
        raise RuntimeError("gemini proxy returned an empty completion")
    u = d.get("usage") or {}
    # Booked against the SAME model key as the official path, deliberately, so a render's cost_report
    # never silently drops to $0 for want of a price row — the failure mode this project has already
    # been bitten by. Two honesty notes that belong in the code, not in someone's memory:
    #   * the $ figure is then priced at GOOGLE's rate, so for proxy-served calls it is an UPPER
    #     BOUND, not the invoice. The real bill is whatever the reseller charges.
    #   * the proxy's OpenAI-compat layer counts image tokens differently from the native SDK
    #     (measured on one identical 1080p frame: 1115 prompt tokens here vs 273 via google-genai),
    #     so token totals are not comparable across the two paths either.
    # The call counter below is what makes the split visible per render.
    from . import perf_metrics as _pm_ok
    _pm_ok.incr("llm.gemini_proxy.ok")
    record_usage(mdl, prompt=int(u.get("prompt_tokens") or 0),
                 completion=int(u.get("completion_tokens") or 0))
    return str(txt)


def _gemini_complete(system, messages, max_tokens, model) -> str:
    _base, _key = _gemini_proxy()
    if _base and _key:
        try:
            return _gemini_proxy_complete(system, messages, max_tokens, model)
        except Exception as _pe:                      # noqa: BLE001 — never fatal, always audible
            from . import perf_metrics as _pm_px
            _pm_px.incr("llm.gemini_proxy.fallback")
            if not globals().get("_PROXY_WARNED"):
                globals()["_PROXY_WARNED"] = True
                # "for the rest of this run" is what this line USED to say, and it was false: the
                # fallback is per CALL — the proxy is tried again on the very next one. Measured on
                # job 218acdfe10, where a single HTTP 500 printed that sentence and the proxy then
                # went on to serve 3567 of 4129 calls: an operator reading it would have priced the
                # whole render at Google rates and over-estimated the bill roughly six-fold.
                print(f"[clipstudio] ⚠ Gemini proxy failed on this call "
                      f"({type(_pe).__name__}: {str(_pe)[:120]}) — answered by the official Google "
                      f"endpoint instead. The proxy is retried on the NEXT call; this warning is "
                      f"printed once. The real split is reported at the end of the render (and in "
                      f"perf_report.json: llm.gemini_proxy.ok / .fallback).", flush=True)
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
    except Exception as _te:                          # noqa: BLE001
        # NOT silent: thinking tokens bill at the OUTPUT rate, so an SDK rename that drops this
        # config quietly multiplies the render's bill (measured probe: 400-900 thought-tokens on a
        # verifier-shaped payload vs ~100 answer tokens). Loud once, and counted.
        from . import perf_metrics as _pm_tc
        _pm_tc.incr("llm.thinking_config.dropped")
        if not globals().get("_THINK_WARNED"):
            globals()["_THINK_WARNED"] = True
            print(f"[clipstudio] ⚠ COST: Gemini thinking_config could NOT be set "
                  f"({type(_te).__name__}) — thought tokens may now be billed as output.",
                  flush=True)
    # callers forward eng_cfg.anthropic_model verbatim — a Claude id must not kill the fallback
    mdl = (model or "").strip()
    if not mdl.lower().startswith("gemini"):
        mdl = _gemini_model()
    resp = cl.models.generate_content(
        model=mdl, contents=contents,
        config=types.GenerateContentConfig(**gcfg))
    _p, _c = _usage_from(resp)
    record_usage(mdl, prompt=_p, completion=_c)
    return getattr(resp, "text", None) or ""


def _claude_vision_budget_ok(messages) -> bool:
    """Claude is the LAST-RESORT vision fallback and costs ~10x Gemini flash on input
    ($3.00 vs $0.30 per 1M). It is silent by design, so a Gemini-side outage could quietly
    re-price a whole render. Allow a generous number of genuine transient fallbacks, then
    refuse — the verifier's existing fail-closed semantics treat a refusal exactly like an
    outage (release-block; never a silent pass). Env: VIDLORE_CLIPSTUDIO_MAX_CLAUDE_VISION."""
    has_image = False
    try:
        for m in (messages or []):
            for part in (m.get("content") or []):
                if isinstance(part, dict) and part.get("type") == "image":
                    has_image = True
                    break
    except Exception:                                  # noqa: BLE001
        return True
    if not has_image:
        return True
    try:
        cap = int(os.environ.get("VIDLORE_CLIPSTUDIO_MAX_CLAUDE_VISION", "50") or 50)
    except (TypeError, ValueError):
        cap = 50
    _CLAUDE_VISION_CALLS[0] += 1
    n = _CLAUDE_VISION_CALLS[0]
    if n == 1:
        print("[clipstudio] ⚠ COST: vision fell back to CLAUDE (~10x Gemini flash input price) — "
              "check the Gemini backend.", flush=True)
    if cap > 0 and n > cap:
        if n == cap + 1:
            print(f"[clipstudio] ⛔ COST GUARD: Claude vision call budget ({cap}) exhausted — "
                  f"refusing further Claude vision calls. Beats verify fail-closed from here "
                  f"(release-block, never a silent pass). Restore the Gemini backend.", flush=True)
        return False
    return True


def _claude_complete(system, messages, max_tokens, eng_cfg, model) -> str:
    import anthropic
    key = _claude_key(eng_cfg)
    if not key:
        return ""
    if not _claude_vision_budget_ok(messages):
        return ""                                      # same shape as "no key": fail-closed
    cl = anthropic.Anthropic(api_key=key)
    mdl = model or (getattr(eng_cfg, "anthropic_model", "") if eng_cfg else "") or _claude_model()
    if not mdl.lower().startswith("claude"):     # a Gemini id forwarded by a caller
        mdl = ((getattr(eng_cfg, "anthropic_model", "") if eng_cfg else "") or _claude_model())
        if not mdl.lower().startswith("claude"):   # eng_cfg itself misconfigured with a non-Claude id
            mdl = "claude-sonnet-4-6"
    resp = cl.messages.create(model=mdl, max_tokens=max_tokens,
                              system=(system or ""), messages=messages)
    _p, _c = _usage_from(resp)
    record_usage(mdl, prompt=_p, completion=_c)
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


_COMPLETE_ORIG = complete


def complete_ex(*, system: str = "", messages, max_tokens: int = 1024,
                eng_cfg=None, model: str = "") -> tuple:
    """Run an LLM completion and report WHO ACTUALLY SERVED it.

    Returns (text, meta) where meta = {"provider", "model", "transport", "served"} describes
    the provider branch that produced the text — not the configured/predicted one. "served" is
    the canonical identity string in exactly vision_config()'s format, so cache keys derived
    from a prediction and keys derived from the actual server are directly comparable. On
    total failure -> ("", {"served": "none", ...}). The provider ladder, retries, backoff and
    fallback order are byte-identical to the historical complete().

    LEGACY-SEAM COMPAT: tests and tools have always stubbed `llm.complete` directly. When
    that name is monkeypatched, honor the stub — route through it and report no provider
    identity ('' -> consumers fall back to the predicted identity, exactly the pre-upgrade
    behavior). The real ladder runs only through the unpatched module function."""
    if _COMPLETE_ORIG is not None and globals().get("complete") is not _COMPLETE_ORIG:
        _out = globals()["complete"](system=system, messages=messages,
                                     max_tokens=max_tokens, eng_cfg=eng_cfg, model=model)
        return _out, {"provider": "", "model": "", "transport": "", "served": ""}
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
    from . import perf_metrics as _pm_llm
    for fn, ident in seq:
        out = fn()
        if out:
            return out, ident()
        _pm_llm.incr("llm.branch_fail")                # eligible-or-skipped branch, no answer
    _pm_llm.incr("llm.total_failure")
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
