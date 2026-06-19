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


def gemini_available() -> bool:
    """Is Gemini USABLE here (creds + SDK)? — independent of which provider is PRIMARY, so Gemini can
    serve as a fallback. Provider order lives in complete()."""
    if not _gcp_cred():
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except Exception:
        return False


def _gemini_client():
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        cred = _gcp_cred()
        # cred is verified to exist on disk; a stale env var pointing at a missing file must
        # not win over it (setdefault would keep the broken value)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred
        try:
            proj = json.load(open(cred)).get("project_id", "")
        except Exception:
            proj = os.environ.get("VIDLORE_GCP_PROJECT", "")
        if not proj:
            raise RuntimeError("no GCP project_id resolved for Gemini/Vertex")
        _CLIENT = genai.Client(vertexai=True, project=proj, location=_gemini_location())
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
    """Run an LLM completion. PRIMARY provider (default Claude) → the OTHER provider as automatic
    fallback. Each provider is RETRIED with exponential backoff on transient errors (rate-limit /
    overload / timeout) so a single hiccup never silently degrades the caller (e.g. analyze collapsing
    to its heuristic path and wrecking relevance). Returns the response text ('' on total failure)."""
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

    # provider ORDER: the selected primary first, then automatic fallbacks. DeepSeek is text-only,
    # so when it's primary, vision calls (image messages) skip BOTH its models and Gemini/Claude
    # serve them. Claude is the LAST resort everywhere except when explicitly chosen as primary.
    prov = _provider()
    if prov in ("deepseek", "ds"):
        # deepseek-v4-pro → deepseek-v4-flash → Gemini → Claude (last)
        seq = (_try_deepseek,
               lambda: _try_deepseek(fast_deepseek_model()),
               _try_gemini,
               _try_claude)
    elif prov in ("gemini", "google", "vertex"):
        seq = (_try_gemini, _try_deepseek, _try_claude)
    else:                                              # anthropic (Claude) primary — explicit choice
        seq = (_try_claude, _try_deepseek, _try_gemini)
    for fn in seq:
        out = fn()
        if out:
            return out
    return ""


def has_llm(eng_cfg=None) -> bool:
    """Is ANY brain available (DeepSeek, Gemini, or Claude)?"""
    return bool(_deepseek_key(eng_cfg)) or gemini_available() or bool(_claude_key(eng_cfg))


def active_provider(eng_cfg=None) -> str:
    """The provider that will actually serve a (text) call — the selected primary if available,
    else the first available fallback in order."""
    avail = {
        "deepseek": (bool(_deepseek_key(eng_cfg)), f"deepseek ({_deepseek_model()})"),
        "anthropic": (bool(_claude_key(eng_cfg)), f"anthropic ({_claude_model()})"),
        "gemini": (gemini_available(), f"gemini ({_gemini_model()}, Vertex)"),
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
