"""A cheaper Gemini endpoint may lower the bill. It may never become a new way for a render to die.

About 95% of a render's API spend is gemini-flash VISION, so the per-token price of that one model
sets the price of the whole product, and an OpenAI-compatible reseller in front of the same model is
worth having. But renders here cost six to eight hours, and the verifier reads an empty answer as
"no verdict" — which is how an outage once quietly downgraded a whole video. So the proxy is a fast
path and nothing more: tried first, and ANY failure falls through to the official Google endpoint,
loudly, once per process.

What "any failure" has to cover, because each of these has a different shape:
  * the host does not resolve, or the connection times out
  * a non-200 (bad key, out of credit, rate limit, 5xx)
  * a 200 whose body is malformed, or has no choices
  * a 200 that carries an EMPTY completion — success-shaped and worthless
"""
from __future__ import annotations

import json
import os

import pytest

from vidlore.clipstudio import llm as L


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setenv("VIDLORE_GEMINI_PROXY_BASE", "https://proxy.test/v1")
    monkeypatch.setenv("VIDLORE_GEMINI_PROXY_KEY", "sk-test")
    L.__dict__.pop("_PROXY_WARNED", None)
    L._USAGE.clear()
    yield
    L.__dict__.pop("_PROXY_WARNED", None)


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok_payload(text="YES", pt=1115, ct=1):
    return {"choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


def _patch_urlopen(monkeypatch, behaviour):
    import urllib.request as u
    monkeypatch.setattr(u, "urlopen", behaviour)


def _patch_sdk(monkeypatch, marker="OFFICIAL"):
    """Stand in for the google-genai path so the fallback is observable without a network call."""
    monkeypatch.setattr(L, "_gemini_complete_official", lambda *a, **k: marker, raising=False)


# ---------------------------------------------------------------- configuration
def test_the_proxy_is_off_unless_both_halves_are_set(monkeypatch):
    monkeypatch.delenv("VIDLORE_GEMINI_PROXY_KEY", raising=False)
    assert L._gemini_proxy() == ("", "")
    monkeypatch.setenv("VIDLORE_GEMINI_PROXY_KEY", "sk-test")
    monkeypatch.delenv("VIDLORE_GEMINI_PROXY_BASE", raising=False)
    assert L._gemini_proxy() == ("", "")


def test_a_configured_proxy_makes_gemini_available_without_the_sdk():
    """It speaks plain HTTP, so it works where google-genai is not installed at all."""
    assert all(L._gemini_proxy())
    assert L.gemini_available() is True


# ---------------------------------------------------------------- the happy path
def test_a_vision_message_is_converted_to_openai_content():
    body = L._openai_content([
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
    ])
    assert body[0] == {"type": "text", "text": "look"}
    assert body[1]["type"] == "image_url"
    assert body[1]["image_url"]["url"].startswith("data:image/png;base64,QUJD")


def test_a_plain_string_stays_a_plain_string():
    assert L._openai_content("hello") == "hello"


def test_the_proxy_answer_is_returned_and_billed(monkeypatch):
    sent = {}

    def fake(req, timeout=None):
        sent["url"] = req.full_url
        sent["auth"] = req.headers.get("Authorization")
        sent["body"] = json.loads(req.data)
        return _Resp(_ok_payload("YES", pt=1115, ct=1))

    _patch_urlopen(monkeypatch, fake)
    out = L._gemini_complete("sys", [{"role": "user", "content": "q"}], 20, "gemini-2.5-flash")
    assert out == "YES"
    assert sent["url"] == "https://proxy.test/v1/chat/completions"
    assert sent["auth"] == "Bearer sk-test"
    assert sent["body"]["model"] == "gemini-2.5-flash"
    assert sent["body"]["messages"][0] == {"role": "system", "content": "sys"}
    # booked under the SAME model key as the official path — never $0 for want of a price row
    e = L._USAGE["gemini-2.5-flash"]
    assert e["calls"] == 1 and e["prompt"] == 1115 and e["completion"] == 1


def test_a_claude_model_id_does_not_leak_into_the_proxy_call(monkeypatch):
    """Callers forward eng_cfg.anthropic_model verbatim; the official path already guards this."""
    sent = {}

    def fake(req, timeout=None):
        sent["body"] = json.loads(req.data)
        return _Resp(_ok_payload())

    _patch_urlopen(monkeypatch, fake)
    L._gemini_complete("", [{"role": "user", "content": "q"}], 20, "claude-opus-4")
    assert sent["body"]["model"].startswith("gemini")


# ---------------------------------------------------------------- every failure falls back
@pytest.mark.parametrize("boom,label", [
    (lambda req, timeout=None: (_ for _ in ()).throw(OSError("dns")), "host does not resolve"),
    (lambda req, timeout=None: (_ for _ in ()).throw(TimeoutError("slow")), "timeout"),
    (lambda req, timeout=None: _Resp({"error": {"message": "no credit"}}), "no choices"),
    (lambda req, timeout=None: _Resp({"choices": []}), "empty choices"),
    (lambda req, timeout=None: _Resp(_ok_payload("")), "empty completion"),
    (lambda req, timeout=None: _Resp(_ok_payload("   ")), "whitespace completion"),
])
def test_any_proxy_failure_falls_back_to_the_official_endpoint(monkeypatch, boom, label, capsys):
    _patch_urlopen(monkeypatch, boom)
    called = {}

    def fake_client():
        called["sdk"] = True
        raise RuntimeError("sdk reached")           # proves control passed to the official path

    monkeypatch.setattr(L, "_gemini_client", fake_client)
    with pytest.raises(RuntimeError, match="sdk reached"):
        L._gemini_complete("", [{"role": "user", "content": "q"}], 20, "gemini-2.5-flash")
    assert called.get("sdk") is True, f"{label}: did not fall back"


def test_an_empty_completion_is_a_failure_not_an_answer(monkeypatch):
    """A verifier that receives "" reads it as 'no verdict'. Success-shaped emptiness must not
    reach it — it must look exactly like the outage it is."""
    _patch_urlopen(monkeypatch, lambda req, timeout=None: _Resp(_ok_payload("")))
    with pytest.raises(RuntimeError, match="empty completion"):
        L._gemini_proxy_complete("", [{"role": "user", "content": "q"}], 20, "gemini-2.5-flash")


def test_the_fallback_is_loud_exactly_once(monkeypatch, capsys):
    _patch_urlopen(monkeypatch, lambda req, timeout=None: (_ for _ in ()).throw(OSError("dns")))
    monkeypatch.setattr(L, "_gemini_client", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            L._gemini_complete("", [{"role": "user", "content": "q"}], 20, "gemini-2.5-flash")
    out = capsys.readouterr().out
    assert out.count("Gemini proxy unavailable") == 1, "warn once per process, not per call"
    assert "the bill is higher" in out


def test_no_proxy_configured_leaves_the_official_path_untouched(monkeypatch):
    monkeypatch.delenv("VIDLORE_GEMINI_PROXY_BASE", raising=False)
    monkeypatch.delenv("VIDLORE_GEMINI_PROXY_KEY", raising=False)
    import urllib.request as u
    monkeypatch.setattr(u, "urlopen",
                        lambda *a, **k: pytest.fail("the proxy must not be called when unset"))
    monkeypatch.setattr(L, "_gemini_client", lambda: (_ for _ in ()).throw(RuntimeError("sdk")))
    with pytest.raises(RuntimeError, match="sdk"):
        L._gemini_complete("", [{"role": "user", "content": "q"}], 20, "gemini-2.5-flash")


def test_the_secret_is_never_printed(monkeypatch, capsys):
    _patch_urlopen(monkeypatch, lambda req, timeout=None: (_ for _ in ()).throw(OSError("dns")))
    monkeypatch.setattr(L, "_gemini_client", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        L._gemini_complete("", [{"role": "user", "content": "q"}], 20, "gemini-2.5-flash")
    assert "sk-test" not in capsys.readouterr().out
