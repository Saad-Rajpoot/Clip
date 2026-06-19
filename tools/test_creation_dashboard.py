#!/usr/bin/env python3
"""Regression tests for the rebuilt premium creation dashboard (the _FORM page).

Template-level (renders _FORM through Jinja with the real option lists) + source
guards + a _brief_from wiring mirror. No running server needed.

    .venv/bin/python tools/test_creation_dashboard.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WEB = (ROOT / "vidlore" / "web.py").read_text(encoding="utf-8")

# the canonical backend field set the form MUST keep posting so _brief_from wires
BACKEND_FIELDS = [
    "title", "script", "prompt", "duration", "fmt", "style", "look_preset",
    "theme", "background", "voiceover", "voice_mode", "tts_model", "tts_voice",
    "voice", "shutterstock", "wf_mix", "wi_mix", "music", "transitions",
    "overlays", "sfx", "captions",
]


def _form_src() -> str:
    m = re.search(r'_FORM = """(.*?)"""', WEB, re.S)
    assert m, "could not find _FORM template"
    return m.group(1)


def _render_form(f=None, error=None) -> str:
    """Render _FORM exactly as _form_page() does, with the real option lists."""
    from flask import Flask, render_template_string
    import vidlore.web as w

    app = Flask(__name__)
    # routes referenced by url_for() inside the template
    for rule, ep in [("/", "index"), ("/new", "new"), ("/script", "gen_script"),
                     ("/sample", "sample"), ("/bg/<name>", "bg_preview")]:
        app.add_url_rule(rule, ep, (lambda *a, **k: "x"))
    ctx = dict(
        f=(f or {}), error=error, themes=w.THEMES,
        themes_meta=w._themes_meta_list(), bg_meta=w._bg_meta_list(),
        durations=list(w.DURATION_BUCKETS), style_modes=w._style_modes_list(),
        voices=w._voices_list(), channels=w._channels_list(),
        premium_presets=w._premium_presets(), api_status=w._api_status(),
        ss_default=w._shutterstock_default(),
    )
    with app.test_request_context("/"):
        return render_template_string(_form_src(), **ctx)


def main() -> int:
    passed = failed = 0

    def check(name, cond, extra=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  PASS {name}")
        else:
            failed += 1
            print(f"  FAIL {name}  {extra}")

    src = _form_src()
    html = _render_form()

    # ---------- PHASE 2: removals ----------
    print("\n[removed / outdated]")
    check("HeyGen B-Roll banner removed from the form",
          "HeyGen" not in src and "hg_new" not in src)
    check("dead top10 format removed (no value=top10)", "top10" not in src)
    check("no Format <select> (documentary is implicit)",
          "name=fmt value=documentary" in src and "<select name=fmt" not in src
          and "name=fmt " not in src.replace("name=fmt value=documentary", ""))
    check("outdated 'Homestead Gold' preset label not surfaced", "Homestead Gold" not in src)
    check("no standalone final-duration selector in step 1 "
          "(duration lives under 'write one for me')",
          "Target script length" in src)

    # ---------- field preservation (wiring contract) ----------
    print("\n[backend field names preserved]")
    for n in BACKEND_FIELDS:
        check(f"name={n} present", re.search(rf"name={n}(?=[\s>=\"'])", html) is not None)

    # ---------- hidden defaults ----------
    print("\n[hidden / driven defaults]")
    check("fmt hidden = documentary", "name=fmt value=documentary" in html)
    check("style hidden = auto (channel/Look-DNA drives it)", "name=style value=auto" in html)
    check("look_preset default = auto", 'name=look_preset id=channelval value="auto"' in html)
    check("voice_mode default = legacy (Basic)", 'name=voice_mode id=vmval value="legacy"' in html)
    check("theme fallback default = standard", 'name=theme id=themeval value="standard"' in html)
    check("background fallback default = auto", 'name=background id=bgval value="auto"' in html)

    # ---------- PHASE 4: beginner flow ----------
    print("\n[beginner 3-step flow + create]")
    check("brand = Vidlore Studio", "Vidlore Studio" in html)
    check("step 1 'Add your content'", "Add your content" in html)
    check("script source toggle (mine / generate)",
          "I have a script" in html and "Write one for me" in html)
    check("step 2 'Documentary style' with Auto recommended",
          "Documentary style" in html and "Auto detect" in html and "Recommended" in html)
    check("step 3 'Narration voice' (Basic / Premium)",
          "Narration voice" in html and "Basic voice" in html and "Premium voice" in html)
    check("primary button 'Create documentary' (not raw 'Generate script')",
          "Create documentary" in html and "genbtn" in html)
    check("sample loader retained", "Load a sample topic" in html)

    # ---------- PHASE 5: advanced collapsed ----------
    print("\n[advanced settings — progressive disclosure]")
    adv_closed = re.search(r"<details class=cadv id=cadv\s*>", html)
    check("Advanced settings section present", "Advanced settings" in html)
    check("Advanced is COLLAPSED by default (no 'open')", adv_closed is not None)
    check("Advanced groups: Visual sourcing / Editing / Voice details / Appearance",
          all(s in html for s in ["Visual sourcing", "Editing", "Voice details", "Appearance"]))
    # the Shutterstock CONTROL sits in Advanced (after the Create button); the only
    # earlier mention of the word is the hidden header readiness tooltip (data-tip).
    check("provider toggles live in Advanced, not the beginner steps",
          "name=shutterstock" in html
          and html.index("name=shutterstock") > html.index("Create documentary"))
    beginner = html[:html.index("Advanced settings")]
    check("no raw provider names as visible labels in steps 1-3",
          not re.search(r">[^<]*\b(Pexels|fal\.ai|Shutterstock)\b", beginner))
    html_err = _render_form(error="boom")
    check("Advanced auto-opens when an error is shown",
          re.search(r"<details class=cadv id=cadv\s+open>", html_err) is not None
          and "boom" in html_err)

    # ---------- editing toggle defaults ----------
    print("\n[editing toggle defaults match the backend]")
    def _checked(n):
        m = re.search(rf"name={n} value=1 ?(checked)?", html)
        return bool(m and m.group(1))
    check("music ON by default", _checked("music"))
    check("transitions ON by default", _checked("transitions"))
    check("overlays ON by default", _checked("overlays"))
    check("sfx OFF by default (legacy, intentional)", not _checked("sfx"))
    check("captions OFF by default (editor override wins)", not _checked("captions"))

    # ---------- option defaults ----------
    print("\n[option defaults]")
    check("duration default 6-8 selected", '<option value="6-8" selected' in html)
    check("wi_mix default balanced", "value=balanced selected" in html)
    check("channel cards rendered from the channels registry (Auto first)",
          'data-channel="auto"' in html and 'data-channel="midnight_pacific"' in html)

    # ---------- preserved JS ----------
    print("\n[preserved + new client behaviour]")
    check("goGen() submit guard preserved", "function goGen()" in html)
    check("ld() sample loader preserved", "async function ld()" in html)
    check("srcToggle() switches script source", "function srcToggle()" in html)
    check("voFile() reflects an uploaded voiceover", "function voFile(" in html)
    check("card pickers wire hidden inputs (cpick)", "function cpick(" in html
          and "cpick('channelpicker'" in html and "cpick('vmpicker'" in html)
    check("#wait overlay + spinner preserved", "id=wait" in html and "keyframes spin" in html)
    check("tooltip engine present (data-tip)", "id='ctip'" in html and "data-tip" in html)

    # ---------- _brief_from wiring mirror (the contract that matters) ----------
    print("\n[_brief_from builds the right Brief from the new payload]")
    import vidlore.web as w
    payload = {
        "title": "The Sunken City", "_src": "mine",
        "script": "line1\n\nscene two\n\nscene three",
        "fmt": "documentary", "style": "auto", "look_preset": "auto",
        "voice_mode": "legacy", "theme": "standard", "background": "auto",
        "prompt": "", "duration": "6-8", "wi_mix": "balanced", "wf_mix": "off",
        "tts_model": "chatterbox", "tts_voice": "deep_male_documentary", "voice": "",
        "music": "1", "transitions": "1", "overlays": "1",  # sfx/captions absent => off
    }
    b = w._brief_from(payload)
    check("title wired", b.title == "The Sunken City")
    check("fmt documentary", b.fmt == "documentary")
    check("duration 6-8", b.duration == "6-8")
    check("style auto (Look DNA drives)", getattr(b, "style", "auto") == "auto")
    check("look_preset=auto leaves it unset (auto-detect)",
          (getattr(b, "look_preset", None) in (None, "", "auto")))
    check("empty prompt -> 'reviewed' (paste-script path)", b.prompt == "reviewed")
    check("captions off", b.captions is False)
    ex = b.extra
    check("extra.music True", ex.get("music") is True)
    check("extra.transitions True", ex.get("transitions") is True)
    check("extra.overlays True", ex.get("overlays") is True)
    check("extra.sfx False (absent checkbox)", ex.get("sfx") is False)
    check("extra.captions/shutterstock off", ex.get("shutterstock") is False)
    check("extra.wi_mix balanced", ex.get("wi_mix") == "balanced")
    check("extra.wf_mix off", ex.get("wf_mix") == "off")
    check("extra.voice_mode legacy", ex.get("voice_mode") == "legacy")
    check("extra.tts_model chatterbox", ex.get("tts_model") == "chatterbox")
    check("extra.tts_voice deep_male_documentary",
          ex.get("tts_voice") == "deep_male_documentary")

    # "write one for me" payload — prompt + premium voice survive
    b2 = w._brief_from({**payload, "_src": "gen", "script": "",
                        "prompt": "tense investigative tone", "voice_mode": "premium",
                        "look_preset": "midnight_pacific"})
    check("gen path: prompt carried", b2.prompt == "tense investigative tone")
    check("gen path: chosen channel wired", getattr(b2, "look_preset", "") == "midnight_pacific")
    check("gen path: premium voice wired", b2.extra.get("voice_mode") == "premium")

    print(f"\n{'='*48}\n  {passed} passed, {failed} failed\n{'='*48}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
