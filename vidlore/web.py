"""Minimal web wizard over the existing pipeline — the browser version of
Vidlore's "generate" flow, but with the script-review step front and centre.

Run:  python -m vidlore.web   (then open http://127.0.0.1:5000)

Flow:  brief form -> generate script -> REVIEW/EDIT in browser ->
        render (live progress) -> preview + download.

Single file, stdlib-threaded background jobs, no database. Same pipeline
the CLI uses, so every provider/cache/differentiator applies here too.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    url_for,
)

from .brief import DURATION_BUCKETS, THEMES, Brief
from .config import load_config
from .license import activate as lic_activate
from .license import is_activated, make_key
from .pipeline import (
    load_brief,
    render_from_script,
    run_dir_for,
    write_script,
)
from .backgrounds import BG_NAMES, bg_meta_list, thumbnail as bg_thumbnail
from .themes import theme_meta, theme_names


def _themes_meta_list() -> list[dict]:
    """All 5 themes packaged for the rich picker — name, title, description
    and a primary/secondary RGB tuple so the template can render the
    preview gradient + accent dot without importing colour logic itself."""
    return [theme_meta(n) for n in theme_names()]


def _bg_meta_list() -> list[dict]:
    """10 backgrounds + an implicit Auto at the front, for the picker grid."""
    return [{"name": "auto", "title": "Auto"}] + bg_meta_list()

# Load .env into the environment at startup so the license secret /
# pre-activation key (and all API keys) are available before any request.
load_config()

app = Flask(__name__)
OUT = Path("output")


@app.before_request
def _license_gate():
    # License gate disabled — app runs without a license key.
    return None

# job_id -> {status, pct, msg, error, title, run_dir}
JOBS: dict[str, dict] = {}


# ---- Dashboard option lists ------------------------------------------- #
def _style_modes_list() -> list:
    """Style Mode cards for the wizard (auto + every defined mode)."""
    from .style_modes import all_modes
    out = [dict(name="auto", label="Auto (match topic)",
                desc="Pick the cinematic personality automatically from the "
                     "theme & topic.")]
    _desc = {
        "standard": "Balanced, neutral documentary pacing and transitions.",
        "epic": "Grand & unhurried — long lingering holds, sweeping music "
                "swells, reflective dissolves, slow majestic camera.",
        "true_crime": "Cold & tense — hard cuts, held evidence shots, low "
                      "tension score, case-file & surveillance graphics.",
    }
    for m in all_modes():
        out.append(dict(name=m.name, label=m.label,
                        desc=_desc.get(m.name, "")))
    return out


_VOICES = [
    ("en-US-GuyNeural", "Guy — deep US male (default narrator)"),
    ("en-US-ChristopherNeural", "Christopher — warm US male"),
    ("en-US-BrianNeural", "Brian — calm US male"),
    ("en-GB-RyanNeural", "Ryan — British male"),
    ("en-US-AriaNeural", "Aria — US female"),
    ("en-US-JennyNeural", "Jenny — soft US female"),
    ("en-GB-SoniaNeural", "Sonia — British female"),
]


def _voices_list() -> list:
    return [dict(id=v, label=lbl) for v, lbl in _VOICES]


def _premium_presets() -> list:
    """The 5 documentary narrator presets for the Premium-voice dropdown."""
    try:
        from .voice_presets import preset_list
        return preset_list()
    except Exception:  # noqa: BLE001
        return []


def _voice_status() -> dict:
    """Backend readiness for the dashboard status chips. Never raises."""
    try:
        from . import tts_backends as tb
        avail = tb.available_backends()  # name -> "" (ready) | reason
        cfg = load_config()
        return {
            "chatterbox": {"ready": avail.get("chatterbox") == "",
                           "reason": avail.get("chatterbox", "n/a")},
            "kokoro": {"ready": avail.get("kokoro") == "",
                       "reason": avail.get("kokoro", "n/a")},
            "edge": {"ready": avail.get("edge") == "",
                     "reason": avail.get("edge", "n/a")},
            "cache": bool(getattr(cfg, "tts_cache", True)),
            "premium_model": getattr(cfg, "tts_model", "chatterbox"),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# --------------------------------------------------------------------- #
# Channel / Look DNA picker — the PRIMARY editorial-identity control.
# Curated subset (the 3 actively-tested presets + Auto) shown in the
# dashboard.  Other presets (homestead, true_crime, netflix_epic,
# standard) remain available via CLI but are NOT exposed here — they
# are legacy / duplicate identities now superseded by the 3 channels.
# --------------------------------------------------------------------- #
def _channels_list() -> list:
    """Editorial-identity presets shown in the dashboard.  Each entry
    maps 1:1 to a YAML in vidlore/look_presets/.  Selecting one sets
    `brief.look_preset` which downstream wires through every system —
    pacing, cards, subtitles, music, footage taste, reveal psychology,
    camera motion, grading, transitions."""
    return [
        dict(name="auto",             label="Auto (no channel)",
             desc="Use brief defaults — pacing, fonts, music, cards "
                  "follow the theme and style mode below."),
        dict(name="midnight_pacific", label="Midnight Pacific",
             desc="Netflix-style cinematic investigative.  Deep dark "
                  "grade, slow patient pacing (4-5 scenes for 6-8 min), "
                  "Helvetica Neue captions, dark_investigation music, "
                  "tension/evidence reveal psychology."),
        dict(name="atlas_explained",  label="Atlas Explained",
             desc="Vox/Harris-style explainer.  Bright editorial grade, "
                  "fast cuts, dense scene count (10-12 for 6-8 min), "
                  "Bricolage Grotesque captions, tech_cyber + climax_"
                  "build music, data reveals (percent / billion)."),
        dict(name="amber_chronicles", label="Amber Chronicles",
             desc="Slow emotional historical.  Warm sepia grade, long "
                  "contemplative holds (2-3 scenes for 6-8 min), "
                  "Playfair Display captions, historical_epic + "
                  "emotional_piano music, emotional/memory reveals."),
    ]


def _api_status() -> list:
    """Build per-key status pills for the dashboard.  Reports which
    backend integrations are actually reachable (key present in .env)
    so the user never picks an option whose key is missing.  Does
    NOT make any network call here — that runs in the live smoke
    test on first request to the relevant feature."""
    cfg = load_config()
    rows = [
        ("Anthropic (script editor)", bool(cfg.anthropic_api_key),
         "required for AI-generated scripts; paste your own script if missing"),
        ("Pexels (primary stock video)", bool(cfg.pexels_api_key),
         "free key at pexels.com/api — primary footage source"),
        ("Shutterstock (fallback)", bool(cfg.shutterstock_api_key),
         "optional tier-2 footage; preview MP4s WATERMARKED on free-trial"),
        ("fal.ai (AI image fallback)", bool(cfg.fal_key),
         "premium AI image generation when stock misses; optional"),
    ]
    return [dict(label=lbl, connected=ok, hint=hint)
            for lbl, ok, hint in rows]


def _shutterstock_default() -> str:
    """Default value for the Shutterstock toggle on the form.  '1' when
    a key is present, '0' otherwise — so a fresh install never tries
    to call an API it doesn't have credentials for."""
    return "1" if load_config().shutterstock_api_key else "0"

_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>{{title}}</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}
body{font:16px/1.5 system-ui,sans-serif;max-width:840px;margin:32px auto;
padding:0 18px;background:#0e1116;color:#e6e6e6}
h1{font-size:22px;margin:0 0 4px} .sub{color:#8b95a5;margin:0 0 22px}
label{display:block;margin:14px 0 5px;font-weight:600;font-size:14px}
input,select,textarea{width:100%;padding:10px;border:1px solid #2b313b;
border-radius:8px;background:#161b22;color:#e6e6e6;font:inherit;box-sizing:border-box}
textarea{min-height:230px;font-family:ui-monospace,monospace;font-size:14px}
.row{display:flex;gap:14px}.row>div{flex:1}
button{margin-top:20px;padding:12px 20px;border:0;border-radius:8px;
background:#3b82f6;color:#fff;font:600 15px system-ui;cursor:pointer}
button.ghost{background:#222a35;color:#cdd5df}
a{color:#6ea8fe} .err{background:#3a1620;border:1px solid #7d2435;
padding:10px 14px;border-radius:8px;margin:14px 0;color:#ffb3c0}
.bar{height:14px;background:#1c222b;border-radius:7px;overflow:hidden;margin:8px 0}
.fill{height:100%;width:0;background:#3b82f6;transition:width .4s}
.card{background:#11161d;border:1px solid #222a35;border-radius:12px;padding:18px}
small{color:#8b95a5} video{width:100%;border-radius:10px;margin-top:14px}
.nav{display:flex;justify-content:space-between;align-items:center;margin:0 0 20px}
.nav a.btn{background:#3b82f6;color:#fff;padding:10px 16px;border-radius:8px;
text-decoration:none;font-weight:600;font-size:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}
.tile{background:#11161d;border:1px solid #222a35;border-radius:12px;overflow:hidden}
.tile img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#1c222b}
.tileph{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;font-size:30px;color:#3a4660;
  background:linear-gradient(135deg,#161b22,#1c222b)}
.tile .m{padding:10px 12px}.tile .m b{font-size:14px;display:block;margin-bottom:6px}
.pill{display:inline-block;font-size:12px;padding:2px 9px;border-radius:999px}
.pill.run{background:#13354d;color:#7fc1ff}.pill.err{background:#3a1620;color:#ffb3c0}
/* Rich theme picker (Vidlore-style grid). Click selects, hidden input
   carries the value; selected card gets a glowing border + check. */
.themes{display:grid;grid-template-columns:1fr;gap:10px;margin:6px 0 4px}
.theme-card{display:flex;align-items:stretch;gap:14px;background:#11161d;
border:1px solid #222a35;border-radius:12px;padding:0;cursor:pointer;
overflow:hidden;transition:border-color .15s,background .15s,transform .08s;
position:relative}
.theme-card:hover{border-color:#3a4660;background:#141a23}
.theme-card.sel{border-color:#3b82f6;background:#0f1a2b;
box-shadow:0 0 0 1px #3b82f6 inset}
.theme-card .swatch{flex:0 0 130px;align-self:stretch;min-height:80px;
position:relative;border-right:1px solid #1a212c}
.theme-card .swatch .dot{position:absolute;left:12px;top:12px;width:14px;
height:14px;border-radius:50%;box-shadow:0 0 0 2px rgba(0,0,0,.3)}
.theme-card .meta{padding:14px 16px 14px 4px;flex:1;min-width:0}
.theme-card .meta b{display:block;font-size:15px;margin-bottom:4px;
color:#e6e6e6;letter-spacing:.2px}
.theme-card .meta span{color:#8b95a5;font-size:13px;line-height:1.45;
display:block}
.theme-card .check{position:absolute;top:12px;right:14px;width:22px;
height:22px;border-radius:50%;background:#3b82f6;color:#fff;display:none;
align-items:center;justify-content:center;font-size:14px;font-weight:700}
.theme-card.sel .check{display:flex}
/* Background picker — 10 thumbnail tiles + Auto (Vidlore "Choose
   background image" step). Smaller cards in a 5-col grid. */
.bgs{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:6px 0 4px}
.bg-card{position:relative;background:#11161d;border:1px solid #222a35;
border-radius:10px;cursor:pointer;overflow:hidden;
transition:border-color .15s,background .15s}
.bg-card:hover{border-color:#3a4660}
.bg-card.sel{border-color:#3b82f6;box-shadow:0 0 0 1px #3b82f6 inset}
.bg-card .thumb{width:100%;aspect-ratio:16/10;background:#1c222b;
display:block;background-size:cover;background-position:center}
.bg-card .thumb.auto{display:flex;align-items:center;justify-content:center;
color:#8b95a5;font-size:13px;
background:repeating-linear-gradient(45deg,#1c222b 0 8px,#161b22 8px 16px)}
.bg-card .lbl{padding:7px 8px;font-size:12px;color:#cdd5df;text-align:center;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bg-card .check{position:absolute;top:6px;right:6px;width:20px;height:20px;
border-radius:50%;background:#3b82f6;color:#fff;display:none;
align-items:center;justify-content:center;font-size:12px;font-weight:700}
.bg-card.sel .check{display:flex}
@media (max-width:640px){.bgs{grid-template-columns:repeat(3,1fr)}}
.empty{color:#8b95a5;text-align:center;padding:50px 0}
/* ---- Professional dashboard chrome ---- */
.topbar{display:flex;align-items:center;justify-content:space-between;
margin:0 0 26px;padding-bottom:16px;border-bottom:1px solid #1c222b}
.brand{display:flex;align-items:center;gap:12px}
.brand .logo{width:38px;height:38px;border-radius:10px;
background:linear-gradient(135deg,#3b82f6,#8b5cf6);display:flex;
align-items:center;justify-content:center;font-weight:800;font-size:20px;
color:#fff;box-shadow:0 4px 14px rgba(59,130,246,.35)}
.brand b{font-size:18px;letter-spacing:.3px}.brand small{display:block;
color:#8b95a5;font-size:12px;margin-top:1px}
.topbar a.btn{background:#1b2330;color:#cdd5df;padding:9px 15px;
border-radius:9px;text-decoration:none;font-weight:600;font-size:13px;
border:1px solid #2b333f}.topbar a.btn:hover{border-color:#3a4660}
.section{background:#11161d;border:1px solid #1f2630;border-radius:14px;
padding:20px 22px;margin:0 0 18px}
.section>h2{font-size:13px;text-transform:uppercase;letter-spacing:1.2px;
color:#7c8aa0;margin:0 0 4px;font-weight:700}
.section>.shint{color:#6b7686;font-size:12.5px;margin:0 0 16px}
.modes{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:6px 0 2px}
.mode-card{position:relative;background:#0f141b;border:1px solid #222a35;
border-radius:11px;padding:13px 15px;cursor:pointer;
transition:border-color .15s,background .15s}
.mode-card:hover{border-color:#3a4660;background:#131a23}
.mode-card.sel{border-color:#8b5cf6;background:#16122b;
box-shadow:0 0 0 1px #8b5cf6 inset}
.mode-card b{display:block;font-size:14px;margin-bottom:3px;color:#e6e6e6}
.mode-card span{color:#8b95a5;font-size:12.5px;line-height:1.4;display:block}
.mode-card .check{position:absolute;top:11px;right:13px;width:20px;height:20px;
border-radius:50%;background:#8b5cf6;color:#fff;display:none;
align-items:center;justify-content:center;font-size:12px;font-weight:700}
.mode-card.sel .check{display:flex}
.toggles{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px}
.tog{display:flex;align-items:center;gap:11px;background:#0f141b;
border:1px solid #222a35;border-radius:10px;padding:11px 14px;cursor:pointer}
.tog:hover{border-color:#3a4660}
.tog input{display:none}
.tog .sw{flex:0 0 38px;height:22px;border-radius:999px;background:#2b333f;
position:relative;transition:background .15s}
.tog .sw::after{content:"";position:absolute;top:2px;left:2px;width:18px;
height:18px;border-radius:50%;background:#8b95a5;transition:transform .15s,background .15s}
.tog input:checked+.sw{background:#1f6feb}
.tog input:checked+.sw::after{transform:translateX(16px);background:#fff}
.tog .tl{font-size:13.5px;font-weight:600}.tog .tl small{display:block;
color:#7c8aa0;font-weight:400;font-size:11.5px}
@media (max-width:640px){.modes,.toggles{grid-template-columns:1fr}}
/* ---- Channel / Look DNA picker — primary editorial-identity card ---- */
.channels{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:6px 0 2px}
.ch-card{position:relative;background:#0f141b;border:1px solid #222a35;
border-radius:12px;padding:14px 16px;cursor:pointer;
transition:border-color .15s,background .15s}
.ch-card:hover{border-color:#3a4660;background:#131a23}
.ch-card.sel{border-color:#22c55e;background:#0c1a13;
box-shadow:0 0 0 1px #22c55e inset}
.ch-card b{display:block;font-size:14px;margin-bottom:4px;color:#e6e6e6}
.ch-card span{color:#7c8aa0;font-size:12px;line-height:1.45;display:block}
.ch-card .check{position:absolute;top:13px;right:14px;width:22px;height:22px;
border-radius:50%;background:#22c55e;color:#fff;display:none;
align-items:center;justify-content:center;font-size:13px;font-weight:700}
.ch-card.sel .check{display:flex}
@media (max-width:640px){.channels{grid-template-columns:1fr}}
/* ---- API status pills (top of form) ---- */
.api-row{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.api-pill{display:flex;align-items:center;gap:7px;background:#11161d;
border:1px solid #1f2630;border-radius:999px;padding:5px 12px;font-size:12px;
color:#cdd5df}
.api-pill .dot{width:8px;height:8px;border-radius:50%}
.api-pill.ok .dot{background:#22c55e;box-shadow:0 0 5px rgba(34,197,94,.6)}
.api-pill.miss .dot{background:#6b7686}
.api-pill.miss{color:#7c8aa0;border-color:#1a212c}
/* ---- Footage source toggle (kill-switch with status sub-line) ---- */
.footage-row{display:flex;flex-direction:column;gap:12px;margin:4px 0 2px}
.footage-card{background:#0f141b;border:1px solid #222a35;border-radius:11px;
padding:13px 16px}
.footage-card .ft-head{display:flex;align-items:center;justify-content:space-between;
gap:14px}
.footage-card .ft-name{font-weight:600;font-size:14px;color:#e6e6e6}
.footage-card .ft-sub{color:#7c8aa0;font-size:12px;margin-top:3px;line-height:1.5}
.footage-card.locked .ft-name{color:#6b7686}
.footage-card .ft-pill{font-size:11px;padding:2px 8px;border-radius:999px;
background:#1c222b;color:#7c8aa0}
.footage-card .ft-pill.primary{background:#13354d;color:#7fc1ff}
.footage-card .ft-pill.ok{background:#0d2818;color:#5fc77f}
.footage-card .ft-pill.warn{background:#3a2a10;color:#f0b656}
.note{background:#0f1b2a;border-left:3px solid #3b82f6;color:#9cb7d8;
padding:9px 13px;border-radius:6px;font-size:12.5px;margin:10px 0 0;
line-height:1.5}
</style></head><body>{{ body|safe }}</body></html>"""

_FORM = """
<style>
.cwrap{max-width:780px;margin:0 auto}
.chead{display:flex;align-items:center;justify-content:space-between;gap:16px;margin:4px 0 26px}
.cbrand{display:flex;align-items:center;gap:13px}
.clogo{width:42px;height:42px;border-radius:11px;background:linear-gradient(135deg,#3b82f6,#8b5cf6);
  display:flex;align-items:center;justify-content:center;font:800 22px system-ui;color:#fff;
  box-shadow:0 6px 18px rgba(59,130,246,.34)}
.cbrand b{display:block;font-size:18px;letter-spacing:.2px}
.cbrand .c2{display:block;color:#8b95a5;font-size:12.5px;margin-top:1px}
.chead-r{display:flex;align-items:center;gap:14px}
.cready{display:inline-flex;align-items:center;gap:7px;color:#9aa6b5;font-size:12.5px;white-space:nowrap}
.cready .cdot{width:8px;height:8px;border-radius:50%;background:#22c55e;box-shadow:0 0 0 3px rgba(34,197,94,.16)}
.cready.warn .cdot{background:#f59e0b;box-shadow:0 0 0 3px rgba(245,158,11,.16)}
.cbtn2{display:inline-flex;align-items:center;gap:7px;background:#161c25;border:1px solid #2a3340;
  color:#cdd5df;padding:9px 14px;border-radius:9px;text-decoration:none;font-weight:600;font-size:13.5px}
.cbtn2:hover{border-color:#3a4660;background:#1a212b;color:#fff}
.cerr{background:#3a1620;border:1px solid #7d2435;padding:12px 15px;border-radius:10px;margin:0 0 18px;color:#ffb3c0;font-size:14px}

.cstep{background:#11161d;border:1px solid #222a35;border-radius:16px;padding:22px 22px 24px;margin:0 0 18px}
.cstep-h{display:flex;align-items:flex-start;gap:13px;margin:0 0 18px}
.cnum{flex:0 0 30px;width:30px;height:30px;border-radius:50%;background:#1c2533;border:1px solid #2f3b4d;
  color:#7fa9ff;font:700 14px system-ui;display:flex;align-items:center;justify-content:center;margin-top:1px}
.cstep-h b{display:block;font-size:16.5px;letter-spacing:.2px}
.cstep-h>div>span{display:block;color:#8b95a5;font-size:13px;margin-top:3px;line-height:1.5}

.clab{display:block;font-weight:600;font-size:13.5px;margin:16px 0 7px;color:#dfe5ec}
.clab:first-of-type{margin-top:0}
.clab small{font-weight:500;color:#8b95a5}
.cin{width:100%;padding:11px 12px;border:1px solid #2b313b;border-radius:10px;background:#0f141b;
  color:#e6e6e6;font:inherit;box-sizing:border-box}
.cin:focus{outline:0;border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,.18)}
.cta{width:100%;min-height:150px;padding:12px;border:1px solid #2b313b;border-radius:10px;background:#0f141b;
  color:#e6e6e6;font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;box-sizing:border-box;resize:vertical}
.cta:focus{outline:0;border-color:#3b82f6;box-shadow:0 0 0 3px rgba(59,130,246,.18)}

.cseg{display:flex;gap:8px;background:#0d1219;border:1px solid #232c38;border-radius:11px;padding:5px;margin:2px 0 16px}
.cseg .csegb{flex:1;margin:0}
.cseg input{position:absolute;opacity:0;pointer-events:none}
.cseg .csegb>span{display:block;text-align:center;padding:9px 10px;border-radius:8px;cursor:pointer;
  font-weight:600;font-size:13.5px;color:#9aa6b5;transition:.12s}
.cseg input:checked+span{background:#1f2a3b;color:#fff;box-shadow:0 1px 0 rgba(255,255,255,.04)}
.cseg .csegb:hover>span{color:#cdd5df}

.cdrop{display:flex;align-items:center;gap:11px;border:1.5px dashed #2f3a49;border-radius:11px;
  padding:14px 16px;background:#0d1219;cursor:pointer;color:#9aa6b5;font-size:13.5px;transition:.12s}
.cdrop:hover{border-color:#3b82f6;background:#0f1622;color:#cdd5df}
.cdrop.has{border-style:solid;border-color:#2f6b3f;background:#0e1a12;color:#9be3ad}
.cdrop input{position:absolute;width:1px;height:1px;opacity:0;overflow:hidden}
.cdrop::before{content:"🎙";font-size:18px;opacity:.85}
.clink{margin:13px 0 0;padding:0;background:none;border:0;color:#6ea8fe;font:600 13px system-ui;cursor:pointer}
.clink:hover{text-decoration:underline}
.chint{color:#7c8696;font-size:12.5px;margin:10px 0 0;line-height:1.5}

.ccards{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.ccard{position:relative;background:#0f141b;border:1.5px solid #232c38;border-radius:13px;padding:15px 16px;
  cursor:pointer;transition:.12s}
.ccard:hover{border-color:#3a4660;background:#121925}
.ccard.sel{border-color:#3b82f6;background:#0f1a2b;box-shadow:0 0 0 1px #3b82f6 inset}
.ccard b{display:block;font-size:14.5px;margin:0 0 5px;color:#eef2f7}
.ccard span{display:block;color:#8b95a5;font-size:12.5px;line-height:1.5}
.ccard .crec{display:inline-block;vertical-align:middle;margin-left:6px;background:#13351f;color:#5fd07f;
  border:1px solid #1f6b39;font:600 10px system-ui;padding:2px 7px;border-radius:20px;letter-spacing:.3px}
.ccard.rec{border-color:#2f6b3f}
.ccard .ctick{position:absolute;top:12px;right:13px;width:20px;height:20px;border-radius:50%;background:#3b82f6;
  color:#fff;font-size:12px;display:none;align-items:center;justify-content:center}
.ccard.sel .ctick{display:flex}
.ccards2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.ccard2{position:relative;background:#0f141b;border:1.5px solid #232c38;border-radius:13px;padding:15px 16px;cursor:pointer;transition:.12s}
.ccard2:hover{border-color:#3a4660;background:#121925}
.ccard2.sel{border-color:#3b82f6;background:#0f1a2b;box-shadow:0 0 0 1px #3b82f6 inset}
.ccard2 b{display:block;font-size:14.5px;margin:0 0 5px;color:#eef2f7}
.ccard2 span{display:block;color:#8b95a5;font-size:12.5px;line-height:1.5}
.ccard2 .ctick{position:absolute;top:12px;right:13px;width:20px;height:20px;border-radius:50%;background:#3b82f6;color:#fff;font-size:12px;display:none;align-items:center;justify-content:center}
.ccard2.sel .ctick{display:flex}
.cvonote{margin:13px 0 0;padding:11px 13px;border-radius:10px;background:#0e1a12;border:1px solid #1f4d2e;color:#9be3ad;font-size:13px;display:none}
.cvonote.show{display:block}

.ccreate{display:block;width:100%;margin:6px 0 0;padding:16px;border:0;border-radius:13px;cursor:pointer;
  background:linear-gradient(135deg,#3b82f6,#6366f1);color:#fff;font:700 16px system-ui;letter-spacing:.2px;
  box-shadow:0 10px 26px rgba(59,130,246,.32)}
.ccreate:hover{filter:brightness(1.07)}
.ccreate:disabled{opacity:.6;cursor:default;filter:none}
.ccreatesub{text-align:center;color:#7c8696;font-size:12.5px;margin:11px 0 4px;line-height:1.5}

.cadv{margin:18px 0 8px;background:#0e131a;border:1px solid #1e2630;border-radius:14px;overflow:hidden}
.cadv>summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:11px;padding:16px 18px;
  font-weight:700;font-size:14.5px;color:#cdd5df}
.cadv>summary::-webkit-details-marker{display:none}
.cadv>summary small{font-weight:500;color:#7c8696;font-size:12.5px}
.cadv>summary .cadvic{width:26px;height:26px;border-radius:8px;background:#1a2230;display:flex;align-items:center;justify-content:center;font-size:14px}
.cadv>summary::after{content:"▾";margin-left:auto;color:#7c8696;font-size:13px;transition:.15s}
.cadv[open]>summary::after{transform:rotate(180deg)}
.cadvbody{padding:4px 18px 20px;border-top:1px solid #1e2630}
.cadvsec{padding:18px 0;border-bottom:1px solid #181f29}
.cadvsec:last-child{border-bottom:0;padding-bottom:4px}
.cadvh{font-weight:700;font-size:13.5px;color:#dfe5ec;margin:0 0 13px;letter-spacing:.3px;text-transform:uppercase}
.cadvh small{display:block;text-transform:none;letter-spacing:0;font-weight:500;color:#7c8696;font-size:12px;margin-top:4px}
.cfield{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:0 0 12px;font-size:13.5px;color:#cdd5df;font-weight:600}
.cfield:last-child{margin-bottom:0}
.cfield .cin{max-width:300px}
.ctog{display:flex;align-items:center;gap:13px;margin:0 0 13px;cursor:pointer}
.ctog:last-child{margin-bottom:0}
.ctog input{position:absolute;opacity:0;width:1px;height:1px}
.ctog .csw{flex:0 0 40px;width:40px;height:23px;border-radius:13px;background:#2a3340;position:relative;transition:.15s}
.ctog .csw::after{content:"";position:absolute;top:2px;left:2px;width:19px;height:19px;border-radius:50%;background:#8b95a5;transition:.15s}
.ctog input:checked+.csw{background:#3b82f6}
.ctog input:checked+.csw::after{transform:translateX(17px);background:#fff}
.ctog input:disabled+.csw{opacity:.4}
.ctog .ctl{font-weight:600;font-size:13.5px;color:#dfe5ec}
.ctog .ctl small{display:block;color:#7c8696;font-weight:500;font-size:12px;margin-top:2px}
.cexp{margin:12px 0 0;border-top:1px dashed #232c38;padding-top:12px}
.cexp>summary{cursor:pointer;color:#8b95a5;font-size:12.5px;font-weight:600;list-style:none}
.cexp>summary::-webkit-details-marker{display:none}
.cexpb{display:inline-block;background:#2a210e;color:#e0b252;border:1px solid #5a4516;font:600 10px system-ui;padding:1px 6px;border-radius:5px;margin-left:7px}

#ctip{position:fixed;z-index:1000;max-width:260px;background:#0a0e14;border:1px solid #2f3a49;color:#dfe5ec;
  font:500 12px/1.5 system-ui;padding:8px 11px;border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,.5);
  pointer-events:none;opacity:0;transform:translateY(4px);transition:opacity .12s,transform .12s}
#ctip.show{opacity:1;transform:translateY(0)}
@media(max-width:620px){.ccards,.ccards2{grid-template-columns:1fr}.chead{flex-direction:column;align-items:flex-start}}
</style>

<div class=cwrap>
  <div class=chead>
    <div class=cbrand>
      <div class=clogo>V</div>
      <div><b>Vidlore Studio</b><span class=c2>AI-powered faceless documentaries</span></div>
    </div>
    <div class=chead-r>
      {% set _ready = api_status[0].connected and api_status[1].connected %}
      <span class="cready{{ '' if _ready else ' warn' }}"
        data-tip="{% for s in api_status %}{{ s.label }}: {{ 'connected' if s.connected else 'not set' }}{% if not loop.last %} · {% endif %}{% endfor %}">
        <span class=cdot></span>{{ 'All systems ready' if _ready else 'Ready (some keys optional)' }}</span>
      <a class=cbtn2 href="{{ url_for('index') }}" data-tip="Browse your rendered documentaries.">▦ My Videos</a>
    </div>
  </div>

  {% if error %}<div class=cerr>{{ error }}</div>{% endif %}

  {% set _gen = (not f.get('script')) and f.get('prompt') %}
  <form method=post action="{{ url_for('gen_script') }}" id=genform enctype="multipart/form-data" onsubmit="return goGen()">
    <input type=hidden name=fmt value=documentary>
    <input type=hidden name=style value=auto>

    <!-- STEP 1 · CONTENT -->
    <div class=cstep>
      <div class=cstep-h><span class=cnum>1</span>
        <div><b>Add your content</b><span>Faceless documentary · give it a title, then bring a script or let Vidlore write one.</span></div>
      </div>

      <label class=clab>Documentary title</label>
      <input class=cin name=title required value="{{ f.get('title','') }}"
        placeholder="The Greek City That Vanished Beneath the Sea"
        data-tip="A clear working title — it guides the automatic style, visuals and narration.">

      <label class=clab>Script</label>
      <div class=cseg>
        <label class=csegb><input type=radio name=_src value=mine {{ '' if _gen else 'checked' }} onchange="srcToggle()"><span>I have a script</span></label>
        <label class=csegb><input type=radio name=_src value=gen {{ 'checked' if _gen else '' }} onchange="srcToggle()"><span>Write one for me</span></label>
      </div>

      <div id=srcmine {{ 'hidden' if _gen else '' }}>
        <textarea class=cta name=script placeholder="Paste your documentary script or narration here.&#10;One blank line between scenes.">{{ f.get('script','') }}</textarea>
        <div class=chint>The first line becomes the title scene. You'll review &amp; edit it before rendering.</div>
      </div>

      <div id=srcgen {{ '' if _gen else 'hidden' }}>
        <textarea class=cta name=prompt placeholder="Optional creative direction — angle, tone, audience, must-hit points. Leave empty to let Vidlore decide.">{{ f.get('prompt','') }}</textarea>
        <label class=clab>Target script length
          <small>— how much narration to write (final video length follows the script).</small></label>
        <select class=cin name=duration
          data-tip="Roughly how many minutes of narration to generate. Vidlore sets the real video length from the finished script.">
          {% for d in durations %}<option value="{{ d }}" {{ 'selected' if f.get('duration','6-8')==d else '' }}>{{ d }} minutes</option>{% endfor %}
        </select>
        <div class=chint>Writing a script needs an Anthropic key. No key? Switch to “I have a script”, or upload a voiceover.</div>
      </div>

      <label class=clab>Voiceover <small>(optional — overrides AI narration)</small></label>
      <label class=cdrop id=cdrop>
        <input type=file name=voiceover accept="audio/*,.mp3,.wav,.m4a,.aac" onchange="voFile(this)">
        <span id=cdroptx>Upload your own narration audio, or let Vidlore generate the voice.</span>
      </label>

      <div><button type=button class=clink onclick="ld()" data-tip="Fill every field with a ready-made example topic.">↧ Load a sample topic</button></div>
    </div>

    <!-- STEP 2 · DOCUMENTARY STYLE -->
    <div class=cstep>
      <div class=cstep-h><span class=cnum>2</span>
        <div><b>Documentary style</b><span>Sets pacing, on-screen graphics, captions, music and colour. Auto reads your topic and picks for you.</span></div>
      </div>
      <input type=hidden name=look_preset id=channelval value="{{ f.get('look_preset','auto') }}">
      <div class=ccards id=channelpicker>
        {% for ch in channels %}
          <div class="ccard{{ ' sel' if f.get('look_preset','auto')==ch.name else '' }}{{ ' rec' if ch.name=='auto' else '' }}"
               data-channel="{{ ch.name }}" data-tip="{{ ch.desc }}">
            <b>{% if ch.name=='auto' %}Auto detect{% else %}{{ ch.label }}{% endif %}{% if ch.name=='auto' %}<span class=crec>Recommended</span>{% endif %}</b>
            <span>{% if ch.name=='auto' %}Vidlore analyses your title &amp; script and applies the best documentary look automatically.{% else %}{{ ch.desc.split('.')[0] }}.{% endif %}</span>
            <div class=ctick>&#10003;</div>
          </div>
        {% endfor %}
      </div>
    </div>

    <!-- STEP 3 · VOICE -->
    <div class=cstep id=cvoicestep>
      <div class=cstep-h><span class=cnum>3</span>
        <div><b>Narration voice</b><span>How the documentary is voiced when you don't upload your own.</span></div>
      </div>
      <input type=hidden name=voice_mode id=vmval value="{{ f.get('voice_mode','legacy') }}">
      <div class=ccards2 id=vmpicker>
        <div class="ccard2{{ ' sel' if f.get('voice_mode','legacy')!='premium' else '' }}" data-vm=legacy
             data-tip="Cloud edge-TTS. Fast, reliable, zero setup — the safe default.">
          <b>Basic voice</b><span>Fast &amp; reliable. Works everywhere, no setup.</span><div class=ctick>&#10003;</div>
        </div>
        <div class="ccard2{{ ' sel' if f.get('voice_mode','legacy')=='premium' else '' }}" data-vm=premium
             data-tip="Self-hosted neural narration (Chatterbox / Kokoro). Highest quality; falls back to Basic if a model isn't installed.">
          <b>Premium voice</b><span>Ultra-real neural narration (self-hosted).</span><div class=ctick>&#10003;</div>
        </div>
      </div>
      <div class=cvonote id=cvonote>🎙 Using your uploaded voiceover — the narration voice above is skipped.</div>
    </div>

    <!-- CREATE -->
    <button type=submit class=ccreate id=genbtn>Create documentary →</button>
    <p class=ccreatesub>Vidlore writes/uses your script, sources visuals, builds motion graphics, mixes music &amp; sound, and runs a quality check. You'll review the script first.</p>

    <!-- ADVANCED -->
    <details class=cadv id=cadv {{ 'open' if error else '' }}>
      <summary><span class=cadvic>⚙</span> Advanced settings <small>— optional. Auto already handles all of this.</small></summary>
      <div class=cadvbody>

        <div class=cadvsec>
          <div class=cadvh>Visual sourcing<small>Free stock video and AI still-images always run automatically as needed. These add extra sources.</small></div>
          <label class=ctog>
            <input type=checkbox name=shutterstock value=1 {{ 'checked' if f.get('shutterstock', ss_default)=='1' and api_status[2].connected else '' }} {{ 'disabled' if not api_status[2].connected else '' }}>
            <span class=csw></span>
            <span class=ctl>Premium stock fallback<small>Shutterstock — used only when free stock misses{{ ' · no token set' if not api_status[2].connected else '' }}</small></span>
          </label>
          <label class=cfield>Archive &amp; web images
            <select class=cin name=wi_mix data-tip="Topic-relevant photos from the web &amp; archives, animated into b-roll. Great for historical / niche subjects.">
              <option value=off {{ 'selected' if f.get('wi_mix')=='off' else '' }}>Off</option>
              <option value=light {{ 'selected' if f.get('wi_mix')=='light' else '' }}>Light</option>
              <option value=balanced {{ 'selected' if f.get('wi_mix','balanced')=='balanced' else '' }}>Balanced (recommended)</option>
              <option value=heavy {{ 'selected' if f.get('wi_mix')=='heavy' else '' }}>Heavy</option>
            </select>
          </label>
          <details class=cexp>
            <summary>Experimental sources</summary>
            <label class=cfield style="margin-top:12px">Web video crawl <span class=cexpb>EXPERIMENTAL</span>
              <select class=cin name=wf_mix data-tip="Attempts to pull video directly from web pages. Rarely yields usable clips; off by default.">
                <option value=off {{ 'selected' if f.get('wf_mix','off')=='off' else '' }}>Off (default)</option>
                <option value=light {{ 'selected' if f.get('wf_mix')=='light' else '' }}>Light</option>
                <option value=balanced {{ 'selected' if f.get('wf_mix')=='balanced' else '' }}>Balanced</option>
                <option value=heavy {{ 'selected' if f.get('wf_mix')=='heavy' else '' }}>Heavy</option>
              </select>
            </label>
          </details>
        </div>

        <div class=cadvsec>
          <div class=cadvh>Editing</div>
          <label class=ctog><input type=checkbox name=music value=1 {{ 'checked' if f.get('music','1')!='0' else '' }}><span class=csw></span><span class=ctl>Background music<small>arc-aware score · channel-matched</small></span></label>
          <label class=ctog><input type=checkbox name=transitions value=1 {{ 'checked' if f.get('transitions','1')!='0' else '' }}><span class=csw></span><span class=ctl>Transitions<small>motivated dissolves between scenes</small></span></label>
          <label class=ctog><input type=checkbox name=overlays value=1 {{ 'checked' if f.get('overlays','1')!='0' else '' }}><span class=csw></span><span class=ctl>Graphics &amp; overlays<small>title cards · callouts · chapters</small></span></label>
          <label class=ctog><input type=checkbox name=sfx value=1 {{ 'checked' if f.get('sfx','1')=='1' else '' }}><span class=csw></span><span class=ctl>Sound design<small>sparse booms / risers</small></span></label>
          <label class=ctog><input type=checkbox name=captions value=1 {{ 'checked' if f.get('captions','0')=='1' else '' }}><span class=csw></span><span class=ctl>Burn captions into the video<small>off by default — toggle on here, or anytime in the editor</small></span></label>
        </div>

        <div class=cadvsec>
          <div class=cadvh>Voice details<small>Used by the narration voice above (ignored when you upload your own).</small></div>
          <label class=cfield>Premium engine
            <select class=cin name=tts_model data-tip="Neural TTS engine for Premium voice.">
              <option value=chatterbox {{ 'selected' if f.get('tts_model','chatterbox')=='chatterbox' else '' }}>Chatterbox (premium)</option>
              <option value=kokoro {{ 'selected' if f.get('tts_model')=='kokoro' else '' }}>Kokoro (fast fallback)</option>
            </select>
          </label>
          <label class=cfield>Premium narrator
            <select class=cin name=tts_voice data-tip="Voice character for Premium narration.">
              {% for p in premium_presets %}<option value="{{ p.key }}" {{ 'selected' if f.get('tts_voice','deep_male_documentary')==p.key else '' }}>{{ p.label }}</option>{% endfor %}
            </select>
          </label>
          <label class=cfield>Basic voice override
            <select class=cin name=voice data-tip="Optional specific edge-TTS voice for Basic narration.">
              <option value="">Auto (documentary default)</option>
              {% for v in voices %}<option value="{{ v.id }}" {{ 'selected' if f.get('voice')==v.id else '' }}>{{ v.label }}</option>{% endfor %}
            </select>
          </label>
        </div>

        <div class=cadvsec>
          <div class=cadvh>Appearance<small>Only applies when Documentary style is “Auto detect” and Auto picks no channel — otherwise the channel controls the look.</small></div>
          <label class=clab>Fallback theme</label>
          <input type=hidden name=theme id=themeval value="{{ f.get('theme','standard') }}">
          <div class=themes id=themepicker>
            {% for tm in themes_meta %}
              {% set g0=tm.grad[0] %}{% set g1=tm.grad[1] %}{% set pc=tm.primary %}
              <div class="theme-card{{ ' sel' if f.get('theme','standard')==tm.name else '' }}"
                   data-theme="{{ tm.name }}"
                   style="--g0:rgb({{g0[0]}},{{g0[1]}},{{g0[2]}});--g1:rgb({{g1[0]}},{{g1[1]}},{{g1[2]}});--pc:rgb({{pc[0]}},{{pc[1]}},{{pc[2]}})">
                <div class=swatch style="background:linear-gradient(135deg,var(--g0),var(--g1))"><div class=dot style="background:var(--pc)"></div></div>
                <div class=meta><b>{{ tm.title }}</b><span>{{ tm.description }}</span></div>
                <div class=check>&#10003;</div>
              </div>
            {% endfor %}
          </div>
          <label class=clab style="margin-top:16px">Card background <small>(fallback slide only — real footage covers the screen)</small></label>
          <input type=hidden name=background id=bgval value="{{ f.get('background','auto') }}">
          <div class=bgs id=bgpicker>
            {% for bg in bg_meta %}
              <div class="bg-card{{ ' sel' if f.get('background','auto')==bg.name else '' }}" data-bg="{{ bg.name }}" title="{{ bg.title }}">
                {% if bg.name == 'auto' %}<div class="thumb auto">Auto</div>{% else %}<div class=thumb style="background-image:url('{{ url_for('bg_preview', name=bg.name) }}')"></div>{% endif %}
                <div class=lbl>{{ bg.title }}</div><div class=check>&#10003;</div>
              </div>
            {% endfor %}
          </div>
        </div>

      </div>
    </details>
  </form>

  <div id=wait style="display:none;position:fixed;inset:0;background:rgba(8,11,16,.94);z-index:999;
    align-items:center;justify-content:center;text-align:center">
    <div>
      <div style="width:54px;height:54px;margin:0 auto 22px;border:5px solid #222a35;border-top-color:#3b82f6;border-radius:50%;animation:spin 1s linear infinite"></div>
      <h2 style=margin:0>Reading your script…</h2>
      <p class=sub>This takes about <b>30–60 seconds</b>. Please don't close this tab or press back.</p>
    </div>
  </div>
</div>
<style>@keyframes spin{to{transform:rotate(360deg)}}</style>

<script>
function goGen(){
  var b=document.getElementById('genbtn');
  b.disabled=true; b.textContent='Working…';
  document.getElementById('wait').style.display='flex';
  return true;
}
async function ld(){
  var r=await fetch("{{ url_for('sample') }}"); var d=await r.json();
  document.querySelector('[name=title]').value=d.title;
  document.querySelector('[name=prompt]').value=d.prompt;
  document.querySelector('[name=script]').value=d.script;
}
function srcToggle(){
  var gen=(document.querySelector('input[name=_src]:checked')||{}).value==='gen';
  document.getElementById('srcmine').hidden=gen;
  document.getElementById('srcgen').hidden=!gen;
}
function voFile(inp){
  var has=inp.files && inp.files.length>0;
  var drop=document.getElementById('cdrop');
  drop.classList.toggle('has',has);
  document.getElementById('cdroptx').textContent=has?('🎙 '+inp.files[0].name+' — your voiceover will be used'):'Upload your own narration audio, or let Vidlore generate the voice.';
  document.getElementById('cvonote').classList.toggle('show',has);
}
// card pickers: set the hidden input + .sel highlight
function cpick(picker, hid, attr){
  var p=document.getElementById(picker); if(!p) return;
  p.addEventListener('click', function(e){
    var c=e.target.closest('[data-'+attr+']'); if(!c) return;
    p.querySelectorAll('.sel').forEach(function(x){x.classList.remove('sel')});
    c.classList.add('sel');
    document.getElementById(hid).value=c.getAttribute('data-'+attr);
  });
}
cpick('channelpicker','channelval','channel');
cpick('vmpicker','vmval','vm');
cpick('themepicker','themeval','theme');
cpick('bgpicker','bgval','bg');
// tooltip engine
(function(){
  var tip=document.createElement('div'); tip.id='ctip'; document.body.appendChild(tip);
  var cur=null;
  function show(el){
    var t=el.getAttribute('data-tip'); if(!t) return;
    tip.textContent=t; tip.classList.add('show'); cur=el;
    var r=el.getBoundingClientRect(); tip.style.left='-9999px'; tip.style.top='0';
    var tw=tip.offsetWidth, th=tip.offsetHeight;
    var x=Math.min(Math.max(8,r.left+r.width/2-tw/2), innerWidth-tw-8);
    var y=r.top-th-9; if(y<8) y=r.bottom+9;
    tip.style.left=x+'px'; tip.style.top=y+'px';
  }
  function hide(){ tip.classList.remove('show'); cur=null; }
  document.addEventListener('mouseover',function(e){var el=e.target.closest('[data-tip]'); if(el&&el!==cur) show(el);});
  document.addEventListener('mouseout',function(e){var el=e.target.closest('[data-tip]'); if(el&&!el.contains(e.relatedTarget)) hide();});
  document.addEventListener('focusin',function(e){var el=e.target.closest('[data-tip]'); if(el) show(el);});
  document.addEventListener('focusout',hide);
  document.addEventListener('keydown',function(e){if(e.key==='Escape')hide();});
  addEventListener('scroll',hide,true);
})();
</script>"""

_REVIEW = """
<h1>Review &amp; edit the script</h1>
<p class=sub>{{n}} scenes · ~{{words}} words. Edit freely — line 1 is the
title, one blank line between scenes (add/remove blank lines to split or
merge). This is the step Vidlore doesn't give you.</p>
<form method=post action="{{ url_for('do_render') }}" class=card>
  <input type=hidden name=title value="{{title}}">
  <input type=hidden name=theme value="{{theme}}">
  <input type=hidden name=duration value="{{duration}}">
  <input type=hidden name=captions value="{{captions}}">
  <input type=hidden name=style value="{{style}}">
  <input type=hidden name=fmt value="{{fmt}}">
  <input type=hidden name=voice value="{{voice}}">
  <input type=hidden name=voice_mode value="{{voice_mode}}">
  <input type=hidden name=tts_model value="{{tts_model}}">
  <input type=hidden name=tts_voice value="{{tts_voice}}">
  <input type=hidden name=background value="{{background}}">
  <input type=hidden name=music value="{{music}}">
  <input type=hidden name=transitions value="{{transitions}}">
  <input type=hidden name=overlays value="{{overlays}}">
  <input type=hidden name=sfx value="{{sfx}}">
  <input type=hidden name=look_preset value="{{look_preset}}">
  <input type=hidden name=shutterstock value="{{shutterstock}}">
  <input type=hidden name=wf_mix value="{{wf_mix}}">
  <input type=hidden name=wi_mix value="{{wi_mix}}">
  <p class=sub style="margin:0 0 12px">
    {% if look_preset and look_preset != 'auto' %}
      📺 Channel: <b style="color:#22c55e">{{look_preset.replace('_',' ').title()}}</b> ·
    {% endif %}
    🎬 Style: <b style="color:#a78bfa">{{style_label}}</b>
     · theme <b>{{theme}}</b> · {{duration}} min
  </p>
  {% if voiceover_name %}<p class=sub style="color:#5fc77f">🎙 Using your
  voiceover: <b>{{voiceover_name}}</b> (AI voice off — split across
  scenes)</p>{% endif %}
  <textarea name=script_text>{{script_text}}</textarea>
  <button type=submit>Render video →</button>
  <a href="{{ url_for('new') }}" style="margin-left:14px">← start over</a>
  <a href="{{ url_for('index') }}" style="margin-left:14px">My Videos</a>
</form>"""

_JOB = """
<style>
.jwrap{max-width:920px;margin:0 auto}
.jhead{display:flex;justify-content:space-between;align-items:center;margin:0 0 22px;gap:12px;flex-wrap:wrap}
.jbrand{display:flex;align-items:center;gap:11px}
.jlogo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,#3b82f6,#2563eb);
  display:flex;align-items:center;justify-content:center;color:#fff;font-size:15px;box-shadow:0 4px 14px rgba(59,130,246,.35)}
.jbrand b{font-size:15px;display:block;line-height:1.2}
.jbrand .j2{color:#8b95a5;font-size:12px}
.jhead-r{display:flex;align-items:center;gap:10px}
.jbadge{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:600;padding:5px 12px;border-radius:999px;
  background:#13354d;color:#7fc1ff;border:1px solid #1d4d6b}
.jbadge.run{background:#13354d;color:#7fc1ff;border:1px solid #1d4d6b}
.jbadge.ok{background:#10301f;color:#69d49a;border:1px solid #1d4d33}
.jbadge.bad{background:#3a1620;color:#ffb3c0;border:1px solid #7d2435}
.jbadge .jdot{width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 7px currentColor}
.jbadge.run .jdot{animation:jpulse 1.1s ease-in-out infinite}
@keyframes jpulse{0%,100%{opacity:.4}50%{opacity:1}}
.jpanel{background:#11161d;border:1px solid #222a35;border-radius:16px;padding:26px 26px;margin:0 0 16px}
.jhero{text-align:center;padding:30px 26px 28px}
.jhero-ic{width:58px;height:58px;border-radius:16px;margin:0 auto 16px;display:flex;align-items:center;justify-content:center;
  background:#13354d;color:#7fc1ff;font-size:24px}
.jhero-ic.ok{background:#10301f;color:#69d49a;box-shadow:0 8px 26px rgba(80,200,140,.18)}
.jhero-ic.bad{background:#3a1620;color:#ff9bab}
.jspin{width:26px;height:26px;border-radius:50%;border:3px solid rgba(127,193,255,.25);border-top-color:#7fc1ff;animation:jspin .8s linear infinite}
@keyframes jspin{to{transform:rotate(360deg)}}
.jh1{font-size:25px;font-weight:800;margin:0 0 8px;letter-spacing:-.01em}
.jlead{color:#9aa6b6;font-size:15px;margin:0 auto;max-width:560px;line-height:1.5}
.jprog{margin:24px auto 0;max-width:560px}
.jprog-top{display:flex;justify-content:space-between;font-size:13.5px;margin:0 0 9px}
.jstage{color:#cdd5df;font-weight:600}.jpct{color:#7fc1ff;font-weight:700;font-variant-numeric:tabular-nums}
.jbar{height:10px;background:#0c1117;border:1px solid #1c2531;border-radius:999px;overflow:hidden;position:relative}
.jfill{height:100%;width:0;border-radius:999px;background:linear-gradient(90deg,#2563eb,#3b82f6,#60a5fa);transition:width .5s ease;
  background-size:200% 100%;animation:jshimmer 2.2s linear infinite}
@keyframes jshimmer{to{background-position:-200% 0}}
.jhint{color:#6b7686;font-size:12.5px;margin:20px auto 0;max-width:520px;line-height:1.5}
.jqa{display:none;align-items:center;gap:7px;margin:16px auto 0;font-size:13px;font-weight:600;padding:6px 14px;border-radius:999px}
.jqa.ok{background:#10301f;color:#69d49a;border:1px solid #1d4d33}
.jqa.warn{background:#332915;color:#e2b552;border:1px solid #5a4a1f}
.jqa.bad{background:#3a1620;color:#ffb3c0;border:1px solid #7d2435}
.jvlabel{font-size:12px;color:#8b95a5;text-transform:uppercase;letter-spacing:.06em;margin:0 0 10px;font-weight:600}
.jvframe{position:relative;width:100%;aspect-ratio:16/9;background:#070a0e;border:1px solid #222a35;border-radius:14px;overflow:hidden}
.jvframe video{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#070a0e;border-radius:14px;margin:0;border:0}
.jacts{display:flex;flex-direction:column;gap:14px;margin:0 0 16px}
.jbtn-primary{display:flex;align-items:center;gap:14px;text-decoration:none;background:linear-gradient(135deg,#3b82f6,#2563eb);
  color:#fff;border-radius:14px;padding:16px 20px;box-shadow:0 8px 22px rgba(37,99,235,.32);transition:transform .08s,box-shadow .15s}
.jbtn-primary:hover{transform:translateY(-1px);box-shadow:0 12px 28px rgba(37,99,235,.42)}
.jbtn-primary .jbic{font-size:22px;flex:none}
.jbtn-primary b{font-size:15.5px;display:block}.jbtn-primary small{color:rgba(255,255,255,.8);font-size:12.5px}
.jrow{display:flex;gap:10px;flex-wrap:wrap}
.jbtn2{flex:1 1 auto;min-width:140px;display:inline-flex;align-items:center;justify-content:center;gap:8px;text-decoration:none;
  background:#1a212c;color:#dbe3ee;border:1px solid #2a3340;border-radius:11px;padding:12px 14px;font-size:13.5px;font-weight:600;
  cursor:pointer;transition:background .15s,border-color .15s,transform .08s}
.jbtn2:hover{background:#222c39;border-color:#3a4660}.jbtn2:active{transform:translateY(.5px)}
.jbtn2:focus-visible{outline:none;box-shadow:0 0 0 2px #0e1116,0 0 0 4px #3b82f6}
.jbtn2 span{font-size:15px;line-height:1}
.jgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:640px){.jgrid{grid-template-columns:1fr}}
.jph{font-size:13px;font-weight:700;color:#cdd5df;margin:0 0 14px;text-transform:uppercase;letter-spacing:.05em}
.jsum{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#1c2531;border-radius:10px;overflow:hidden}
.jcell{display:flex;justify-content:space-between;align-items:center;gap:10px;background:#11161d;padding:11px 14px}
.jk{color:#8b95a5;font-size:13px}.jv{color:#e6e6e6;font-weight:600;font-size:13.5px;text-align:right}
.jqaline{font-size:14px;color:#cdd5df;line-height:1.5;margin:0 0 6px}.jqaline.ok{color:#9fe0bb}
.jlink{background:none;border:0;color:#7fc1ff;font:600 13px system-ui;cursor:pointer;padding:8px 0 0;margin:6px 0 0}
.jlink:hover{text-decoration:underline}
.jtech{margin:12px 0 0;display:grid;grid-template-columns:1fr;gap:1px;background:#1c2531;border-radius:10px;overflow:hidden}
.jerrp{background:#0c1117;border:1px solid #2a3340;border-radius:10px;padding:12px 14px;margin:14px 0 0;color:#ff9bab;
  font:12px ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;max-height:200px;overflow:auto}
.jtip{position:fixed;z-index:99999;max-width:260px;background:#0b1117;color:#eaf1fb;border:1px solid #314256;border-radius:8px;
  padding:7px 10px;font:12px/1.45 system-ui;box-shadow:0 10px 28px rgba(0,0,0,.55);pointer-events:none;display:none}
</style>
<div class=jwrap>
  <div class=jhead>
    <div class=jbrand><span class=jlogo>▶</span><div><b>Vidlore Studio</b><span class=j2>AI documentary render</span></div></div>
    <div class=jhead-r>
      <span class="jbadge run" id=jbadge><span class=jdot></span>Rendering</span>
      <a class=jbtn2 style="flex:none;min-width:0" href="{{ url_for('index') }}" data-tip="Back to your video library.">▦ My Videos</a>
    </div>
  </div>

  <div class="jpanel jhero" id=jhero>
    <div id=jrender>
      <div class=jhero-ic><span class=jspin></span></div>
      <h1 class=jh1>Creating your documentary</h1>
      <p class=jlead>Vidlore is assembling visuals, motion graphics, captions, music, and the final quality check.</p>
      <div class=jprog>
        <div class=jprog-top><span class=jstage id=jstage>Preparing scenes</span><span class=jpct id=jpct>0%</span></div>
        <div class=jbar><div class=jfill id=jfill></div></div>
      </div>
      <p class=jhint>You can leave this page and come back — the render keeps going. The Review Editor opens automatically when it is ready.</p>
    </div>
    <div id=jdone style=display:none>
      <div class="jhero-ic ok"><span style=font-size:26px>✓</span></div>
      <h1 class=jh1>Your documentary is ready</h1>
      <p class=jlead>Review your video, make scene-level changes, or download the final MP4.</p>
      <span class=jqa id=jqa data-tip="The automatic quality checks performed on the final video."></span>
    </div>
    <div id=jerr style=display:none>
      <div class="jhero-ic bad"><span style=font-size:26px>!</span></div>
      <h1 class=jh1>The video could not be completed</h1>
      <p class=jlead>Your project is still safe. Try rendering again or return to the dashboard.</p>
      <div class=jrow style="justify-content:center;margin-top:20px">
        <form method=post action="/job/{{job_id}}/retry" style=display:inline>
          <button class=jbtn2 style="flex:none;background:#3b82f6;color:#fff;border-color:#3b82f6" type=submit data-tip="Render this project again — cached work is reused.">↻ Try again</button></form>
        <a class=jbtn2 style=flex:none href="{{ url_for('index') }}" data-tip="Go back to your videos.">Return to dashboard</a>
        <button class=jlink id=jerrtog type=button data-tip="Show the raw technical error.">View technical details</button>
      </div>
      <pre class=jerrp id=jerrp style=display:none></pre>
    </div>
  </div>

  <div class=jpanel id=jvideo style=display:none>
    <div class=jvlabel>Final rendered preview</div>
    <div class=jvframe><video id=vid controls playsinline></video></div>
  </div>

  <div class=jacts id=jacts style=display:none>
    <a class=jbtn-primary id=edlink href="#" data-tip="Open the scene editor to make small changes before exporting again.">
      <span class=jbic>🎬</span><span><b>Open Review Editor</b><small>Fine-tune scenes, replace visuals, adjust captions, and re-render.</small></span></a>
    <div class=jrow>
      <a class=jbtn2 id=dl download data-tip="Download the latest rendered MP4 file."><span>⬇</span> Download MP4</a>
      <a class=jbtn2 href="{{ url_for('new') }}" data-tip="Start a brand-new documentary."><span>＋</span> Create another</a>
      <a class=jbtn2 href="{{ url_for('index') }}" data-tip="See all your rendered videos."><span>▦</span> My Videos</a>
    </div>
  </div>

  <div class=jgrid id=jpanels style=display:none>
    <div class=jpanel style=margin:0>
      <div class=jph>Project details</div>
      <div class=jsum id=jsummary></div>
    </div>
    <div class=jpanel id=jqapanel style="margin:0;display:none">
      <div class=jph>Quality check</div>
      <div id=jqabody></div>
      <button class=jlink id=jtechtog type=button data-tip="See the detailed quality metrics.">View technical details</button>
      <div class=jtech id=jtech style=display:none></div>
    </div>
  </div>
</div>
<script>
const id="{{job_id}}";var _polling=true;
function _e(s){return String(s==null?'':s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
function stageLabel(msg){var m=(msg||'').toLowerCase();
  if(/queue|prepar/.test(m))return 'Preparing scenes';
  if(/narrat|voice|tts/.test(m))return 'Recording narration';
  if(/ai|still|imagen|generat/.test(m))return 'Generating AI stills';
  if(/footage|visual|fetch|clip|search/.test(m))return 'Selecting visuals';
  if(/motion|graphic|card|map|chart/.test(m))return 'Building motion graphics';
  if(/music|sfx|sound|audio|mix/.test(m))return 'Mixing music & sound';
  if(/caption|subtitle/.test(m))return 'Adding captions';
  if(/assembl|cross|render|encod|thumb/.test(m))return 'Rendering final video';
  if(/final|check|qa|output/.test(m))return 'Checking final output';
  if(/done|ready|complet/.test(m))return 'Video ready';
  return msg||'Working…';}
function setBadge(cls,txt){var b=document.getElementById('jbadge');b.className='jbadge '+cls;
  b.innerHTML=(cls==='run'?'<span class=jdot></span>':'')+txt;}
async function showDone(s){
  document.getElementById('jrender').style.display='none';
  document.getElementById('jdone').style.display='block';
  document.getElementById('jvideo').style.display='block';
  document.getElementById('jacts').style.display='flex';
  document.getElementById('jpanels').style.display='grid';
  setBadge('ok','✓ Video ready');
  var bust='?v='+Date.now();
  var v=document.getElementById('vid');v.src='/job/'+id+'/file/video'+bust;
  document.getElementById('dl').href='/job/'+id+'/file/video'+bust;
  if(s.slug)document.getElementById('edlink').href='/e/'+s.slug;
  try{var r=await fetch('/job/'+id+'/summary');var d=await r.json();if(d&&d.ok)renderSummary(d);}catch(e){}}
function renderSummary(s){
  var rows=[];function row(k,v){if(v!=null&&v!=='')rows.push('<div class=jcell><span class=jk>'+_e(k)+'</span><span class=jv>'+_e(v)+'</span></div>');}
  row('Duration',s.duration);row('Scenes',s.scenes);row('Resolution',s.resolution);
  row('Frame rate',s.fps?(s.fps+' fps'):null);row('Captions',s.captions?'On':'Off');
  if(s.look)row('Look',s.look);if(s.size_mb)row('File size',s.size_mb+' MB');if(s.completed)row('Completed',s.completed);
  document.getElementById('jsummary').innerHTML=rows.join('')||'<div class=jcell><span class=jk>No details available</span><span class=jv></span></div>';
  if(s.qa_verdict){var pass=s.qa_verdict==='PASS';
    var q=document.getElementById('jqa');q.className='jqa '+(pass?'ok':(s.qa_verdict==='FAIL'?'bad':'warn'));
    q.textContent=pass?'✓ Quality checked':'Review recommended';q.style.display='inline-flex';
    if(pass)setBadge('ok','✓ Quality checked');
    document.getElementById('jqapanel').style.display='block';
    document.getElementById('jqabody').innerHTML='<div class="jqaline'+(pass?' ok':'')+'">'+(pass?'✓ ':'')+_e(s.qa_summary||(pass?'No issues found in the final video.':'A few checks are worth a review.'))+'</div>';
    var t=[];if(s.black_frames!=null)t.push(['Black frames',s.black_frames]);
    if(s.lufs!=null)t.push(['Audio loudness',s.lufs+' LUFS']);if(s.resolution)t.push(['Resolution',s.resolution]);
    if(s.size_mb)t.push(['Output size',s.size_mb+' MB']);if(s.cuts!=null)t.push(['Cuts',s.cuts]);t.push(['Render job',id]);
    document.getElementById('jtech').innerHTML=t.map(function(x){return '<div class=jcell><span class=jk>'+_e(x[0])+'</span><span class=jv>'+_e(x[1])+'</span></div>';}).join('');}}
function showError(err){document.getElementById('jrender').style.display='none';
  document.getElementById('jerr').style.display='block';setBadge('bad','Failed');
  document.getElementById('jerrp').textContent=err||'Unknown error.';}
async function poll(){if(!_polling)return;
  try{var r=await fetch('/job/'+id+'/status');var s=await r.json();
    document.getElementById('jfill').style.width=(s.pct||0)+'%';
    document.getElementById('jpct').textContent=(s.pct||0)+'%';
    document.getElementById('jstage').textContent=stageLabel(s.msg);
    if(s.status=='done'){_polling=false;showDone(s);return;}
    if(s.status=='error'){_polling=false;showError(s.error);return;}
  }catch(e){}
  setTimeout(poll,1500);}
document.getElementById('jtechtog').onclick=function(){var t=document.getElementById('jtech');
  var o=t.style.display==='none';t.style.display=o?'grid':'none';this.textContent=o?'Hide technical details':'View technical details';};
document.getElementById('jerrtog').onclick=function(){var p=document.getElementById('jerrp');
  var o=p.style.display==='none';p.style.display=o?'block':'none';this.textContent=o?'Hide technical details':'View technical details';};
// compact tooltip engine (data-tip / title; 300ms delay, viewport clamp, focus, Esc)
(function(){var el,t,cur;function txt(e){return (e.getAttribute('data-tip')||e.getAttribute('title')||'').trim();}
function find(n){for(;n&&n!==document;n=n.parentElement)if(n.getAttribute&&(n.getAttribute('data-tip')||n.getAttribute('title')))return n;return null;}
function show(n){var x=txt(n);if(!x)return;if(n.hasAttribute('title')){n.setAttribute('data-t',n.getAttribute('title'));n.removeAttribute('title');}
  if(!el){el=document.createElement('div');el.className='jtip';document.body.appendChild(el);}el.textContent=x;el.style.display='block';cur=n;
  var r=n.getBoundingClientRect(),w=el.offsetWidth,h=el.offsetHeight,p=8,vw=innerWidth,vh=innerHeight;
  var L=r.left+r.width/2-w/2,T=r.top-h-8;if(T<p)T=r.bottom+8;L=Math.max(p,Math.min(L,vw-w-p));T=Math.max(p,Math.min(T,vh-h-p));
  el.style.left=Math.round(L)+'px';el.style.top=Math.round(T)+'px';}
function hide(){if(cur&&cur.getAttribute('data-t')){cur.setAttribute('title',cur.getAttribute('data-t'));cur.removeAttribute('data-t');}cur=null;if(t){clearTimeout(t);t=null;}if(el)el.style.display='none';}
document.addEventListener('mouseover',function(e){var n=find(e.target);if(!n||n===cur)return;hide();t=setTimeout(function(){show(n);},300);},true);
document.addEventListener('mouseout',function(e){if(find(e.target))hide();},true);
document.addEventListener('focusin',function(e){var n=find(e.target);if(n){hide();show(n);}},true);
document.addEventListener('focusout',hide,true);document.addEventListener('keydown',function(e){if(e.key==='Escape')hide();},true);
addEventListener('scroll',hide,true);})();
poll();
</script>"""

_DASH = """
<div class=nav>
  <div><h1 style=margin:0>My Videos</h1>
  <p class=sub style=margin:2px:0:0>{{ph}} — your faceless video library</p></div>
  <a class=btn href="{{ url_for('new') }}">+ New video</a>
</div>
{% if jobs %}
<h3 style=margin:6px:0:10px>In progress</h3>
<div class=grid style=margin-bottom:26px>
  {% for jid,j in jobs %}
  <div class=tile><div class=m>
    <b>{{j.title}}</b>
    <span class="pill {{'err' if j.status=='error' else 'run'}}">
      {{ 'failed' if j.status=='error' else j.pct ~ '%' }}</span><br><br>
    <a href="{{ url_for('job_page', job_id=jid) }}">open progress →</a>
  </div></div>
  {% endfor %}
</div>
{% endif %}
{% if incomplete %}
<h3 style=margin:6px:0:10px>Incomplete / failed (resumable)</h3>
<p class=sub style=margin:0:0:12px>Server restart ho gaya tha ya render crash kar gaya — ye
projects disk pe save hain. Resume karein, sirf failed phase re-run hogi (cache reuse).</p>
<div class=grid style=margin-bottom:26px>
  {% for v in incomplete %}
  <div class=tile><div class=m>
    <b>{{v.title}}</b>
    <span class=pill style="background:#5a2a1f;color:#ffb19a">failed</span>
    <br><small>{{v.when}}</small><br><br>
    <form method=post action="{{ url_for('resume_render', slug=v.slug) }}"
          style=display:inline>
      <button type=submit class=btn style=margin:0>↻ Resume render</button>
    </form>
    &nbsp;
    <form method=post action="{{ url_for('delete_video', slug=v.slug) }}"
          style=display:inline onsubmit="return confirm('Delete this project?')">
      <button type=submit class=ghost style=margin:0>🗑</button>
    </form>
  </div></div>
  {% endfor %}
</div>
{% endif %}
{% if vids %}
<div class=grid>
  {% for v in vids %}
  <a class=tile href="{{ url_for('video_page', slug=v.slug) }}"
     style=text-decoration:none;color:inherit>
    <div class=tileph>▶</div>
    <div class=m><b>{{v.title}}</b><small>{{v.when}}</small></div>
  </a>
  {% endfor %}
</div>
{% elif not jobs %}
<div class=empty>No videos yet.<br><br>
  <a class=btn href="{{ url_for('new') }}">+ Create your first video</a></div>
{% endif %}"""

_VIDEO = """
<div class=nav>
  <div><h1 style=margin:0>{{title}}</h1></div>
  <span>
    {% if editor_ok %}<a href="{{ url_for('editor_page', slug=slug) }}" class=btn
       style="background:#2f6df0;color:#fff">🎬 Open Review Editor</a>{% endif %}
    <a href="{{ url_for('index') }}" style=margin-left:12px>← My Videos</a>
  </span>
</div>
<div class=card>
  <video controls src="{{ url_for('video_file', slug=slug, kind='video') }}"></video>
  {% if editor_ok %}
  <p style=margin-top:16px>
    <a href="{{ url_for('editor_page', slug=slug) }}" class=btn
       style="background:#2f6df0;color:#fff;font-size:15px;padding:10px 18px">🎬 Open Review Editor</a>
  </p>
  {% else %}
  <p style="margin-top:16px;color:#f0a070" class=sub>🎬 Review Editor unavailable — {{ editor_reason }}</p>
  {% endif %}
  <p style=margin-top:16px>
    <a href="{{ url_for('video_file', slug=slug, kind='video') }}" download>⬇ Download MP4</a>
    {% if has_srt %} · <a href="{{ url_for('video_file', slug=slug, kind='srt') }}" target=_blank>captions (.srt)</a>{% endif %}
     · <a href="{{ url_for('new') }}">+ new video</a>
  </p>
</div>
<details style=margin-top:22px>
  <summary style="cursor:pointer;font-weight:600">✎ Edit script &amp; re-render</summary>
  <form method=post action="{{ url_for('edit_script', slug=slug) }}"
        class=card style=margin-top:10px>
    <p class=sub style=margin:0:0:10px>Tweak wording or scene breaks
    (line 1 = title; one blank line between scenes). Only the scenes you
    actually change are re-rendered — the rest is reused from cache.</p>
    <textarea name=script_text>{{ script_text }}</textarea>
    <button type=submit>Save &amp; re-render →</button>
  </form>
</details>
{% if scenes %}
<h3 style=margin:26px:0:8px>Scenes</h3>
<p class=sub style=margin:0:0:14px>Don't like a scene's visual? Regenerate
just that one — every other scene is reused from cache, so it's fast.
(Vidlore can't do this — one change = full re-render.)</p>
<div class=card>
{% for s in scenes %}
  <div style="display:flex;gap:12px;align-items:flex-start;
       padding:10px 0;border-bottom:1px solid #222a35">
    <div style=flex:1>
      <small>scene {{s.idx}}{% if s.v %} · img v{{s.v}}{% endif %}{% if s.vv %} · voice v{{s.vv}}{% endif %}</small><br>
      {{ s.text }}
    </div>
    <div style="display:flex;flex-direction:column;gap:6px">
      <form method=post action="{{ url_for('regen', slug=slug, idx=s.idx) }}">
        <button class=ghost style=margin:0;white-space:nowrap>↻ Visual</button>
      </form>
      <form method=post action="{{ url_for('revoice', slug=slug, idx=s.idx) }}">
        <button class=ghost style=margin:0;white-space:nowrap>🔊 Re-voice</button>
      </form>
    </div>
  </div>
{% endfor %}
</div>
{% endif %}
<form method=post action="{{ url_for('delete_video', slug=slug) }}"
      style=margin-top:24px
      onsubmit="return confirm('Delete this video and all its files? This cannot be undone.')">
  <button class=ghost style="background:#3a1620;color:#ffb3c0">🗑 Delete video</button>
</form>"""

PH = "Vidlore"  # product name placeholder (rename freely)


def _page(body_html: str, **ctx) -> str:
    inner = render_template_string(body_html, ph=PH, **ctx)
    return render_template_string(_PAGE, title=PH, body=inner)


def _safe_dir(slug: str) -> Path:
    """Resolve output/<slug> safely (no path traversal)."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,59}", slug or ""):
        abort(404)
    d = OUT / slug
    if d.resolve().parent != OUT.resolve() or not d.is_dir():
        abort(404)
    return d


def _list_videos() -> list[dict]:
    """Finished videos = output/<slug>/<slug>.mp4 on disk (survives
    restarts; no database needed)."""
    if not OUT.exists():
        return []
    out: list[dict] = []
    for d in OUT.iterdir():
        if not d.is_dir():
            continue
        mp4 = d / f"{d.name}.mp4"
        if not mp4.exists():
            continue
        title = d.name
        sj = d / "script.json"
        if sj.exists():
            try:
                title = json.loads(sj.read_text(encoding="utf-8")).get("title") or title
            except Exception:  # noqa: BLE001
                pass
        out.append(
            dict(
                slug=d.name,
                title=title,
                has_thumb=(d / "thumbnail.jpg").exists(),
                mtime=mp4.stat().st_mtime,
                when=time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(mp4.stat().st_mtime)
                ),
            )
        )
    return sorted(out, key=lambda v: v["mtime"], reverse=True)


def _list_incomplete() -> list[dict]:
    """Incomplete renders = output/<slug>/ with brief.json + script.json
    but NO final <slug>.mp4 (assembly never finished, or render crashed).
    These survive server restarts (the JOBS in-memory dict does not),
    and the dashboard now surfaces them with a Resume button so the
    user can retry without losing work. Bug fix 2026-05-26: previously
    a failed render disappeared from the dashboard after a server
    restart and the user had no way back to it."""
    if not OUT.exists():
        return []
    out: list[dict] = []
    for d in OUT.iterdir():
        if not d.is_dir():
            continue
        mp4 = d / f"{d.name}.mp4"
        if mp4.exists():
            continue                       # complete — _list_videos handles it
        bj = d / "brief.json"
        sj = d / "script.json"
        # Only show as resumable if BOTH a brief and a script exist —
        # otherwise the render never got past the script stage and
        # there's nothing to resume from.
        if not (bj.exists() and sj.exists()):
            continue
        title = d.name
        try:
            title = json.loads(sj.read_text(encoding="utf-8")).get("title") or title
        except Exception:  # noqa: BLE001
            pass
        # Most recent activity = newest file in the dir (cache/work/log)
        try:
            latest = max(
                (p.stat().st_mtime for p in d.rglob("*") if p.is_file()),
                default=d.stat().st_mtime,
            )
        except OSError:
            latest = d.stat().st_mtime
        out.append(
            dict(
                slug=d.name,
                title=title,
                mtime=latest,
                when=time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(latest)
                ),
            )
        )
    return sorted(out, key=lambda v: v["mtime"], reverse=True)


_ACTIVATE = """<h1>🔒 Activate {{ph}}</h1>
<p class=sub>Yeh software chalane ke liye license key zaroori hai.
Apni key niche daal kar <b>Activate</b> dabayein.
<br><small>Ek key sirf ek PC par chalti hai.</small></p>
{% if error %}<div class=err>{{ error }}</div>{% endif %}
<div class=card>
  <form method=post action="{{ url_for('do_activate') }}">
    <label>License key</label>
    <input name=key autocomplete=off autofocus required
           placeholder="VR-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX">
    <button type=submit>Activate</button>
  </form>
</div>
<p style=margin-top:18px><small>Key nahi hai? Software dene wale se
rabta karein. Ye PC: <code>{{ mid }}</code></small></p>"""


@app.get("/activate")
def activate_page():
    if is_activated():
        return redirect(url_for("index"))
    from .license import machine_id
    return _page(_ACTIVATE, error=None, mid=machine_id())


@app.post("/activate")
def do_activate():
    ok, msg = lic_activate(request.form.get("key", ""))
    if ok:
        return redirect(url_for("index"))
    from .license import machine_id
    return _page(_ACTIVATE, error=msg, mid=machine_id())


_ADMIN = """<h1>🔑 Key Generator <small>(owner only)</small></h1>
{% if not enabled %}
<div class=err>Yeh page abhi <b>OFF</b> hai. On karne ke liye apni
<code>.env</code> file me yeh line daalein (customer ki copy me NA dalein):
<br><br><code>VIDLORE_ADMIN_TOKEN=apna-secret-password</code><br><br>
Phir server restart karein.</div>
{% else %}
<p class=sub>Customer/device ka naam likhein aur button dabayein —
neeche key ban jayegi.</p>
{% if error %}<div class=err>{{ error }}</div>{% endif %}
{% if key %}
<div class=card style="border-color:#2f6f3f">
  <label>✅ Nayi key (copy karke customer ko dein)</label>
  <input id=k value="{{ key }}" readonly
         style="font:600 18px ui-monospace,monospace;color:#7fffa6"
         onclick="this.select()">
  <button type=button class=ghost style=margin-top:10px
          onclick="navigator.clipboard.writeText(document.getElementById('k').value);this.textContent='Copied!'">
    Copy</button>
  <p><small>For: <b>{{ label }}</b></small></p>
</div>
{% endif %}
<div class=card>
  <form method=post action="{{ url_for('admin_keys') }}">
    <label>Admin password</label>
    <input name=token type=password autocomplete=off required
           value="{{ token }}" placeholder="apna VIDLORE_ADMIN_TOKEN">
    <label>Customer / device ka naam</label>
    <input name=label autocomplete=off placeholder="jaise: Ahmed ka laptop">
    <button type=submit>Generate Key</button>
  </form>
</div>
{% endif %}"""


@app.route("/admin/keys", methods=["GET", "POST"])
def admin_keys():
    admin_tok = os.environ.get("VIDLORE_ADMIN_TOKEN", "").strip()
    if not admin_tok:
        return _page(_ADMIN, enabled=False, key=None, error=None,
                     label="", token="")
    if request.method == "GET":
        return _page(_ADMIN, enabled=True, key=None, error=None,
                     label="", token="")
    given = (request.form.get("token") or "").strip()
    if given != admin_tok:
        return _page(_ADMIN, enabled=True, key=None,
                     error="Galat admin password.", label="", token="")
    label = (request.form.get("label") or "").strip()
    return _page(_ADMIN, enabled=True, key=make_key(label), error=None,
                 label=label or "(no name)", token=given)


@app.get("/")
def index():
    active = [
        (jid, j) for jid, j in JOBS.items() if j["status"] != "done"
    ]
    # Filter incomplete-on-disk to hide projects already shown as live
    # JOBS (avoid duplicate rows for a render currently in flight).
    _running_slugs = {Path(j.get("run_dir", "")).name
                       for _, j in active if j.get("run_dir")}
    incomplete = [
        v for v in _list_incomplete() if v["slug"] not in _running_slugs
    ]
    return _page(_DASH, vids=_list_videos(), jobs=active,
                  incomplete=incomplete)


@app.get("/new")
def new():
    return _form_page({})


@app.get("/bg-preview/<name>")
def bg_preview(name: str):
    """Serve a small cached PNG thumbnail for the background picker grid.
    Generated on first request, then served from disk for instant re-loads.
    Unknown names 404 so a stale template can't poison the cache."""
    p = bg_thumbnail(name)
    if not p or not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/png",
                     conditional=True, max_age=86400)


@app.get("/sample")
def sample():
    base = Path(__file__).resolve().parent.parent / "examples"
    import yaml

    by = yaml.safe_load((base / "sample_brief.yaml").read_text(encoding="utf-8"))
    return jsonify(
        title=by.get("title", ""),
        prompt=by.get("prompt", ""),
        script=(base / "sample_script.txt").read_text(encoding="utf-8"),
    )


_PREVIEW_LINES = [
    "In 1982, Forbes listed Pablo Escobar among the richest men in the world.",
    "And then, everything changed.",
]


@app.get("/voice-status")
def voice_status():
    """Backend readiness for the dashboard chips."""
    return jsonify(_voice_status())


@app.get("/voice-preview")
def voice_preview():
    """Generate a ~10-15s preview with the selected engine + preset so the
    user can listen BEFORE a full render. Falls back to Kokoro if the chosen
    premium engine isn't ready. No paid APIs."""
    from . import tts_backends as tb
    from .voice_presets import get_preset
    from .ffmpeg_tool import run

    model = (request.args.get("model") or "chatterbox").strip().lower()
    voice = (request.args.get("voice") or "deep_male_documentary").strip()
    if model not in ("chatterbox", "kokoro"):
        model = "chatterbox"
    backend = tb.get_backend(model)
    why = backend.is_available() if backend else "unknown backend"
    if why and model != "kokoro":  # graceful fallback to the fast backend
        kb = tb.get_backend("kokoro")
        if kb and kb.is_available() == "":
            backend, model, why = kb, "kokoro", ""
    if why:
        return jsonify(error=f"{model} not ready: {why}"), 503

    cfg = load_config()
    preset = get_preset(voice)
    pdir = OUT / "_voice_previews"
    pdir.mkdir(parents=True, exist_ok=True)
    parts = [pdir / f"{model}_{voice}_{i}.wav"
             for i in range(len(_PREVIEW_LINES))]
    items = [tb.SynthItem(t, p) for t, p in zip(_PREVIEW_LINES, parts)]
    try:
        backend.synth_batch(items, preset, device=cfg.tts_device, settings={})
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"{model} synth failed: {str(e)[:200]}"), 500
    # resample each to 44100/stereo, concat, normalize to -16 LUFS
    norm_parts = []
    for i, p in enumerate(parts):
        np_ = pdir / f"{model}_{voice}_n{i}.wav"
        run(["-i", str(p), "-ar", "44100", "-ac", "2", str(np_)])
        norm_parts.append(np_)
    clist = pdir / f"{model}_{voice}_list.txt"
    clist.write_text("\n".join(f"file '{p.name}'" for p in norm_parts) + "\n",
                     encoding="utf-8")
    raw = pdir / f"{model}_{voice}_raw.wav"
    run(["-f", "concat", "-safe", "0", "-i", str(clist), "-c", "copy", str(raw)])
    out = pdir / f"{model}_{voice}.wav"
    try:
        run(["-i", str(raw), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-ar", "44100", "-ac", "2", str(out)])
    except Exception:  # noqa: BLE001
        out = raw
    return send_file(out.resolve(), mimetype="audio/wav", conditional=True,
                     max_age=0)


def _truthy(v) -> bool:
    return str(v) in ("1", "on", "true", "yes")


def _form_page(f: dict, error=None) -> str:
    """Render the wizard with all option lists (one place to keep in sync)."""
    return _page(_FORM, error=error, f=f, themes=THEMES,
                 themes_meta=_themes_meta_list(), bg_meta=_bg_meta_list(),
                 durations=list(DURATION_BUCKETS),
                 style_modes=_style_modes_list(), voices=_voices_list(),
                 channels=_channels_list(),
                 premium_presets=_premium_presets(),
                 api_status=_api_status(),
                 ss_default=_shutterstock_default())


def _brief_from(form) -> Brief:
    b = Brief(
        title=form["title"].strip(),
        prompt=(form.get("prompt") or "reviewed").strip() or "reviewed",
        fmt=form.get("fmt", "documentary"),
        duration=form.get("duration", "6-8"),
        theme=form.get("theme", "history"),
        voice=(form.get("voice") or "").strip() or None,
        captions=_truthy(form.get("captions", "0")),
        background=form.get("background", "auto"),
        style=form.get("style", "auto"),
    )
    # CHANNEL / LOOK DNA — the primary editorial-identity control.
    # When 'auto' (no channel), brief.look_preset stays None and the
    # render runs on the legacy style-mode + theme path.  When a real
    # channel is selected, it routes through resolve_look() in the
    # pipeline and overrides pacing / cards / subtitles / music /
    # camera / grade / reveals.  Maps 1:1 to vidlore/look_presets/*.yaml.
    _look = (form.get("look_preset") or "auto").strip().lower()
    if _look and _look != "auto":
        b.look_preset = _look
    # Per-render output toggles (applied to Config in _run_job). Defaults
    # match the engine defaults (music/transitions/overlays ON, sfx OFF).
    # 'shutterstock' toggle sets VIDLORE_SHUTTERSTOCK env (also off
    # automatically when no token is configured).
    # Web Footage Engine mix — 4-way radio (off / light / balanced /
    # heavy). Stored in extra and translated to WEB_FOOTAGE_MIX +
    # WEB_FOOTAGE_ENGINE env vars by _run_job(). Keeps the cost ceiling
    # honest: 'off' fully bypasses the web crawl, no DuckDuckGo hits.
    _wfm = (form.get("wf_mix") or "off").strip().lower()
    if _wfm not in ("off", "light", "balanced", "heavy"):
        _wfm = "off"
    # Web Image Discovery mix — sister to wf_mix. Drives WEB_IMAGE_*.
    _wim = (form.get("wi_mix") or "balanced").strip().lower()
    if _wim not in ("off", "light", "balanced", "heavy"):
        _wim = "balanced"
    # Voice Mode — Upload (wins, handled by brief.voiceover) / Premium Local /
    # Legacy Basic. Premium is OFF unless explicitly selected. Stored in extra
    # and translated to VIDLORE_TTS_* env by _run_job().
    _vm = (form.get("voice_mode") or "legacy").strip().lower()
    if _vm not in ("upload", "premium", "legacy"):
        _vm = "legacy"
    _tts_model = (form.get("tts_model") or "chatterbox").strip().lower()
    if _tts_model not in ("chatterbox", "kokoro"):
        _tts_model = "chatterbox"
    _tts_voice = (form.get("tts_voice") or "deep_male_documentary").strip()
    b.extra = {
        "music": _truthy(form.get("music", "1")),
        "transitions": _truthy(form.get("transitions", "1")),
        "overlays": _truthy(form.get("overlays", "1")),
        "sfx": _truthy(form.get("sfx", "0")),
        "shutterstock": _truthy(form.get("shutterstock", "0")),
        "wf_mix": _wfm,
        "wi_mix": _wim,
        "voice_mode": _vm,
        "tts_model": _tts_model,
        "tts_voice": _tts_voice,
    }
    return b


@app.post("/script")
def gen_script():
    f = request.form
    try:
        brief = _brief_from(f)
    except (ValueError, KeyError) as e:
        return _form_page(f, error=str(e))
    cfg = load_config()
    # Save an uploaded voiceover into the run dir so it persists and the
    # render step picks it up automatically (the pipeline already splits
    # it per-scene via narrate_from_file).
    vo_name = ""
    up = request.files.get("voiceover")
    if up and up.filename:
        ext = Path(up.filename).suffix.lower() or ".mp3"
        if ext in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"):
            rd = run_dir_for(brief, OUT)
            rd.mkdir(parents=True, exist_ok=True)
            for old in rd.glob("voiceover.*"):
                old.unlink(missing_ok=True)
            saved = rd / f"voiceover{ext}"
            up.save(str(saved))
            vo_name = up.filename
            # Set it NOW so the script step transcribes the spoken words
            # and uses them as the script (footage matches the voice +
            # exact alignment). render_from_script re-discovers it later.
            brief.voiceover = str(saved)
    script_file = None
    pasted = (f.get("script") or "").strip()
    if pasted:
        tmp = Path(tempfile.mkstemp(suffix=".txt")[1])
        tmp.write_text(pasted, encoding="utf-8")
        script_file = str(tmp)
    elif not cfg.has_llm and not brief.voiceover:
        return _form_page(
            f, error="No ANTHROPIC_API_KEY set, so paste a script, upload a "
            "voiceover, or click “Load sample topic” to continue.")
    try:
        st = write_script(brief, cfg, OUT, script_file=script_file)
    except Exception as e:  # noqa: BLE001
        return _form_page(f, error=f"{type(e).__name__}: {e}")
    ex = getattr(brief, "extra", None) or {}
    from .style_modes import resolve_style
    mode = resolve_style(getattr(brief, "style", "auto"), theme=brief.theme,
                         title=brief.title, prompt=brief.prompt)
    return _page(
        _REVIEW,
        title=brief.title, theme=brief.theme, duration=brief.duration,
        captions="1" if brief.captions else "0",
        style=getattr(brief, "style", "auto"), fmt=brief.fmt,
        voice=brief.voice or "", background=brief.background,
        music="1" if ex.get("music", True) else "0",
        transitions="1" if ex.get("transitions", True) else "0",
        overlays="1" if ex.get("overlays", True) else "0",
        sfx="1" if ex.get("sfx", False) else "0",
        shutterstock="1" if ex.get("shutterstock", False) else "0",
        wf_mix=str(ex.get("wf_mix", "off") or "off").lower(),
        wi_mix=str(ex.get("wi_mix", "balanced") or "balanced").lower(),
        voice_mode=str(ex.get("voice_mode", "legacy")),
        tts_model=str(ex.get("tts_model", "chatterbox")),
        tts_voice=str(ex.get("tts_voice", "deep_male_documentary")),
        look_preset=getattr(brief, "look_preset", "") or "auto",
        style_label=mode.label,
        script_text=st.script_txt.read_text(encoding="utf-8"),
        n=len(st.script.scenes), words=st.script.word_count,
        voiceover_name=vo_name,
    )


# ── V3.4.2 — PROJECT-SETTINGS PERSISTENCE (editor-rerender fidelity) ──────────
# The dashboard resolves per-render settings (MG mode, audio/source toggles,
# voice, niche, recipe lock) into brief.extra + env. load_brief() does NOT
# round-trip brief.extra, so an editor RE-RENDER used to silently reset those to
# defaults — most importantly MG, which now defaults ON, would flip an MG-OFF
# project ON. We snapshot the RESOLVED settings to render_settings.json at render
# time and restore them on a rerender, so the original mode is preserved unless
# the editor explicitly changes it (editor overrides apply AFTER the restore).
def _persist_render_settings(run_dir: Path, brief: "Brief", cfg) -> None:
    """Best-effort snapshot of the resolved per-render settings. Never fatal."""
    try:
        snap = dict(getattr(brief, "extra", None) or {})
        snap["mg"] = os.environ.get("VIDLORE_MOTION_GRAPHICS") == "1"
        snap["ai_video"] = os.environ.get("VIDLORE_AI_VIDEO") == "1"   # expect False
        snap.setdefault("music", bool(getattr(cfg, "music_enabled", True)))
        snap.setdefault("sfx", bool(getattr(cfg, "sfx_enabled", False)))
        snap.setdefault("transitions", bool(getattr(cfg, "transitions_enabled", True)))
        snap.setdefault("overlays", bool(getattr(cfg, "overlays_enabled", True)))
        snap["captions"] = bool(getattr(brief, "captions", False))
        snap["voice_backend"] = os.environ.get("VIDLORE_TTS_BACKEND", "legacy")
        snap["_schema"] = "render_settings/1"
        (run_dir / "render_settings.json").write_text(
            json.dumps(snap, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _restore_render_settings(run_dir: Path, brief: "Brief") -> str:
    """Seed brief.extra from a prior render's render_settings.json so an editor
    rerender preserves the original mode. `setdefault` only fills keys the editor
    has not already set this session. Legacy projects (no snapshot) get a
    documented safe migration default — MG OFF (how every pre-V3.4.2 project was
    actually rendered). Returns a short status string for logging."""
    try:
        if not isinstance(getattr(brief, "extra", None), dict):
            brief.extra = {}
        sp = run_dir / "render_settings.json"
        if sp.exists():
            snap = json.loads(sp.read_text(encoding="utf-8"))
            snap.pop("_schema", None)
            for k, v in snap.items():
                brief.extra.setdefault(k, v)
            return "restored %d setting(s) from render_settings.json (mg=%s)" % (
                len(snap), snap.get("mg"))
        brief.extra.setdefault("mg", False)
        brief.extra.setdefault("_settings_migrated", "legacy_no_snapshot_mg_off")
        return "no render_settings.json (legacy project) → migration default MG OFF"
    except Exception:  # noqa: BLE001
        return "settings-restore skipped (error)"


def _run_job(job_id: str, brief: Brief) -> None:
    def prog(pct: int, msg: str) -> None:
        JOBS[job_id].update(pct=pct, msg=msg)

    try:
        cfg = load_config()
        # apply the wizard's per-render output toggles
        ex = getattr(brief, "extra", None) or {}
        if "music" in ex:
            cfg.music_enabled = bool(ex["music"])
        if "transitions" in ex:
            cfg.transitions_enabled = bool(ex["transitions"])
        if "overlays" in ex:
            cfg.overlays_enabled = bool(ex["overlays"])
        if "sfx" in ex:
            cfg.sfx_enabled = bool(ex["sfx"])
        # Footage-source kill-switch — VIDLORE_SHUTTERSTOCK env var
        # consulted by _shutterstock_fetch() at every call site.  We
        # SET it here per-render so the wizard's toggle wins over any
        # ambient shell setting.  '1' = allowed (and token must be
        # present); '0' = disabled regardless of token.
        os.environ["VIDLORE_SHUTTERSTOCK"] = "1" if ex.get(
            "shutterstock", False) else "0"
        # Web Footage Engine — translate the wizard's mix radio into
        # WEB_FOOTAGE_ENGINE + WEB_FOOTAGE_MIX env vars consulted by
        # web_footage.cfg_from_env() inside fetch_footage().  'off' is
        # the only explicit kill-switch; the other tiers all enable
        # the engine and just change the % of scenes it tries to fill.
        _wfm = str(ex.get("wf_mix", "off")).lower()
        if _wfm == "off":
            os.environ["WEB_FOOTAGE_ENGINE"] = "0"
        else:
            os.environ["WEB_FOOTAGE_ENGINE"] = "1"
            os.environ["WEB_FOOTAGE_MIX"] = _wfm
        # Web Image Discovery — sister env-var path. AI images keep
        # firing as the next-tier fallback regardless of this setting.
        _wim = str(ex.get("wi_mix", "balanced")).lower()
        if _wim == "off":
            os.environ["WEB_IMAGE_ENGINE"] = "0"
        else:
            os.environ["WEB_IMAGE_ENGINE"] = "1"
            os.environ["WEB_IMAGE_MIX"] = _wim
        # Voice Mode -> VIDLORE_TTS_* env. Upload-voiceover wins inside the
        # pipeline regardless, so for 'upload'/'legacy' we keep premium OFF.
        _vm = str(ex.get("voice_mode", "legacy")).lower()
        if _vm == "premium" and not brief.voiceover:
            os.environ["VIDLORE_TTS_BACKEND"] = "premium_local"
            os.environ["VIDLORE_TTS_MODEL"] = str(ex.get("tts_model", "chatterbox"))
            os.environ["VIDLORE_TTS_VOICE"] = str(ex.get("tts_voice", "deep_male_documentary"))
            cfg.tts_backend = "premium_local"
            cfg.tts_model = os.environ["VIDLORE_TTS_MODEL"]
            cfg.tts_voice = os.environ["VIDLORE_TTS_VOICE"]
        else:
            os.environ["VIDLORE_TTS_BACKEND"] = "legacy"
            cfg.tts_backend = "legacy"
        # MOTION-GRAPHICS ENGINE — explicit, documented PRODUCTION DEFAULT
        # (V3.4.2). The 71-primitive director is the flagship dashboard feature
        # but is gated behind VIDLORE_MOTION_GRAPHICS (off unless "1").
        # Historically nothing in the dashboard set it, so every portal render
        # silently ran MG OFF (legacy graphic_kind path). We now set it EXPLICITLY
        # here, every render (direct assignment, never setdefault — so a prior
        # job's value can't leak across renders, matching the shutterstock/web
        # toggles above):
        #   • default (extra absent / truthy) → MG ON,
        #   • extra["mg"] is False → MG OFF (the Review-Editor rerender restores
        #     the original project's recorded mode here; strict-replay forces OFF).
        # CLI is unaffected — it never calls _run_job. Rollback: flip "1"→"0".
        _mg_extra = ex.get("mg", None)
        os.environ["VIDLORE_MOTION_GRAPHICS"] = (
            "1" if (_mg_extra is None or _mg_extra) else "0")
        # Review-Editor global audio/caption overrides (Phase 4). Always reset
        # the music-volume env so settings never leak between concurrent jobs.
        try:
            from . import editor_manifest as _EM
            _g = _EM.load_overrides(Path(JOBS[job_id]["run_dir"])).get("global", {})
            if _g.get("captions_enabled") is not None:
                brief.captions = bool(_g["captions_enabled"])
            if _g.get("music_enabled") is not None:
                cfg.music_enabled = bool(_g["music_enabled"])
            os.environ["VIDLORE_MUSIC_VOLUME"] = str(
                float(_g["music_volume"]) if _g.get("music_volume") is not None else 1.0)
            # DOC_012 — editor LOOK override: force a niche/look family
            # (resolve_look honors brief.look_preset). "" / "auto" → auto-look.
            _lp = (_g.get("look_preset") or "").strip()
            if _lp and _lp.lower() != "auto":
                try:
                    brief.look_preset = _lp
                except Exception:  # noqa: BLE001
                    pass
            # Editorial RECIPE controls (Review-Editor): 'New Variation'
            # re-rolls the per-video recipe; 'Lock Look' pins it.  Carried
            # on brief.extra, read by resolve_look()'s auto-look branch
            # (editorial_recipe.py).  Only meaningful in auto-look mode.
            try:
                _ev = _g.get("editorial_variation")
                _el = _g.get("editorial_recipe_lock")
                if not isinstance(getattr(brief, "extra", None), dict):
                    brief.extra = {}
                if _ev is not None:
                    brief.extra["editorial_variation"] = int(_ev)
                if isinstance(_el, dict) and _el.get("niche"):
                    brief.extra["editorial_recipe_lock"] = _el
                else:
                    brief.extra.pop("editorial_recipe_lock", None)
                _rov = _g.get("recipe_overrides")
                if isinstance(_rov, dict) and any(
                        v is not None for v in _rov.values()):
                    brief.extra["recipe_overrides"] = _rov
                else:
                    brief.extra.pop("recipe_overrides", None)
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            os.environ["VIDLORE_MUSIC_VOLUME"] = "1"
        # P4 — render the SAME dir the job/editor targets (never recompute a
        # conflicting path from the title).
        # V3.4.2 — snapshot the resolved settings BEFORE the render so a later
        # editor rerender can restore this exact mode (MG/audio/source/voice).
        _persist_render_settings(Path(JOBS[job_id]["run_dir"]), brief, cfg)
        render_from_script(brief, cfg, OUT, progress=prog,
                           run_dir=Path(JOBS[job_id]["run_dir"]))
        # Review-Editor: on success, clear the one-shot regen flags + record the
        # rendered-edits signature so the 'unsaved' indicator clears and a later
        # re-export won't silently re-roll already-regenerated scenes.
        try:
            from . import editor_manifest as _EMr
            _rd = Path(JOBS[job_id]["run_dir"])
            _EMr.mark_rendered(_rd)
            JOBS[job_id].update(pct=98, msg="Checking final output…")
            _EMr.refresh_render_metrics(_rd)   # P3 — honest QA metrics on success
        except Exception:  # noqa: BLE001
            pass
        JOBS[job_id].update(status="done", pct=100, msg="Done")
    except Exception as e:  # noqa: BLE001
        JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}")


@app.post("/render")
def do_render():
    f = request.form
    try:
        brief = _brief_from(f)
    except (ValueError, KeyError) as e:
        return _form_page(f, error=str(e))
    run_dir = run_dir_for(brief, OUT)
    run_dir.mkdir(parents=True, exist_ok=True)
    # If a voiceover was uploaded on the form step it lives in the run
    # dir — use it (the pipeline splits it per-scene; AI voice is off).
    vo = next(iter(sorted(run_dir.glob("voiceover.*"))), None)
    if vo is not None:
        brief.voiceover = str(vo)
    # Persist the (possibly edited) script so the pipeline's hash-diff
    # logic regenerates only what changed.
    (run_dir / "script.txt").write_text(f.get("script_text", ""), encoding="utf-8")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = dict(status="running", pct=0, msg="Queued…",
                        error="", title=brief.title, run_dir=str(run_dir))
    threading.Thread(target=_run_job, args=(job_id, brief),
                     daemon=True).start()
    return redirect(url_for("job_page", job_id=job_id))


@app.get("/job/<job_id>")
def job_page(job_id: str):
    if job_id not in JOBS:
        abort(404)
    return _page(_JOB, job_id=job_id, title=JOBS[job_id]["title"])


@app.get("/job/<job_id>/status")
def job_status(job_id: str):
    j = JOBS.get(job_id) or abort(404)
    return jsonify(status=j["status"], pct=j["pct"], msg=j["msg"],
                   error=j["error"],
                   slug=Path(j.get("run_dir", "")).name)


@app.get("/job/<job_id>/file/<kind>")
def job_file(job_id: str, kind: str):
    j = JOBS.get(job_id) or abort(404)
    rd = Path(j["run_dir"])
    slug = rd.name
    paths = {
        "video": rd / f"{slug}.mp4",
        "thumb": rd / "thumbnail.jpg",
        "srt": rd / f"{slug}.srt",
    }
    p = paths.get(kind)
    if not p or not p.exists():
        abort(404)
    return send_file(p.resolve())


def _fmt_dur(secs) -> "Optional[str]":
    try:
        secs = int(float(secs))   # truncate (matches the player's displayed time)
        return f"{secs // 60}:{secs % 60:02d}"
    except Exception:  # noqa: BLE001
        return None


def _probe_resolution(mp4: Path) -> "Optional[str]":
    try:
        import imageio_ffmpeg
        import subprocess as _sp
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        out = _sp.run([ff, "-i", str(mp4)], capture_output=True, text=True).stderr
        m = re.search(r"(\d{3,4})x(\d{3,4})", out)
        return f"{m.group(1)} × {m.group(2)}" if m else None
    except Exception:  # noqa: BLE001
        return None


@app.get("/job/<job_id>/summary")
def job_summary(job_id: str):
    """Premium results dashboard data (read once on completion): duration, scenes,
    resolution, QA, Look, captions, file facts. Additive — no existing route changes."""
    j = JOBS.get(job_id) or abort(404)
    rd = Path(j["run_dir"])
    slug = rd.name
    mp4 = rd / f"{slug}.mp4"
    if not mp4.exists():
        return jsonify(ok=False)
    meta = _counts(rd, "render_meta.json")
    metrics = _counts(rd, "render_metrics.json")
    qa = metrics.get("qa") or {}
    try:
        look = (meta.get("editorial_recipe_summary") or "").split("·")[0].strip() or None
    except Exception:  # noqa: BLE001
        look = None
    try:
        import datetime as _dt
        completed = _dt.datetime.fromtimestamp(mp4.stat().st_mtime).strftime("%b %d, %I:%M %p")
    except Exception:  # noqa: BLE001
        completed = None
    return jsonify(
        ok=True, title=j.get("title"), slug=slug,
        duration=_fmt_dur(meta.get("video_seconds")),
        scenes=meta.get("scenes"), fps=meta.get("fps"),
        resolution=_probe_resolution(mp4),
        size_mb=round(mp4.stat().st_size / 1048576),
        completed=completed,
        captions=(rd / f"{slug}.srt").exists(),
        look=look,
        cuts=meta.get("cuts"), transitions=meta.get("transitions_motivated"),
        qa_verdict=qa.get("verdict"), qa_summary=qa.get("summary"),
        black_frames=metrics.get("black_frames"),
        lufs=(metrics.get("audio") or {}).get("lufs"),
        editor_ok=(rd / "script.json").exists(),
    )


def _counts(d: Path, name: str) -> dict:
    f = d / name
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _scene_rows(d: Path) -> tuple[str, list[dict]]:
    """(title, [{idx,text,v,vv}]) from script.json + the variant files."""
    title = d.name
    scenes: list[dict] = []
    sj = d / "script.json"
    if sj.exists():
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
            title = data.get("title") or title
            vs = _counts(d, "variants.json")
            vvs = _counts(d, "voice_variants.json")
            for i, s in enumerate(data.get("scenes", [])):
                txt = (s.get("narration") or "").strip()
                scenes.append(
                    dict(
                        idx=i,
                        v=int(vs.get(str(i), 0)),
                        vv=int(vvs.get(str(i), 0)),
                        text=txt[:160] + ("…" if len(txt) > 160 else ""),
                    )
                )
        except Exception:  # noqa: BLE001
            pass
    return title, scenes


def _start_render_job(d: Path, msg: str) -> str:
    """Kick a background render for an existing project dir; return the job_id.
    The pipeline's hash-diff + per-scene cache means only what actually changed
    is recomputed (unchanged scenes are copied from cache, no LLM/TTS/AI)."""
    if not (d / "script.json").exists():
        abort(404)
    brief = load_brief(d)
    if brief is None:  # legacy render without brief.json
        title, _ = _scene_rows(d)
        brief = Brief(title=title, prompt="reviewed")
    # V3.4.2 — preserve the ORIGINAL project settings on an editor rerender
    # (load_brief drops brief.extra). Editor overrides still apply on top inside
    # _run_job. Legacy projects fall back to a documented MG-OFF migration default.
    _settings_status = _restore_render_settings(d, brief)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = dict(status="running", pct=0, msg=msg, error="",
                        title=brief.title, run_dir=str(d),
                        settings=_settings_status)
    threading.Thread(target=_run_job, args=(job_id, brief),
                     daemon=True).start()
    return job_id


def _start_render(d: Path, msg: str):
    """Kick a render and redirect to its live progress page."""
    return redirect(url_for("job_page", job_id=_start_render_job(d, msg)))


def _nocap_path(d: Path) -> Path:
    return d / "editor_cache" / "preview_nocap.mp4"


def _nocap_fresh(d: Path) -> bool:
    """A no-caption proxy is usable if it exists and is at least as new as the
    project's captioned MP4 (same visual content, captions stripped)."""
    nc, mp4 = _nocap_path(d), d / f"{d.name}.mp4"
    return nc.exists() and mp4.exists() and nc.stat().st_mtime >= mp4.stat().st_mtime - 2


_NOCAP_PRESERVE = ("{slug}.mp4", "{slug}.srt", "render_meta.json", "render_metrics.json",
                   "render_black_frame_metrics.json", "thumbnail.jpg", "script.json",
                   "script.txt", "script.baseline.json", "sfx_cue_sheet.json",
                   "variants.json", "voice_variants.json", "motion_graphics_manifest.json")


def _prepare_caption_proxy(d: Path, job_id: str) -> None:
    """Issue 2 — render a NO-CAPTION preview proxy (captions=False) so the editor can
    toggle captions live (HTML caption overlay over a clean base). To reproduce the
    project's EXACT footage (its cache + variants.json pin the per-scene clips — a
    fresh temp render re-fetches different / watermarked stock), this renders IN the
    project dir and BACKS UP + RESTORES every output file render touches, so the real
    captioned MP4 is preserved. Only the no-caption result is kept, at
    editor_cache/preview_nocap.mp4."""
    import shutil as _sh
    ecache = d / "editor_cache"
    ecache.mkdir(exist_ok=True)
    bdir = ecache / f"nocap_backup_{job_id}"
    _sh.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True)
    slug = d.name
    preserve = []
    for tmpl in _NOCAP_PRESERVE:
        fn = tmpl.format(slug=slug)
        if fn not in preserve:
            preserve.append(fn)
    try:
        cfg = load_config()
        brief = load_brief(d)
        if brief is None:
            title, _ = _scene_rows(d)
            brief = Brief(title=title, prompt="reviewed")
        try:
            brief.captions = False
        except Exception:  # noqa: BLE001
            pass
        # snapshot the files render will overwrite, so the project stays intact
        for fn in preserve:
            p = d / fn
            if p.exists():
                _sh.copy2(p, bdir / fn)

        def prog(pct: int, msg: str) -> None:
            JOBS[job_id].update(pct=int(pct), msg="Caption preview · " + msg)
        # reuse the project's EXACT cache + footage decisions → matching visuals
        render_from_script(brief, cfg, OUT, progress=prog, run_dir=d)
        produced = d / f"{slug}.mp4"
        ok = produced.exists() and produced.stat().st_size > 100_000
        if ok:
            _sh.copy2(produced, _nocap_path(d))
        # ALWAYS restore the real (captioned) project outputs
        for fn in preserve:
            b = bdir / fn
            if b.exists():
                _sh.copy2(b, d / fn)
        if ok:
            JOBS[job_id].update(status="done", pct=100, msg="Caption preview ready")
        else:
            JOBS[job_id].update(status="error", error="proxy produced no MP4")
    except Exception as e:  # noqa: BLE001
        # best-effort restore on failure
        for fn in preserve:
            b = bdir / fn
            if b.exists():
                try:
                    _sh.copy2(b, d / fn)
                except Exception:  # noqa: BLE001
                    pass
        JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}")
    finally:
        _sh.rmtree(bdir, ignore_errors=True)


@app.get("/e/<slug>/caption-proxy/status")
def editor_caption_proxy_status(slug: str):
    d = _ed_dir(slug)
    return jsonify(ready=_nocap_fresh(d))


@app.post("/e/<slug>/caption-proxy")
def editor_caption_proxy(slug: str):
    """Build (or reuse) the no-caption preview proxy. Returns ready=True immediately if
    a fresh one exists, else kicks a background job and returns its id to poll."""
    d = _ed_dir(slug)
    if _nocap_fresh(d):
        return jsonify(ok=True, ready=True)
    # de-dupe: if a proxy build for THIS project is already running, return it
    # (a second build would rmtree the first's work dir mid-render).
    for jid, j in JOBS.items():
        if j.get("title") == "caption preview" and j.get("run_dir") == str(d) \
                and j.get("status") == "running":
            return jsonify(ok=True, ready=False, job_id=jid)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = dict(status="running", pct=0, msg="Preparing caption preview…",
                        error="", title="caption preview", run_dir=str(d))
    threading.Thread(target=_prepare_caption_proxy, args=(d, job_id), daemon=True).start()
    return jsonify(ok=True, ready=False, job_id=job_id)


@app.get("/e/<slug>/file/nocap-video")
def editor_nocap_video(slug: str):
    d = _ed_dir(slug)
    p = _nocap_path(d)
    if not p.exists():
        abort(404)
    # NB: resolve() to an absolute path — werkzeug's send_file resolves a
    # RELATIVE path against the Flask app root (vidlore/), not the CWD, so a
    # bare str(p) 500s with FileNotFoundError. Matches the sibling routes.
    return send_file(str(p.resolve()), mimetype="video/mp4", conditional=True)


def _bump_and_render(d: Path, name: str, idx: int, label: str):
    """Bump scene `idx`'s counter in `name` then re-render — only that
    scene's cache entry is invalidated, the rest is reused."""
    if not (d / "script.json").exists():
        abort(404)
    cur = _counts(d, name)
    cur[str(idx)] = int(cur.get(str(idx), 0)) + 1
    (d / name).write_text(json.dumps(cur), encoding="utf-8")
    return _start_render(d, f"{label} scene {idx}…")


def _current_script_text(d: Path) -> str:
    st = d / "script.txt"
    if st.exists():
        return st.read_text(encoding="utf-8")
    sj = d / "script.json"
    if sj.exists():
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
            return (
                (data.get("title", "") + "\n\n"
                 + "\n\n".join(s.get("narration", "")
                               for s in data.get("scenes", []))).strip()
            )
        except Exception:  # noqa: BLE001
            pass
    return ""


@app.get("/v/<slug>")
def video_page(slug: str):
    d = _safe_dir(slug)
    if not (d / f"{slug}.mp4").exists():
        abort(404)
    title, scenes = _scene_rows(d)
    # Review Editor needs script.json (the editable model). Show the button
    # when it's present; otherwise surface a clear reason rather than hiding.
    editor_ok = (d / "script.json").exists()
    editor_reason = ("" if editor_ok else
                     "this render has no script.json (older video) — "
                     "re-render once to enable the Review Editor")
    return _page(
        _VIDEO, slug=slug, title=title, scenes=scenes,
        script_text=_current_script_text(d),
        has_thumb=(d / "thumbnail.jpg").exists(),
        has_srt=(d / f"{slug}.srt").exists(),
        editor_ok=editor_ok, editor_reason=editor_reason,
    )


@app.post("/v/<slug>/edit")
def edit_script(slug: str):
    d = _safe_dir(slug)
    if not (d / "script.json").exists():
        abort(404)
    (d / "script.txt").write_text(request.form.get("script_text", ""), encoding="utf-8")
    return _start_render(d, "Re-rendering edited script…")


@app.post("/v/<slug>/resume")
def resume_render(slug: str):
    """Resume an incomplete / failed render directly from its slug.
    Used by the dashboard's "Incomplete" section when the in-memory
    JOBS entry is gone (e.g. after a server restart) but the project
    dir on disk still has script.json + brief.json + cache. Same
    semantics as /job/<id>/retry but addressed by slug instead of
    job_id. Bug fix 2026-05-26."""
    d = _safe_dir(slug)
    if not (d / "script.json").exists() or not (d / "brief.json").exists():
        abort(404)
    return _start_render(d, "Resuming failed render…")


@app.post("/job/<job_id>/retry")
def retry_job(job_id: str):
    """Re-fire a failed (or finished) render with the same brief +
    script. Uses the saved run_dir so script.json + brief.json + cache
    are all reused — only the failed phase re-runs. The new render
    inherits any code fixes shipped after the first attempt (e.g. the
    Too-many-open-files RLIMIT_NOFILE bump applied 2026-05-26)."""
    info = JOBS.get(job_id)
    if not info:
        abort(404)
    rd = info.get("run_dir")
    if not rd or not Path(rd).exists():
        abort(404)
    return _start_render(Path(rd), "Retrying render…")


@app.post("/v/<slug>/regen/<int:idx>")
def regen(slug: str, idx: int):
    return _bump_and_render(_safe_dir(slug), "variants.json", idx,
                            "Regenerating visual for")


@app.post("/v/<slug>/revoice/<int:idx>")
def revoice(slug: str, idx: int):
    return _bump_and_render(_safe_dir(slug), "voice_variants.json", idx,
                            "Re-voicing")


@app.post("/v/<slug>/delete")
def delete_video(slug: str):
    d = _safe_dir(slug)  # path-traversal safe
    shutil.rmtree(d, ignore_errors=True)
    # drop any finished/failed job rows pointing at the removed dir
    for jid in [j for j, v in JOBS.items()
                if Path(v["run_dir"]).name == slug
                and v["status"] != "running"]:
        JOBS.pop(jid, None)
    return redirect(url_for("index"))


@app.get("/v/<slug>/file/<kind>")
def video_file(slug: str, kind: str):
    d = _safe_dir(slug)
    paths = {
        "video": d / f"{slug}.mp4",
        "thumb": d / "thumbnail.jpg",
        "srt": d / f"{slug}.srt",
    }
    p = paths.get(kind)
    if not p or not p.exists():
        abort(404)
    return send_file(p.resolve())


# ── Review Editor (Phase 1 — read-only) ──────────────────────────────── #
_EDITOR = """
<div id=ed-root data-slug="{{slug}}"></div>
<style>
/* ====================  EDITING STUDIO  ====================
   Full-width, spacious, CapCut-desktop-like. Overrides the 840px
   page cap so the editor uses the whole screen. */
body{max-width:none!important;margin:0!important;padding:0!important;
  background:#0a0d12!important;color:#e7eef6}
#ed-root{padding:14px clamp(16px,2.1vw,40px) 20px}
.edhead{display:flex;align-items:center;gap:14px;flex-wrap:nowrap;margin-bottom:14px;
  background:#10151c;border:1px solid #1e2731;border-radius:14px;padding:11px 16px;
  position:sticky;top:0;z-index:30}
.edhl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;flex:1;min-width:0}
.edhr{display:flex;align-items:center;gap:9px;flex:none}
.edhead h1{font-size:20px;font-weight:650;margin:0;letter-spacing:-.01em;
  max-width:38ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@media(max-width:820px){.edhead{flex-wrap:wrap}.edhr{flex-wrap:wrap}}
.edchip{font-size:12px;padding:5px 12px;border-radius:20px;border:1px solid #26303c;
  color:#9fb0c2;background:#0e141b;white-space:nowrap}
.edchip.ok{color:#62cc83;border-color:#2e5e42;background:#102019}
.edchip.warn{color:#e2b552;border-color:#5e5230;background:#221d11}
.edchip.bad{color:#e2864e;border-color:#5e3630;background:#221411}
a.edchip{text-decoration:none}a.edchip:hover{border-color:#3b82f6;color:#d4e4f7}

.edwrap{display:grid;grid-template-columns:308px minmax(0,1fr) 366px;
  grid-template-rows:minmax(0,1fr) 210px;
  grid-template-areas:"scenes preview inspector" "timeline timeline timeline";
  gap:14px;height:calc(100vh - 116px);min-height:540px}
.edscenes{grid-area:scenes;overflow-y:auto;background:#10151c;border:1px solid #1e2731;border-radius:14px;padding:10px}
.edpreview{grid-area:preview;display:flex;flex-direction:column;gap:11px;min-width:0;min-height:0}
.edinspector{grid-area:inspector;overflow-y:auto;background:#10151c;border:1px solid #1e2731;border-radius:14px;padding:16px}
.edtimeline{grid-area:timeline;overflow-x:auto;overflow-y:auto;background:#10151c;border:1px solid #1e2731;border-radius:14px;padding:14px 16px;position:relative}

/* SCENE LIST */
.scrow{display:flex;gap:11px;padding:9px;border-radius:11px;cursor:pointer;border:1px solid transparent;
  margin-bottom:5px;transition:background .12s,border-color .12s;position:relative}
.scrow:hover{background:#161d27}
.scrow.sel{background:#152334;border-color:#2f5d8a;box-shadow:inset 3px 0 0 #3b82f6}
.scrow.dragging{opacity:.45}
.scrow.dropabove{box-shadow:inset 0 3px 0 #3b82f6}
.scrow.dropbelow{box-shadow:inset 0 -3px 0 #3b82f6}
.scgrip{flex:none;align-self:center;color:#4a5564;font-size:15px;line-height:1;cursor:grab;
  width:13px;text-align:center;user-select:none}
.scrow:hover .scgrip{color:#7c8aa0}
.scrow.dragging .scgrip{cursor:grabbing}
.scrow img{width:90px;height:51px;object-fit:cover;border-radius:7px;background:#0c1116;flex:none}
.scrow .scmeta{min-width:0;display:flex;flex-direction:column;justify-content:center}
.scrow .sctt{font-size:13px;font-weight:500;color:#dde6f0;line-height:1.3;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.scb{display:inline-block;font-size:10px;padding:2px 8px;border-radius:11px;margin:4px 4px 0 0;
  background:#1a232f;color:#9fb0c2;border:1px solid #29333f}
.scb.ai{color:#c79bf0;border-color:#4a3a63}.scb.card{color:#7fd0a0;border-color:#2e5e42}
.scb.skip{color:#e0844e;border-color:#5e3630;text-decoration:line-through}
.scb.foot{color:#7fb0e0;border-color:#2e4a63}.scb.edited{color:#ffd76a;border-color:#5e5230}

/* PREVIEW — big + central */
.edpreview>video,video{width:100%;flex:1;min-height:0;background:#000;border-radius:14px;
  border:1px solid #1e2731;object-fit:contain;max-height:none}
#edpchips{color:#7f8c9b;font-size:12px;text-align:center;letter-spacing:.02em}

/* INSPECTOR */
.insp h3{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:#6f7d8c;
  margin:18px 0 8px;padding-bottom:6px;border-bottom:1px solid #1b2530}
.insp h3:first-child{margin-top:0}
.insp .narr{font-size:13.5px;line-height:1.55;color:#d7dee6}
.insp .kw{display:inline-block;font-size:11px;padding:3px 9px;border-radius:12px;background:#15202b;
  color:#9fb6cc;margin:3px 3px 0 0}
.insp .fld{display:flex;gap:8px;font-size:13px;margin:4px 0}.insp .fld b{color:#7f97a8;min-width:88px;flex:none}
.dots{letter-spacing:2px;color:#e0b24e}

/* TIMELINE — taller, clearer, higher contrast */
.tk{display:flex;align-items:center;height:40px;margin:5px 0;position:relative}
.tklab{width:78px;font-size:11px;font-weight:600;color:#8593a3;flex:none;text-transform:uppercase;letter-spacing:.05em}
.tkrow{position:relative;flex:1;height:33px;background:#0b1119;border-radius:7px;border:1px solid #161e28}
.blk{position:absolute;top:2px;height:29px;border-radius:5px;font-size:10px;color:#eaf1f8;display:flex;align-items:center;
  overflow:hidden;white-space:nowrap;padding:0 7px;box-sizing:border-box;cursor:pointer;border:1px solid #2a3a4a;transition:filter .1s}
.blk:hover{filter:brightness(1.28)}
.blk.v{background:#1b3147;background-size:cover;background-position:center}.blk.v.sel{outline:2px solid #5fa8ff;z-index:3}
.blk.c{background:#1e4030}.blk.cap{background:#2a2440}.blk.m{background:#3a2a1e}.blk.vo{background:#23303a}
.playhead{position:absolute;top:4px;bottom:4px;width:2.5px;background:#ff5a5a;z-index:5;pointer-events:none;
  border-radius:2px;box-shadow:0 0 7px rgba(255,90,90,.55)}

/* CONTROL / AUDIO BAR under the preview */
.audiobar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:13px;color:#aeb9c6;
  background:#10151c;border:1px solid #1e2731;border-radius:12px;padding:12px 15px}
.audiobar label{display:flex;align-items:center;gap:6px;margin:0}
.audiobar input[type=range]{accent-color:#3b82f6}
.audiobar select{background:#0e141b;border:1px solid #2a3340;color:#dfe7ef;border-radius:7px;padding:5px 8px;font-size:12px}
.audtag{font-size:10px;font-weight:700;letter-spacing:.08em;color:#7c8aa0;background:#10161e;border:1px solid #232c38;border-radius:5px;padding:2px 6px;white-space:nowrap}
/* CapCut-clean: Menu dropdown + whole-video popover + reusable popup */
.edmenuwrap{position:relative;display:inline-block}
.edpop{position:absolute;top:calc(100% + 6px);right:0;min-width:216px;background:#141b24;border:1px solid #2a3340;border-radius:10px;box-shadow:0 14px 38px rgba(0,0,0,.55);padding:6px;z-index:120}
.edpop[hidden]{display:none}
.edpopgrp{font-size:10px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#6f7d8c;padding:8px 10px 3px}
.edpopitem{display:block;width:100%;text-align:left;background:none;border:0;color:#dfe7ef;font-size:13px;padding:8px 10px;border-radius:7px;cursor:pointer;white-space:nowrap}
.edpopitem:hover{background:#1d2733}
.edpopitem.danger{color:var(--danger-text,#f2c3a0)}
.edpopitem.danger:hover{background:var(--danger-soft,#2a1714)}
.edpopsep{height:1px;background:#222c38;margin:5px 4px}
.edvback{position:fixed;inset:0;background:rgba(6,9,13,.55);z-index:150}
.edvback[hidden]{display:none}
#edvpop{position:fixed;top:50%;left:50%;right:auto;transform:translate(-50%,-50%);width:min(560px,92vw);max-height:84vh;overflow:auto;z-index:160;padding:16px 18px}
.edpoph{display:flex;align-items:center;justify-content:space-between;font-size:15px;font-weight:600;color:#eef3f8;margin-bottom:12px}
.edpopx{background:none;border:0;color:#8a93a6;font-size:15px;cursor:pointer;border-radius:6px;width:28px;height:28px}
.edpopx:hover{background:#1d2733;color:#dfe7ef}
.edvsec{margin-bottom:8px}
.edvseclbl{font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:#6f7d8c;margin-bottom:8px}
.edvsummary{display:flex;align-items:center;gap:10px;width:100%;background:#10151c;border:1px solid #1e2731;border-radius:10px;padding:9px 12px;color:#aeb9c6;cursor:pointer;font-size:13px;text-align:left}
.edvsummary:hover{border-color:#2a3a4d;background:#131b24}
.edvslead{font-weight:600;color:#dfe7ef}
.edvspills{flex:1;color:#7c8aa0;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.edvscog{color:#8a93a6;font-size:15px}
/* scene-row ••• overflow menu */
.edscenes .scrow{position:relative}
.scdots{position:absolute;top:5px;right:5px;width:24px;height:24px;border:0;background:rgba(20,27,36,.86);color:#8a93a6;border-radius:6px;cursor:pointer;font-size:16px;line-height:22px;text-align:center;padding:0;opacity:0;transition:opacity .12s;z-index:3}
.edscenes .scrow:hover .scdots,.scdots:focus{opacity:1}
.scdots:hover{background:#1d2733;color:#dfe7ef}
.rmenu{min-width:174px}
/* timeline: scene-number-primary block labels */
#edtlb .blk.v{display:flex;align-items:center;gap:3px;overflow:hidden}
.blknum{flex:none;font-weight:700;font-size:10px;font-variant-numeric:tabular-nums;background:rgba(0,0,0,.5);color:#fff;padding:0 4px;border-radius:4px;line-height:15px}
.blktxt{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
#edtlb .blk.v.sel .blknum{background:var(--accent,#3b82f6)}
/* Bug 2 — grabbable playhead knob + click/drag-to-seek cursors */
.phknob{position:absolute;top:-4px;left:50%;transform:translateX(-50%);width:16px;height:15px;border-radius:3px 3px 7px 7px;background:var(--playhead,#ff5a5a);box-shadow:0 1px 5px rgba(0,0,0,.6);cursor:ew-resize;pointer-events:auto;z-index:12}
.phknob:hover{filter:brightness(1.18)}
#edtlb.scrubbing{cursor:ew-resize;user-select:none}
#edtlb .tkruler,#edtlb .tkruler .tkrul{cursor:ew-resize}
/* Bug 1 — instant replaced-visual pip in the preview corner (non-blocking) */
.edrepov{position:absolute;top:10px;right:10px;width:154px;background:#0e141b;border:1px solid #2f5d8a;border-radius:9px;overflow:hidden;z-index:6;box-shadow:0 8px 22px rgba(0,0,0,.55)}
.edrepov[hidden]{display:none}
.edrepov img{width:100%;height:86px;object-fit:cover;display:block;background:#0c1116}
.edrepovlbl{padding:6px 8px;font-size:10.5px;line-height:1.3;color:#bfe0c4}
/* ===== Live draft preview layer stack (over the rendered MP4) ===== */
.edlay{position:absolute;inset:0;z-index:5;pointer-events:none;display:flex;align-items:center;justify-content:center;overflow:hidden}
.edlay[hidden]{display:none}
#edlayvisual{background-size:cover;background-position:center;background-repeat:no-repeat;background-color:#0a0e14}
#edlayvisual.kb{animation:edkb 14s ease-out forwards}
@keyframes edkb{from{transform:scale(1.02)}to{transform:scale(1.13)}}
.edlayvid{width:100%;height:100%;object-fit:cover;display:block}
#edlaycard{z-index:6}
.dcfull{width:100%;height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;background:linear-gradient(180deg,rgba(8,10,14,.5),rgba(8,10,14,.78));padding:6%}
.dctitle{font-size:clamp(18px,3.4vw,40px);font-weight:800;color:#fff;text-shadow:0 2px 10px rgba(0,0,0,.6);max-width:82%;line-height:1.15}
.dcbody{margin-top:3%;font-size:clamp(11px,1.5vw,18px);color:#e8eef5;max-width:74%;line-height:1.4}
.dcnum{font-size:clamp(40px,9vw,118px);font-weight:900;color:#ffd479;text-shadow:0 3px 14px rgba(0,0,0,.6)}
.dcsub{margin-top:2%;font-size:clamp(12px,1.8vw,22px);color:#fff;text-transform:uppercase;letter-spacing:.05em}
.dcdoc{width:78%;max-height:80%;overflow:hidden;background:#f4f1ea;color:#1a1813;border-radius:4px;padding:5% 6%;box-shadow:0 10px 40px rgba(0,0,0,.6)}
.dcdoctitle{font-weight:800;font-size:clamp(13px,1.9vw,22px);border-bottom:2px solid #c9a24a;padding-bottom:8px;margin-bottom:10px}
.dcdocbody{font-size:clamp(10px,1.4vw,15px);line-height:1.5;color:#3a342a}
.dclower{position:absolute;left:5%;right:5%;bottom:9%;background:rgba(10,14,20,.82);border-left:4px solid var(--accent,#3b82f6);border-radius:6px;padding:10px 16px;font-size:clamp(13px,1.8vw,20px);font-weight:700;color:#fff;text-align:left}
/* V1.4.4 — draft-card safety: never overflow the stage, clamp long body text, and mark
   the overlay as an editing aid (only ever shown for an EDITED card). */
#edlaycard .dcfull{max-height:100%;overflow:hidden}
#edlaycard .dclower{max-height:34%;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical}
#edlaycard .dctitle{max-width:84%;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
#edlaycard .dcbody{display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.edcarddraft{outline:2px dashed rgba(110,160,230,.45);outline-offset:-4px}
.edcardnote{position:absolute;left:50%;bottom:9%;transform:translateX(-50%);display:flex;align-items:center;gap:7px;
  font-size:12px;font-weight:600;color:#dfe9f6;background:rgba(8,11,16,.84);border:1px solid #36506f;border-radius:20px;
  padding:6px 14px;max-width:88%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;box-shadow:0 4px 14px rgba(0,0,0,.42)}
.edcardnotedot{width:7px;height:7px;border-radius:50%;background:#6db1ff;box-shadow:0 0 7px #6db1ff;flex:none}
.edcardhide{position:absolute;left:0;right:0;bottom:0;padding:8px 12px;text-align:center;font-size:12px;color:#cfe0c7;background:linear-gradient(transparent,rgba(8,11,16,.85))}
.eddraftlbl{position:absolute;left:10px;bottom:10px;z-index:8;display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;color:#bfe0c4;background:rgba(8,11,16,.72);border:1px solid #2f5d4a;border-radius:20px;padding:3px 10px;pointer-events:none}
.eddraftlbl[hidden]{display:none}
.eddraftdot{width:7px;height:7px;border-radius:50%;background:#54c98a;box-shadow:0 0 6px #54c98a}
/* Live status pill (Generating…, New visual added) — top-centre of the preview stage */
.edstatusmsg{position:absolute;left:50%;top:10px;transform:translateX(-50%);z-index:9;display:flex;align-items:center;gap:7px;
  font-size:12px;font-weight:600;color:#dfe9f6;background:rgba(8,11,16,.82);border:1px solid #36506f;border-radius:20px;padding:5px 13px;pointer-events:none;box-shadow:0 4px 14px rgba(0,0,0,.4)}
.edstatusmsg[hidden]{display:none}
.edstatusdot{width:7px;height:7px;border-radius:50%;background:#6db1ff;box-shadow:0 0 7px #6db1ff;animation:edstatpulse 1.1s ease-in-out infinite}
@keyframes edstatpulse{0%,100%{opacity:.45}50%{opacity:1}}
/* Live caption overlay (shown over a no-caption proxy when captions are ON) */
.edlaycaptions{position:absolute;left:0;right:0;bottom:7%;z-index:7;display:flex;justify-content:center;pointer-events:none;padding:0 8%}
.edlaycaptions[hidden]{display:none}
.edlaycaptions .edcapln{font-family:'Playfair Display',Georgia,serif;font-weight:700;font-size:clamp(15px,2.5vw,30px);line-height:1.25;color:#fff;text-align:center;
  text-shadow:0 2px 5px rgba(0,0,0,.85),0 0 2px rgba(0,0,0,.9);max-width:100%}
/* Tooltip engine */
.edtip{position:fixed;z-index:99999;max-width:264px;background:#0b1117;color:#eaf1fb;border:1px solid #314256;border-radius:8px;
  padding:7px 10px;font-size:12px;line-height:1.45;box-shadow:0 10px 28px rgba(0,0,0,.55);pointer-events:none;display:none}
/* Layers panel (inspector) */
.edlayers{display:flex;flex-direction:column;gap:2px}
.edlayrow{display:flex;align-items:center;gap:9px;padding:5px 4px;border-radius:7px}
.edlayrow:hover{background:#131b24}
.edlayeye{width:30px;height:26px;border:1px solid #2a3340;background:#0e141b;color:#6f7d8c;border-radius:7px;cursor:pointer;font-size:14px;line-height:1;flex:none}
.edlayeye.on{color:#dfe7ef;border-color:#2f5d8a;background:#152334}
.edlayeye:hover{border-color:#3a4856}
.edlayname{font-size:13px;color:#dfe7ef;font-weight:500}
.edlaynote{margin-left:auto;font-size:10.5px;color:#6f7d8c}
/* timeline direct drag-reorder (P1) */
#edtlb .blk.v[draggable=true]{cursor:grab}
#edtlb .blk.v.tldragging{opacity:.4;cursor:grabbing}
#edtlb .blk.v.tldropl{box-shadow:inset 3px 0 0 var(--accent,#3b82f6),0 0 0 1px var(--accent,#3b82f6)}
#edtlb .blk.v.tldropr{box-shadow:inset -3px 0 0 var(--accent,#3b82f6),0 0 0 1px var(--accent,#3b82f6)}

/* BUTTONS */
.edbtns{display:flex;gap:9px;flex-wrap:wrap;margin-top:10px}
.edbtn{background:#1b2836;border:1px solid #2f4a63;color:#cfe0f0;border-radius:9px;padding:8px 14px;font-size:13px;cursor:pointer;transition:background .12s}
.edbtn:hover{background:#233649}
.ghost{background:#141b24;border:1px solid #26303c;color:#aebccb;border-radius:9px;padding:8px 14px;font-size:13px;cursor:pointer;transition:background .12s}
.ghost:hover{background:#1b232e}
#edexport{background:#3b82f6;color:#fff;border-color:#3b82f6;font-weight:600;padding:9px 18px;border-radius:10px}
#edexport:hover{background:#2f70e0}

.muted{color:#6f7d8c;font-size:12.5px}
.warnstrip{display:flex;gap:8px;flex-wrap:wrap;margin:-4px 0 12px}
.warnb{font-size:12px;padding:4px 11px;border-radius:12px;background:#241d10;color:#e6bd63;border:1px solid #5e5230}
.fld2{margin:9px 0}.flab{display:block;font-size:10px;color:#7f8c9b;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}
#cardfields input,#cardfields textarea,.edinspector input[type=file]{width:100%;box-sizing:border-box;background:#0e141b;
  border:1px solid #2a3340;color:#dfe7ef;border-radius:8px;padding:8px 10px;font-size:13px;font-family:inherit}
.searchgrid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:8px}
.sres{width:100%;height:58px;object-fit:cover;border-radius:7px;cursor:pointer;border:2px solid transparent}
.sres:hover{border-color:#5fa8ff}

/* scrollbars + responsive */
.edscenes::-webkit-scrollbar,.edinspector::-webkit-scrollbar,.edtimeline::-webkit-scrollbar{width:10px;height:10px}
.edscenes::-webkit-scrollbar-thumb,.edinspector::-webkit-scrollbar-thumb,.edtimeline::-webkit-scrollbar-thumb{background:#26303d;border-radius:6px}
.edscenes::-webkit-scrollbar-thumb:hover,.edinspector::-webkit-scrollbar-thumb:hover{background:#33414f}
@media(max-width:1000px){.edwrap{grid-template-columns:1fr;grid-template-rows:auto 56vh auto 230px;
  grid-template-areas:"scenes" "preview" "inspector" "timeline";height:auto;min-height:0}
  .edscenes{max-height:240px}}

/* ---- PHASE 2 : preview transport, timeline zoom, collapsible inspector ---- */
/* preview transport bar (scene nav + play + time) */
.edtransport{display:flex;align-items:center;justify-content:center;gap:6px;flex-wrap:wrap;
  background:#10151c;border:1px solid #1e2731;border-radius:12px;padding:8px 12px}
.tbtn{background:#1a2330;border:1px solid #2a3340;color:#dfe7ef;border-radius:8px;
  min-width:38px;height:34px;padding:0 11px;font-size:14px;cursor:pointer;display:flex;align-items:center;gap:5px;transition:background .12s}
.tbtn:hover{background:#22303f}.tbtn.play{background:#3b82f6;border-color:#3b82f6;color:#fff;min-width:46px}
.tbtn.play:hover{background:#2f70e0}
.ttime{font-variant-numeric:tabular-nums;font-size:12.5px;color:#9fb0c2;margin:0 8px;min-width:92px;text-align:center}
.edpreview>video{flex:1}

/* timeline header (zoom toolbar) */
.tlhead{display:flex;align-items:center;gap:8px;margin-bottom:8px;position:sticky;left:0}
.tlhead .tlt{font-size:11px;font-weight:600;color:#8593a3;text-transform:uppercase;letter-spacing:.05em}
.tlhead .sp{flex:1}
.zbtn{background:#1a2330;border:1px solid #2a3340;color:#cfe0f0;border-radius:7px;width:30px;height:28px;font-size:14px;cursor:pointer;display:inline-flex;align-items:center;justify-content:center}
.zbtn:hover{background:#22303f}
.tlbody{min-width:100%;position:relative}
.tklab{position:sticky;left:0;z-index:4;background:#10151c}
.blk.sel-clip{outline:2px solid #5fa8ff;outline-offset:0;z-index:4;filter:brightness(1.15)}

/* collapsible inspector sections */
.insec{border:1px solid #1b2530;border-radius:11px;margin-bottom:9px;background:#0e141b;overflow:hidden}
.insec>.inhd{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none;
  padding:11px 13px;font-size:12px;font-weight:600;letter-spacing:.04em;color:#cdd8e4;text-transform:uppercase}
.insec>.inhd:hover{background:#131b24}
.insec>.inhd .car{margin-left:auto;color:#6f7d8c;font-size:11px;transition:transform .15s}
.insec.col>.inhd .car{transform:rotate(-90deg)}
.insec>.inbody{padding:4px 13px 13px}
.insec.col>.inbody{display:none}
.insec.danger{border-color:#3a2632}.insec.danger>.inhd{color:#e89292}
.insec h3{margin:12px 0 6px}.insec h3:first-child{margin-top:4px}

/* tips panel */
.tips{margin-top:12px;border:1px solid #1b2530;border-radius:11px;background:#0e141b;overflow:hidden}
.tips>summary{cursor:pointer;padding:10px 13px;font-size:12px;color:#9fb6cc;font-weight:600;list-style:none}
.tips>summary::-webkit-details-marker{display:none}
.tips ul{margin:0;padding:4px 22px 13px;font-size:12.5px;color:#aeb9c6;line-height:1.7}

/* ===== PHASE 3 FEATURE COMPONENTS (drag-file · toasts · action bar) ===== */
/* toast notifications (var(--token, fallback) so they work pre/post design-system) */
.edtoasts{position:fixed;right:18px;bottom:18px;z-index:1000;display:flex;flex-direction:column;gap:9px;max-width:380px}
.edtoast{display:flex;gap:10px;align-items:flex-start;background:var(--bg-elevated,#141c26);
  border:1px solid var(--border-strong,#27313e);border-radius:12px;padding:11px 12px;
  box-shadow:var(--shadow-pop,0 12px 30px rgba(0,0,0,.45));color:var(--text-primary,#e7eef6);
  font-size:13px;opacity:0;transform:translateY(8px);transition:opacity .22s,transform .22s}
.edtoast.in{opacity:1;transform:none}.edtoast.out{opacity:0;transform:translateY(8px)}
.edtoast .etic{flex:none;width:18px;text-align:center;font-size:13px;line-height:1.5;color:var(--text-secondary,#9fb0c2)}
.edtoast.success .etic{color:var(--success,#54c98a)}.edtoast.error .etic{color:var(--danger,#ef6f5a)}
.edtoast.progress .etic{color:var(--accent,#3b82f6)}
.edtoast .etmain{min-width:0;flex:1}.edtoast .ettxt{display:block;line-height:1.4}
.edtoast .etbar{height:4px;border-radius:3px;background:rgba(255,255,255,.08);margin-top:7px;overflow:hidden}
.edtoast .etbar>i{display:block;height:100%;width:0;background:var(--accent,#3b82f6);border-radius:3px;transition:width .15s}
.edtoast .etdet{margin-top:6px}
.edtoast .etdet summary{cursor:pointer;color:var(--text-muted,#8593a3);font-size:11.5px}
.edtoast .etdet pre{white-space:pre-wrap;word-break:break-word;font-size:11px;color:var(--text-muted,#8593a3);margin:5px 0 0}
.edtoast .etx{flex:none;background:none;border:0;color:var(--text-muted,#6f7d8c);font-size:16px;line-height:1;cursor:pointer;padding:0 2px}
.edtoast .etx:hover{color:var(--text-primary,#e7eef6)}
/* spinner + busy */
.spin{display:inline-block;width:12px;height:12px;border:2px solid rgba(255,255,255,.25);border-top-color:#fff;
  border-radius:50%;animation:spin .7s linear infinite;vertical-align:-1px}
@keyframes spin{to{transform:rotate(360deg)}}
.isbusy{opacity:.75;cursor:progress}
/* file-drop states */
.scrow.filedrop{outline:2px dashed var(--accent,#3b82f6);outline-offset:-3px;background:var(--accent-soft,#15233a)}
.edpvdrop{position:absolute;inset:0;display:none;align-items:center;justify-content:center;border-radius:14px;
  background:rgba(9,13,19,.80);border:2px dashed var(--accent,#5fa8ff);z-index:8;pointer-events:none}
.edpreview.filedrop .edpvdrop{display:flex}
.edpvdropc{text-align:center;color:#dbe7f5;font-size:14px;font-weight:600;display:flex;flex-direction:column;gap:8px;align-items:center;padding:0 20px}
.edpvdropic{font-size:30px;color:var(--accent,#5fa8ff)}
/* inspector dropzone */
.edrop{border:1.5px dashed var(--border-strong,#2d3a48);border-radius:12px;padding:15px 12px;text-align:center;cursor:pointer;
  background:var(--bg-app,#0c1117);transition:border-color .15s,background .15s;margin-bottom:4px}
.edrop:hover,.edrop:focus{border-color:var(--accent,#3b82f6);background:var(--accent-soft,#111b2b);outline:none}
.edrop.filedrop{border-color:var(--accent,#3b82f6);border-style:solid;background:var(--accent-soft,#15233a)}
.edropic{font-size:22px;color:var(--text-secondary,#7c8aa0);line-height:1}
.edropt{font-size:13px;font-weight:600;color:var(--text-primary,#d7e2ee);margin-top:5px}
.edrops{font-size:11px;color:var(--text-muted,#7a8696);margin-top:3px}
/* easy action bar */
.edactions{display:flex;flex-wrap:wrap;gap:9px;margin:2px 0 14px}
.edag{display:flex;align-items:center;gap:5px;background:var(--bg-app,#0c1117);border:1px solid var(--border-soft,#1e2731);
  border-radius:10px;padding:5px 7px 5px 9px}
.edagl{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted,#6f7d8c);font-weight:700;margin-right:3px}
.qbtn{display:inline-flex;align-items:center;gap:5px;background:var(--bg-elevated,#1a2230);border:1px solid var(--border-soft,#28333f);
  color:var(--text-primary,#d6e2ee);border-radius:8px;height:30px;padding:0 10px;font-size:12.5px;cursor:pointer;transition:background .12s,border-color .12s}
.qbtn:hover{background:var(--bg-hover,#222e3c);border-color:var(--accent,#3b82f6)}
.qbtn:disabled{opacity:.4;cursor:not-allowed}
.qbtn .qic{font-style:normal;font-size:13px;color:var(--text-secondary,#9fb0c2);line-height:1}
.qbtn.danger{color:var(--danger,#ef8a78);border-color:transparent}
.qbtn.danger:hover{background:var(--danger-soft,#2a1714);border-color:var(--danger,#7d3a30)}
.qbtn.danger .qic{color:var(--danger,#ef8a78)}
/* inspector title + danger ghost */
.insptitle{display:flex;align-items:baseline;gap:9px;margin:0 0 12px}
.insptitle h2{font-size:16px;font-weight:650;margin:0;color:var(--text-primary,#eaf1f8);letter-spacing:-.01em}
.insprole{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted,#7c8aa0);font-weight:600}
.ghost.danger{color:var(--danger,#ef8a78);border-color:var(--danger,#5e3630)}
.ghost.danger:hover{background:var(--danger-soft,#2a1714)}

/* ============ EDITING STUDIO DESIGN SYSTEM (synthesized) ============ */
/* ============================================================================
   EDITING STUDIO — DESIGN SYSTEM (SHIPPING SPEC)
   Synthesis of three proposals. Calm graphite-navy dark · ONE restrained blue.
   Pure CSS · additive only · reuses exact class names · production values.
   ========================================================================= */

/* ===== TOKENS ===== */
:root{
  /* --- Surfaces (cool graphite/navy, layered by one believable elevation ramp) --- */
  --bg-app:        #0a0d12;   /* deepest app canvas (root) */
  --bg-panel:      #10151c;   /* panel bodies (scenes/preview/inspector/timeline) */
  --bg-elevated:   #161d27;   /* raised: cards, inputs, clips, sticky labels, sections */
  --bg-hover:      #1a2330;   /* hover wash on rows/controls */
  --bg-selected:   #152334;   /* selected row/clip tint (navy lift) */

  /* --- Borders (cool-gray hairlines, two weights) --- */
  --border-soft:   #1e2731;   /* default separators, resting panel edges */
  --border-strong: #26303c;   /* interactive/control edges, internal dividers */
  --border-hover:  #33414f;   /* edge on control hover (one step up from strong) */

  /* --- Text (3-step ramp) --- */
  --text-primary:   #e7eef6;  /* titles, active values */
  --text-secondary: #9fb0c2;  /* labels, secondary copy */
  --text-muted:     #6f7d8c;  /* meta, hints, placeholders, disabled */

  /* --- Accent (single restrained blue — active + primary ONLY) --- */
  --accent:        #3b82f6;   /* primary actions, active/selected state */
  --accent-strong: #2f70e0;   /* hover / pressed */
  --accent-soft:   rgba(59,130,246,0.12);   /* low-chroma tint bg behind active items */
  --accent-border: #2f5d8a;   /* selected/active border on navy surfaces */
  --focus-ring:    rgba(59,130,246,0.50);   /* keyboard focus outer glow */

  /* --- Status (each color carries ONE meaning) --- */
  --success:       #62cc83;   /* success / card present only */
  --success-soft:  rgba(98,204,131,0.12);   /* success tint bg */
  --success-text:  #bfe9cc;   /* legible success label on dark */
  --warning:       #e2b552;   /* warnings / unsaved / edited only (amber) */
  --warning-soft:  rgba(226,181,82,0.12);   /* warning tint bg */
  --warning-text:  #f0dca6;   /* legible warning label on dark */
  --danger:        #e2864e;   /* destructive only */
  --danger-soft:   rgba(226,134,78,0.12);   /* danger tint bg */
  --danger-text:   #f2c3a0;   /* legible danger label on dark */

  /* --- Radii --- */
  --radius-sm:   6px;     /* chips, inputs, small clips, transport btns */
  --radius-md:   9px;     /* buttons, rows, controls, sections */
  --radius-lg:  13px;     /* panels */
  --radius-pill: 999px;   /* status pills, badges, range track */

  /* --- Spacing rhythm (4 / 8 / 12 / 16 / 24 / 32) --- */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;

  /* --- Depth (soft, cool, real lift — never glossy) --- */
  --shadow-panel: 0 1px 0 0 rgba(255,255,255,0.035) inset,   /* 1px light top edge */
                  0 8px 24px -12px rgba(0,0,0,0.60),
                  0 2px 6px -3px rgba(0,0,0,0.50);
  --shadow-pop:   0 1px 0 0 rgba(255,255,255,0.05) inset,
                  0 16px 40px -12px rgba(0,0,0,0.70),
                  0 4px 12px -4px rgba(0,0,0,0.55);

  /* --- Control sizing --- */
  --btn-h: 36px;   /* buttons (primary/secondary/quiet) */
  --ctl-h: 30px;   /* compact controls: chips, inputs, select, transport */

  /* --- Type --- */
  --font-ui: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI",
             "Inter", Roboto, Helvetica, Arial, sans-serif,
             "Apple Color Emoji", "Segoe UI Emoji";
  --font-mono: "SF Mono", "JetBrains Mono", "Roboto Mono", ui-monospace,
               Menlo, Consolas, monospace;

  /* --- Timeline clip LOUDNESS HIERARCHY -----------------------------------
     visual = strongest (saturated navy-blue) > card = secondary (green)
     > voice = calm steel > music = distinct warm amber > caption = lightest
     (low-sat violet). Each: -bg fill, -bd border/left-bar, -tx label color.
     Selection is an accent RING layered over the clip — never recolors it. */
  --clip-visual-bg:  #1b3147;  --clip-visual-bd:  #356192;  --clip-visual-tx:  #cfe3f7;
  --clip-card-bg:    #1e4030;  --clip-card-bd:    #2f7a54;  --clip-card-tx:    #cdeed9;
  --clip-caption-bg: #2a2440;  --clip-caption-bd: #4a3f73;  --clip-caption-tx: #d6cef0;
  --clip-music-bg:   #3a2a1e;  --clip-music-bd:   #6e4f33;  --clip-music-tx:   #ecd4ba;
  --clip-voice-bg:   #23303a;  --clip-voice-bd:   #3c5263;  --clip-voice-tx:   #c6d6e2;
  --playhead:        #ff5a5a;

  /* --- Track lane backdrop (recedes behind clips) --- */
  --tk-lane:        #0d1218;
  --tk-tick:        #1e2731;   /* ruler tick on lane (== border-soft) */
}

/* Box model + smoothing baseline (additive, safe) */
#ed-root *,#ed-root *::before,#ed-root *::after{ box-sizing:border-box; }

/* ===== COMPONENTS ===== */

/* ---------------------------------------------------------------------------
   ROOT
   ------------------------------------------------------------------------- */
#ed-root{
  background:
    radial-gradient(1200px 600px at 50% -10%, #0d1119 0%, transparent 60%),
    var(--bg-app);
  color:var(--text-primary);
  font-family:var(--font-ui);
  font-size:13px;
  line-height:1.45;
  letter-spacing:0.1px;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility;
  padding:clamp(var(--space-3), 1.6vw, var(--space-6));
  min-height:100vh;
}

/* ---------------------------------------------------------------------------
   HEADER  .edhead / .edhl / .edhr
   ------------------------------------------------------------------------- */
.edhead{
  position:sticky;
  top:clamp(var(--space-3), 1.6vw, var(--space-6));
  z-index:50;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:var(--space-4);
  padding:var(--space-3) var(--space-4);
  margin-bottom:var(--space-4);
  background:color-mix(in srgb, var(--bg-panel) 88%, transparent);
  -webkit-backdrop-filter:blur(12px) saturate(120%);
  backdrop-filter:blur(12px) saturate(120%);
  border:1px solid var(--border-soft);
  border-radius:var(--radius-lg);
  box-shadow:var(--shadow-panel);
}
.edhl,.edhr{ display:flex; align-items:center; gap:var(--space-2); min-width:0; }
.edhl{ flex:1 1 auto; min-width:0; }
.edhl h1{
  margin:0 var(--space-1);
  font-size:15px;
  font-weight:650;
  letter-spacing:0.1px;
  color:var(--text-primary);
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
  max-width:42ch;
}

/* ---------------------------------------------------------------------------
   CHIPS / STATUS PILLS  .edchip (+ .ok .warn .bad)  ·  #edunsaved
   ------------------------------------------------------------------------- */
.edchip{
  display:inline-flex;
  align-items:center;
  gap:6px;
  height:24px;
  padding:0 10px;
  font-size:11.5px;
  font-weight:550;
  line-height:1;
  letter-spacing:0.2px;
  color:var(--text-secondary);
  background:var(--bg-elevated);
  border:1px solid var(--border-soft);
  border-radius:var(--radius-pill);
  white-space:nowrap;
  text-decoration:none;
  transition:background .15s ease, border-color .15s ease, color .15s ease;
}
a.edchip:hover{
  color:var(--text-primary);
  background:var(--bg-hover);
  border-color:var(--border-strong);
}
/* leading status dot for state variants */
.edchip.ok::before,.edchip.warn::before,.edchip.bad::before{
  content:""; width:6px; height:6px; border-radius:50%; flex:none;
}
.edchip.ok  { color:var(--success-text); background:var(--success-soft); border-color:rgba(98,204,131,0.28); }
.edchip.ok::before  { background:var(--success); box-shadow:0 0 0 2px rgba(98,204,131,0.18); }
.edchip.warn{ color:var(--warning-text); background:var(--warning-soft); border-color:rgba(226,181,82,0.28); }
.edchip.warn::before{ background:var(--warning); box-shadow:0 0 0 2px rgba(226,181,82,0.18); }
.edchip.bad { color:var(--danger-text);  background:var(--danger-soft);  border-color:rgba(226,134,78,0.30); }
.edchip.bad::before { background:var(--danger);  box-shadow:0 0 0 2px rgba(226,134,78,0.18); }

/* unsaved indicator — quiet amber, present only when dirty */
#edunsaved{
  color:var(--warning-text);
  background:var(--warning-soft);
  border-color:rgba(226,181,82,0.30);
  font-weight:600;
}
#edunsaved::before{
  content:""; width:6px; height:6px; border-radius:50%; flex:none;
  background:var(--warning); box-shadow:0 0 0 2px rgba(226,181,82,0.18);
}

/* ---------------------------------------------------------------------------
   WARNING STRIP  .warnstrip (#edwarn) / .warnb
   ------------------------------------------------------------------------- */
.warnstrip{
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  gap:var(--space-2);
  padding:var(--space-2) var(--space-3);
  margin-bottom:var(--space-4);
  background:linear-gradient(0deg, var(--warning-soft), var(--warning-soft)), var(--bg-panel);
  border:1px solid rgba(226,181,82,0.26);
  border-radius:var(--radius-md);
  box-shadow:var(--shadow-panel);
}
.warnb{
  display:inline-flex;
  align-items:center;
  gap:6px;
  height:24px;
  padding:0 10px;
  font-size:11.5px;
  font-weight:550;
  color:var(--warning-text);
  background:rgba(226,181,82,0.10);
  border:1px solid rgba(226,181,82,0.30);
  border-radius:var(--radius-pill);
}
.warnb::before{
  content:"!"; display:grid; place-items:center; flex:none;
  width:14px; height:14px; border-radius:50%;
  font-size:10px; font-weight:800; color:#1c1810;
  background:var(--warning);
}

/* ---------------------------------------------------------------------------
   WORKSPACE GRID  .edwrap
   ------------------------------------------------------------------------- */
.edwrap{
  display:grid;
  grid-template-columns:minmax(248px,300px) minmax(0,1fr) minmax(300px,360px);
  grid-template-rows:minmax(0,1fr) auto;
  grid-template-areas:
    "scenes preview inspector"
    "timeline timeline timeline";
  gap:var(--space-4);
  align-items:stretch;
}
.edscenes   { grid-area:scenes; }
.edpreview  { grid-area:preview; }
.edinspector{ grid-area:inspector; }
.edtimeline { grid-area:timeline; }

/* Shared panel surface — one elevation, hairline edge, gentle real lift */
.edscenes,.edpreview,.edinspector,.edtimeline{
  background:var(--bg-panel);
  border:1px solid var(--border-soft);
  border-radius:var(--radius-lg);
  box-shadow:var(--shadow-panel);
  min-height:0;
}

/* ---------------------------------------------------------------------------
   SCENES PANEL  .edscenes (#edscenes) / .scrow ...
   ------------------------------------------------------------------------- */
.edscenes{
  display:flex;
  flex-direction:column;
  padding:var(--space-2);
  gap:var(--space-1);
  overflow-y:auto;
  max-height:calc(100vh - 240px);
}

/* premium media-card row */
.scrow{
  position:relative;
  display:grid;
  grid-template-columns:16px 90px 1fr;
  align-items:center;
  gap:var(--space-3);
  padding:var(--space-2);
  border:1px solid transparent;
  border-radius:var(--radius-md);
  background:transparent;
  cursor:pointer;
  user-select:none;
  transition:background .14s ease, border-color .14s ease, box-shadow .14s ease, transform .06s ease;
}
.scrow + .scrow{ margin-top:2px; }
.scrow:hover{ background:var(--bg-hover); border-color:var(--border-soft); }
.scrow.sel{
  background:var(--bg-selected);
  border-color:var(--accent-border);
  box-shadow:0 0 0 1px rgba(59,130,246,0.18),
             0 1px 0 0 rgba(255,255,255,0.04) inset,
             0 6px 18px -10px rgba(0,0,0,0.6);
}
/* active accent spine on selected row */
.scrow.sel::before{
  content:""; position:absolute; left:5px; top:10px; bottom:10px; width:3px;
  border-radius:var(--radius-pill); background:var(--accent);
  box-shadow:0 0 8px rgba(59,130,246,0.45);
}
.scrow.sel .sctt{ color:var(--text-primary); }

/* drag handle */
.scgrip{
  display:grid; place-items:center;
  width:16px; height:100%;
  color:var(--text-muted);
  cursor:grab;
  border-radius:var(--radius-sm);
  opacity:0;
  transition:opacity .14s ease, color .14s ease;
}
.scgrip::before{
  content:""; width:8px; height:14px;
  background:radial-gradient(currentColor 1px, transparent 1.4px) 0 0 / 4px 4px;
  opacity:.8;
}
.scrow:hover .scgrip,.scrow.sel .scgrip{ opacity:1; }
.scgrip:hover{ color:var(--text-secondary); }
.scgrip:active{ cursor:grabbing; }

/* thumbnail */
.scrow img{
  width:90px; height:51px;
  object-fit:cover;
  border-radius:var(--radius-sm);
  background:var(--bg-app);
  border:1px solid var(--border-soft);
  box-shadow:0 1px 3px rgba(0,0,0,0.4);
}

.scmeta{ min-width:0; display:flex; flex-direction:column; gap:5px; }
.sctt{
  font-size:12.5px;
  font-weight:580;
  color:var(--text-primary);
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.scrow:not(.sel) .sctt{ color:var(--text-secondary); }
.scb{ display:flex; flex-wrap:wrap; gap:5px; }

/* drag + drop affordances */
.scrow.dragging{
  opacity:.85;
  background:var(--bg-elevated);
  border-color:var(--border-strong);
  box-shadow:var(--shadow-pop);
  transform:scale(1.01);
  cursor:grabbing;
}
.scrow.dropabove,.scrow.dropbelow{ background:var(--bg-hover); }
.scrow.dropabove::after,.scrow.dropbelow::after{
  content:""; position:absolute; left:8px; right:8px; height:2px;
  background:var(--accent); border-radius:var(--radius-pill);
  box-shadow:0 0 8px rgba(59,130,246,0.6);
}
.scrow.dropabove::after{ top:-2px; }
.scrow.dropbelow::after{ bottom:-2px; }

/* scene badges (.ai .card .skip .foot .edited) */
.scb > *{
  display:inline-flex; align-items:center;
  height:18px; padding:0 7px;
  font-size:10px; font-weight:650; letter-spacing:0.4px; text-transform:uppercase;
  border-radius:var(--radius-sm);
  color:var(--text-muted);
  background:var(--bg-elevated);
  border:1px solid var(--border-soft);
  white-space:nowrap;
}
.scb .ai    { color:#bcd2f5; background:var(--accent-soft);  border-color:rgba(59,130,246,0.30); }
.scb .card  { color:var(--success-text); background:var(--success-soft); border-color:rgba(98,204,131,0.28); }
.scb .edited{ color:var(--warning-text); background:var(--warning-soft); border-color:rgba(226,181,82,0.28); }
.scb .skip  { color:var(--text-muted); background:transparent; border-style:dashed; text-decoration:line-through; text-decoration-color:var(--text-muted); }
.scb .foot  { color:var(--clip-voice-tx); background:var(--bg-elevated); border-color:var(--border-strong); }

/* ---------------------------------------------------------------------------
   PREVIEW PANEL  .edpreview / video#edvid / .edtransport / .audiobar
   — the visual focal point
   ------------------------------------------------------------------------- */
.edpreview{
  display:flex;
  flex-direction:column;
  gap:var(--space-3);
  padding:var(--space-4);
  min-width:0;
}
video#edvid{
  flex:1 1 auto;
  width:100%;
  aspect-ratio:16 / 9;
  min-height:0;
  object-fit:contain;
  background:radial-gradient(120% 120% at 50% 0%, #0e131a 0%, #05070a 100%);
  border-radius:var(--radius-md);
  /* recessed editing-canvas frame: inner hairline + inset bezel + soft outer lift */
  box-shadow:
    inset 0 0 0 1px rgba(255,255,255,0.04),
    inset 0 0 0 6px rgba(8,11,15,0.55),
    0 18px 50px -20px rgba(0,0,0,0.82);
  outline:1px solid var(--border-strong);
  outline-offset:-1px;
}

/* transport */
.edtransport{
  display:flex;
  align-items:center;
  gap:var(--space-2);
  padding:6px;
  background:var(--bg-elevated);
  border:1px solid var(--border-soft);
  border-radius:var(--radius-md);
}
.tbtn{
  display:inline-flex; align-items:center; justify-content:center;
  height:32px; min-width:32px; padding:0 9px;
  color:var(--text-secondary);
  background:transparent;
  border:1px solid transparent;
  border-radius:var(--radius-sm);
  cursor:pointer;
  font-size:13px;
  transition:background .14s ease, color .14s ease, border-color .14s ease, transform .06s ease;
}
.tbtn:hover{ background:var(--bg-hover); color:var(--text-primary); border-color:var(--border-soft); }
.tbtn:active{ background:var(--bg-selected); transform:translateY(0.5px); }
.tbtn:focus-visible{ outline:none; box-shadow:0 0 0 2px var(--bg-panel), 0 0 0 4px var(--focus-ring); }
.tbtn.play{
  width:40px; height:40px; min-width:40px;
  color:#fff;
  background:var(--accent);
  border-color:transparent;
  box-shadow:0 1px 0 rgba(255,255,255,0.18) inset, 0 2px 8px -2px rgba(59,130,246,0.6);
}
.tbtn.play:hover{ background:var(--accent-strong); color:#fff; }
.tbtn.play:active{ transform:translateY(0.5px); }
.ttime{
  margin-left:auto;
  padding:0 10px;
  font-family:var(--font-mono);
  font-size:12px;
  font-variant-numeric:tabular-nums;
  letter-spacing:0.3px;
  color:var(--text-secondary);
}
.ttime span,.ttime b,.ttime strong{ color:var(--text-muted); font-weight:inherit; }

/* preview meta chips line */
#edpchips{
  display:flex; flex-wrap:wrap; align-items:center; gap:6px;
  font-size:11.5px; color:var(--text-muted);
}
#edpchips > span{
  padding:3px 8px;
  background:var(--bg-elevated);
  border:1px solid var(--border-soft);
  border-radius:var(--radius-pill);
  color:var(--text-secondary);
}

/* audio / look bar */
.audiobar{
  display:flex; flex-wrap:wrap; align-items:center; gap:var(--space-4);
  padding:var(--space-2) var(--space-3);
  background:var(--bg-elevated);
  border:1px solid var(--border-soft);
  border-radius:var(--radius-md);
  font-size:12px;
  color:var(--text-secondary);
}
.audiobar label{ display:inline-flex; align-items:center; gap:7px; cursor:pointer; user-select:none; }

/* ---------------------------------------------------------------------------
   FORM CONTROLS — shared, premium, calm (custom checkbox/range; styled select/text)
   Scoped to editor regions to avoid leaking global input styling.
   ------------------------------------------------------------------------- */
.audiobar input[type="checkbox"],
.insp input[type="checkbox"],
#cardfields input[type="checkbox"]{
  appearance:none; -webkit-appearance:none;
  width:16px; height:16px; margin:0; flex:none;
  border:1px solid var(--border-strong);
  border-radius:5px;
  background:var(--bg-app);
  cursor:pointer;
  position:relative;
  transition:background .14s ease, border-color .14s ease, box-shadow .14s ease;
}
.audiobar input[type="checkbox"]:hover,
.insp input[type="checkbox"]:hover,
#cardfields input[type="checkbox"]:hover{ border-color:var(--accent-border); }
.audiobar input[type="checkbox"]:checked,
.insp input[type="checkbox"]:checked,
#cardfields input[type="checkbox"]:checked{ background:var(--accent); border-color:var(--accent); }
.audiobar input[type="checkbox"]:checked::after,
.insp input[type="checkbox"]:checked::after,
#cardfields input[type="checkbox"]:checked::after{
  content:""; position:absolute; left:5px; top:2px; width:4px; height:8px;
  border:solid #fff; border-width:0 2px 2px 0; transform:rotate(43deg);
}
.audiobar input[type="checkbox"]:focus-visible,
.insp input[type="checkbox"]:focus-visible,
#cardfields input[type="checkbox"]:focus-visible{ outline:none; box-shadow:0 0 0 3px var(--focus-ring); }

.audiobar input[type="range"],
.insp input[type="range"]{
  appearance:none; -webkit-appearance:none;
  height:4px; width:120px; margin:0;
  background:var(--border-strong);
  border-radius:var(--radius-pill);
  cursor:pointer;
}
.audiobar input[type="range"]::-webkit-slider-thumb,
.insp input[type="range"]::-webkit-slider-thumb{
  -webkit-appearance:none; appearance:none;
  width:14px; height:14px; border-radius:50%;
  background:var(--text-primary);
  border:2px solid var(--bg-panel);
  box-shadow:0 1px 3px rgba(0,0,0,0.5);
  transition:background .14s ease, transform .08s ease;
}
.audiobar input[type="range"]::-webkit-slider-thumb:hover,
.insp input[type="range"]::-webkit-slider-thumb:hover{ background:var(--accent); }
.audiobar input[type="range"]:active::-webkit-slider-thumb,
.insp input[type="range"]:active::-webkit-slider-thumb{ transform:scale(1.08); background:var(--accent); }
.audiobar input[type="range"]::-moz-range-thumb,
.insp input[type="range"]::-moz-range-thumb{
  width:14px; height:14px; border-radius:50%; border:2px solid var(--bg-panel);
  background:var(--text-primary); box-shadow:0 1px 3px rgba(0,0,0,0.5);
}
.audiobar input[type="range"]:focus-visible,
.insp input[type="range"]:focus-visible{ outline:none; box-shadow:0 0 0 3px var(--focus-ring); }

/* select + text/number/textarea (scoped) */
.audiobar select,
.insp input[type="text"],.insp input[type="number"],.insp input:not([type]),
.insp textarea,.insp select,
#cardfields input,#cardfields textarea,#cardfields select,
.fld input,.fld textarea,.fld select{
  appearance:none; -webkit-appearance:none;
  height:var(--ctl-h);
  padding:0 var(--space-3);
  font-family:var(--font-ui);
  font-size:12.5px;
  color:var(--text-primary);
  background:var(--bg-app);
  border:1px solid var(--border-strong);
  border-radius:var(--radius-sm);
  transition:border-color .14s ease, box-shadow .14s ease, background .14s ease;
}
.audiobar select,.insp select,#cardfields select,.fld select{
  padding-right:28px;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'><path d='M2.5 4.5L6 8l3.5-3.5' fill='none' stroke='%239fb0c2' stroke-width='1.4' stroke-linecap='round' stroke-linejoin='round'/></svg>");
  background-repeat:no-repeat;
  background-position:right 9px center;
  cursor:pointer;
}
.insp textarea,#cardfields textarea,.fld textarea{
  height:auto; min-height:72px; padding:var(--space-2) var(--space-3); line-height:1.5; resize:vertical;
}
.fld input,.fld select,.fld textarea,
#cardfields input,#cardfields select,#cardfields textarea{ width:100%; }
.audiobar select:hover,
.insp input:hover,.insp textarea:hover,.insp select:hover,
#cardfields input:hover,#cardfields textarea:hover,#cardfields select:hover,
.fld input:hover,.fld textarea:hover,.fld select:hover{ border-color:var(--border-hover); }
.audiobar select:focus,.audiobar select:focus-visible,
.insp input:focus,.insp textarea:focus,.insp select:focus,
#cardfields input:focus,#cardfields textarea:focus,#cardfields select:focus,
.fld input:focus,.fld textarea:focus,.fld select:focus{
  outline:none; border-color:var(--accent); background:var(--bg-elevated);
  box-shadow:0 0 0 3px var(--focus-ring);
}
.insp ::placeholder,#cardfields ::placeholder,.fld ::placeholder{ color:var(--text-muted); }

/* ---------------------------------------------------------------------------
   INSPECTOR PANEL  .edinspector (#edinsp) / .insec / .inhd / .car / .inbody
   ------------------------------------------------------------------------- */
.edinspector{
  display:flex;
  flex-direction:column;
  padding:var(--space-2);
  gap:var(--space-2);
  overflow-y:auto;
  max-height:calc(100vh - 240px);
}

.insec{
  background:var(--bg-elevated);
  border:1px solid var(--border-soft);
  border-radius:var(--radius-md);
  overflow:hidden;
}
.inhd{
  display:flex; align-items:center; gap:var(--space-2);
  padding:var(--space-3);
  cursor:pointer;
  user-select:none;
  font-size:12.5px; font-weight:600; letter-spacing:0.2px;
  color:var(--text-primary);
  background:transparent;
  transition:background .14s ease;
}
.inhd:hover{ background:var(--bg-hover); }
/* caret: CSS triangle, points down when open, right when collapsed.
   Covers both [open]/.open and .collapsed/[aria-collapsed] toggling schemes. */
.car{
  margin-left:auto; flex:none;
  width:0; height:0;
  border-left:5px solid transparent;
  border-right:5px solid transparent;
  border-top:6px solid var(--text-muted);
  transition:transform .18s ease, border-top-color .14s ease;
}
.inhd:hover .car{ border-top-color:var(--text-secondary); }
.insec.collapsed .car,
.insec[aria-collapsed="true"] .car,
.insec:not([open]):not(.open) .car{ transform:rotate(-90deg); }
.insec[open] .car,.insec.open .car{ transform:rotate(0deg); }

.inbody{
  padding:var(--space-3);
  border-top:1px solid var(--border-soft);
  display:flex; flex-direction:column; gap:var(--space-3);
  font-size:12px; color:var(--text-secondary);
}
.insec.collapsed .inbody,
.insec[aria-collapsed="true"] .inbody{ display:none; }

/* danger section (e.g. destructive zone) */
.insec.danger{ border-color:rgba(226,134,78,0.28); }
.insec.danger .inhd{ color:var(--danger-text); }
.insec.danger .inbody{ border-top-color:rgba(226,134,78,0.22); }

/* inspector content primitives */
.insp h3{
  margin:0 0 var(--space-1);
  font-size:11px; font-weight:650; text-transform:uppercase; letter-spacing:0.5px;
  color:var(--text-muted);
}
.narr{ font-size:12.5px; line-height:1.55; color:var(--text-secondary); }
.kw{ display:flex; flex-wrap:wrap; gap:6px; }
.kw > *{
  display:inline-flex; align-items:center;
  padding:3px 9px;
  font-size:11px; color:#bcd2f5;
  background:var(--accent-soft);
  border:1px solid rgba(59,130,246,0.26);
  border-radius:var(--radius-pill);
}
.fld{ display:flex; flex-direction:column; gap:6px; }
.fld > label,.fld > span:first-child{
  font-size:11px; font-weight:560; letter-spacing:0.2px; color:var(--text-muted);
}
.dots{ display:inline-flex; gap:5px; align-items:center; }
.dots > *{ width:7px; height:7px; border-radius:50%; background:var(--border-strong); }
.dots > .on{ background:var(--accent); box-shadow:0 0 6px rgba(59,130,246,0.5); }

.tips{
  font-size:11.5px;
  color:var(--text-muted);
  border-top:1px solid var(--border-soft);
  padding-top:var(--space-2);
}
.tips summary{
  cursor:pointer; color:var(--text-secondary); font-weight:550;
  list-style:none; user-select:none;
}
.tips summary::-webkit-details-marker{ display:none; }
.tips summary::before{ content:"\25B8 "; color:var(--text-muted); }
.tips[open] summary::before{ content:"\25BE "; }
.tips[open] summary{ margin-bottom:var(--space-2); color:var(--text-primary); }

/* ---------------------------------------------------------------------------
   BUTTON HIERARCHY
   PRIMARY  #edexport  ·  SECONDARY .edbtn  ·  QUIET .ghost  ·  .danger variant
   ------------------------------------------------------------------------- */
.edbtn,.ghost,#edexport{
  display:inline-flex; align-items:center; justify-content:center; gap:7px;
  height:var(--btn-h);
  padding:0 var(--space-4);
  font-family:var(--font-ui);
  font-size:12.5px; font-weight:600; letter-spacing:0.2px; line-height:1;
  border-radius:var(--radius-md);
  border:1px solid transparent;
  cursor:pointer;
  white-space:nowrap;
  user-select:none;
  text-decoration:none;
  transition:background .14s ease, border-color .14s ease, color .14s ease,
             box-shadow .14s ease, transform .06s ease;
}
.edbtn:focus-visible,.ghost:focus-visible,#edexport:focus-visible{
  outline:none; box-shadow:0 0 0 2px var(--bg-panel), 0 0 0 4px var(--focus-ring);
}
.edbtn:active,.ghost:active,#edexport:active{ transform:translateY(0.5px); }
.edbtn:disabled,.ghost:disabled,#edexport:disabled,
.edbtn[disabled],.ghost[disabled],#edexport[disabled]{
  opacity:.45; cursor:not-allowed; transform:none; box-shadow:none; pointer-events:none;
}

/* SECONDARY — neutral elevated surface */
.edbtn{
  color:var(--text-primary);
  background:var(--bg-elevated);
  border-color:var(--border-strong);
  box-shadow:0 1px 0 rgba(255,255,255,0.035) inset, 0 2px 5px -3px rgba(0,0,0,0.5);
}
.edbtn:hover{ background:var(--bg-hover); border-color:var(--border-hover); }
.edbtn:active{ background:var(--bg-selected); }

/* QUIET — text-weight, no chrome until hover (#edundo, #edresetall) */
.ghost{
  color:var(--text-secondary);
  background:transparent;
  border-color:transparent;
}
.ghost:hover{ background:var(--bg-hover); color:var(--text-primary); border-color:var(--border-soft); }
.ghost:active{ background:var(--bg-selected); }

/* PRIMARY — the single saturated blue action */
#edexport{
  color:#fff;
  background:var(--accent);
  border-color:var(--accent);
  box-shadow:0 1px 0 rgba(255,255,255,0.18) inset, 0 4px 14px -4px rgba(59,130,246,0.55);
}
#edexport:hover{ background:var(--accent-strong); border-color:var(--accent-strong);
  box-shadow:0 1px 0 rgba(255,255,255,0.18) inset, 0 6px 18px -4px rgba(59,130,246,0.6); }
#edexport:active{ background:var(--accent-strong); box-shadow:0 1px 6px -2px rgba(59,130,246,0.5); }
#edexport:focus-visible{ box-shadow:0 0 0 2px var(--bg-panel), 0 0 0 4px var(--focus-ring),
  0 4px 14px -4px rgba(59,130,246,0.5); }

/* DANGER variant — destructive only; outline that fills on hover.
   Apply .danger to .edbtn or .ghost (e.g. a destructive reset/delete). */
.edbtn.danger{
  color:var(--danger-text);
  background:var(--danger-soft);
  border-color:rgba(226,134,78,0.36);
  box-shadow:none;
}
.edbtn.danger:hover{ color:#fff; background:var(--danger); border-color:var(--danger); }
.edbtn.danger:focus-visible{ box-shadow:0 0 0 2px var(--bg-panel), 0 0 0 4px rgba(226,134,78,0.45); }
.ghost.danger{ color:var(--danger-text); background:transparent; border-color:transparent; }
.ghost.danger:hover{ color:#fff; background:var(--danger); border-color:var(--danger); }
.ghost.danger:focus-visible{ box-shadow:0 0 0 2px var(--bg-panel), 0 0 0 4px rgba(226,134,78,0.45); }
/* #edresetall is a quiet header reset — nudge toward danger affordance on hover */
#edresetall:hover{ color:var(--danger-text); }

/* ---------------------------------------------------------------------------
   TIMELINE  .edtimeline / .tlhead / .tlt / .zbtn / .tlbody (#edtlb)
             .tk / .tklab / .tkrow / .blk / .playhead
   ------------------------------------------------------------------------- */
.edtimeline{
  display:flex;
  flex-direction:column;
  min-height:200px;
  padding:0;
  overflow:hidden;
}
.tlhead{
  display:flex; align-items:center; gap:var(--space-3);
  padding:var(--space-2) var(--space-4);
  border-bottom:1px solid var(--border-soft);
  background:var(--bg-panel);
}
.tlt{
  font-size:11px; font-weight:650; text-transform:uppercase; letter-spacing:0.5px;
  color:var(--text-muted);
}
.zbtn{
  display:inline-flex; align-items:center; justify-content:center;
  width:28px; height:28px; padding:0;
  font-size:13px; line-height:1;
  color:var(--text-secondary);
  background:var(--bg-elevated);
  border:1px solid var(--border-strong);
  border-radius:var(--radius-sm);
  cursor:pointer;
  transition:background .14s ease, color .14s ease, border-color .14s ease, transform .06s ease;
}
.zbtn:first-of-type{ margin-left:auto; }
.zbtn:hover{ background:var(--bg-hover); color:var(--text-primary); border-color:var(--border-hover); }
.zbtn:active{ background:var(--bg-selected); transform:translateY(0.5px); }
.zbtn:focus-visible{ outline:none; box-shadow:0 0 0 2px var(--bg-panel), 0 0 0 4px var(--focus-ring); }
.zbtn + .zbtn{ margin-left:-1px; }

.tlbody{
  position:relative;
  overflow:auto;
  padding:var(--space-2) 0;
  max-height:34vh;
}

/* track row */
.tk{ display:flex; align-items:stretch; min-height:44px; }
.tk + .tk{ margin-top:var(--space-1); }

/* sticky label */
.tklab{
  position:sticky; left:0; z-index:5;
  flex:0 0 96px; width:96px;
  display:flex; align-items:center; gap:6px;
  padding:0 var(--space-3);
  font-size:10.5px; font-weight:600; letter-spacing:0.4px; text-transform:uppercase;
  color:var(--text-muted);
  background:var(--bg-panel);
  border-right:1px solid var(--border-soft);
}

/* track lane — faint backdrop with ruler ticks (decorative, independent of JS px math) */
.tkrow{
  position:relative;
  flex:1 1 auto; min-width:0;
  margin:4px var(--space-3) 4px 0;
  border-radius:var(--radius-sm);
  background:
    repeating-linear-gradient(90deg,
      transparent 0, transparent 63px,
      var(--tk-tick) 63px, var(--tk-tick) 64px),
    var(--tk-lane);
  box-shadow:inset 0 0 0 1px var(--border-soft);
}

/* clip block — absolute-positioned (JS sets left/width in px), color-coded by track */
.blk{
  position:absolute;
  top:4px; bottom:4px;
  display:flex; align-items:center;
  padding:0 8px 0 11px;
  font-size:11px; font-weight:560; letter-spacing:0.2px;
  color:var(--text-primary);
  border:1px solid transparent;
  border-radius:var(--radius-sm);
  overflow:hidden; white-space:nowrap; text-overflow:ellipsis;
  cursor:pointer; user-select:none;
  box-shadow:0 1px 0 0 rgba(255,255,255,0.05) inset, 0 2px 5px -3px rgba(0,0,0,0.55);
  transition:filter .12s ease, box-shadow .12s ease, transform .06s ease;
}
/* left identity bar per clip (grabbability + reinforces track loudness) */
.blk::before{
  content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  border-radius:var(--radius-sm) 0 0 var(--radius-sm);
}
.blk:hover{ filter:brightness(1.12); z-index:3; }
.blk:active{ transform:translateY(0.5px); }

/* per-track LOUDNESS HIERARCHY: visual(loudest) > card > voice > music(distinct) > caption(lightest) */
.blk.v   { background:var(--clip-visual-bg);  color:var(--clip-visual-tx);  border-color:var(--clip-visual-bd);  }
.blk.v::before  { background:var(--clip-visual-bd); }
.blk.c   { background:var(--clip-card-bg);    color:var(--clip-card-tx);    border-color:var(--clip-card-bd);    }
.blk.c::before  { background:var(--clip-card-bd); }
.blk.vo  { background:var(--clip-voice-bg);   color:var(--clip-voice-tx);   border-color:var(--clip-voice-bd);   }
.blk.vo::before { background:var(--clip-voice-bd); }
/* MUSIC — distinct warm bed, subtle waveform-ish texture marks it as audio */
.blk.m   {
  color:var(--clip-music-tx); border-color:var(--clip-music-bd);
  background:
    repeating-linear-gradient(90deg,
      rgba(0,0,0,0.16) 0 5px, transparent 5px 10px),
    var(--clip-music-bg);
}
.blk.m::before  { background:var(--clip-music-bd); }
/* CAPTION — lightest/quietest: low-sat violet, slightly dimmed */
.blk.cap { background:var(--clip-caption-bg); color:var(--clip-caption-tx); border-color:var(--clip-caption-bd); opacity:.94; }
.blk.cap::before{ background:var(--clip-caption-bd); }

/* selected clip — single blue ring layered OVER the clip's own color, lifts above siblings */
.blk.sel-clip{
  border-color:var(--accent);
  z-index:4;
  box-shadow:
    0 0 0 1px var(--accent),
    0 0 0 4px var(--accent-soft),
    0 1px 0 0 rgba(255,255,255,0.10) inset,
    0 6px 16px -8px rgba(0,0,0,0.6);
}
.blk.sel-clip::before{ width:4px; background:var(--accent); box-shadow:0 0 8px rgba(59,130,246,0.6); }

/* playhead — the one red, hairline with a grab cap, above all tracks */
.playhead{
  position:absolute; top:0; bottom:0; width:2px;
  background:var(--playhead);
  box-shadow:0 0 8px rgba(255,90,90,0.55);
  z-index:10; pointer-events:none;
}
.playhead::before{
  content:""; position:absolute; top:-1px; left:50%; transform:translateX(-50%);
  width:11px; height:8px; border-radius:2px 2px 5px 5px;
  background:var(--playhead);
  box-shadow:0 1px 3px rgba(0,0,0,0.5);
}

/* ---------------------------------------------------------------------------
   SCROLLBARS — thin, cool, hover-darkening (WebKit + Firefox), scoped to scrollers
   ------------------------------------------------------------------------- */
.edscenes,.edinspector,.tlbody{ scrollbar-width:thin; scrollbar-color:#2b3744 transparent; }
.edscenes::-webkit-scrollbar,
.edinspector::-webkit-scrollbar,
.tlbody::-webkit-scrollbar{ width:10px; height:10px; }
.edscenes::-webkit-scrollbar-track,
.edinspector::-webkit-scrollbar-track,
.tlbody::-webkit-scrollbar-track{ background:transparent; }
.edscenes::-webkit-scrollbar-thumb,
.edinspector::-webkit-scrollbar-thumb,
.tlbody::-webkit-scrollbar-thumb{
  background:#2b3744;
  border:3px solid transparent;
  background-clip:padding-box;
  border-radius:var(--radius-pill);
}
.edscenes::-webkit-scrollbar-thumb:hover,
.edinspector::-webkit-scrollbar-thumb:hover,
.tlbody::-webkit-scrollbar-thumb:hover{ background:#3a4856; background-clip:padding-box; }
.edscenes::-webkit-scrollbar-corner,
.edinspector::-webkit-scrollbar-corner,
.tlbody::-webkit-scrollbar-corner{ background:transparent; }

/* ---------------------------------------------------------------------------
   RESPONSIVE — collapse to single column on narrow viewports
   ------------------------------------------------------------------------- */
@media (max-width:1000px){
  .edwrap{
    grid-template-columns:minmax(0,1fr);
    grid-template-areas:
      "preview"
      "scenes"
      "inspector"
      "timeline";
  }
  .edscenes,.edinspector{ max-height:400px; }
}

/* ===== DECISIONS =====
   ---------------------------------------------------------------------------
   WHO WON EACH AREA
   - TOKENS: shared spine from all three (identical surfaces/borders/text/accent/
     status/spacing — strong convergence). Depth shadows from P3 (1px light top
     edge + soft low-spread drop = real "graphite lift", no gloss). Clip palette
     from P1/P2 (bg + bd + tx triplet per track; loudness encoded in fill+label).
   - HEADER / CHIPS / WARNSTRIP: P1's restraint + dot system, with P3's solid
     warnb "!" badge. color-mix only on header bg (graceful fallback noted).
   - SCENES: P3's CSS-grid row + dotted scgrip + accent spine, P1's badge recipes
     (.skip strikethrough, .ai blue, .card green, .edited amber).
   - PREVIEW: P2's inset editing-canvas frame on #edvid (THE focal move) + P2/P3
     transport. This is the single biggest "it's an editor, not an admin panel" cue.
   - FORM CONTROLS: P1's fully custom checkbox + range thumb (most premium, no
     reliance on accent-color), scoped to .audiobar/.insp/#cardfields/.fld.
   - INSPECTOR: P3's CSS-triangle caret (rotates for open/collapsed, both schemes
     handled), P1/P2 section + field primitives.
   - BUTTONS: P1's FLAT precise hierarchy (export = solid blue, edbtn = elevated,
     ghost = bare). Rejected P2/P3 gradient export — flat reads calmer/more Linear
     and the brief forbids heavy gradients. Tiny inset highlight only.
   - TIMELINE: P2's absolute-positioned .blk in a lane-with-ticks .tkrow + the
     "selection is a RING over the clip's own color" principle (clip never loses
     track identity). Music gets a faint waveform texture; caption dimmed .94.

   TOP 3 WATCH-OUTS FOR IMPLEMENTATION
   1) .insec collapse toggle: caret + body hide key off .collapsed AND
      [aria-collapsed="true"] (and [open]/.open for the caret). Wire ONE of these
      in your JS to whichever you already toggle; if you use inline display on
      .inbody instead, the body-hide rule is harmless but align the caret selector.
   2) Clip positioning is ABSOLUTE: .blk uses position:absolute + top/bottom:4px,
      so your JS MUST set inline left/width (px) on each .blk, and .tkrow is the
      positioning context (already position:relative). The 64px ruler tick pitch
      is purely decorative — it does not need to match your real time scale.
   3) Modern-CSS reliance: color-mix() (header bg only) and backdrop-filter
      (header) need Chromium/FF/Safari 16.2+. Both degrade gracefully (near-solid
      panel). aspect-ratio on #edvid is well-supported but verify your target.
      Panel max-heights use calc(100vh - 240px); if the header/warnstrip grow,
      bump that offset or convert .edscenes/.edinspector parents to flex.
   --------------------------------------------------------------------------- */

/* ===== RECONCILE — align the design system to the exact DOM this editor emits =====
   (id-scoped so these always win over the generic design-system rules above) */
/* keep the proven full-height workspace + resizable-panel CSS vars (DS left height:auto) */
@media(min-width:1001px){
  /* P1 — fluid panel defaults: scale with viewport (narrow laptops → big
     monitors) so the editor is never cramped at 1366 nor tiny at 2560, while
     the center preview stays dominant. User drags still override via --ed-*. */
  .edwrap{height:calc(100vh - 116px);min-height:540px;
    grid-template-columns:var(--ed-left,clamp(232px,18vw,392px)) minmax(0,1fr) var(--ed-right,clamp(300px,22vw,468px));
    grid-template-rows:minmax(0,1fr) var(--ed-tl,210px)}
  .edscenes,.edinspector{max-height:none}
  .edtimeline{max-height:none}
}
/* preview: keep the DS canvas frame but not its aspect-ratio (flex fills the cell) */
.edpreview video#edvid{aspect-ratio:auto;flex:1 1 0;min-height:0}
/* scene badges are INDIVIDUAL chips wrapped in .scbadges (not a DS wrapper) */
#edscenes .scbadges{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px}
#edscenes .scb{display:inline-flex;align-items:center;height:18px;padding:0 8px;font-size:10px;font-weight:650;
  letter-spacing:.3px;text-transform:uppercase;border-radius:var(--radius-sm);color:var(--text-muted);
  background:var(--bg-elevated);border:1px solid var(--border-soft);white-space:nowrap}
#edscenes .scb.ai{color:#bcd2f5;background:var(--accent-soft);border-color:rgba(59,130,246,.30)}
#edscenes .scb.card{color:var(--success-text);background:var(--success-soft);border-color:rgba(98,204,131,.28)}
#edscenes .scb.edited{color:var(--warning-text);background:var(--warning-soft);border-color:rgba(226,181,82,.28)}
#edscenes .scb.skip{color:var(--text-muted);background:transparent;border-style:dashed;text-decoration:line-through}
#edscenes .scb.foot{color:var(--clip-voice-tx);background:var(--bg-elevated);border-color:var(--border-strong)}
#edscenes .scb.miss{color:var(--danger-text);background:var(--danger-soft);border-color:rgba(226,134,78,.32)}
/* drag grip: keep the glyph, drop the DS dot pattern; subtle-always, brighter on hover */
#edscenes .scgrip{opacity:.5;font-size:14px;width:16px}
#edscenes .scgrip::before{content:none!important}
#edscenes .scrow:hover .scgrip{opacity:1;color:var(--text-secondary)}
/* keywords are INDIVIDUAL chips; dots are a TEXT string of glyphs */
#edinsp .kw{display:inline-flex;align-items:center;padding:3px 9px;font-size:11px;color:#bcd2f5;
  background:var(--accent-soft);border:1px solid rgba(59,130,246,.26);border-radius:var(--radius-pill);margin:3px 4px 0 0}
#edinsp .dots{display:inline;letter-spacing:2px;color:var(--warning)}
/* collapsible inspector uses .col (not DS .collapsed/[open]); caret is a glyph (not a CSS triangle) */
#edinsp .car{margin-left:auto;width:auto;height:auto;border:0;color:var(--text-muted);font-size:11px;transition:transform .16s}
#edinsp .insec.col>.inhd .car{transform:rotate(-90deg)}
#edinsp .insec:not(.col)>.inhd .car{transform:none}
#edinsp .insec.col>.inbody{display:none}
#edinsp .insec:not(.col)>.inbody{display:flex;flex-direction:column;gap:var(--space-3)}
/* DS makes inspector/scenes flex columns; children must NOT shrink (keep content height -> panel scrolls) */
#edinsp>*{flex:0 0 auto}
#edinsp .insec{flex:0 0 auto}
#edscenes .scrow{flex:0 0 auto}
/* resizable-panel drag handles */
.edwrap{position:relative}
.edrz{position:absolute;z-index:25;background:transparent;touch-action:none;user-select:none}
.edrz:not(.horiz){width:11px;cursor:col-resize;transform:translateX(-50%)}
.edrz.horiz{left:0;right:0;height:11px;cursor:row-resize;transform:translateY(-50%)}
.edrz::after{content:"";position:absolute;background:var(--border-strong);opacity:0;transition:opacity .14s,background .14s}
.edrz:not(.horiz)::after{left:50%;top:8px;bottom:8px;width:2px;transform:translateX(-50%);border-radius:2px}
.edrz.horiz::after{top:50%;left:14px;right:14px;height:2px;transform:translateY(-50%);border-radius:2px}
.edrz:hover::after,.edrz.dragging::after{opacity:1;background:var(--accent)}
@media(max-width:1000px){.edrz{display:none}}
/* timeline selection toggles .blk.v.sel (JS) — give it the premium accent ring */
#edtlb .blk.sel,#edtlb .blk.v.sel{border-color:var(--accent);z-index:4;outline:none;
  box-shadow:0 0 0 1px var(--accent),0 0 0 4px var(--accent-soft),
    0 1px 0 0 rgba(255,255,255,.10) inset,0 6px 16px -8px rgba(0,0,0,.6)}
/* productization: compact audio toolbar + collapsible "Look & style" panel */
.audiobar .audvol,.audiobar .audlook{display:inline-flex;align-items:center;gap:7px}
.audiobar .audval{font-variant-numeric:tabular-nums;color:var(--text-secondary);min-width:34px}
.audiobar .audsep{width:1px;height:18px;background:var(--border-soft)}
.audiobar .audsig{font-size:11px;color:var(--text-muted)}
.edproj{background:var(--bg-elevated);border:1px solid var(--border-soft);border-radius:var(--radius-md);overflow:hidden}
.edproj>summary{display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none;list-style:none;
  padding:9px 13px;font-size:12px;font-weight:600;color:var(--text-secondary)}
.edproj>summary::-webkit-details-marker{display:none}
.edproj>summary::before{content:"\\25B8";color:var(--text-muted);font-size:10px;transition:transform .15s}
.edproj[open]>summary::before{transform:rotate(90deg)}
.edproj[open]>summary{color:var(--text-primary)}
.edprojmeta{margin-left:auto;font-weight:400;font-size:11px;color:var(--text-muted);font-variant-numeric:tabular-nums}
.edprojbody{padding:6px 13px 13px;border-top:1px solid var(--border-soft);display:flex;flex-direction:column;gap:10px}
.projrow{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.projlbl{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--text-muted);font-weight:700}
.projval{font-size:12px;color:var(--text-secondary);line-height:1.5}
.projcolor{width:30px;height:24px;padding:0;border:1px solid var(--border-strong);border-radius:6px;background:var(--bg-app);cursor:pointer}
.edprojbody select{height:28px;font-size:12px}
.projadv{border-top:1px dashed var(--border-soft);padding-top:8px}
.projadv>summary{cursor:pointer;list-style:none;font-size:11px;color:var(--text-muted);font-weight:600}
.projadv>summary::-webkit-details-marker{display:none}
.projadv>summary::before{content:"\\25B8 ";color:var(--text-muted)}
.projadv[open]>summary::before{content:"\\25BE "}
.projadvrow{margin-top:8px;gap:6px}
/* warning area: minor = light inline chips, serious = full amber strip (DS) */
.warnstrip.minor{background:none;border:0;box-shadow:none;padding:0 2px;gap:8px}
.warnchip{display:inline-flex;align-items:center;gap:7px;height:25px;padding:0 11px;font-size:11.5px;
  color:var(--text-secondary);background:var(--bg-panel);border:1px solid var(--border-soft);border-radius:var(--radius-pill)}
.warnchip::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--warning);flex:none}
/* timeline time-ruler + collapsible secondary tracks */
.tkruler{min-height:20px;margin-bottom:3px}
.tkruler .tkrul{background:none!important;box-shadow:none;border:0;position:relative;height:18px;margin:0 var(--space-3) 0 0}
.rtick{position:absolute;top:0;font-size:9.5px;color:var(--text-muted);white-space:nowrap;transform:translateX(-1px)}
.rtick>i{display:block;width:1px;height:6px;background:var(--border-strong)}
.rtick>b{font-weight:500;font-variant-numeric:tabular-nums;display:block;margin-top:1px}
.tklabtog{cursor:pointer}
.tklabtog:hover{color:var(--text-secondary)}
.tk.tkmin{min-height:22px}
.tk.tkmin .tkrow{min-height:13px;opacity:.45}
.tkrowmin{background:repeating-linear-gradient(90deg,transparent 0 8px,var(--tk-tick) 8px 9px),var(--tk-lane)}

/* ===== PREVIEW-FIRST LAYOUT (v1.1) — make the video the visual hero ===== */
/* the video gets its OWN growing stage; controls below stay compact */
.edpreview{gap:8px!important;padding:10px!important;min-height:0}
.edpvstage{flex:1 1 auto;min-height:120px;display:flex;align-items:center;justify-content:center;position:relative;
  border-radius:var(--radius-md);overflow:hidden;
  background:radial-gradient(120% 120% at 50% 0%,#0e131a 0%,#05070a 100%);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.04),inset 0 0 0 5px rgba(8,11,15,.55),0 16px 44px -20px rgba(0,0,0,.82);
  outline:1px solid var(--border-strong);outline-offset:-1px}
.edpreview video#edvid,.edpvstage>video{width:100%!important;height:100%!important;flex:none!important;min-height:0!important;
  object-fit:contain;background:transparent;border:0;outline:0;box-shadow:none;border-radius:0;aspect-ratio:auto}
.edpvdrop{border-radius:var(--radius-md)}
/* slim transport so it never steals preview height; seek bar replaces native controls */
.edtransport{flex-wrap:nowrap!important;gap:6px!important;padding:5px 9px!important}
.edtransport .tbtn{height:30px;min-width:30px;padding:0 8px;font-size:13px}
.edtransport .tbtn.play{width:34px;height:34px;min-width:34px}
.edtransport .tbtn.on{background:var(--accent);border-color:var(--accent);color:#fff}
.tbseek{flex:1 1 auto;min-width:60px;height:4px;margin:0 4px;cursor:pointer;
  -webkit-appearance:none;appearance:none;background:var(--border-strong);border-radius:var(--radius-pill)}
.tbseek::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:13px;height:13px;border-radius:50%;
  background:var(--accent);border:2px solid var(--bg-panel);box-shadow:0 1px 3px rgba(0,0,0,.5);cursor:pointer}
.tbseek::-moz-range-thumb{width:13px;height:13px;border-radius:50%;background:var(--accent);border:2px solid var(--bg-panel)}
.edtransport .ttime{margin-left:0;min-width:auto;white-space:nowrap}
/* THEATER / large-preview mode — hide secondary controls + shrink timeline */
.edpreview.theater .audiobar,.edpreview.theater .edproj{display:none}
@media(min-width:1001px){.edwrap.theater{grid-template-rows:minmax(0,1fr) 116px}}

</style>
{% raw %}
<script>
(function(){
const R=document.getElementById('ed-root'), SLUG=R.dataset.slug;
let M=null,T=null,sel=0,TLZOOM=1,THUMBBUST=Date.now();
// ---- Menu dropdown + whole-video settings popover (CapCut-clean) ----
function __edCloseMenu(){var m=document.getElementById('edmenupop');if(m)m.hidden=true;var b=document.getElementById('edmenu');if(b)b.setAttribute('aria-expanded','false');}
function __edMenuInit(){var b=document.getElementById('edmenu');if(!b)return;b.onclick=function(e){e.stopPropagation();var m=document.getElementById('edmenupop');if(!m)return;var willShow=m.hidden;m.hidden=!willShow;b.setAttribute('aria-expanded',willShow?'true':'false');};}
window.__edResetAll=async function(){if(!confirm('Reset ALL changes on this project? This cannot be undone.'))return;try{await jpost('/e/'+SLUG+'/reset-all');await reload();if(window.toast)toast('All changes reset','success');}catch(e){alert(e.message);}};
window.__edOpenVideoSettings=function(){__edCloseMenu();var p=document.getElementById('edvpop'),bk=document.getElementById('edvback');if(p)p.hidden=false;if(bk)bk.hidden=false;};
window.__edCloseVideoSettings=function(){var p=document.getElementById('edvpop'),bk=document.getElementById('edvback');if(p)p.hidden=true;if(bk)bk.hidden=true;};
window.__edRefreshPreview=function(){__edCloseMenu();var v=document.getElementById('edvid');if(v){var s=v.querySelector('source');if(s)s.src='/v/'+SLUG+'/file/video?v='+Date.now();try{v.load();}catch(e){}}if(window.toast)toast('Preview refreshed','success');};
window.__edMenuAct=function(a){__edCloseMenu();
  if(a==='dashboard'){window.location='/';}
  else if(a==='refresh'){reload();}
  else if(a==='reset'){__edResetAll();}
  else if(a==='video'){__edOpenVideoSettings();}
  else if(a==='preview'){__edRefreshPreview();}
  else if(a==='layout'){if(window.__edResetLayout)__edResetLayout();}
};
if(!window.__edPopGlobals){window.__edPopGlobals=true;
  document.addEventListener('click',function(e){
    var t=e.target;
    var m=document.getElementById('edmenupop');if(m&&!m.hidden&&t&&t.closest&&!t.closest('.edmenuwrap'))__edCloseMenu();
    if(t&&t.closest&&!t.closest('.rmenu')&&!t.closest('.scdots')&&window.__rmClose)__rmClose();
  });
  document.addEventListener('keydown',function(e){if(e.key==='Escape'){
    if(window.__rmClose)__rmClose();
    var m=document.getElementById('edmenupop');if(m&&!m.hidden){__edCloseMenu();return;}
    var p=document.getElementById('edvpop');if(p&&!p.hidden)__edCloseVideoSettings();}});
}
let TLHIDE=(function(){try{return JSON.parse(localStorage.getItem('vf_ed_tracks')||'{}')||{};}catch(e){return {};}})();
window.__edTLtoggle=function(k){TLHIDE[k]=!TLHIDE[k];try{localStorage.setItem('vf_ed_tracks',JSON.stringify(TLHIDE));}catch(e){}renderTimeline();};
const esc=s=>(s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=s=>{s=Math.max(0,Math.round(s||0));return Math.floor(s/60)+':'+String(s%60).padStart(2,'0');};
function srcMix(o){o=o||{};const map={pexels_stock:'Stock',ai_fal:'AI',ai_pollinations:'AI',web_image_selected:'Web',web_footage_selected:'Web',archive_org:'Archive'};
  const agg={};for(const k in o){if(o[k]>0){const n=map[k]||k;agg[n]=(agg[n]||0)+o[k];}}
  return Object.keys(agg).map(k=>k+' ×'+agg[k]).join(' · ')||'—';}

async function jpost(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});
  let j={};try{j=await r.json();}catch(e){}if(!r.ok||j.ok===false)throw new Error(j.error||('HTTP '+r.status));return j;}
async function fetchData(){const a=await Promise.all([fetch('/e/'+SLUG+'/manifest.json').then(r=>r.json()),
  fetch('/e/'+SLUG+'/timeline.json').then(r=>r.json())]);if(a[0].error)throw new Error(a[0].error);M=a[0];T=a[1];}
// ===== TOOLTIP ENGINE — one delegated component for every [data-tip] / [title] =====
// Reads data-tip (preferred) or the native title, and strips the native title while
// our tooltip is up so the slow browser tooltip never double-fires. Delegated on
// document, so it survives every inspector / scene-list / timeline re-render.
var __edTipEl=null,__edTipTimer=null,__edTipFor=null;
function __edTipText(el){return (el.getAttribute('data-tip')||el.getAttribute('title')||'').trim();}
function __edTipFind(t){for(var e=t;e&&e!==document;e=e.parentElement){
  if(e.getAttribute&&((e.getAttribute('data-tip')||'')||(e.getAttribute('title')||'')))return e;}return null;}
function __edTipShow(el){
  var txt=__edTipText(el);if(!txt)return;
  if(el.hasAttribute('title')){el.setAttribute('data-tiptitle',el.getAttribute('title'));el.removeAttribute('title');}
  if(!__edTipEl){__edTipEl=document.createElement('div');__edTipEl.className='edtip';__edTipEl.setAttribute('role','tooltip');document.body.appendChild(__edTipEl);}
  __edTipEl.textContent=txt;__edTipEl.style.display='block';__edTipFor=el;
  var r=el.getBoundingClientRect(),tw=__edTipEl.offsetWidth,th=__edTipEl.offsetHeight,pad=8,vw=window.innerWidth,vh=window.innerHeight;
  var x=r.left+r.width/2-tw/2, y=r.top-th-8;            // default: above the control
  if(y<pad)y=r.bottom+8;                                 // flip below if it would clip off the top
  x=Math.max(pad,Math.min(x,vw-tw-pad));                 // clamp inside the viewport (no clipping)
  y=Math.max(pad,Math.min(y,vh-th-pad));
  __edTipEl.style.left=Math.round(x)+'px';__edTipEl.style.top=Math.round(y)+'px';}
function __edTipHide(){
  if(__edTipFor&&__edTipFor.getAttribute&&__edTipFor.getAttribute('data-tiptitle')){
    __edTipFor.setAttribute('title',__edTipFor.getAttribute('data-tiptitle'));__edTipFor.removeAttribute('data-tiptitle');}
  __edTipFor=null;if(__edTipTimer){clearTimeout(__edTipTimer);__edTipTimer=null;}
  if(__edTipEl)__edTipEl.style.display='none';}
function __edTipInit(){
  if(window.__edTipReady)return;window.__edTipReady=true;
  document.addEventListener('mouseover',function(e){var el=__edTipFind(e.target);if(!el||el===__edTipFor)return;
    __edTipHide();__edTipTimer=setTimeout(function(){__edTipShow(el);},320);},true);
  document.addEventListener('mouseout',function(e){if(__edTipFind(e.target))__edTipHide();},true);
  document.addEventListener('focusin',function(e){var el=__edTipFind(e.target);if(el){__edTipHide();__edTipShow(el);}},true);
  document.addEventListener('focusout',__edTipHide,true);
  document.addEventListener('keydown',function(e){if(e.key==='Escape')__edTipHide();},true);
  window.addEventListener('scroll',__edTipHide,true);}
// ===== LIVE PREVIEW STATUS (P5) — a compact transient pill over the center stage =====
function __edStatus(txt,autoMs){
  var el=document.getElementById('edstatusmsg');
  if(!el){var stage=document.querySelector('.edpvstage');if(!stage)return;
    el=document.createElement('div');el.id='edstatusmsg';el.className='edstatusmsg';stage.appendChild(el);}
  if(window.__edStatusT){clearTimeout(window.__edStatusT);window.__edStatusT=null;}
  if(!txt){el.hidden=true;el.textContent='';return;}
  el.innerHTML='<span class=edstatusdot></span>'+txt;el.hidden=false;
  if(autoMs)window.__edStatusT=setTimeout(function(){el.hidden=true;},autoMs);}
// ===== LIVE CAPTIONS (P1 / Issue 2) =====
// The rendered MP4 has captions BURNED IN, so to toggle them live we render captions
// as an HTML overlay (from the .srt) over a NO-CAPTION proxy of the same video. The
// proxy is generated once (background) and cached; until it exists, captions OFF kicks
// that build. The final export still burns captions (captions_enabled is honored).
var __edCues=null;
function __edSrtT(s){s=(s||'').trim().replace(',','.');var p=s.split(':');
  return p.length<3?0:((+p[0])*3600+(+p[1])*60+(parseFloat(p[2])||0));}
async function __edCaptionsLoad(){
  if(__edCues)return __edCues;__edCues=[];
  if(!(M.assets&&M.assets.srt))return __edCues;
  try{var r=await fetch('/v/'+SLUG+'/file/srt?v='+(M.video_mtime||0));var txt=await r.text();
    txt.replace(/\\r/g,'').split(/\\n\\n+/).forEach(function(b){
      var lines=b.split('\\n').filter(function(x){return x.trim();});if(lines.length<2)return;
      var tl=null,ti=-1;for(var k=0;k<lines.length;k++){if(lines[k].indexOf('-->')>=0){tl=lines[k];ti=k;break;}}
      if(!tl)return;var mm=tl.split('-->');var text=lines.slice(ti+1).join(' ').trim();
      if(text)__edCues.push({s:__edSrtT(mm[0]),e:__edSrtT(mm[1]),t:text});});
  }catch(e){}
  return __edCues;}
function __edCaptionsOn(){return !(M.global&&M.global.captions_enabled===false);}
function __edCaptionSync(){
  var lc=document.getElementById('edlaycaptions');if(!lc)return;
  if(!window.__edNocapReady||!__edCaptionsOn()||!__edCues||!__edCues.length){lc.hidden=true;lc.innerHTML='';return;}
  var v=document.getElementById('edvid'),t=v?v.currentTime:0,cur=null;
  for(var i=0;i<__edCues.length;i++){if(t>=__edCues[i].s&&t<__edCues[i].e){cur=__edCues[i];break;}}
  if(cur){lc.innerHTML='<div class=edcapln>'+esc(cur.t)+'</div>';lc.hidden=false;}
  else{lc.hidden=true;lc.innerHTML='';}}
function __edUseNocapBase(){
  window.__edNocapReady=true;var v=document.getElementById('edvid');if(!v)return;
  if(v.dataset.nocap==='1'){__edCaptionSync();return;}
  var keep=v.currentTime||0,playing=!v.paused;
  var src=v.querySelector('source');if(src){src.src='/e/'+SLUG+'/file/nocap-video?v='+(M.video_mtime||0);}
  v.dataset.nocap='1';
  try{for(var i=0;i<v.textTracks.length;i++)v.textTracks[i].mode='hidden';}catch(e){}  // no double captions
  try{v.load();v.currentTime=keep;if(playing)v.play();}catch(e){}
  __edCaptionSync();}
function __edPollProxy(job_id){
  if(window.__edProxyIv)clearInterval(window.__edProxyIv);
  window.__edProxyIv=setInterval(async function(){
    try{var r=await fetch('/job/'+job_id+'/status');var j=await r.json();
      if(j.status==='done'){clearInterval(window.__edProxyIv);__edUseNocapBase();__edStatus('Caption preview ready',1800);}
      else if(j.status==='error'){clearInterval(window.__edProxyIv);__edStatus('');if(window.toast)toast('Could not prepare the caption preview','error');}
      else __edStatus('Preparing instant caption preview… '+(j.pct||0)+'%');}catch(e){}},2500);}
async function __edEnsureNocap(kick){
  try{var r=await fetch('/e/'+SLUG+'/caption-proxy/status');var j=await r.json();
    if(j&&j.ready){__edUseNocapBase();return true;}}catch(e){}
  if(kick){__edStatus('Preparing instant caption preview…');
    try{var r2=await fetch('/e/'+SLUG+'/caption-proxy',{method:'POST'});var j2=await r2.json();
      if(j2&&j2.ready){__edUseNocapBase();return true;}
      if(j2&&j2.job_id)__edPollProxy(j2.job_id);}catch(e){}}
  return false;}
async function __edCaptionsInit(){
  await __edCaptionsLoad();
  // LIVE CAPTIONS (Issue 2 — re-enabled 2026-06-03). The production renderer now emits a
  // FOOTAGE-MATCHED no-caption base (editor_cache/preview_nocap.mp4) alongside the
  // captioned MP4 in the SAME render, so the two differ ONLY by the burned captions. We
  // swap the preview to that clean base and render captions as an HTML overlay toggled by
  // captions_enabled — instant ON/OFF with the footage unchanged, no reload, no re-render.
  // We only CHECK for a fresh base here (no heavy build is kicked on open). If one isn't
  // present (a project rendered before this change), the burned MP4 stays in place and we
  // stay honest: a single re-render produces the base.
  var ready=await __edEnsureNocap(false);
  if(!ready&&!__edCaptionsOn())
    __edStatus('Captions are OFF for the final video — re-render once to enable the live toggle',3600);}
async function boot(){R.innerHTML='<p class=muted>Loading editor… (building manifest + scene posters on first open)</p>';
  try{await fetchData();mount();paint();}catch(e){R.innerHTML='<p class=muted>Could not load the editor: '+esc(e.message)+'</p>';}}
async function reload(){THUMBBUST=Date.now();try{await fetchData();paint();__edCaptionSync();}catch(e){alert('Reload failed: '+e.message);}}
function mount(){
  R.innerHTML=
   '<div class=edhead>'
   +'<div class=edhl>'
   +'<a href="/v/'+SLUG+'" class=edchip title="Back to the video page">‹ Back</a>'
   +'<h1 title="'+esc(M.title)+'">'+esc(M.title)+'</h1>'
   +'<span class=edchip>'+M.duration_pretty+'</span>'
   +'<span class=edchip>'+M.scenes.length+' scenes</span>'
   +'<span class=edchip id=edsrc></span><span class=edchip id=edqa></span>'
   +'<span class=edchip id=edunsaved style="display:none" data-tip="Edits you have made but not yet rendered into the final MP4. Click Apply & re-render to bake them in."></span>'
   +'</div>'
   +'<div class=edhr>'
   +'<button class=ghost id=edundo data-tip="Undo your last change (visual, card text, layer, reorder).">↶ Undo</button>'
   +'<button class=edbtn id=edexport data-tip="Apply your edits and create an updated final MP4. Everything in the live preview is baked into the exported video." style="background:#2f6df0;color:#fff;border-color:#2f6df0">🎬 Apply &amp; re-render</button>'
   +'<div class=edmenuwrap>'
     +'<button class=ghost id=edmenu title="More options" aria-haspopup=menu aria-expanded=false>☰ Menu</button>'
     +'<div class=edpop id=edmenupop hidden role=menu>'
       +'<div class=edpopgrp>Project</div>'
       +'<button class=edpopitem role=menuitem onclick="__edMenuAct(\\'dashboard\\')">Return to dashboard</button>'
       +'<button class=edpopitem role=menuitem onclick="__edMenuAct(\\'refresh\\')">Refresh project</button>'
       +'<button class="edpopitem danger" role=menuitem onclick="__edMenuAct(\\'reset\\')">Reset all changes</button>'
       +'<div class=edpopsep></div>'
       +'<div class=edpopgrp>Video settings</div>'
       +'<button class=edpopitem role=menuitem onclick="__edMenuAct(\\'video\\')">Whole video settings…</button>'
       +'<div class=edpopsep></div>'
       +'<div class=edpopgrp>Advanced</div>'
       +'<button class=edpopitem role=menuitem onclick="__edMenuAct(\\'preview\\')">Refresh preview</button>'
       +'<button class=edpopitem role=menuitem onclick="__edMenuAct(\\'layout\\')">Reset panel layout</button>'
     +'</div>'
   +'</div>'
   +'</div>'
   +'</div>'
   +'<div id=edwarn class=warnstrip></div>'
   +'<div class=edwrap>'
   +'<div class=edscenes id=edscenes></div>'
   +'<div class=edpreview id=edpv ondragover="__edZoneOver(event)" ondragleave="__edZoneLeave(event)" ondrop="__edPreviewDrop(event)">'
   +'<div class=edpvstage><video id=edvid playsinline><source src="/v/'+SLUG+'/file/video?v='+(M.video_mtime||0)+'" type=video/mp4>'
   +(M.assets.srt?'<track kind=subtitles src="/v/'+SLUG+'/file/srt?v='+(M.video_mtime||0)+'" default>':'')+'</video>'
   +'<div class=edlay id=edlayvisual hidden></div>'
   +'<div class=edlay id=edlaycard hidden></div>'
   +'<div class=edlaycaptions id=edlaycaptions hidden></div>'
   +'<div class=eddraftlbl id=eddraftlbl hidden><span class=eddraftdot></span>Live draft preview</div>'
   +'</div>'
   +'<div class=edtransport>'
     +'<button class=tbtn id=tbprev title="Previous scene (↑)">⏮</button>'
     +'<button class=tbtn id=tbreplay title="Replay this scene">⟲</button>'
     +'<button class="tbtn play" id=tbplay title="Play / Pause (space)">▶</button>'
     +'<button class=tbtn id=tbnext title="Next scene (↓)">⏭</button>'
     +'<input type=range id=tbseek class=tbseek min=0 max=1000 value=0 title="Seek">'
     +'<span class=ttime id=tbtime>0:00 / 0:00</span>'
     +'<button class=tbtn id=tbmute title="Mute / unmute preview">\\ud83d\\udd0a</button>'
     +'<button class=tbtn id=tbtheater title="Large preview (theater)">\\u2195</button>'
     +'<button class=tbtn id=tbfs title="Fullscreen">\\u26f6</button>'
   +'</div>'
   +'<button class=edvsummary id=edvsummary type=button title="Open whole-video settings — music, subtitles, style, quality" onclick="__edOpenVideoSettings()"><span class=edvslead>Whole video settings</span><span class=edvspills id=edvspills></span><span class=edvscog>⚙</span></button>'
   +'<div class=edpvdrop><div class=edpvdropc><span class=edpvdropic>\\u21e9</span>'
     +'<span>Drop an image or video to replace this scene</span></div></div></div>'
   +'<div class="edinspector insp" id=edinsp></div>'
   +'<div class=edtimeline id=edtl></div>'
   +'<div class=edrz id=edrzL title="Drag to resize — double-click to reset"></div>'
   +'<div class=edrz id=edrzR title="Drag to resize — double-click to reset"></div>'
   +'<div class="edrz horiz" id=edrzT title="Drag to resize — double-click to reset"></div>'
   +'</div>'
   +'<div class=edvback id=edvback hidden onclick="__edCloseVideoSettings()"></div>'
   +'<div class=edpop id=edvpop hidden role=dialog aria-label="Whole video settings" aria-modal=true>'
     +'<div class=edpoph>Whole video settings'
       +'<button class=edpopx type=button onclick="__edCloseVideoSettings()" title="Close" aria-label=Close>✕</button></div>'
     +'<div class=edvsec><div class=edvseclbl>Audio &amp; subtitles</div>'
       +'<div id=edaudio class=audiobar></div></div>'
     +'<details class=edproj id=edproj open><summary>Video style'
       +'<span class=edprojmeta id=edpchips></span></summary>'
       +'<div id=edprojbody class=edprojbody></div></details>'
   +'</div>';
  __edInitResize();
  __edTipInit();
  __edCaptionsInit();
  var V=document.getElementById('edvid');
  V.addEventListener('timeupdate',onTime);
  V.addEventListener('play',function(){var b=document.getElementById('tbplay');if(b)b.textContent='⏸';});
  V.addEventListener('pause',function(){var b=document.getElementById('tbplay');if(b)b.textContent='▶';});
  var _tb;
  if(_tb=document.getElementById('tbplay'))_tb.onclick=function(){if(V.paused)V.play();else V.pause();};
  if(_tb=document.getElementById('tbprev'))_tb.onclick=function(){select(Math.max(0,sel-1));seekSel();};
  if(_tb=document.getElementById('tbnext'))_tb.onclick=function(){select(Math.min(M.scenes.length-1,sel+1));seekSel();};
  if(_tb=document.getElementById('tbreplay'))_tb.onclick=function(){seekSel();V.play();};
  if(_tb=document.getElementById('tbfs'))_tb.onclick=function(){if(V.requestFullscreen)V.requestFullscreen();};
  // seek scrubber (custom — native controls are hidden so this IS the seek bar)
  var _sk=document.getElementById('tbseek');
  if(_sk){_sk.oninput=function(){var d=(V.duration&&isFinite(V.duration))?V.duration:((T&&T.total)||0);if(d)V.currentTime=(_sk.value/1000)*d;};}
  // theater / large-preview toggle (hides audio + look controls, shrinks timeline)
  var _mu=document.getElementById('tbmute');
  if(_mu){_mu.onclick=function(){V.muted=!V.muted;_mu.textContent=V.muted?'\\ud83d\\udd07':'\\ud83d\\udd0a';_mu.classList.toggle('on',V.muted);};}
  var _th=document.getElementById('tbtheater');
  if(_th){var on=(localStorage.getItem('vf_ed_theater')==='1');
    var ap=function(){var w=document.querySelector('.edwrap'),p=document.getElementById('edpv');
      if(w)w.classList.toggle('theater',on);if(p)p.classList.toggle('theater',on);_th.classList.toggle('on',on);
      if(typeof __edPositionHandles==='function')__edPositionHandles();};
    ap();
    _th.onclick=function(){on=!on;try{localStorage.setItem('vf_ed_theater',on?'1':'0');}catch(e){}ap();};}
  // native-controls fallback ONLY if the video genuinely fails to load
  V.addEventListener('error',function(){try{V.setAttribute('controls','');}catch(e){}});
  // keyboard: space = play/pause, ↑/↓ = prev/next scene (ignored while typing)
  document.addEventListener('keydown',function(e){
    if(/^(INPUT|TEXTAREA|SELECT)$/.test((e.target&&e.target.tagName)||''))return;
    var vv=document.getElementById('edvid');if(!vv)return;
    if(e.code==='Space'){e.preventDefault();if(vv.paused)vv.play();else vv.pause();}
    else if(e.key==='ArrowUp'){e.preventDefault();select(Math.max(0,sel-1));seekSel();}
    else if(e.key==='ArrowDown'){e.preventDefault();select(Math.min(M.scenes.length-1,sel+1));seekSel();}});
  document.getElementById('edexport').onclick=edExport;
  document.getElementById('edundo').onclick=async()=>{try{await jpost('/e/'+SLUG+'/undo');await reload();}catch(e){alert(e.message);}};
  __edMenuInit();
}
async function edExport(){
  const _xb=document.getElementById('edexport');
  if(_xb&&_xb.disabled)return;                       // guard: render already starting
  let p={};try{p=await fetch('/e/'+SLUG+'/pending').then(r=>r.json());}catch(e){}
  if(!p.edited_scenes&&!(p.global&&Object.keys(p.global).length)){alert('No edits yet. Edit a card, change audio, or replace a visual, then click Apply.');return;}
  const RC={card_rebake:'card-only edit — fast',scene_footage:'visual edit — re-fetches changed scenes',audio_only:'audio-only — near-instant',timeline_reflow:'structural edit — timeline reflows',none:'audio/caption only'};
  const impact=RC[p.render_class]||p.render_class;
  const note=p.llm_cost?'':'\\n✓ No AI/script cost — unchanged scenes reuse cache.';
  if(!confirm('Apply edits & re-render?\\n\\nImpact: '+impact+'\\n'
    +(p.card_edits||0)+' card · '+(p.cards_removed||0)+' card removed · '+(p.visual_uploads||0)+' visual · '
    +(p.prompt_edits||0)+' prompt · '+(p.scenes_removed||0)+' deleted'+(p.reordered?' · reordered':'')+note+'\\n\\nTakes a few minutes.'))return;
  if(_xb){_xb.disabled=true;_xb.textContent='⏳ Starting…';}
  try{const j=await jpost('/e/'+SLUG+'/export');if(j.job_url){window.location=j.job_url;return;}}
  catch(e){alert('Export failed: '+e.message);}
  if(_xb){_xb.disabled=false;_xb.innerHTML='🎬 Apply &amp; re-render';}   // re-enable only if we did NOT navigate
}
function paint(){
  const r=M.render||{},q=(r.qa||{}).verdict||'';
  document.getElementById('edsrc').textContent='Sources: '+srcMix(r.sources);
  const qe=document.getElementById('edqa');
  const QL={PASS:'✓ Quality checked',WARN:'⚠ Quality: review',FAIL:'⚠ Quality: issues found'};
  qe.textContent=q?(QL[q]||('Quality: '+q)):'';
  qe.className='edchip '+(q==='PASS'?'ok':q==='FAIL'?'bad':q==='WARN'?'warn':'');
  var _qs=(r.qa||{}).summary||'';
  qe.title=(q==='PASS'?'The final video passed automatic checks (no black frames, healthy loudness).':q==='FAIL'?'Automatic checks found a problem in the final video.':q==='WARN'?'Automatic checks flagged something minor to review.':'')+(_qs?' — '+_qs:'');
  document.getElementById('edpchips').textContent='fps '+(r.fps||'?')+' · cuts '+(r.cuts||'?')+' · transitions '+(r.transitions_motivated||0);
  if(sel>=M.scenes.length)sel=M.scenes.length-1;if(sel<0)sel=0;
  // warnings — humanise, drop the QA duplicate (shown in the header chip),
  // minor → compact chips; serious blockers → full amber strip.
  function _humanW(w){var s=String(w);
    if(/captions?\s*(are\s*)?off/i.test(s))return 'Captions are off';
    var mv=s.match(/music\s*volume\s*(\d+)\s*%/i);if(mv)return 'Music volume is '+mv[1]+'%';
    var sd=s.match(/(\d+)\s*scene[^.]*delet/i);if(sd)return sd[1]+' scene'+(sd[1]==='1'?'':'s')+' deleted';
    if(s.toLowerCase().trim().indexOf('qa')===0)return null;  // QA shown in the header chip
    return s.charAt(0).toUpperCase()+s.slice(1);}
  var minorW=(M.warnings||[]).map(_humanW).filter(Boolean);
  var serious=(q==='FAIL');
  var wel=document.getElementById('edwarn');
  if(serious){wel.className='warnstrip serious';
    wel.innerHTML='<span class=warnb>'+esc(((r.qa||{}).summary)||'This project has blocking issues — review before rendering.')+'</span>'
      +minorW.map(function(t){return '<span class=warnb>'+esc(t)+'</span>';}).join('');
    wel.style.display='flex';}
  else if(minorW.length){wel.className='warnstrip minor';
    wel.innerHTML=minorW.map(function(t){return '<span class=warnchip>'+esc(t)+'</span>';}).join('');
    wel.style.display='flex';}
  else{wel.style.display='none';}
  paintAudio();renderScenes();renderTimeline();select(sel);
  paintUnsaved();
  if(typeof __edPositionHandles==='function')__edPositionHandles();
}
async function paintUnsaved(){
  var el=document.getElementById('edunsaved');if(!el)return;
  try{
    var p=await fetch('/e/'+SLUG+'/pending').then(function(r){return r.json();});
    var n=(p.card_edits||0)+(p.cards_removed||0)+(p.visual_uploads||0)+(p.prompt_edits||0)
         +(p.scenes_removed||0)+(p.regen||0)+(p.reordered?1:0)
         +(p.global&&Object.keys(p.global).length?1:0);
    if(n>0&&p.rendered_clean){el.textContent='✓ '+n+' edit'+(n===1?'':'s')+' applied';el.className='edchip ok';el.style.display='';
      el.title='Your edits are baked into the current video. Make a new change to re-enable Apply & re-render.';}
    else if(n>0){el.textContent='● '+n+' unsaved';el.className='edchip warn';el.style.display='';
      el.title='You have '+n+' unsaved edit(s). Click “Apply & re-render” to bake them in.';}
    else{el.style.display='none';}
  }catch(e){el.style.display='none';}
}
async function edGlobal(patch){try{await jpost('/e/'+SLUG+'/global',patch);await reload();}catch(e){alert(e.message);}}
function paintAudio(){
  const g=M.global||{},el=document.getElementById('edaudio');if(!el)return;
  const mOn=g.music_enabled!==false,vol=(g.music_volume!=null?g.music_volume:1),caps=!!g.captions_enabled;
  // DOC_012 — editor signature (which look + why) + a Look override dropdown.
  const sig=M.editor_signature||{}, curLook=g.look_preset||'auto';
  const LOOKS=[['auto','Auto (detect)'],['true_crime','True Crime'],['midnight_pacific','Investigation / Spy'],['amber_chronicles','History'],['netflix_epic','Cinematic Epic'],['atlas_explained','Explainer'],['homestead','Homestead'],['standard','Standard']];
  const sigTxt = sig.look ? ('auto: '+String(sig.look).replace(/_/g,' ')+(sig.niche?' · '+String(sig.niche).replace(/_/g,' '):'')) : '';
  // Editorial RECIPE (Layer 2) — niche is a SOFT prior; each video gets a
  // deterministic recipe within the niche envelope.  'New Variation'
  // re-rolls it (different look, same niche); 'Lock Look' pins it so
  // re-renders / edits keep the same look.  Only shown in auto-look mode
  // (a forced Look preset bypasses the recipe).
  const recSum=(curLook==='auto')?(M.editorial_recipe_summary||''):'';
  const locked=!!(g.editorial_recipe_lock);
  // selective per-axis overrides (pin map / accent while the rest stays auto)
  const _ro=g.recipe_overrides||{}, _rec=M.editorial_recipe||{};
  const curMap=(_ro.map_style!=null?_ro.map_style:(_rec.map_style||''));
  function _hx(a){if(!a||a.length<3)return '#7a8aa0';var f=function(x){x=(Math.max(0,Math.min(255,x|0))).toString(16);return x.length<2?'0'+x:x;};return '#'+f(a[0])+f(a[1])+f(a[2]);}
  const curAccHex=_hx(_ro.accent!=null?_ro.accent:_rec.accent);
  const roKeys=Object.keys(_ro).filter(function(k){return _ro[k]!=null;});
  // compact toolbar — only the essentials, always visible
  el.innerHTML='<span class=audtag title="These settings apply to the WHOLE video on the next Apply &amp; re-render — not just the selected scene">WHOLE VIDEO</span>'
   +'<label title="Turn the background-music bed on or off in the re-rendered video"><input type=checkbox id=edmusic '+(mOn?'checked':'')+'> Background music</label>'
   +'<label class=audvol title="Loudness of the background-music bed in the re-rendered video. This does NOT change the preview volume — use the speaker button on the player for that.">Music volume <input type=range id=edvol min=0 max=1.5 step=0.05 value='+vol+'><span class=audval>'+(Math.round(vol*100))+'%</span></label>'
   +'<label data-tip="Show or hide captions. Updates the live preview instantly and sets whether captions are burned into the final exported video."><input type=checkbox id=edcaps '+(caps?'checked':'')+'> Captions in final video</label>'
   +'<span class=audsep></span>'
   +'<label class=audlook data-tip="Pick the cinematic look — grade, fonts and card colours. Applied to the whole video on the next Apply &amp; re-render.">Look <select id=edlook>'+LOOKS.map(function(o){return '<option value="'+o[0]+'"'+(o[0]===curLook?' selected':'')+'>'+o[1]+'</option>';}).join('')+'</select></label>'
   +(sigTxt?'<span class=audsig title="Auto-detected cinematic look">'+esc(sigTxt)+'</span>':'');
  // project look & style — moved into the collapsible panel under the preview
  var pjEl=document.getElementById('edprojbody');
  if(pjEl){
    if(curLook!=='auto'){
      pjEl.innerHTML='<div class=muted>A specific Look is selected, so the automatic editorial recipe is paused. Set Look back to “Auto (detect)” to use variation &amp; style controls.</div>';
    }else if(!recSum){
      pjEl.innerHTML='<div class=muted>No editorial recipe for this project yet.</div>';
    }else{
      pjEl.innerHTML='<div class=projrow><span class=projlbl>Recipe</span><span class=projval>'+esc(recSum)+'</span></div>'
       +'<div class=projrow>'
         +'<button class=edbtn id=edvary title="Re-roll a new look (same niche), then Apply edits to re-render">New variation</button>'
         +'<button class="'+(locked?'edbtn':'ghost')+'" id=edlock title="Pin this exact look so future re-renders keep it">'+(locked?'Locked — click to unlock':'Lock this look')+'</button>'
         +(locked?'<span class=muted>re-renders keep this look</span>':'')
       +'</div>'
       +'<div class=projrow><span class=projlbl>Map</span>'
         +'<select id=edrmap>'+['','satellite','dark','political','tactical','parchment','news'].map(function(m){return '<option value="'+m+'"'+(curMap===m?' selected':'')+'>'+(m?m:'auto')+'</option>';}).join('')+'</select>'
         +'<span class=projlbl>Accent</span><input type=color id=edracc value="'+curAccHex+'" class=projcolor>'
         +'<button class=ghost id=edrclr>Clear overrides</button>'
         +(roKeys.length?'<span class=muted>pinned: '+esc(roKeys.join(', '))+'</span>':'')
       +'</div>'
       +'<details class=projadv><summary>Advanced look controls</summary><div class=projrow projadvrow>'
         +'<select id=edrden><option value=auto>density: auto</option><option value=minimal>density: minimal</option><option value=balanced>density: balanced</option><option value=rich>density: rich</option></select>'
         +'<select id=edrmus><option value=auto>music: auto</option><option value=quiet>music: quiet</option><option value=present>music: present</option><option value=forward>music: forward</option></select>'
         +'<select id=edrmot><option value=auto>motion: auto</option><option value=calm>motion: calm</option><option value=active>motion: active</option></select>'
         +'<select id=edrtr><option value=auto>transitions: auto</option><option value=cut>cut-dominant</option><option value=dissolve>premium dissolve</option><option value=dynamic>dynamic</option></select>'
         +'<select id=edrsub><option value=auto>subtitle: auto</option><option value=minimal>subtitle: minimal</option><option value=bold_lower>subtitle: bold</option><option value=serif_lower>subtitle: serif</option><option value=mono_lower>subtitle: mono</option></select>'
         +'<select id=edrlt><option value=auto>lower-third: auto</option><option value=scrim_panel>lower-third: scrim</option><option value=case_file_tag>lower-third: case-file</option><option value=dossier_id_card>lower-third: dossier ID</option><option value=era_label>lower-third: era label</option><option value=serif_engraved>lower-third: engraved</option></select>'
       +'</div></details>';
    }
  }
  document.getElementById('edmusic').onchange=e=>edGlobal({music_enabled:e.target.checked});
  document.getElementById('edcaps').onchange=function(e){
    // Instant live toggle: edGlobal re-fetches + repaints (no video reload, so the
    // footage-matched nocap base stays put); sync the HTML overlay right after so the
    // change is visible immediately even while the preview is PAUSED.
    edGlobal({captions_enabled:e.target.checked}).then(function(){__edCaptionSync();});};
  document.getElementById('edvol').onchange=e=>edGlobal({music_volume:parseFloat(e.target.value)});
  document.getElementById('edlook').onchange=e=>edGlobal({look_preset:e.target.value});
  var _vb=document.getElementById('edvary'); if(_vb)_vb.onclick=function(){edGlobal({editorial_variation:((g.editorial_variation||0)+1),editorial_recipe_lock:null});};
  var _lb=document.getElementById('edlock'); if(_lb)_lb.onclick=function(){edGlobal(locked?{editorial_recipe_lock:null}:{editorial_recipe_lock:(M.editorial_recipe||{})});};
  function _setRO(k,v){var ro=Object.assign({},g.recipe_overrides||{});if(v===null||v===''){delete ro[k];}else{ro[k]=v;}edGlobal({recipe_overrides:ro});}
  function _hexRgb(s){s=String(s||'').replace('#','');return [parseInt(s.slice(0,2),16),parseInt(s.slice(2,4),16),parseInt(s.slice(4,6),16)];}
  var _rm=document.getElementById('edrmap'); if(_rm)_rm.onchange=function(e){_setRO('map_style',e.target.value||null);};
  var _ra=document.getElementById('edracc'); if(_ra)_ra.onchange=function(e){_setRO('accent',_hexRgb(e.target.value));};
  var _rc=document.getElementById('edrclr'); if(_rc)_rc.onclick=function(){edGlobal({recipe_overrides:{}});};
  // Advanced Look Controls — each select nudges ONE recipe axis (the rest
  // stays auto). 'auto' clears the override for that axis.
  var _dn=document.getElementById('edrden'); if(_dn)_dn.onchange=function(e){var m={minimal:0.8,balanced:1.0,rich:1.2};_setRO('density',m[e.target.value]==null?null:m[e.target.value]);};
  var _mu=document.getElementById('edrmus'); if(_mu)_mu.onchange=function(e){var m={quiet:0.7,present:0.85,forward:1.0};_setRO('music_bed',m[e.target.value]==null?null:m[e.target.value]);};
  var _mo=document.getElementById('edrmot'); if(_mo)_mo.onchange=function(e){var m={calm:1.4,active:0.85};_setRO('hold_mult',m[e.target.value]==null?null:m[e.target.value]);};
  var _tr=document.getElementById('edrtr'); if(_tr)_tr.onchange=function(e){var P={cut:['cut','cut','slow_dissolve'],dissolve:['slow_dissolve','dissolve','cut','page_wipe'],dynamic:['blur_cut','dissolve','cut','geo_push']};_setRO('transition_palette',P[e.target.value]||null);};
  // subtitle family + lower-third family — string axes map straight through.
  var _su=document.getElementById('edrsub'); if(_su)_su.onchange=function(e){_setRO('subtitle_style',e.target.value==='auto'?null:e.target.value);};
  var _ltf=document.getElementById('edrlt'); if(_ltf)_ltf.onchange=function(e){_setRO('lower_third_family',e.target.value==='auto'?null:e.target.value);};
}

function badges(sc){
  // only high-value chips: source (Footage/AI/Replaced) · Card · Edited · Missing
  var e=sc.edit_status||{},out=[];
  if(e.missing_visual||sc.missing_visual)out.push('<span class="scb miss" data-tip="This scene has no usable visual yet — replace or regenerate it before exporting.">Missing</span>');
  if(e.visual_upload)out.push('<span class="scb ai" data-tip="The visual for this scene was replaced or generated by you (not the original).">Replaced</span>');
  else{var sb=sc.source_badge||'';
    if(sb&&sb!=='Card'){var cls=/upload|ai|image|generat/i.test(sb)?'ai':'foot';
      var lab=/upload/i.test(sb)?'Replaced':(/^ai$|image|generat/i.test(sb)?'AI':sb);
      var btip=cls==='ai'?'This scene uses an AI-generated still image.':'This scene uses real stock footage.';
      out.push('<span class="scb '+cls+'" data-tip="'+btip+'">'+esc(lab)+'</span>');}}
  if(sc.card&&!sc.card.removed)out.push('<span class="scb card" data-tip="This scene shows a graphic card (title / stat / diagram) over the footage.">Card</span>');
  else if(sc.card&&sc.card.removed)out.push('<span class="scb skip" data-tip="The card on this scene is hidden — the footage underneath will show in the final video.">Card removed</span>');
  if(e.dirty)out.push('<span class="scb edited" data-tip="You have unsaved edits on this scene — they bake in on the next Apply & re-render.">Edited</span>');
  return '<div class=scbadges>'+out.join('')+'</div>';
}
function _scTip(sc,i){return '#'+(i+1)+(sc.role?'  ·  '+sc.role:'')
  +(sc.card&&sc.card.title_label?'  ·  card: '+sc.card.title_label:'')
  +'\\n'+(sc.narration||'').slice(0,160)+'\\n\\nClick to select · drag the grip to reorder · drop a file to replace the visual';}
function renderScenes(){
  document.getElementById('edscenes').innerHTML=M.scenes.map((sc,i)=>
   '<div class="scrow'+(i===sel?' sel':'')+'" data-i='+i+' draggable=true'
   +' onclick="__edsel('+sc.scene_index+')" title="'+esc(_scTip(sc,i))+'"'
   +' ondragstart="__edDragStart(event,'+i+')" ondragend="__edDragEnd(event)"'
   +' ondragover="__edDragOver(event,'+i+')" ondragleave="__edDragLeave(event)" ondrop="__edDrop(event,'+i+')">'
   +'<span class=scgrip title="Drag to reorder">\\u2630</span>'
   +'<img loading=lazy src="/e/'+SLUG+'/thumb/'+sc.scene_index+'?t='+THUMBBUST+'" onerror="this.style.visibility=\\'hidden\\'">'
   +'<div class=scmeta><div class=sctt>#'+(i+1)+' '+esc(sc.narration.slice(0,80))+'</div>'+badges(sc)+'</div>'
   +'<button class=scdots title="More actions" aria-label="More actions" onclick="__edRowMenu(event,'+i+')">\\u22ef</button>'
   +'</div>').join('');
}
function dots(n){n=n||0;return '●'.repeat(Math.max(0,Math.min(5,n)))+'○'.repeat(Math.max(0,5-n));}
function fieldInput(f){const v=esc(f.value);
  return f.multiline?('<textarea data-fkey="'+f.key+'" rows=2>'+v+'</textarea>')
                    :('<input data-fkey="'+f.key+'" value="'+v+'">');}
function sec(title,body,col,danger){
  var key=String(title).split(/[ /(]/)[0];
  return '<div class="insec'+(col?' col':'')+(danger?' danger':'')+'" data-sec="'+esc(key)+'">'
    +'<div class=inhd onclick="__edToggleSec(this)">'+title+'<span class=car>\\u25be</span></div>'
    +'<div class=inbody>'+body+'</div></div>';}
window.__edToggleSec=function(h){h.parentNode.classList.toggle('col');};
window.__edGotoSec=function(i,key){var s=document.querySelector('#edinsp .insec[data-sec="'+key+'"]');
  if(s){s.classList.remove('col');s.scrollIntoView({block:'nearest'});}};
window.__edPickFile=function(i){var f=document.getElementById('edfile');if(!f)return;
  f.onchange=function(){if(f.files&&f.files[0]){__edAcceptFile(i,f.files[0]);f.value='';}};f.click();};
function renderInspector(i){
  const sc=M.scenes[i],el=document.getElementById('edinsp');
  // ---- SCENE (story) ----
  var sceneBody='<div class=muted>'+fmt(sc.start)+'–'+fmt(sc.end)+' · '+sc.duration.toFixed(1)+'s</div>'
    +'<h3>Narration</h3><div class=narr>'+esc(sc.narration)+'</div>'
    +((sc.keywords&&sc.keywords.length)?'<h3>Keywords</h3><div>'+sc.keywords.map(k=>'<span class=kw>'+esc(k)+'</span>').join('')+'</div>':'')
    +(sc.visual?'<h3>Visual direction</h3><div class=narr>'+esc(sc.visual)+'</div>':'')
    +'<h3>Pacing</h3><div class=fld><b>Shot</b><span>'+esc(sc.shot_type||'—')+'</span></div>'
    +'<div class=fld><b>Intensity</b><span class=dots>'+dots(sc.energy)+'</span></div>';
  // ---- VISUAL (replace / regenerate / search) ----
  var up=sc.edit_status&&sc.edit_status.visual_upload;
  var visBody='<div class=edrop tabindex=0 role=button onclick="__edPickFile('+i+')" '
      +'ondragover="__edZoneOver(event)" ondragleave="__edZoneLeave(event)" ondrop="__edInspDrop(event,'+i+')">'
      +'<div class=edropic>\\u21e9</div><div class=edropt>Drag an image or video here</div>'
      +'<div class=edrops>or click to browse · JPG · PNG · WEBP · MP4 · MOV · WEBM</div></div>'
    +(up?('<div class=fld><b>Replacement</b><span>'+esc(up.split("/").pop())+'</span></div>'
      +'<img src="/e/'+SLUG+'/upload/'+sc.scene_index+'?t='+Date.now()+'" style="width:100%;border-radius:8px;margin:6px 0">'
      +'<div class=edbtns><button class=ghost onclick="__edClearVisual('+i+')" title="Restore the original generated visual">\\u21a9 Reset visual</button></div>'):'')
    +'<div class=fld2 style=margin-top:10px><input id=edsearchq placeholder="Search replacement photos…" onkeydown="if(event.key===\\'Enter\\')__edSearch('+i+')"></div>'
    +'<div class=edbtns><button class=edbtn onclick="__edSearch('+i+')">Search Pexels</button>'
      +'<button class=ghost onclick="__edSearch('+i+',\\'pixabay\\')">Pixabay</button></div>'
    +'<div id=edresults class=searchgrid></div>'
    +((sc.source_badge==='Card'||sc.visual_prompt_edited)?'<div class=muted>Tip: stock photos suit footage scenes; AI/document scenes are best left to Regenerate.</div>':'');
  // ---- CARD / TEXT ----
  var cardBody,hasCard=!!sc.card;
  if(sc.card){const c=sc.card;
    if(c.removed){cardBody='<div class=muted>Card removed — it will not render.</div><div class=edbtns><button class=ghost onclick="__edRestoreCard('+i+')" title="Bring this card back — keeps your other edits on this scene">↩ Restore card</button></div>';}
    else{cardBody='<div id=cardfields>'+(c.fields||[]).map(f=>'<div class=fld2><label class=flab>'+esc(f.label)+'</label>'+fieldInput(f)+'</div>').join('')+'</div>'
      +'<div class=edbtns><button class=edbtn onclick="__edSaveCard('+i+')">Save text</button>'
      +'<button class=ghost onclick="__edRemoveCard('+i+')">Remove card</button></div>';}
  } else {cardBody='<div class=muted>No graphic card on this scene.</div>';}
  var cardTitle='Card / Text'+(sc.card&&sc.card.title_label?(' — '+esc(sc.card.title_label)):'');
  // ---- AUDIO (per-scene narration) ----
  var audioBody='<div class=edbtns><button class=edbtn onclick="__edRegen('+i+',\\'voice\\')" title="Re-synthesize this scene\\'s narration on the next render">Re-voice this scene</button></div>'
    +'<div class=muted>Re-voicing regenerates only this scene\\'s narration. Music &amp; captions are project-wide — set them in the toolbar under the preview.</div>'
    +(sc.edit_status&&sc.edit_status.regen?'<div class=muted>↻ queued for the next render</div>':'');
  // ---- ADVANCED (AI image prompt) ----
  var promptBody='<div class=muted style="margin-bottom:6px">The text prompt used to generate this scene’s visual. Editing it changes what gets regenerated.</div>'
    +'<div id=promptbox><textarea id=edprompt rows=3>'+esc(sc.visual||'')+'</textarea></div>'
    +'<div class=edbtns><button class=edbtn onclick="__edPrompt('+i+')">Save prompt</button>'
    +'<button class=edbtn onclick="__edRegen('+i+',\\'visual\\')">Regenerate</button></div>'
    +(sc.edit_status&&sc.edit_status.regen?'<div class=muted>↻ will regenerate on next render</div>':'');
  // ---- CAPTIONS ----
  var cc=(sc.captions||{}).preview_cues||[];
  var capBody=cc.length?('<div class=narr>'+cc.map(c=>esc(c.text)).join(' ')+'</div>'):'<div class=muted>No captions in this window.</div>';
  // ---- SCENE ACTIONS + DELETE (danger) ----
  var actBody='<div class=edbtns>'
    +'<button class=ghost onclick="__edMove('+i+',1)" title="Move this scene earlier">↑ Move up</button>'
    +'<button class=ghost onclick="__edMove('+i+',0)" title="Move this scene later">↓ Move down</button>'
    +'<button class=ghost onclick="__edResetScene('+i+')" title="Undo every edit on this scene">↩ Reset scene</button></div>';
  var delBody='<div class=edbtns><button class="ghost danger"'+(M.scenes.length<=1?' disabled title="A video needs at least one scene"':' title="Remove this scene from the video"')+' onclick="__edRemoveScene('+i+')">Delete scene</button></div>';
  // ---- ACTION BAR (PRIMARY actions only; secondary behind ••• + collapsed sections) ----
  var hasCardLive=sc.card&&!sc.card.removed;
  var qCard=sc.card?('<div class=edag><span class=edagl>Card</span>'
    +(hasCardLive?'<button class=qbtn onclick="__edGotoSec('+i+',\\'Card\\')" title="Edit the card text"><i class=qic>\\u270e</i>Edit</button>'
      +'<button class=qbtn onclick="__edCardPreview('+i+')" title="Preview the graphic card"><i class=qic>\\u25a3</i>Preview</button>'
      :'<button class=qbtn onclick="__edRestoreCard('+i+')" title="Restore the removed card (keeps other edits)"><i class=qic>\\u21a9</i>Restore</button>')
    +'</div>'):'';
  var actionsBar='<div class=edactions>'
    +'<div class=edag><span class=edagl>Visual</span>'
      +'<button class=qbtn onclick="__edPickFile('+i+')" title="Replace with your own image or video"><i class=qic>\\u2913</i>Replace</button>'
      +'<button class=qbtn onclick="__edGenerateNew('+i+')" data-tip="Generate a new AI still image for this scene and show it in the live preview."><i class=qic>\\u27f3</i>Generate new</button></div>'
    +qCard
    +'<div class=edag><span class=edagl>Scene</span>'
      +'<button class=qbtn'+(i===0?' disabled':'')+' onclick="__edMove('+i+',1)" title="Move this scene earlier"><i class=qic>\\u2191</i>Up</button>'
      +'<button class=qbtn'+(i===M.scenes.length-1?' disabled':'')+' onclick="__edMove('+i+',0)" title="Move this scene later"><i class=qic>\\u2193</i>Down</button>'
      +'<button class=qbtn onclick="__edRowMenu(event,'+i+')" title="More: re-voice, reset, delete"><i class=qic>\\u22ef</i>More</button></div>'
    +'</div>';
  // ---- LAYERS (context-aware visibility toggles; live preview + card→render) ----
  var loff=(sc.edit_status&&sc.edit_status.layers_off)||[];
  var hasRepl=!!(sc.edit_status&&sc.edit_status.visual_upload);
  var capsOn=!!(sc.captions&&sc.captions.enabled);
  function _lrow(name,label,vis,note){return '<div class=edlayrow><button class="edlayeye'+(vis?' on':'')+'" onclick="__edLayer('+i+',\\''+name+'\\','+(vis?0:1)+')" title="'+(vis?'Hide':'Show')+' — '+label+'" aria-pressed="'+vis+'">'+(vis?'\\ud83d\\udc41':'\\u2298')+'</button><span class=edlayname>'+label+'</span>'+(note?'<span class=edlaynote>'+note+'</span>':'')+'</div>';}
  var layersBody='<div class=edlayers>'
    +(hasRepl?_lrow('visual','Visual',loff.indexOf('visual')<0,'replacement'):'')
    +(sc.card?_lrow('card','Card / Graphic',(!sc.card.removed&&loff.indexOf('card')<0),''):'')
    +_lrow('captions','Captions',capsOn,'whole video')
    +'</div>'+(sc.card?'<div class=muted>Hide a layer to preview without it. A hidden card is also removed from the final video.</div>':'');
  el.innerHTML='<div class=insptitle><h2>Scene '+(i+1)+'</h2><span class=insprole>'+esc(sc.role||'scene')+'</span></div>'
    +'<input type=file id=edfile accept=".jpg,.jpeg,.png,.webp,.mp4,.mov,.webm,.mkv,.m4v" style="display:none">'
    +actionsBar
    +sec('Layers',layersBody,false)
    +sec('Visual',visBody,true)
    +sec(cardTitle,cardBody,!hasCard)
    +sec('Scene details',sceneBody,true)
    +sec('Advanced',promptBody,true)
    +'<details class=tips><summary>Editing Studio tips</summary><ul>'
      +'<li>Click a scene on the left — or a clip on the timeline — to select it.</li>'
      +'<li><b>Space</b> = play / pause · <b>↑ / ↓</b> = previous / next scene.</li>'
      +'<li>Drag a photo or video onto the scene or preview to replace its visual. Your original is never overwritten.</li>'
      +'<li>Drag scenes by the grip to reorder; zoom the timeline with <b>−</b> / <b>+</b>.</li>'
      +'<li>Click <b>Apply edits &amp; re-render</b> when done — unchanged scenes reuse cache.</li>'
    +'</ul></details>';
  // P2 — live card text: instant draft-overlay update + debounced silent save (no
  // reload, so the field keeps focus while typing).
  var _cf=document.getElementById('cardfields');
  if(_cf){var _cst;_cf.addEventListener('input',function(){
    clearTimeout(_cst);_cst=setTimeout(function(){__edSaveCardSilent(i);},600);});}
}
function dur(s){s=Math.max(0,(s.end||0)-(s.start||0));return s.toFixed(1)+'s';}
function _tlRuler(tot){
  var step=tot<=20?5:tot<=60?10:tot<=180?30:tot<=600?60:120;
  var out='<div class="tk tkruler"><div class=tklab></div><div class="tkrow tkrul">';
  for(var t=0;t<=tot+0.01;t+=step){var l=100*t/tot;
    out+='<span class=rtick style="left:'+l+'%"><i></i><b>'+fmt(t)+'</b></span>';}
  return out+'</div></div>';
}
function renderTimeline(){
  const tot=T.total||1,tr=T.tracks;
  function track(lab,items,cls,labf,tipf,key){
    var hidden=key&&TLHIDE[key];
    var lh=key?('<div class="tklab tklabtog" onclick="__edTLtoggle(\\''+key+'\\')" title="Show / hide this track">'+(hidden?'\\u25b8':'\\u25be')+' '+lab+'</div>')
              :('<div class=tklab>'+lab+'</div>');
    if(hidden)return '<div class="tk tkmin">'+lh+'<div class="tkrow tkrowmin"></div></div>';
    let row='<div class=tk>'+lh+'<div class=tkrow>';
    (items||[]).forEach(b=>{const l=100*b.start/tot,w=Math.max(0.4,100*(b.end-b.start)/tot);
      const bg=(cls==='v'&&b.scene_index!=null)?(';background-image:url(/e/'+SLUG+'/thumb/'+b.scene_index+'?t='+THUMBBUST+')'):'';
      const tip=tipf?tipf(b):esc(labf(b));
      // Bug 3 — width-tiered labels: narrow = number only; medium = number +
      // short keyword (first word); wide = number + narration. Full text = tooltip.
      var _lab=esc(labf(b)||''),_kw=_lab.split(' ').slice(0,1).join(' ');
      var inner=(cls==='v'&&b.scene_index!=null)
        ?('<span class=blknum>'+(b.scene_index+1)+'</span>'+(w>=6?'<span class=blktxt>'+_lab+'</span>':(w>=3.2?'<span class=blktxt>'+_kw+'</span>':'')))
        :_lab;
      var dragattr=(cls==='v'&&b.scene_index!=null)?(' draggable=true ondragstart="__edTLDragStart(event,'+b.scene_index+')" ondragend="__edTLDragEnd(event)" ondragover="__edTLDragOver(event,'+b.scene_index+')" ondrop="__edTLDrop(event,'+b.scene_index+')"'):'';
      row+='<div class="blk '+cls+'" data-i="'+(b.scene_index!=null?b.scene_index:'')+'" style="left:'+l+'%;width:'+w+'%'+bg+'"'+dragattr+' '
        +(b.scene_index!=null?'onclick="__edsel('+b.scene_index+')"':'')+' title="'+tip+'">'+inner+'</div>';});
    return row+'</div></div>';
  }
  document.getElementById('edtl').innerHTML=
    '<div class=tlhead><span class=tlt>Timeline</span><span class=sp></span>'
     +'<button class=zbtn onclick="__edResetLayout()" title="Reset panel layout" style="width:auto;padding:0 9px;font-size:11px">\\u21ba Layout</button>'
     +'<button class=zbtn id=tzout title="Zoom out">\\u2212</button>'
     +'<button class=zbtn id=tzfit title="Fit whole video">\\u25ad</button>'
     +'<button class=zbtn id=tzin title="Zoom in">+</button></div>'
   +'<div class=tlbody id=edtlb style="width:'+(TLZOOM*100)+'%">'
     +_tlRuler(tot)
     +track('Visual',tr.visual,'v',b=>b.label,b=>esc(b.label)+' · '+dur(b))
     +track('Cards',tr.card,'c',b=>b.kind_label,null,'card')
     +track('Captions',tr.caption,'cap',b=>b.text,null,'caption')
     +track('Music',tr.music,'m',b=>b.label,null,'music')
     +track('Voice',tr.voice,'vo',b=>'VO',null,'voice')
     +'<div class=playhead id=edph><div class=phknob title="Drag to scrub the video"></div></div>'
   +'</div>';
  var z;
  if(z=document.getElementById('tzin'))z.onclick=function(){setZoom(TLZOOM*1.6);};
  if(z=document.getElementById('tzout'))z.onclick=function(){setZoom(TLZOOM/1.6);};
  if(z=document.getElementById('tzfit'))z.onclick=function(){setZoom(1);};
  __edTLInit();onTime();scrollTLToSel();
}
// Bug 2 — timeline click-to-seek + playhead drag-scrub with the real mouse.
function __edTLSeekX(clientX){
  var body=document.getElementById('edtlb'),v=document.getElementById('edvid');if(!body||!v)return;
  var row=body.querySelector('.tkrow');if(!row)return;
  var rb=row.getBoundingClientRect(),tot=(T&&T.total)||v.duration||1;
  var t=Math.max(0,Math.min(tot-0.05,((clientX-rb.left)/Math.max(1,rb.width))*tot));
  try{v.currentTime=t;}catch(e){}
  onTime();
}
function __edTLInit(){
  var body=document.getElementById('edtlb');if(!body)return;
  var dragging=false;
  // Use MOUSE events (fired by real mice, touch-compat, AND automation) with
  // document-level move/up so a drag that leaves the timeline still scrubs +
  // clamps. (pointerdown alone is not emitted by some automation backends.)
  function _move(e){if(dragging)__edTLSeekX(e.clientX);}
  function _up(){if(dragging){dragging=false;body.classList.remove('scrubbing');
    document.removeEventListener('mousemove',_move);document.removeEventListener('mouseup',_up);}}
  function _down(e){
    var t=e.target;
    if(t&&t.closest&&(t.closest('.blk')||t.closest('.tklab')))return;  // blocks select scenes; labels toggle tracks
    dragging=true;body.classList.add('scrubbing');
    document.addEventListener('mousemove',_move);document.addEventListener('mouseup',_up);
    __edTLSeekX(e.clientX);e.preventDefault();
  }
  body.addEventListener('mousedown',_down);
  // pointerdown too, for real pen/touch users (harmless duplicate via the guard)
  body.addEventListener('pointerdown',function(e){if(e.pointerType&&e.pointerType!=='mouse')_down(e);});
}
function setZoom(z){TLZOOM=Math.max(1,Math.min(9,z));var b=document.getElementById('edtlb');
  if(b)b.style.width=(TLZOOM*100)+'%';onTime();scrollTLToSel();}
function scrollTLToSel(){var b=document.querySelector('#edtlb .blk.v.sel');
  if(b&&b.scrollIntoView)try{b.scrollIntoView({inline:'center',block:'nearest'});}catch(e){}}
// Safe preview anchor INSIDE a scene. The assembled MP4 DISSOLVES between scenes
// (up to ~0.85s — assemble.py _TRANSITIONS), so seeking to the raw scene start lands
// in the incoming dissolve where the PREVIOUS scene is still visible in the flattened
// MP4. Land far enough in to clear any dissolve (~0.9s floor), grow a little for long
// scenes, but never past ~half the scene (short-scene clamp). Preview-only — the final
// render timing is NOT affected.
function __edSeekAnchor(sc){if(!sc)return 0;var st=+sc.start||0,en=+sc.end||st,
  dur=Math.max(0.1,en-st),off=Math.min(Math.max(0.9,dur*0.15),dur*0.5);return st+off;}
function seekSel(){var V=document.getElementById('edvid'),s=M.scenes[sel];
  if(V&&s){try{V.currentTime=__edSeekAnchor(s);}catch(e){}}}
function onTime(){
  const v=document.getElementById('edvid'),tot=T.total||1,ph=document.getElementById('edph');
  if(ph){const row=document.querySelector('#edtlb .tkrow');
    if(row){ph.style.left=(row.offsetLeft+(v.currentTime/tot)*row.offsetWidth)+'px';}}
  const tt=document.getElementById('tbtime');if(tt)tt.textContent=fmt(v.currentTime)+' / '+fmt(v.duration||tot);
  var sk=document.getElementById('tbseek'),sd=(v.duration&&isFinite(v.duration))?v.duration:tot;if(sk&&sd&&document.activeElement!==sk)sk.value=Math.round((v.currentTime/sd)*1000);
  // auto-highlight current scene (row + timeline block). When PAUSED — i.e. a seek /
  // scrub / playhead-drag, NOT live playback — also move the INSPECTOR to the same
  // scene so row + inspector + timeline + playhead + preview never diverge (the bug
  // was that the inspector lagged on the previously-clicked scene). Skip the inspector
  // rebuild while a field inside #edinsp is focused (don't steal focus / thrash).
  const cur=M.scenes.findIndex(s=>v.currentTime>=s.start&&v.currentTime<s.end);
  if(cur>=0&&cur!==sel){markSel(cur);
    if(v.paused){var ae=document.activeElement;
      if(!(ae&&ae.closest&&ae.closest('#edinsp')))renderInspector(cur);}}
  __edDraftSync();
  __edCaptionSync();
}
function markSel(i){sel=i;document.querySelectorAll('.scrow').forEach(r=>r.classList.toggle('sel',+r.dataset.i===i));
  document.querySelectorAll('.blk.v').forEach(b=>b.classList.toggle('sel',+b.dataset.i===M.scenes[i].scene_index));}
// ===== LIVE DRAFT PREVIEW LAYER STACK =====
// Draft layers sit ON TOP of the rendered MP4 and reflect the user's edits for the
// scene at the current playback time. Re-render only bakes the FINAL MP4.
function __edActiveScene(){var v=document.getElementById('edvid');var t=v?v.currentTime:0;
  var i=M.scenes.findIndex(s=>t>=s.start&&t<s.end);
  if(i<0){var n=M.scenes.length;            // end-of-video / past last s.end → clamp to final scene
    if(n&&t>=M.scenes[n-1].start)return n-1;return (sel>=0?sel:0);}
  return i;}
function __edLayerOff(sc,name){var L=(sc&&sc.edit_status&&sc.edit_status.layers_off)||[];return L.indexOf(name)>=0;}
// Kinds whose live HTML draft is clean + compact (a lower-third / simple title). Complex
// motion-graphic kinds (statement/kinetic, classified/document reveal, callout, numbers,
// maps, routes, infographics…) are NOT here — they fall back to an honest note instead
// of a broken oversized approximation, and the base MP4 keeps showing them.
var __edSimpleCardKinds={lower_third:1,lower:1,title_card:1,title:1,chapter:1,chapter_card:1,
  location:1,location_title:1,label:1,name_reveal:1,name:1};
function __edCardSimple(c){return !!(c&&__edSimpleCardKinds[String(c.graphic_kind||'').toLowerCase()])
  && String(c.graphic_body||'').length<=160;}
function __edDraftCardHTML(c){var t=esc(c.graphic_text||''),b=esc(c.graphic_body||''),k=c.graphic_kind||'';
  if(/location|label|lower|name|chapter/.test(k))return '<div class=dclower>'+t+(b?' &middot; '+b:'')+'</div>';
  return '<div class=dcfull><div class=dctitle>'+t+'</div>'+(b?'<div class=dcbody>'+b+'</div>':'')+'</div>';}
var _draftKey='';
function __edDraftSync(force){
  var lv=document.getElementById('edlayvisual'),lc=document.getElementById('edlaycard'),lbl=document.getElementById('eddraftlbl');
  if(!lv||!lc)return;
  var i=__edActiveScene(),sc=M.scenes[i];if(!sc){lv.hidden=lc.hidden=true;if(lbl)lbl.hidden=true;return;}
  var es=sc.edit_status||{},uf=es.visual_upload,cardOff=__edLayerOff(sc,'card'),visOff=__edLayerOff(sc,'visual');
  // The base MP4 ALREADY renders every card, so only draw a card overlay when the card
  // was EDITED this session (preview the NEW text) or hidden — never duplicate an
  // unedited card.
  var cardEdited=!!(sc.card&&sc.card.edited&&!sc.card.removed&&!cardOff);
  // cheap re-sync guard (rebuild only when the meaningful state changes)
  var key=i+'|'+(uf||'')+'|'+visOff+'|'+(cardEdited?JSON.stringify(sc.card.graphic_text)+'|'+JSON.stringify(sc.card.graphic_body)+'|'+(sc.card.graphic_kind||''):'')+'|'+cardOff+'|'+(sc.card&&sc.card.removed)+'|'+cardEdited;
  if(!force&&key===_draftKey)return; _draftKey=key;
  // VISUAL layer — full-stage replacement
  if(uf&&!visOff){var url='/e/'+SLUG+'/upload/'+sc.scene_index,isVid=/\\.(mp4|mov|webm|mkv|m4v)$/i.test(uf);
    if(isVid){lv.style.backgroundImage='';lv.className='edlay';lv.innerHTML='<video class=edlayvid src="'+url+'?t='+THUMBBUST+'" muted playsinline loop autoplay></video>';}
    else{lv.innerHTML='';lv.className='edlay kb';lv.style.backgroundImage='url('+url+'?t='+THUMBBUST+')';}
    lv.hidden=false;}
  else{lv.hidden=true;lv.style.backgroundImage='';lv.innerHTML='';}
  // CARD layer — hidden-card note · clean draft of an EDITED simple card · honest note
  // for an EDITED complex card · nothing for an unedited card (base MP4 already shows it).
  if(sc.card&&(sc.card.removed||cardOff)){
    lc.className='edlay';lc.innerHTML='<div class=edcardhide>Card hidden — the footage underneath will show in the final video</div>';lc.hidden=false;}
  else if(cardEdited&&__edCardSimple(sc.card)){
    lc.className='edlay edcarddraft';lc.innerHTML=__edDraftCardHTML(sc.card);lc.hidden=false;}
  else if(cardEdited){
    lc.className='edlay';lc.innerHTML='<div class=edcardnote><span class=edcardnotedot></span>Text saved — the final card styling appears after re-render.</div>';lc.hidden=false;}
  else{lc.hidden=true;lc.innerHTML='';}
  if(lbl)lbl.hidden=!((!lv.hidden)||(!lc.hidden));
}
function select(i){if(i<0||i>=M.scenes.length)return;markSel(i);renderInspector(i);scrollTLToSel();__edDraftSync(true);
  const r=document.querySelector('.scrow[data-i="'+i+'"]');if(r)r.scrollIntoView({block:'nearest'});}
window.__edsel=function(i){const v=document.getElementById('edvid');
  const pos=M.scenes.findIndex(s=>s.scene_index===i);const k=pos>=0?pos:i;
  if(v&&M.scenes[k]){try{v.currentTime=__edSeekAnchor(M.scenes[k]);}catch(e){}}select(k);};
function sidx(i){return M.scenes[i].scene_index;}
window.__edSaveCard=async function(i){const vals={};
  document.querySelectorAll('#cardfields [data-fkey]').forEach(el=>vals[el.dataset.fkey]=el.value);
  try{await jpost('/e/'+SLUG+'/scene/'+sidx(i)+'/card',vals);await reload();}
  catch(e){alert('Save failed: '+e.message);}};
// P2 — live card text: build a draft card from the current field values (approx),
// + a silent debounced save (no reload, so the textarea keeps focus while typing).
window.__edSaveCardSilent=async function(i){const vals={};
  document.querySelectorAll('#cardfields [data-fkey]').forEach(el=>vals[el.dataset.fkey]=el.value);
  try{await jpost('/e/'+SLUG+'/scene/'+sidx(i)+'/card',vals);
    // refresh the read-model (NOT the inspector DOM, so the field keeps focus) so
    // the draft card overlay shows the REAL re-encoded card text.
    try{var a=await Promise.all([fetch('/e/'+SLUG+'/manifest.json').then(r=>r.json()),
      fetch('/e/'+SLUG+'/timeline.json').then(r=>r.json())]);if(a[0]&&a[0].scenes){M=a[0];T=a[1];}}catch(e){}
    __edDraftSync(true);if(typeof paintUnsaved==='function')paintUnsaved();}catch(e){}};
window.__edRemoveCard=async function(i){if(!confirm('Remove the card from this scene?'))return;
  try{await jpost('/e/'+SLUG+'/scene/'+sidx(i)+'/card/remove');await reload();}catch(e){alert(e.message);}};
window.__edRestoreCard=async function(i){
  try{await jpost('/e/'+SLUG+'/scene/'+sidx(i)+'/card/restore');await reload();toast('Card restored','success');}catch(e){toast('Could not restore card: '+e.message,'error');}};
window.__edResetScene=async function(i){
  try{await jpost('/e/'+SLUG+'/scene/'+sidx(i)+'/reset');await reload();toast('Scene reset to original','success');}catch(e){toast('Reset failed: '+e.message,'error');}};
window.__edUpload=function(i){const f=document.getElementById('edfile');
  if(!f||!f.files||!f.files[0]){toast('Choose an image or video first, or drag one onto the scene.','info');return;}
  __edAcceptFile(i,f.files[0]);};
window.__edPrompt=async function(i){const p=(document.getElementById('edprompt')||{}).value||'';
  try{await jpost('/e/'+SLUG+'/scene/'+sidx(i)+'/prompt',{prompt:p});await reload();toast('Prompt saved','success');}catch(e){toast('Could not save prompt: '+e.message,'error');}};
window.__edRegen=async function(i,kind){
  var t=toast(kind==='voice'?'Queuing re-voice…':'Queuing regenerate…','progress',{progress:true});t.progress(0.5);
  try{const r=await fetch('/e/'+SLUG+'/scene/'+sidx(i)+'/regen/'+kind,{method:'POST'});
    if(!r.ok)throw new Error('HTTP '+r.status);await reload();
    t.done(kind==='voice'?'Will re-voice on next render':'Will regenerate on next render');}
  catch(e){t.fail('Could not queue '+(kind==='voice'?'re-voice':'regenerate'),e.message);}};
// Issue 3 — Generate new visual NOW (on-demand fal.ai still) and show it live.
window.__edGenerateNew=async function(i){
  __edStatus('Generating new visual…');
  var t=toast('Generating new visual…','progress',{progress:true});t.progress(0.4);
  try{const r=await fetch('/e/'+SLUG+'/scene/'+sidx(i)+'/generate',{method:'POST'});
    var j={};try{j=await r.json();}catch(e){}
    if(!r.ok||!j.ok){t.fail('Could not generate a new visual',(j&&j.error)||('HTTP '+r.status));__edStatus('');return;}
    t.progress(0.85);
    await reload();                 // manifest now has the generated visual_override
    select(i);seekSel();            // seek into this scene so the full center preview shows it
    __edDraftSync(true);
    __edStatus('New visual added',1800);
    t.done('New visual generated and added to the live preview.');}
  catch(e){t.fail('Could not generate a new visual',e.message);__edStatus('');}};
window.__edMove=async function(i,up){
  try{const r=await fetch('/e/'+SLUG+'/scene/'+sidx(i)+'/move/'+up,{method:'POST'});
    if(r.ok)await reload();}catch(e){alert(e.message);}};
// ===== TOAST / FEEDBACK SYSTEM =====
function toast(msg,kind,opts){opts=opts||{};
  var host=document.getElementById('edtoasts');
  if(!host){host=document.createElement('div');host.id='edtoasts';host.className='edtoasts';document.body.appendChild(host);}
  var t=document.createElement('div');t.className='edtoast '+(kind||'info');
  var ic={info:'\\u2139',success:'\\u2713',error:'\\u26a0',progress:'\\u2191'}[kind||'info']||'';
  t.innerHTML='<span class=etic>'+ic+'</span><div class=etmain><span class=ettxt></span>'
    +(opts.progress?'<div class=etbar><i></i></div>':'')
    +(opts.details?'<details class=etdet><summary>Details</summary><pre></pre></details>':'')+'</div>'
    +'<button class=etx aria-label=Dismiss>\\u00d7</button>';
  t.querySelector('.ettxt').textContent=msg;
  if(opts.details)t.querySelector('.etdet pre').textContent=opts.details;
  t.querySelector('.etx').onclick=close;
  host.appendChild(t);requestAnimationFrame(function(){t.classList.add('in');});
  var timer=null,ttl=(opts.ttl!=null)?opts.ttl:(kind==='error'?6500:3000);
  function close(){if(timer)clearTimeout(timer);t.classList.add('out');setTimeout(function(){if(t.parentNode)t.remove();},240);}
  if(kind!=='progress'&&ttl>0)timer=setTimeout(close,ttl);
  return {el:t,
    text:function(m){t.querySelector('.ettxt').textContent=m;return this;},
    progress:function(f){var b=t.querySelector('.etbar>i');if(b)b.style.width=Math.round(Math.max(0,Math.min(1,f))*100)+'%';return this;},
    done:function(m){t.className='edtoast success in';t.querySelector('.etic').textContent='\\u2713';t.querySelector('.ettxt').textContent=m||'Done';var b=t.querySelector('.etbar>i');if(b)b.style.width='100%';if(timer)clearTimeout(timer);timer=setTimeout(close,2200);return this;},
    fail:function(m,d){t.className='edtoast error in';t.querySelector('.etic').textContent='\\u26a0';t.querySelector('.ettxt').textContent=m||'Something went wrong';if(d){var det=document.createElement('details');det.className='etdet';det.innerHTML='<summary>Details</summary><pre></pre>';det.querySelector('pre').textContent=d;t.querySelector('.etmain').appendChild(det);}if(timer)clearTimeout(timer);timer=setTimeout(close,7000);return this;},
    close:close};
}
function __edBusy(el,on,label){if(!el)return;if(on){el.dataset._t=el.dataset._t||el.innerHTML;el.disabled=true;el.classList.add('isbusy');if(label)el.innerHTML='<span class=spin></span> '+label;}
  else{el.disabled=false;el.classList.remove('isbusy');if(el.dataset._t){el.innerHTML=el.dataset._t;delete el.dataset._t;}}}

// ===== FILE VALIDATION + UPLOAD (drag-file-onto-scene to replace visual) =====
var ED_IMG=['jpg','jpeg','png','webp'], ED_VID=['mp4','mov','webm','mkv','m4v'], ED_MAXMB=250;
function __edFileErr(f){var ext=(f.name.split('.').pop()||'').toLowerCase();
  if(ED_IMG.indexOf(ext)<0&&ED_VID.indexOf(ext)<0)return 'Unsupported file “.'+ext+'”. Use JPG, PNG, WEBP, MP4, MOV or WEBM.';
  if(f.size===0)return 'That file appears to be empty.';
  var mb=f.size/1048576;if(mb>ED_MAXMB)return 'File is too large ('+mb.toFixed(0)+' MB). The limit is '+ED_MAXMB+' MB.';
  return null;}
function __edAcceptFile(i,file){
  var err=__edFileErr(file);if(err){toast(err,'error');return;}
  var t=toast('Uploading '+file.name+'…','progress',{progress:true});
  var fd=new FormData();fd.append('file',file);
  var xhr=new XMLHttpRequest();xhr.open('POST','/e/'+SLUG+'/scene/'+sidx(i)+'/upload');
  xhr.upload.onprogress=function(e){if(e.lengthComputable)t.progress(e.loaded/e.total);};
  xhr.onload=function(){var r={};try{r=JSON.parse(xhr.responseText);}catch(e){}
    if(xhr.status>=200&&xhr.status<300&&r.ok){t.done('Visual replaced — re-render to bake it into the final video');reload();}
    else{t.fail('Upload failed: '+((r&&r.error)||('HTTP '+xhr.status)),'Scene '+(i+1)+' · '+file.name);}};
  xhr.onerror=function(){t.fail('Upload failed. Please check the file and try again.');};
  xhr.send(fd);}
function __edIsFileDrag(ev){try{return ev.dataTransfer&&Array.prototype.indexOf.call(ev.dataTransfer.types||[],'Files')>=0;}catch(e){return false;}}

// ---- drag-and-drop scene reorder (non-destructive; Move up/down remain as fallback) ----
let DRAG=null;
function __edClearDrop(){document.querySelectorAll('.scrow').forEach(function(r){
  r.classList.remove('dragging','dropabove','dropbelow','filedrop');});}
window.__edDragStart=function(ev,i){
  var _xb=document.getElementById('edexport');
  if(_xb&&_xb.disabled){try{ev.preventDefault();}catch(e){}return;}  // no reorder while a render is starting/running
  DRAG=i;ev.dataTransfer.effectAllowed='move';
  try{ev.dataTransfer.setData('text/plain',String(i));}catch(e){}
  ev.currentTarget.classList.add('dragging');};
window.__edRowMenu=function(ev,i){ev.stopPropagation();ev.preventDefault();__rmClose();
  var sc=M.scenes[i],hasCard=!!(sc&&sc.card);
  var html='<button class=edpopitem onclick="__rmClose();__edRegen('+i+',\\'voice\\')" title="Re-synthesize the narration">Re-voice narration</button>'
    +'<button class=edpopitem onclick="__rmClose();__edResetScene('+i+')">Reset this scene</button>';
  if(sc&&sc.edit_status&&sc.edit_status.visual_upload){html+='<button class=edpopitem onclick="__rmClose();__edClearVisual('+i+')" title="Restore the original generated visual">Use original visual</button>';}
  if(hasCard){html+=(sc.card.removed
      ?'<button class=edpopitem onclick="__rmClose();__edRestoreCard('+i+')">Restore card</button>'
      :'<button class=edpopitem onclick="__rmClose();__edCardPreview('+i+')">Preview card</button>');}
  html+='<div class=edpopsep></div><button class="edpopitem danger" onclick="__rmClose();__edRemoveScene('+i+')">Delete scene</button>';
  var m=document.createElement('div');m.className='edpop rmenu';m.id='__rmenu';m.innerHTML=html;document.body.appendChild(m);
  var r=ev.currentTarget.getBoundingClientRect(),mw=m.offsetWidth,mh=m.offsetHeight;
  var left=Math.min(r.right-mw,window.innerWidth-mw-8),top=r.bottom+4;
  if(top+mh>window.innerHeight-8)top=Math.max(8,r.top-mh-4);
  m.style.position='fixed';m.style.left=Math.max(8,left)+'px';m.style.top=top+'px';m.style.right='auto';};
window.__rmClose=function(){var m=document.getElementById('__rmenu');if(m)m.remove();};
window.__edDragEnd=function(ev){DRAG=null;__edClearDrop();};
window.__edDragOver=function(ev,i){
  if(__edIsFileDrag(ev)){ev.preventDefault();ev.dataTransfer.dropEffect='copy';
    ev.currentTarget.classList.add('filedrop');return;}
  if(DRAG===null||DRAG===i)return;ev.preventDefault();
  ev.dataTransfer.dropEffect='move';
  var row=ev.currentTarget,rc=row.getBoundingClientRect(),below=(ev.clientY-rc.top)>rc.height/2;
  row.classList.toggle('dropbelow',below);row.classList.toggle('dropabove',!below);};
window.__edDragLeave=function(ev){ev.currentTarget.classList.remove('dropabove','dropbelow','filedrop');};
window.__edDrop=async function(ev,i){ev.preventDefault();
  if(__edIsFileDrag(ev)){ev.currentTarget.classList.remove('filedrop');__edClearDrop();
    var f=ev.dataTransfer.files&&ev.dataTransfer.files[0];if(f){select(i);__edAcceptFile(i,f);}return;}
  var src=DRAG;__edClearDrop();
  if(src===null||src===i){DRAG=null;return;}
  var row=ev.currentTarget,rc=row.getBoundingClientRect(),below=(ev.clientY-rc.top)>rc.height/2;
  var b=below?i+1:i;                 // insert before original-visible index b
  var to=(src<b)?b-1:b;              // convert to (visible-without-src) index
  DRAG=null;
  try{var r=await fetch('/e/'+SLUG+'/scene/'+sidx(src)+'/reorder/'+to,{method:'POST'}).then(function(x){return x.json();});
    if(r&&r.ok===false){toast('Reorder failed: '+(r.error||'unknown'),'error');return;}
    sel=to;await reload();
    select(Math.max(0,Math.min(to,M.scenes.length-1)));  // keep the moved scene selected
    toast('Scene moved','success');
  }catch(e){toast('Reorder failed: '+e.message,'error');}};
// ---- timeline direct drag-reorder (P1) — reuses the override reorder backend ----
function __edTLClearDrop(){document.querySelectorAll('#edtlb .blk.tldropl,#edtlb .blk.tldropr').forEach(function(x){x.classList.remove('tldropl','tldropr');});}
window.__edReorderApply=async function(src,to){
  if(src===null||src<0||to<0||to===src){DRAG=null;return;}
  DRAG=null;
  try{var r=await fetch('/e/'+SLUG+'/scene/'+sidx(src)+'/reorder/'+to,{method:'POST'}).then(function(x){return x.json();});
    if(r&&r.ok===false){toast('Reorder failed: '+(r.error||'unknown'),'error');return;}
    sel=to;await reload();select(Math.max(0,Math.min(to,M.scenes.length-1)));toast('Scene moved','success');
  }catch(e){toast('Reorder failed: '+e.message,'error');}};
window.__edTLDragStart=function(ev,i){var xb=document.getElementById('edexport');if(xb&&xb.disabled){try{ev.preventDefault();}catch(e){}return;}
  DRAG=i;ev.dataTransfer.effectAllowed='move';try{ev.dataTransfer.setData('text/plain',String(i));}catch(e){}ev.currentTarget.classList.add('tldragging');};
window.__edTLDragOver=function(ev,i){if(DRAG===null||DRAG===i)return;ev.preventDefault();ev.dataTransfer.dropEffect='move';
  __edTLClearDrop();var b=ev.currentTarget,rc=b.getBoundingClientRect();b.classList.add(((ev.clientX-rc.left)>rc.width/2)?'tldropr':'tldropl');};
window.__edTLDragEnd=function(ev){DRAG=null;if(ev&&ev.currentTarget)ev.currentTarget.classList.remove('tldragging');__edTLClearDrop();};
window.__edTLDrop=function(ev,i){ev.preventDefault();var src=DRAG;__edTLClearDrop();
  document.querySelectorAll('#edtlb .blk.tldragging').forEach(function(x){x.classList.remove('tldragging');});
  if(src===null||src===i){DRAG=null;return;}
  var b=ev.currentTarget,rc=b.getBoundingClientRect(),after=(ev.clientX-rc.left)>rc.width/2;
  var ins=after?i+1:i,to=(src<ins)?ins-1:ins;__edReorderApply(src,to);};
// ---- preview + inspector dropzone file-drop wiring ----
window.__edZoneOver=function(ev){if(!__edIsFileDrag(ev))return;ev.preventDefault();ev.dataTransfer.dropEffect='copy';ev.currentTarget.classList.add('filedrop');};
window.__edZoneLeave=function(ev){if(ev.relatedTarget&&ev.currentTarget.contains&&ev.currentTarget.contains(ev.relatedTarget))return;ev.currentTarget.classList.remove('filedrop');};
window.__edPreviewDrop=function(ev){if(!__edIsFileDrag(ev))return;ev.preventDefault();
  ev.currentTarget.classList.remove('filedrop');
  var f=ev.dataTransfer.files&&ev.dataTransfer.files[0];if(f)__edAcceptFile(sel,f);};
window.__edInspDrop=function(ev,i){if(!__edIsFileDrag(ev))return;ev.preventDefault();
  ev.currentTarget.classList.remove('filedrop');
  var f=ev.dataTransfer.files&&ev.dataTransfer.files[0];if(f)__edAcceptFile(i,f);};
// ===== RESIZABLE PANELS (left / right / timeline) — persisted to localStorage =====
var ED_RZ={left:[250,460,308],right:[320,520,366],tl:[190,420,210]};  // [min,max,default]
function __edLayoutGet(){try{return JSON.parse(localStorage.getItem('vf_ed_layout')||'{}')||{};}catch(e){return {};}}
function __edLayoutSet(o){try{localStorage.setItem('vf_ed_layout',JSON.stringify(o));}catch(e){}}
function __edApplyLayout(){var w=document.querySelector('.edwrap');if(!w)return;var o=__edLayoutGet();
  if(o.left!=null)w.style.setProperty('--ed-left',o.left+'px');
  if(o.right!=null)w.style.setProperty('--ed-right',o.right+'px');
  if(o.tl!=null)w.style.setProperty('--ed-tl',o.tl+'px');
  __edPositionHandles();}
function __edPositionHandles(){var w=document.querySelector('.edwrap');if(!w)return;
  var wb=w.getBoundingClientRect();
  var sc=document.querySelector('.edscenes'),ins=document.querySelector('.edinspector'),tl=document.querySelector('.edtimeline');
  var hL=document.getElementById('edrzL'),hR=document.getElementById('edrzR'),hT=document.getElementById('edrzT');
  if(!sc||!hL)return;
  var topH=(tl?tl.getBoundingClientRect().top:wb.bottom)-wb.top-7;
  hL.style.left=(sc.getBoundingClientRect().right-wb.left+3)+'px';hL.style.height=topH+'px';
  hR.style.left=(ins.getBoundingClientRect().left-wb.left-10)+'px';hR.style.height=topH+'px';
  hT.style.top=((tl?tl.getBoundingClientRect().top:wb.bottom)-wb.top-10)+'px';}
function __edInitResize(){
  var w=document.querySelector('.edwrap');if(!w)return;__edApplyLayout();
  function drag(handle,axis){var H=document.getElementById(handle);if(!H)return;
    H.addEventListener('pointerdown',function(ev){
      if(window.innerWidth<=1000)return;             // disabled in stacked layout
      ev.preventDefault();try{H.setPointerCapture(ev.pointerId);}catch(e){}H.classList.add('dragging');
      document.body.style.cursor=axis==='tl'?'row-resize':'col-resize';
      function mv(e){var wb=w.getBoundingClientRect(),b=ED_RZ[axis],v;
        if(axis==='left')v=e.clientX-wb.left;
        else if(axis==='right')v=wb.right-e.clientX;
        else v=wb.bottom-e.clientY;
        v=Math.max(b[0],Math.min(b[1],Math.round(v)));
        w.style.setProperty('--ed-'+axis,v+'px');__edPositionHandles();}
      function up(e){try{H.releasePointerCapture(ev.pointerId);}catch(_){}H.classList.remove('dragging');
        document.body.style.cursor='';H.removeEventListener('pointermove',mv);H.removeEventListener('pointerup',up);
        var o=__edLayoutGet();o[axis]=parseInt(w.style.getPropertyValue('--ed-'+axis))||ED_RZ[axis][2];__edLayoutSet(o);}
      H.addEventListener('pointermove',mv);H.addEventListener('pointerup',up);});
    H.addEventListener('dblclick',function(){w.style.removeProperty('--ed-'+axis);
      var o=__edLayoutGet();delete o[axis];__edLayoutSet(o);__edPositionHandles();toast('Panel size reset','success');});
  }
  drag('edrzL','left');drag('edrzR','right');drag('edrzT','tl');
  window.addEventListener('resize',__edPositionHandles);
}
window.__edResetLayout=function(){var w=document.querySelector('.edwrap');if(!w)return;
  w.style.removeProperty('--ed-left');w.style.removeProperty('--ed-right');w.style.removeProperty('--ed-tl');
  __edLayoutSet({});__edPositionHandles();toast('Layout reset to default','success');};
window.__edRemoveScene=async function(i){if(!confirm('Delete this whole scene from the video?'))return;
  try{const r=await fetch('/e/'+SLUG+'/scene/'+sidx(i)+'/remove-scene',{method:'POST'});
    if(r.ok)await reload();}catch(e){alert(e.message);}};
window.__edCardPreview=function(i){window.open('/e/'+SLUG+'/card-preview/'+sidx(i),'_blank');};
window.__edClearVisual=async function(i){try{await fetch('/e/'+SLUG+'/scene/'+sidx(i)+'/clear-visual',{method:'POST'});await reload();toast('Original visual restored','success');}catch(e){toast('Could not restore: '+e.message,'error');}};
window.__edSearch=async function(i,src){
  const q=(document.getElementById('edsearchq')||{}).value||'',box=document.getElementById('edresults');
  if(!q){box.innerHTML='<div class=muted>Type a search term.</div>';return;}
  box.innerHTML='<div class=muted>Searching…</div>';
  try{const d=await fetch('/e/'+SLUG+'/search?src='+(src||'pexels')+'&q='+encodeURIComponent(q)).then(r=>r.json());
    if(d.error){box.innerHTML='<div class=muted>'+esc(d.error)+'</div>';return;}
    if(!d.results||!d.results.length){box.innerHTML='<div class=muted>No results.</div>';return;}
    box.innerHTML=d.results.map(function(r){return '<img class=sres src="'+r.thumb+'" title="'+esc(r.src+" "+r.w+"x"+r.h)+'" data-f="'+encodeURIComponent(r.full)+'" onclick="__edUseSearch('+i+',this.dataset.f)">';}).join('');
  }catch(e){box.innerHTML='<div class=muted>Search failed.</div>';}
};
window.__edUseSearch=async function(i,full){
  try{await jpost('/e/'+SLUG+'/scene/'+sidx(i)+'/use-search',{url:decodeURIComponent(full)});await reload();
    toast('Visual replaced — re-render to bake it into the final video','success');}
  catch(e){toast('Could not use that image: '+e.message,'error');}
};
window.__edLayer=async function(i,name,on){
  try{await jpost('/e/'+SLUG+'/scene/'+sidx(i)+'/layer/'+name+'/'+on);await reload();
    var n=name==='card'?'Card / Graphic':name==='captions'?'Captions':'Visual';
    toast(n+(String(on)==='1'?' shown':' hidden')+(name==='card'?' — also affects the final video':' in live preview'),'success');}
  catch(e){toast('Could not toggle layer: '+e.message,'error');}};
boot();
})();
</script>
{% endraw %}
"""


def _manifest_stale(d: Path) -> bool:
    mp = d / "editor_manifest.json"
    if not mp.exists():
        return True
    try:
        mt = mp.stat().st_mtime
        mp4 = next(iter(sorted(d.glob("*.mp4"))), None)
        if mp4 and mt < mp4.stat().st_mtime:
            return True
        ovr = d / "edits" / "user_overrides.json"
        if ovr.exists() and mt < ovr.stat().st_mtime:
            return True
        return False
    except OSError:
        return True


def _editor_not_found(slug: str) -> str:
    """P6 — beginner-friendly missing-project page (with a Back to dashboard
    action) instead of a bare browser 404."""
    body = (
        '<div style="max-width:560px;margin:13vh auto;text-align:center;color:#cdd6e0">'
        '<div style="font-size:46px;margin-bottom:12px">🔍</div>'
        '<h1 style="font-size:22px;margin:0 0 10px">This project could not be found</h1>'
        '<p style="color:#8a93a6;line-height:1.55;margin:0 auto;max-width:440px">'
        'It may have been moved, renamed, or deleted. Return to the dashboard and '
        'choose another project.</p>'
        '<a href="/" style="display:inline-block;margin-top:20px;background:#2f6df0;'
        'color:#fff;text-decoration:none;padding:11px 22px;border-radius:9px;'
        'font-weight:600">&larr; Back to dashboard</a>'
        '</div>')
    return _page(body, title="Project not found")


@app.get("/e/<slug>")
def editor_page(slug: str):
    # P6 — render a friendly not-found page instead of abort(404).
    ok = bool(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,59}", slug or ""))
    d = OUT / slug
    if not (ok and d.is_dir() and d.resolve().parent == OUT.resolve()
            and (d / "script.json").exists()):
        return _editor_not_found(slug), 404
    return _page(_EDITOR, slug=slug, title=slug)


@app.get("/e/<slug>/manifest.json")
def editor_manifest_json(slug: str):
    d = _safe_dir(slug)
    if not (d / "script.json").exists():
        abort(404)
    if _manifest_stale(d):
        from . import editor_manifest as EM
        try:
            EM.write_manifest(d)
        except Exception as e:  # noqa: BLE001
            return jsonify(error=f"manifest build failed: {e}"), 500
    return send_file((d / "editor_manifest.json").resolve(),
                     mimetype="application/json", max_age=0)


@app.get("/e/<slug>/timeline.json")
def editor_timeline_json(slug: str):
    d = _safe_dir(slug)
    p = d / "editor_cache" / "timeline.json"
    if _manifest_stale(d) or not p.exists():
        from . import editor_manifest as EM
        try:
            # thumbs=False: the manifest.json request (fired in parallel by
            # fetchData) extracts posters — don't double the ffmpeg work here.
            EM.write_manifest(d, thumbs=False)
        except Exception as e:  # noqa: BLE001
            return jsonify(error=f"timeline build failed: {e}"), 500
    if not p.exists():
        abort(404)
    return send_file(p.resolve(), mimetype="application/json", max_age=0)


@app.get("/e/<slug>/thumb/<int:idx>")
def editor_thumb(slug: str, idx: int):
    d = _safe_dir(slug)
    p = d / "editor_cache" / "thumbs" / f"sc{idx:03d}.jpg"
    if not p.exists():
        abort(404)
    return send_file(p.resolve(), mimetype="image/jpeg", max_age=3600)


# ── Phase 2: non-destructive editing (overrides) ─────────────────────── #
def _ed_dir(slug: str) -> Path:
    d = _safe_dir(slug)
    if not (d / "script.json").exists():
        abort(404)
    return d


# ── RC5: Review-Editor manual-replacement relevance gate ─────────────────── #
# A manual replacement (drag-drop / file upload / chosen stock-search result) is
# one of the three producers that NEVER reach the in-pipeline fail-closed gate
# (RC5_VISUAL_RELEVANCE_REPORT.md, "STEP 9"). A user can therefore drop an
# anime/game DVD cover, a strategy-game UI screenshot, an infographic, a poster
# or a party logo straight onto a scene and it sails into the render. This is the
# SMALLEST safe integration: it does NOT redesign the editor UI — it only runs
# `visual_relevance.classify_junk_metadata` (filename/metadata) + `graphic_signal`
# (pixels) on the candidate and returns a status the route surfaces in its JSON.
# HARD-junk classes are BLOCKED by default; an explicit advanced override
# (force=1) is allowed but LOGGED (never a silent bypass). A clean asset keeps the
# existing replacement behaviour unchanged.
_HARD_JUNK_HINT = ("game", "anime", "dvd", "cover", "ui", "hud", "screenshot",
                   "poster", "infographic", "logo", "meme")


def _editor_scene_context(run_dir, idx: int) -> dict:
    """Best-effort narration/keywords for scene `idx` (for the on-topic junk
    exemption + a richer log entry). Never raises."""
    try:
        from . import editor_manifest as EM
        base = (EM._load_json(Path(run_dir) / "script.baseline.json")
                or EM._load_json(Path(run_dir) / "script.json") or {})
        scs = base.get("scenes", [])
        if 0 <= idx < len(scs):
            s = scs[idx]
            return {"narration": s.get("narration", "") or "",
                    "keywords": " ".join(s.get("keywords", []) or [])}
    except Exception:                                          # noqa: BLE001
        pass
    return {"narration": "", "keywords": ""}


def _log_replacement_decision(run_dir, idx: int, entry: dict) -> None:
    """Append one manual-replacement decision (accept / warning / reject /
    override) to edits/replacement_audit.jsonl — the audit trail an advanced
    override is NEVER allowed to skip. Records timestamps + asset hashes so a
    forced-junk replacement is always attributable. Never raises."""
    try:
        import time as _t
        rec = dict(entry)
        rec.setdefault("ts", _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime()))
        rec.setdefault("scene_index", idx)
        p = Path(run_dir) / "edits" / "replacement_audit.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:                                          # noqa: BLE001
        pass


def validate_replacement_asset(data, filename, *, is_video=False, narration="",
                               keywords="", source_url="", query=""):
    """Pure, Flask-free relevance check for a manual Review-Editor replacement.

    Runs `visual_relevance.classify_junk_metadata` over the filename + source
    URL + search query (with the scene narration as the on-topic exemption) and
    `visual_relevance.graphic_signal` over the candidate's pixels, then returns a
    status dict the route surfaces verbatim:

        {
          "status": "accepted" | "warning" | "rejected",
          "hard_junk": bool,          # a blocked-by-default junk class
          "reason": str,              # human-readable cause ("" when clean)
          "hits": [str],              # junk-metadata token hits
          "graphic_dom": float|None,  # pixel designed-graphic score
          "looks_designed": bool,
          "asset_sha1": str,          # hash of the candidate bytes (for the log)
          "asset": str,               # candidate filename
        }

    POLICY: a HARD junk class (game/anime/dvd/cover/ui/poster/infographic/logo/
    meme/…) OR a clearly designed-graphic frame ⇒ "rejected" (the caller blocks
    it unless force=1). A softer signal (graphic-ish but under the hard bar, or
    metadata junk the narration partially excuses) ⇒ "warning" (accepted, but the
    UI is told). Everything clean ⇒ "accepted". Defensive: any internal failure
    degrades to "accepted" with a note so the editor is never bricked — the
    in-pipeline post-pass + the frame-sweep QA remain the backstops.

    `data` may be bytes OR a path to the already-written candidate file; we hash
    the bytes and (if a real image/clip is available) run the pixel probe on it.
    """
    import hashlib as _hl
    import tempfile as _tf

    out = {"status": "accepted", "hard_junk": False, "reason": "", "hits": [],
           "graphic_dom": None, "looks_designed": False, "asset_sha1": "",
           "asset": filename or ""}
    tmp_path = None
    try:
        from . import visual_relevance as VR

        # --- candidate bytes + hash (for the audit log + a temp file to probe) ---
        raw = None
        probe_path = None
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        elif data:
            try:
                probe_path = str(data)
                raw = Path(data).read_bytes()
            except Exception:                                  # noqa: BLE001
                raw = None
        if raw:
            try:
                out["asset_sha1"] = _hl.sha1(raw).hexdigest()
            except Exception:                                  # noqa: BLE001
                out["asset_sha1"] = ""

        # --- (1) metadata hard-reject (filename + url + query vs narration) ---
        try:
            isj, reason, hits = VR.classify_junk_metadata(
                title=filename or "", slug=filename or "", url=source_url or "",
                query=query or "", narration=narration or "")
        except Exception:                                      # noqa: BLE001
            isj, reason, hits = False, "", []
        out["hits"] = list(hits or [])
        meta_hard = bool(isj) and any(
            any(h.startswith(k) or k in h for k in _HARD_JUNK_HINT)
            for h in (hits or []))

        # --- (2) pixel designed-graphic probe (best-effort) ---
        if probe_path is None and raw:
            try:
                ext = (Path(filename).suffix or ".bin").lower()
                fd, tmp_path = _tf.mkstemp(suffix=ext)
                os.close(fd)
                Path(tmp_path).write_bytes(raw)
                probe_path = tmp_path
            except Exception:                                  # noqa: BLE001
                probe_path = None
        looks_designed = False
        gdom = None
        if probe_path:
            try:
                g = VR.graphic_signal(probe_path, is_video=bool(is_video))
                gdom = g.get("graphic_dom")
                looks_designed = bool(g.get("looks_designed"))
            except Exception:                                  # noqa: BLE001
                looks_designed, gdom = False, None
        out["graphic_dom"] = gdom
        out["looks_designed"] = looks_designed

        # --- (3) verdict ---
        if isj and meta_hard:
            out["status"] = "rejected"
            out["hard_junk"] = True
            out["reason"] = (reason or "junk-metadata")
            return out
        if looks_designed:
            out["status"] = "rejected"
            out["hard_junk"] = True
            out["reason"] = (f"designed-graphic-not-footage "
                             f"(graphic_dom={gdom})")
            return out
        if isj:
            # metadata junk that wasn't a HARD class (e.g. a soft 'template'
            # token) → warn, accept, let the user decide.
            out["status"] = "warning"
            out["reason"] = (reason or "junk-metadata") + " (soft)"
            return out
        out["status"] = "accepted"
        return out
    except Exception as e:                                     # noqa: BLE001
        out["status"] = "accepted"
        out["reason"] = f"validation-skipped ({type(e).__name__})"
        return out
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:                                  # noqa: BLE001
                pass


def _guard_manual_replacement(d, idx, data, filename, *, is_video, source_url="",
                              query="", force=False, override_reason=""):
    """Shared wrapper used by every manual-replacement route. Validates the
    candidate, LOGS the decision, and returns either (None, status_dict) to BLOCK
    (caller returns the rejected JSON, original visual untouched) or
    (status_dict, None) to PROCEED with the save. A `force` override never blocks
    but is always logged with the original-asset hash + chosen asset + reason."""
    ctx = _editor_scene_context(d, idx)
    st = validate_replacement_asset(
        data, filename, is_video=is_video, narration=ctx.get("narration", ""),
        keywords=ctx.get("keywords", ""), source_url=source_url, query=query)
    blocked = (st["status"] == "rejected")
    if blocked and force:
        # advanced override: allow, but record an immutable audit entry.
        _log_replacement_decision(d, idx, {
            "action": "override_forced", "result": "accepted_via_override",
            "status_before_override": "rejected", "reason": st.get("reason"),
            "hits": st.get("hits"), "graphic_dom": st.get("graphic_dom"),
            "asset": st.get("asset"), "asset_sha1": st.get("asset_sha1"),
            "source_url": source_url, "query": query,
            "override_reason": (override_reason or "")[:300]})
        st = dict(st)
        st["status"] = "accepted_override"
        return st, None
    _log_replacement_decision(d, idx, {
        "action": "validate", "result": st["status"], "reason": st.get("reason"),
        "hits": st.get("hits"), "graphic_dom": st.get("graphic_dom"),
        "asset": st.get("asset"), "asset_sha1": st.get("asset_sha1"),
        "source_url": source_url, "query": query})
    if blocked:
        return None, st
    return st, None


@app.post("/e/<slug>/scene/<int:idx>/card")
def editor_card_save(slug: str, idx: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    fields = request.get_json(silent=True) or {}
    res = EM.save_card_override(d, idx, fields)
    return jsonify(res), (200 if res.get("ok") else 400)


@app.post("/e/<slug>/scene/<int:idx>/card/remove")
def editor_card_remove(slug: str, idx: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.remove_card_override(d, idx))


@app.post("/e/<slug>/scene/<int:idx>/card/restore")
def editor_card_restore(slug: str, idx: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.restore_card(d, idx))


@app.post("/e/<slug>/scene/<int:idx>/layer/<name>/<int:on>")
def editor_set_layer(slug: str, idx: int, name: str, on: int):
    """Live-preview layer visibility toggle (card hidden also reaches the render)."""
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    res = EM.set_layer(d, idx, name, bool(on))
    return jsonify(res), (200 if res.get("ok") else 400)


@app.post("/e/<slug>/scene/<int:idx>/upload")
def editor_upload(slug: str, idx: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    up = request.files.get("file")
    if not up or not up.filename:
        return jsonify(ok=False, error="Please choose an image or video file."), 400
    # P5 — server-side size cap (defense-in-depth; the JS caps at 250 MB too).
    # Measure via the stream so we reject BEFORE reading a huge file into memory.
    MAX_BYTES = 300 * 1024 * 1024
    try:
        up.stream.seek(0, 2)
        _sz = up.stream.tell()
        up.stream.seek(0)
    except Exception:  # noqa: BLE001
        _sz = up.content_length or 0
    if _sz and _sz > MAX_BYTES:
        return jsonify(ok=False,
                       error=f"That file is too large ({_sz // (1024 * 1024)} MB). "
                             f"The limit is 300 MB."), 413
    blob = up.read()
    # RC5 relevance gate — block hard-junk replacements (game/anime/cover/UI/
    # poster/infographic/logo/meme/designed-graphic) BEFORE saving; original
    # visual stays intact. force=1 is an explicit, LOGGED advanced override.
    _is_vid = (Path(up.filename).suffix or "").lower() in (
        ".mp4", ".mkv", ".webm", ".mov", ".m4v")
    _force = str(request.values.get("force", "")).strip().lower() in ("1", "true", "yes", "on")
    _ovr_reason = (request.values.get("override_reason", "") or "")
    proceed, blocked = _guard_manual_replacement(
        d, idx, blob, up.filename, is_video=_is_vid, force=_force,
        override_reason=_ovr_reason)
    if proceed is None:
        return jsonify(ok=False, status="rejected", relevance=blocked,
                       error="This image looks like a " + (blocked.get("reason") or
                             "non-documentary graphic") + ". It was blocked to keep "
                             "the video relevant. Choose real footage, or re-submit "
                             "with the advanced override if you're certain."), 422
    res = EM.save_visual_override(d, idx, up.filename, blob)
    if res.get("ok"):
        res["relevance"] = {k: proceed[k] for k in ("status", "reason", "hits",
                            "graphic_dom", "looks_designed") if k in proceed}
    return jsonify(res), (200 if res.get("ok") else 400)


@app.post("/e/<slug>/scene/<int:idx>/reset")
def editor_reset_scene(slug: str, idx: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.reset_scene(d, idx))


@app.post("/e/<slug>/reset-all")
def editor_reset_all(slug: str):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.reset_all(d))


@app.get("/e/<slug>/upload/<int:idx>")
def editor_upload_file(slug: str, idx: int):
    d = _ed_dir(slug)
    updir = d / "edits" / "uploads"
    cand = sorted(updir.glob(f"sc{idx:03d}.*")) if updir.exists() else []
    if not cand:
        abort(404)
    return send_file(cand[0].resolve(), max_age=0)


@app.get("/e/<slug>/pending")
def editor_pending(slug: str):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.pending_summary(d))


@app.post("/e/<slug>/global")
def editor_global(slug: str):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.save_global_override(d, request.get_json(silent=True) or {}))


@app.post("/e/<slug>/scene/<int:idx>/remove-scene")
def editor_remove_scene(slug: str, idx: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.remove_scene(d, idx))


@app.post("/e/<slug>/scene/<int:idx>/move/<int:up>")
def editor_move_scene(slug: str, idx: int, up: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.move_scene(d, idx, -1 if up else 1))


@app.post("/e/<slug>/scene/<int:idx>/reorder/<int:to>")
def editor_reorder_scene(slug: str, idx: int, to: int):
    """Drag-reorder: move baseline scene `idx` to visible position `to`."""
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.reorder_scene(d, idx, to))


@app.post("/e/<slug>/scene/<int:idx>/prompt")
def editor_prompt(slug: str, idx: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    p = (request.get_json(silent=True) or {}).get("prompt", "")
    return jsonify(EM.save_visual_prompt(d, idx, p))


@app.post("/e/<slug>/scene/<int:idx>/regen/<kind>")
def editor_regen(slug: str, idx: int, kind: str):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.set_regen(d, idx, kind))


@app.post("/e/<slug>/scene/<int:idx>/generate")
def editor_generate(slug: str, idx: int):
    """Issue 3 — on-demand AI still generation. Generates a NEW image NOW via fal.ai,
    saves it as a visual_override (so it persists, undoes, and the final render uses it),
    and returns the path so the editor shows it INSTANTLY in the live draft layer. No
    full production render is triggered."""
    import time as _time
    import tempfile as _tf
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    cfg = load_config()
    if not cfg.fal_key:
        return jsonify(ok=False,
                       error="AI image generation isn't configured (no FAL_KEY). "
                             "Use Replace to add your own image instead."), 400
    base = EM._load_json(d / "script.baseline.json") or EM._load_json(d / "script.json") or {}
    scs = base.get("scenes", [])
    if not (0 <= idx < len(scs)):
        abort(404)
    s = dict(scs[idx])
    osc = EM.load_overrides(d)["scenes"].get(EM.scene_stable_id(d, idx), {})
    prompt = (osc.get("visual_prompt_override") or s.get("visual")
              or s.get("narration") or "").strip()
    if not prompt:
        return jsonify(ok=False, error="This scene has no prompt to generate from."), 400
    # fal caches by (model, prompt); vary the prompt so each click yields a FRESH,
    # differently-composed still (and a different cache key).
    hints = ["alternate composition", "wide establishing shot, environmental context",
             "closer detail, shallow depth of field", "different angle, dramatic natural light",
             "fresh framing, eye-level, documentary"]
    seed = int(_time.time() * 1000) % 1000000
    gprompt = f"{prompt}. {hints[seed % len(hints)]} (v{seed})."
    try:
        from . import footage as F
        tmp = Path(_tf.mkdtemp(prefix="edgen_")) / "gen.jpg"
        ok = F._fal_image(gprompt, tmp, cfg.fal_key, cfg.fal_model, seed)
        if not ok or not tmp.exists() or tmp.stat().st_size < 4000:
            return jsonify(ok=False, error="Generation didn't return an image — please try again."), 502
        data = tmp.read_bytes()
        res = EM.save_visual_override(d, idx, "generated.jpg", data)
        try:
            tmp.unlink()
        except Exception:  # noqa: BLE001
            pass
        if not res.get("ok"):
            return jsonify(res), 400
        res["generated"] = True
        return jsonify(res), 200
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=f"Generation failed: {type(e).__name__}: {e}"), 500


@app.get("/e/<slug>/card-preview/<int:idx>")
def editor_card_preview(slug: str, idx: int):
    """Cheap preview: re-render JUST this scene's card PNG (no video render)."""
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    import tempfile
    ov = EM.load_overrides(d)["scenes"]
    sid = EM.scene_stable_id(d, idx)
    osc = ov.get(sid, {})
    base = EM._load_json(d / "script.baseline.json") or EM._load_json(d / "script.json") or {}
    scs = base.get("scenes", [])
    if not (0 <= idx < len(scs)):
        abort(404)
    s = dict(scs[idx])
    cto = osc.get("card_text_override") or {}
    kind = (s.get("graphic_kind") or "").strip()
    if not kind or osc.get("card_removed"):
        abort(404)
    gt = cto.get("graphic_text", s.get("graphic_text", ""))
    gb = cto.get("graphic_body", s.get("graphic_body", ""))
    try:
        from . import footage as F
        from .script_gen import Scene
        sc = Scene(index=0, narration=s.get("narration", ""), keywords=s.get("keywords", []),
                   graphic_kind=kind, graphic_text=gt, graphic_body=gb)
        tmp = Path(tempfile.mkdtemp(prefix="cardprev_"))
        cfg = load_config()
        out = F.build_graphic_images([sc], cfg, tmp, theme={"accent": (214, 64, 54)})
        png = next((tmp / v for v in (out or {}).values() if (tmp / v).exists()), None)
        if not png:
            png = next(iter(sorted(tmp.glob("*.png"))), None)
        if not png or not png.exists():
            return jsonify(error="card preview unavailable — it will render on Apply"), 503
        return send_file(png.resolve(), mimetype="image/png", max_age=0)
    except Exception as e:  # noqa: BLE001
        return jsonify(error=f"preview unavailable: {str(e)[:120]}"), 503


@app.post("/e/<slug>/undo")
def editor_undo(slug: str):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.undo(d))


@app.post("/e/<slug>/scene/<int:idx>/clear-visual")
def editor_clear_visual(slug: str, idx: int):
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    return jsonify(EM.clear_visual(d, idx))


@app.get("/e/<slug>/search")
def editor_search(slug: str):
    """Stock/web image search for a replacement visual. Pexels + Pixabay are
    curated stock (no YouTube thumbnails / blog graphics / posters by nature);
    a min-dimension gate drops low-quality results."""
    _ed_dir(slug)
    q = (request.args.get("q") or "").strip()
    src = (request.args.get("src") or "pexels").lower()
    if not q:
        return jsonify(results=[], error="Type something to search.")
    cfg = load_config()
    import requests
    out = []
    try:
        if src == "pexels":
            key = cfg.pexels_api_key or os.environ.get("PEXELS_API_KEY", "")
            if not key:
                return jsonify(results=[], error="Stock search needs a Pexels API key (PEXELS_API_KEY).")
            r = requests.get("https://api.pexels.com/v1/search",
                             headers={"Authorization": key},
                             params={"query": q, "per_page": 16, "orientation": "landscape"},
                             timeout=15)
            for p in (r.json().get("photos", []) if r.ok else []):
                out.append({"thumb": p["src"]["medium"], "full": p["src"]["large2x"],
                            "src": "Pexels", "w": p["width"], "h": p["height"],
                            "credit": p.get("photographer", "")})
        elif src == "pixabay":
            key = os.environ.get("PIXABAY_API_KEY", "")
            if not key:
                return jsonify(results=[], error="Pixabay search needs PIXABAY_API_KEY.")
            r = requests.get("https://pixabay.com/api/",
                             params={"key": key, "q": q, "image_type": "photo",
                                     "per_page": 16, "safesearch": "true",
                                     "orientation": "horizontal"}, timeout=15)
            for p in (r.json().get("hits", []) if r.ok else []):
                out.append({"thumb": p["webformatURL"], "full": p.get("largeImageURL") or p["webformatURL"],
                            "src": "Pixabay", "w": p["imageWidth"], "h": p["imageHeight"],
                            "credit": p.get("user", "")})
        else:
            return jsonify(results=[], error="Unknown source.")
    except Exception as e:  # noqa: BLE001
        return jsonify(results=[], error=f"Search failed: {str(e)[:100]}")
    # quality gate: drop low-res (text-heavy/poster crops tend to be small/odd-ratio)
    out = [x for x in out if x.get("w", 0) >= 800 and x.get("h", 0) >= 450]
    return jsonify(results=out[:12])


@app.post("/e/<slug>/scene/<int:idx>/use-search")
def editor_use_search(slug: str, idx: int):
    """Download a chosen search result + set it as the scene's visual override."""
    d = _ed_dir(slug)
    body = request.get_json(silent=True) or {}
    url = body.get("url", "")
    query = (body.get("query") or body.get("q") or "")
    force = str(body.get("force", "")).strip().lower() in ("1", "true", "yes", "on")
    ovr_reason = body.get("override_reason", "") or ""
    if not url.startswith(("http://", "https://")):
        return jsonify(ok=False, error="bad url"), 400
    import requests
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        if "image" not in r.headers.get("content-type", "") or len(r.content) < 2000:
            return jsonify(ok=False, error="not a usable image"), 400
        from . import editor_manifest as EM
        ext = ".png" if "png" in r.headers.get("content-type", "") else ".jpg"
        fname = "search" + ext
        # RC5 relevance gate — same fail-closed policy as upload, with the source
        # url + search query feeding the metadata classifier.
        proceed, blocked = _guard_manual_replacement(
            d, idx, r.content, fname, is_video=False, source_url=url,
            query=query, force=force, override_reason=ovr_reason)
        if proceed is None:
            return jsonify(ok=False, status="rejected", relevance=blocked,
                           error="That result looks like a " + (blocked.get("reason")
                                 or "non-documentary graphic") + " and was blocked. "
                                 "Pick a different image, or override if certain."), 422
        res = EM.save_visual_override(d, idx, fname, r.content)
        if res.get("ok"):
            res["relevance"] = {k: proceed[k] for k in ("status", "reason", "hits",
                                "graphic_dom", "looks_designed") if k in proceed}
        return jsonify(res), (200 if res.get("ok") else 400)
    except Exception as e:  # noqa: BLE001
        return jsonify(ok=False, error=f"download failed: {str(e)[:100]}"), 500


@app.post("/e/<slug>/export")
def editor_export(slug: str):
    """Phase 3: fold the user's non-destructive edits into script.json
    (baseline-safe, LLM-skipped for card edits) and kick a re-render that
    reuses every unchanged scene from cache. Returns the job URL for the
    editor to navigate to the live progress page."""
    d = _ed_dir(slug)
    from . import editor_manifest as EM
    res = EM.apply_overrides(d)
    if not res.get("ok"):
        return jsonify(res), 400
    summ = EM.pending_summary(d)
    job_id = _start_render_job(d, "Applying your edits & re-rendering…")
    return jsonify(ok=True, job_url=url_for("job_page", job_id=job_id),
                   changed=res.get("changed"), summary=summ)


# =====================================================================
# HEYGEN B-ROLL TOOL — separate lightweight pipeline
# =====================================================================
# This is a STAND-ALONE feature.  None of these routes / templates /
# handlers touch the main documentary engine (write_script / render_
# from_script / pipeline.py / look_dna / assemble / captions / etc.).
#
# Flow:
#   GET  /heygen-broll/new                 → upload form
#   POST /heygen-broll/upload              → save file + start job
#   GET  /heygen-broll/job/<id>            → progress page
#   GET  /heygen-broll/job/<id>/status     → JSON status (polled)
#   GET  /heygen-broll/job/<id>/download   → final MP4
#   GET  /heygen-broll/job/<id>/plan       → plan.json (debug)

_HG_FORM = """
<div class=topbar>
  <div class=brand>
    <div class=logo style="background:#22c55e">B</div>
    <div><b>HeyGen B-Roll Tool</b><small>simple stock-overlay on
      avatar/talking-head videos · no captions / no music / no AI imagery</small></div>
  </div>
  <a class=btn href="{{ url_for('index') }}">← Studio</a>
</div>

{% if error %}<div class=err>{{error}}</div>{% endif %}
<form method=post action="{{ url_for('hg_upload') }}" class=genform
      id=hgform enctype="multipart/form-data" onsubmit="return goHg()">

<div class=section>
  <h2>1 · Upload Avatar Video</h2>
  <p class=shint>Upload an MP4 from HeyGen or any other avatar / talking-head
    tool. The original audio is preserved — we only overlay stock B-roll
    on top of the visible avatar at the moments where it adds context.</p>
  <label>MP4 file <small>(required · up to ~40 min)</small></label>
  <input type=file name=video accept="video/mp4,.mp4" required>
  <label style="margin-top:14px">Video title <small>(optional but
    recommended — improves stock search)</small></label>
  <input name=title placeholder="How elderly people fall asleep in 5 minutes">
  <label style="margin-top:14px">Script / transcript <small>(optional — if
    pasted, we use this verbatim; otherwise we transcribe the audio
    with Whisper)</small></label>
  <textarea name=script style="min-height:140px"
    placeholder="Paste the full narration here if you have it (any language). Leave empty to auto-transcribe from the uploaded audio."></textarea>
</div>

<div class=section>
  <h2>2 · B-Roll Coverage</h2>
  <p class=shint>How much of the video should be covered by stock footage.
    Never reaches 100% — the avatar always returns naturally between
    bursts so the video still feels human.</p>
  <select name=coverage>
    <option value=light    >Light · 50% coverage (avatar-heavy)</option>
    <option value=balanced selected>Balanced · 65% coverage (default)</option>
    <option value=heavy    >Heavy · 75% coverage</option>
    <option value=maximum  >Maximum · 80% coverage</option>
  </select>
</div>

<div class=section>
  <h2>3 · Avatar Corner Mode <small style="text-transform:none;letter-spacing:0;color:#7c8aa0;font-weight:500"> · optional</small></h2>
  <p class=shint>Keep the avatar visible as a circular portrait in a
    corner while B-roll plays full-screen. The corner rotates every
    ~3.5 minutes (top-left → top-right → bottom-right → bottom-left)
    so the eye stays fresh on long videos. Lightweight — no AI
    background removal.</p>
  <label class=tog>
    <input type=checkbox name=pip value=1>
    <span class=sw></span>
    <span class=tl>Avatar Corner Mode<small>round portrait, rotating
      corners every ~3.5 min</small></span>
  </label>
</div>

<div class=section>
  <h2>4 · Stock Source</h2>
  <p class=shint>Free sources only — Pexels primary, Pixabay fallback.
    No Shutterstock, no AI image generation, no paid APIs.</p>
  <div class=api-row>
    <span class="api-pill {{ 'ok' if pexels_ok else 'miss' }}">
      <span class=dot></span>Pexels {{ '✓' if pexels_ok else '✗ no key' }}
    </span>
    <span class="api-pill {{ 'ok' if pixabay_ok else 'miss' }}">
      <span class=dot></span>Pixabay {{ '✓' if pixabay_ok else '✗ no key' }}
    </span>
    <span class="api-pill {{ 'ok' if anthropic_ok else 'miss' }}">
      <span class=dot></span>Anthropic {{ '✓' if anthropic_ok else '✗ heuristic keywords' }}
    </span>
  </div>
  {% if not pexels_ok and not pixabay_ok %}
  <div class=note style="border-color:#f59e0b;background:#3a2a10;color:#f0b656">
    ⚠ Both stock sources missing. Set <code>PEXELS_API_KEY</code> or
    <code>PIXABAY_API_KEY</code> in <code>.env</code> before running.
  </div>
  {% endif %}
</div>

<button type=submit id=hgbtn style="width:100%;font-size:16px;padding:15px">
  ✨ Add B-Roll →
</button>
</form>

<div id=hgwait style="display:none;position:fixed;inset:0;
  background:rgba(8,11,16,.94);z-index:999;
  align-items:center;justify-content:center;text-align:center">
  <div>
    <div style="width:54px;height:54px;margin:0 auto 22px;border:5px solid #222a35;
      border-top-color:#22c55e;border-radius:50%;animation:spin 1s linear infinite"></div>
    <h2 style=margin:0>Uploading and starting job…</h2>
    <p class=sub>Long videos may take a minute to upload. Don't close the tab.</p>
  </div>
</div>
<style>@keyframes spin{to{transform:rotate(360deg)}}</style>
<script>
function goHg(){
  var b=document.getElementById('hgbtn'); b.disabled=true;
  b.textContent='Uploading…';
  document.getElementById('hgwait').style.display='flex';
  return true;
}
</script>
"""

_HG_JOB = """
<div class=topbar>
  <div class=brand>
    <div class=logo style="background:#22c55e">B</div>
    <div><b>HeyGen B-Roll Tool</b><small>job {{job_id}}</small></div>
  </div>
  <a class=btn href="{{ url_for('hg_new') }}">+ Another</a>
</div>

<h1>Rendering…</h1>
<p class=sub>{{title or 'Avatar video B-roll overlay'}}</p>
<div class=card>
  <div class=bar><div class=fill id=hgfill style="background:#22c55e"></div></div>
  <p id=hgmsg>Starting…</p>
  <div id=hgdone style=display:none>
    <video id=hgvid controls style="width:100%;border-radius:8px;margin-top:10px"></video>
    <p style="margin-top:14px">
      <a id=hgdl download class=btn>⬇ Download MP4</a>
      <a id=hgplan target=_blank style="margin-left:14px">plan.json</a>
      <a href="{{ url_for('hg_new') }}" style="margin-left:14px">make another</a>
    </p>
    <p class=sub id=hgsummary></p>
  </div>
  <div id=hgfailed style=display:none class=err></div>
</div>
<script>
const jobId="{{job_id}}";
async function poll(){
  let r=await fetch(`/heygen-broll/job/${jobId}/status`);
  let s=await r.json();
  document.getElementById('hgfill').style.width=s.pct+'%';
  document.getElementById('hgmsg').textContent=s.msg+' ('+s.pct+'%)';
  if(s.status=='done'){
    document.getElementById('hgmsg').style.display='none';
    document.getElementById('hgdone').style.display='block';
    let dl=`/heygen-broll/job/${jobId}/download`;
    document.getElementById('hgvid').src=dl;
    document.getElementById('hgdl').href=dl;
    document.getElementById('hgplan').href=`/heygen-broll/job/${jobId}/plan`;
    document.getElementById('hgsummary').textContent =
      `${s.n_broll} B-roll clips · ${s.coverage_pct}% coverage · ` +
      `${s.duration_s.toFixed(1)}s · language: ${s.language||'?'}`;
    return;
  }
  if(s.status=='error'){
    document.getElementById('hgmsg').style.display='none';
    document.getElementById('hgfailed').style.display='block';
    document.getElementById('hgfailed').textContent=s.error;
    return;
  }
  setTimeout(poll,1500);
}
poll();
</script>
"""


def _hg_form_page(error=None) -> str:
    cfg = load_config()
    return _page(_HG_FORM, error=error,
                 pexels_ok=bool(cfg.pexels_api_key),
                 pixabay_ok=bool(os.environ.get("PIXABAY_API_KEY", "").strip()),
                 anthropic_ok=bool(cfg.anthropic_api_key))


@app.get("/heygen-broll/new")
def hg_new():
    return _hg_form_page()


@app.post("/heygen-broll/upload")
def hg_upload():
    """Save the uploaded MP4 to disk and start a background pipeline
    thread.  The job's progress is then polled by the browser via
    /heygen-broll/job/<id>/status until status='done' or 'error'."""
    from .heygen_broll import (BROLL_JOBS, JobState, run_pipeline,
                                COVERAGE_PRESETS, DEFAULT_COVERAGE)
    f = request.files.get("video")
    if not f or not f.filename:
        return _hg_form_page(error="Please choose an MP4 file to upload.")
    ext = Path(f.filename).suffix.lower()
    if ext not in (".mp4", ".mov", ".m4v", ".webm"):
        return _hg_form_page(error=f"Unsupported video format: {ext}")
    job_id = uuid.uuid4().hex[:12]
    job_dir = OUT / "heygen_broll" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    saved = job_dir / f"input{ext}"
    f.save(str(saved))

    title    = (request.form.get("title") or "").strip()
    script   = (request.form.get("script") or "").strip()
    pip      = request.form.get("pip") in ("1", "on", "true")
    cov_key  = (request.form.get("coverage") or "balanced").lower()
    coverage = COVERAGE_PRESETS.get(cov_key, DEFAULT_COVERAGE)

    BROLL_JOBS[job_id] = JobState(
        job_id=job_id, title=title, input_path=str(saved),
        msg="Queued…", pct=1)

    def _worker():
        try:
            run_pipeline(job_id, saved, title=title, script=script,
                         coverage=coverage, pip=pip, output_dir=job_dir)
        except Exception:                                       # noqa: BLE001
            pass   # JobState already records the error
    threading.Thread(target=_worker, daemon=True).start()
    return redirect(url_for("hg_job_page", job_id=job_id))


@app.get("/heygen-broll/job/<job_id>")
def hg_job_page(job_id: str):
    from .heygen_broll import BROLL_JOBS
    j = BROLL_JOBS.get(job_id)
    if j is None:
        abort(404)
    return _page(_HG_JOB, job_id=job_id, title=j.title)


@app.get("/heygen-broll/job/<job_id>/status")
def hg_job_status(job_id: str):
    from .heygen_broll import BROLL_JOBS
    j = BROLL_JOBS.get(job_id)
    if j is None:
        abort(404)
    return jsonify(
        status=j.status, pct=j.pct, msg=j.msg, error=j.error,
        n_chunks=j.n_chunks, n_broll=j.n_broll,
        coverage_pct=j.coverage_pct, duration_s=j.duration_s,
        language=j.language, title=j.title)


@app.get("/heygen-broll/job/<job_id>/download")
def hg_job_download(job_id: str):
    from .heygen_broll import BROLL_JOBS
    j = BROLL_JOBS.get(job_id)
    if j is None or not j.output_path:
        abort(404)
    p = Path(j.output_path)
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="video/mp4",
                     as_attachment=True, download_name=f"broll_{job_id}.mp4")


@app.get("/heygen-broll/job/<job_id>/plan")
def hg_job_plan(job_id: str):
    """Serve the plan.json for debugging."""
    from .heygen_broll import BROLL_JOBS
    j = BROLL_JOBS.get(job_id)
    if j is None:
        abort(404)
    p = Path(j.output_path).parent / "plan.json" if j.output_path else None
    if not p or not p.exists():
        abort(404)
    return send_file(str(p), mimetype="application/json")


def main() -> None:
    # Port is configurable (VIDLORE_PORT). Default 5000, but on macOS the
    # AirPlay Receiver squats on :5000, so the Mac launcher sets 5050.
    try:
        port = int(os.environ.get("VIDLORE_PORT", "5000").strip() or 5000)
    except ValueError:
        port = 5000
    print(f"{PH} web wizard → http://127.0.0.1:{port}  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, threaded=True)


if __name__ == "__main__":
    main()
