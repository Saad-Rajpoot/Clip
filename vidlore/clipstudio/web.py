"""ClipStudio web portal — a small Flask app to run the full pipeline from a browser.

Paste a script + topic, optionally upload your own voiceover MP3, click Create — it runs
produce_auto() in a background thread (Claude brain, HD downloads, caption-dodge, watermark-crop,
anti-repeat all active), streams live progress, and serves the finished MP4 to download.

Run:  python -m vidlore.clipstudio.web         (defaults to http://127.0.0.1:5151)
Env:  VIDLORE_CLIPSTUDIO_PORT (5151), VIDLORE_CLIPSTUDIO_PORTAL_DIR (output root).

Self-contained / clone-only — does NOT touch the main Vidlore portal (that's web.py at the repo
root, port 5050). Different module, different port.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Flask, request, jsonify, send_file, redirect, url_for, abort

from .config import ClipConfig, engine_config, _NCPU
from . import llm as _llm
from .orchestrate import produce_auto

_ROOT = Path(os.environ.get("VIDLORE_CLIPSTUDIO_PORTAL_DIR",
                            str(Path.home() / "Desktop" / "clipstudio_output" / "portal")))
_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024     # 200 MB voiceover cap

# in-memory job registry  id -> {phase, log[], done, ok, output, error, started}
_JOBS: dict = {}
_LOCK = threading.Lock()
# Renders are SERIALIZED through this lock. The LLM layer reads the chosen provider/model from the
# process-global env at call-time, so two concurrent jobs with different brains would clobber each
# other mid-run. One render at a time keeps each job on its own brain (and avoids CPU/network
# thrash on a single machine); a second submission queues behind the first.
_RENDER_LOCK = threading.Lock()
_JOBS_KEEP = 60          # newest finished jobs kept in the registry (output files stay on disk)
_JOBS_TTL = 24 * 3600    # finished jobs older than this are evicted from the registry


def _evict_jobs():
    """Bound the in-memory registry — a long-lived portal must not grow it forever.
    Only DONE jobs are evicted; their files remain under the portal dir."""
    with _LOCK:
        done = sorted((jid for jid, j in _JOBS.items() if j.get("done")),
                      key=lambda jid: _JOBS[jid].get("started", 0.0))
        now = time.time()
        # TTL anchors on COMPLETION (fallback: start) — a long render must still get its
        # full download window after finishing
        drop = [jid for jid in done
                if now - _JOBS[jid].get("finished", _JOBS[jid].get("started", 0.0)) > _JOBS_TTL]
        overflow = len(done) - len(drop) - _JOBS_KEEP
        if overflow > 0:
            drop += [jid for jid in done if jid not in drop][:overflow]
        for jid in drop:
            _JOBS.pop(jid, None)


# ---------------------------------------------------------------------------
def _run_job(jid: str, project_dir: Path, *, topic: str, title: str, movie_hint: str,
             script_text: str, voiceover: str | None, max_sources: int, policy: str,
             theme: str, verify: bool, voice_provider: str = "", voice_preset: str = "",
             provider: str = "", ds_model: str = "", turbo: bool = True,
             captions_enabled: bool = True, caption_style: str = "professional",
             resume: bool = False, review_mode: bool = False):
    # FULL render log persisted to the project dir — the in-memory buffer keeps only the last 400
    # lines, so window-QC / breakout selection / composed index maps / caption merge / audit remap /
    # assembly-invariant / post-render QA lines scroll off and can never be audited after the fact.
    # `output/build.log` keeps the COMPLETE line-numbered log for every render (best-effort; a log
    # write must never break a render).
    _log_path = project_dir / "output" / "build.log"
    try:
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(_log_path, "a", encoding="utf-8")
        _log_fh.write(f"\n===== render start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"job={jid} =====\n")
        _log_fh.flush()
    except Exception:
        _log_fh = None

    def log(m):
        with _LOCK:
            j = _JOBS.get(jid)
            _el = (time.time() - j['started']) if j is not None else 0.0
            if j is not None:
                j["log"].append(f"[{_el:6.1f}s] {m}")
                j["log"] = j["log"][-400:]
                # cheap phase hint from the orchestrator's "N/9 ·" prefixes
                if "·" in m and m.split("/")[0].strip().isdigit():
                    j["phase"] = m.strip()
            if _log_fh is not None:
                try:
                    _log_fh.write(f"[{_el:7.1f}s] {m}\n")
                    _log_fh.flush()
                except Exception:
                    pass

    try:
        if _RENDER_LOCK.locked():
            log("queued — waiting for the current render to finish…")
        # Hold the render lock for this job's whole run so its brain/env can't be clobbered by a
        # concurrent job. Set BOTH env keys fully (not just on a deepseek job) so no stale model id
        # bleeds in: a non-DeepSeek job still resets the DeepSeek model to the pro default for any
        # fallback. DeepSeek-v4-pro is the default; Claude stays the last-resort fallback only.
        with _RENDER_LOCK:
            os.environ["VIDLORE_CLIPSTUDIO_LLM_PROVIDER"] = provider or "deepseek"
            os.environ["VIDLORE_CLIPSTUDIO_DEEPSEEK_MODEL"] = ds_model or "deepseek-v4-pro"
            # Turbo = saturate all CPU cores (cut/ASR/download fan-out). load_clip_config() inside
            # produce_auto reads this at construction, so it must be set before that call. Set it
            # explicitly each run (symmetric) so a prior job's choice never bleeds in.
            os.environ["VIDLORE_CLIPSTUDIO_MAX_CPU"] = "1" if turbo else "0"
            # REVIEW mode = accept the render even if some beats have no proven footage, flagging them
            # loudly instead of release-blocking. Set symmetrically every run so 'block' (production
            # default) is restored and a prior review retry never bleeds into a fresh job.
            os.environ["VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE"] = "warn" if review_mode else "block"
            res = produce_auto(
                str(project_dir), topic=topic, script_text=script_text, movie_hint=movie_hint,
                policy=policy, max_sources=max_sources, theme=theme, title=title,
                captions=bool(captions_enabled), caption_style=caption_style,
                voiceover=voiceover, use_tts=True, verify=verify, do_build=True,
                voice_provider=voice_provider, voice_preset=voice_preset,
                resume=resume, progress=log,
            )
        with _LOCK:
            j = _JOBS[jid]
            j["done"] = True
            j["finished"] = time.time()
            j["ok"] = bool(res.get("output"))
            j["output"] = res.get("output")
            j["summary"] = res.get("summary", {})
            j["status"] = "ok" if j["ok"] else "error"
            j["retryable"] = not j["ok"]
    except BaseException as e:                            # noqa: BLE001
        # BaseException: a stray SystemExit from a dependency dies silently in a thread —
        # the job would spin "running…" forever with done never set
        #
        # CONTENT vs TRANSIENT. A release-block is a judgment about this render: re-running it
        # unchanged cannot help. Reported as a bare error it looked retryable, and the pipeline was
        # restarted 8 times over 11 hours until the vision API died — at which point the gate had
        # nothing left to reject and the render published. `retryable` is persisted so no driver
        # (or human watching the portal) has to infer that from a message string.
        try:
            from .verify import NonRetryableBuildError as _NRB, VisionBackendError as _VBE
            _content = isinstance(e, _NRB)
            _vision = isinstance(e, _VBE)
        except Exception:
            _content = _vision = False
        with _LOCK:
            j = _JOBS[jid]
            j["done"] = True
            j["finished"] = time.time()
            j["ok"] = False
            j["error"] = str(e)[:600]
            # VISION-OUTAGE is its own status: NOT a content failure (footage may be fine, nobody
            # could check it) and IS retryable — once the API backend/billing is restored, Resume
            # re-verifies and completes. It must never read as 'footage gap'.
            j["status"] = ("vision_down" if _vision else
                           "content_failure" if _content else "error")
            j["retryable"] = _vision or (not _content)
            j["error_type"] = type(e).__name__
            if _vision:
                j["vision_kind"] = getattr(e, "kind", "down")
            j["log"].append(("VISION BACKEND UNAVAILABLE (retryable once restored): " if _vision else
                             "CONTENT FAILURE (not retryable): " if _content else "FATAL: ")
                            + str(e)[:300])
        # full traceback (absolute paths) only to the server console, not the browser
        traceback.print_exc()
        if _log_fh is not None:
            try:
                _log_fh.write("FATAL: " + str(e)[:400] + "\n")
            except Exception:
                pass
    finally:
        if _log_fh is not None:
            try:
                _log_fh.write(f"===== render end {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                _log_fh.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
_PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>ClipStudio</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
 :root{--bg:#0d1117;--card:#161b22;--bd:#30363d;--fg:#e6edf3;--mut:#8b949e;--acc:#d4a017;--ok:#3fb950}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
   font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
 .wrap{max-width:780px;margin:0 auto;padding:28px 18px 60px}
 h1{font-size:24px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 24px}
 .card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:22px;margin-bottom:18px}
 label{display:block;font-weight:600;margin:14px 0 6px}
 .hint{color:var(--mut);font-weight:400;font-size:13px}
 input[type=text],textarea,select{width:100%;background:#0d1117;color:var(--fg);
   border:1px solid var(--bd);border-radius:8px;padding:10px 12px;font:inherit}
 textarea{min-height:170px;resize:vertical}
 .row{display:flex;gap:14px}.row>div{flex:1}
 button{background:var(--acc);color:#1a1a1a;border:0;border-radius:9px;padding:13px 22px;
   font:600 16px/1 inherit;cursor:pointer;margin-top:20px}
 button:disabled{opacity:.5;cursor:not-allowed}
 .brain{font-size:13px;color:var(--mut);margin-top:10px}
 pre{background:#010409;border:1px solid var(--bd);border-radius:8px;padding:14px;
   max-height:340px;overflow:auto;font:12px/1.45 ui-monospace,Menlo,monospace;white-space:pre-wrap}
 .bar{height:6px;background:#21262d;border-radius:4px;overflow:hidden;margin:6px 0 14px}
 .bar>i{display:block;height:100%;background:var(--acc);width:0;transition:width .4s}
 a.dl{display:inline-block;background:var(--ok);color:#06210f;text-decoration:none;
   padding:13px 22px;border-radius:9px;font-weight:700;margin-top:6px}
 .phase{font-weight:600;color:var(--acc)}
 /* caption design selector */
 .capwrap.off .capgrid{opacity:.35;pointer-events:none;filter:grayscale(.6)}
 .capgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:9px;margin-top:6px}
 .capcard{cursor:pointer;border:2px solid var(--bd);border-radius:10px;overflow:hidden;
   background:#0d1117;transition:border-color .15s,transform .1s}
 .capcard:hover{transform:translateY(-1px)}
 .capcard.sel{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}
 .capcard input{position:absolute;opacity:0;width:0;height:0}
 .capsample{display:flex;align-items:flex-end;justify-content:center;min-height:66px;padding:10px 8px;
   text-align:center;line-height:1.15;
   background:linear-gradient(135deg,#2a2f3a 0%,#3a3324 55%,#1c1f27 100%)}
 .capsample .cw{display:inline-block}
 .capname{font-size:12px;font-weight:600;padding:6px 6px 7px;text-align:center;color:var(--mut);
   border-top:1px solid var(--bd)}
 .capcard.sel .capname{color:var(--acc)}
 .capblurb{font-size:11px;color:var(--mut);margin-top:4px;min-height:28px}
</style></head><body><div class=wrap>
<h1>🎬 ClipStudio</h1>
<p class=sub>Script + (optional) voiceover → finished movie-clip video. __BRAIN__</p>
__BODY__
</div>__SCRIPT__</body></html>"""

_FORM = """
<form class=card method=post action="/create" enctype="multipart/form-data" id=f>
 <label>Topic / title <span class=hint>(the video's headline)</span></label>
 <input type=text name=topic placeholder="Daenerys Was Winning... Until One Decision Burned It All Down." required>
 <div class=row>
  <div><label>Movie / show hint <span class=hint>(optional)</span></label>
   <input type=text name=movie placeholder="Game of Thrones"></div>
  <div><label>Max sources <span class=hint>(footage clips)</span></label>
   <input type=text name=max_sources value="8"></div>
 </div>
 <label>AI brain <span class=hint>(which API analyzes the script + matches scenes)</span></label>
 <select name=brain>
__BRAIN_OPTS__
 </select>
 <label>Narration script <span class=hint>(the spoken text — one or more paragraphs)</span></label>
 <textarea name=script required placeholder="She is standing on the steps in Meereen..."></textarea>
 <label>Voiceover audio <span class=hint>(optional MP3/WAV — used instead of TTS, force-aligned to the script)</span></label>
 <input type=file name=voiceover accept="audio/*">
 <div class=row>
  <div><label>AI voice <span class=hint>(used when no voiceover uploaded)</span></label>
   <select name=voice_provider>
    <option value="kokoro">AI · Kokoro (neural, recommended)</option>
    <option value="chatterbox">AI · Chatterbox (best, needs install)</option>
    <option value="elevenlabs">ElevenLabs (cloud, needs key)</option>
    <option value="edge">Edge (free, basic)</option></select></div>
  <div><label>Voice character</label>
   <select name=voice_preset>
    <option value="deep_male_documentary">Deep male — documentary</option>
    <option value="warm_male_explainer">Warm male — explainer</option>
    <option value="dark_investigative_narrator">Dark — investigative</option>
    <option value="mature_female_documentary">Mature female — documentary</option>
    <option value="calm_female_storyteller">Calm female — storyteller</option></select></div>
 </div>
 <div class=row>
  <div><label>Theme</label>
   <select name=theme><option>history</option><option>crime</option><option>modern</option>
   <option>minimalist</option><option>standard</option></select></div>
  <div><label>AI verify <span class=hint>(slower, higher quality)</span></label>
   <select name=verify><option value="1">on</option><option value="0">off (faster)</option></select></div>
  <div><label>Turbo <span class=hint>(use all __NCPU__ CPU cores — faster render)</span></label>
   <select name=turbo>__TURBO_OPTS__</select></div>
 </div>
 <div class=capwrap id=capWrap>
  <div class=row>
   <div style="max-width:150px"><label>Captions</label>
    <select name=captions id=capOn><option value="1">On</option><option value="0">Off</option></select></div>
   <div><label>Caption design <span class=hint>(one style for the whole video)</span></label>
    <div class=capgrid id=capGrid>__CAPTION_CARDS__</div>
    <input type=hidden name=caption_style id=capStyle value="professional"></div>
  </div>
 </div>
 <button type=submit id=go>Create video</button>
 <div class=brain>Sources are downloaded under your own approved-for-testing policy. Verify rights before publishing.</div>
</form>
<script>
 f.addEventListener('submit',function(){go.disabled=true;go.textContent='Starting…'});
 // caption design: click-to-select cards + ON/OFF toggle hides the grid
 (function(){
   var grid=document.getElementById('capGrid'), hid=document.getElementById('capStyle'),
       on=document.getElementById('capOn'), wrap=document.getElementById('capWrap');
   if(!grid)return;
   function sel(name){
     hid.value=name;
     grid.querySelectorAll('.capcard').forEach(function(c){
       c.classList.toggle('sel', c.dataset.name===name);
       var r=c.querySelector('input'); if(r) r.checked=(c.dataset.name===name);
     });
   }
   grid.querySelectorAll('.capcard').forEach(function(c){
     c.addEventListener('click',function(){sel(c.dataset.name);});
   });
   function toggle(){ wrap.classList.toggle('off', on.value==='0'); }
   on.addEventListener('change',toggle);
   sel(hid.value||'professional'); toggle();
 })();
</script>"""


def _job_page(jid: str) -> str:
    j = _JOBS.get(jid) or {}
    if j.get("captions_enabled", True):
        from .caption_presets import CAPTION_PRESETS, DEFAULT_STYLE
        _p = CAPTION_PRESETS.get(j.get("caption_style", DEFAULT_STYLE))
        _cap_txt = f"Captions: {_p.label if _p else 'Professional'}"
    else:
        _cap_txt = "Captions: Off"
    body = f"""
<div class=card>
 <div>Status: <span class=phase id=ph>starting…</span></div>
 <div class=bar><i id=pbar></i></div>
 <div class=brain>{_cap_txt}</div>
 <div id=dl></div>
 <pre id=log>working…</pre>
</div>"""
    script = """<script>
 var done=false;
 async function poll(){
   if(done)return;
   try{
     const r=await fetch('/status/__JID__');const j=await r.json();
     document.getElementById('ph').textContent=j.phase||(j.done?(j.ok?'done ✓':'failed ✗'):'running…');
     document.getElementById('log').textContent=(j.log||[]).join('\\n');
     document.getElementById('log').scrollTop=1e9;
     // crude progress from the "N/9" phase
     let pct=5;const m=(j.phase||'').match(/(\\d+)\\/9/);if(m)pct=Math.round(m[1]/9*100);
     if(j.done)pct=100;document.getElementById('pbar').style.width=pct+'%';
     if(j.done){done=true;
       if(j.ok){document.getElementById('dl').innerHTML=
         '<a class=dl href="/download/__JID__">⬇ Download MP4</a>'+
         '<div class=brain>'+(j.summary&&j.summary.sources_used?(j.summary.sources_used+' sources · '):'')+
         (j.summary&&j.summary.segments?(j.summary.segments+' scenes'):'')+'</div>';}
       else{
         var content=(j.status==='content_failure');
         var vision=(j.status==='vision_down');
         document.getElementById('ph').textContent=vision?'vision API unavailable ✗':(content?'footage gap ✗':'failed ✗');
         var h='<div style="margin-top:10px">';
         if(vision){
           var vk=j.vision_kind||'down';
           var msg=(vk==='billing')?'The AI vision service is out of credits, so footage could not be verified. Top up billing (Gemini AI Studio or Anthropic), then click Resume — it re-verifies only (no re-download / re-index / re-match).':(vk==='auth')?'The vision API key is invalid or unauthorized. Fix it in .env, then click Resume.':'The vision backend was unreachable. Check your connection, then click Resume.';
           h+='<a class=dl href="#" onclick="return retryResume()">↻ Resume (re-verify)</a>';
           h+='<div class=brain>'+msg+'</div>';
         }else{
           h+='<a class=dl href="#" onclick="return retryResume()">↻ Resume / retry from where it failed</a>';
           if(content){
             h+='<a class=dl style="background:#8b6f1a;color:#fff;margin-left:8px" href="#" onclick="return retryDraft()">▶ Build a review draft anyway</a>';
             h+='<div class=brain>Some beats have no proven footage. “Resume” re-checks fast — finished stages (index/match/verify) are not redone. “Review draft” airs those beats flagged for manual review — not for publication.</div>';
           }else{
             h+='<div class=brain>Resume continues from the failed stage — completed stages are skipped, not redone.</div>';
           }
         }
         h+='</div>';
         document.getElementById('dl').innerHTML=h;
       }
     }
   }catch(e){}
   setTimeout(poll, done?9999:2500);
 }
 async function _retry(mode){
   try{ await fetch('/retry/__JID__'+(mode?('?mode='+mode):''),{method:'POST'}); location.reload(); }
   catch(e){}
   return false;
 }
 function retryResume(){ return _retry(''); }
 function retryDraft(){ return _retry('review'); }
 poll();
</script>""".replace("__JID__", jid)
    return (_PAGE.replace("__BODY__", body).replace("__SCRIPT__", script)
            .replace("__BRAIN__", ""))    # job page has no subtitle brain tag — strip the placeholder


# the selectable AI brains. DeepSeek-v4-pro is the default/primary; Claude is the last-resort
# fallback only. value = "<provider>" or "deepseek:<model>".
_BRAINS = [
    ("deepseek:deepseek-v4-pro",   "DeepSeek V4 Pro — best quality (default)"),
    ("deepseek:deepseek-v4-flash", "DeepSeek V4 Flash — faster, lighter"),
    ("anthropic",                  "Claude (Anthropic) — fallback brain"),
    ("gemini",                     "Gemini (Google Vertex)"),
]


def _current_brain() -> str:
    """The brain value matching the live env, so the form opens on what's actually configured."""
    try:
        prov = _llm._provider()
    except Exception:
        prov = "deepseek"
    if prov in ("gemini", "google", "vertex"):
        return "gemini"
    if prov == "anthropic":
        return "anthropic"
    try:
        mdl = _llm._deepseek_model()
    except Exception:
        mdl = "deepseek-v4-pro"
    return f"deepseek:{mdl}"


def _brain_options() -> str:
    cur = _current_brain()
    return "\n".join(
        f'  <option value="{v}"{" selected" if v == cur else ""}>{label}</option>'
        for v, label in _BRAINS)


def _turbo_on() -> bool:
    """Live state of the all-cores Turbo switch (VIDLORE_CLIPSTUDIO_MAX_CPU), so the form opens on
    whatever is currently configured (the project .env defaults it ON)."""
    return os.environ.get("VIDLORE_CLIPSTUDIO_MAX_CPU", "").strip().lower() in ("1", "true", "yes", "on")


def _turbo_options() -> str:
    on = _turbo_on()
    return (f'<option value="1"{" selected" if on else ""}>on — fastest</option>'
            f'<option value="0"{" selected" if not on else ""}>off — leave a core free</option>')


def _caption_cards() -> str:
    """Build the caption-design selector cards from the preset registry — each card shows a compact,
    realistic sample ('Power is never given.') approximating the preset's real typography, colour,
    weight, outline/shadow and active-word treatment (the word 'never' takes the accent colour)."""
    from html import escape
    from .caption_presets import preset_choices
    cards = []
    for p in preset_choices():
        pv = p["preview"]
        # the sample text sits on the sample gradient; backplate presets get a translucent box
        box = ("" if pv["backplate"] == "transparent"
               else f"background:{pv['backplate']};padding:3px 9px;border-radius:4px;")
        sample_style = (f"font-family:{escape(pv['family'])},Arial,sans-serif;"
                        f"font-weight:{pv['weight']};color:{pv['color']};"
                        f"text-shadow:{pv['text_shadow']};{box}")
        cards.append(
            f'<label class=capcard data-name="{p["name"]}" title="{escape(p["blurb"])}">'
            f'<input type=radio name=_capr value="{p["name"]}">'
            f'<div class=capsample><span class=cw style="{sample_style}">'
            f'Power is <span style="color:{pv["accent"]}">never</span> given.</span></div>'
            f'<div class=capname>{escape(p["label"])}</div></label>')
    return "".join(cards)


def _shell(body: str) -> str:
    prov = "?"
    try:
        prov = _llm.active_provider(engine_config())
    except Exception:
        pass
    return (_PAGE.replace("__BODY__", body).replace("__SCRIPT__", "")
            .replace("__BRAIN__", f"Brain: {prov}"))


@app.route("/")
def index():
    return _shell(_FORM.replace("__BRAIN_OPTS__", _brain_options())
                       .replace("__TURBO_OPTS__", _turbo_options())
                       .replace("__CAPTION_CARDS__", _caption_cards())
                       .replace("__NCPU__", str(_NCPU)))


@app.route("/create", methods=["POST"])
def create():
    topic = (request.form.get("topic") or "").strip()
    script_text = (request.form.get("script") or "").strip()
    if not topic or not script_text:
        return redirect(url_for("index"))
    movie = (request.form.get("movie") or "").strip()
    theme = (request.form.get("theme") or "history").strip()
    verify = (request.form.get("verify") or "1").strip() != "0"
    turbo = (request.form.get("turbo") or "1").strip() != "0"   # all-CPU-cores render (default on)
    voice_provider = (request.form.get("voice_provider") or "kokoro").strip()
    voice_preset = (request.form.get("voice_preset") or "deep_male_documentary").strip()
    # AI brain selection (validated against an allowlist — the portal is unauthenticated, so never
    # forward a raw form value into the provider/model env). Defaults to DeepSeek V4 Pro.
    brain = (request.form.get("brain") or "deepseek:deepseek-v4-pro").strip()
    provider, ds_model = "deepseek", "deepseek-v4-pro"
    if brain in ("anthropic", "claude"):
        provider, ds_model = "anthropic", ""
    elif brain in ("gemini", "google", "vertex"):
        provider, ds_model = "gemini", ""
    elif brain.startswith("deepseek"):
        provider = "deepseek"
        _m = brain.split(":", 1)[1] if ":" in brain else "deepseek-v4-pro"
        ds_model = _m if _m in ("deepseek-v4-pro", "deepseek-v4-flash") else "deepseek-v4-pro"
    try:
        max_sources = max(2, min(40, int(request.form.get("max_sources") or 8)))
    except Exception:
        max_sources = 8

    # CAPTION settings — validated server-side against the preset registry (the portal is
    # unauthenticated, so an unknown/manipulated value must never reach the worker raw). ON by
    # default; an unknown style safely falls back to the default preset.
    from .caption_presets import VALID_STYLES, DEFAULT_STYLE
    captions_enabled = (request.form.get("captions") or "1").strip() not in ("0", "false", "off", "no")
    caption_style = (request.form.get("caption_style") or DEFAULT_STYLE).strip().lower()
    if caption_style not in VALID_STYLES:
        caption_style = DEFAULT_STYLE

    _evict_jobs()
    jid = uuid.uuid4().hex[:10]
    proj = _ROOT / jid
    proj.mkdir(parents=True, exist_ok=True)

    vo_path = None
    f = request.files.get("voiceover")
    if f and f.filename:
        ext = os.path.splitext(f.filename)[1].lower() or ".mp3"
        vo_path = str(proj / f"voiceover{ext}")
        f.save(vo_path)

    with _LOCK:
        _JOBS[jid] = {"phase": "queued", "log": [], "done": False, "ok": False,
                      "output": None, "error": "", "summary": {}, "started": time.time(),
                      # machine-readable outcome (see _run_job): "queued"/"ok"/"error"/
                      # "content_failure". `retryable` says whether re-running unchanged could
                      # possibly help — a content verdict never can.
                      "status": "queued", "retryable": True, "error_type": "",
                      # per-job caption settings (shown on the status page; no cross-job leakage)
                      "captions_enabled": captions_enabled, "caption_style": caption_style}

    _kw = dict(
        topic=topic, title=topic, movie_hint=movie, script_text=script_text,
        voiceover=vo_path, max_sources=max_sources,
        policy=os.environ.get("VIDLORE_CLIPSTUDIO_PORTAL_POLICY", "approved_testing").strip(),
        theme=theme, verify=verify,
        voice_provider=voice_provider, voice_preset=voice_preset,
        provider=provider, ds_model=ds_model, turbo=turbo,
        captions_enabled=captions_enabled, caption_style=caption_style)
    # remember the launch kwargs so "Resume / Retry" can replay this render on the SAME dir with
    # resume=True (skipping every stage that already finished) instead of starting a fresh job.
    with _LOCK:
        _JOBS[jid]["params"] = dict(_kw)
    t = threading.Thread(target=_run_job, args=(jid, proj), kwargs=_kw, daemon=True)
    t.start()
    return redirect(url_for("job", jid=jid))


@app.route("/retry/<jid>", methods=["POST"])
def retry(jid):
    """Resume a finished-but-failed job on its OWN project dir with resume=True — every stage that
    already completed (analyze/discover/download/index/match/verify/recover) is skipped, so a render
    that died or content-blocked continues from where it stopped instead of restarting from zero.
    ?mode=review re-runs in review mode (accept + flag weak beats) so a footage-gap render still
    produces a watchable draft."""
    with _LOCK:
        j = _JOBS.get(jid)
        params = (j or {}).get("params")
        busy = bool(j and not j.get("done"))
    if j is None or params is None:
        # params live only in the in-memory registry; an evicted/old job can't be replayed
        return ("This job is no longer resumable (it aged out of the queue). Start a new render.",
                410)
    if busy:
        return ("This render is still running — let it finish before retrying.", 409)
    proj = _ROOT / jid
    if not (proj / "project.json").exists():
        return ("Nothing to resume — the project state for this job is gone.", 410)
    review_mode = (request.values.get("mode") or "").strip().lower() == "review"
    with _LOCK:
        j.update({"phase": "queued", "done": False, "ok": False, "output": None, "error": "",
                  "status": "queued", "retryable": True, "error_type": "",
                  "started": time.time(), "finished": None})
        j["log"].append(f"↻ resume{' (review mode)' if review_mode else ''} — reusing project, "
                        f"skipping already-completed stages")
    t = threading.Thread(target=_run_job, args=(jid, proj),
                         kwargs={**params, "resume": True, "review_mode": review_mode}, daemon=True)
    t.start()
    return redirect(url_for("job", jid=jid))


@app.route("/job/<jid>")
def job(jid):
    if jid not in _JOBS:
        abort(404)
    return _job_page(jid)


@app.route("/status/<jid>")
def status(jid):
    with _LOCK:
        j = _JOBS.get(jid)
        if not j:
            return jsonify({"error": "unknown job"}), 404
        # status/retryable are part of the contract: a caller deciding whether to re-run must not
        # have to pattern-match an error string to learn that a content verdict cannot be retried
        return jsonify({k: j.get(k) for k in ("phase", "log", "done", "ok", "error", "summary",
                                              "status", "retryable", "error_type")})


@app.route("/download/<jid>")
def download(jid):
    with _LOCK:
        j = _JOBS.get(jid)
    if not j or not j.get("output") or not Path(j["output"]).exists():
        abort(404)
    return send_file(j["output"], as_attachment=True,
                     download_name=f"clipstudio_{jid}.mp4")


def main():
    port = int(os.environ.get("VIDLORE_CLIPSTUDIO_PORT", "5151").strip() or 5151)
    host = os.environ.get("VIDLORE_CLIPSTUDIO_HOST", "127.0.0.1").strip()
    print(f"ClipStudio portal → http://{host}:{port}")
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
