#!/usr/bin/env python3
"""Regression tests for the premium post-render results dashboard (the _JOB page).

Template-level (renders _JOB through Jinja) + source guards + logic mirrors. No running
server needed.

    .venv/bin/python tools/test_post_render_dashboard.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = (ROOT / "vidlore" / "web.py").read_text(encoding="utf-8")


def _render_job():
    m = re.search(r'_JOB = """(.*?)"""', WEB, re.S)
    job = m.group(1)
    from flask import Flask, render_template_string
    app = Flask(__name__)
    app.add_url_rule("/", "index", lambda: "x")
    app.add_url_rule("/new", "new", lambda: "x")
    with app.test_request_context("/"):
        return render_template_string(job, job_id="JOBID123", title="My Doc", ph="Vidlore")


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

    html = _render_job()

    # ---- structure: three states ----
    check("rendering state present (#jrender, stage, fill)",
          "id=jrender" in html and "id=jstage" in html and "id=jfill" in html and "id=jpct" in html)
    check("complete state present (#jdone, video, actions, panels)",
          "id=jdone" in html and "id=jvideo" in html and "id=jacts" in html and "id=jpanels" in html)
    check("error state present (#jerr + retry form + tech toggle)",
          "id=jerr" in html and "/job/JOBID123/retry" in html and "id=jerrtog" in html)
    check("success heading is 'Your documentary is ready' (not 'Rendering…')",
          "Your documentary is ready" in html and "<h1>Rendering" not in html)
    check("rendering heading is 'Creating your documentary'", "Creating your documentary" in html)

    # ---- premium actions, no browser-default raw-link group ----
    check("primary action: Open Review Editor button (#edlink, not a bare link)",
          "id=edlink" in html and "jbtn-primary" in html and "Open Review Editor" in html)
    check("secondary action buttons present (Download / Create / My Videos)",
          "Download MP4" in html and "Create another" in html and "My Videos" in html)
    check("thumbnail feature REMOVED (no Thumbnail button / file/thumb ref)",
          "Thumbnail" not in html and "/file/thumb" not in html)
    check("no old raw '· thumbnail · My Videos · make another' link row",
          "make another" not in html and "· thumbnail ·" not in html)
    check("buttons use jbtn classes (no browser-default link styling)",
          html.count("jbtn2") >= 3)

    # ---- video card: cache-bust (no thumbnail poster — feature removed) ----
    check("video preview card with 16:9 frame", "jvframe" in html and "aspect-ratio:16/9" in html)
    check("MP4 cache-busted with ?v=Date.now()",
          "'/job/'+id+'/file/video'+bust" in html and "bust='?v='+Date.now()" in html)
    check("'Final rendered preview' caption present", "Final rendered preview" in html)

    # ---- summary + QA panels ----
    check("project summary renders Duration/Scenes/Resolution rows",
          "row('Duration'" in html and "row('Scenes'" in html and "row('Resolution'" in html)
    check("QA panel: passed -> 'Quality checked', else 'Review recommended'",
          "Quality checked" in html and "Review recommended" in html)
    check("QA technical-details toggle (#jtechtog/#jtech)", "id=jtechtog" in html and "id=jtech" in html)
    check("technical details include black frames + loudness + render job",
          "Black frames" in html and "LUFS" in html and "Render job" in html)

    # ---- polling correctness ----
    check("polling stops after completion (_polling=false on done/error)",
          "_polling=false" in html and "if(!_polling)return" in html)
    check("status drives pct/stage; no fake completion", "/job/'+id+'/status" in html and "s.status=='done'" in html)
    check("summary fetched once on done", "/job/'+id+'/summary" in html)

    # ---- tooltips ----
    check("tooltip engine present (data-tip, 300ms, clamp, Esc)",
          "data-tip" in html and "jtip" in html and "Escape" in html)
    tips = ["Open the scene editor", "Download the latest rendered MP4",
            "all your rendered videos", "Start a brand-new documentary", "quality checks"]
    check("beginner tooltips on key actions", all(t in html for t in tips), [t for t in tips if t not in html])

    # ---- responsive guard ----
    check("responsive CSS guard present (@media stack)", "@media(max-width:640px)" in html)

    # ---- stageLabel mirror (humanized, real-data only) ----
    def stage(msg):
        m = (msg or "").lower()
        if re.search(r"queue|prepar", m): return "Preparing scenes"
        if re.search(r"narrat|voice|tts", m): return "Recording narration"
        if re.search(r"ai|still|imagen|generat", m): return "Generating AI stills"
        if re.search(r"footage|visual|fetch|clip|search", m): return "Selecting visuals"
        if re.search(r"motion|graphic|card|map|chart", m): return "Building motion graphics"
        if re.search(r"music|sfx|sound|audio|mix", m): return "Mixing music & sound"
        if re.search(r"caption|subtitle", m): return "Adding captions"
        if re.search(r"assembl|cross|render|encod|thumb", m): return "Rendering final video"
        if re.search(r"final|check|qa|output", m): return "Checking final output"
        if re.search(r"done|ready|complet", m): return "Video ready"
        return msg or "Working…"
    check("stage: 'Queued…' -> Preparing scenes", stage("Queued…") == "Preparing scenes")
    check("stage: 'Assembling video (crossfades)…' -> Rendering final video",
          stage("Assembling video (crossfades)…") == "Rendering final video")
    check("stage: 'Checking final output…' -> Checking final output",
          stage("Checking final output…") == "Checking final output")
    check("stage: 'Done' -> Video ready", stage("Done") == "Video ready")
    check("stage: 'Narration 85s' -> Recording narration", stage("Narration 85s") == "Recording narration")

    # ---- backend helpers + routes (additive, nothing removed) ----
    def fmt_dur(secs):
        try:
            secs = int(float(secs))
            return f"{secs // 60}:{secs % 60:02d}"
        except Exception:
            return None
    check("duration format: 85.67 -> 1:25", fmt_dur(85.67) == "1:25")
    check("duration format: 605 -> 10:05", fmt_dur(605) == "10:05")
    check("duration format: bad input -> None", fmt_dur("x") is None)
    check("job_summary route added (additive)", '@app.get("/job/<job_id>/summary")' in WEB and "def job_summary" in WEB)
    for r in ["/job/<job_id>/status", "/job/<job_id>/file/<kind>", "/job/<job_id>/retry"]:
        check(f"existing route preserved: {r}", f'"{r}"' in WEB)
    check("summary returns qa + meta facts",
          all(k in WEB for k in ("qa_verdict", "resolution", "black_frames", "duration=", "scenes=", "editor_ok")))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
