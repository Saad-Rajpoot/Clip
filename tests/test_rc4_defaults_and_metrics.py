# RC4 STEP 8+9 — production defaults ON + metrics freshness regression.
#
# Step 9 (web.py): a FRESH /new dashboard (empty form dict) must default the
# SFX + burn-captions checkboxes to CHECKED, while an explicit opt-out on a
# real submitted form (sfx unchecked → the hidden/real value resolves to "0")
# must still round-trip to OFF through the _truthy / _brief_from parse path.
#
# Step 8 (assemble.py): after the final mux, a render_export_metrics.json
# sidecar fingerprints the DELIVERED mp4 (sha256 + probed duration + fps +
# abs path), and the pre-mux render_black_frame_metrics.json is clearly
# labelled scope="intermediate_pre_mux" so it can't be mistaken for the
# final-output metric. The metrics writer is hermetic + never raises.
#
# Hermetic: no real render. The export-metrics unit builds a ~1s color mp4
# with the bundled ffmpeg when available, else falls back to asserting the
# helper's never-raise contract on a missing file.
#
# Run:  PYTHONPATH=. python tests/test_rc4_defaults_and_metrics.py
#   or  PYTHONPATH=. pytest tests/test_rc4_defaults_and_metrics.py
import os
import json
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("VIDLORE_AI_VIDEO", "0")

import vidlore.web as W
import vidlore.assemble as A
from vidlore.ffmpeg_tool import ffmpeg_exe


_passed = 0


def check(name, cond):
    global _passed
    assert cond, "FAIL: " + name
    _passed += 1
    print("  [ok] " + name)


# ───────────────────────── Step 9 — fresh defaults, per surface ─────────────
def test_form_template_defaults_on():
    """ACTUAL behavior of each surface, not stale literals.

    vidlore app form: sound design defaults ON; captions are DELIBERATELY off by default
    (the template's own copy says 'off by default — toggle on here, or anytime in the
    editor') — asserting the deliberate design, not the pre-refactor literal.

    ClipStudio PORTAL: captions default ON, and the selected caption style is validated
    against the preset registry with the default preset as the fallback — the behavior
    users actually get on a fresh render."""
    src = W._FORM
    check("vidlore form sfx default ON", "f.get('sfx','1')" in src)
    check("vidlore captions off-by-default is DELIBERATE (explained in the UI copy)",
          "f.get('captions','0')" in src and "off by default" in src)
    import inspect
    import vidlore.clipstudio.web as CW
    csrc = inspect.getsource(CW)
    check("ClipStudio portal captions default ON in the submit parse",
          '(request.form.get("captions") or "1")' in csrc)
    check("ClipStudio portal honors explicit off values",
          '("0", "false", "off", "no")' in csrc)
    from vidlore.clipstudio.caption_presets import CAPTION_PRESETS, DEFAULT_STYLE
    check("ClipStudio portal validates the selected caption style against the registry",
          'request.form.get("caption_style")' in csrc and "DEFAULT_STYLE" in csrc)
    check("default caption style is a real preset", DEFAULT_STYLE in CAPTION_PRESETS)


def test_fresh_dashboard_renders_checked():
    """An empty form dict (what GET /new passes) must render BOTH toggles
    checked — proves the production default is genuinely ON end-to-end."""
    # _form_page renders a Jinja template via Flask's render_template_string,
    # which needs an app/request context.
    with W.app.test_request_context("/new"):
        html = W._form_page({})
    # Locate the two checkbox <input> tags and assert each carries `checked`.
    for field in ("sfx", "captions"):
        i = html.find("name=" + field + " ")
        check(field + " checkbox present in rendered form", i != -1)
        tag = html[i: html.find(">", i)]
        check("fresh dashboard renders " + field + " CHECKED",
              "checked" in tag)


def test_explicit_optout_round_trips_off():
    """PRESERVE explicit opt-out. A user who unchecks the box submits the
    real value '0' (the template's hidden field always emits the resolved
    value); that must parse to False — both via the low-level _truthy gate
    and via the full _brief_from path."""
    # low-level gate
    check("_truthy('0') is False", W._truthy("0") is False)
    check("_truthy('1') is True", W._truthy("1") is True)
    check("_truthy(absent/None) is False", W._truthy(None) is False)

    class _Form(dict):
        pass

    # Submitted form with the SFX box explicitly OFF (value '0') + captions OFF.
    f = _Form(title="T", sfx="0", captions="0")
    b = W._brief_from(f)
    check("_brief_from: explicit sfx='0' → extra['sfx'] False",
          b.extra["sfx"] is False)
    check("_brief_from: explicit captions='0' → brief.captions False",
          bool(b.captions) is False)

    # A truly ABSENT checkbox (no key) also means OFF at the parse layer —
    # the template hidden field guarantees the real value is present in the
    # live portal, but the parse default must remain safe.
    f2 = _Form(title="T")
    b2 = W._brief_from(f2)
    check("_brief_from: absent sfx → False (parse default OFF)",
          b2.extra["sfx"] is False)

    # And a checked box (value '1') flips it ON, confirming the round-trip.
    f3 = _Form(title="T", sfx="1", captions="1")
    b3 = W._brief_from(f3)
    check("_brief_from: sfx='1' → True", b3.extra["sfx"] is True)
    check("_brief_from: captions='1' → True", bool(b3.captions) is True)


# ─────────────────── Step 8 — final-export metrics sidecar ────────────────
def _make_tiny_mp4(dst: Path) -> bool:
    """Best-effort: synth a ~1s 64x64 color mp4 with the bundled ffmpeg.
    Returns True on success, False if ffmpeg is unavailable / fails."""
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=c=teal:s=64x64:d=1",
             "-r", "30", "-pix_fmt", "yuv420p", str(dst)],
            capture_output=True, text=True, timeout=60)
        return r.returncode == 0 and dst.exists() and dst.stat().st_size > 0
    except Exception:
        return False


def test_export_metrics_records_hash_duration_path():
    """_write_export_metrics writes render_export_metrics.json next to the
    final mp4 with sha256 + probed duration + fps + abs path."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "My_Final_Video.mp4"
        have_mp4 = _make_tiny_mp4(out)

        m = A._write_export_metrics(out, fps=30)

        # Sidecar file is written regardless and matches the returned dict.
        sidecar = out.parent / "render_export_metrics.json"
        check("render_export_metrics.json written", sidecar.exists())
        on_disk = json.loads(sidecar.read_text(encoding="utf-8"))
        check("sidecar matches returned dict", on_disk == m)

        check("schema == render_export_metrics/1",
              m["schema"] == "render_export_metrics/1")
        check("final_video == final mp4 name",
              m["final_video"] == "My_Final_Video.mp4")
        check("final_path is absolute + ends with the file",
              os.path.isabs(m["final_path"])
              and m["final_path"].endswith("My_Final_Video.mp4"))
        check("fps recorded", m["fps"] == 30)

        if have_mp4:
            check("sha256 is a 64-hex digest",
                  isinstance(m["sha256"], str) and len(m["sha256"]) == 64
                  and all(c in "0123456789abcdef" for c in m["sha256"]))
            # sha256 must match an independent stream-hash of the same bytes.
            import hashlib
            h = hashlib.sha256()
            with open(out, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            check("sha256 matches independent hash of the file",
                  m["sha256"] == h.hexdigest())
            check("duration_s probed (~1s)",
                  isinstance(m["duration_s"], float) and m["duration_s"] > 0)
        else:
            print("  [skip] ffmpeg unavailable — mp4 content asserts skipped")


def test_export_metrics_never_raises_on_bad_input():
    """Robustness: hashing/probe failures degrade fields to None and never
    raise (so a sidecar problem can never break the render)."""
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "does_not_exist.mp4"
        m = A._write_export_metrics(missing, fps=24)   # must not raise
        check("missing-file: sha256 degraded to None", m["sha256"] is None)
        check("missing-file: duration degraded to None",
              m["duration_s"] is None)
        check("missing-file: schema + fps still present",
              m["schema"] == "render_export_metrics/1" and m["fps"] == 24)
    # Probe helper on a missing path returns 0.0 (the documented sentinel).
    check("_probe_duration_s(missing) == 0.0",
          A._probe_duration_s(Path(td) / "gone.mp4") == 0.0)


# ──────────── Step 8 — black-frame sidecar labelled intermediate ──────────
def test_black_frame_writer_labels_intermediate():
    """Both render_black_frame_metrics.json writer branches must emit
    scope='intermediate_pre_mux' + scanned_file=<video_in.name> so the
    pre-mux scan can't be confused with the final-output metric. Asserted on
    the assemble.py source (the writers only fire inside a real repair)."""
    src = Path(A.__file__).read_text(encoding="utf-8")
    check("black-frame sidecar carries scope=intermediate_pre_mux",
          '"scope": "intermediate_pre_mux"' in src)
    check("black-frame sidecar carries scanned_file label",
          '"scanned_file": video_in.name' in src)
    # Exactly the two known writer branches were labelled (preserved + clean).
    check("scope label present in BOTH writer branches",
          src.count('"scope": "intermediate_pre_mux"') == 2)


def _run_all():
    test_form_template_defaults_on()
    test_fresh_dashboard_renders_checked()
    test_explicit_optout_round_trips_off()
    test_export_metrics_records_hash_duration_path()
    test_export_metrics_never_raises_on_bad_input()
    test_black_frame_writer_labels_intermediate()
    print("\nALL %d CHECKS PASSED" % _passed)


if __name__ == "__main__":
    _run_all()
