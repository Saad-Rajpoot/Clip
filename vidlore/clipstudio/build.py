"""Stage 6 — build the final video.

Constructs the engine's own objects from our segments + cut clips and calls
`vidlore.assemble.assemble()` to render — reusing the entire renderer (grade, captions,
transitions, letterbox blurred-fill) unchanged. Clips are injected via `beat_clips`, exactly
how the engine's own `locked_visuals.json` override works (DESIGN.md §3).

TTS uses the engine's free edge-tts `narrate()`; if that has no network, a silent-narration
fallback (per-scene silence of the estimated duration) still lets the video assemble so the
clip selection can be reviewed visually.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from .models import ClipProject, ScriptSegment
from .config import ClipConfig, ffmpeg_exe


def _placeholder_clip(proj: ClipProject, idx: int) -> Path:
    """Black clip for a segment that ended up with no candidate."""
    out = proj.clips_dir / f"seg_{idx:03d}_blank.mp4"
    if not out.exists():
        subprocess.run(
            [ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=3",
             "-c:v", "libx264", "-t", "3", "-pix_fmt", "yuv420p", str(out)],
            capture_output=True, timeout=60,
        )
    return out


def _watermark_crop_filter(corner: str) -> str:
    """A same-fraction punch-in crop (keeps 16:9) that drops the corner where a channel watermark
    sits, then rescales to fill — so a watermarked source can be KEPT (relevance) instead of dropped.
    Keeps 84% of each axis away FROM the logo corner (~19% zoom). corner ∈ br|bl|tr|tl."""
    k = 0.84
    # keep the region OPPOSITE the logo corner: logo on the right → keep left (x=0); logo on the
    # bottom → keep top (y=0); and vice-versa.
    x = "0" if corner in ("br", "tr") else f"iw*{1 - k:.3f}"
    y = "0" if corner in ("br", "bl") else f"ih*{1 - k:.3f}"
    # no rescale here: assemble normalizes every clip to 1920x1080 with AR preserved — a hard
    # scale=1280:720 would soften 1080p sources (720p round-trip) and stretch non-16:9 film
    return f"crop=iw*{k:.3f}:ih*{k:.3f}:{x}:{y}"


def _detect_logo_corner(src_path: str, ocr_engine) -> str:
    """OCR a few SOURCE frames and vote on the corner (br|bl|tr|tl) where persistent edge text (a
    channel logo / CTA) sits, so the crop drops the right corner. Defaults to bottom-right."""
    import re as _re3
    from collections import Counter
    try:
        from PIL import Image
    except Exception:
        return "br"
    from .match import _OCR_JUNK
    ff = ffmpeg_exe()
    votes: Counter = Counter()
    for off in (5, 20, 45):
        tmp = f"{src_path}.corner_{off}.jpg"
        try:
            subprocess.run([ff, "-y", "-loglevel", "error", "-ss", str(off), "-i", str(src_path),
                            "-frames:v", "1", tmp], capture_output=True, timeout=20)
            if not Path(tmp).exists():
                continue
            W, H = Image.open(tmp).size
            res, _el = ocr_engine(tmp)
            for box, txt, conf in (res or []):
                if float(conf) < 0.4:
                    continue
                if not (_OCR_JUNK.search(txt) or len(_re3.findall(r"[A-Za-z]", txt)) >= 6):
                    continue
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                bw = (max(xs) - min(xs)) / max(1, W)
                cx = (min(xs) + max(xs)) / 2 / max(1, W); cy = (min(ys) + max(ys)) / 2 / max(1, H)
                if bw > 0.45:
                    continue        # frame-wide strip = burned subtitle/ticker, never a corner logo
                # a logo sits in a CORNER: off-center on BOTH axes — bottom-CENTER subtitles
                # (cy~0.85, cx~0.5) must not vote bl/br on centroid jitter
                if (cx < 0.3 or cx > 0.7) and (cy < 0.3 or cy > 0.7):
                    votes[("b" if cy > 0.5 else "t") + ("r" if cx > 0.5 else "l")] += 1
        except Exception:
            pass
        finally:
            try:
                import os as _o
                _o.remove(tmp)
            except Exception:
                pass
    return votes.most_common(1)[0][0] if votes else "br"


def _watermarked_source_corners(proj, ocr_engine, progress=None) -> dict:
    """{source_id: corner} for every SOURCE_OK source carrying a persistent channel watermark — its
    clips get a punch-in crop at cut time so the source can be KEPT (relevance) not dropped."""
    from . import index as _index
    from .match import _source_is_watermarked
    out: dict = {}
    if ocr_engine is None:
        return out
    for src in proj.sources:
        if getattr(src, "status", "") != "ok" or not src.local_path or not Path(src.local_path).exists():
            continue
        try:
            shots = _index.load_shots(proj, src.id)
        except Exception:
            continue
        if _source_is_watermarked(shots):
            out[src.id] = _detect_logo_corner(src.local_path, ocr_engine)
            if progress:
                progress(f"build: watermark-crop source {src.id} (corner={out[src.id]})")
    return out


# Detail-enhance chain applied to EVERY cut: the "1080p" uploads of TV scenes are mostly SOFT
# re-encodes (measured Laplacian var 2-18 — fake HD). Light denoise first (so sharpening
# amplifies real edges, not encode noise), then unsharp + contrast-adaptive sharpen.
# Empirically 3.3× Laplacian on a soft S03E03 source with no visible halos.
_CAS = "hqdn3d=1.5:1.5:3:3,unsharp=5:5:0.9:3:3:0.0,cas=0.5"


def _ken_burns_filter(dur: float, src_w: int = 0, zoom_to: float = 1.10) -> str:
    """A slow cinematic push-in (Ken Burns) over the clip — what the competitor does on a held
    key shot. Pre-upscale (lanczos) so the zoom never reveals soft pixels, then zoompan from
    1.0→zoom_to across the clip's frames, output 1920x1080@30."""
    frames = max(1, int(round(max(0.5, dur) * 30)))
    return (f"scale=2560:1440:force_original_aspect_ratio=increase:flags=lanczos,crop=2560:1440,"
            f"zoompan=z='min(1.0+{zoom_to - 1.0:.4f}*on/{frames},{zoom_to:.3f})':d=1:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30,setsar=1,{_CAS}")


def _image_kenburns_clip(img_path: str, dest: Path, dur: float, zoom_to: float = 1.12) -> Optional[Path]:
    """Render a still IMAGE (web exact-scene fallback) as a 1080p Ken-Burns motion clip of
    `dur` seconds — a slow push-in over the photo so a still never looks frozen/cheap."""
    dur = max(0.6, float(dur))
    frames = max(1, int(round(dur * 30)))
    vf = (f"scale=2560:1440:force_original_aspect_ratio=increase:flags=lanczos,crop=2560:1440,"
          f"zoompan=z='min(1.0+{zoom_to - 1.0:.4f}*on/{frames},{zoom_to:.3f})':d={frames}:"
          f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30,setsar=1,{_CAS}")
    cmd = [ffmpeg_exe(), "-y", "-loop", "1", "-t", f"{dur:.3f}", "-i", str(img_path),
           "-vf", vf, "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=240)
    except Exception:
        return None
    return dest if (p.returncode == 0 and dest.exists() and dest.stat().st_size > 0) else None


def _upscale_filter(src_w: int) -> str:
    """Normalize + sharpen every cut. SD sources get a lanczos upscale first; everything gets
    contrast-adaptive sharpening (the footage chain is otherwise uniformly soft)."""
    if src_w and src_w < 1280:
        return ("scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop=1920:1080,unsharp=5:5:0.8:3:3:0.4,setsar=1,{_CAS}")
    return _CAS


def _recut_to_duration(src_path: str, start: float, need: float, src_dur: float,
                       dest: Path, crop_filter: str = "", zoom: float = 0.0,
                       src_w: int = 0) -> Optional[Path]:
    """Cut a clip of length `need` from the source so it is AT LEAST as long as the narration
    scene — otherwise the renderer loops a short clip to fill the audio (the 2-second 'same scene
    repeating' bug). Extends past the detected shot into continuous source footage as needed.
    `zoom` applies a slow Ken Burns push-in (held key shot); `src_w` enables SD sharpen-upscale."""
    start = max(0.0, start)
    end = start + need
    if src_dur and end > src_dur:                     # clamp to the tail, keep full length
        end = src_dur
        start = max(0.0, end - need)
    dur = max(0.5, end - start)
    # filter chain: watermark crop → (Ken Burns zoom | SD sharpen-upscale). Zoom already
    # upscales+normalizes to 1080, so it subsumes the SD-upscale step.
    vf_parts = [crop_filter] if crop_filter else []
    if zoom:
        vf_parts.append(_ken_burns_filter(dur, src_w, zoom_to=float(zoom)))
    else:
        up = _upscale_filter(src_w)
        if up:
            vf_parts.append(up)
    cmd = [
        ffmpeg_exe(), "-y", "-ss", f"{start:.3f}", "-i", str(src_path), "-t", f"{dur:.3f}", "-an",
    ]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += [
        # CRF 18 + medium: the clips are short and re-encoded again by assemble + the letterbox
        # bake, so start crisp (CRF 20/veryfast compounded into visible softness)
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(dest),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        return None
    return dest if (p.returncode == 0 and dest.exists() and dest.stat().st_size > 0) else None


def _apply_cinematic_letterbox(mp4: Path, bar_h: int = 132) -> bool:
    """Bake clean cinematic black bars (top + bottom) onto the FINAL video — the wide 'film' frame an
    expert editor uses when dissecting one scene (see the reference competitor cut). drawbox OVERLAYS
    the bars (no crop, no zoom — the full image is kept); the captions are lifted above the bottom bar
    by raising the caption margin before assemble(), so nothing is covered."""
    tmp = mp4.with_name(mp4.stem + "_cine.mp4")
    vf = (f"drawbox=x=0:y=0:w=iw:h={bar_h}:color=black@1.0:t=fill,"
          f"drawbox=x=0:y=ih-{bar_h}:w=iw:h={bar_h}:color=black@1.0:t=fill")
    cmd = [ffmpeg_exe(), "-y", "-i", str(mp4), "-vf", vf,
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=1800)
        if r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            tmp.replace(mp4)
            return True
    except Exception:
        pass
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass
    return False


def _apply_slow_motion(clip_path: Path, factor: float, dest: Path) -> Path:
    """Re-encode a clip slowed by `factor` (setpts). Done clip-side BEFORE assemble() so the
    engine's fps=30 normalization can't undo it. Caller cut the source shorter by 1/factor so the
    stretched output still exactly fills its beat (narration stays in sync)."""
    cmd = [
        ffmpeg_exe(), "-y", "-i", str(clip_path), "-vf", f"setpts={factor:.3f}*PTS", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(dest),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception:
        return clip_path
    return dest if (p.returncode == 0 and dest.exists() and dest.stat().st_size > 0) else clip_path


def _freeze_punchline(clip: Path, dest: Path, t1: float, total: float,
                      style: str = "bw") -> Optional[Path]:
    """Competitor-style analytical FREEZE: play the beat live until t1, then hold that exact
    frame for the rest — the 'screenshot' moment under the key narration line. style='bw' is
    the signature B&W punchline (with a shutter click mixed in by the caller); style='still'
    is the quieter COLOR analytical still the competitor uses for mid-argument observations
    (no SFX) — both appear in their edit, B&W is reserved for the biggest moments."""
    rest = max(0.6, total - t1)
    nloop = max(1, int(round(rest * 30)))
    grade = ("hue=s=0,eq=contrast=1.22:brightness=0.06,noise=c0s=6:c0f=t,"
             if style == "bw" else
             "eq=contrast=1.08:saturation=1.05,noise=c0s=4:c0f=t,")
    # NOTE: tpad stop_mode=clone silently no-ops in the bundled ffmpeg — the `loop` filter is
    # the reliable way to hold a frame. Explicit split (an input label can't feed two chains).
    fc = (f"[0:v]split=2[va][vb];"
          f"[va]trim=0:{t1:.3f},setpts=PTS-STARTPTS[live];"
          f"[vb]trim={t1:.3f}:{t1 + 0.12:.3f},setpts=PTS-STARTPTS,"
          f"loop=loop={nloop}:size=1:start=2,setpts=N/30/TB,"
          f"{grade}"
          f"trim=0:{rest:.3f},setpts=PTS-STARTPTS[frz];"
          f"[live][frz]concat=n=2:v=1:a=0[out]")
    cmd = [ffmpeg_exe(), "-y", "-i", str(clip), "-filter_complex", fc, "-map", "[out]",
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dest)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=240)
    except Exception:
        return None
    return dest if (p.returncode == 0 and dest.exists() and dest.stat().st_size > 0) else None


# ---------------------------------------------------------------------------
# REAL-AUDIO BREAKOUTS — the competitor's signature authenticity move: the narration goes
# QUIET and the source scene plays with its OWN dialogue for ~4-5s, then narration resumes
# ("evidence" for the claim just made). Used 1-3×/video, only where the narration genuinely
# references a spoken line (dialogue-locked beats) — natural, never forced.
# ---------------------------------------------------------------------------

# A breakout must be the MOVIE's ORIGINAL dialogue — NEVER another YouTuber's voice-over. A
# video essay narrates OVER the clip (CTA, meta-analysis "this scene…", "in psychology…"); its
# audio reads as commentary, not in-character dialogue. These phrases mark competitor narration.
_NARRATION_RX = re.compile(
    r"\b(subscrib|like button|hit (that|the) like|smash (that|the) like|comment(s| below)?|"
    r"this video|this breakdown|this analysis|this essay|the channel|link(s)? (in|below)|"
    r"patreon|if you (enjoyed|liked)|thanks for watching|let'?s (talk|break|dive)|"
    r"in (today'?s|this) video|stay tuned|next time|don'?t forget to|"
    # meta-commentary about the scene (essayist describing, not a character speaking):
    r"this (scene|line|moment|shot|episode|sequence|exchange)\b|that (scene|line|moment)\b|"
    r"in psychology|psycholog(ical|y)|the (philosophical|emotional|psychological) (core|center|heart)|"
    r"represent(s|ed)?|symboliz|foreshadow|what this (means|tells|shows|reveals)|"
    r"notice (how|that)|here'?s (why|what|the)|the (most|single) [a-z]+ line|"
    r"the (energy|air|mood|tone) (shifts|changes)|everything (stops|changes)|"
    r"writers?|the show|the series|gave us|gives us|the scene works|"
    # third-person ESSAY/commentary phrasing — a narrator ANALYZING the scene, not a character
    # IN it (observed leaking into breakouts: "Basically what the red wedding is to Tywin...",
    # "So here he is, ready to discuss...", "all that warmth he once had for her got buried"):
    r"basically|essentially|so here (?:he|she|they|it|we)\b|"
    r"all that (?:warmth|passion|love|rage|anger|hate|hatred|ambition|grief|pain|power|fear|tenderness|loyalty)\b|"
    r"(?:he|she|it|they)'?s not just\b|not just [a-z]+ing\b|"
    r"what (?:the )?[a-z' ]{2,30} (?:is|was|means|represents) to [a-z]|"
    r"writing a new \w+|\w+ in the same song\b|got buried\b|ready to discuss\b|"
    r"once had for\b|has just (?:orchestrated|won|killed|become|ended|destroyed)\b|"
    # more essay-narration tells (observed leaking into a breakout: "Tyrion knows that Lannister
    # promises are written in sand..."): essayist metaphors + 3rd-person interpretation. Tuned NOT
    # to hit in-character dialogue ("he knows that we are here" stays clean — only reali[sz]es/
    # understands/represents trigger the 3rd-person branch).
    r"written in (?:sand|stone|blood)|this (?:proves|shows us|reveals|tells us|is what makes)|"
    r"which is (?:why|exactly why)|(?:he|she|they|it) (?:reali[sz]es|understands|represents|embodies) (?:that|the|how|why)|"
    # ANALYTICAL / EVALUATIVE essay tells — a narrator GRADING a character or scene in the third
    # person (never an in-character line). Observed leaking into a breakout from a commentary source:
    # "He doubles Bronn's pay to stay loyal. The black water proved Tyrion's strategic brilliance."
    r"(?:strategic|tactical|political|military|sheer|pure|cold) (?:brilliance|genius|mastery|mind|acumen|calculus)|"
    r"(?:proved?|proves|cements?|demonstrat\w+|showcas\w+|underscores?) "
    r"(?:his|her|their|the|[a-z]+'s)[a-z ]{0,18}(?:brilliance|genius|strateg\w+|cunning|ruthless\w+|"
    r"madness|loyalty|dominance|downfall|character|arc|nature)|"
    r"\b[a-z]+'s (?:strategic|tactical|political) (?:brilliance|genius|mind|prowess)|"
    # essayist RECAP tells — third-person summary of the SHOW across time, never in-character
    # dialogue (observed leaking into a breakout window from a 'Cersei's Fatal Mistake' analysis
    # source: "...walked out untouched. Cersei spent multiple seasons after the purple wedding..."):
    r"most people (?:think|believe|assume|forget|miss|don'?t)|people (?:think|assume|forget) that|"
    r"(?:spent|spends|spend|spending) (?:the (?:next|rest|following)|multiple|several|two|three|"
    r"four|five|years|seasons|episodes)|(?:multiple|several|the next|the following|over \w+) "
    r"(?:seasons|episodes)|(?:red|purple|green) wedding|"
    r"character('?s)? arc)\b", re.I)


def _is_narration(text: str) -> bool:
    """True if `text` reads as competitor voice-over / essay narration, not movie dialogue."""
    return bool(_NARRATION_RX.search(text or ""))


# A breakout airs a scene's OWN dialogue. A RETROSPECTIVE recap line — a character in a LATER scene
# referring back to the event ("Last time I was here, I killed my father with a crossbow" = S7 Tyrion
# recapping the S4 act) — is the wrong era/scene even though it's in-character. Bar such lines from
# breakouts (they describe the act in the past, the breakout should BE the act).
_BREAKOUT_RECAP_RX = re.compile(
    r"\blast time (?:i|we|you|he|she|they)\b|"
    r"\b(?:i|you|he|she|they) killed (?:my|your|his|her|their) (?:own )?(?:father|son|mother|brother|sister|king)\b|"
    r"\b(?:years?|seasons?) (?:ago|later)\b|\ball those years\b|\bback then\b",
    re.I)


def _is_recap_line(text: str) -> bool:
    """True if a breakout transcript is a later-scene RETROSPECTIVE recap (wrong era/scene)."""
    return bool(_BREAKOUT_RECAP_RX.search(text or ""))


_COMPARISON_RX = re.compile(
    r"\b(unlike|compared to|in contrast|whereas|versus|\bvs\.?\b|as opposed to|"
    r"the difference between|side by side|by comparison|just like (?:in )?(?:house of|the )|"
    r"reminds (?:us|me) of)\b", re.I)


def _script_wants_comparison(segments) -> bool:
    """True if the NARRATION explicitly compares to another show/installment — only then may a
    different-era / different-installment clip legitimately air (e.g. 'unlike House of the Dragon')."""
    for s in segments or []:
        if _COMPARISON_RX.search(getattr(s, "text", "") or ""):
            return True
    return False


def _dialogue_aware_dur(src_path: str, start: float, lo: float = 3.0,
                        hi: float = 10.0):
    """(duration, transcript). Ends the breakout on a COMPLETE spoken line, not mid-word.
    Whisper the source from `start`, find the LATEST natural stop (punctuation or >=0.45s
    gap) within [lo, hi]. Also returns the joined transcript so the caller can REJECT a clip
    whose audio is competitor narration rather than the movie's own dialogue."""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return max(lo, min(hi, 5.5)), ""
    try:
        import subprocess as _sp
        import tempfile as _tf
        import os as _os
        fd, wav = _tf.mkstemp(suffix=".wav")
        _os.close(fd)
        _sp.run([ffmpeg_exe(), "-y", "-ss", f"{max(0.0, start):.3f}", "-t", f"{hi + 1.5:.2f}",
                 "-i", str(src_path), "-vn", "-ar", "16000", "-ac", "1", wav],
                capture_output=True, timeout=120)
        m = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _i = m.transcribe(wav, word_timestamps=True, vad_filter=False)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append((str(w.word or "").strip(), float(w.start), float(w.end)))
        try:
            _os.unlink(wav)
        except OSError:
            pass
        text = " ".join(w[0] for w in words)
        if not words:
            return max(lo, min(hi, 5.0)), ""
        stops = []                                     # candidate end-times (relative to start)
        for i, (txt, _ws, we) in enumerate(words):
            if we > hi + 0.4:
                break
            ends_sentence = txt[-1:] in ".!?…"
            gap_after = (words[i + 1][1] - we) if i + 1 < len(words) else 1.0
            if ends_sentence:
                stops.append(min(hi, we + 0.30))       # let the line breathe before the cut
            elif gap_after >= 0.45:
                stops.append(min(hi, we + min(0.30, gap_after / 2)))
        good = [t for t in stops if lo <= t <= hi]
        if good:
            return round(max(good), 3), text           # latest complete line within the window
        last_end = min(hi, words[-1][2] + 0.30)        # no clean stop → run to last word
        return round(max(lo, min(hi, last_end)), 3), text
    except Exception:
        return max(lo, min(hi, 5.5)), ""


def _extract_breakout(src_path: str, start: float, dur: float, vdest: Path,
                      adest: Path, src_w: int = 0) -> Optional[float]:
    """Cut the breakout VIDEO (enhanced, 1080p30, silent) + its AUDIO (2-pass loudnorm to
    narration level, faded) from the source. Skips leading scene silence so the narration
    pause never dangles over a mute shot; `dur` is treated as the MAX — the real length is
    chosen to end on a complete spoken line (3-10s). Returns the exact duration or None."""
    import json as _json10
    import re as _re10
    try:
        # 1) leading-silence probe — a shot often opens 1-2s before anyone speaks
        ps = subprocess.run(
            [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(src_path),
             "-t", f"{dur:.3f}", "-vn", "-af", "silencedetect=noise=-30dB:d=0.4",
             "-f", "null", "-"], capture_output=True, timeout=120)
        _txt = (ps.stderr or b"").decode("utf-8", "ignore")
        _m0 = _re10.search(r"silence_start:\s*(-?[\d.]+)", _txt)
        if _m0 and float(_m0.group(1)) <= 0.15:
            _m1 = _re10.search(r"silence_end:\s*([\d.]+)", _txt)
            if _m1:
                _lead = float(_m1.group(1))
                if 0.8 <= _lead < dur - 1.2:
                    start = max(0.0, start + _lead - 0.25)
        # DIALOGUE-AWARE LENGTH: end on a complete spoken line within [3, dur] seconds, never
        # mid-word. Bounded by what's left of the source. (`dur` arrives as the 10s max cap.)
        try:
            from .ingest import probe as _probe0
            _srcdur = float(_probe0(src_path).get("duration", 0.0) or 0.0)
        except Exception:
            _srcdur = 0.0
        _hi = min(float(dur), (_srcdur - start - 0.1) if _srcdur else float(dur))
        if _hi >= 3.2:
            dur, _bk_text = _dialogue_aware_dur(str(src_path), start, lo=3.0, hi=_hi)
        else:
            dur, _bk_text = max(2.0, _hi), ""
        # COMPETITOR-VOICEOVER GUARD: a breakout must be the movie's OWN dialogue, never another
        # YouTuber's narration. If the clip's audio reads as commentary/CTA (essay voice-over),
        # reject it — the beat keeps its footage. (env VIDLORE_CLIPSTUDIO_BREAKOUT_VOICE_GUARD)
        import os as _os10
        if _os10.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_VOICE_GUARD", "1").strip() \
                not in ("0", "false", "no") and _is_narration(_bk_text):
            return None
        pv = subprocess.run(
            [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(src_path),
             "-t", f"{dur:.3f}", "-an",
             "-vf", (_upscale_filter(src_w) if src_w and src_w < 1280
                     else f"scale=1920:1080:force_original_aspect_ratio=increase,"
                          f"crop=1920:1080,setsar=1,{_CAS}") + ",fps=30",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(vdest)],
            capture_output=True, timeout=300)
        # 2) audio — 2-PASS loudnorm: single-pass on a 2-6s clip lands ~1.5-2 LU hot vs the
        # narration bed (measured on the Jaqen v3 render); measure first, then normalize
        _LN = "loudnorm=I=-16.5:TP=-2.0:LRA=11"
        p1 = subprocess.run(
            [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(src_path),
             "-t", f"{dur:.3f}", "-vn", "-af", _LN + ":print_format=json",
             "-f", "null", "-"], capture_output=True, timeout=120)
        _ln = _LN
        try:
            _jt = (p1.stderr or b"").decode("utf-8", "ignore")
            _jm = _re10.search(r"\{[^{}]*\"input_i\"[^{}]*\}", _jt, _re10.S)
            _d2 = _json10.loads(_jm.group(0))
            _ln = (_LN + f":measured_I={_d2['input_i']}:measured_TP={_d2['input_tp']}"
                   f":measured_LRA={_d2['input_lra']}:measured_thresh={_d2['input_thresh']}"
                   f":offset={_d2['target_offset']}:linear=true")
        except Exception:
            pass                                       # fall back to single-pass
        pa = subprocess.run(
            [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(src_path),
             "-t", f"{dur:.3f}", "-vn",
             "-af", f"{_ln},afade=t=in:st=0:d=0.12,"
                    f"afade=t=out:st={max(0.1, dur - 0.15):.3f}:d=0.15",
             "-ar", "44100", "-ac", "2", str(adest)],
            capture_output=True, timeout=300)
        if pv.returncode != 0 or pa.returncode != 0 or not vdest.exists() or not adest.exists():
            return None
        from .ingest import probe
        da = probe(adest).get("duration", 0.0) or dur
        dv = probe(vdest).get("duration", 0.0) or dur
        return round(min(da, dv), 3)
    except Exception:
        return None


def _quote_run_in(qwords: list, twords: list) -> int:
    """Longest leading-run of the quote spoken in the transcript (1-word ASR drift tolerated
    via 3-word sub-windows). 0 = not spoken here."""
    if len(qwords) < 3 or not twords:
        return 0
    tstr = " " + " ".join(twords) + " "
    best = 0
    for j in range(len(qwords), 2, -1):
        if (" " + " ".join(qwords[:j]) + " ") in tstr:
            best = j
            break
    if best == 0:
        for i in range(len(qwords) - 2):
            if (" " + " ".join(qwords[i:i + 3]) + " ") in tstr:
                best = 3
                break
    return best


_ESSAYISH_RX = re.compile(
    r"\b(why|theor(?:y|ies)|explained?|breakdown|analysis|analy(?:z|s)e|video essay|essay|"
    r"you missed|missed (?:it|this)|real (?:reason|plan|meaning|story|mission)|really|"
    r"secret|hidden|truth|history of|the story of|what if|top \d+|rank(?:ed|ing)|"
    r"documentary|deep dive|everything (?:we|you) know|"
    # narrated commentary / interview / promo — NOT clean scene audio for a breakout:
    r"psycholog|toxic|the (?:core|meaning|psychology) of|best scenes|supercut|"
    r"featurette|behind the|on playing|interview|reacts?|reaction|"
    # edit/essay/reaction TITLES that slipped onto a Tywin S01E07 video and aired commentary:
    r"the scene where|the scene that|fatal mistake|biggest mistake|costly mistake|"
    r"lost because|lost (?:her|his|the) [a-z]+ because|almost cries|motivational|tribute|edit\b|amv|"
    r"educat(?:es|ing|ion)|watching|watch ?along|first time|the sack of|the story of|"
    r"vs fan|plot hole|q ?& ?a|explains?|commentary)\b", re.I)

_EN_FN_WORDS = {
    "the", "and", "you", "that", "this", "what", "have", "will", "they", "with",
    "your", "for", "not", "but", "was", "his", "her", "she", "him", "are", "were",
    "don't", "can't", "won't", "didn't", "there", "here", "when", "then", "them",
    "who", "would", "could", "should", "from", "about", "want", "know", "going",
}


def _english_ish(words: list) -> bool:
    """True when an ASR word list reads as ENGLISH speech (>=12% common function words).
    Foreign-dub sources (FR/RO uploads) pass dialogue verification via translated captions,
    but their AUDIO must never become a breakout."""
    if len(words) < 4:
        return False
    return sum(1 for w in words if w in _EN_FN_WORDS) / len(words) >= 0.12


def _breakout_src_ok(src, shots) -> bool:
    """Audio-trust gate for real-audio breakouts: only IN-SHOW dialogue may play. Video
    essays/documentaries narrate wall-to-wall and QUOTE the very lines our script quotes —
    a breakout would air another narrator's voice over our pause. Excluded on either
    signal: essay-ish title, or extreme speech coverage (real scenes are bursty: most
    shots carry little or no dialogue, essays speak over >=75% of shots).

    NOTE: breakout selection scans proj.sources directly (not the match pool), so the match-pool
    reaction-drop does NOT protect it — the reaction/facecam check must be repeated HERE. The
    _ESSAYISH_RX \\b...\\b boundary also misses the PLURAL 'Reactions' (a real leak: a "TOP ...
    Reactions!" video aired a profanity-laced reaction over a pause), so use _REACTION_TITLE too."""
    title = getattr(src, "title", "") or ""
    if _ESSAYISH_RX.search(title):
        return False
    try:
        from .discover import _REACTION_TITLE
        if _REACTION_TITLE.search(title):
            return False
    except Exception:
        pass
    if shots:
        rich = sum(1 for sh in shots
                   if len((getattr(sh, "transcript", "") or "").split()) >= 6)
        if rich / len(shots) >= 0.75:
            return False
    return True


def _breakout_window_luma(src_path, start: float, dur: float) -> float:
    """Mean luma (YAVG, 0–255) across a breakout's air window — one signalstats pass over
    [start, start+dur] on a 128×72 decode. Returns **-1.0 if unreadable** (a failed probe is
    'unknown', never 'dark', so a transient ffmpeg error can never wrongly reject a shot)."""
    try:
        import re as _rel9
        p = subprocess.run(
            [ffmpeg_exe(), "-y", "-hide_banner", "-nostats",
             "-ss", f"{max(0.0, float(start)):.3f}", "-i", str(src_path),
             "-t", f"{max(0.5, float(dur)):.3f}", "-an",
             "-vf", "scale=128:72,signalstats,"
                    "metadata=print:key=lavfi.signalstats.YAVG:file=-",
             "-f", "null", "-"],
            capture_output=True, timeout=120)
        blob = (p.stdout or b"").decode("utf-8", "ignore") \
            + (p.stderr or b"").decode("utf-8", "ignore")
        vals = [float(m) for m in _rel9.findall(r"YAVG=([0-9.]+)", blob)]
        return (sum(vals) / len(vals)) if vals else -1.0
    except Exception:
        return -1.0


def _combine_opening_hook(segments):
    """Analyze frequently SPLITS the opening hook into tiny per-beat fragments — e.g.
    "Seize him." / "Cut his throat." / "Stop." / "Wait — I've changed my mind." — each below the
    3-word quote floor, so the literal opening line never becomes a cold-open candidate. Stitch the
    LEADING run of consecutive quoted opening beats (within the first 6) into ONE hook quote mapped
    to the FIRST beat. Returns (first_seg, combined_quote, {combined_beat_indices}) or None."""
    frag = []
    for seg in (segments or [])[:6]:
        q = (getattr(seg, "quote", "") or "").strip()
        if q:
            frag.append((seg, q))
        elif frag:
            break                                      # hook ended at the first quote-less beat
    if len(frag) >= 2 and len(" ".join(q for _s, q in frag).split()) >= 3:
        return frag[0][0], " ".join(q for _s, q in frag), {s.index for s, _q in frag}
    return None


def _select_breakouts(proj, segments, total: float, work: Path, log) -> list:
    """Pick the 1-3 most NATURAL breakout moments: beats whose narration QUOTES a line, located
    by searching the line in every source's own ASR — the breakout plays the SHOT where the line
    is actually spoken (independent of which clip the matcher picked for the beat)."""
    import re as _re9
    from . import index as _index

    def _rw(t):
        return _re9.findall(r"[a-z']+", (t or "").lower())

    # quote pool: per-beat quotes PLUS the analysis' verbatim anchor DIALOGUE lines (per-beat
    # quotes are often paraphrases; anchor dialogue is demanded verbatim) — each anchor line is
    # mapped to the beat whose narration overlaps it most (that beat is "about" the line)
    quote_segs = [(seg, seg.quote) for seg in segments
                  if len((getattr(seg, "quote", "") or "").split()) >= 3]
    try:
        ana = (getattr(proj, "meta", None) or {}).get("analysis", {})
        seen_q = {q.lower() for _s, q in quote_segs}
        for sc in (ana.get("anchor_scenes") or []):
            for dlg in (sc.get("dialogue") or []):
                if len(dlg.split()) < 3 or dlg.lower() in seen_q:
                    continue
                dw = set(_rw(dlg))
                best_seg, best_ov = None, 0
                for seg in segments:
                    if not seg.text:
                        continue
                    ov = len(dw & set(_rw(seg.text)))
                    if ov > best_ov:
                        best_ov, best_seg = ov, seg
                if best_seg is not None and best_ov >= 1:
                    quote_segs.append((best_seg, dlg))
    except Exception:
        pass
    # COLD-OPEN HOOK: stitch a fragmented opening hook ("Seize him." / "Cut his throat." / "Stop." /
    # "Wait — I've changed my mind.") into ONE scene-0 candidate (replacing the individual fragments),
    # so the literal opening real scene can air as the cold-open instead of just one mid-fragment.
    _ohook = _combine_opening_hook(segments)
    if _ohook is not None:
        _ohs, _ohq, _ohidx = _ohook
        quote_segs = [(s, q) for (s, q) in quote_segs if s.index not in _ohidx]
        quote_segs.insert(0, (_ohs, _ohq))
    # Do NOT early-return on an empty quote pool. Per-beat quotes / anchor dialogue are routinely
    # absent for ANALYTICAL/essay scripts (the narrator analyses the scene rather than quoting a line)
    # and the LLM returns anchor dialogue inconsistently under load — so quote_segs can be empty even
    # when the footage holds perfectly good in-character lines. The EVIDENCE-MINING pass below
    # ("always runs") finds natural real-audio moments from anchor-source dialogue overlapping the
    # narration, fully gated (era / recap / commentary / wrong-character Face-ID / burned-text / luma);
    # the terminal `if not cands: return []` is the real bail-out. (Was: empty quote pool → 0 breakouts
    # even on a 16-min essay with rich dialogue footage — the Drogon render.)
    srcs = [s for s in proj.sources
            if s.status == "ok" and s.local_path and Path(s.local_path).exists()]
    shots_of = {}
    for s in srcs:
        try:
            shots_of[s.id] = _index.load_shots(proj, s.id)
        except Exception:
            shots_of[s.id] = []
    ok_audio = {s.id for s in srcs if _breakout_src_ok(s, shots_of.get(s.id) or [])}
    _src_excluded = len(srcs) - len(ok_audio)
    if _src_excluded:
        log(f"build: breakout audio gate — {_src_excluded} narration-style/"
            f"foreign source(s) excluded, {len(ok_audio)} eligible")
    # breakout AUDIT counters — surfaced as ONE summary line per render so v4+ reports exactly how
    # many candidates were found and why each was rejected (commentary / recap-wrong-era /
    # wrong-character / dark / dedup / later-era source).
    _rej = {"later_era_source": 0, "commentary": 0, "recap": 0, "wrong_char": 0,
            "dark": 0, "burned_text": 0, "dedup": 0, "spacing": 0, "window_commentary": 0}
    from .match import _ocr_text_heavy as _txt9
    import os as _os9b
    _tgate9 = _os9b.environ.get("VIDLORE_CLIPSTUDIO_TEXT_GATE", "1").strip() \
        not in ("0", "false", "no", "")
    cands = []
    # STRONG-VERBATIM set: (seg_idx, src_id, round(start,1)) for matches where 4+ CONSECUTIVE scripted
    # words are spoken verbatim in the footage's own ASR. The exact scripted line located inside the
    # footage's audio is a STRONGER scene-identity proof than a face — a wrong scene almost never
    # contains the same 4-word line. So these are EXEMPT from the wrong-character Face-ID gate (a dim /
    # profile / over-the-shoulder shot that Face-ID can't confirm still airs IF it speaks the exact
    # line). Every OTHER gate still applies (era / recap / commentary / burned-text / luma). This is
    # what lets an iconic quoted line — e.g. an opening "Seize him. Cut his throat." cold-open — air
    # even when the throne-room shot is too dim for Face-ID. (User's cold-open / "let the scene speak".)
    _verbatim_strong = set()
    _verbatim_first = None        # (seg_idx, key) of the EARLIEST verbatim quote = cold-open hook
    for seg, _q in quote_segs:
        qw = _rw(_q)[:8]
        best = None
        for s in srcs:
            if s.id not in ok_audio:
                continue
            for sh in shots_of.get(s.id, []):
                if _tgate9 and _txt9(sh):
                    continue                           # burned-in text never airs
                run = _quote_run_in(qw, _rw(getattr(sh, "transcript", "")))
                if run >= 3:
                    # +3: a verbatim script quote located in the footage's own ASR is the
                    # strongest naturalness signal — it must outrank mined evidence lines
                    score = run + 3 + (2 if (getattr(s, "extra", None) or {}).get("anchor_verified")
                                       else 0)
                    if best is None or score > best[0]:
                        best = (score, s, sh, run)
        if best is not None:
            cands.append((best[0], seg.index, best[1], best[2], _q))
            _k = (seg.index, best[1].id, round(float(best[2].start), 1))
            if best[3] >= 4:                           # 4+ exact consecutive words → trust over Face-ID
                _verbatim_strong.add(_k)
            if _verbatim_first is None or seg.index < _verbatim_first[0]:
                _verbatim_first = (seg.index, _k)
    # COLD-OPEN HOOK: the EARLIEST verbatim quote that falls in the opening stretch is the hook — the
    # user wants the real scene of the opening quoted line (e.g. "Seize him. Cut his throat.") to air
    # right at the start. Exempt it from the Face-ID gate too, so a short 3-word opening quote over a
    # dim throne-room shot still airs (a wrong scene won't speak that exact opening line).
    if _verbatim_first is not None and _verbatim_first[0] <= max(5, len(segments) // 12):
        _verbatim_strong.add(_verbatim_first[1])
    if True:
        # EVIDENCE MINING — always runs, not just as a fallback. The competitor's edit uses the
        # scene's OWN spoken lines as evidence every ~15-30s (narration makes a point, the scene
        # proves it in its own voice). The LLM quote can be a paraphrase or a hallucination —
        # the footage's ASR is ground truth. A dialogue-rich shot qualifies when its words
        # OVERLAP a beat's narration (the show speaks about the same thing the narration is
        # explaining — exactly when a real-audio breakout feels natural).
        from .segment import _STOP as _STOP9
        # restrict mining to ANCHOR-scene sources (episode-code / verified / scene-title match) —
        # a compilation's unrelated episode dialogue must not become a breakout
        _ana9 = (getattr(proj, "meta", None) or {}).get("analysis", {})
        _ep9 = (_ana9.get("episode_hint") or "").lower().replace(" ", "")
        # ERA-COHERENCE for breakouts: a breakout airs a FULL movie scene with its OWN audio, so a
        # LATER-season shot (a bearded S7 Tyrion on a beach over an S4E10 privy scene) reads as an
        # obvious continuity break. Bar sources whose title declares ONLY seasons later than the core
        # scene. EARLIER seasons stay eligible — the script narrates backstory/flashbacks (birth,
        # the S1 slap, the S3 Casterly Rock denial). Fails open: no core season / no season in the
        # title → unchanged. Affects breakouts only, never the regular footage pool.
        import re as _re9b
        def _seasons9(txt):
            out = set()
            for m in _re9b.finditer(r"s(?:eason)?\s*0?(\d{1,2})\s*(?:e|x|ep|episode)\s*0?\d{1,2}"
                                    r"|(\d{1,2})x\d{2}", txt or "", _re9b.I):
                out.add(int(m.group(1) or m.group(2)))
            for m in _re9b.finditer(r"\bseason\s*0?(\d{1,2})\b", txt or "", _re9b.I):
                out.add(int(m.group(1)))
            return out
        _core_seasons9 = set()
        for _sc9 in (_ana9.get("anchor_scenes") or []):
            _core_seasons9 |= _seasons9((_sc9.get("episode", "") or "") + " "
                                        + (_sc9.get("query", "") or ""))
        _core_max9 = max(_core_seasons9) if _core_seasons9 else 0
        # comparison exception: a later-era / different-installment clip may air ONLY when the script
        # explicitly compares ("unlike House of the Dragon ...") — otherwise the era-gate stays strict.
        _allow_compare9 = _script_wants_comparison(segments)
        # main cast (character + actor names) — a breakout SHOULD feature one of them; a wrong-
        # character shot (e.g. a bearded man on a boat over a Tyrion/Tywin scene) must not air.
        _main_faces9 = set()
        for _ch9 in (_ana9.get("characters") or []):
            for _k9 in ("name", "actor"):
                _v9 = (_ch9.get(_k9) or "").strip().lower()
                if _v9:
                    _main_faces9.add(_v9)
        _tier1, _tier2 = set(), set()
        _mv9 = {w for w in _rw(_ana9.get("movie_title", "") or "")}
        for s in srcs:
            if s.id not in ok_audio:
                continue
            _t9 = (s.title or "").lower()
            if _core_max9 and not _allow_compare9:
                _ss9b = _seasons9(_t9)
                if _ss9b and min(_ss9b) > _core_max9:
                    _rej["later_era_source"] += 1
                    continue                           # purely later-season source → never a breakout
            if (getattr(s, "extra", None) or {}).get("anchor_verified") or \
                    (_ep9 and _ep9 in _t9.replace(" ", "")):
                _tier1.add(s.id)                       # the EXACT scene/episode — trusted
                continue
            for sc9 in (_ana9.get("anchor_scenes") or []):
                toks9 = {w for w in _rw((sc9.get("name", "") + " " + sc9.get("query", "")))
                         if len(w) > 3 and w not in _mv9}      # movie-name tokens match EVERY title
                if sum(1 for w in toks9 if w in _t9) >= 2:
                    _tier2.add(s.id)                   # scene-titled, but could be another episode
                    break
        def _mine_tier(tier_srcs, min_ov):
            found = []
            for s in tier_srcs:
                for sh in shots_of.get(s.id, []):
                    if _tgate9 and _txt9(sh):
                        continue                       # burned-in text never airs
                    raw9 = _rw(getattr(sh, "transcript", ""))
                    if not _english_ish(raw9):
                        continue                       # foreign-dub audio never breaks out
                    tw = [w for w in raw9 if len(w) > 2 and w not in _STOP9]
                    if len(tw) < 4 or not (2.0 <= (sh.end - sh.start) <= 9.0):
                        continue
                    tset = set(tw)

                    def _ov_count(bw):
                        n = 0
                        for b in bw:
                            for t in tset:
                                if b == t or (b.startswith(t) and len(b) - len(t) <= 2) \
                                        or (t.startswith(b) and len(t) - len(b) <= 2):
                                    n += 1
                                    break
                        return n
                    best_seg, best_ov = None, -1
                    for seg in segments:
                        if not seg.text:
                            continue
                        ov = _ov_count({w for w in _rw(seg.text) if len(w) > 2
                                        and w not in _STOP9})
                        if ov > best_ov:
                            best_ov, best_seg = ov, seg
                    if best_seg is None or best_ov < min_ov:
                        continue
                    # score: beat-relevance + how much is actually SPOKEN in the shot
                    score = best_ov * 2 + min(4, len(tw) // 3)
                    found.append((score, best_seg.index, s, sh,
                                  (getattr(sh, "transcript", "") or "")[:60]))
            return found

        # tier 1 (the EXACT episode/scene) AND tier 2 (scene-titled): demand REAL narration overlap
        # (>=2 content words) so a breakout line is actually ABOUT what the narration is saying at
        # that point — a single shared word let tangential lines air (e.g. a Tywin-throne "I did not
        # do" over a strangling beat). Mined evidence MERGES with located verbatim quotes; only if
        # nothing matched at all do we fall back to tier-1 "any line" (the scene IS the subject).
        mined = (_mine_tier([s for s in srcs if s.id in _tier1], 2)
                 + _mine_tier([s for s in srcs if s.id in _tier2], 2))
        if not cands and not mined:
            mined = _mine_tier([s for s in srcs if s.id in _tier1], 0)
        _seen_shot = {(c[2].id, int(float(c[3].start) * 10)) for c in cands}
        for c in mined:
            kk = (c[2].id, int(float(c[3].start) * 10))
            if kk not in _seen_shot:
                _seen_shot.add(kk)
                cands.append(c)
    if not cands:
        log("build: no breakout — no spoken line relates to the narration (natural skip)")
        return []
    # COLD-OPEN: scene 0 is normally the title overlay, but the OPENING verbatim quote (the hook,
    # e.g. "Seize him. Cut his throat.") is allowed to air at scene 0 — KEEP a scene-0 candidate only
    # when it is a strong-verbatim / cold-open match, NEVER a generic evidence-mined one.
    _cold_key = (_verbatim_first[1] if (_verbatim_first is not None
                 and _verbatim_first[0] <= max(5, len(segments) // 12)) else None)
    cands = [c for c in cands
             if c[1] >= 1 or (c[1], c[2].id, round(float(c[3].start), 1)) in _verbatim_strong]
    # process the cold-open FIRST (reserve its slot before n_max / spacing fills up), then by score
    cands.sort(key=lambda x: (0 if (x[1], x[2].id, round(float(x[3].start), 1)) == _cold_key else 1,
                              -x[0]))
    # competitor density: one real-audio evidence moment every ~28s where material allows
    # (their edit: 13 in 180s on a single-scene essay; ours stays match-gated so sparse
    # scripts naturally get fewer) — was a hard 1-3 cap
    n_max = max(1, min(8, 1 + int(total // 28)))
    picked = []
    _picked_word_sets = []                             # for spoken-content dedup
    for c in cands:
        if len(picked) >= n_max:
            break
        if any(abs(c[1] - p[1]) < 2 for p in picked):
            _rej["spacing"] += 1
            continue                                   # >=2-scene spacing — natural, not forced
        # CONTENT DEDUP: never play the SAME spoken line as a breakout twice (an expert editor
        # wouldn't repeat "you raped her, you killed her children" in two places)
        _cw = {w for w in _rw(c[4])[:10] if len(w) > 2}
        if _cw and any(len(_cw & p) / max(1, len(_cw | p)) >= 0.5 for p in _picked_word_sets):
            _rej["dedup"] += 1
            continue
        # COMMENTARY-AUDIO gate: a breakout plays the SHOT's OWN audio. If that audio is a
        # YouTuber/essayist ANALYZING the scene (third-person commentary) rather than the movie's
        # in-character dialogue, it must never air over the narration pause. The shot transcript IS
        # the airing audio. Essay sources slip the title/coverage gates AND match by overlap (their
        # commentary about the scene naturally overlaps the script about the same scene), so this
        # CONTENT check on the airing audio is the reliable catch. (Observed: 3 of 4 breakouts on
        # the Tywin S01E07 video were essay narration — "Basically what the red wedding is to
        # Tywin...", "all that warmth he once had for her got buried...".)
        if _is_narration(getattr(c[3], "transcript", "") or ""):
            _rej["commentary"] += 1
            log(f"build: breakout skipped before scene {c[1]} — source audio is commentary/"
                f"narration, not in-character movie dialogue")
            continue
        # WRONG-ERA / WRONG-SCENE RECAP gate: a later-scene retrospective ("Last time I was here, I
        # killed my father with a crossbow" = S7 Tyrion recapping the S4 act) is in-character but the
        # wrong era/scene. Bar it unless the script explicitly asks for a comparison.
        if not _allow_compare9 and _is_recap_line(getattr(c[3], "transcript", "") or ""):
            _rej["recap"] += 1
            log(f"build: breakout skipped before scene {c[1]} — retrospective recap line "
                f"(wrong era/scene, not the scene's own dialogue)")
            continue
        # WRONG-ERA SOURCE gate — verbatim / cold-open candidates are NOT tier-era-filtered (the
        # evidence-miner pre-filters its sources, the verbatim-quote loop does not), so re-check here:
        # a source whose title declares ONLY later seasons than the core scene never airs (unless the
        # script explicitly compares). Applies to all candidates; mined ones already passed, so no-op.
        if _core_max9 and not _allow_compare9:
            _css9 = _seasons9((c[2].title or "").lower())
            if _css9 and min(_css9) > _core_max9:
                _rej["later_era_source"] += 1
                log(f"build: breakout skipped before scene {c[1]} — later-era source "
                    f"(declares season > core scene S{_core_max9})")
                continue
        # WRONG-CHARACTER gate: a breakout airs an iconic MOMENT, so the shot must feature a confirmed
        # MAIN character (Face-ID). A shot showing no main-cast face (e.g. a bearded man on a boat over
        # a Tyrion/Tywin scene) must not air. Active only when the cast is known; a breakout is optional
        # polish, so a missed Face-ID just means no breakout there (safe), never a wrong one.
        _verbatim_ok = (c[1], c[2].id, round(float(c[3].start), 1)) in _verbatim_strong
        if _main_faces9 and not _verbatim_ok:
            _fids9 = {f.lower() for f in (getattr(c[3], "face_ids", None) or [])}
            if not (_fids9 & _main_faces9):
                _rej["wrong_char"] += 1
                log(f"build: breakout skipped before scene {c[1]} — shot shows no confirmed main "
                    f"character (wrong-character/scene guard)")
                continue
        elif _verbatim_ok and _main_faces9:
            _fids9b = {f.lower() for f in (getattr(c[3], "face_ids", None) or [])}
            if not (_fids9b & _main_faces9):
                log(f"build: breakout before scene {c[1]} — Face-ID unconfirmed but EXACT scripted "
                    f"line is spoken verbatim in this shot (audio-match overrides face guard)")
        if _tgate9 and getattr(c[2], "local_path", None) and (
                _frame_has_burned_text(c[2].local_path, float(c[3].start) + 0.8)
                or _frame_has_burned_text(
                    c[2].local_path,
                    float(c[3].start) + min(3.0, max(0.8, (float(c[3].end)
                                                           - float(c[3].start)) * 0.6)))):
            _rej["burned_text"] += 1
            continue                                   # air-time probe: text never airs
        # LEGIBILITY GATE: a breakout is meant to be an ICONIC, instantly-recognizable movie
        # moment. A near-black / heavily-degraded shot whose subject is unidentifiable reads as
        # a frozen dead frame even when its own ASR overlaps the narration — observed: the
        # VHS-glitched "Angels with Even Filthier Souls" in-movie film (mean-luma ~55) aired ~8s
        # over a present-day Macaulay-Culkin beat. Reject on the air-window's mean luma; the beat
        # then keeps its normally-selected, well-exposed footage (a safe fallback, not a gap).
        # A negative probe (unreadable) is treated as unknown and never rejects.
        # env VIDLORE_CLIPSTUDIO_BREAKOUT_LUMA_FLOOR — 0 disables the gate.
        _lflo9 = float(_os9b.environ.get(
            "VIDLORE_CLIPSTUDIO_BREAKOUT_LUMA_FLOOR", "62") or 62)
        # A verbatim / cold-open match IS the exact scene we asked for, and many iconic scenes are
        # LEGITIMATELY dim (candle-lit throne room, night). Don't reject those on the legibility
        # floor — only on TRUE near-black (subject genuinely invisible). Other breakouts keep the
        # full floor. This is what lets a dim "Power is power" / "Cut his throat" cold-open air.
        if _verbatim_ok:
            _lflo9 = min(_lflo9, 24.0)
        if _lflo9 > 0 and getattr(c[2], "local_path", None):
            _bwin9 = min(9.0, max(2.0, float(c[3].end) - float(c[3].start)))
            _blum9 = _breakout_window_luma(c[2].local_path, float(c[3].start), _bwin9)
            if 0.0 <= _blum9 < _lflo9:
                _rej["dark"] += 1
                log(f"build: breakout skipped before scene {c[1]} — too dark/illegible to "
                    f"air (mean-luma {_blum9:.0f} < floor {_lflo9:.0f})")
                continue
        picked.append(c)
        _picked_word_sets.append({w for w in _rw(c[4])[:10] if len(w) > 2})

    # COLD-OPEN OWNS THE OPENING: drop any OTHER breakout that lands inside the combined hook beats
    # (e.g. an evidence-mined "I've changed my mind" at beat 3 that duplicates the cold-open). It is
    # redundant with the cold-open, AND for the VO word-cut it would splice INSIDE the hook span and
    # push the hook's tail beyond the locate window — forcing a needless fallback.
    if _ohook is not None and _cold_key is not None:
        _hb = _ohook[2]
        picked = [c for c in picked
                  if c[1] not in _hb
                  or (c[1], c[2].id, round(float(c[3].start), 1)) == _cold_key]

    # ── BREAKOUT AUDIT (one greppable line per render) — candidates found, per-reason rejections, kept.
    # Grep: `grep BREAKOUT-AUDIT <log>` for the summary; `grep BREAKOUT-OK <log>` for accepted ones. ──
    log(f"[BREAKOUT-AUDIT] candidates={len(cands)} pre_extract_accepted={len(picked)} | "
        f"rejected commentary={_rej['commentary']} recap/wrong-era={_rej['recap']} "
        f"wrong-character={_rej['wrong_char']} dark={_rej['dark']} burned-text={_rej['burned_text']} "
        f"dedup={_rej['dedup']} spacing={_rej['spacing']} | pre-filtered "
        f"later-era-source={_rej['later_era_source']} essay/foreign-source={_src_excluded}")
    # SELF-DIAGNOSTIC: surface WHY when too few breakouts were accepted, so we can decide (per the
    # 'conservative Face-ID gate' policy) whether to loosen — only on real evidence, never pre-emptively.
    if len(cands) >= 2 and len(picked) <= max(0, len(cands) // 6):
        _top = max(_rej, key=_rej.get) if any(_rej.values()) else "none"
        if _rej["wrong_char"] >= max(1, len(cands) // 2):
            log(f"[BREAKOUT-AUDIT] NOTE: low accept ({len(picked)}/{len(cands)}) — the conservative "
                f"Face-ID gate is the MAIN limiter (wrong-character={_rej['wrong_char']}). Loosen ONLY "
                f"if this render visibly has too few breakouts.")
        else:
            log(f"[BREAKOUT-AUDIT] NOTE: low accept ({len(picked)}/{len(cands)}) — main reason: "
                f"{_top}={_rej.get(_top, 0)} (not the Face-ID gate).")

    out = []
    for score, idx, src, sh, _q in sorted(picked, key=lambda p: p[1]):
        # pass the 10s MAX cap — _extract_breakout finds the real length that ends on a
        # complete spoken line (3-10s), so an iconic quote is never chopped mid-word
        dur = 10.0
        v = work / f"breakout_{idx:03d}.mp4"
        a = work / f"breakout_{idx:03d}.wav"
        real = _extract_breakout(src.local_path, float(sh.start), dur, v, a,
                                 int(getattr(src, "width", 0) or 0))
        if not (real and real > 1.5):
            continue
        # POST-EXTRACTION WINDOW-AUDIO gate — the matched SHOT line passed the commentary gate, but
        # _extract_breakout extends the window to a full spoken line (3-10s), which can BLEED into
        # adjacent essay/commentary narration from the SAME source (observed: a 'Cersei's Fatal
        # Mistake ...' analysis source matched the in-character "chaos is a ladder" line, but the
        # aired window continued into "...Cersei spent multiple seasons after the purple wedding").
        # Validate the AIRED window's transcript (every shot it spans), not just the matched line.
        _w0, _w1 = float(sh.start), float(sh.start) + float(real)
        _wtxt = " ".join(
            (getattr(s2, "transcript", "") or "").strip()
            for s2 in shots_of.get(src.id, [])
            if s2.end > _w0 and s2.start < _w1 and (getattr(s2, "transcript", "") or "").strip())
        if _wtxt and _is_narration(_wtxt):
            _rej["window_commentary"] += 1
            log(f"build: breakout REJECTED post-extract before scene {idx} — aired audio window is "
                f"commentary/narration, not in-character dialogue "
                f"(src={(src.title or src.id)[:40]!r})")
            continue                                   # try the next candidate; none clean → no breakout
        _is_cold = (idx, src.id, round(float(sh.start), 1)) == _cold_key
        _entry = {"seg_index": idx, "dur": real, "video": v, "audio": a}
        if _is_cold:
            # carry the hook quote + the beats it was stitched from, so the VO word-cut (default ON
            # for uploaded voiceover) can locate the narrator's rendition of the hook and replace it.
            _entry["cold_open"] = True
            _entry["hook_quote"] = _q
            _entry["hook_beats"] = sorted(_ohook[2]) if _ohook is not None else [idx]
        out.append(_entry)
        # ACCEPTED breakout provenance — scene index + SOURCE TITLE + source timestamp + the line.
        # Tag the opening verbatim hook as COLD-OPEN so the audit shows it aired at the start.
        _cold = "COLD-OPEN " if _is_cold else ""
        log(f"[BREAKOUT-OK] #{len(out)} {_cold}before-scene={idx} dur={real:.1f}s "
            f"src@{float(sh.start):.0f}s src={(src.title or src.id)[:52]!r} line={_q[:42]!r}")
    # ALWAYS report the FINAL accepted count after post-extraction (the pre_extract_accepted count
    # above can shrink here when an aired window is rejected as commentary) — so the audit is never
    # ambiguous about how many breakouts actually aired.
    log(f"[BREAKOUT-AUDIT] final accepted={len(out)} (post-extract; "
        f"window_commentary rejections={_rej['window_commentary']})")
    return out


def _splice_audio(full: Path, splices: list, work: Path) -> Optional[Path]:
    """Insert audio segments into `full` at the given ORIGINAL-timeline times.
    splices = [(T_seconds, wav_path, dur), ...] ascending."""
    if not splices:
        return None
    ins, fparts, labels = ["-i", str(full)], [], []
    prev = 0.0
    n = 0
    for i, (t, wav, _d) in enumerate(splices, start=1):
        ins += ["-i", str(wav)]
        fparts.append(f"[0:a]atrim={prev:.3f}:{t:.3f},asetpts=PTS-STARTPTS[s{n}]")
        labels.append(f"[s{n}]")
        n += 1
        labels.append(f"[{i}:a]")
        prev = t
    fparts.append(f"[0:a]atrim={prev:.3f},asetpts=PTS-STARTPTS[s{n}]")
    labels.append(f"[s{n}]")
    fc = ";".join(fparts) + f";{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"
    dest = work / "narration_breakouts.wav"
    p = subprocess.run([ffmpeg_exe(), "-y", *ins, "-filter_complex", fc, "-map", "[out]",
                        "-ar", "44100", "-ac", "2", str(dest)],
                       capture_output=True, timeout=600)
    return dest if (p.returncode == 0 and dest.exists() and dest.stat().st_size > 0) else None


_FRAME_TXT_CACHE: dict = {}


def _frame_has_burned_text(src_path, t) -> bool:
    """OCR the ACTUAL frame at t (accurate ffmpeg -ss seek). The indexed keyframe can miss
    overlay text that appears mid-shot (comment-card motion graphics build up AFTER the
    keyframe — observed: essay source clean keyframe OCR, cartoon+tweet-cards 1s later).
    USER RULE: footage with readable on-screen text never airs."""
    try:
        from . import ocr as _ocr_rt
        if not _ocr_rt.available():
            return False
        import os as _os12
        import re as _re12
        import tempfile
        key = (str(src_path), round(float(t), 1))
        if key in _FRAME_TXT_CACHE:
            return _FRAME_TXT_CACHE[key]
        fd, png = tempfile.mkstemp(suffix=".png")
        _os12.close(fd)
        p = subprocess.run([ffmpeg_exe(), "-y", "-ss", f"{max(0.0, float(t)):.2f}",
                            "-i", str(src_path), "-frames:v", "1", png],
                           capture_output=True, timeout=60)
        val = False
        if p.returncode == 0 and Path(png).exists() and Path(png).stat().st_size > 0:
            txt = _ocr_rt.read_text(png)
            val = (len(_re12.findall(r"[A-Za-z']{3,}", txt or "")) >= 3
                   or _ocr_rt.has_big_text(png))     # large overlay caption — geometry test
        try:
            _os12.unlink(png)
        except OSError:
            pass
        _FRAME_TXT_CACHE[key] = val
        return val
    except Exception:
        return False


def _refine_pause_times(narration_audio: Path, needs: dict) -> dict:
    """Snap estimated pause points to REALITY: transcribe the narration once with whisper
    word timestamps and return {beat_index: actual_end_of_boundary_word + breath}. The
    build's WordTimings are proportional estimates (edge-tts emits no reliable per-word
    boundaries) — a pause placed off a real word boundary cuts a word in half."""
    import re as _re11
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return {}
    try:
        m = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _info = m.transcribe(str(narration_audio), word_timestamps=True,
                                   vad_filter=False)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append((_re11.sub(r"[^a-z']", "", (w.word or "").lower()),
                              float(w.start), float(w.end)))
    except Exception:
        return {}
    out = {}
    for old, (t_est, tok) in needs.items():
        tt = _re11.sub(r"[^a-z']", "", (tok or "").lower())
        if not tt:
            continue
        best = None
        for wtxt, _ws, we in words:
            if wtxt == tt and abs(we - t_est) <= 2.5:
                if best is None or abs(we - t_est) < abs(best - t_est):
                    best = we
        if best is not None:
            out[old] = best + 0.12
    return out


def _locate_hook_span(vo_path, hook_quote, log, *, max_scan: float = 35.0, max_span: float = 18.0):
    """Locate the narrator's rendition of the cold-open hook in the uploaded VOICEOVER via WORD-LEVEL
    whisper timestamps (NOT beat durations). Returns (h0, h1, match_ratio) — start of the first
    matched hook word, end of the last — or None if the match is weak. The semantic gate: the VO
    must actually OPEN with the hook words (>=60% matched, in order, near t=0). Word boundaries are
    the primary source; beat durations are only a downstream reference/fallback."""
    import re as _re
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    hook_w = [w for w in _re.findall(r"[a-z']+", (hook_quote or "").lower()) if len(w) > 1 or w == "i"]
    if len(hook_w) < 3:
        return None
    try:
        m = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _ = m.transcribe(str(vo_path), word_timestamps=True, vad_filter=False)
        vow = []
        for s in segs:
            for w in (s.words or []):
                tok = _re.sub(r"[^a-z']", "", (w.word or "").lower())
                if tok:
                    vow.append((tok, float(w.start), float(w.end)))
            if vow and vow[-1][1] > max_scan:
                break
    except Exception:
        return None
    if len(vow) < 3:
        return None
    # Match the hook words IN ORDER from a VO start position, SKIPPING interleaved narration words
    # (the opening hook fragments are often split by narration: "Seize him. Cut his throat." [Two
    # seconds pass...] "Stop. Wait — I've changed my mind."). Bounded by max_span so it never runs
    # away; gated on coverage + a near-t=0 start so it only fires when the VO genuinely OPENS with it.
    n = len(hook_w)
    best = None
    for s0 in range(min(6, len(vow))):
        hi, vi, matched, first_t, last_t = 0, s0, 0, None, None
        while hi < n and vi < len(vow):
            if first_t is not None and vow[vi][1] - first_t > max_span:
                break
            if vow[vi][0] == hook_w[hi]:
                first_t = vow[vi][1] if first_t is None else first_t
                last_t = vow[vi][2]
                matched += 1
                hi += 1
            vi += 1
        if matched >= 3 and first_t is not None and first_t <= 3.0 \
                and (best is None or matched > best[0]):
            best = (matched, first_t, last_t)
    if best is None:
        return None
    matched, h0, h1 = best
    ratio = matched / n
    if ratio < 0.70 or h1 is None or h1 <= h0:             # semantic gate → caller falls back
        return None
    return (h0, h1, ratio)


def _audio_trim_prepend(vo_path, h1: float, clip_audio, d_clip: float, work: Path):
    """New narration track = [cold-open clip audio, trimmed to d_clip] + [voiceover from h1 onward].
    The prepend MUST be exactly d_clip long — the same quantity used for Δ and the cold-open scene
    duration — else the clip's own audio/video duration mismatch desyncs the whole remainder."""
    dest = work / "narration_vocut.wav"
    fc = (f"[0:a]atrim=0:{d_clip:.3f},asetpts=PTS-STARTPTS[c];"
          f"[1:a]atrim=start={h1:.3f},asetpts=PTS-STARTPTS[v];"
          f"[c][v]concat=n=2:v=0:a=1[out]")
    p = subprocess.run([ffmpeg_exe(), "-y", "-i", str(clip_audio), "-i", str(vo_path),
                        "-filter_complex", fc, "-map", "[out]", "-ar", "44100", "-ac", "2",
                        str(dest)], capture_output=True, timeout=600)
    return dest if (p.returncode == 0 and dest.exists() and dest.stat().st_size > 0) else None


def _apply_coldopen_vocut(proj, segments, scenes, narration, co, work, log, protect_idx=None):
    """Cold-open VO WORD-CUT — DEFAULT ON for word-aligned uploaded voiceover; kill switch
    env VIDLORE_CLIPSTUDIO_VO_CUT=0 (the caller also gates on narration._vo_word_aligned, so TTS
    renders never reach here). REPLACE the
    narrator's opening-hook words with the real-scene clip: locate the hook in the VOICEOVER by
    word-level timestamps, cut that span, prepend the clip, DROP the hook beats, shift the rest by
    Δ. Returns (segments, scenes, narration, bmap, idx_map, dropped_count) on success, or None to
    fall back to the proven insert behaviour. ANY uncertainty falls back — never breaks a render.
    Works on COPIES so a fallback leaves the caller's segments/scenes/narration pristine. protect_idx
    = scene indices that must NOT be dropped/trimmed (mid-video breakout pseudo-scenes) → fall back if
    the cut would touch one."""
    import copy as _copy
    from vidlore.script_gen import Scene as _EScene
    from vidlore.tts import NarratedScene as _NScene
    from .models import ScriptSegment as _CSeg
    protect = set(protect_idx or ())
    try:
        clip_audio, clip_video = co.get("audio"), co.get("video")
        d_clip = float(co.get("dur") or 0.0)
        hook_q = co.get("hook_quote") or ""
        if not (clip_audio and clip_video and d_clip > 1.0 and hook_q):
            return None
        loc = _locate_hook_span(narration.audio, hook_q, log)
        if loc is None:
            log("build: VO-cut fallback — opening hook not confidently located in the voiceover")
            return None
        h0, h1, ratio = loc
        ns_list = list(narration.scenes)
        cum, _t = {}, 0.0
        for ns in ns_list:
            cum[ns.index] = (_t, _t + float(ns.duration))
            _t += float(ns.duration)
        tol = 0.30
        dropped, survive, straddle = [], [], None
        for ns in ns_list:
            cs, ce = cum[ns.index]
            if ce <= h1 + tol:
                dropped.append(ns.index)               # scene audio fully inside the cut span
            elif cs >= h1 - tol:
                survive.append(ns.index)               # scene fully after the cut
            elif straddle is None:
                straddle = ns.index                    # the ONE scene crossing the cut point → trim
                survive.append(ns.index)
            else:
                log("build: VO-cut fallback — more than one beat straddles the cut point")
                return None
        if not dropped or not survive:
            log("build: VO-cut fallback — no clean hook/remainder split")
            return None
        # never cut/trim a mid-video breakout pseudo-scene (would corrupt that breakout)
        if protect & (set(dropped) | ({straddle} if straddle is not None else set())):
            log("build: VO-cut fallback — the cut would touch a mid-video breakout scene")
            return None
        new_audio = _audio_trim_prepend(narration.audio, h1, clip_audio, d_clip, work)
        if new_audio is None:
            log("build: VO-cut fallback — audio trim/prepend failed")
            return None
        delta = d_clip - h1
        seg_by = {s.index: s for s in segments}
        sc_by = {s.index: s for s in scenes}
        ns_by = {n.index: n for n in ns_list}
        new_segs, new_scs, new_ns, idx_map, bmap, cap_specs = [], [], [], {}, {}, []
        # cold-open pseudo-scene at index 0 (the real-scene clip replaces the hook narration)
        new_segs.append(_CSeg(index=0, text="", est_duration=d_clip))
        _sb = _EScene(index=0, narration="", keywords=[], visual="breakout")
        _sb.intensity, _sb.role, _sb.shot_type = 1, "evidence", "archival"
        new_scs.append(_sb)
        new_ns.append(_NScene(index=0, audio=Path(clip_audio), duration=d_clip, words=[]))
        bmap[0] = Path(clip_video)
        cap_specs.append({"start": 0.0, "dur": d_clip, "audio": str(clip_audio)})
        ni = 1
        for idx in sorted(survive):
            seg, sc, ns = seg_by.get(idx), sc_by.get(idx), ns_by.get(idx)
            if seg is None or ns is None:
                continue
            # work on COPIES — leave the caller's objects pristine so any later error/fallback is safe
            seg = _copy.copy(seg)
            seg.index = ni
            new_segs.append(seg)
            if sc is not None:
                sc = _copy.copy(sc)
                sc.index = ni
                new_scs.append(sc)
            ns = _copy.copy(ns)
            ns.index = ni
            _ws = [_copy.copy(w) for w in (getattr(ns, "words", None) or [])]
            if idx == straddle:
                # keep only the TAIL (words after the cut); the leading hook words went with the audio.
                cs, ce = cum[idx]
                ns.duration = max(0.1, ce - h1)
                _ws = [w for w in _ws if float(w.start) >= h1 - 0.05]
            for w in _ws:
                w.start = float(w.start) + delta
                w.end = float(w.end) + delta
            ns.words = _ws
            new_ns.append(ns)
            idx_map[idx] = ni
            ni += 1
        # mid-video breakout captions: DROP any inside the cut span (their audio was removed),
        # shift the rest by Δ; then add the cold-open cap (already at index 0 of cap_specs).
        for cs in (getattr(narration, "_breakout_caps", None) or []):
            if float(cs["start"]) < h1 - tol:
                continue
            cap_specs.append({"start": round(float(cs["start"]) + delta, 3),
                              "dur": cs["dur"], "audio": cs["audio"]})
        narration.audio = new_audio
        narration.scenes = new_ns
        narration._breakout_caps = cap_specs               # narration.total is a derived property
        log(f"build: VO-CUT cold-open — hook in VO [{h0:.2f}-{h1:.2f}s] match={ratio:.0%}; cut the "
            f"narrator's hook, dropped {len(dropped)} beat(s), prepended {d_clip:.1f}s clip (Δ={delta:+.2f}s)")
        return new_segs, new_scs, narration, bmap, idx_map, len(dropped)
    except Exception as e:                                 # noqa: BLE001
        log(f"build: VO-cut fallback — exception ({str(e)[:90]})")
        return None


def _apply_breakouts(proj, segments, scenes, narration, picks, work, log):
    """Insert breakout pseudo-scenes BEFORE their beats: reindex segments/scenes/narration,
    shift later word timings, splice the breakout audio into the narration track. Returns
    (segments, scenes, narration, breakout_clip_map, idx_map)."""
    from vidlore.script_gen import Scene as _EScene
    from vidlore.tts import NarratedScene as _NScene
    from .models import ScriptSegment as _CSeg
    ins_before = {p["seg_index"]: p for p in picks}
    sc_by_idx = {sc.index: sc for sc in scenes}
    ns_by_idx = {ns.index: ns for ns in narration.scenes}
    new_segs, new_scs, new_ns = [], [], []
    bmap, idx_map, splices = {}, {}, []

    # --- PASS 1 — CLAUSE-SAFE PAUSE points. Beat boundaries are duration-driven and
    # usually fall MID-SENTENCE ("...Varys warns | him plainly..."), so pausing exactly at
    # the boundary sounds broken. If the in-progress sentence finishes within the first
    # ~3.5s of the beat, DELAY the pause (and the breakout scene) to that full stop: the
    # previous scene plays delta longer, this beat starts delta shorter.
    # Sentence-end signals (ASR-style scripts often carry NO punctuation): (a) word ends
    # with .!?…  (b) a real TTS pause gap >=0.28s follows  (c) the next word is
    # Capitalized AND is a common word — "These/They/For/Men" capitalized mid-text marks
    # a new sentence, while a capitalized NAME (Arya, Varys) does not.
    from .segment import _STOP as _STOPB
    _cap_ok = set(_STOPB) | {"men", "man", "people", "everyone", "nobody",
                             "now", "later", "then", "think", "look", "watch"}

    def _sent_end(tok, nxt, gap):
        tok, nxt = (tok or "").strip(), (nxt or "").strip()
        return (tok[-1:] in ".!?…" or gap >= 0.28
                or (nxt[:1].isupper() and nxt.lower().strip(".,!?") in _cap_ok))

    _starts, _t0 = {}, 0.0
    for _n in narration.scenes:
        _starts[_n.index] = _t0
        _t0 += float(_n.duration)
    _ordered = sorted(ins_before)
    _deltas, _needs = {}, {}
    for old in _ordered:
        _deltas[old] = 0.0
        _nsx = ns_by_idx.get(old)
        _prv = ns_by_idx.get(old - 1)
        _prev_ok = True
        if _prv is not None and getattr(_prv, "words", None):
            _firstw = ((_nsx.words[0].word if (_nsx is not None and
                        getattr(_nsx, "words", None)) else "") or "").strip()
            _prev_ok = _sent_end((_prv.words[-1].word or "").strip(), _firstw, 0.0)
        if _prev_ok or _nsx is None or not getattr(_nsx, "words", None):
            continue
        _tc = _starts.get(old, 0.0)
        _capd = min(3.5, max(0.0, float(_nsx.duration) - 0.8))
        _ws = _nsx.words
        for _k in range(len(_ws)):
            _rel = float(_ws[_k].end) - _tc
            if _rel > _capd:
                break
            _nxt = (_ws[_k + 1].word if _k + 1 < len(_ws) else "")
            _gap = (float(_ws[_k + 1].start) - float(_ws[_k].end)
                    if _k + 1 < len(_ws) else 0.0)
            if _sent_end(_ws[_k].word, _nxt, _gap):
                _deltas[old] = _rel + 0.10             # small breath after the full stop
                _needs[old] = (_tc + _rel, _ws[_k].word)
                break
    # the per-word times above are PROPORTIONAL ESTIMATES (edge-tts emits no reliable
    # boundaries — see tts.py) and drift grows with delta; snap each pause to the REAL
    # end of its boundary word via whisper word timestamps on the narration track
    if _needs:
        _real = _refine_pause_times(Path(narration.audio), _needs)
        for old, _t_abs in _real.items():
            _nsx = ns_by_idx.get(old)
            _capd = min(3.5, max(0.0, float(_nsx.duration) - 0.8)) if _nsx else 3.5
            _nd = _t_abs - _starts.get(old, 0.0)
            if 0.0 < _nd <= _capd:
                _deltas[old] = _nd

    t_cursor, shift = 0.0, 0.0
    cap_specs = []                                      # breakout captions: final-timeline start
    for seg in list(segments):
        old = seg.index
        dur_here, delta_here, shift_before = 0.0, 0.0, shift
        if old in ins_before:
            p = ins_before[old]
            delta_here = float(_deltas.get(old, 0.0))
            ni = len(new_segs)
            new_segs.append(_CSeg(index=ni, text="", est_duration=p["dur"]))
            _sb = _EScene(index=ni, narration="", keywords=[], visual="breakout")
            _sb.intensity, _sb.role, _sb.shot_type = 1, "evidence", "archival"
            new_scs.append(_sb)
            new_ns.append(_NScene(index=ni, audio=Path(p["audio"]),
                                  duration=p["dur"], words=[]))
            bmap[ni] = Path(p["video"])
            splices.append((t_cursor + delta_here, Path(p["audio"]), p["dur"]))
            # FINAL-timeline start = original insert time + breakout durations spliced BEFORE it
            cap_specs.append({"start": round(t_cursor + delta_here + shift_before, 3),
                              "dur": float(p["dur"]), "audio": str(p["audio"])})
            if delta_here and len(new_ns) >= 2:        # [-1] is the pseudo scene itself
                new_ns[-2].duration = float(new_ns[-2].duration) + delta_here
                log(f"build:   breakout pause moved +{delta_here:.2f}s to the sentence end")
            shift += p["dur"]
            dur_here = p["dur"]
        ni = len(new_segs)
        seg.index = ni
        new_segs.append(seg)
        sc = sc_by_idx.get(old)
        if sc is not None:
            sc.index = ni
            new_scs.append(sc)
        ns = ns_by_idx.get(old)
        if ns is not None:
            ns.index = ni
            _orig_dur = float(ns.duration)
            _pause_t = t_cursor + delta_here
            for w in ns.words:
                # words spoken BEFORE a delayed pause stay with the previous scene's
                # window — they only carry shifts from EARLIER insertions
                if dur_here and w.start < _pause_t - 1e-6:
                    w.start += shift_before
                    w.end += shift_before
                elif shift:
                    w.start += shift
                    w.end += shift
            if delta_here:
                ns.duration = max(0.6, _orig_dur - delta_here)
            new_ns.append(ns)
            t_cursor += _orig_dur                      # original timeline, pre-splice
        idx_map[old] = ni
    full = _splice_audio(Path(narration.audio), splices, work)
    if full is None:
        raise RuntimeError("breakout audio splice failed")
    narration.audio = full
    narration.scenes = new_ns
    try:
        narration.total = float(getattr(narration, "total", 0.0)) + shift
    except Exception:
        pass
    try:
        narration._breakout_caps = cap_specs           # consumed post-assemble for word captions
    except Exception:
        pass
    return new_segs, new_scs, narration, bmap, idx_map


def _split_clip_sequential(clip: Path, lens: list, out_dir: Path, idx: int) -> list:
    """Split a breakout clip into sequential sub-clips matching the planned beat lengths
    (continuous scene playing through — never a replay)."""
    parts, cum = [], 0.0
    for m, L in enumerate(lens):
        dest = out_dir / f"beat_{idx:03d}_{m}_bk.mp4"
        p = subprocess.run([ffmpeg_exe(), "-y", "-ss", f"{cum:.3f}", "-i", str(clip),
                            "-t", f"{max(0.6, L):.3f}", "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                            str(dest)], capture_output=True, timeout=120)
        if p.returncode != 0 or not dest.exists():
            return []
        parts.append(dest)
        cum += max(0.6, L)
    return parts


def _freeze_continuation(frz_clip: Path, dest: Path, duration: float) -> Optional[Path]:
    """Extend a punchline FREEZE across the scene's remaining beats: hold the SAME frozen frame
    (with living grain) — competitor freezes run 4-8s, far longer than one beat."""
    from .ingest import probe
    d = probe(frz_clip).get("duration", 0.0)
    if d <= 0.2:
        return None
    png = dest.with_suffix(".png")
    try:
        p1 = subprocess.run([ffmpeg_exe(), "-y", "-ss", f"{max(0.0, d - 0.08):.3f}",
                             "-i", str(frz_clip), "-frames:v", "1", str(png)],
                            capture_output=True, timeout=60)
        if p1.returncode != 0 or not png.exists():
            return None
        p2 = subprocess.run([ffmpeg_exe(), "-y", "-loop", "1", "-i", str(png),
                             "-t", f"{max(0.5, duration):.3f}",
                             "-vf", "noise=c0s=6:c0f=t,scale=1920:1080,setsar=1,fps=30",
                             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
                            capture_output=True, timeout=180)
        return dest if (p2.returncode == 0 and dest.exists() and dest.stat().st_size > 0) else None
    finally:
        try:
            png.unlink(missing_ok=True)
        except OSError:
            pass


def _click_wav(work: Path) -> Optional[Path]:
    """A short camera-shutter-style tick (synthesized — no asset, no license)."""
    out = work / "shutter_click.wav"
    if out.exists():
        return out
    # mechanical camera-shutter "ka-chak": crisp tick + delayed lower thunk, QUIET
    # (calibrated to ~-22 dB peak — v1 white-noise at -1.9 dB was far too loud, the first
    # bandpass rework at -70 dB was inaudible; gains are measured, not guessed)
    cmd = [ffmpeg_exe(), "-y",
           "-f", "lavfi", "-i", "anoisesrc=d=0.030:c=pink:a=0.9",
           "-f", "lavfi", "-i", "anoisesrc=d=0.040:c=brown:a=0.9",
           "-filter_complex",
           "[0:a]bandpass=f=2400:w=1800,volume=70,afade=t=out:st=0.008:d=0.022[t1];"
           "[1:a]bandpass=f=900:w=700,volume=95,afade=t=out:st=0.012:d=0.028,adelay=60|60[t2];"
           "[t1][t2]amix=inputs=2:normalize=0[out]",
           "-map", "[out]", "-ar", "44100", "-ac", "2", str(out)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=60)
    except Exception:
        return None
    return out if (p.returncode == 0 and out.exists()) else None


def _mix_clicks(narration, marks: list, work: Path, log) -> None:
    """Mix the shutter click into the narration track at the freeze moments (assemble muxes
    narration.audio as the voice track, so this needs no engine change)."""
    marks = [t for t in marks if t and t > 0.05]
    if not marks:
        return
    click = _click_wav(work)
    if not click:
        return
    try:
        ins = ["-i", str(Path(narration.audio).resolve())]
        fparts, mix = [], ["[0:a]"]
        for i, t in enumerate(marks, start=1):
            ins += ["-i", str(click)]
            ms = int(t * 1000)
            fparts.append(f"[{i}:a]adelay={ms}|{ms}[c{i}]")
            mix.append(f"[c{i}]")
        fc = ";".join(fparts) + f";{''.join(mix)}amix=inputs={len(mix)}:normalize=0[out]"
        dest = Path(narration.audio).with_name("narration_clicks.wav")
        p = subprocess.run([ffmpeg_exe(), "-y", *ins, "-filter_complex", fc, "-map", "[out]",
                            "-ar", "44100", str(dest)], capture_output=True, timeout=300)
        if p.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            narration.audio = dest
            log(f"build: {len(marks)} freeze click(s) mixed into narration")
    except Exception as e:                                 # noqa: BLE001
        log(f"build: click mix skipped ({str(e)[:60]})")


def _silent_narration(segments: list[ScriptSegment], work: Path, cfg: ClipConfig):
    """Build a Narration with per-scene silence — used when TTS is unavailable."""
    from vidlore.tts import Narration, NarratedScene, _spread_words
    work.mkdir(parents=True, exist_ok=True)
    scenes, total = [], 0.0
    for seg in segments:
        dur = max(cfg.min_clip_sec, seg.est_duration)
        a = work / f"sil_{seg.index:03d}.wav"
        subprocess.run(
            [ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", f"{dur:.3f}", str(a)], capture_output=True, timeout=60)
        scenes.append(NarratedScene(index=seg.index, audio=a, duration=round(dur, 3),
                                    words=_spread_words(seg.text, total, dur)))
        total += dur
    full = work / "sil_full.wav"
    subprocess.run(
        [ffmpeg_exe(), "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
         "-t", f"{total:.3f}", str(full)], capture_output=True, timeout=120)
    return Narration(scenes=scenes, audio=full)


def _chunked_whisper_words(audio_path: Path, total: float, *, window: float = 90.0,
                           overlap: float = 6.0):
    """faster-whisper word timestamps over LONG audio, transcribed in OVERLAPPING windows.

    A single whisper pass over a 16-min voiceover drifts several seconds (observed: a word spoken
    at ~37s timestamped at ~43s), which then drags captions out of sync. Whisper is accurate WITHIN
    a short clip, so we transcribe ~90s windows and offset each window's word times by its absolute
    start. The first `overlap` seconds of every window after the first are skipped (already covered
    by the previous window's tail) to avoid duplicates. Returns [(word, abs_start, abs_end)]."""
    import math
    import os
    import subprocess
    import tempfile
    from vidlore.align import _model
    model = _model()
    step = max(10.0, window - overlap)
    words: list = []
    last_end = 0.0
    i = 0
    while i * step < total:
        ws = i * step
        wdur = min(window, total - ws)
        if wdur <= 0.2:
            break
        clip = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run([ffmpeg_exe(), "-y", "-ss", f"{ws:.3f}", "-t", f"{wdur:.3f}",
                            "-i", str(audio_path), "-ar", "16000", "-ac", "1", clip],
                           capture_output=True, timeout=180)
            segs, _ = model.transcribe(clip, word_timestamps=True, beam_size=1,
                                       condition_on_previous_text=False, vad_filter=True)
            skip = overlap if i > 0 else 0.0
            for s in segs:
                for w in (s.words or []):
                    if w.start is None or w.end is None:
                        continue
                    ls, le = float(w.start), float(w.end)
                    if not (math.isfinite(ls) and math.isfinite(le)) or le < ls:
                        continue
                    if ls < skip:                       # covered by the previous window's tail
                        continue
                    a, b = ws + ls, ws + le
                    if a < last_end - 1.0:              # keep ~monotonic across window seams
                        continue
                    txt = (w.word or "").strip()
                    if txt:
                        words.append((txt, a, b))
                        last_end = max(last_end, b)
        except Exception:
            pass
        finally:
            try:
                os.remove(clip)
            except OSError:
                pass
        i += 1
    return words


def _align_words_to_hyp(flat: list, hyp: list):
    """Sequence-align script words `flat` to a dense (word, start, end) hypothesis stream — same
    logic as the engine's align_script, but accepts a PRE-BUILT hyp so we can feed the chunked,
    locally-accurate transcription above. Returns a (start, end) per script word, or None."""
    import re as _r
    from difflib import SequenceMatcher
    if not hyp or not flat:
        return None

    def _norm(t):
        return _r.sub(r"[^\w]", "", t.lower(), flags=_r.UNICODE)

    s_norm = [_norm(w) for w in flat]
    h_norm = [_norm(w) for w, _, _ in hyp]
    if not any(s_norm) or not any(h_norm):
        return None
    times: list = [None] * len(flat)
    sm = SequenceMatcher(a=s_norm, b=h_norm, autojunk=False)
    for i1, j1, nn in sm.get_matching_blocks():
        for k in range(nn):
            _, st, en = hyp[j1 + k]
            times[i1 + k] = (st, en)
    n_anchor = sum(1 for t in times if t is not None)
    if n_anchor < max(8, int(0.45 * len(flat))):        # too little matched → unreliable
        return None
    anchors = [(i, t) for i, t in enumerate(times) if t is not None]
    fi, ft = anchors[0]
    for i in range(fi):
        times[i] = (max(0.0, ft[0]), ft[0])
    li, lt = anchors[-1]
    for i in range(li + 1, len(times)):
        times[i] = (lt[1], lt[1])
    for (ia, ta), (ib, tb) in zip(anchors, anchors[1:]):
        if ib - ia <= 1:
            continue
        gap = ib - ia
        t0, t1 = ta[1], tb[0]
        stepv = (t1 - t0) / gap if t1 > t0 else 0.0
        for k in range(1, gap):
            s = t0 + stepv * (k - 1)
            e = t0 + stepv * k
            times[ia + k] = (s, max(s, e))
    return [t if t is not None else (0.0, 0.0) for t in times]


def _synced_narration_from_file(script, audio_path: str, workdir: Path, log=None):
    """Caption-sync for an uploaded voiceover — apply Whisper word-alignment PER-SCENE-TOLERANT.

    The engine's `narrate_from_file` aligns the voice, but guards it with an ALL-OR-NOTHING
    plausibility gate: if even ONE scene's aligned span looks long it discards the WHOLE alignment
    and falls back to a proportional split. On a long (10min+) voiceover that gate trips easily, and
    the proportional timing drifts — captions appear seconds before the voice says them (observed:
    an 11s frozen caption on a 16-min render). Here a lone odd scene is CLAMPED locally instead of
    throwing away the other ~180 exactly-synced scenes, so on-screen text lands on the real voice.

    Returns a Narration with true per-word timings, or None to let the caller use the engine path.
    Raises RuntimeError on an egregious script/voiceover length mismatch (same protection as the
    engine), so the caller falls through to TTS rather than stretching a tiny script over a long VO.
    """
    import math
    import os
    try:
        from vidlore.tts import Narration, NarratedScene, WordTiming, _slice_scene, _wav_duration
        from vidlore.ffmpeg_tool import run as _ffrun
    except Exception:
        return None

    def _log(m):
        if log:
            log(m)

    workdir.mkdir(parents=True, exist_ok=True)
    master = workdir / "voiceover_master.wav"
    _ffrun(["-i", str(Path(audio_path).resolve()), "-ar", "44100", "-ac", "2", str(master)])
    total = float(_wav_duration(master))
    scene_tok = [s.narration.split() for s in script.scenes]
    flat = [w for toks in scene_tok for w in toks]
    n = len(script.scenes)
    if not flat or total <= 1.0:
        return None
    bounds = [0]
    for toks in scene_tok:
        bounds.append(bounds[-1] + len(toks))

    # HARD length-mismatch gate (mirrors the engine): a script that is a tiny SUBSET of a long VO
    # would stretch wildly — refuse so the caller can fall through to TTS. Env override available.
    wc = len(flat)
    if wc >= 40 and os.environ.get("VIDLORE_ALLOW_VO_MISMATCH") != "1":
        exp_lo, exp_hi = wc / 4.0, wc / 1.5
        if total < exp_lo * 0.65 or total > exp_hi * 1.35:
            raise RuntimeError(
                f"[caption-sync] voiceover/script length mismatch: script={wc} words "
                f"(expects {exp_lo:.0f}-{exp_hi:.0f}s at 90-240 wpm), voiceover={total:.1f}s")

    # CHUNKED alignment: a single whisper pass over long audio drifts; transcribe in windows so
    # word times stay locally accurate, then sequence-align the script to that dense stream.
    hyp = _chunked_whisper_words(master, total)
    aligned = _align_words_to_hyp(flat, hyp)
    if not aligned or len(aligned) != len(flat):
        return None                                       # alignment failed → engine fallback
    if any(not (math.isfinite(s) and math.isfinite(e)) for s, e in aligned):
        return None
    if aligned[-1][1] < 0.5 * total:                      # huge un-anchored tail → unreliable
        return None

    wcount = [max(1, bounds[i + 1] - bounds[i]) for i in range(n)]
    wsum = float(sum(wcount)) or 1.0
    scenes, prev_end, clamped = [], 0.0, 0
    for i, sc in enumerate(script.scenes):
        a, b = bounds[i], bounds[i + 1]
        start = prev_end
        if i == n - 1:
            end = total
        elif b > a:
            end = max(aligned[b - 1][1], start + 0.2)
        else:
            end = start + 0.2
        # per-scene balloon clamp: a single mis-aligned scene can't become a multi-second freeze
        # (the engine handles this by rejecting EVERYTHING; we clamp just the offender)
        exp = total * wcount[i] / wsum
        cap = max(12.0, 4.0 * exp)
        if end - start > cap:
            end = start + cap
            clamped += 1
        end = min(max(end, start + 0.2), total)
        w_times = []
        for k in range(a, b):
            ws = min(max(start, aligned[k][0]), end)
            we = min(max(ws, aligned[k][1]), end)
            w_times.append(WordTiming(flat[k], ws, we))
        wav = workdir / f"scene_{sc.index:03d}.wav"
        _slice_scene(master, start, end, total, wav)
        scenes.append(NarratedScene(sc.index, wav, max(0.2, end - start), w_times))
        prev_end = end
    _log(f"build: captions word-synced to voiceover — {len(flat)} words"
         + (f", {clamped} scene(s) clamped" if clamped else ""))
    return Narration(scenes=scenes, audio=master, reused=0)


import re as _re

# theme → engine music-library bucket (assets/music/<bucket>/*.mp3).
# Keys cover the REAL engine theme names (vidlore.themes: crime|history|modern|minimalist|
# standard) — plus topical aliases for callers that pass a niche keyword instead.
_MUSIC_BUCKET = {
    "history": "historical_epic", "crime": "dark_investigation", "modern": "tech_cyber",
    "minimalist": "neutral", "standard": "neutral",
    "true_crime": "dark_investigation", "mystery": "mystery",
    "war": "military_tension", "tech": "tech_cyber", "nature": "emotional_piano",
    "finance": "financial", "survival": "survival_urgency",
}
# words that mark a dramatic / high-impact beat (held shot + stronger grade)
_PEAK_RX = _re.compile(r"\b(death|dies?|died|kill(s|ed|ing)?|murder|destroy|destroyed|poison|"
                       r"betray|blood|war|dead|corpse|scream|purple wedding|execution|fatal|"
                       r"revenge|monster|cruel|cruelty)\b", _re.I)
_STOPE = set("the a an of to in on at for and or but with from into as it its this that is are "
             "was were be it's his her their he she they you your we our us him them not no who "
             "which what when where why how then than too very just only also one new".split())


def _resolve_music(music, theme_name: str, total: float, work: Path):
    """A cinematic background bed. User-supplied path wins; else the engine's theme-aware
    compose_score(); else a deterministic track from the matching music bucket."""
    import os
    if music and Path(music).exists():
        return music
    bucket = _MUSIC_BUCKET.get(theme_name, "historical_epic")
    try:                                              # engine's crossfaded, ducked, arc-aware score
        from vidlore.musiclib import compose_score
        # RETENTION MUSIC ARC: a dynamic emotional curve, not a flat bed — punchy hook intensity,
        # ease back so the early narration breathes, a mid build, then SWELL into the climax
        # (~80%) and soften for the outro. Dynamics keep the ear engaged. (env: MUSIC_ARC=0 → flat)
        _t = max(2.0, float(total))
        if os.environ.get("VIDLORE_CLIPSTUDIO_MUSIC_ARC", "1").strip() not in ("0", "false", "no"):
            cues = [{"t": 0.0, "category": bucket, "intensity": 4},          # hook: strong open
                    {"t": _t * 0.08, "category": bucket, "intensity": 2},    # ease: let the VO land
                    {"t": _t * 0.35, "category": bucket, "intensity": 3},    # build
                    {"t": _t * 0.62, "category": bucket, "intensity": 4},
                    {"t": _t * 0.82, "category": bucket, "intensity": 5},    # climax SWELL
                    {"t": _t * 0.95, "category": bucket, "intensity": 3}]    # outro soften
        else:
            cues = [{"t": 0.0, "category": bucket, "intensity": 3},
                    {"t": max(1.0, total * 0.6), "category": bucket, "intensity": 4}]
        dest = work / "score.wav"
        p = compose_score(cues, max(2.0, float(total)), dest)
        if p and Path(p).exists():
            return str(p)
    except Exception:
        pass
    try:                                              # reliable fallback: pick a track directly
        import vidlore.musiclib as _ml
        base = Path(_ml.__file__).resolve().parent / "assets" / "music"
        tracks = sorted((base / bucket).glob("*.mp3")) or sorted(base.glob("*/*.mp3"))
        if tracks:
            return str(tracks[len(tracks) // 3])
    except Exception:
        pass
    return None


def _pick_emphasis(text: str, keywords: list[str]) -> str:
    """The single spoken word to punch in the captions (must actually appear in the line)."""
    low = text.lower()
    for k in keywords:
        if k and len(k) > 3 and k.lower() not in _STOPE and k.lower() in low:
            return k
    return ""


def _assign_editorial(scenes, segments) -> None:
    """Populate the engine's per-scene editorial signals (role / intensity / emphasis / shot_type)
    so the renderer varies cut-rate, motion and grade like a human editor instead of one flat mode."""
    # SAFE editorial only: emphasis word (caption punch) + shot-type (motion personality, NOT cut
    # count) + a LONG-HOLD reveal on dramatic peaks. We deliberately DO NOT raise intensity or set
    # density roles — those make the engine subdivide a scene into more beats than we have clips for,
    # which restarts the single clip = the loop bug. (One-clip-per-scene + engine sub-cutting are
    # architecturally incompatible; richer intra-scene cutting needs multi-clip scenes — future work.)
    # With multi-clip-per-scene (build_video supplies one DISTINCT clip per engine sub-beat), the
    # engine's editorial subdivision is now SAFE — a denser scene just consumes more distinct clips
    # instead of looping one. So we drive the full editorial system: dramatic peaks HOLD (reveal),
    # lists cut faster (escalation), bookends are hook/resolution. The beat count is recomputed from
    # these same energies/roles in build_video so clip-count always matches the engine's cut-count.
    n = len(scenes)
    shots = ["establishing", "detail", "reaction", "wide", "tracking", "portrait"]
    for i, (sc, seg) in enumerate(zip(scenes, segments)):
        sc.emphasis = _pick_emphasis(seg.text, seg.keywords)
        sc.shot_type = shots[i % len(shots)]
        txt = seg.text
        if i == 0:
            sc.role, sc.intensity = "hook", 3
        elif i >= n - 1:
            sc.role, sc.intensity = "resolution", 2
        elif _PEAK_RX.search(txt):
            sc.role, sc.intensity = "reveal", 4          # dramatic → long held shot
        elif txt.count(",") >= 2 or _re.search(r"\bevery\b", txt, _re.I):
            sc.role, sc.intensity = "escalation", 4      # a list → brisker cutting (more sub-beats)
        else:
            sc.role, sc.intensity = "evidence", 3


def _clip_has_burned_text(clip_path: Path, ocr_engine) -> bool:
    """Does this RAW cut clip carry on-frame text of its OWN (a ripped source's burned-in dialogue
    subtitle or a channel logo)? At the cut stage our narration caption is NOT baked yet, so any
    readable text here is the source's — reliable, unlike the sparse single-keyframe index OCR that
    missed subtitles which flicker on/off with dialogue. Samples a few frames; conservative (needs a
    confident multi-letter read) so clean footage is never falsely flagged."""
    if ocr_engine is None or not Path(clip_path).exists():
        return False
    import os as _os2
    import re as _re2
    ff = ffmpeg_exe()
    for off in (0.3, 1.0, 1.8, 2.6):
        tmp = f"{clip_path}.ocr_{int(off * 10)}.jpg"
        try:
            subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{off:.2f}", "-i", str(clip_path),
                            "-frames:v", "1", "-vf", "scale=854:-1", tmp],
                           capture_output=True, timeout=20)
            if not Path(tmp).exists():
                continue
            res, _el = ocr_engine(tmp)
            confident = [txt for _b, txt, conf in (res or [])
                         if float(conf) >= 0.5 and len(_re2.findall(r"[A-Za-z]", txt)) >= 4]
            if len(confident) >= 2 or any(len(_re2.findall(r"[A-Za-z]", t)) >= 6 for t in confident):
                return True
        except Exception:
            pass
        finally:
            try:
                _os2.remove(tmp)
            except Exception:
                pass
    return False


# A channel intro/outro slate / social-links / CTA card — NOT scene footage. Unlike a source
# DIALOGUE subtitle (which we keep, just suppressing our own caption over it), a branding card
# must be REMOVED from the video entirely. Matches the cards the user flagged (ExploreWesteros
# social-links outro, FilmIsNow promo, "All links in the description / Don't forget to Like").
_BRANDING_RX = re.compile(
    r"\.com\b|\.net\b|www\.|@[A-Za-z0-9_]{3,}|subscrib|like (and|&) sub|hit (the|that) "
    r"(like|bell)|don'?t forget to|links? (are )?in the (description|bio)|follow|"
    r"channel|patreon|watch (more|this one)|filmisnow|movie extras|explorewesteros|"
    r"check out|new video|comment below|"
    # AI-upscaler / generator watermarks — the footage is a synthetic render, not the show:
    r"magnific|topaz|gigapixel|upscayl|upscaled|stable ?diffusion|midjourney|runway|krea", re.I)


def _clip_branding_text(clip_path: Path, ocr_engine) -> bool:
    """STRICTER than _clip_has_burned_text: True only when the clip shows a channel branding /
    social-links / CTA slate (must be REMOVED), not a mere in-scene dialogue subtitle (kept)."""
    if ocr_engine is None or not Path(clip_path).exists():
        return False
    import os as _os2
    ff = ffmpeg_exe()
    for off in (0.2, 0.8, 1.6, 2.6, 3.6):
        tmp = f"{clip_path}.brand_{int(off * 10)}.jpg"
        try:
            subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{off:.2f}", "-i", str(clip_path),
                            "-frames:v", "1", "-vf", "scale=854:-1", tmp], capture_output=True, timeout=20)
            if not Path(tmp).exists():
                continue
            res, _el = ocr_engine(tmp)
            # LOW confidence gate (0.3): branding/CTA/AI-watermark tokens are so specific that a
            # faint read still means a card — a faint tiled "Magnific" AI-upscaler watermark read
            # at ~0.4 confidence used to slip the 0.45 gate and stay in the video.
            joined = " ".join(str(txt) for _b, txt, conf in (res or []) if float(conf) >= 0.30)
            if _BRANDING_RX.search(joined):
                return True
        except Exception:
            pass
        finally:
            try:
                _os2.remove(tmp)
            except Exception:
                pass
    return False


def _freeze_replace(prev_clip: Path, dest: Path, duration: float) -> Optional[Path]:
    """Replace a branding/junk clip with a still freeze of the PREVIOUS clean clip's last frame
    (timing preserved, the card never airs) — reuses the freeze-continuation still renderer."""
    return _freeze_continuation(prev_clip, dest, duration)


def _ass_ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _breakout_caption_ass(caps: list, out_ass: Path, log=None) -> Optional[Path]:
    """Build an ASS overlay that captions the SPOKEN dialogue during each real-audio breakout,
    word-by-word (karaoke fill) — large, bold, gold, distinct from the white narration caption,
    so when the scene's own voice plays the viewer reads exactly what's said. Word timings come
    from whispering each breakout's audio (its own dialogue)."""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    import re as _re
    lines = []
    try:
        m = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        return None
    for cap in caps:
        try:
            segs, _i = m.transcribe(str(cap["audio"]), word_timestamps=True, vad_filter=False)
        except Exception:
            continue
        words = []
        for s in segs:
            for w in (s.words or []):
                wt = (w.word or "").strip()
                if wt:
                    words.append((wt, float(w.start), float(w.end)))
        if not words:
            continue
        base = float(cap["start"])
        dur = float(cap["dur"])
        # group into short lines (<=5 words) that fit one row
        grp, cur = [], []
        for w in words:
            cur.append(w)
            if len(cur) >= 5:
                grp.append(cur); cur = []
        if cur:
            grp.append(cur)
        for line in grp:
            ws = base + line[0][1]
            we = min(base + dur, base + line[-1][2] + 0.10)
            if we <= ws:
                continue
            txt = ""
            for (wt, a, b) in line:
                cs = max(6, int(round((b - a) * 100)))     # karaoke fill centiseconds
                safe = wt.replace("{", "(").replace("}", ")").replace("\\", "")
                txt += f"{{\\kf{cs}}}{safe} "
            lines.append(f"Dialogue: 0,{_ass_ts(ws)},{_ass_ts(we)},BK,,0,0,0,,"
                         f"{{\\fad(120,120)}}{txt.strip()}")
    if not lines:
        return None
    # bold gold karaoke, heavy black outline, lower-centre but lifted; the unsung (secondary)
    # colour is a dim white that fills to bright gold word-by-word as each word is spoken.
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nPlayResX: 1920\nPlayResY: 1080\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding\n"
        # PrimaryColour = filled/sung = bright gold (&H0023C8FF BGR); Secondary = dim white;
        # Outline = black; Alignment 2 = bottom-centre, big MarginV lifts it above the bar.
        "Style: BK,Arial Black,66,&H0023C8FF,&H00C8C8C8,&H00000000,&H64000000,-1,0,1,4,2,2,"
        "120,120,150,1\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    try:
        Path(out_ass).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
        if log:
            log(f"build: breakout captions — {len(lines)} word-by-word line(s)")
        return out_ass
    except Exception:
        return None


def _burn_breakout_captions(video: Path, caps: list, work: Path, log=None) -> bool:
    """Burn the word-by-word breakout captions onto the final video (engine untouched)."""
    if not caps:
        return False
    ass = _breakout_caption_ass(caps, work / "breakout_caps.ass", log)
    if ass is None:
        return False
    out = video.with_name(video.stem + "_bkcap.mp4")
    # libass needs an escaped path inside the filter string
    _ap = str(ass).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    cmd = [ffmpeg_exe(), "-y", "-i", str(video), "-vf", f"ass='{_ap}'",
           "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-movflags", "+faststart", str(out)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=1200)
    except Exception:
        return False
    if p.returncode == 0 and out.exists() and out.stat().st_size > 0:
        Path(out).replace(video)                       # same dir — atomic, no `os` needed
        return True
    if log:
        log(f"build: breakout-caption burn failed ({(p.stderr or b'')[-180:]!r})")
    return False


def build_video(proj: ClipProject, segments: list[ScriptSegment], cfg: ClipConfig, *,
                voice: str = "", captions: bool = True, title: str = "",
                theme_name: str = "history", music: Optional[str] = None,
                voiceover: Optional[str] = None, voice_provider: str = "",
                voice_preset: str = "",
                use_tts: bool = True, progress=None) -> Path:
    """Render the matched clips into a final MP4 via the engine's assemble()."""
    from vidlore.script_gen import Script, Scene
    from vidlore.footage import FootageItem
    from vidlore.themes import theme as get_theme
    from vidlore.assemble import assemble
    from vidlore.config import load_config

    def log(m):
        if progress:
            progress(m)

    eng = load_config()
    import os
    os.environ.setdefault("VIDLORE_MUSIC_VOLUME", "1.15")   # present cinematic bed under the VO
    work = proj.output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    out_path = proj.output_dir / "final.mp4"
    sel_by_idx = {s.segment_index: s for s in proj.selections}

    # 1) engine Script from our segments (narration text per scene)
    scenes = [Scene(index=seg.index, narration=seg.text,
                    keywords=seg.keywords, visual=seg.expected_visual)
              for seg in segments]
    _assign_editorial(scenes, segments)               # role / intensity / emphasis / shot_type
    script = Script(title=title or proj.name, scenes=scenes)

    # 2) narration — user voiceover (forced-aligned) > edge-tts > silent fallback
    narration = None
    if voiceover and Path(voiceover).exists():
        try:
            # caption-sync FIRST: per-scene-tolerant word alignment (the engine's all-or-nothing
            # gate silently drifts long voiceovers). Falls back to the engine path if it can't align.
            narration = _synced_narration_from_file(script, str(Path(voiceover).resolve()), work / "vo", log)
            if narration is not None:
                log(f"build: narration {narration.total:.1f}s (user voiceover, word-synced captions)")
            else:
                from vidlore.tts import narrate_from_file
                narration = narrate_from_file(script, str(Path(voiceover).resolve()), work / "vo")
                log(f"build: narration {narration.total:.1f}s (user voiceover, forced-aligned)")
            if narration is not None:
                # mark word-aligned uploaded VO — the cold-open VO word-cut needs REAL word
                # boundaries (whisper-aligned), never the proportional estimates of a TTS render.
                try:
                    narration._vo_word_aligned = True
                except Exception:
                    pass
        except Exception as e:                            # noqa: BLE001
            log(f"build: voiceover align failed ({str(e)[:90]}) — falling back to TTS")
            narration = None
    if narration is None and use_tts:
        # voice provider: AI neural (chatterbox/kokoro, local, no key) > ElevenLabs (cloud,
        # needs key) > edge (free). Any failure degrades to edge, then to silent — a run
        # never dies for lack of a voice. Reuses the engine's tts stack unchanged.
        _prov = (voice_provider or getattr(cfg, "voice_provider", "") or "edge").strip().lower()
        _preset = (voice_preset or getattr(cfg, "voice_preset", "")
                   or "deep_male_documentary").strip()
        _edge_voice = voice or eng.voice or "en-US-GuyNeural"
        if _prov in ("chatterbox", "kokoro"):
            try:
                from vidlore.tts import narrate_premium
                narration = narrate_premium(script, work / "tts", preset_key=_preset,
                                            backend_name=_prov, cache_dir=work / "tts_cache",
                                            device="auto")
                log(f"build: narration {narration.total:.1f}s (AI voice · {_prov} · {_preset})")
            except Exception as e:                   # noqa: BLE001
                log(f"build: AI voice '{_prov}' failed ({str(e)[:80]}) — edge-tts fallback")
                narration = None
        if narration is None:
            try:
                from vidlore.tts import narrate
                _p = "elevenlabs" if (_prov == "elevenlabs" and eng.elevenlabs_api_key) else "edge"
                _vc = (voice or eng.voice) if _p == "elevenlabs" else _edge_voice
                narration = narrate(script, _vc or "en-US-GuyNeural",
                                    work / "tts", cache_dir=work / "tts_cache", provider=_p,
                                    el_api_key=eng.elevenlabs_api_key, el_model=eng.elevenlabs_model,
                                    fallback_voice=_edge_voice)
                log(f"build: narration {narration.total:.1f}s "
                    f"({'ElevenLabs AI voice' if _p == 'elevenlabs' else 'edge-tts'})")
            except Exception as e:                   # noqa: BLE001
                log(f"build: TTS unavailable ({str(e)[:80]}) — using silent narration")
                narration = None
    if narration is None:
        narration = _silent_narration(segments, work / "silent", cfg)
        log(f"build: narration {narration.total:.1f}s (silent fallback)")

    # 2b) REAL-AUDIO BREAKOUTS — narration pauses, the source scene plays with its OWN voice
    #     (cold-open/evidence moments, 1-3 per video, only on dialogue-locked beats). ALWAYS ON
    #     (the feature is wanted, including on uploaded voiceovers); env still overrides.
    #     CAPTION SYNC (was the bug): a breakout splices a movie clip (its own audio) into the
    #     timeline, pausing the narration. `_apply_breakouts` already shifts every later word-time
    #     by the breakout duration (so word starts stay correct) and splices the audio — the ONLY
    #     break was that `captions._group` had no notion of the silent gap, so the cue spanning it
    #     FROZE for the whole breakout (10-13s) and swallowed the next word ("...evil. This"). Fix:
    #     `_group` hard-cuts a cue at a >=1s gap, and the breakout window is added to the caption-
    #     suppress list below (the breakout's OWN dialogue is captioned by _burn_breakout_captions).
    _breakout_clip: dict = {}
    if os.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUTS", "1").strip() not in ("0", "false", "no"):
        try:
            _bks = _select_breakouts(proj, segments, getattr(narration, "total", 0.0), work, log)
            if _bks:
                _co = next((b for b in _bks if b.get("cold_open")), None)
                _mid = [b for b in _bks if not b.get("cold_open")]
                _bidx = {}
                # mid-video breakouts first (they don't touch the opening beats) — INSERT as usual
                if _mid:
                    segments, scenes, narration, _breakout_clip, _bidx = _apply_breakouts(
                        proj, segments, scenes, narration, _mid, work, log)
                # cold-open: VO word-cut (replace) if the strict hook gate passes; otherwise the
                # proven INSERT path. Default ON, but ONLY for word-aligned UPLOADED voiceover (the
                # narration._vo_word_aligned check below) — never TTS. Kill switch:
                # VIDLORE_CLIPSTUDIO_VO_CUT=0 (also false/no) forces the old insert behaviour.
                if _co:
                    _vocut = os.environ.get("VIDLORE_CLIPSTUDIO_VO_CUT", "1").strip() \
                        not in ("0", "false", "no")
                    _done = False
                    # only on a word-aligned uploaded voiceover (real word boundaries — never TTS
                    # proportional estimates), and protect mid-video breakout scenes from the cut
                    if _vocut and getattr(narration, "_vo_word_aligned", False):
                        _res = _apply_coldopen_vocut(proj, segments, scenes, narration, _co, work, log,
                                                     protect_idx=set(_breakout_clip.keys()))
                        if _res is not None:
                            segments, scenes, narration, _co_bmap, _co_idx, _drop = _res
                            # compose old→final index map + remap mid-video clip scenes through the
                            # cold-open reindex; DROP any key the cold-open removed (never keep stale)
                            _breakout_clip = {_co_idx[k]: v for k, v in _breakout_clip.items()
                                              if k in _co_idx}
                            _breakout_clip.update(_co_bmap)
                            _bidx = ({o: _co_idx[p] for o, p in _bidx.items() if p in _co_idx}
                                     if _bidx else dict(_co_idx))
                            _done = True
                    if not _done:                          # fallback / flag-off → normal insert
                        segments, scenes, narration, _b2, _i2 = _apply_breakouts(
                            proj, segments, scenes, narration, [_co], work, log)
                        _breakout_clip.update(_b2)
                        _bidx = ({o: _i2.get(p, p) for o, p in _bidx.items()} if _bidx else _i2)
                if _bidx:
                    sel_by_idx = {_bidx[s.segment_index]: s for s in proj.selections
                                  if s.segment_index in _bidx}
        except Exception as e:                            # noqa: BLE001
            log(f"build: breakouts skipped ({str(e)[:80]})")

    # 3) MULTI-CLIP-PER-SCENE — supply ONE distinct clip per engine sub-beat so an energetic scene
    #    cuts between DIFFERENT relevant clips instead of looping one (the old loop bug). The beat
    #    count k per scene is computed with the SAME energies/roles passed to assemble() below.
    from vidlore.assemble import plan_beats
    ndur = {sc.index: sc.duration for sc in narration.scenes}
    _durs = [ndur.get(seg.index, seg.est_duration) for seg in segments]
    _eng = [sc.energy for sc in scenes]
    _rol = [sc.role for sc in scenes]
    nbeats = [0] * len(segments)
    try:
        for tup in plan_beats(_durs, target=3.4, bmin=2.4, energies=_eng, roles=_rol):
            j = int(tup[0])
            if 0 <= j < len(nbeats):
                nbeats[j] += 1
    except Exception as e:                               # noqa: BLE001
        log(f"build: plan_beats unavailable ({str(e)[:60]}) — 1 clip/scene")
    nbeats = [max(1, k) for k in nbeats]

    # watermark-CROP: KEEP a watermarked source but punch-in-crop its channel logo off-frame at cut
    # time (vs dropping the whole source) — preserves relevance. One OCR engine, reused for the
    # caption-dodge pass below. (cfg.watermark_mode: "crop" default | "drop".)
    _wm_mode = (getattr(cfg, "watermark_mode", "crop") or "crop").lower()
    _ocr_eng = None
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_eng = RapidOCR()
    except Exception:
        _ocr_eng = None
    wm_corners = (_watermarked_source_corners(proj, _ocr_eng, progress=log)
                  if _wm_mode == "crop" else {})

    footage, beat_clips = [], {}
    total_clips, padded, sm_n = 0, 0, 0
    # GLOBAL anti-repetition: the multi-clip beat_windows (chosen + alternates) are deduped only
    # WITHIN a scene, but the SAME high-scoring (source,moment) window appears in many scenes'
    # alternate lists — so it got re-cut across scenes (the leftover repetition the user saw). Track
    # every (source, ~3s in-point bucket) we cut and always prefer the FRESHEST (never-used, else
    # least-recently-used) window for each beat, so the same moment isn't shown twice where the pool
    # allows it. Also stride Option-B fills FAR apart so same-source fills don't look duplicated.
    import numpy as _np
    import os as _os3
    used_at: dict = {}
    _air_ct: dict = {}        # times each (sid,in) window has AIRED across the whole video
    # SCENE-WALK: per-source playhead — beats from one source ADVANCE chronologically through
    # it (the essayist plays the scene FORWARD), instead of cutting back to recycled windows.
    # This is what kills the "same shot again and again" look structurally.
    _playhead: dict = {}
    freeze_marks: list = []   # (scene_index, t1) — analytical B&W freeze moments (+ click SFX)
    _last_frz_seg = None      # last frozen scene (B&W or color still) — for hold continuation
    gbeat = 0
    recent_emb: list = []         # CLIP embeds of the last few cut clips → avoid back-to-back repeats
    from . import index as _index
    _SHOTS_B: dict = {}
    _EMB_B: dict = {}
    # cosine ≥ this between two shots' CLIP embeds = "shows the same thing" (e.g. another Jon-in-furs
    # close-up). phash was too composition-sensitive (rated near-identical clips as different); the CLIP
    # embedding captures SUBJECT/scene similarity, which is what reads as repetition. Env-tunable.
    from .config import _f as _cfg_f, _i as _cfg_i      # tolerant env parses (bad value → default)
    _SIMT = _cfg_f("VIDLORE_CLIPSTUDIO_VISUAL_SIM", 0.86)

    def _wkey(sid, in_p):
        return (sid, round(float(in_p) / 3.0))

    def _shots_b(sid):
        if sid not in _SHOTS_B:
            try:
                _SHOTS_B[sid] = _index.load_shots(proj, sid)
            except Exception:
                _SHOTS_B[sid] = []
        return _SHOTS_B[sid]

    # USER RULE: footage with readable burned-in text (tweet/quote overlays, lyric cards,
    # hard subtitles) NEVER airs. match.py gates new selections; this build-time check also
    # covers windows from OLDER selections and verify-promoted alternates.
    from .match import _ocr_text_heavy as _txt_heavy
    _TGATE = os.environ.get("VIDLORE_CLIPSTUDIO_TEXT_GATE", "1").strip() not in ("0", "false", "no", "")

    def _shot_has_text(sid, t):
        if not _TGATE:
            return False
        for _sh in _shots_b(sid):
            if _sh.start <= float(t) < _sh.end:
                return _txt_heavy(_sh)
        return False

    def _emb_b(sid):
        if sid not in _EMB_B:
            try:
                _EMB_B[sid] = _index.load_embeds(proj, sid)
            except Exception:
                _EMB_B[sid] = None
        return _EMB_B[sid]

    def _win_embed(sid, in_p):
        em = _emb_b(sid)
        if em is None:
            return None
        for sh in _shots_b(sid):
            if sh.start <= float(in_p) < sh.end:
                r = getattr(sh, "embed_row", -1)
                return em[r] if (0 <= r < len(em)) else None
        return None

    def _near_recent(e):
        # is this look visually like one of the last few cut clips? (back-to-back repeat guard)
        if e is None:
            return False
        for _re in recent_emb[-5:]:
            try:
                if _re is not None and float(_np.dot(e, _re)) >= _SIMT:
                    return True
            except Exception:
                pass
        return False

    # PIXEL-SPACE spaced repeat guard. CLIP embeds judge SEMANTIC similarity — two wide shots of
    # the same dark room pass as "different" while the EYE sees the same picture. The guard that
    # matches perception is a frame hash, with SPACING: a look may return after enough beats
    # (the competitor re-airs shots ~50s apart), but never near-term. Calibrated on the
    # competitor's edit: distant tasteful re-airs OK, near repeats killed.
    _aired_hashes: list = []          # ahash per aired beat, in air order
    _HASH_WIN = 18                    # a look may not return within this many beats (~45s)
    import cv2 as _cv2

    def _frame_hash(src_path, t):
        try:
            cap = _cv2.VideoCapture(str(src_path))
            cap.set(_cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000)
            ok, img = cap.read()
            cap.release()
            if not ok:
                return None
            g = _cv2.cvtColor(_cv2.resize(img, (8, 8)), _cv2.COLOR_BGR2GRAY)
            return (g > g.mean()).flatten()
        except Exception:
            return None

    def _looks_recent(h):
        if h is None:
            return False
        for prev in _aired_hashes[-_HASH_WIN:]:
            try:
                if prev is not None and int(_np.sum(prev != h)) <= 8:
                    return True
            except Exception:
                pass
        return False

    def _next_distinct_shot(sid, after_t):
        # SHOT-AWARE walk: the next detected shot boundary at/after `after_t` whose LOOK is not
        # a near-term repeat — raw seconds-walking inside one static take produced visually
        # identical consecutive beats (a council room doesn't change in 3 seconds; the camera
        # CUT does). Returns a start time, falling back to after_t.
        try:
            src_obj = proj.source(sid)
            sp = src_obj.local_path if src_obj else None
            for sh in _shots_b(sid):
                if sh.end - sh.start < 1.2 or sh.start < after_t - 0.25:
                    continue
                if _TGATE and _txt_heavy(sh):
                    continue                           # burned-in text never airs
                if sp and _looks_recent(_frame_hash(sp, min(sh.start + 1.2, sh.end - 0.2))):
                    continue
                if _TGATE and sp and (
                        _frame_has_burned_text(
                            sp, min(sh.start + 1.0, max(sh.start, sh.end - 0.4)))
                        or _frame_has_burned_text(
                            sp, min(sh.start + 2.2, max(sh.start, sh.end - 0.4)))):
                    continue                           # air-time probe (keyframe can miss)
                return max(after_t, sh.start)
        except Exception:
            pass
        return after_t

    # PER-SCENE VISUAL BUDGET (anti-repeat, desync-safe). A scene can show only as many beats as it has
    # distinct LOOKS — count them by clustering its OWN beat_windows' CLIP-embeds (cosine ≥ _SIMT = 1
    # cluster). If plan_beats wants more sub-beats than that, the extra beats would otherwise REPLAY the
    # scene's first clip — because assemble computes its OWN k from `energies`, and any beat past the
    # clips we supply falls to `item = base` (assemble.py). So we don't just cap our clip count (that
    # DESYNCS us from assemble and caused the "same shot 3× in the first 4s"); we LOWER each
    # over-budget scene's ENERGY and recompute, so BOTH our nbeats AND assemble's k drop together.
    def _scene_distinct(_sel):
        cl = []
        for _w in (getattr(_sel, "beat_windows", []) or []):
            _e = _win_embed(_w[0], _w[1])
            if _e is not None and not any(float(_np.dot(_e, _c)) >= _SIMT for _c in cl):
                cl.append(_e)
        return max(1, len(cl))

    _D = {seg.index: _scene_distinct(sel_by_idx.get(seg.index)) for seg in segments}
    # GLOBAL distinct-look budget: cluster ALL beat_windows' embeds — that's the total number of
    # different-LOOKING shots in the whole project. Asking for more beats than this means SOME look
    # reappears (the spread-out reuse). Cap total beats to it (≥ 1/scene) so each look shows ~once.
    _gclusters: list = []
    for _s in proj.selections:
        for _w in (getattr(_s, "beat_windows", []) or []):
            _e = _win_embed(_w[0], _w[1])
            if _e is not None and not any(float(_np.dot(_e, _c)) >= _SIMT for _c in _gclusters):
                _gclusters.append(_e)
    _gbudget = max(len(segments), len(_gclusters) or len(segments))

    # MAIN-MOMENT HOLD + KEN BURNS — like the competitor, the key moments should LINGER (one held
    # shot for the whole beat) with a slow push-in, instead of fast multi-cutting through everything.
    # A scene is a "main moment" if it's a dramatic peak (_PEAK_RX) OR a high-confidence pick on
    # the exact (anchor) scene. Capped to ~1/3 of scenes so the video doesn't drag.
    _anchor_sids_b = set()
    try:
        for _s in proj.selections:
            if (getattr(proj.source(_s.source_id), "extra", None) or {}).get("anchor_verified"):
                _anchor_sids_b.add(_s.source_id)
    except Exception:
        pass
    _conf = {i: getattr(s, "confidence", 0.0) for i, s in sel_by_idx.items()}
    # RETENTION PACING: a held scene becomes ONE static shot for its whole duration, so a long
    # narration sentence → a 7-10s motionless hold that drags. Hold only DRAMATIC PEAKS plus a
    # SMALL set of the strongest short/medium beats (cap ~1/5), and NEVER hold a long scene
    # (>~6s) — those read as dead air and tank retention. Env-tunable.
    from .config import _f as _cfg_f2
    _hold_frac = _cfg_f2("VIDLORE_CLIPSTUDIO_HOLD_FRACTION", 0.20)
    _hold_max_s = _cfg_f2("VIDLORE_CLIPSTUDIO_HOLD_MAX_SEC", 6.0)
    _hold = set()
    _cand_hold = []
    for _pos, _seg in enumerate(segments):
        _sel = sel_by_idx.get(_seg.index)
        _sdur = float(ndur.get(_seg.index, getattr(_seg, "est_duration", 0.0)) or 0.0)
        _too_long = _sdur > _hold_max_s
        if _PEAK_RX.search(_seg.text) and not _too_long:
            _hold.add(_pos)                                  # dramatic peak always holds (if not long)
        elif _sel and _conf.get(_seg.index, 0.0) >= 0.66 and not _too_long:
            _cand_hold.append((_conf[_seg.index], _pos))     # strong pick → hold candidate
    _cand_hold.sort(reverse=True)
    for _c, _pos in _cand_hold[: max(1, int(len(segments) * _hold_frac))]:
        _hold.add(_pos)
    _hold -= {p for p, _sg in enumerate(segments) if _sg.index in _breakout_clip}
    # HOOK: the first ~2 beats decide whether the viewer stays — they must be DYNAMIC (motion,
    # push-in), never a slow static freeze. Force the opening beats out of the hold set.
    _hook_n = int(_cfg_f("VIDLORE_CLIPSTUDIO_HOOK_BEATS", 2))
    _hold -= set(range(max(0, _hook_n)))
    hold_pos = _hold                                         # frozen set of "held" scene positions
    # freeze STYLE split (competitor): B&W + shutter click only on the BIGGEST punchlines;
    # the rest get the quieter color analytical still with no SFX
    _bw_pos = set(sorted(hold_pos,
                         key=lambda p: -_conf.get(segments[p].index, 0.0))[:3])
    if hold_pos:
        log(f"build: {len(hold_pos)} main-moment hold+zoom scenes "
            f"({len(_bw_pos)} B&W punchline · {len(hold_pos) - len(_bw_pos)} color still)")

    _energies_eff = list(_eng)
    for _p in hold_pos:                                      # held scene = ONE shot for the beat
        _energies_eff[_p] = 1
    _nb = list(nbeats)        # plan_beats failing here must fall back to the stage-3 counts,
    for _it in range(40):     # never to zeros — k=0 would divide-by-zero at slice_dur below
        _nb_try = [0] * len(segments)
        try:
            for _tup in plan_beats(_durs, target=3.4, bmin=2.4, energies=_energies_eff, roles=_rol):
                _j = int(_tup[0])
                if 0 <= _j < len(_nb_try):
                    _nb_try[_j] += 1
        except Exception:
            break
        _nb = [max(1, kk) for kk in _nb_try]
        _over = False
        # (a) no scene exceeds its OWN distinct looks
        for _pos, _seg in enumerate(segments):
            if _nb[_pos] > _D[_seg.index] and _energies_eff[_pos] > 1:
                _energies_eff[_pos] = max(1, _energies_eff[_pos] - 1)
                _over = True
        # (b) total beats ≤ the GLOBAL distinct-look budget — trim the highest-energy scene
        if sum(_nb) > _gbudget:
            _cand = [(_energies_eff[_p], _nb[_p], _p) for _p in range(len(segments))
                     if _energies_eff[_p] > 1 and _nb[_p] > 1]
            if _cand:
                _cand.sort(reverse=True)
                _energies_eff[_cand[0][2]] -= 1
                _over = True
        if not _over:
            break
    # `scene.energy` is a read-only property, so we pass `_energies_eff` straight to assemble() below
    # — that makes assemble's plan_beats give the SAME k as our nbeats. The energy-lowering loop has
    # already minimized _nb as far as the distinct-look budget allows; a long low-energy scene can
    # still plan more beats than its distinct looks, and capping BELOW assemble's k here would make
    # assemble's excess beats replay the scene's FIRST clip (`item = base`). Supplying k clips with
    # the LAST one padded is the controlled repeat — so build keeps assemble's count.
    # FINAL recompute from the FINAL energies: the loop can exit (iteration-40 exhaustion, or an
    # exception after a lowering) with _energies_eff mutated AFTER the last plan — assemble gets
    # _energies_eff, so counts AND per-beat lengths must derive from exactly those values.
    _lens_by_pos: dict = {}
    try:
        _nb_fin = [0] * len(segments)
        for _tup in plan_beats(_durs, target=3.4, bmin=2.4, energies=_energies_eff, roles=_rol):
            _j = int(_tup[0])
            if 0 <= _j < len(_nb_fin):
                _nb_fin[_j] += 1
                _lens_by_pos.setdefault(_j, []).append(float(_tup[1]))
        _nb = [max(1, kk) for kk in _nb_fin]
    except Exception:
        _lens_by_pos = {}
    nbeats = [max(1, kk) for kk in _nb]
    if _energies_eff != _eng:
        log("build: visual budget — lowered scene energies so plan_beats converges with the "
            "distinct-look budget (no first-clip replay)")

    for pos, seg in enumerate(segments):
        sel = sel_by_idx.get(seg.index)
        k = nbeats[pos]
        # cut each beat to its REAL planned length — plan_beats is non-uniform (a reveal's hold
        # beat is ~2.4x the average), so an average-length cut would stream-loop mid-shot on
        # exactly the held peak beat
        _lens = list(_lens_by_pos.get(pos) or [])
        if len(_lens) != k or sum(_lens) <= 0:
            _lens = [ndur.get(seg.index, seg.est_duration) / k] * k
        is_peak = bool(_PEAK_RX.search(seg.text))
        if seg.index in _breakout_clip:
            # breakout scene: the prepared clip plays through (split sequentially if the plan
            # wants multiple beats) — its REAL audio is already spliced into the narration
            _bclip = _breakout_clip[seg.index]
            # register the breakout's look in the air-guard so the SAME shot can't re-air
            # as ordinary footage a few beats later (breakouts bypass the windows walk)
            _aired_hashes.append(_frame_hash(str(_bclip), 1.0))
            clips_for_scene = (_split_clip_sequential(_bclip, _lens, proj.clips_dir, seg.index)
                               if k > 1 else [_bclip]) or [_bclip]
            while len(clips_for_scene) < k:
                clips_for_scene.append(clips_for_scene[-1])
            total_clips += len(clips_for_scene)
            footage.append(FootageItem(index=seg.index, path=clips_for_scene[0], is_video=True))
            beat_clips[seg.index] = clips_for_scene[:k]
            gbeat += k
            continue
        # IMAGE FALLBACK scene: no relevant YouTube clip existed, so a verified exact-scene
        # still was fetched — render it as a Ken-Burns motion still across the beat's k slots
        _img = getattr(sel, "image_path", "") if sel else ""
        if _img and Path(_img).exists():
            clips_for_scene = []
            for m in range(k):
                per_beat = max(cfg.min_clip_sec, _lens[m]) + 0.5
                _kc = proj.clips_dir / f"beat_{seg.index:03d}_{m}_img.mp4"
                _z = 1.08 + 0.02 * (m % 2)            # alternate push-in so multi-beat stills vary
                got = _image_kenburns_clip(_img, _kc, per_beat, zoom_to=_z)
                if got:
                    clips_for_scene.append(Path(got))
            if clips_for_scene:
                while len(clips_for_scene) < k:
                    clips_for_scene.append(clips_for_scene[-1])
                total_clips += len(clips_for_scene)
                footage.append(FootageItem(index=seg.index, path=clips_for_scene[0], is_video=True))
                beat_clips[seg.index] = clips_for_scene[:k]
                gbeat += k
                log(f"build: 🖼 image-still beat {seg.index} ({(sel.image_meta or {}).get('source','web')})")
                continue
        windows_avail = list(getattr(sel, "beat_windows", []) or [])
        clips_for_scene = []
        for m in range(k):
            per_beat = max(cfg.min_clip_sec, _lens[m]) + 0.5
            # held scene: beats after the freeze CONTINUE the held frame — the hold spans the
            # whole scene (one long competitor-style freeze), not just the first beat
            if (pos in hold_pos and m >= 1 and clips_for_scene
                    and _last_frz_seg == seg.index):
                _cont = proj.clips_dir / f"beat_{seg.index:03d}_{m}_frzc.mp4"
                _gotc = _freeze_continuation(clips_for_scene[0], _cont, per_beat)
                if _gotc:
                    clips_for_scene.append(Path(_gotc))
                    gbeat += 1
                    continue
            src, start = None, 0.0
            chosen_w = None
            if windows_avail:
                # a window airs ONCE, ever, AND its picture must not be a NEAR-TERM repeat
                # (pixel-hash spacing — nearby in-points of a static scene are visual clones
                # even when the timestamps differ, and CLIP embeds can't see that)
                _fresh = []
                for w in windows_avail:
                    if _air_ct.get(_wkey(w[0], w[1]), 0) >= 1:
                        continue
                    if _shot_has_text(w[0], float(w[1])):
                        continue                       # burned-in text never airs
                    _wsrc = proj.source(w[0])
                    _h = _frame_hash(_wsrc.local_path, float(w[1]) + 1.2) if (
                        _wsrc and _wsrc.local_path) else None
                    if _looks_recent(_h):
                        continue
                    _fresh.append((w, _h))
                if _fresh:
                    _fresh.sort(key=lambda wh: used_at.get(_wkey(wh[0][0], wh[0][1]), -1))
                    for _wh in _fresh:
                        # AIR-TIME text probe: OCR the frames that will actually air — the
                        # indexed keyframe can miss mid-shot overlay text (tweet cards,
                        # word-by-word animated captions). Probe across the WHOLE cut span.
                        _wsp = proj.source(_wh[0][0])
                        _wsp = _wsp.local_path if _wsp else None
                        _probe_ts, _pt = [], 0.8
                        while _pt < min(per_beat - 0.2, 4.6):
                            _probe_ts.append(_pt)
                            _pt += 1.2
                        if _TGATE and _wsp and any(
                                _frame_has_burned_text(_wsp, float(_wh[0][1]) + _q9)
                                for _q9 in (_probe_ts or [0.8])):
                            continue
                        chosen_w, _ch_hash = _wh
                        break
                    if chosen_w is not None:
                        windows_avail.remove(chosen_w)
                # else: every window aired/looks recent → fall through to the shot-aware walk
                # below, which finds the next DIFFERENT-looking shot instead of a replay
            if chosen_w is not None:
                sid, in_p = chosen_w[0], float(chosen_w[1])
                src = proj.source(sid)
                # lead-in must NOT cross the shot boundary backwards: 0.2s of the PREVIOUS
                # (unrelated) shot flashing at the beat opening is how a cartoon/tweet-card
                # frame aired at a scene start — snap to the containing shot's own start
                _csh = next((s9 for s9 in _shots_b(sid)
                             if s9.start <= in_p < s9.end), None)
                start = max(0.0, in_p - 0.2, (float(_csh.start) if _csh else 0.0))
                used_at[_wkey(sid, in_p)] = gbeat
                _air_ct[_wkey(sid, in_p)] = _air_ct.get(_wkey(sid, in_p), 0) + 1
                _aired_hashes.append(_ch_hash)
                _e = _win_embed(sid, in_p)
                if _e is not None:
                    recent_emb.append(_e)
            elif sel and sel.source_id:
                # fill: SHOT-AWARE walk — continue the chosen source from its playhead, snapped
                # to the next shot whose look is not a near-term repeat
                src = proj.source(sel.source_id)
                sid = sel.source_id
                start = _next_distinct_shot(sid, max(_playhead.get(sid, 0.0),
                                                     sel.in_point - 0.2))
                if src:
                    used_at[_wkey(sid, start)] = gbeat
                    _aired_hashes.append(_frame_hash(src.local_path, start + 1.2)
                                         if src.local_path else None)
            gbeat += 1
            if not (src and src.local_path and Path(src.local_path).exists()):
                continue
            factor = 1.30 if (is_peak and m == k - 1 and k >= 2) else 1.0   # held "land" = slow-mo
            src_need = per_beat / factor
            if src.duration and start + src_need > src.duration:
                start = max(0.0, src.duration - src_need)
            # advance the scene-walk playhead past what this beat consumed (slight overlap
            # headroom so a hard cut never lands mid-frame of the previous beat's tail)
            _playhead[src.id] = max(_playhead.get(src.id, 0.0), start + src_need - 0.25)
            dest = proj.clips_dir / f"beat_{seg.index:03d}_{m}.mp4"
            _cf = _watermark_crop_filter(wm_corners[src.id]) if src.id in wm_corners else ""
            # competitor-style: a push-in on nearly EVERY shot (gentle), stronger on held moments
            _zoom = 1.10 if pos in hold_pos else 1.055
            if pos < _hook_n:                            # hook beats: a stronger, more arresting push-in
                _zoom = 1.12
            _sw = int(getattr(src, "width", 0) or 0)
            rc = _recut_to_duration(src.local_path, start, src_need, src.duration, dest,
                                    crop_filter=_cf, zoom=_zoom, src_w=_sw)
            if rc and factor != 1.0:
                sm = _apply_slow_motion(Path(rc), factor,
                                        proj.clips_dir / f"beat_{seg.index:03d}_{m}_sm.mp4")
                if str(sm) == str(rc):
                    # slow-mo failed — the 1/factor-short cut would visibly loop in its beat;
                    # re-cut the full beat length un-slowed to a FRESH path (re-encoding onto
                    # dest would truncate the valid short cut if this attempt also fails)
                    _full = proj.clips_dir / f"beat_{seg.index:03d}_{m}_full.mp4"
                    if _recut_to_duration(src.local_path, start, per_beat, src.duration, _full,
                                          crop_filter=_cf, zoom=_zoom, src_w=_sw):
                        os.replace(_full, dest)               # same dir — atomic
                    else:
                        try:
                            _full.unlink(missing_ok=True)
                        except OSError:
                            pass
                else:
                    rc = sm
                    sm_n += 1
            if rc and pos in hold_pos and m == 0:
                # analytical punchline: live → FREEZE under the key line. B&W + shutter click
                # for the top punchlines (competitor's signature), quiet color still for the
                # rest of the held analytical moments (their mid-argument observation stills)
                _style = "bw" if pos in _bw_pos else "still"
                _t1 = min(max(1.0, per_beat * 0.4), max(0.8, per_beat - 1.2))
                _frz = proj.clips_dir / f"beat_{seg.index:03d}_{m}_frz.mp4"
                _got = _freeze_punchline(Path(rc), _frz, _t1, per_beat, style=_style)
                if _got:
                    rc = _got
                    _last_frz_seg = seg.index
                    if _style == "bw":
                        freeze_marks.append((seg.index, _t1))
            if rc:
                clips_for_scene.append(Path(rc))
        if not clips_for_scene:
            fb = (sel.clip_path if (sel and sel.clip_path and Path(sel.clip_path).exists())
                  else str(_placeholder_clip(proj, seg.index)))
            clips_for_scene = [Path(fb)]
        while len(clips_for_scene) < k:                   # pad: the ONLY place a repeat can re-enter
            clips_for_scene.append(clips_for_scene[-1])
            padded += 1
        total_clips += len(clips_for_scene)
        footage.append(FootageItem(index=seg.index, path=clips_for_scene[0], is_video=True))
        beat_clips[seg.index] = clips_for_scene[:k]
    log(f"build: {total_clips} beat-clips / {len(segments)} scenes "
        f"(avg {total_clips / max(1, len(segments)):.1f}/scene · {padded} padded · {sm_n} slow-mo)")

    # 3b) CAPTION-SUPPRESSION WINDOWS — when the chosen footage already carries on-screen text (a
    #     ripped clip's burned-in dialogue subtitle, or a channel watermark/logo), our OWN narration
    #     caption must NOT also appear in that window: stacked text reads as messy/unprofessional
    #     (user request). Compute the on-timeline (start,end) of every text-bearing scene and hand
    #     them to assemble(), which then drops the burned caption there — letting the clip's own text
    #     own the frame, with no overlap. Footage stays (relevance preserved); only our caption yields.
    import os as _os
    _suppress_on = _os.environ.get("VIDLORE_CLIPSTUDIO_CAPTION_DODGE", "1").strip() not in ("0", "false", "no", "")

    # 3a) BRANDING-CARD REMOVAL — a channel intro/outro slate, social-links / CTA card, or promo
    #     (ExploreWesteros, FilmIsNow, "links in the description", "don't forget to like") is NOT
    #     scene footage and must NEVER air. We OCR the ACTUAL cut clips (reliable even when the
    #     source was indexed without OCR — the recurring leak) and replace any branding clip with
    #     a still freeze of the previous CLEAN clip (timing preserved). env BRANDING_GATE.
    if _ocr_eng is not None and _os.environ.get("VIDLORE_CLIPSTUDIO_BRANDING_GATE", "1").strip() \
            not in ("0", "false", "no", ""):
        _blens = {segments[_p].index: list(_ls) for _p, _ls in _lens_by_pos.items()}
        _last_clean = None
        _replaced = 0
        for seg in segments:
            if seg.index in _breakout_clip:               # breakouts are dialogue-verified already
                clips0 = beat_clips.get(seg.index) or []
                if clips0:
                    _last_clean = clips0[-1]
                continue
            clips = beat_clips.get(seg.index) or []
            _ls = _blens.get(seg.index) or []
            for m, cp in enumerate(list(clips)):
                if _clip_branding_text(Path(cp), _ocr_eng):
                    _d = (_ls[m] if m < len(_ls) and _ls[m] > 0 else 3.0) + 0.5
                    if _last_clean is not None:
                        _fr = proj.clips_dir / f"beat_{seg.index:03d}_{m}_nobrand.mp4"
                        _got = _freeze_replace(Path(_last_clean), _fr, _d)
                        if _got:
                            clips[m] = Path(_got)
                            _replaced += 1
                            continue
                    # no clean predecessor → drop to a black placeholder rather than air the card
                    clips[m] = _placeholder_clip(proj, seg.index)
                    _replaced += 1
                else:
                    _last_clean = cp
            beat_clips[seg.index] = clips
            # keep the scene's lead FootageItem in sync if its first clip was replaced
            if clips:
                for _fi in footage:
                    if _fi.index == seg.index:
                        _fi.path = clips[0]
                        break
        if _replaced:
            log(f"build: branding-card removal — {_replaced} channel/CTA-slate clip(s) "
                f"freeze-replaced (never airs)")

    suppress_wins = []
    if captions and _suppress_on:
        # reuse the OCR engine initialized for watermark-crop above
        # Map each scene to its EXACT caption time-span straight from the aligned word timings —
        # `narration.scenes[i].words[j].start/end` are absolute on the final timeline, the SAME times
        # the burned caption uses. (Cumulative scene durations drifted to ~2× the real timeline in the
        # render context; word-spans are always bounded by narration.total, so the window can't slip.)
        _total = float(getattr(narration, "total", 0.0) or 0.0)
        span_by_idx = {}
        for ns in getattr(narration, "scenes", []):
            ws = getattr(ns, "words", None) or []
            if ws:
                a = min(float(w.start) for w in ws)
                b = max(float(w.end) for w in ws)
                if _total > 0:
                    a, b = max(0.0, min(a, _total)), max(0.0, min(b, _total))
                span_by_idx[ns.index] = (round(a, 2), round(b, 2))
        # PER-BEAT windows need the engine's REAL beat lengths — plan_beats is deliberately
        # non-uniform (a hook's beat 0 ~2.3x, a reveal's hold ~2.4x the average), so equal
        # slices would leave seconds of stacked text on exactly the dramatic beats. Reuse the
        # final plan computed for the cut stage above (same energies assemble receives).
        _beat_lens: dict = {segments[_p].index: list(_ls) for _p, _ls in _lens_by_pos.items()}
        for seg in segments:
            # OCR the ACTUAL cut clips of this scene — at the cut stage our caption is NOT baked yet,
            # so ANY readable on-frame text is the SOURCE's own burned-in subtitle/logo (reliable,
            # unlike the sparse single-keyframe index OCR which missed subtitles that flicker on/off).
            # One subtitled beat among clean ones must not silence the caption for the whole scene —
            # suppress only that beat's REAL slice (+0.3s slack each side).
            clips = beat_clips.get(seg.index) or []
            span = span_by_idx.get(seg.index)
            if not clips or not span:
                continue
            a, b = span
            lens = _beat_lens.get(seg.index) or []
            if len(lens) != len(clips) or sum(lens) <= 0:
                lens = [(b - a) / len(clips)] * len(clips)    # fallback: equal slices
            else:
                _sc = (b - a) / sum(lens)                     # plan is narration-duration space;
                lens = [x * _sc for x in lens]                # rescale to the caption word-span
            t = a
            for m, cp in enumerate(clips):
                t2 = t + lens[m]
                if _clip_has_burned_text(Path(cp), _ocr_eng):
                    suppress_wins.append((round(max(a, t - 0.3), 2), round(min(b, t2 + 0.3), 2)))
                t = t2
        if suppress_wins:
            suppress_wins.sort()
            merged = [list(suppress_wins[0])]
            for a2, b2 in suppress_wins[1:]:
                if a2 <= merged[-1][1] + 0.05:
                    merged[-1][1] = max(merged[-1][1], b2)
                else:
                    merged.append([a2, b2])
            suppress_wins = [(round(a2, 2), round(b2, 2)) for a2, b2 in merged]
            log(f"build: caption-dodge on {len(suppress_wins)} text-bearing window(s) "
                f"(no caption over burned subtitles/logos)")

    # A real-audio breakout plays the movie's OWN dialogue (captioned separately by
    # _burn_breakout_captions). The main narration caption must not show over that window:
    # `_group`'s gap-cut already stops a cue from spanning it, and marking the window here
    # guarantees no boundary word bleeds in. Independent of the OCR caption-dodge above.
    if captions:
        _bk_added = 0
        for _bc in (getattr(narration, "_breakout_caps", None) or []):
            try:
                _bs = float(_bc["start"])
                suppress_wins.append((round(_bs, 2), round(_bs + float(_bc["dur"]), 2)))
                _bk_added += 1
            except Exception:                                    # noqa: BLE001
                pass
        if _bk_added:
            suppress_wins = sorted(set(suppress_wins))
            log(f"build: caption-suppress over {_bk_added} real-audio breakout window(s) "
                f"(breakout dialogue captioned separately; main caption stays voice-locked)")

    # 4) theme + 5) render
    th = get_theme(theme_name)
    # NEUTRAL footage grade for movie-clip videos: the engine themes tint footage (history =
    # warm/green colorbalance + desat + paper grain), which reads as murky "low quality" next to
    # reference channels that keep the show's ORIGINAL grade. Keep theme captions/cards; replace
    # the grade with a gentle neutral pop and drop the tinting overlays. (Env-overridable.)
    import os as _os9
    if _os9.environ.get("VIDLORE_CLIPSTUDIO_NEUTRAL_GRADE", "1").strip() not in ("0", "false", "no"):
        th = {**th, "grade": "eq=contrast=1.05:saturation=1.04", "overlay_effects": []}
    # CINEMATIC MODE — for a single-scene deep-dive the editor frames the whole video like film:
    # gentle 2.3:1 letterbox bars + captions lifted clear of the bottom bar. On by default for
    # single_scene videos (matches the reference competitor's look); override via env.
    _vtype = (proj.meta.get("analysis", {}) or {}).get("video_type", "")
    _cine_env = os.environ.get("VIDLORE_CLIPSTUDIO_CINEMATIC", "").strip().lower()
    cinematic = (_cine_env in ("1", "true", "yes", "on")) or (_cine_env == "" and _vtype == "single_scene")
    _bar_h = _cfg_i("VIDLORE_CLIPSTUDIO_LETTERBOX_PX", 132)
    if cinematic:
        try:
            cap = dict(th.get("caption", {}))
            cap["margin_v"] = max(int(cap.get("margin_v", 60) or 60), _bar_h + 26)
            th = {**th, "caption": cap}
            log(f"build: cinematic letterbox ON ({_bar_h}px bars · captions lifted to {cap['margin_v']}px · type={_vtype})")
        except Exception as e:                            # noqa: BLE001
            log(f"build: cinematic caption-lift skipped ({str(e)[:60]})")
            cinematic = False
    if freeze_marks:
        _fstarts = {}
        for _ns in getattr(narration, "scenes", []):
            _ws = getattr(_ns, "words", None) or []
            if _ws:
                _fstarts[_ns.index] = min(float(_w.start) for _w in _ws)
        _mix_clicks(narration,
                    [_fstarts.get(_i, 0.0) + _t for _i, _t in freeze_marks if _i in _fstarts],
                    work, log)
    log(f"build: assembling {len(footage)} scenes → {out_path.name}")
    # assemble() internally does len() on these per-scene lists, so pass them explicitly as
    # the engine's own pipeline does (its None defaults are a latent bug that never fires there).
    result = assemble(
        footage, narration, th, work, out_path,
        beat_clips=beat_clips, captions=captions,
        # music: a real cinematic bed. The engine's full sidechain-duck chain only fires when a valid
        # track PATH is passed (the bare call earlier passed None after a revert). We generate/select
        # one here; build_video's caller verifies it's actually audible in the rendered audio.
        music=_resolve_music(music, theme_name, getattr(narration, "total", 0.0) + 1.0, work),
        transitions=True, title=script.title,
        chapters=[(seg.keywords[0] if seg.keywords else "") for seg in segments],
        energies=_energies_eff,
        emphasis=[sc.emphasis for sc in scenes],
        # single-scene deep-dives stay PURE footage (like the reference cut) — suppress the engine's
        # generated info/illustration cards that read as non-footage relevance misses.
        graphics=([("", "", "") for _ in scenes] if cinematic
                  else [(sc.graphic_kind, sc.graphic_text, sc.graphic_body) for sc in scenes]),
        graphic_assets={},
        shot_types=[sc.shot_type for sc in scenes],
        roles=[sc.role for sc in scenes],
        caption_suppress_windows=suppress_wins,
    )
    result = Path(result)
    if cinematic:
        if _apply_cinematic_letterbox(result, _bar_h):
            log(f"build: cinematic bars baked onto {result.name}")
        else:
            log("build: cinematic letterbox post-pass failed — kept flat frame")
    # WORD-BY-WORD BREAKOUT CAPTIONS — during real-audio breakouts the narration caption is
    # silent, so caption the SCENE's own spoken line (large, bold, gold karaoke). Engine
    # untouched; burned as a clipstudio post-pass. (env VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS)
    if captions and os.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1").strip() \
            not in ("0", "false", "no"):
        _caps = list(getattr(narration, "_breakout_caps", None) or [])
        if _caps and _burn_breakout_captions(result, _caps, work, log):
            log(f"build: word-by-word breakout captions burned ({len(_caps)} scene-line(s))")
    log(f"build: done → {result}")
    return result
