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
    clips get a punch-in crop at cut time so the source can be KEPT (relevance) not dropped.

    Two detectors, OR'd: the OCR-keyword one (needs the logo to OCR into a known junk token) and
    the PIXEL static-corner one (match._source_corner_logo) — a stylized/graffiti bug OCRs as
    garbage and slipped the keyword path entirely (observed: 'BLACK TRVLLS' aired on 16 beats)."""
    import os as _os2
    from . import index as _index
    from .match import _source_is_watermarked, _source_corner_logo, _source_edge_logo
    pixel_on = _os2.environ.get("VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE", "1").strip() \
        not in ("0", "false", "no")
    edge_on = _os2.environ.get("VIDLORE_CLIPSTUDIO_EDGE_LOGO_GATE", "1").strip() \
        not in ("0", "false", "no")
    #  A side badge is removed by the WIDTH half of the punch-in crop, so the vertical half of the
    #  code is free; 'b*' keeps the top, which is where faces sit far more often than feet.
    _EDGE2CORNER = {"l": "bl", "r": "br"}
    out: dict = {}
    for src in proj.sources:
        if getattr(src, "status", "") != "ok" or not src.local_path or not Path(src.local_path).exists():
            continue
        try:
            shots = _index.load_shots(proj, src.id)
        except Exception:
            continue
        if ocr_engine is not None and _source_is_watermarked(shots):
            out[src.id] = _detect_logo_corner(src.local_path, ocr_engine)
            if progress:
                progress(f"build: watermark-crop source {src.id} (corner={out[src.id]}, ocr)")
            continue
        if pixel_on:
            corner = _source_corner_logo(shots)
            if corner:
                out[src.id] = corner
                if progress:
                    progress(f"build: watermark-crop source {src.id} (corner={corner}, pixel-static)")
                continue
        if edge_on:
            #  Only after the corner detector has passed: a badge on the side border at mid-height
            #  is invisible to it by construction (see match._source_edge_logo).
            side = _source_edge_logo(shots)
            if side:
                out[src.id] = _EDGE2CORNER[side]
                if progress:
                    progress(f"build: watermark-crop source {src.id} "
                             f"(edge={side} → crop {out[src.id]}, pixel-static)")
    return out


# Detail-enhance chain applied to EVERY cut: the "1080p" uploads of TV scenes are mostly SOFT
# re-encodes (measured Laplacian var 2-18 — fake HD). Light denoise first (so sharpening
# amplifies real edges, not encode noise), then unsharp + contrast-adaptive sharpen.
# Empirically 3.3× Laplacian on a soft S03E03 source with no visible halos.
_CAS = "hqdn3d=1.5:1.5:3:3,unsharp=5:5:0.9:3:3:0.0,cas=0.5"


def _ken_burns_filter(dur: float, src_w: int = 0, zoom_to: float = 1.10) -> str:
    """A slow cinematic push-in (Ken Burns) over the clip — what the competitor does on a held
    key shot. Pre-upscale (lanczos) so the zoom never reveals soft pixels, then zoompan from
    1.0→zoom_to across the clip's frames, output 1920x1080@30.

    TIME-NEUTRALITY (P0 fix): `zoompan` with `d=1` emits one OUTPUT frame per INPUT frame, so the
    caller's `-t dur` OUTPUT limit pulls `dur*30` input frames through it — on a non-30fps source
    that consumes `dur*30/src_fps` SECONDS of source, not `dur` (23.976fps → 1.25× over-consume ran
    the cut straight into a source's Max/WarnerMedia outro slate past the window-QC-cleared end;
    50/60fps → under-consume aired as accidental slow-motion). Prepending `fps=30` resamples the
    seeked input to exactly 30fps FIRST, so 1 output second == 1 source second for every source fps
    and the window-QC-cleared [start, start+dur] range is exactly what airs. On a true 30fps source
    `fps=30` is a no-op (no regression); other rates get frame-rate conversion (24→30 gains 3:2-style
    repeats) — the relevance-safe direction, since the cleared range now airs verbatim."""
    frames = max(1, int(round(max(0.5, dur) * 30)))
    return (f"fps=30,scale=2560:1440:force_original_aspect_ratio=increase:flags=lanczos,crop=2560:1440,"
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


def _breakout_window_admissible(aired_text: str, movie_title: str, *, beat_text: str = "",
                                beat_subject: str = "", promised_quote: str = "",
                                quote_authored: bool = False, relevance_required: bool = True,
                                eng_cfg=None) -> tuple:
    """May this window air as a breakout AT THIS BEAT? Returns (ok, reason, verdicts).

    TWO STAGES, each sampled N times. The split is the whole design:

      STAGE 1 — IDENTIFY, BLIND. Sees the show name and the clip's ASR and NOTHING ELSE. Answers
        intelligible / speaker / scene / in_story. It cannot be led, because it is never told what
        the answer should be. Decided by MAJORITY, because real ASR is partly garbled almost every
        time and one mute vote must not veto a clip the other samples can plainly place — except
        that a single CONFIDENT "narrator" reading still vetoes, since airing a rival essayist is
        the one error with no upside.
      STAGE 2 — BELONG. Sees the narration, the promised line, the ASR *and stage 1's independent
        scene label*, and answers one question: is this line the narration's own evidence here?
        Decided by UNANIMITY. This is the half the owner asked never to compromise, and the half a
        single dissent has repeatedly been right about.

    Why blind. The single-stage version put "NARRATION AT THIS MOMENT" and "LINE THE NARRATION
    PROMISED" in the same prompt as the transcript, and a judge handed unintelligible ASR answered
    from the promise instead of the audio. Measured, job 6a26707939 scene 113: the clip's ASR was
    "We'll use to hear it, cut some black. I will do us" and all three samples replied "Melisandre
    declares Jon Snow is the prince that was promised", belongs=true — which is a paraphrase of the
    beat they had just been shown, not of anything in the audio. Blinding stage 1 removes the leak;
    `intelligible` makes "I cannot tell what this says" an answer instead of a guess.

    Why every earlier gate failed: they judged a PROXY, and each proxy fell to an input that
    satisfies it without satisfying the intent — title tokens ('scene' matched inside "All Scenes"),
    an overlap COUNT (min_ov=2), a stoplist that still let pronouns through, ±2-char prefix fuzz
    ("children's"≈"children"), season strings, face crops. On job benjen_v2 a Benjen Stark essay
    aired FOUR breakouts of a Season-1 Cersei/Ned conversation; beat 112 won its Cersei line on two
    fuzzy tokens. Judging the two artifacts the viewer judges — the words HEARD and the words the
    NARRATION says — leaves no proxy to game.

    A LOCATED QUOTE IS NOT AN EXEMPTION. It used to be one: if find_quote_span placed the beat's
    scripted line in this source, relevance was skipped entirely on the theory that an author had
    already chosen it. Job 6a26707939 disproved the theory twice in one render. Scene 18 promised
    "I have seen the future in the flames." and the locator matched, at phrase ratio 0.8, audio that
    actually says "I don't know, your grace. I CAN'T see the future in the flames" — the negation of
    the promise, the opposite scene, the opposite meaning. All three judges said belongs=false and
    the exemption admitted it anyway; only a downstream coverage floor (0.67 < 0.70) kept it off the
    timeline. Scene 123 was the same shape. A fuzzy locator is not an author, so `quote_authored` is
    now recorded and reported and decides nothing. A genuine anchor still airs on merit: stage 2 is
    told in as many words that the exact promised line DOES belong.

    FAILS CLOSED (no breakout) on: unintelligible audio, a narration verdict, a not-belongs verdict,
    low confidence, an unparseable reply, a missing sample, no LLM. A breakout is optional polish —
    refusing one costs nothing, while airing an unrelated conversation is the most damaging thing
    this feature can do. A CODE fault (NameError/AttributeError/TypeError) is logged loudly and
    re-raised: it must never hide inside the fail-closed catch (cf. `recovery: skipped (NameError)`,
    dead for months).

    Env: VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_SAMPLES (N, default 3 — do NOT run at 1 or 2; measured,
    beat 156 admitted on 2 of 6 single samples and unanimity is what caught it),
    VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_MIN_CONF (default 0.70),
    VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_CHECK=0 drops stage 2 only (identification still runs)."""
    import os as _os_ad
    txt = (aired_text or "").strip()
    if len(txt.split()) < 3:
        return False, "too short to classify", []
    try:
        _n = max(1, min(5, int(_os_ad.environ.get(
            "VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_SAMPLES", "3") or 3)))
    except (TypeError, ValueError):
        _n = 3
    try:
        _min_conf = float(_os_ad.environ.get(
            "VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_MIN_CONF", "0.70") or 0.70)
    except (TypeError, ValueError):
        _min_conf = 0.70

    from . import llm as _llm_d
    if eng_cfg is None:                                    # self-serve: build_video carries no cfg
        try:
            from .config import engine_config as _ec_d
            eng_cfg = _ec_d()
        except Exception:                                  # noqa: BLE001
            eng_cfg = None
    if not _llm_d.has_llm(eng_cfg):
        return False, "no LLM available to judge the window (fail-closed)", []

    _show = movie_title or "a film/TV show"

    # ---------------------------------------------------------------- stage 1: identify, blind
    _sys1 = (
        "You are shown an automatic-speech-recognition transcript of the soundtrack of one short "
        "clip from " + _show + ". Identify it. You are NOT told anything about why the clip was "
        "chosen, and you must not guess at that.\n"
        "ASR is noisy: expect wrong words, fused words and nonsense fragments. PARTIAL garble is "
        "normal and is not a reason to give up — most usable clips contain some.\n"
        "  line  - copy, VERBATIM FROM THE TRANSCRIPT, the longest stretch you can actually read "
        "as real dialogue. Copy the words exactly as they appear even where you believe you know "
        "the real line. Empty string if the transcript is word-salad end to end.\n"
        "  intelligible - true if `line` is a recoverable line of dialogue and you can say who "
        "speaks it. False only if there is nothing readable to quote.\n"
        "  speaker - name the character speaking INSIDE the story; the exact string \"narrator\" "
        "ONLY if these words are a video-essay author narrating ABOUT the show from outside it.\n"
        "  scene   - one short clause naming the scene these words come from, or \"\" if unsure.\n"
        "  in_story - true if this is the show's own dialogue rather than commentary about it.\n"
        "  confidence - how sure you are of the identification.\n"
        'Reply ONLY: {"intelligible":true|false,"line":"...","speaker":"...","scene":"...",'
        '"in_story":true|false,"confidence":0.0-1.0}')
    _usr1 = f"CLIP AUDIO (ASR): {txt[:600]}"

    def _parse(out, keys):
        import json as _json_d
        import re as _re_d
        m = _re_d.search(r"\{.*\}", out or "", _re_d.S)
        if not m:
            return None
        v = _json_d.loads(m.group(0))
        return {k: f(v.get(k)) for k, f in keys.items()}

    def _grounded(line: str) -> bool:
        """Is `line` actually READ OFF the transcript, or invented? Deterministic, not a vote.

        This is the anti-confabulation floor. A sample may only claim it can hear something if it
        can quote it, and the quote has to be in the audio — which no amount of prompt-following
        guarantees on its own. It is also what makes the blinding hold: even a stage-1 sample that
        somehow guessed the essay's subject cannot smuggle it in, because those words are not in
        the transcript to copy."""
        import re as _re_g
        lw = [w for w in _re_g.findall(r"[a-z']{3,}", (line or "").lower())]
        if len(lw) < 3:                                    # too little quoted to verify anything
            return False
        tw = set(_re_g.findall(r"[a-z']{3,}", txt.lower()))
        return sum(1 for w in lw if w in tw) >= max(3, int(0.7 * len(lw)))

    def _ident(_i):
        out = _llm_d.complete(system=_sys1, max_tokens=200,
                              messages=[{"role": "user", "content": _usr1}], eng_cfg=eng_cfg)
        v = _parse(out, {
            # a reply that simply omits `intelligible` must not silently mean "no"
            "intelligible": lambda x: True if x is None else bool(x),
            "line": lambda x: str(x or "").strip(),
            "speaker": lambda x: str(x or "").strip(),
            "scene": lambda x: str(x or "").strip(),
            "in_story": lambda x: bool(x),
            "confidence": lambda x: float(x or 0)})
        if v is not None:
            v["grounded"] = _grounded(v["line"])
        return v

    def _sample(fn):
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=_n) as _ex:                  # N samples cost one call of wall time
            return list(_ex.map(fn, range(_n)))

    try:
        ident = _sample(_ident)
    except (NameError, AttributeError, TypeError):         # a CODE fault must never fail closed
        raise
    except Exception as e:                                 # noqa: BLE001 — provider/parse failures
        return False, f"judge error ({type(e).__name__}) — fail-closed", []

    good = [s for s in ident if s]
    if len(good) != _n:                                    # a missing sample is a NO, not a maybe
        return False, f"only {len(good)}/{_n} verdicts parsed — fail-closed", good

    # MAJORITY, not unanimity, on identification — and only samples that could actually place the
    # clip get a vote on who is speaking. Measured: job 6a26707939's one genuinely correct breakout
    # (beat 53, "Stannis says he will risk everything" over "This is the right time, and I will risk
    # everything ... we march to victory") carries ASR garble in the same window ("we kind of want to
    # supply line some clears"), and one sample in three called the whole thing unintelligible. Real
    # ASR is partly garbled almost every time, so a single mute vote cannot be a veto or the gate
    # only ever says no. RELEVANCE keeps strict unanimity below — that is the half that must not bend.
    _placed = [s for s in good if s["intelligible"] and s["grounded"]]
    if len(_placed) * 2 <= _n:                             # a tie is still a refusal
        _ung = sum(1 for s in good if s["intelligible"] and not s["grounded"])
        return False, (f"the clip's audio is not intelligible enough to identify "
                       f"({len(_placed)}/{_n} quoted a line that is actually in it"
                       + (f", {_ung} quoted words the audio does not contain" if _ung else "")
                       + ")"), good
    _narr = [s for s in _placed
             if (not s["in_story"] or s["speaker"].lower() == "narrator")
             and s["confidence"] >= _min_conf]
    if _narr:                                              # ONE confident essayist call vetoes
        _b = _narr[0]
        return False, (f"a narrator speaking ABOUT the show, not the show — speaker={_b['speaker']!r} "
                       f"in_story={_b['in_story']} conf={_b['confidence']:.2f} "
                       f"({len(_narr)}/{_n} heard commentary)"), good
    _named = [s for s in _placed
              if s["in_story"] and s["speaker"].lower() != "narrator"
              and s["confidence"] >= _min_conf]
    if len(_named) * 2 <= _n:
        _b = _placed[0]
        return False, (f"not identifiable as the show's own dialogue with confidence — "
                       f"speaker={_b['speaker']!r} conf={_b['confidence']:.2f} "
                       f"({len(_named)}/{_n} placed it confidently)"), good

    _lead = {id(s) for s in _named}                        # identity, not ==: samples can be equal
    good = _named + [s for s in good if id(s) not in _lead]  # a confident reading leads the record
    _sp = good[0]["speaker"] or "in-character"
    if not relevance_required:                             # env kill-switch → identification only
        return True, f"{_sp} — the show's own dialogue (identified, {_n}/{_n})", good

    # ---------------------------------------------------------------- stage 2: does it belong here
    _sys2 = (
        "A video essay about " + _show + " is about to cut to a clip and let the clip's OWN audio "
        "play, so the scene speaks for itself. Decide whether that cut is honest at this exact "
        "moment.\n"
        "An independent viewer has already identified the clip without being told anything about "
        "the essay; their reading is given as CLIP IDENTIFIED AS. Trust it over your own reading "
        "of the raw transcript.\n"
        "  belongs - true if the narration and this line are about the SAME SPECIFIC thing: the "
        "line is spoken in the scene the narration describes, OR the narration reports or "
        "paraphrases what this line says, OR it is the exact line the narration promised. A line "
        "that merely MENTIONS the same noun as the narration (a throne, children, a heart, killing, "
        "a walker) does NOT belong. A different scene, different characters or a different point in "
        "the story does NOT belong. A line that STATES THE OPPOSITE of what the narration promised "
        "does NOT belong, however many words it shares. If you would have to explain the connection "
        "to a viewer, it does NOT belong.\n"
        "When unsure, answer belongs=false. An off-topic cut is far worse than no cut.\n"
        'Reply ONLY: {"belongs":true|false,"why":"...","confidence":0.0-1.0}')
    _usr2 = (f"NARRATION AT THIS MOMENT: {(beat_text or '(none)')[:400]}\n"
             + (f"NARRATION SUBJECT: {beat_subject[:120]}\n" if beat_subject else "")
             + (f"LINE THE NARRATION PROMISED: {promised_quote[:200]}\n" if promised_quote else "")
             + f"CLIP IDENTIFIED AS: {good[0]['speaker']} — {good[0]['scene'] or 'unnamed scene'}\n"
             + (f"CLEAREST LINE IN THE CLIP: {good[0]['line'][:200]}\n"
                if good[0].get("line") else "")
             + f"CLIP AUDIO (ASR): {txt[:600]}")

    def _belong(_i):
        out = _llm_d.complete(system=_sys2, max_tokens=160,
                              messages=[{"role": "user", "content": _usr2}], eng_cfg=eng_cfg)
        return _parse(out, {"belongs": lambda x: bool(x),
                            "why": lambda x: str(x or "").strip(),
                            "confidence": lambda x: float(x or 0)})

    try:
        rel = _sample(_belong)
    except (NameError, AttributeError, TypeError):
        raise
    except Exception as e:                                 # noqa: BLE001
        return False, f"judge error ({type(e).__name__}) — fail-closed", good

    for s, r in zip(good, rel):                            # one merged record per sample
        s["belongs"] = bool(r and r["belongs"])
        s["belongs_confidence"] = float(r["confidence"]) if r else 0.0
        s["belongs_why"] = (r or {}).get("why", "")
    if len([r for r in rel if r]) != _n:
        return False, f"only {len([r for r in rel if r])}/{_n} verdicts parsed — fail-closed", good

    votes = [bool(r["belongs"]) and r["confidence"] >= _min_conf for r in rel]
    if all(votes):
        return True, (f"{_sp} — belongs at this beat (judged blind, {_n}/{_n}"
                      + (", quote-anchored" if quote_authored else "") + ")"), good
    _bad = next(s for s, v in zip(good, votes) if not v)
    return False, (f"off-topic for this beat — speaker={_bad['speaker']!r} "
                   f"scene={_bad['scene'][:60]!r} belongs={_bad['belongs']} "
                   f"conf={_bad['belongs_confidence']:.2f} ({sum(votes)}/{_n} would admit)"), good


def _breakout_line_is_dialogue(aired_text: str, movie_title: str, eng_cfg=None) -> tuple:
    """SEMANTIC authority on 'is this the movie speaking, or a YouTuber speaking ABOUT the movie?'

    Every previous defence here ENUMERATED: essay-ish words in the source TITLE (_ESSAYISH_RX) and
    essay-ish phrases in the transcript (_NARRATION_RX). Enumeration cannot win — there are
    unlimited ways to title a clickbait essay and unlimited ways to narrate one. Measured leaks, each
    of which needed a NEW keyword after the fact: 'The Scene Tyrion Exposed The Spy Using Three
    Lies', then "Varys's Absolute Humiliation of Tyrion Lannister" whose aired 'breakout' was the
    narrator saying '...of authority in Westeros, but as he opens the door to the small council'.
    Neither title nor line contained any listed keyword, so both gates passed them.

    So ASK instead of MATCH. A breakout airs a scene's own voice; whether a transcript is a
    character speaking IN the story or an essayist speaking ABOUT it is a semantic judgment, and one
    cheap text call answers it reliably. Only breakout CANDIDATES reach here (a handful per render),
    so the cost is negligible.

    FAILS CLOSED: returns (False, reason) on a narration verdict, low confidence, an unparseable
    reply, no LLM, or any exception. A breakout is optional polish — refusing one costs nothing,
    while airing a rival's voice-over is the single most damaging thing this feature can do.
    Returns (is_dialogue, reason)."""
    txt = (aired_text or "").strip()
    if len(txt.split()) < 3:
        return False, "too short to classify"
    try:
        from . import llm as _llm_d
        if eng_cfg is None:                                # self-serve: build_video carries no cfg
            try:
                from .config import engine_config as _ec_d
                eng_cfg = _ec_d()
            except Exception:
                eng_cfg = None
        if not _llm_d.has_llm(eng_cfg):
            return False, "no LLM available to classify dialogue vs narration (fail-closed)"
        _sys = (
            "You classify a transcript snippet taken from a YouTube video about "
            + (movie_title or "a film/TV show") + ".\n"
            "Decide if the snippet is:\n"
            "  'dialogue'  = words spoken BY A CHARACTER INSIDE the story, to another character "
            "(first/second person, in-world, e.g. 'Tell no one what?', 'I never asked for this').\n"
            "  'narration' = a video-essay narrator/commentator talking ABOUT the story from "
            "outside it (third-person description, analysis, recap, mid-sentence explanatory prose, "
            "e.g. '...of authority in Westeros, but as he opens the door', 'this scene shows us').\n"
            "If it reads like an essayist explaining, summarizing or analysing — even partially, "
            "even mid-sentence — answer 'narration'. When genuinely unsure, answer 'narration'.\n"
            'Reply ONLY: {"kind":"dialogue"|"narration","confidence":0.0-1.0}')
        out = _llm_d.complete(system=_sys, max_tokens=60,
                              messages=[{"role": "user", "content": f"Snippet: {txt[:600]}"}],
                              eng_cfg=eng_cfg)
        import json as _json_d
        import re as _re_d
        m = _re_d.search(r"\{.*\}", out or "", _re_d.S)
        if not m:
            return False, "classifier gave no parseable verdict (fail-closed)"
        v = _json_d.loads(m.group(0))
        kind = str(v.get("kind", "")).strip().lower()
        conf = float(v.get("confidence", 0) or 0)
        if kind == "dialogue" and conf >= 0.6:
            return True, f"in-character dialogue (conf {conf:.2f})"
        return False, f"classified '{kind or 'unknown'}' (conf {conf:.2f}) — not in-character dialogue"
    except Exception as e:                                 # noqa: BLE001
        return False, f"classifier error ({type(e).__name__}) — fail-closed"


def _echoes_own_narration(aired_text: str, script_stream: str, min_run: int = 6) -> int:
    """Longest run of consecutive words the AIRED breakout audio shares with THIS video's OWN
    narration script. A real in-character movie line ('I still remember seeing my father's fleet
    burn in Lannisport') shares ~nothing with our analysis prose; but a same-topic ESSAY source's
    NARRATION echoes our script almost verbatim (measured: an aired 'breakout' window read
    '…somewhere in King's Landing, Cersei is about to hear one of them' — a 9-word run straight
    out of our own script). A long shared run is proof the 'dialogue' is the narrator, not a
    character — a keyword-free, general backstop to _NARRATION_RX. `script_stream` is a pre-built
    space-delimited lowercase word stream of the whole narration (built once by the caller)."""
    import re as _re_en
    aw = _re_en.findall(r"[a-z']+", (aired_text or "").lower())
    if len(aw) < min_run or not script_stream:
        return 0
    best = 0
    n = len(aw)
    for i in range(n):
        # longest run starting at i that appears verbatim in the script stream
        for j in range(n, i + best, -1):     # only try to BEAT the current best
            if j - i < min_run:
                break
            if (" " + " ".join(aw[i:j]) + " ") in script_stream:
                best = j - i
                break
    return best


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
        return _pick_breakout_stop(words, lo, hi), text
    except Exception:
        return max(lo, min(hi, 5.5)), ""


def _pick_breakout_stop(words: list, lo: float, hi: float) -> float:
    """Duration to cut a breakout on a COMPLETE spoken line (pure — unit-testable). `words` is a
    list of (text, start, end) relative to the breakout start. Prefer the LATEST complete stop in
    [lo, hi]; if none, a SHORT complete utterance followed by real silence ends AT ITS OWN STOP
    (floored 2.0s) instead of stretching to `hi` (the '10s window / 1.2s speech' dead-air bug)."""
    if not words:
        return round(max(2.0, min(hi, 5.0)), 3)
    stops = []                                         # candidate end-times (relative to start)
    for i, (txt, _ws, we) in enumerate(words):
        if we > hi + 0.4:
            break
        ends_sentence = txt[-1:] in ".!?…"
        gap_after = (words[i + 1][1] - we) if i + 1 < len(words) else 1.0
        if ends_sentence:
            stops.append(min(hi, we + 0.30))           # let the line breathe before the cut
        elif gap_after >= 0.45:
            stops.append(min(hi, we + min(0.30, gap_after / 2)))
    good = [t for t in stops if lo <= t <= hi]
    if good:
        return round(max(good), 3)                     # latest complete line within the window
    # NO complete stop in [lo, hi]: a SHORT complete utterance ("Anyone can be killed.") followed by
    # real silence must END AT ITS OWN STOP (floored 2.0s), NOT stretch to `hi`. Prefer the EARLIEST
    # sentence-final stop below `lo` whose next word is >= 2s away (a genuine trailing gap).
    early = []
    for i, (txt, _ws, we) in enumerate(words):
        if we > hi:
            break
        if txt[-1:] in ".!?…":
            nxt = words[i + 1][1] if i + 1 < len(words) else (we + 5.0)
            if nxt - we >= 2.0:                        # complete line + a real trailing gap
                early.append(we + 0.30)
    if early:
        return round(max(2.0, min(hi, min(early))), 3)
    last_end = min(hi, words[-1][2] + 0.30)            # continuous dialogue → run to last word
    return round(max(2.0, min(hi, last_end)), 3)


# Quote-anchored breakout window padding. Small: the in-point is the line's own audio onset, so
# these only stop the first consonant / last syllable being clipped.
_BK_LEAD_S = 0.35
_BK_TAIL_S = 0.45
# Breakouts are optional editorial polish, but they are also the only inserts that air a source's
# own picture AND sound at full prominence.  A 360p source enlarged to a 1080p container is still
# 360p footage.  Keep this as a hard native-source contract (not an env-tunable ranking bonus): if
# the local file cannot be probed, or its native height is below 720, it cannot supply a breakout.
_BK_MIN_NATIVE_SHORT_EDGE = 720
_BK_MIN_NATIVE_LONG_EDGE = 1280
# Compatibility name retained for old diagnostics/tests; the gate below uses
# both decoded dimensions, not this height value alone.
_BK_MIN_NATIVE_HEIGHT = _BK_MIN_NATIVE_SHORT_EDGE
# Phrase-alignment tiers for correcting a breakout caption against the known source line — see
# _correct_breakout_words. Above MIN, context alone decides; between FUZZY and MIN a slot must also
# look phonetically like an ASR slip; below FUZZY the audio is speaking a different line entirely.
_BK_CAP_ALIGN_MIN = 0.80
_BK_CAP_ALIGN_FUZZY = 0.60

# Typographic apostrophes/backticks → ASCII before any breakout-line tokenizing. Official HBO
# uploads write "don’t" (U+2019) where fan uploads write "don't"; the tokenizer's [a-z'] class
# split "don’t" into "don"+"t", which broke the cross-source dedup Jaccard for the SAME Olenna
# line from two uploads (job 5462677f95: the confession aired twice, 3.4 min apart).
_BK_APOS_TR = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"})


def _breakout_native_hd_ok(probe_info) -> bool:
    """Pure native-resolution admission rule for a real-audio breakout.

    `probe_info` must be the result of probing the bytes that would actually be cut.  Persisted
    source metadata is deliberately not a fallback: it can be stale after a download recovery or
    can describe the requested format rather than the local file.  Unknown therefore fails closed.
    """
    if not isinstance(probe_info, dict):
        return False
    try:
        width = int(probe_info.get("width") or 0)
        height = int(probe_info.get("height") or 0)
        short, long = sorted((width, height))
    except (TypeError, ValueError, OverflowError):
        return False
    return short >= _BK_MIN_NATIVE_SHORT_EDGE and long >= _BK_MIN_NATIVE_LONG_EDGE


def _probe_breakout_native_dimensions(src_path) -> dict:
    """Probe the local source bytes used by a breakout; return normalized native dimensions."""
    try:
        from .ingest import probe as _probe_bk_hd
        info = _probe_bk_hd(Path(src_path)) or {}
        return {
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
        }
    except Exception:                                  # noqa: BLE001 — unknown probe fails closed
        return {"width": 0, "height": 0}


def _breakout_video_filter(src_w: int, crop_corner: str = "",
                           legibility_vf: str = "") -> str:
    """Build the breakout picture chain: logo crop -> shadow grade -> 1080 normalization."""
    vf_parts = []
    if crop_corner:
        vf_parts.append(_watermark_crop_filter(crop_corner))
    if legibility_vf:
        vf_parts.append(legibility_vf)
    if src_w and src_w < 1280:
        vf_parts.append(_upscale_filter(src_w))
    else:
        vf_parts.append(f"scale=1920:1080:force_original_aspect_ratio=increase,"
                        f"crop=1920:1080,setsar=1,{_CAS}")
    vf_parts.append("fps=30")
    return ",".join(vf_parts)


def _bk_dedup_same_line(cq: str, pq: str) -> bool:
    """Cross-source same-line check for breakout dedup: normalized full-line similarity.
    Substring/Jaccard on the first 10 tokens miss ASR/transcription variants of the SAME spoken
    line ('do you? Well,' vs 'to you? What?'); SequenceMatcher on the full normalized quotes is
    robust to a one-word divergence while genuinely different dialogue stays far below 0.8."""
    if not cq or not pq:
        return False
    from difflib import SequenceMatcher as _SM_bk
    return _SM_bk(None, cq, pq).ratio() >= 0.8


def _extract_breakout(src_path: str, start: float, dur: float, vdest: Path,
                      adest: Path, src_w: int = 0, crop_corner: str = "",
                      min_dur: float = 0.0, quality_meta: Optional[dict] = None) -> Optional[float]:
    """Cut the breakout VIDEO (enhanced, 1080p30, silent) + its AUDIO (2-pass loudnorm to
    narration level, faded) from the source. Skips leading scene silence so the narration
    pause never dangles over a mute shot; `dur` is treated as the MAX — the real length is
    chosen to end on a complete spoken line (3-10s). Returns the exact duration or None.
    `crop_corner`: punch-in crop that drops a channel bug's corner — breakout clips are cut
    directly from the source, so build_video's per-clip watermark crop never touches them.
    `quality_meta`, when supplied, receives the exact legibility grade applied for audit."""
    import json as _json10
    import re as _re10
    try:
        # 1) leading-silence probe — a shot often opens 1-2s before anyone speaks.
        # SKIPPED for a quote-anchored window (min_dur>0): `start` is then the QUOTE's own audio
        # onset, so there is no dead lead to trim and skipping ahead would clip the first word.
        ps = subprocess.run(
            [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(src_path),
             "-t", f"{dur:.3f}", "-vn", "-af", "silencedetect=noise=-30dB:d=0.4",
             "-f", "null", "-"], capture_output=True, timeout=120)
        _txt = (ps.stderr or b"").decode("utf-8", "ignore")
        _m0 = _re10.search(r"silence_start:\s*(-?[\d.]+)", _txt)
        if min_dur <= 0 and _m0 and float(_m0.group(1)) <= 0.15:
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
        # FRAME-LOCK the breakout length. The video is cut with fps=30, so it lands on
        # round(len*30)/30; the audio was cut with -t len, so it lands on exactly len. They differ
        # by up to half a frame EACH — measured 9.480s audio vs 9.467s video — and downstream the
        # audio splice uses its length while the video segment uses its own, so every breakout
        # leaked that mismatch into the timeline (the residual +0.127s the sync invariant caught
        # after the three assemble-side drift fixes). Quantise the target to whole frames up front
        # and cut BOTH to it, so audio == video == a frame-multiple by construction.
        _qfps = 30.0
        _hi = round(_hi * _qfps) / _qfps
        if _hi >= 3.2:
            # `lo` is the floor the dialogue-aware search may not undercut. For a quote-anchored
            # window that floor is the QUOTE'S OWN LENGTH: the whole point is to air the complete
            # intended line, so a search that ends on "a complete spoken line" earlier than the
            # quote's last word would still truncate it. This is how the payoff was lost —
            # breakout #2 ended 5.15s before "…essence of nightshade to help him sleep".
            _lo = max(3.0, min(float(min_dur or 0.0), _hi))
            dur, _bk_text = _dialogue_aware_dur(str(src_path), start, lo=_lo, hi=_hi)
        else:
            dur, _bk_text = max(2.0, _hi), ""
        # the dialogue-aware search returns a fresh length ending on a spoken line; re-lock it to a
        # whole frame so the -t below cuts audio and video to the SAME frame-multiple
        dur = round(float(dur) * _qfps) / _qfps
        # COMPETITOR-VOICEOVER GUARD: a breakout must be the movie's OWN dialogue, never another
        # YouTuber's narration. If the clip's audio reads as commentary/CTA (essay voice-over),
        # reject it — the beat keeps its footage. (env VIDLORE_CLIPSTUDIO_BREAKOUT_VOICE_GUARD)
        import os as _os10
        if _os10.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_VOICE_GUARD", "1").strip() \
                not in ("0", "false", "no") and _is_narration(_bk_text):
            return None
        # Regular selected clips already receive this presentation-only shadow lift in cut.py.
        # Breakouts bypass cut_selection and previously skipped it entirely, leaving the exact same
        # dim source window much darker than a normal beat.  Probe the final, dialogue-adjusted
        # window and apply the same grade before the 1080 normalization.  Any unexpected failure is
        # caught by this function's outer guard and omits the optional breakout.
        from .cut import legibility_filter as _legibility_filter
        _legibility_vf, _legibility_note = _legibility_filter(src_path, start, dur)
        if quality_meta is not None:
            quality_meta["legibility_grade"] = _legibility_note or ""
        pv = subprocess.run(
            [ffmpeg_exe(), "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(src_path),
             "-t", f"{dur:.3f}", "-an",
             "-vf", _breakout_video_filter(src_w, crop_corner, _legibility_vf),
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
        dv = probe(vdest).get("duration", 0.0) or dur
        da = probe(adest).get("duration", 0.0) or dur
        # CONFORM AUDIO TO VIDEO, exactly. The video is authoritative — it is quantised to whole
        # frames and a segment cannot be a fractional frame — but `-t` on a 30fps video and a
        # 44100Hz wav land on different grids (measured 278 vs 279 frames for the same request), so
        # a request-side quantise alone still leaves a 1-frame gap. Trim-or-pad the audio to the
        # video's measured length so the breakout's audio splice and its video segment are the SAME
        # length by construction — the guarantee F5's atomicity needs.
        if abs(da - dv) > 1e-3:
            _tmp = adest.with_suffix(".conform.wav")
            _cf = subprocess.run(
                [ffmpeg_exe(), "-y", "-i", str(adest),
                 "-af", f"apad,atrim=end={dv:.6f},asetpts=N/SR/TB",
                 "-ar", "44100", "-ac", "2", str(_tmp)],
                capture_output=True, timeout=120)
            if _cf.returncode == 0 and _tmp.exists():
                _tmp.replace(adest)
                da = probe(adest).get("duration", 0.0) or dv
        return round(dv, 3)
    except Exception:
        return None


# The narration making an explicit promise to the viewer ("listen for…", "the word I told you to
# listen for"). Such a promise names its payoff, and the edit owes the viewer that line.
_PROMISE_RX = re.compile(
    r"\b(?:listen (?:out )?for|watch for|told you to listen(?: for)?|"
    r"the word i told you|keep an ear (?:out )?for|there it is)\b", re.I)
# Words that are ABOUT the promise, not the promised thing. A promise followed by one of these is a
# TEASE ("listen for SOMETHING") — it names nothing yet, so it yields nothing.
_PROMISE_STOP = {"something", "anything", "word", "words", "name", "line", "moment", "thing",
                 "this", "that", "these", "those", "it's", "its", "the", "one", "listen",
                 "hear", "here", "there", "when", "what", "which", "somewhere"}


def _promised_terms(segments) -> set:
    """Content words the narration explicitly promises the viewer will HEAR.

    Take the LAST promise phrase in a beat (so "the word I told you to listen for" ends after
    "for", not after "told you") and then the FIRST content word following it, in that beat only:

        "That's the word I told you to listen for. Nightshade."   -> {"nightshade"}
        "Before we go in, I want you to listen for something."    -> {}   (a tease; names nothing)

    Not spilling into the next beat is deliberate. The hook teases for ~15 seconds before naming
    anything ("...one word gets spoken. It isn't an insult... It's the name of a poison"), so
    reading ahead just harvests the next sentence's first noun. The PAYOFF beat is where the word is
    finally said, and that is the one worth reserving a breakout slot for."""
    out: set = set()
    for s in (segments or []):
        t = getattr(s, "text", "") or ""
        ms = list(_PROMISE_RX.finditer(t))
        if not ms:
            continue
        for w in re.findall(r"[A-Za-z']{4,}", t[ms[-1].end():]):
            wl = w.lower()
            if wl in _PROMISE_STOP or wl in _BK_FUNC:
                break                      # a tease, or promise-vocabulary → this beat names nothing
            out.add(wl)
            break
    return out


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


# function/near-universal words — a run made only of these ("do you know the", "where did you")
# is a GENERIC interrogative prefix, not a distinctive verbatim quote.
_BK_FUNC = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "but", "is", "are", "was",
            "were", "be", "been", "he", "she", "it", "they", "you", "i", "we", "my", "your",
            "his", "her", "do", "did", "does", "that", "this", "what", "who", "how", "when",
            "where", "why", "so", "if", "for", "with", "at", "as", "by", "me", "him", "them",
            "there", "here", "have", "has", "had", "will", "would", "can", "could", "them"}


def _verbatim_bypass_ok(qw: list, run: int) -> bool:
    """May a verbatim quote-in-footage match OVERRIDE the Face-ID wrong-character gate? Only for a
    STRONG match: >= 4 consecutive matched words, covering >= 70% of the quote, and containing at
    least one DISTINCTIVE content word. A bare 3-4-word generic prefix ('Do you know the', 'Where
    did you learn') is insufficient — that over-matched a DIFFERENT spoken line and stole the gate
    (breakout_017/055 aired the wrong occurrence). Coverage alone separates 'Do you know the [story
    of Harrenhal]' (4/7 = 0.57, rejected) from 'Anyone can be killed' (4/4 = 1.0, kept)."""
    if run < 4 or not qw:
        return False
    matched = qw[:run]
    coverage = run / max(1, len(qw))
    has_content = any(w not in _BK_FUNC and len(w) > 2 for w in matched)
    return coverage >= 0.70 and has_content


# A REVIEW DRAFT EXISTS TO BE SEEN. IT MUST NOT BE WITHHELD OVER ONE IMPERFECT BEAT.
#
# Job 0ca9dc4c2f died at five separate end-of-build gates in a row, each one throwing away a fully
# rendered video. Four were bugs and were fixed. The fifth was not a bug at all — the caption
# readability gate was simply never told that a review draft is allowed to be imperfect, while the
# footage, unverified-exact and black-frame gates all already knew. Counting the rest afterwards
# found the same omission in twelve more content-quality gates.
#
# So the distinction is made once, here, and it is NOT "how bad is it":
#
#   INTEGRITY — the artifact is wrong or unprovable. Wrong owner, broken lineage, a frame that
#   changed under the verifier, a malformed binding. These say "this may not be our footage at
#   all", and they stay fatal in BOTH modes. No draft is worth shipping footage we cannot place.
#
#   CONTENT QUALITY — the artifact is honestly ours and imperfect. A sub-HD still, an unresolved
#   beat, a verdict we could not obtain. These are exactly what a human opens a draft to look at,
#   and in review mode they are recorded and delivered instead of raised.
#
# Production ('block', the default) is unchanged: every one of these is still fail-closed.
#
# The FIRST attempt classified by keywords in the message and mis-filed "verified still is 512x288;
# a real 1280x720 owner is required" as an ownership fault because the word "owner" appeared in it.
# The SECOND classified by the gate's own name PREFIX — better, but still prose. A message gets
# reworded, f-string-interpolated, or concatenated from a sub-gate, and nothing notices: by the time
# this was audited, "caption readability:" was a registered prefix that no gate raised any more.
# Worse, prefixes cannot be ENUMERATED, so a gate that forgot to declare one looked exactly like a
# gate that has none — and four portal renders died in a row, each on a different undeclared member
# of the same family.
#
# So the classification now lives on the exception's `kind`, in release_policy.KIND_CLASS: one
# machine field, one writer, greppable, and enumerable by tests/test_terminal_raise_census.py, which
# refuses to let a new terminal raise skip its declaration. Anything undeclared is integrity.
from .release_policy import review_draft_mode                              # noqa: E402  (re-export)


def content_defect_is_deliverable(exc) -> bool:
    """Can a review draft carry this defect, or must the render still fail?

    Takes the EXCEPTION, not its message. Only a gate that has declared itself a content gate in
    release_policy.KIND_CLASS may be forgiven, and only in review mode. Anything else — including a
    gate nobody has classified yet — stays fatal.
    """
    from .release_policy import deliverable as _deliverable
    return _deliverable(exc)


def _asr_wav_words(wav_path) -> tuple:
    """Re-ASR an EXTRACTED breakout audio clip → (ordered_words, joined_text, speech_seconds).
    This is the GROUND TRUTH of what a breakout actually says (post-loudnorm), unlike the source's
    indexed shot transcript.  Overlapping, EOF-anchored short windows prevent Whisper's plausible
    prefix-only result from hiding the last several seconds of dialogue. () on failure."""
    try:
        from .breakout_asr import transcribe_breakout_words, speech_seconds
        timed, status = transcribe_breakout_words(wav_path, with_status=True)
        if not status.get("complete", False):
            return ([], "", 0.0)      # a window we failed to hear is not ground truth — the
        words = [w[0] for w in timed] # caller skips the breakout rather than judge a partial read
        return (words, " ".join(words), speech_seconds(timed))
    except Exception:
        return ([], "", 0.0)


# Deterministic contraction / possessive canonicalization so whisper's inconsistent tokenization
# ("I've" vs "I have", "don't" vs "do not") aligns BOTH the quote and the aired transcript to one form
# — instead of discarding all apostrophe tokens (which let unrelated audio pass an "I've"/"don't" quote).
_CONTRACTIONS = {
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "would've": "would have", "could've": "could have", "should've": "should have",
    "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "cannot",
    "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
    "couldn't": "could not", "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "mustn't": "must not", "needn't": "need not", "ain't": "is not",
    "it's": "it is", "that's": "that is", "he's": "he is", "she's": "she is",
    "there's": "there is", "here's": "here is", "what's": "what is", "who's": "who is",
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "i'll": "i will", "you'll": "you will", "we'll": "we will", "he'll": "he will",
    "she'll": "she will", "they'll": "they will", "it'll": "it will",
    "i'd": "i would", "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "let's": "let us",
}


def _canon_tokens(words: list) -> list:
    """Canonicalize a token list: lowercase, strip punctuation, EXPAND contractions to their full
    word sequence, and reduce possessives ("tywin's" -> "tywin"). Deterministic; never drops
    meaningful content."""
    out = []
    for w in words:
        w = w.strip(".,!?…\"'").lower()
        if not w:
            continue
        if w in _CONTRACTIONS:
            out.extend(_CONTRACTIONS[w].split())
        elif w.endswith("'s") or w.endswith("’s"):
            out.append(w[:-2])                         # possessive → root noun
        elif "'" in w or "’" in w:
            out.append(w.replace("'", "").replace("’", ""))
        else:
            out.append(w)
    return out


def _ordered_coverage(quote_words: list, aired_words: list) -> float:
    """Fraction of the quote's CONTENT words that appear IN ORDER in the aired transcript (an
    ordered/LCS ratio, not unordered presence). Contractions are canonicalized on BOTH sides first.
    A quote with NO meaningful content words after canonicalization (e.g. "don't", "can't", "I've",
    or a bare possessive) is INSUFFICIENT EVIDENCE and returns 0.0 — it is never treated as a full
    match, so unrelated audio can never pass such a degenerate quote. 0..1."""
    q = _canon_tokens(quote_words)
    a = _canon_tokens(aired_words)
    qc = [w for w in q if w not in _BK_FUNC and len(w) > 2]
    if len(qc) < 2:
        return 0.0                                     # <2 distinctive content words (e.g. "don't",
        # "can't"→"cannot", "I've", a bare possessive, or a generic "do you know" prefix) is
        # INSUFFICIENT EVIDENCE — never a full match, so unrelated audio can never pass it.
    # ASR-NEAR-MISS TOLERANCE. This gate must judge the same way candidates are FOUND
    # (index.find_quote_span), or a line located through garble is then dropped here on that same
    # garble. Measured: "Perhaps some ESSENCE of nightshade" aired as "a MESSENCE of nightshade" —
    # the candidate was generated (fuzzy) then this exact-match gate scored it 0.17 and dropped the
    # video's promised nightshade payoff. A content word counts when it matches exactly OR is a
    # clear phonetic near-miss (SequenceMatcher ≥ 0.8) — never a mere prefix, so distinct words stay
    # distinct.
    from difflib import SequenceMatcher as _SM

    def _same(x, y):
        if x == y:
            return True
        if abs(len(x) - len(y)) > 3:
            return False
        return _SM(None, x, y).ratio() >= 0.8
    # In-order match that SKIPS a missing quote word instead of burning the aired pointer to the
    # end. The old greedy loop advanced `i` to len(a) whenever a quote word was absent, so one
    # substitution killed every later word: "Perhaps SOME essence…" vs aired "Perhaps A messence…"
    # scored 1/6 because 'some'≠'a' consumed the whole remainder, dropping the nightshade payoff.
    # Now a word that cannot be found (from the current position) is simply not counted, and the
    # pointer stays put for the next word.
    i = matched = 0
    for w in qc:
        j = i
        while j < len(a) and not _same(a[j], w):
            j += 1
        if j < len(a):                                 # found in order → consume up to it
            matched += 1
            i = j + 1
        # not found → skip this quote word, keep i for the next (do NOT burn the pointer)
    return matched / len(qc)


def _narr_dup_run(quote_words: list, segments, idx: int, window: int = 2) -> int:
    """Longest run of consecutive words the breakout quote shares with the NARRATION of the beats
    around it ([idx-window .. idx+window]) — i.e. does the narrator SAY the same line right before/
    after the scene plays it (the duplication 'sandwich'). 0 = no meaningful overlap."""
    if not quote_words:
        return 0
    import re as _redup
    best = 0
    for s in segments:
        _si = int(getattr(s, "index", -1))
        if _si == idx or abs(_si - idx) > window:
            continue
        tw = _redup.findall(r"[a-z']+", (getattr(s, "text", "") or "").lower())
        if not tw:
            continue
        tstr = " " + " ".join(tw) + " "
        for j in range(len(quote_words), 3, -1):          # need >= 4 consecutive shared words
            for i in range(len(quote_words) - j + 1):
                if (" " + " ".join(quote_words[i:i + j]) + " ") in tstr:
                    best = max(best, j)
                    break
            if best >= j:
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
    # ESSAY/clickbait STRUCTURE (measured leak: 'The Scene Tyrion Exposed The Spy Using Three
    # Lies' aired 3 narration 'breakouts' — its title has no 'explained/breakdown' keyword but is
    # unmistakably an essay): 'the scene <name> exposed/reveals/…', 'exposed/outsmarted the
    # spy/traitor/plan/council', 'using N lies/tricks/words'. Structural, so raw-clip titles
    # ('Tyrion & Bronn HD S2', 'Varys small council scene') never match.
    r"the scene \w+ (?:exposed|reveals?|proves?|explains?|shows?|breaks?)|"
    r"(?:exposed?|outsmart(?:ed|s)?|outplay(?:ed|s)?|outwit(?:ted|s)?|tricked|fooled|caught) "
    r"(?:the |a )?(?:spy|traitor|mole|leak|liar|plan|scheme|trap|council|small council)|"
    r"using (?:\d+|one|two|three|four|five|six|seven) (?:lies|tricks|words|moves|rules|secrets|clues|steps|questions)|"
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


def _breakout_cursor_probe(video: Path, dur: float) -> str:
    """Screen-recording MOUSE-CURSOR detector for an extracted breakout clip — '' when clean,
    else a reject reason. Job 5462677f95 aired a 12.6s breakout with a white cursor burned at
    frame-centre for its whole duration ('game of thrones Joffrey death' was a screen capture);
    no gate looks at small static mid-frame furniture.

    Deliberately BOUNDED to dodge the candle-sconce precedent (static scene furniture fires
    positionally-consistent-edge detectors): (1) only runs on breakout clips (seconds long, one
    scene, high stakes); (2) requires the SCENE to move (global inter-frame diff — a locked-off
    shot is skipped as undecidable); (3) the blob must be small, bright, near-ACHROMATIC (white/
    grey cursor — a candle sconce/flame is warm, high channel spread) and frozen (per-pixel std
    ~0 across frames while the scene changes); (4) away from corners (corner logos have their own
    detector + crop path). A miss costs one breakout candidate, never footage.
    Env: VIDLORE_CLIPSTUDIO_BREAKOUT_CURSOR_GATE=0 disables."""
    import os as _os_cur
    if _os_cur.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_CURSOR_GATE", "1").strip() \
            in ("0", "false", "no"):
        return ""
    try:
        import numpy as np
        import tempfile as _tf
        from PIL import Image
        frames = []
        # sample past the first 30% — breakout clips open on a cross-dissolve from the previous
        # beat, and a global fade makes every ring look "moving" (a night-sky star survived the
        # ring test exactly this way)
        with _tf.TemporaryDirectory(prefix="bkcur_") as td:
            for k, frac in enumerate((0.30, 0.50, 0.70, 0.90)):
                fp = Path(td) / f"f{k}.png"
                subprocess.run(
                    [ffmpeg_exe(), "-y", "-loglevel", "error",
                     "-ss", f"{max(0.0, frac * float(dur)):.3f}", "-i", str(video),
                     "-frames:v", "1", "-vf", "scale=640:360", str(fp)],
                    capture_output=True, timeout=60)
                if fp.exists():
                    frames.append(np.asarray(Image.open(fp).convert("RGB"),
                                             dtype="float32"))
        if len(frames) < 3:
            return ""
        gray = [f.mean(axis=2) for f in frames]
        # (2) the scene must move — a locked-off shot is undecidable, skip
        moves = [float(np.abs(gray[i + 1] - gray[i]).mean()) for i in range(len(gray) - 1)]
        if max(moves) < 2.0:
            return ""
        stack = np.stack(gray)                            # [n, 360, 640]
        rgb_mean = np.stack(frames).mean(axis=0)          # [360, 640, 3]
        chroma = rgb_mean.max(axis=2) - rgb_mean.min(axis=2)
        mask = ((stack.mean(axis=0) > 190.0)              # bright
                & (stack.std(axis=0) < 3.5)               # frozen while the scene moves
                & (chroma < 18.0))                        # achromatic (white/grey, not flame)
        # (4) corners are another detector's jurisdiction
        mask[:44, :116] = mask[:44, 524:] = mask[316:, :116] = mask[316:, 524:] = False
        if not mask.any():
            return ""
        # small connected component = cursor-sized furniture (4-neighbour flood fill)
        seen = np.zeros_like(mask, dtype=bool)
        ys, xs = np.nonzero(mask)
        for y0, x0 in zip(ys, xs):
            if seen[y0, x0]:
                continue
            comp = [(int(y0), int(x0))]
            seen[y0, x0] = True
            qi = 0
            while qi < len(comp) and len(comp) <= 900:
                cy, cx = comp[qi]
                qi += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < 360 and 0 <= nx < 640 and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        comp.append((ny, nx))
            # measured on the real leak: a 1280-wide screen-recording's arrow shrinks to a
            # ~6x6 white blob (~13px) once the 360p source is decoded at 640x360
            if 8 <= len(comp) <= 500:
                cys = [p[0] for p in comp]
                cxs = [p[1] for p in comp]
                if min(cys) < 4 or min(cxs) < 4 or max(cys) >= 356 or max(cxs) >= 636:
                    continue    # frame-edge sliver (upload border artifact, sits under bars)
                # SOLID-WHITE FROZEN CORE — a rendered cursor saturates (mean 255, std 0);
                # a specular rim-light on a near-static shot peaks ~235 and shimmers
                # (measured: real cursor 11 core px, the Bran-hood highlight FP 0)
                _mean0 = stack.mean(axis=0)
                _std0 = stack.std(axis=0)
                _core = sum(1 for cy, cx in comp
                            if _mean0[cy, cx] > 245.0 and _std0[cy, cx] < 1.5)
                if _core < 3:
                    continue
                if (max(cys) - min(cys)) <= 36 and (max(cxs) - min(cxs)) <= 36:
                    # THE CANDLE-SCONCE/STAR DISCRIMINATOR: a cursor floats OVER moving
                    # content, static scene furniture (a star in a night sky, a lit sconce
                    # on a wall) sits IN a static region. Require the ring around the blob
                    # to actually move across the samples.
                    y0 = max(0, min(cys) - 12); y1 = min(360, max(cys) + 13)
                    x0 = max(0, min(cxs) - 12); x1 = min(640, max(cxs) + 13)
                    ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
                    for cy, cx in comp:
                        ring[cy - y0, cx - x0] = False
                    _rstd = float(stack.std(axis=0)[y0:y1, x0:x1][ring].mean())
                    if _rstd >= 4.0:
                        return (f"cursor-overlay(~{len(comp)}px @ "
                                f"{int(np.mean(cxs))},{int(np.mean(cys))})")
        return ""
    except Exception:                                    # noqa: BLE001
        return ""


def _breakout_window_luma(src_path, start: float, dur: float) -> float:
    """LEGIBILITY score (0–255-ish) across a breakout's air window, on a 128×72 decode.

    Was plain mean luma (YAVG) — which measures BRIGHTNESS, not legibility, and rejected iconic
    candle-lit scenes wholesale. Measured: the S07E03 Olenna–Jaime confession probes YAVG 28–45 and
    Purple-Wedding interiors 48–59 — every one under the 62 floor, every one perfectly readable on
    screen, seven candidates killed in one render. What actually separates "readable dim scene"
    from "dead near-black frame" is CONTRAST: a lit face against shadow spreads the histogram even
    when the mean is low, while true near-black has both a low mean and a collapsed spread.

    Score = max(YAVG, spread) where spread = YHIGH − YLOW (signalstats' 10th→90th percentile
    range). A bright flat frame passes on YAVG exactly as before; a dim-but-lit scene passes on
    spread (confession: spread ≈ 90+); genuine near-black fails both (YAVG ~8, spread ~15).
    Returns **-1.0 if unreadable** (a failed probe is 'unknown', never 'dark')."""
    try:
        import re as _rel9
        p = subprocess.run(
            [ffmpeg_exe(), "-y", "-hide_banner", "-nostats",
             "-ss", f"{max(0.0, float(start)):.3f}", "-i", str(src_path),
             "-t", f"{max(0.5, float(dur)):.3f}", "-an",
             "-vf", "scale=128:72,signalstats,"
                    "metadata=print:key=lavfi.signalstats.YAVG:file=-:direct=1,"
                    "metadata=print:key=lavfi.signalstats.YLOW:file=-:direct=1,"
                    "metadata=print:key=lavfi.signalstats.YHIGH:file=-:direct=1",
             "-f", "null", "-"],
            capture_output=True, timeout=120)
        blob = (p.stdout or b"").decode("utf-8", "ignore") \
            + (p.stderr or b"").decode("utf-8", "ignore")
        yavg = [float(m) for m in _rel9.findall(r"YAVG=([0-9.]+)", blob)]
        ylow = [float(m) for m in _rel9.findall(r"YLOW=([0-9.]+)", blob)]
        yhigh = [float(m) for m in _rel9.findall(r"YHIGH=([0-9.]+)", blob)]
        if not yavg:
            return -1.0
        mean = sum(yavg) / len(yavg)
        spread = 0.0
        if ylow and yhigh and len(ylow) == len(yhigh):
            spread = sum(h - l for h, l in zip(yhigh, ylow)) / len(yhigh)
        return max(mean, spread)
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


def _select_breakouts(proj, segments, total: float, work: Path, log, cfg=None) -> list:
    """Pick the 1-3 most NATURAL breakout moments: beats whose narration QUOTES a line, located
    by searching the line in every source's own ASR — the breakout plays the SHOT where the line
    is actually spoken (independent of which clip the matcher picked for the beat)."""
    import re as _re9
    from . import index as _index
    from .relevance_contract import (
        _confirm_prompted_quote_span_unprompted as _confirm_quote9,
        _quote_confirmation_summary as _quote_confirmation_summary9,
        _quote_requires_exact_contiguous_match as _quote_exact_required9,
        _prompted_quote_candidate_spans as _quote_candidate_spans9,
        QUOTE_RETRIEVAL_MAX_OCCURRENCES_PER_STREAM as _quote_occurrence_bound9,
    )
    if cfg is None:
        from .config import load_clip_config as _load_clip_config9
        cfg = _load_clip_config9()

    # PERSISTENT BREAKOUT AUDIT (work/breakout_audit.json): the [BREAKOUT-*] lines only reach the
    # live progress stream, which portal/CLI runs discard — after the render there was no way to
    # answer "which breakouts were rejected and why". Capture every breakout log line here and
    # persist alongside the summary counters + accepted entries; _apply_breakouts appends the
    # final on-timeline air times.
    _audit_lines: list = []
    _orig_log = log

    def log(m):                                        # shadows the param for this function body
        if "breakout" in str(m).lower():
            _audit_lines.append(str(m))
        _orig_log(m)

    def _rw(t):
        return _re9.findall(r"[a-z']+", (t or "").translate(_BK_APOS_TR).lower())

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
    # BANNED sources (fan-film / AI-recreation) never open a breakout. This is the MOST damaging
    # place a banned upload can land: a breakout hands it the narration's own pause and airs its
    # audio as if it were the show's. The breakout miner reads proj.sources directly, so match's
    # ban does not reach here — apply it explicitly (measured: an "ALTERNATE ENDING" AI recreation
    # aired 9.9s of synthetic Oberyn/Mountain footage under the real "You raped her" line).
    from .match import banned_source_ids as _banned_bk
    _bk_banned = _banned_bk(proj)
    if _bk_banned:
        _n_bk = len(srcs)
        srcs = [s for s in srcs if s.id not in _bk_banned]
        if len(srcs) != _n_bk:
            log(f"build: breakout — {_n_bk - len(srcs)} BANNED source(s) excluded "
                f"(fan-film / AI-recreation never airs its own audio)")
    # CROSS-SHOW purity for breakouts: a breakout airs a FULL scene with its own audio in the most
    # prominent slots (the cold-open included), so a franchise sibling (House of the Dragon in a
    # Game of Thrones video) is the single most jarring possible miss. match.py already drops wrong-
    # show sources from the regular footage pool, but the breakout miner reads proj.sources directly
    # — so it must apply the same gate, or an HotD clip that never enters the pool can still open the
    # video. (Observed: an Aemond/HotD clip aired as the cold-open of a Daenerys render.)
    from .discover import _wrong_installment as _wrong_show9
    _bk_show9 = ((getattr(proj, "meta", None) or {}).get("analysis", {}) or {}).get("movie_title", "") or ""
    if _bk_show9:
        _bk_n0 = len(srcs)
        srcs = [s for s in srcs if not _wrong_show9(_bk_show9, s.title or "")]
        if len(srcs) < _bk_n0:
            log(f"build: breakout wrong-show gate — dropped {_bk_n0 - len(srcs)} "
                f"franchise-sibling source(s) (e.g. House of the Dragon in a {_bk_show9} video)")
    # NATIVE-HD CONTRACT.  Probe the local bytes now, after download/recovery is complete, rather
    # than trusting `SourceVideo.height`: that metadata can remain 1080 while a fallback file on
    # disk is only 360p.  This filter is breakout-only.  Regular matching keeps its relevance-first
    # behavior, while the optional full-picture/audio insert is omitted when no publishable source
    # exists.  Unknown probe data fails closed for the same reason.
    _bk_native_dims = {}
    _bk_hd_srcs = []
    for _s_hd9 in srcs:
        _dim9 = _probe_breakout_native_dimensions(_s_hd9.local_path)
        _bk_native_dims[_s_hd9.id] = _dim9
        if _breakout_native_hd_ok(_dim9):
            _bk_hd_srcs.append(_s_hd9)
            continue
        _w9 = int(_dim9.get("width") or 0)
        _h9 = int(_dim9.get("height") or 0)
        _why9 = (f"native {_w9}x{_h9} < {_BK_MIN_NATIVE_LONG_EDGE}x"
                 f"{_BK_MIN_NATIVE_SHORT_EDGE}" if _w9 and _h9 else
                 "native dimensions unknown")
        log(f"build: breakout HD gate — omitted source {(_s_hd9.title or _s_hd9.id)[:52]!r} "
            f"({_why9}; upscaling cannot create HD detail)")
    _bk_lowres_excluded = len(srcs) - len(_bk_hd_srcs)
    srcs = _bk_hd_srcs
    if _bk_lowres_excluded:
        log(f"build: breakout HD gate — {_bk_lowres_excluded} source(s) excluded; "
            f"{len(srcs)} native-HD source(s) eligible")
    shots_of = {}
    quote_retrieval_words_of = {}
    for s in srcs:
        try:
            shots_of[s.id] = _index.load_shots(proj, s.id)
        except Exception:
            shots_of[s.id] = []
        try:
            _retrieval_ok9, _retrieval_streams9, _retrieval_reason9, _retrieval_complete9 = \
                _index._load_quote_retrieval_streams_result(
                    proj, s, cfg, require_complete=True)
        except Exception:
            _retrieval_ok9, _retrieval_streams9, _retrieval_complete9 = False, [], False
        quote_retrieval_words_of[s.id] = (
            [stream["words"] for stream in _retrieval_streams9]
            if _retrieval_ok9 and _retrieval_complete9 else [])
    ok_audio = {s.id for s in srcs if _breakout_src_ok(s, shots_of.get(s.id) or [])}
    _src_excluded = len(srcs) - len(ok_audio)
    if _src_excluded:
        log(f"build: breakout audio gate — {_src_excluded} narration-style/"
            f"foreign source(s) excluded, {len(ok_audio)} eligible")
    # breakout AUDIT counters — surfaced as ONE summary line per render so v4+ reports exactly how
    # many candidates were found and why each was rejected (commentary / recap-wrong-era /
    # wrong-character / dark / dedup / later-era source).
    _rej = {"later_era_source": 0, "commentary": 0, "recap": 0, "wrong_char": 0,
            "unidentified": 0,
            "dark": 0, "burned_text": 0, "dedup": 0, "spacing": 0, "window_commentary": 0,
            "qa_excluded": 0, "off_topic": 0}
    # per-beat admission verdicts, persisted into breakout_audit.json so a render's breakout
    # decisions can be re-read offline without calling the judge again
    _bk_admit_verdicts: dict = {}
    # lines a PRIOR render's post-render QA proved un-airable (audio masked / black window) —
    # see _persist_breakout_qa_exclusions; the beat gets a different candidate or none
    _qa_excl: set = set()
    try:
        import json as _json_qx
        _qxf = work.parent / "breakout_qa_exclude.json"   # output/, survives the work-dir wipe
        if _qxf.exists():
            _qa_excl = {_norm_bk_line(e.get("line", ""))
                        for e in (_json_qx.loads(_qxf.read_text(encoding="utf-8")) or {})
                        .get("exclude", []) if e.get("line")}
            if _qa_excl:
                log(f"build: breakout selection honors {len(_qa_excl)} post-render-QA "
                    f"exclusion(s) from a prior run")
    except Exception:                                      # noqa: BLE001
        _qa_excl = set()
    from .match import _ocr_text_heavy as _txt9
    from .match import _shot_subtitle_band as _sub9
    from .match import _source_corner_logo as _logo9
    import os as _os9b
    _tgate9 = _os9b.environ.get("VIDLORE_CLIPSTUDIO_TEXT_GATE", "1").strip() \
        not in ("0", "false", "no", "")
    _sbgate9 = _os9b.environ.get("VIDLORE_CLIPSTUDIO_SUBBAND_GATE", "1").strip() \
        not in ("0", "false", "no", "")
    _cngate9 = _os9b.environ.get("VIDLORE_CLIPSTUDIO_CORNER_LOGO_GATE", "1").strip() \
        not in ("0", "false", "no", "")

    _ggate9 = _os9b.environ.get("VIDLORE_CLIPSTUDIO_GRAPHICS_GATE", "1").strip() \
        not in ("0", "false", "no")

    def _texty9(sh):
        """Burned overlay text on a breakout shot — OCR-readable OR a script-agnostic subtitle
        band (Arabic/Turkish burned subs OCR to nothing readable but must never open a breakout)
        — OR a hard-tier designed-graphics shot (news CGI / game-UI parody / illustration; the
        pool gate repeats here per the repeat-every-pool-drop doctrine, persisted-flag only)."""
        if (_tgate9 and _txt9(sh)) or (_sbgate9 and _sub9(sh)):
            return True
        try:
            return _ggate9 and int(getattr(sh, "graphics_flag", -1) or -1) >= 2
        except (TypeError, ValueError):
            return False
    cands = []
    # EXPLICIT candidate ORIGIN, keyed by (seg_idx, src_id, round(start,1)) and set WHERE the
    # candidate is created — 'verbatim_quote' (a scripted quote located in the footage's own ASR) or
    # 'evidence_mined' (a dialogue-rich shot overlapping the beat's narration). This is the candidate's
    # true type; it is NOT the same as `_verbatim_strong` (which is merely Face-ID-bypass eligibility),
    # so downstream coverage thresholds must key on THIS, not on bypass membership.
    _cand_origin = {}
    # Prompted ASR is retrieval only.  A quote candidate receives NONE of the verbatim privileges
    # below until a second, narrow decoder run on physically extracted audio (with no prompt or
    # hotwords) confirms that exact hit.  Keep the confirmed span candidate-specific: a beat may
    # have several authored/anchor lines, and recomputing ``seg.quote`` at extraction used to anchor
    # a winning correction/anchor candidate to a different line.
    _quote_confirmation_by_candidate: dict = {}
    _quote_confirmation_attempts: list = []

    def _quote_candidate_key9(seg_idx, src, shot, quote):
        return (int(seg_idx), str(src.id), round(float(shot.start), 1),
                " ".join(_rw(quote)))

    def _confirm_quote_hit9(src, quote, prompted_span, *, exact_required):
        try:
            evidence = _confirm_quote9(
                proj, src, quote, prompted_span, cfg,
                exact_contiguous_required=bool(exact_required))
        except Exception as exc:                         # optional breakout: uncertainty rejects
            evidence = {"status": "inconclusive",
                        "reason": f"confirmation_error:{type(exc).__name__}"}
        summary = _quote_confirmation_summary9(evidence)
        _quote_confirmation_attempts.append({
            "source_id": str(getattr(src, "id", "") or ""),
            "quote": str(quote or "")[:240],
            "prompted_span": [round(float(prompted_span[0]), 3),
                               round(float(prompted_span[1]), 3),
                               round(float(prompted_span[2]), 3)],
            "confirmation": summary,
        })
        return evidence, summary

    def _quote_confirmation_counts9():
        return {
            "attempted": len(_quote_confirmation_attempts),
            "confirmed": sum(
                1 for row in _quote_confirmation_attempts
                if (row.get("confirmation") or {}).get("status") == "confirmed"),
            "rejected": sum(
                1 for row in _quote_confirmation_attempts
                if (row.get("confirmation") or {}).get("status") == "rejected"),
            "inconclusive": sum(
                1 for row in _quote_confirmation_attempts
                if (row.get("confirmation") or {}).get("status") == "inconclusive"),
        }
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
    # SAME LINE, TWO SPELLINGS, TWICE THE DECODING.
    #
    # This search reads the quote only through its normalised token run and through the
    # short-quote exactness rule, which counts those same tokens — so two scripts writing
    # "I demand a trial by combat!" and "I demand a trial by combat." pose one question and get one
    # answer. Measured on a real 180-scene job: 757 unprompted decodes over 19 authored quotes that
    # are only 17 distinct lines, and the two duplicated pairs alone cost 160 of those decodes
    # (21.1%). This memo removes exactly that repetition and nothing else.
    #
    # It is emphatically NOT an early exit. The loop below keeps the BEST-scoring confirmed
    # candidate, so stopping at the first confirmation would change which window airs — that is a
    # quality change wearing an optimisation's clothes, and it is not made here.
    _quote_search_memo: dict = {}
    _quote_search_stats = {"hit": 0, "miss": 0}

    def _locate_quote9(_q):
        """Retrieve then independently confirm a scripted quote in downloaded footage.

        Returns ``(score, source, shot, run, confirmed_span, confirmation_summary)`` or None.
        The persisted, vocabulary-prompted word stream only proposes locations; it cannot prove
        its own prompt-derived phrase.  Every returned candidate has therefore survived the
        separate unprompted narrow-audio decoder.
        """
        qw = _rw(_q)[:8]
        _exact_required = _quote_exact_required9(_q, index_module=_index)
        _memo_key = (tuple(qw), bool(_exact_required))
        if _memo_key in _quote_search_memo:
            _quote_search_stats["hit"] += 1
            return _quote_search_memo[_memo_key]
        _quote_search_stats["miss"] += 1
        best = None
        for s in srcs:
            if s.id not in ok_audio:
                continue
            # WORD-STREAM PASS first. Per-shot matching below can only see a line that lies wholly
            # inside ONE shot, because _assign_transcript bins ASR words into shots by midpoint. Two
            # measured losses in the same clip:
            #   "Any man who must | say I am the king is no true king."  (split at the 126.58 cut)
            #   "Maester. Perhaps | a messence of nightshade to help him | sleep."  (4 shots + garble)
            # The second produced NO candidate at all — the video's promised payoff, never even
            # considered. The word stream is continuous, so cuts and garble stop mattering.
            try:
                _general_words9 = _index.load_words(proj, s.id)
            except Exception:
                _general_words9 = []
            _seen_spans9 = set()
            _confirmed_stream_hit9 = False
            _candidate_streams9 = [("general_names_only_asr", _general_words9)]
            _candidate_streams9.extend(
                (f"authored_prompt_retrieval_chunk_{chunk_index}", chunk_words)
                for chunk_index, chunk_words in enumerate(
                    quote_retrieval_words_of.get(s.id) or []))
            for _stream_kind9, _words9 in _candidate_streams9:
                try:
                    _spans9 = _quote_candidate_spans9(
                        _words9, _q, exact_contiguous_required=_exact_required,
                        index_module=_index)
                except Exception:
                    _spans9 = []
                # Optional breakouts do not need to exhaust an unbounded hallucination storm. The
                # strict whole-pool publication contract separately marks a truncated source
                # indeterminate; here the safe result is simply no breakout from untried hints.
                for _span in _spans9[:_quote_occurrence_bound9]:
                    _span_key9 = tuple(round(float(value), 3) for value in _span)
                    if _span_key9 in _seen_spans9:
                        continue
                    _seen_spans9.add(_span_key9)
                    _confirmation, _summary = _confirm_quote_hit9(
                        s, _q, _span, exact_required=_exact_required)
                    _summary = dict(_summary or {})
                    _summary["retrieval_stream"] = _stream_kind9
                    _confirmed_span = (_confirmation.get("confirmed_span")
                                       if _confirmation.get("status") == "confirmed" else None)
                    if _confirmed_span:
                        _cqs, _cqe, _cratio = (float(_confirmed_span[0]),
                                                float(_confirmed_span[1]),
                                                float(_confirmed_span[2]))
                        _sh = next((x for x in shots_of.get(s.id, [])
                                    if float(x.start) <= _cqs < float(x.end)), None)
                    else:
                        _sh = None
                    if _sh is not None and not _texty9(_sh):
                        _run = max(3, int(round(_cratio * len(qw))))
                        score = _run + 3 + (
                            2 if (getattr(s, "extra", None) or {}).get("anchor_verified") else 0)
                        if best is None or score > best[0]:
                            best = (score, s, _sh, _run,
                                    (_cqs, _cqe, _cratio), _summary)
                        _confirmed_stream_hit9 = True
            if _confirmed_stream_hit9:
                continue                           # continuous word stream is better evidence
            for sh in shots_of.get(s.id, []):
                if _texty9(sh):
                    continue                           # burned-in text never airs
                run = _quote_run_in(qw, _rw(getattr(sh, "transcript", "")))
                if run >= 3:
                    # Shot transcripts are derived from the same prompted word stream, so even this
                    # fallback is only another retrieval hint.  It may become a quote candidate only
                    # if the independent decoder finds the full phrase near this shot.
                    _hint = (float(sh.start), float(sh.end),
                             min(1.0, float(run) / max(1, len(qw))))
                    _confirmation, _summary = _confirm_quote_hit9(
                        s, _q, _hint, exact_required=_exact_required)
                    _confirmed_span = (_confirmation.get("confirmed_span")
                                       if _confirmation.get("status") == "confirmed" else None)
                    if not _confirmed_span:
                        continue
                    _cqs, _cqe, _cratio = (float(_confirmed_span[0]),
                                            float(_confirmed_span[1]),
                                            float(_confirmed_span[2]))
                    _confirmed_sh = next(
                        (x for x in shots_of.get(s.id, [])
                         if float(x.start) <= _cqs < float(x.end)), None)
                    if _confirmed_sh is None or _texty9(_confirmed_sh):
                        continue
                    _confirmed_run = max(3, int(round(_cratio * len(qw))))
                    # +3: a verbatim script quote located in the footage's own ASR is the
                    # strongest naturalness signal — it must outrank mined evidence lines
                    score = _confirmed_run + 3 + (
                        2 if (getattr(s, "extra", None) or {}).get("anchor_verified") else 0)
                    if best is None or score > best[0]:
                        best = (score, s, _confirmed_sh, _confirmed_run,
                                (_cqs, _cqe, _cratio), _summary)
        _quote_search_memo[_memo_key] = best
        return best

    def _admit_quote9(seg, _q, best):
        nonlocal _verbatim_first
        cands.append((best[0], seg.index, best[1], best[2], _q))
        _k = (seg.index, best[1].id, round(float(best[2].start), 1))
        _cand_origin[_k] = "verbatim_quote"        # created from a scripted quote
        _quote_confirmation_by_candidate[
            _quote_candidate_key9(seg.index, best[1], best[2], _q)] = {
                "confirmed_span": [round(float(best[4][0]), 3),
                                   round(float(best[4][1]), 3),
                                   round(float(best[4][2]), 3)],
                "confirmation": dict(best[5] or {}),
                # The confirmation artifact is bound to the exact string that was decoded against.
                # When two beats quote the same line with different punctuation the search is
                # memoised across them, so the binding can name the sibling spelling. Record what
                # THIS beat asked for, so the audit never quietly implies otherwise.
                "authored_quote_as_requested": str(_q),
            }
        # Face-ID BYPASS requires a STRONG verbatim match (>=4 words, >=70% coverage, a content
        # word) — a bare 3-4-word generic prefix used to steal the gate and air a DIFFERENT line.
        if _verbatim_bypass_ok(_rw(_q)[:8], best[3]):
            _verbatim_strong.add(_k)
        if _verbatim_first is None or seg.index < _verbatim_first[0]:
            _verbatim_first = (seg.index, _k, _rw(_q)[:8], best[3])

    _unlocated9 = []
    for seg, _q in quote_segs:
        best = _locate_quote9(_q)
        if best is None:
            _unlocated9.append((seg, _q))
            continue
        _admit_quote9(seg, _q, best)
    # QUOTE-HALLUCINATION CORRECTION — one bounded LLM re-ask per render. The per-beat `quote` comes
    # from the analysis LLM and is routinely a paraphrase or an outright invention (measured: a beat
    # carrying quote='I killed Joffrey.' — words never spoken on screen; the real confession is
    # phrased differently). An invented quote locates NOWHERE in any source's word stream, so the
    # script's promised payoff silently produces zero candidates. Fix: batch every unlocatable quote
    # into ONE fact-check call asking for the exact on-screen wording, then re-run the SAME locator
    # on the corrections. A correction only counts if it now locates in the footage's own ASR — the
    # ASR stays ground truth, so a second hallucination changes nothing and no gate is weakened.
    # Env: VIDLORE_CLIPSTUDIO_QUOTE_CORRECT=0 disables.
    if _unlocated9 and _os9b.environ.get("VIDLORE_CLIPSTUDIO_QUOTE_CORRECT", "1").strip() \
            not in ("0", "false", "no", ""):
        try:
            from . import llm as _llm9
            _lines9 = "\n".join(
                f"{i}. narration: {((getattr(t, 'text', '') or '')[:160])!r}"
                f" | claimed quote: {q[:120]!r}"
                for i, (t, q) in enumerate(_unlocated9))
            _qtxt9 = _llm9.complete(
                system=("You are a dialogue fact-checker for "
                        + (_bk_show9 or "the show/film named in the narration")
                        + ". For each numbered item, the narration references a moment and claims a "
                          "quoted line, but that wording was NOT found in the episode audio. Reply "
                          "with one line per item: '<number>. <the exact dialogue as spoken on "
                          "screen>' — or '<number>. NONE' if no such line exists. Only real "
                          "on-screen wording; never invent."),
                messages=[{"role": "user", "content": _lines9}], max_tokens=800)
            import re as _re9q
            _fixed9 = {}
            for _m9 in _re9q.finditer(r"^\s*(\d+)[.)]\s*(.+?)\s*$", _qtxt9 or "", _re9q.M):
                _c9 = _m9.group(2).strip().strip('"“”‘’\'')
                if _c9 and _c9.upper() != "NONE" and len(_c9.split()) >= 3:
                    _fixed9[int(_m9.group(1))] = _c9
            _nfix9 = 0
            for _i9, (seg, _q0) in enumerate(_unlocated9):
                _qc9 = _fixed9.get(_i9)
                if not _qc9 or _qc9.lower() == _q0.lower():
                    continue
                best = _locate_quote9(_qc9)
                if best is not None:
                    _admit_quote9(seg, _qc9, best)
                    _nfix9 += 1
                    log(f"build: quote corrected (beat {seg.index}): {_q0[:40]!r} → "
                        f"{_qc9[:60]!r} — located in footage ASR")
            log(f"build: quote check — {len(_unlocated9)} scripted quote(s) unlocatable; "
                f"corrective re-ask recovered {_nfix9}")
        except Exception as _e9q:                          # noqa: BLE001
            log(f"build: quote correction skipped ({str(_e9q)[:80]})")
    _quote_confirm_counts9 = _quote_confirmation_counts9()
    if _quote_confirm_counts9["attempted"]:
        log("[BREAKOUT-QUOTE-CONFIRM] "
            f"attempted={_quote_confirm_counts9['attempted']} "
            f"confirmed={_quote_confirm_counts9['confirmed']} "
            f"rejected={_quote_confirm_counts9['rejected']} "
            f"inconclusive={_quote_confirm_counts9['inconclusive']} "
            "(prompted ASR is retrieval-only)")
    # COLD-OPEN HOOK: the EARLIEST verbatim quote in the opening stretch is the hook — air the real
    # scene of the opening quoted line right at the start. It is still Face-ID-exempt ONLY when it is
    # itself a STRONG verbatim match (same bar as any other bypass) — a weak 3-word opening prefix no
    # longer buys Face-ID immunity just for being first.
    if _verbatim_first is not None and _verbatim_first[0] <= max(5, len(segments) // 12):
        if _verbatim_bypass_ok(_verbatim_first[2], _verbatim_first[3]):
            _verbatim_strong.add(_verbatim_first[1])
    if True:
        # EVIDENCE MINING — always runs, not just as a fallback. The competitor's edit uses the
        # scene's OWN spoken lines as evidence every ~15-30s (narration makes a point, the scene
        # proves it in its own voice). The LLM quote can be a paraphrase or a hallucination —
        # the footage's ASR is ground truth. A dialogue-rich shot qualifies when its words
        # OVERLAP a beat's narration (the show speaks about the same thing the narration is
        # explaining — exactly when a real-audio breakout feels natural).
        from .segment import _STOP as _STOP9
        # MINER-ONLY extra stopwords for the beat-overlap count: generic verbs/adverbs made
        # semantically empty matches count as "topic overlap" — 'never' + 'seize/seized' let an
        # S2 war-council clip air mid-trial-argument (audited 3/10, twice). Content overlap must
        # come from words that actually carry the beat's subject.
        _STOP9 = set(_STOP9) | {"never", "ever", "always", "said", "says", "tell", "told",
                                "thing", "things", "every", "little", "great", "want",
                                "wants", "know", "knows", "make", "makes", "made"}
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
        # BEAT-LOCAL breakout era (see the gate below): the beat the breakout airs before has its
        # own era (own words / event mapping / anchor-scene inheritance) — computed with the same
        # machinery verify uses, so a multi-era video (S4 wedding + S7 confession anchors) stops
        # legitimizing S7 sources for S4 beats.
        from . import era as _era9
        _ana_shim9 = type("A", (), {"anchor_scenes": _ana9.get("anchor_scenes"),
                                    "movie_title": _ana9.get("movie_title", ""),
                                    "characters": _ana9.get("characters"),
                                    "actors": _ana9.get("actors")})()
        _event_eras9 = _era9.event_eras_from(_ana_shim9)
        _anchor_eras9 = _era9.anchor_token_eras(_ana_shim9)
        _global_era9v = str(_ana9.get("episode_hint") or "")
        _gver9 = bool(_ana9.get("episode_hint_verified"))
        _single9v = (_ana9.get("video_type") == "single_scene")

        def _beat_season9(sidx):
            s = next((x for x in segments if getattr(x, "index", None) == sidx), None)
            if s is None:
                return None
            try:
                be = _era9.beat_era(s, _global_era9v, single_scene=_single9v,
                                    global_verified=_gver9, event_eras=_event_eras9,
                                    anchor_eras=_anchor_eras9)
                return _era9.parse_season(be) if be else None
            except Exception:
                return None
        # comparison exception: a later-era / different-installment clip may air ONLY when the script
        # explicitly compares ("unlike House of the Dragon ...") — otherwise the era-gate stays strict.
        _allow_compare9 = _script_wants_comparison(segments)
        # main cast (character + actor names) — a breakout SHOULD feature one of them; a wrong-
        # character shot (e.g. a bearded man on a boat over a Tyrion/Tywin scene) must not air.
        _main_faces9 = set()
        _char2actor9 = {}                      # canonical character name -> actor, for the
        for _ch9 in (_ana9.get("characters") or []):   # three-state identity test below
            _n9 = (_ch9.get("name") or "").strip()
            if _n9:
                _char2actor9[_n9.lower()] = (_ch9.get("actor") or "").strip()
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
                    if _texty9(sh):
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
        try:
            _min_ov9 = max(1, int(_os9b.environ.get(
                "VIDLORE_CLIPSTUDIO_BREAKOUT_MINE_MIN_OV", "2") or 2))
        except (TypeError, ValueError):
            _min_ov9 = 2
        mined = (_mine_tier([s for s in srcs if s.id in _tier1], _min_ov9)
                 + _mine_tier([s for s in srcs if s.id in _tier2], _min_ov9))
        if not cands and not mined:
            # LAST-RESORT tier used to accept ZERO-overlap lines — that is how an S2 war-council
            # clip aired mid-trial-argument and a Bronn/Shae consolation aired under Olenna's
            # confession (audited 2-3/10 both times). The floor now holds even here: a breakout
            # that shares no content with the narration is worse than no breakout at all.
            mined = _mine_tier([s for s in srcs if s.id in _tier1], _min_ov9)
        _seen_shot = {(c[2].id, int(float(c[3].start) * 10)) for c in cands}
        for c in mined:
            kk = (c[2].id, int(float(c[3].start) * 10))
            if kk not in _seen_shot:
                _seen_shot.add(kk)
                cands.append(c)
                _cand_origin.setdefault((c[1], c[2].id, round(float(c[3].start), 1)), "evidence_mined")
    if not cands:
        log("build: no breakout — no spoken line relates to the narration (natural skip)")
        # A no-breakout result is still an auditable decision.  In particular, preserve whether
        # prompted quote hits were independently rejected or the confirmation decoder was merely
        # inconclusive; those states must never collapse into a silent "no candidates".
        try:
            import json as _json9_empty
            work.mkdir(parents=True, exist_ok=True)
            (work / "breakout_audit.json").write_text(_json9_empty.dumps({
                "candidates": 0,
                "rejected_counts": dict(_rej),
                "pre_filtered_essay_or_foreign_sources": _src_excluded,
                "pre_filtered_low_resolution_sources": _bk_lowres_excluded,
                "accepted": [],
                "quote_confirmation_counts": _quote_confirmation_counts9(),
                "quote_confirmation_attempts": _quote_confirmation_attempts,
                "admission_verdicts": {},
                "log_lines": _audit_lines,
            }, indent=1), encoding="utf-8")
        except Exception:
            pass
        return []
    # COLD-OPEN: scene 0 is normally the title overlay, but the OPENING verbatim quote (the hook,
    # e.g. "Seize him. Cut his throat.") is allowed to air at scene 0 — KEEP a scene-0 candidate only
    # when it is a strong-verbatim / cold-open match, NEVER a generic evidence-mined one.
    _cold_key = (_verbatim_first[1] if (_verbatim_first is not None
                 and _verbatim_first[0] <= max(5, len(segments) // 12)) else None)
    cands = [c for c in cands
             if c[1] >= 1 or (c[1], c[2].id, round(float(c[3].start), 1)) in _verbatim_strong]
    # process the cold-open FIRST (reserve its slot before n_max / spacing fills up), then by score
    # PROMISED PAYOFF — a line the narration explicitly tells the viewer to listen for outranks a
    # merely high-scoring one. This video spends 9 minutes on "Somewhere in those ninety seconds,
    # one word gets spoken… It's the name of a poison", then says "There it is. That's the word I
    # told you to listen for. Nightshade." over footage of a garden wedding — and the viewer never
    # hears Tywin say it. A promise the edit doesn't keep is worse than no promise.
    #
    # Priority ONLY, never a bypass: this reorders candidates, and every gate below (wrong-character,
    # commentary, era, dark, burned-text, dedup, spacing) still has to pass. A promised line that
    # can't be aired safely still doesn't air.
    _promised = _promised_terms(segments)
    if _promised:
        log(f"build: breakout — narration promises the viewer will hear: {sorted(_promised)}")

    def _keeps_a_promise(c) -> bool:
        return bool(_promised and any(t in (c[4] or "").lower() for t in _promised))

    cands.sort(key=lambda x: (0 if (x[1], x[2].id, round(float(x[3].start), 1)) == _cold_key else 1,
                              0 if _keeps_a_promise(x) else 1,
                              -x[0]))
    # competitor density: one real-audio evidence moment every ~28s where material allows
    # (their edit: 13 in 180s on a single-scene essay; ours stays match-gated so sparse
    # scripts naturally get fewer) — was a hard 1-3 cap
    n_max = max(1, min(8, 1 + int(total // 28)))
    picked = []
    _picked_word_sets = []                             # for spoken-content dedup
    _picked_meta = []                                  # (source_id, win0, win1, norm_quote) per pick

    def _bk_win(cc):
        _s0 = float(cc[3].start)
        _len = min(10.0, max(3.0, float(cc[3].end) - _s0))
        return _s0, _s0 + _len

    # RESERVE. The render picked 8 and aired 3: five died AFTER extraction (dialogue classifier,
    # narration echo, coverage floor) and nothing replaced them, because selection stops at n_max
    # while 32 candidates were never even evaluated. Picking deeper changes WHICH are tried, never
    # how many air — the extract loop still stops at n_max. Spacing and dedup are evaluated in score
    # order against the full picked list, so a reserve candidate is already >=2 beats from every
    # pick including ones that later die: conservative, never wrong.
    import os as _os_bk          # this function has no module-level `os` — a bare one raises
    try:                         # NameError inside a fail-open catch, which is how a whole stage
        _bk_reserve = max(0, int(  # once died silently for months in this project
            _os_bk.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_RESERVE", "6") or 6))
    except (TypeError, ValueError):
        _bk_reserve = 6
    for c in cands:
        if len(picked) >= n_max + _bk_reserve:
            break
        if any(abs(c[1] - p[1]) < 2 for p in picked):
            _rej["spacing"] += 1
            continue                                   # >=2-scene spacing — natural, not forced
        # identity fields for content dedup (the dedup itself runs AFTER all the content gates
        # below, so a superset-replacement can never swap in a candidate that the wrong-character /
        # recap / era / commentary / luma gates would have rejected).
        _cw = {w for w in _rw(c[4])[:10] if len(w) > 2}
        # punctuation-free normalized quote — 'beast,' vs 'beast' must not defeat the
        # substring/similarity dedup keys
        _cq = " ".join(_rw(c[4]))
        _cw0, _cw1 = _bk_win(c)
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
        # POST-RENDER-QA exclusion (prior run proved this line un-airable in the final mix)
        if _qa_excl and _norm_bk_line(c[4]) in _qa_excl:
            _rej["qa_excluded"] += 1
            log(f"build: breakout skipped before scene {c[1]} — line failed a prior render's "
                f"post-render QA (excluded)")
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
        # BEAT-LOCAL era gate. The core-max above spans ALL anchors, so a multi-era video (S4
        # wedding + S7 confession anchors → core_max 7) legitimized S7 sources for S4 beats:
        # measured, 2 of 3 aired breakouts were S7 Winterfell scenes spliced under S4 King's
        # Landing narration and graded WRONG on era — the exact flaw the competitor was marked
        # down for. The breakout airs BEFORE scene c[1]; when THAT beat declares a season (own
        # words / event mapping / anchor inheritance) a source declaring ONLY later seasons never
        # airs there. The confession-anchored beats keep their S7 breakouts (their beat era IS
        # season 7); comparison scripts stay exempt.
        if not _allow_compare9:
            _bs9 = _beat_season9(c[1])
            if _bs9:
                _css9b = _era9.title_seasons(c[2].title or "")
                if _css9b and min(_css9b) > _bs9:
                    _rej["later_era_source"] += 1
                    log(f"build: breakout skipped before scene {c[1]} — source declares only "
                        f"season(s) later than the beat's era (S{_bs9})")
                    continue
        # WRONG-CHARACTER gate: a breakout airs an iconic MOMENT, so the shot must feature a confirmed
        # MAIN character (Face-ID). A shot showing no main-cast face (e.g. a bearded man on a boat over
        # a Tyrion/Tywin scene) must not air. Active only when the cast is known; a breakout is optional
        # polish, so a missed Face-ID just means no breakout there (safe), never a wrong one.
        _verbatim_ok = (c[1], c[2].id, round(float(c[3].start), 1)) in _verbatim_strong
        # IDENTITY: three states, never conflated. The old test was `face_ids & main_cast`, which
        # sounds like a wrong-character check and is not one — MEASURED on a real render, 0 of 625
        # Face-ID name instances fall outside the main cast, because `match()` is an argmax over a
        # roster that contains only main cast. So the test was `bool(face_ids)` in disguise:
        #   - it rejected 25 candidates, EVERY ONE of them merely unidentified. Looking at all 18
        #     distinct shots: 10 plainly show main cast (including Tommen on the Iron Throne
        #     abolishing trial by combat — the video's central scene, wanted by 7 beats), 5 are the
        #     right scene but too wide/dark for anyone to be identifiable, and only 3 are genuinely
        #     off-cast. Precision at the job it claims to do: 3/18 = 17%.
        #   - and it PASSED the real thing: 7 of 15 named candidates carried a main-cast face that
        #     is NOT the beat's required person, all 7 passed, and 2 aired.
        # Face-ID names only 13.4% of shots here, so "no name" is overwhelmingly *unknown*.
        _conf9 = {f.lower() for f in (getattr(c[3], "face_ids", None) or []) if str(f).strip()}
        _seg9 = next((x for x in segments if getattr(x, "index", None) == c[1]), None)
        _tgt9, _full9 = set(), False
        if _seg9 is not None and getattr(_seg9, "required_kind", "") in ("actor", "character"):
            from .match import resolve_face_targets as _rft9
            _tgt9, _full9 = _rft9(getattr(_seg9, "required_entity", ""), _char2actor9)
        # a source proven to BE this scene is evidence no face crop can give
        _scene_proof9 = _verbatim_ok or (c[2].id in _tier1) or (c[2].id in _tier2)
        if _main_faces9:
            if _conf9 and _tgt9 and _full9 and not (_conf9 & _tgt9):
                # CONFIRMED WRONG — a named main-cast face that is not who this beat is about.
                # This branch never fired before; it is the rejection the gate was supposed to make.
                _rej["wrong_char"] += 1
                log(f"build: breakout skipped before scene {c[1]} — shot shows "
                    f"{sorted(_conf9)[:2]}, beat needs {sorted(_tgt9)[:2]} (wrong character)")
                continue
            if not _conf9 and not _scene_proof9:
                # UNKNOWN, and nothing else vouches for the scene. Named honestly so the audit
                # line stops reading as "wrong character" for what is really a Face-ID miss.
                _rej["unidentified"] += 1
                log(f"build: breakout skipped before scene {c[1]} — no confident Face-ID and no "
                    f"scene proof (unidentified, not wrong)")
                continue
            if not _conf9 and _verbatim_ok:
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
        # floor — only on TRUE near-black (subject genuinely invisible). The same holds for a
        # candidate whose shot carries a CONFIRMED main-cast face: Face-ID recognizing someone IS
        # proof the shot is legible — a luma floor overruling a confirmed face is self-
        # contradictory. Measured: 7 candidates with confirmed faces (they had already passed the
        # wrong-character guard) died at YAVG 45–59 under the flat 62 floor, including
        # Purple-Wedding interiors. Full floor now applies only when the cast is UNKNOWN (no
        # Face-ID evidence either way).
        _face_confirmed9 = bool(_main_faces9
                                and ({f.lower() for f in (getattr(c[3], "face_ids", None) or [])}
                                     & _main_faces9))
        if _verbatim_ok or _face_confirmed9:
            _lflo9 = min(_lflo9, 24.0)
        if _lflo9 > 0 and getattr(c[2], "local_path", None):
            _bwin9 = min(9.0, max(2.0, float(c[3].end) - float(c[3].start)))
            _blum9 = _breakout_window_luma(c[2].local_path, float(c[3].start), _bwin9)
            if 0.0 <= _blum9 < _lflo9:
                _rej["dark"] += 1
                log(f"build: breakout skipped before scene {c[1]} — too dark/illegible to "
                    f"air (mean-luma {_blum9:.0f} < floor {_lflo9:.0f})")
                continue
        # CONTENT DEDUP (runs AFTER every content gate, so `c` is fully vetted before it may replace
        # a pick) — an expert editor never airs the SAME moment twice. A candidate duplicates an
        # already-picked breakout by ANY of: (a) same source + overlapping extracted window (the
        # beat-37/40 bug: both were src@0.0s of the same trial clip → the identical video), (b) its
        # normalized quote is a substring of / superset of a picked quote, (c) high token overlap
        # (Jaccard ≥ 0.5 on the first 10 words). On a substring/overlap pair the LONGER, more-complete
        # quote wins (it too has passed all gates); genuinely distinct dialogue moments are kept.
        _dup_i = None
        for _i, (_psrc, _pw0, _pw1, _pq) in enumerate(_picked_meta):
            _same_src_win = (_psrc == c[2].id and _cw0 < _pw1 - 0.3 and _cw1 > _pw0 + 0.3)
            _substr = bool(_cq and _pq and (_cq in _pq or _pq in _cq))
            _tok = (len(_cw & _picked_word_sets[_i])
                    / max(1, len(_cw | _picked_word_sets[_i])) >= 0.5) if _cw else False
            # (d) cross-SOURCE same line: full-line similarity — catches transcription variants
            # of one spoken line from two different uploads (the 192/234 double confession)
            _line_sim = _bk_dedup_same_line(_cq, _pq)
            if _same_src_win or _substr or _tok or _line_sim:
                _dup_i = _i
                break
        if _dup_i is not None:
            if len(_cq) > len(_picked_meta[_dup_i][3]):    # keep the more complete quote
                picked[_dup_i] = c
                _picked_word_sets[_dup_i] = _cw
                _picked_meta[_dup_i] = (c[2].id, _cw0, _cw1, _cq)
            _rej["dedup"] += 1
            continue
        picked.append(c)
        _picked_word_sets.append({w for w in _rw(c[4])[:10] if len(w) > 2})
        _picked_meta.append((c[2].id, _cw0, _cw1, _cq))

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
        f"wrong-character={_rej['wrong_char']} unidentified={_rej['unidentified']} "
        f"dark={_rej['dark']} burned-text={_rej['burned_text']} "
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
    _seg_by_idx = {s.index: s for s in segments}
    # our OWN narration script as a word stream — the backstop that catches an essay source's
    # narration re-aired as a 'breakout' (its words echo our script; a real character line does not)
    import re as _re_ns
    _script_stream = " " + " ".join(
        _re_ns.findall(r"[a-z']+", " ".join((getattr(s, "text", "") or "") for s in segments).lower())
    ) + " "
    for score, idx, src, sh, _q in sorted(picked, key=lambda p: p[1]):
        if len(out) >= n_max:
            break                     # the reserve exists to REPLACE deaths, not to add breakouts
        # pass the 10s MAX cap — _extract_breakout finds the real length that ends on a
        # complete spoken line (3-10s), so an iconic quote is never chopped mid-word
        dur = 10.0
        v = work / f"breakout_{idx:03d}.mp4"
        a = work / f"breakout_{idx:03d}.wav"
        # corner-bug sources: breakouts are cut straight from the source (build_video's per-clip
        # watermark crop never sees them), so punch-in-crop the bug corner here (memoized detector)
        _bk_corner = _logo9(shots_of.get(src.id) or []) if _cngate9 else ""
        # QUOTE-ANCHORED WINDOW. The in-point used to be the SHOT boundary (float(sh.start)), and
        # the window only ever grew forward — so a line that begins before its shot was unreachable
        # by construction. Measured: the thesis line "Any man who must say I am the king is no true
        # king" ends at 129.64s and its shot starts at 130.05s, so breakout #1 opened 0.36s AFTER
        # the most important line in the video and could never have included it.
        #
        # Anchor on the quote's own audio span instead (word-level ASR, so shot boundaries are
        # irrelevant) and size the window to CONTAIN the whole line.
        _bk_start, _bk_min = float(sh.start), 0.0
        _candidate_origin9 = _cand_origin.get(
            (idx, src.id, round(float(sh.start), 1)), "evidence_mined")
        _qtext = str(_q or "").strip()
        _confirmation_record9 = _quote_confirmation_by_candidate.get(
            _quote_candidate_key9(idx, src, sh, _q), {})
        _confirmation_summary9 = dict(
            _confirmation_record9.get("confirmation") or {})
        _span = (_confirmation_record9.get("confirmed_span")
                 if _candidate_origin9 == "verbatim_quote" else None)
        if _span:
            _qs, _qe, _qr = _span
            _bk_start = max(0.0, _qs - _BK_LEAD_S)
            _bk_min = (_qe - _bk_start) + _BK_TAIL_S      # never end before the line does
            dur = max(dur, _bk_min + 0.5)
            log(f"build: breakout scene {idx} — quote-anchored window "
                f"[{_bk_start:.2f}→{_bk_start + _bk_min:.2f}] (shot starts {float(sh.start):.2f}, "
                f"phrase match {_qr})")
        elif _qtext:
            if _candidate_origin9 == "verbatim_quote":
                # Defensive invariant: admission marks a candidate verbatim only after
                # confirmation. Never silently downgrade such a bookkeeping mismatch.
                log(f"build: breakout scene {idx} — confirmed-quote span missing from candidate "
                    f"provenance; line is NOT spoken in independently confirmed evidence; "
                    f"refusing verbatim privileges")
                continue
            # Evidence-mined dialogue remains eligible only as ordinary semantic evidence.  Say
            # explicitly that it is NOT spoken in an independently confirmed authored-quote path;
            # it gets no quote anchor, promise hint, cold-open, identity or luma privilege.
            log(f"build: breakout scene {idx} — candidate line is NOT spoken in independently "
                f"confirmed authored-quote evidence; treating it as evidence-mined only")
        _bk_quality = {}
        _native_dim = _bk_native_dims.get(src.id, {})
        real = _extract_breakout(src.local_path, _bk_start, dur, v, a,
                                 int(_native_dim.get("width") or 0), crop_corner=_bk_corner,
                                 min_dur=_bk_min, quality_meta=_bk_quality)
        if not (real and real > 1.5):
            continue
        if _bk_quality.get("legibility_grade"):
            log(f"build: breakout scene {idx} — {_bk_quality['legibility_grade']}")
        # POST-EXTRACTION WINDOW-AUDIO gate — the matched SHOT line passed the commentary gate, but
        # _extract_breakout extends the window to a full spoken line (3-10s), which can BLEED into
        # adjacent essay/commentary narration from the SAME source (observed: a 'Cersei's Fatal
        # Mistake ...' analysis source matched the in-character "chaos is a ladder" line, but the
        # aired window continued into "...Cersei spent multiple seasons after the purple wedding").
        # Validate the AIRED window's transcript (every shot it spans), not just the matched line.
        # validate the window that ACTUALLY aired — with a quote-anchored in-point that is no
        # longer the shot's start, so reading sh.start here would gate the wrong span
        _w0, _w1 = float(_bk_start), float(_bk_start) + float(real)
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
        # NARRATION-ECHO backstop: a same-topic essay source's NARRATION duplicates our own script
        # (keyword-free, so _NARRATION_RX misses it). If the aired window shares a long verbatim run
        # with our narration, it's the narrator, not a character — reject. (Measured: 3 aired
        # 'breakouts' from an essay source echoed our script by 5-9 words; a real character line
        # shares ~0.) env VIDLORE_CLIPSTUDIO_BREAKOUT_ECHO_RUN (default 6; 0 disables).
        _echo_run = int(_os9b.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_ECHO_RUN", "6") or 6)
        if _echo_run > 0 and _wtxt and _echoes_own_narration(_wtxt, _script_stream, _echo_run) >= _echo_run:
            _rej["window_commentary"] += 1
            log(f"build: breakout REJECTED post-extract before scene {idx} — aired audio DUPLICATES "
                f"our own narration script ({_echo_run}+ word run) — it is the narrator, not "
                f"in-character dialogue (src={(src.title or src.id)[:40]!r})")
            continue
        # SEMANTIC AUTHORITY (the gate that ends the whack-a-mole): the two checks above ENUMERATE
        # essay keywords — in the source title and in the transcript — and enumeration keeps losing
        # to phrasings nobody listed ("Varys's Absolute Humiliation of Tyrion Lannister" airing
        # "...of authority in Westeros, but as he opens the door"). ASK instead of MATCH: is this
        # aired line a character speaking INSIDE the story, or an essayist speaking ABOUT it? One
        # cheap text call, only on candidates that already survived every other gate. FAILS CLOSED
        # (narration / low confidence / no LLM / error → reject): a breakout is optional polish, so
        # refusing one costs nothing while airing a rival's voice-over is the worst possible outcome.
        # env VIDLORE_CLIPSTUDIO_BREAKOUT_DIALOGUE_CHECK=0 disables (back to keyword-only).
        # ...and it must belong AT THIS BEAT. "Is this the show speaking?" was never the whole
        # question: on job benjen_v2 four Season-1 Cersei/Ned lines (Robert drinking, Jaime, the
        # Iron Throne) were in-character dialogue and passed this gate cleanly, then aired inside a
        # Benjen Stark essay. Beat 112 bound its Cersei line on two fuzzy tokens — "children's"↔
        # children and "heart"↔heart — while its own quote located in ZERO of 102 word streams. So
        # the window is now identified BLIND (no beat in the prompt) and only then judged against
        # the narration, N times, unanimously.
        # A LOCATED QUOTE BUYS NOTHING. It used to skip the relevance half outright; job 6a26707939
        # scene 18 then matched "I have seen the future in the flames." (phrase 0.8) against audio
        # saying "I CAN'T see the future in the flames" and admitted it 3/3-against. `_span` is
        # recorded for the audit and decides nothing.
        if _os9b.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_DIALOGUE_CHECK", "1").strip() \
                not in ("0", "false", "no", "") and _wtxt:
            _sg9 = _seg_by_idx.get(idx)
            _relevance_on = _os9b.environ.get(
                "VIDLORE_CLIPSTUDIO_BREAKOUT_ADMIT_CHECK", "1").strip() not in ("0", "false", "no")
            _beat_txt9 = getattr(_sg9, "text", "") or ""
            _dlg_ok, _dlg_why, _verdicts9 = _breakout_window_admissible(
                _wtxt, _bk_show9,
                beat_text=_beat_txt9,
                beat_subject=(f"{getattr(_sg9, 'required_kind', '')}: "
                              f"{getattr(_sg9, 'required_entity', '')}"
                              if getattr(_sg9, "required_entity", "") else ""),
                promised_quote=(_qtext if _span else ""),
                quote_authored=bool(_span),
                # ADMIT_CHECK=0 → identification only (the old dialogue-vs-narration behaviour)
                relevance_required=_relevance_on)
            # persist the whole question, not just the answer: a post-mortem that has to re-read
            # build.log to learn WHICH window was refused has already lost the evidence
            _bk_admit_verdicts[idx] = {"ok": _dlg_ok, "why": _dlg_why,
                                       "quote_anchored": bool(_span), "verdicts": _verdicts9,
                                       "unprompted_quote_confirmation": _confirmation_summary9,
                                       "source": (src.title or src.id)[:120],
                                       "beat_text": _beat_txt9[:300],
                                       "promised_quote": (_qtext if _span else "")[:200],
                                       "aired_text": (_wtxt or "")[:400]}
            if not _dlg_ok:
                _rej["off_topic"] += 1
                log(f"build: breakout REJECTED post-extract before scene {idx} — {_dlg_why} "
                    f"(src={(src.title or src.id)[:40]!r})")
                continue
        # CUT-WINDOW FLAG VALIDATION on the aired breakout window — _extract_breakout extends
        # the cut to a full spoken line, which can cross into an adjacent shot carrying burned
        # subs or unreadable murk (the full-source corner bug is already punch-in-cropped via
        # crop_corner, so it is NOT a reject reason here).
        if _os9b.environ.get("VIDLORE_CLIPSTUDIO_WINDOW_QC", "1").strip() \
                not in ("0", "false", "no"):
            from .match import _shot_unreadable as _dark9
            _bk_dirty = ""
            for _s9 in shots_of.get(src.id, []):
                if _s9.end > _w0 + 0.05 and _s9.start < _w1 - 0.05:
                    if _sbgate9 and _sub9(_s9):
                        _bk_dirty = f"subs(shot {_s9.index})"
                        break
                    if _dark9(_s9):
                        _bk_dirty = f"unreadable(shot {_s9.index})"
                        break
            if not _bk_dirty:
                # SCREEN-RECORDING CURSOR — small white frozen blob mid-frame while the scene
                # moves (the 12.6s cursor breakout of job 5462677f95). Probes the EXTRACTED
                # clip, so it sees exactly what would air.
                _cur9 = _breakout_cursor_probe(v, float(real))
                if _cur9:
                    _bk_dirty = _cur9
            if _bk_dirty:
                _rej["window_commentary"] += 0        # tracked separately in the log line
                # breakouts are quote-locked (policy=exact by definition): the aired window is
                # NEVER shifted to a different moment — reject and try the next candidate
                log(f"window-qc: rejected breakout before scene {idx} policy=exact — aired "
                    f"window overlaps {_bk_dirty} (never shifted; trying next candidate) "
                    f"(src={(src.title or src.id)[:40]!r})")
                continue                               # try the next candidate
        _is_cold = (idx, src.id, round(float(sh.start), 1)) == _cold_key
        # NARRATOR DUPLICATION — does the narration around this beat SPEAK the same line the scene
        # is about to play (the pre-empt/echo 'sandwich')? Record it honestly. A mid-video VO word-cut
        # to remove the duplicate stays DEFAULT OFF (unsafe — the historical caption-desync bug), so
        # the safe response is to RECORD it (and optionally skip the redundant breakout via env) rather
        # than damage sync. Cold-opens are exempt: their VO-cut path IS proven/atomic.
        _dup_run = 0 if _is_cold else _narr_dup_run(_rw(_q), segments, idx)
        # DEFAULT-SKIP a mid-video breakout the narration duplicates: mid-video VO-cut stays OFF
        # (unsafe), so retaining the narrator-dialogue-narrator repetition is worse than dropping the
        # redundant breakout. Cold-opens are exempt (their VO-cut is atomic). env=0 to keep+log only.
        if (_dup_run >= 4 and not _is_cold
                and _os9b.environ.get("VIDLORE_CLIPSTUDIO_SKIP_DUP_BREAKOUT", "1").strip()
                not in ("0", "false", "no")):
            log(f"build: breakout before scene {idx} SKIPPED — narrator duplicates the line "
                f"({_dup_run}-word overlap) and mid-video VO-cut is off (SKIP_DUP_BREAKOUT)")
            continue
        # RE-ASR the EXTRACTED audio (ground truth of what actually airs). If re-ASR is unavailable or
        # returns no reliable words, the breakout is UNVERIFIED — we do NOT fall back to the indexed
        # shot transcript and call it 'aired_transcript'; we SKIP the breakout (it never airs) so a
        # breakout's aired dialogue is always its own measured audio. env=0 keeps the legacy behavior.
        from .config import _f as _cfg_fbk
        _bk_verify = _os9b.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_REASR", "1").strip() \
            not in ("0", "false", "no")
        _aw, _atext, _aspk = _asr_wav_words(a)
        if _bk_verify and len(_aw) < 2:
            log(f"build: breakout before scene {idx} SKIPPED — extracted audio could not be re-ASR'd "
                f"(no reliable words) → UNVERIFIED, not aired")
            continue
        _ocov = _ordered_coverage(_rw(_q), _aw) if _aw else 1.0
        # CANDIDATE-TYPE-AWARE ordered-coverage floor, keyed on the candidate's EXPLICIT ORIGIN (not
        # Face-ID-bypass membership). verbatim_quote → >=70%; a COLD-OPEN is a verbatim quote and gets
        # the SAME strong floor (no `not _is_cold` exemption — it is the most prominent breakout);
        # evidence_mined standalone dialogue → >=50% (its semantic relevance to the beat is established
        # by the mining overlap + commentary gate). A 25% one-word coincidence is never enough.
        _origin = _candidate_origin9
        _is_verbatim_type = (_origin == "verbatim_quote") or _is_cold
        _cand_type = "verbatim" if _is_verbatim_type else "mined"
        _mincov = (_cfg_fbk("VIDLORE_CLIPSTUDIO_BREAKOUT_MIN_COVERAGE_VERBATIM", 0.70) if _is_verbatim_type
                   else _cfg_fbk("VIDLORE_CLIPSTUDIO_BREAKOUT_MIN_COVERAGE_MINED", 0.50))
        if _bk_verify and _ocov < _mincov:             # applies to COLD-OPENS too
            log(f"build: breakout before scene {idx}{' (COLD-OPEN)' if _is_cold else ''} DROPPED — "
                f"aired-audio ordered coverage of the {_cand_type} line {_ocov:.2f} < {_mincov:.2f}; "
                f"aired={_atext[:50]!r} quote={_q[:40]!r}")
            continue
        _entry = {"seg_index": idx, "dur": real, "video": v, "audio": a,
                  "candidate_type": _cand_type, "accepted_coverage_floor": round(_mincov, 2)}
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
        # TRUTHFUL AUDIT: aired_transcript = re-ASR of the ACTUAL extracted audio (ground truth).
        # SPEAKER ATTRIBUTION is not established (Face-ID proves who is VISIBLE, not who is SPEAKING,
        # and we run no diarization) → speaker is always "unknown"; the visible faces are recorded
        # SEPARATELY as visible_faces.
        _aired_tx = _atext if _aw else (_wtxt or "")
        _cov = _ocov
        _fid = getattr(sh, "face_ids", None) or []
        # source_t records the in-point that ACTUALLY aired (quote-anchored when we located the
        # line), not the shot boundary — it is provenance, and it is also the dedup/identity key
        _entry["_audit"] = {"seg_index": idx, "cold_open": _is_cold, "dur_s": round(real, 2),
                            "source_id": src.id, "source_title": (src.title or "")[:120],
                            "source_native_width": int(_native_dim.get("width") or 0),
                            "source_native_height": int(_native_dim.get("height") or 0),
                            "legibility_grade": _bk_quality.get("legibility_grade", ""),
                            "source_t": round(float(_bk_start), 1), "line": _q[:160],
                            "shot_t": round(float(sh.start), 1),
                            "quote_anchored": bool(_span),
                            "unprompted_quote_confirmation": _confirmation_summary9,
                            "aired_transcript": _aired_tx[:300],
                            "line_coverage": round(_cov, 2),
                            "candidate_type": _cand_type,
                            "candidate_origin": _origin,
                            "accepted_coverage_floor": round(_mincov, 2),
                            "speaker": "unknown",
                            "visible_faces": list(_fid),
                            "standalone_utterance": bool(_cov >= _mincov and not _is_narration(_aired_tx)),
                            "narrator_duplication_words": _dup_run}
    # ALWAYS report the FINAL accepted count after post-extraction (the pre_extract_accepted count
    # above can shrink here when an aired window is rejected as commentary) — so the audit is never
    # ambiguous about how many breakouts actually aired.
    log(f"[BREAKOUT-AUDIT] final accepted={len(out)} (post-extract; "
        f"window_commentary rejections={_rej['window_commentary']}, "
        f"off_topic rejections={_rej['off_topic']})")
    try:
        import json as _json9
        (work / "breakout_audit.json").write_text(_json9.dumps({
            "candidates": len(cands),
            "rejected_counts": dict(_rej),
            "pre_filtered_essay_or_foreign_sources": _src_excluded,
            "pre_filtered_low_resolution_sources": _bk_lowres_excluded,
            "accepted": [e["_audit"] for e in out],
            "quote_confirmation_counts": _quote_confirmation_counts9(),
            "quote_confirmation_attempts": _quote_confirmation_attempts,
            # every admission verdict, kept or rejected — so "why did THAT breakout air?" is
            # answerable offline, without re-calling the judge
            "admission_verdicts": {str(k): v for k, v in _bk_admit_verdicts.items()},
            "log_lines": _audit_lines,
        }, indent=1), encoding="utf-8")
    except Exception:
        pass                                           # audit persistence must never fail a render
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
        if t > prev + 1e-3:                            # emit the original slice only if NON-empty —
            fparts.append(f"[0:a]atrim={prev:.3f}:{t:.3f},asetpts=PTS-STARTPTS[s{n}]")
            labels.append(f"[s{n}]")                   # a t=0 splice (a cold-open prepend) would else
            n += 1                                     # build an empty atrim=0:0 segment that fails concat
        labels.append(f"[{i}:a]")
        prev = t
    fparts.append(f"[0:a]atrim={prev:.3f},asetpts=PTS-STARTPTS[s{n}]")
    labels.append(f"[s{n}]")
    fc = ";".join(fparts) + f";{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"
    dest = work / "narration_breakouts.wav"
    # NEVER read==write: a SECOND splice pass (e.g. a cold-open spliced AFTER the mid-breakout pass,
    # whose narration.audio is already this exact file) would hand ffmpeg the same path as input -i
    # AND output → the splice fails. That failure is what silently disabled the cold-open on uploaded
    # voiceover. Write to a fresh numbered sibling when the input is the default dest.
    if Path(full).resolve() == dest.resolve():
        k = 1
        while (work / f"narration_breakouts_{k}.wav").exists():
            k += 1
        dest = work / f"narration_breakouts_{k}.wav"
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
            _wrds = _re12.findall(r"[A-Za-z']{3,}", txt or "")
            # A SINGLE long word counts too — parity with the post-cut detector
            # (_clip_has_burned_text, which fires on any confident box of >=6 letters). Without
            # this the cut-time probe needed >=3 words or a >=6.5%-height box, so a one-word
            # third-party listicle title ("2. NEEDLE") sailed through the candidate loop and was
            # only caught after the cut — by which point the only remaining move was to hide OUR
            # caption and air the foreign graphic anyway (job 69d80e9dd4, scene at 8:34).
            # Catching it HERE means the loop simply picks a different window.
            val = (len(_wrds) >= 3
                   or any(len(w) >= 6 for w in _wrds)
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
        _coa = co.get("_audit", {}) or {}
        cap_specs.append({"start": 0.0, "dur": d_clip, "audio": str(clip_audio),
                          "video": str(clip_video), "seg_index": co.get("seg_index"),
                          "source_id": _coa.get("source_id", ""),
                          "source_t": _coa.get("source_t"), "line": _coa.get("line", ""),
                          "candidate_type": _coa.get("candidate_type", "verbatim"),
                          "accepted_coverage_floor": _coa.get("accepted_coverage_floor", 0.70)})
        co.setdefault("_audit", {})["aired_at_s"] = 0.0
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
        # Preserve every carried identity field (seg_index/source/line/video) — only start moves.
        for cs in (getattr(narration, "_breakout_caps", None) or []):
            if float(cs["start"]) < h1 - tol:
                continue
            cap_specs.append({**cs, "start": round(float(cs["start"]) + delta, 3)})
        cap_specs.sort(key=lambda c: float(c.get("start", 0.0)))
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
    import copy as _copy
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

    # PRESERVE prior breakout captions across this insertion (do NOT overwrite): when a later
    # insertion (e.g. the cold-open after the mid breakouts) reindexes earlier breakout scenes,
    # their caption windows must SHIFT by this call's insertions that precede them and merge with
    # this call's — else the earlier breakouts lose their captions and their suppress windows go
    # stale (a narration caption then bleeds over the breakout's own dialogue). `_ins_pts` records
    # each insertion's INPUT-timeline position + duration for that per-position shift.
    _prior_caps0 = list(getattr(narration, "_breakout_caps", None) or [])
    _ins_pts = []
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
            _ins_pts.append((t_cursor + delta_here, float(p["dur"])))
            _pa = p.get("_audit", {}) or {}
            # FINAL-timeline start = original insert time + breakout durations spliced BEFORE it.
            # Carry stable identity (beat + source id/t + line + video/audio path) on the caption
            # spec so downstream artifacts (suppress windows, burn, audit) all key off the same
            # provenance and survive reindexing.
            cap_specs.append({"start": round(t_cursor + delta_here + shift_before, 3),
                              "dur": float(p["dur"]), "audio": str(p["audio"]),
                              "video": str(p["video"]), "seg_index": p.get("seg_index"),
                              "source_id": _pa.get("source_id", ""),
                              "source_t": _pa.get("source_t"), "line": _pa.get("line", ""),
                              "candidate_type": _pa.get("candidate_type", "mined"),
                              "accepted_coverage_floor": _pa.get("accepted_coverage_floor", 0.50)})
            p.setdefault("_audit", {})["aired_at_s"] = cap_specs[-1]["start"]
            if delta_here and len(new_ns) >= 2:        # [-1] is the pseudo scene itself
                new_ns[-2].duration = float(new_ns[-2].duration) + delta_here
                log(f"build:   breakout pause moved +{delta_here:.2f}s to the sentence end")
            shift += p["dur"]
            dur_here = p["dur"]
        ni = len(new_segs)
        # ATOMIC: reindex + word-shift on COPIES so a `_splice_audio` failure below leaves the
        # caller's segments/scenes/narration PRISTINE. (Mutating in place then raising on a failed
        # splice left the word-times shifted but the audio un-spliced → every caption lagged the
        # voice by the breakout duration. Seen as a constant ~4.4s caption desync when a t=0
        # cold-open splice failed and the caller swallowed the exception.)
        seg = _copy.copy(seg)
        seg.index = ni
        new_segs.append(seg)
        sc = sc_by_idx.get(old)
        if sc is not None:
            sc = _copy.copy(sc)
            sc.index = ni
            new_scs.append(sc)
        ns = ns_by_idx.get(old)
        if ns is not None:
            ns = _copy.copy(ns)
            ns.words = [_copy.copy(w) for w in (ns.words or [])]
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
    # append the FINAL on-timeline air times to the persisted breakout audit (written by
    # _select_breakouts) — "aired_at_s" answers WHERE in the finished video each breakout plays.
    try:
        import json as _json9b
        _baud = work / "breakout_audit.json"
        if _baud.exists():
            _data = _json9b.loads(_baud.read_text(encoding="utf-8"))
            _times = {p.get("seg_index"): (p.get("_audit") or {}).get("aired_at_s") for p in picks}
            for e in _data.get("accepted", []):
                if e.get("seg_index") in _times and _times[e["seg_index"]] is not None:
                    e["aired_at_s"] = _times[e["seg_index"]]
            _baud.write_text(_json9b.dumps(_data, indent=1), encoding="utf-8")
    except Exception:
        pass                                           # audit persistence must never fail a render
    try:
        narration.total = float(getattr(narration, "total", 0.0)) + shift
    except Exception:
        pass
    # PRESERVE + SHIFT + MERGE prior breakout captions (never overwrite): each prior caption shifts
    # by this call's insertions that precede its window in the input timeline, then merges with this
    # call's own caps. Consumed post-assemble for suppress windows + word-by-word breakout captions.
    try:
        _merged_caps = []
        for _pc in _prior_caps0:
            _t = float(_pc.get("start", 0.0))
            _sh = sum(_d for (_pos, _d) in _ins_pts if _pos <= _t + 1e-6)
            _merged_caps.append({**_pc, "start": round(_t + _sh, 3)})
        _merged_caps.extend(cap_specs)
        _merged_caps.sort(key=lambda c: float(c.get("start", 0.0)))
        narration._breakout_caps = _merged_caps
    except Exception:
        narration._breakout_caps = cap_specs
    return new_segs, new_scs, narration, bmap, idx_map


def _split_clip_sequential(clip: Path, lens: list, out_dir: Path, idx: int,
                           *, suffix: str = "bk") -> list:
    """Split one owned clip into sequential sub-clips matching planned beat lengths.

    ``suffix`` is only a filename/audit label.  Provenance stays with ``clip``; callers must
    register every returned derivative against that root before it can enter assembly.
    """
    parts, cum = [], 0.0
    for m, L in enumerate(lens):
        dest = out_dir / f"beat_{idx:03d}_{m}_{suffix}.mp4"
        p = subprocess.run([ffmpeg_exe(), "-y", "-ss", f"{cum:.3f}", "-i", str(clip),
                            "-t", f"{max(0.6, L):.3f}", "-an", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                            str(dest)], capture_output=True, timeout=120)
        if p.returncode != 0 or not dest.exists() or dest.stat().st_size <= 0:
            return []
        parts.append(dest)
        cum += max(0.6, L)
    return parts


# schema for the owned-derivative memo — bump when this function's output could change
def _sha256_file(path) -> str:
    """Content identity of a produced artifact — never its path, size or mtime."""
    import hashlib as _hl_s
    _h = _hl_s.sha256()
    with open(path, "rb") as _fh:
        for _blk in iter(lambda: _fh.read(1 << 20), b""):
            _h.update(_blk)
    return _h.hexdigest()


_FIT_MEMO_SCHEMA = 2      # 2: the entry also records the digest of the file it wrote
_FIT_MEMO_STATS = {"hit": 0, "miss": 0}


def _fit_verified_selection_clip(clip: Path, dest: Path, duration: float,
                                 *, crop_filter: str = "", zoom_to: float = 1.055,
                                 frame_exact: bool = False) -> Optional[Path]:
    """Make a time-safe 1080p derivative of *the verified cut clip only*.

    The old build reopened the long source and selected another ``beat_window`` (or a mechanical
    shot-walk) after verification.  That is the scene-34 defect: ``seg_034.mp4`` contained Varys,
    while ``beat_034_0.mp4`` was independently cut from an Olenna alternate.  This helper never
    sees a source id or source timeline.  It can trim, grade, crop and gently move the selected
    clip, and when narration needs longer than the selected window it holds the selected clip's
    final decoded frame.  It therefore cannot cross a shot/source/scene boundary.

    Duration is checked after encoding.  A failed/short derivative returns ``None`` so the caller
    blocks the build; it must never fall through to an unrelated source window or placeholder.
    """
    clip, dest = Path(clip), Path(dest)
    need = max(0.6, float(duration))
    if not clip.exists() or clip.stat().st_size <= 0:
        return None
    # CONTENT-ADDRESSED DERIVATIVE MEMO. This encode runs for every beat on every build pass, and a
    # render that self-heals and rebuilds pays for all of them again even where nothing about the
    # beat changed. The result is a pure function of its inputs, so it is keyed on ALL of them and
    # on nothing else: the selected clip's own BYTES (not its path or mtime — a re-cut writes the
    # same filename), the duration asked for, the crop, the zoom, the boundary contract in force,
    # and a schema version so a change to this function invalidates every entry.
    # A hit is re-validated exactly like a fresh encode: the file must exist, be non-empty, and
    # still satisfy `need` on probe — and the caller then puts it through the same
    # `_lineage_derive` proof either way. The memo can therefore skip work; it cannot approve any.
    _memo = dest.with_suffix(dest.suffix + ".key.json")
    _key = None
    try:
        import hashlib as _hl_fit
        import json as _js_fit
        _h = _hl_fit.sha256()
        with open(clip, "rb") as _fh:
            for _blk in iter(lambda: _fh.read(1 << 20), b""):
                _h.update(_blk)
        _key = {"schema": _FIT_MEMO_SCHEMA, "clip_sha256": _h.hexdigest(),
                "need": round(need, 4), "crop": str(crop_filter or ""),
                "zoom": round(float(zoom_to), 6), "frame_exact": bool(frame_exact)}
        # THE OUTPUT IS AN INPUT TO ITS OWN VALIDITY. Matching every input and finding a file of the
        # right duration at `dest` is not proof that the file is the one this memo wrote:
        # `_crop_clip_corner` does `out.replace(src)` on exactly this path during the caption-dodge
        # sweep, so the derivative can be rewritten IN PLACE after the entry is recorded. The key
        # would still match, the duration would still pass, and the memo would hand back a
        # caption-dodge-cropped clip as the plain derivative — applying a crop twice, or applying
        # one where none was asked for. So the entry also records the digest of what it wrote, and
        # a hit re-derives that digest from disk.
        _blob = _js_fit.loads(_memo.read_text(encoding="utf-8"))
        if dest.exists() and dest.stat().st_size > 0 \
                and _blob.get("key") == _key \
                and _blob.get("out_sha256") == _sha256_file(dest) \
                and _ffprobe_duration(dest) + (2.0 / 30.0) >= need:
            _FIT_MEMO_STATS["hit"] += 1
            return dest
    except Exception:                                    # noqa: BLE001 — a memo fault must never
        _key = None                                      # cost the encode it is memoising
    _FIT_MEMO_STATS["miss"] += 1
    # tpad is applied *after* the selection-only motion/normalisation chain.  When a short cut has
    # to fill a longer narration beat, do not clone its final decoded frame: container/frame
    # rounding can expose the first frame *after* the declared selection (and a shot ending on a
    # reverse cut would then freeze the wrong character for most of the beat).  Stop at the same
    # stable, in-window 88% sample used by the lineage bank and hold that frame instead.  The held
    # pixels therefore remain both inside the verified window and visually provable against it.
    # A generous pad plus an exact output frame count avoids stream looping.
    # A FRAME-EXACT clip (cut.cut_contract == "halfopen_v1") cannot contain the frame at `out` —
    # cut.py filtered it out on the frame's own decoded timestamp — so its true final frame is
    # verified in-window and may be held for the whole pad. Nothing is discarded.
    # Anything NOT certified keeps the conservative stop: container/frame rounding may have left
    # the first frame of the following shot at the tail, and cloning it would freeze the wrong
    # character for most of the beat (measured: 126 of 174 output frames on one scene).
    clip_duration = _ffprobe_duration(clip)
    vf = []
    if clip_duration > 0 and need > clip_duration + (2.0 / 30.0) and not frame_exact:
        safe_end = min(max(1.0 / 30.0, clip_duration - (2.0 / 30.0)),
                       max(1.0 / 30.0, clip_duration * 0.88))
        vf.extend([f"trim=end={safe_end:.3f}", "setpts=PTS-STARTPTS"])
    if crop_filter:
        vf.append(crop_filter)
    vf.append(_ken_burns_filter(need, zoom_to=zoom_to))
    vf.append(f"tpad=stop_mode=clone:stop_duration={need + 1.0:.3f}")
    frames = max(1, int(round(need * 30)))
    cmd = [
        ffmpeg_exe(), "-y", "-i", str(clip), "-an", "-vf", ",".join(vf),
        "-frames:v", str(frames), "-r", "30",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=300)
    except Exception:
        return None
    if p.returncode != 0 or not dest.exists() or dest.stat().st_size <= 0:
        return None
    got = _ffprobe_duration(dest)
    if got + (2.0 / 30.0) < need:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    if _key is not None:                                 # record the memo only for a VALIDATED file
        # `os` is imported LOCALLY on purpose. Module-level `os` is not reliably bound in this file
        # — every neighbouring function does its own `import os as _os_xx` — and an unbound name
        # inside a fail-open `except Exception` is invisible: the memo silently never records, every
        # build stays a miss, and a stray .tmp accumulates. That is exactly the class this repo
        # keeps an AST guard for, and it is how the first draft of this memo failed. Measured:
        # hit 0 / miss 2 on a warm run, with owned.mp4.key.json.tmp left on disk.
        import json as _js_w
        import os as _os_w
        _tmp = _memo.with_suffix(_memo.suffix + ".tmp")
        try:
            _tmp.write_text(_js_w.dumps({"key": _key, "out_sha256": _sha256_file(dest)}),
                        encoding="utf-8")
            _os_w.replace(_tmp, _memo)                   # atomic: old entry or new, never partial
        except Exception:                                # noqa: BLE001 — the memo is an
            try:                                         # optimisation and must never fail the
                _tmp.unlink(missing_ok=True)              # encode it is memoising
            except OSError:
                pass
    return dest


# ── BREAKOUT ARTIFACT COMPOSITION ──────────────────────────────────────────────────────────
# A breakout owns FOUR parallel artifacts that must stay in lockstep through every timeline
# transformation (mid-insertion, then cold-open insertion or VO word-cut):
#   • clip_map  {final_scene_index -> breakout video Path}    (consumed in the beat loop)
#   • bidx      {original_beat_index -> final_scene_index}     (remaps proj.selections)
#   • captions  narration._breakout_caps (start/dur + provenance, final timeline)
#   • audit     breakout_audit.json accepted[] (aired_at_s + provenance)
# The release-blocking black-breakout bug was a SILENT `.update()` that left the mid clip_map
# keyed by PRE-cold-open scene indices after the cold-open insertion reindexed every scene: the
# stale key pointed the wrong scene at the clip and the real breakout scene got no video → its
# audio aired over black. These helpers are the ONE composition mechanism every path uses so the
# four artifacts can never drift apart again.
def _compose_breakout_state(clip_map, bidx, idx_map, new_clip, new_bidx, *, log=None):
    """Fold a reindexing insertion into the running breakout state.
    `idx_map` = {scene_index_before_this_insertion -> scene_index_after} (from the insertion).
    Existing `clip_map` KEYS and `bidx` VALUES are remapped through idx_map; any that the
    insertion dropped (not in idx_map) are removed and logged (never left stale). This
    insertion's own `new_clip`/`new_bidx` (already in post-insertion indices) are then merged.
    Returns (clip_map, bidx)."""
    idx_map = idx_map or {}
    out_clip, out_bidx = {}, {}
    for _sidx, _vid in (clip_map or {}).items():
        _n = idx_map.get(_sidx, _sidx if not idx_map else None)
        if _n is None:
            if log:
                log(f"build: breakout compose — DROPPED stale clip key scene={_sidx} "
                    f"(removed by insertion, video={Path(str(_vid)).name})")
            continue
        out_clip[_n] = _vid
    for _beat, _sidx in (bidx or {}).items():
        _n = idx_map.get(_sidx, _sidx if not idx_map else None)
        if _n is None:
            if log:
                log(f"build: breakout compose — dropped stale bidx beat={_beat} (scene {_sidx} removed)")
            continue
        out_bidx[_beat] = _n
    out_clip.update(new_clip or {})
    out_bidx.update(new_bidx or {})
    return out_clip, out_bidx


def _ffprobe_duration(path) -> float:
    """Video duration in seconds via the ffmpeg banner (ffprobe is not on PATH here). 0.0 on any
    failure — callers treat 0.0 as an INVALID/empty clip."""
    try:
        r = subprocess.run([ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
        if not m:
            return 0.0
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        return 0.0


def _validate_breakout_assembly(scenes, narration, clip_map, caps, captions_on, *, log=None):
    """PRE-PUBLISH CONTRACT for breakout assembly. Every breakout pseudo-scene must be ATOMIC:
    exactly one mapped video that exists + probes non-zero, a real audio splice, one caption spec
    (when captions are on), a unique final index, and NO clip-map key pointing at an ordinary
    narration scene. Returns (ok, problems) — `problems` is a list of human strings. A failure means
    'do not publish as-is': the caller repairs the mapping or rolls the breakouts back entirely
    (never audio-over-black)."""
    problems = []
    bk_scene_idx = {sc.index for sc in scenes if getattr(sc, "visual", "") == "breakout"}
    ns_by_idx = {ns.index: ns for ns in getattr(narration, "scenes", [])}
    # 1) every clip-map key is a real breakout pseudo-scene
    for _idx in clip_map:
        if _idx not in bk_scene_idx:
            problems.append(f"clip-map key scene={_idx} is NOT a breakout pseudo-scene "
                            f"(would paint a clip over ordinary narration)")
    # 2) every breakout pseudo-scene has exactly one mapped, on-disk, non-zero video
    for _idx in sorted(bk_scene_idx):
        if _idx not in clip_map:
            problems.append(f"breakout scene={_idx} has NO mapped video (audio would air over black)")
            continue
        _vid = clip_map[_idx]
        if not Path(str(_vid)).exists():
            problems.append(f"breakout scene={_idx} video missing on disk ({_vid})")
        elif _ffprobe_duration(_vid) <= 0.1:
            problems.append(f"breakout scene={_idx} video has zero/unreadable duration ({_vid})")
        # 3) the pseudo-scene must carry its real audio splice (non-empty audio file)
        _ns = ns_by_idx.get(_idx)
        _au = getattr(_ns, "audio", None) if _ns is not None else None
        if _au is None or not Path(str(_au)).exists():
            problems.append(f"breakout scene={_idx} missing its spliced audio ({_au})")
    # 4) captions must be EXACTLY WIRED to the breakout scenes (captions on): one per scene, each
    # caption's final_index a unique breakout pseudo-scene, and each caption's own video equal to
    # that scene's mapped clip — so a caption can never suppress narration over ordinary footage nor
    # burn the wrong line over a breakout (missing / duplicate / cross-wired all caught here).
    if captions_on:
        _caps = list(caps or [])
        _fis = [c.get("final_index") for c in _caps]
        if len(_caps) != len(bk_scene_idx):
            problems.append(f"caption specs ({len(_caps)}) != breakout scenes ({len(bk_scene_idx)}) "
                            f"— a breakout would lose its caption or a stale caption would suppress "
                            f"narration over ordinary footage")
        if any(_f is None for _f in _fis):
            problems.append("a breakout caption has no final_index (unwired caption)")
        elif len(set(_fis)) != len(_fis):
            problems.append(f"breakout caption final_index values are not unique ({_fis}) — "
                            f"two captions target one scene / a scene is unwired")
        elif set(_fis) != bk_scene_idx:
            problems.append(f"caption final_index set {sorted(set(_fis))} != breakout scenes "
                            f"{sorted(bk_scene_idx)} (cross-wired or missing caption)")
        for _c in _caps:
            _fi = _c.get("final_index")
            if _fi in clip_map and str(_c.get("video", "")) != str(clip_map[_fi]):
                problems.append(f"caption at scene {_fi} video {Path(str(_c.get('video',''))).name} "
                                f"!= clip_map video {Path(str(clip_map[_fi])).name} (cross-wired)")
            # provenance: the caption's audio must be THIS pseudo-scene's spliced narration audio
            _ns2 = ns_by_idx.get(_fi)
            _sa = getattr(_ns2, "audio", None) if _ns2 is not None else None
            if _sa is not None and _c.get("audio") and str(_c.get("audio")) != str(_sa):
                problems.append(f"caption at scene {_fi} audio {Path(str(_c.get('audio'))).name} "
                                f"!= pseudo-scene audio {Path(str(_sa)).name} (provenance mismatch)")
    # 5) no two breakout scenes point at the SAME video (a stale/mis-composed key would alias one
    # clip onto two scenes, leaving the other's real clip unaired)
    _vids = [str(v) for v in clip_map.values()]
    if len(set(_vids)) != len(_vids):
        problems.append("two breakout scenes map to the same video (aliased clip → one airs unmapped)")
    ok = not problems
    if log and not ok:
        for _p in problems:
            log(f"build: breakout INVARIANT FAIL — {_p}")
    return ok, problems


def _finalize_breakout_audit(work: Path, caps, clip_map, narration, *, log=None):
    """Rewrite breakout_audit.json accepted[] with the DEFINITIVE final-timeline facts, keyed by
    STABLE identity (source_id + source_t + normalized line, with the audio path as the tiebreak) —
    never a fragile intermediate scene index. Adds final_index, aired_start/end and validation
    status so the audit always matches the finished video."""
    try:
        import json as _json
        _baud = work / "breakout_audit.json"
        if not _baud.exists():
            return 0, 0                                     # no audit file (e.g. unit test) → no-op
        data = _json.loads(_baud.read_text(encoding="utf-8"))
        # reverse clip_map: final scene index -> video path, to attach the final index by caption
        _vid_to_final = {str(v): k for k, v in (clip_map or {}).items()}

        def _norm(s):
            return " ".join((s or "").lower().split())

        # index caps by stable identity for matching against accepted entries
        cap_by_key = {}
        for c in (caps or []):
            key = (str(c.get("source_id", "")), c.get("source_t"), _norm(c.get("line", "")))
            cap_by_key[key] = c
            cap_by_key.setdefault(("audio", str(c.get("audio", ""))), c)
        for e in data.get("accepted", []):
            key = (str(e.get("source_id", "")), e.get("source_t"), _norm(e.get("line", "")))
            c = cap_by_key.get(key)
            if c is None:
                continue
            e["aired_at_s"] = float(c["start"])
            e["aired_end_s"] = round(float(c["start"]) + float(c["dur"]), 3)
            _fi = c.get("final_index")
            if _fi is None:
                _fi = _vid_to_final.get(str(c.get("video", "")))
            if _fi is not None:
                e["final_index"] = _fi
            e["video"] = str(c.get("video", "")) or e.get("video", "")
            e["validated"] = bool(_fi is not None and str(c.get("video", "")) in _vid_to_final)
        _acc = data.get("accepted", [])
        _val = sum(1 for e in _acc if e.get("validated"))
        # explicit top-level final counts (accepted_count was previously null). qa_passed is written
        # by build_video AFTER the post-render QA gate runs (unknown here → left as-is / None).
        data["accepted_count"] = len(_acc)
        data["validated_count"] = _val
        data.setdefault("qa_passed", None)
        data["final_timeline_seconds"] = round(float(getattr(narration, "total", 0.0)), 2)
        _baud.write_text(_json.dumps(data, indent=1), encoding="utf-8")
        if log:
            log(f"build: breakout audit finalized — {len(_acc)} accepted, {_val} validated, "
                f"aired times keyed to stable identity")
        return _val, len(_acc)
    except Exception as _e:                                # noqa: BLE001
        if log:
            log(f"build: breakout audit finalize skipped ({str(_e)[:70]})")
    return 0, 0


def _qa_extract_frame(ff, src, t: float, out: Path) -> bool:
    """Extract ONE frame from `src` at time `t` to a PNG and confirm it DECODES (file exists and
    PIL can open it). Returns False on any extraction/decode failure — the QA gate treats that as a
    probe FAILURE (fail closed), never a silent pass."""
    try:
        out.unlink(missing_ok=True)
        subprocess.run([ff, "-y", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
                        "-frames:v", "1", str(out)], check=True, timeout=30)
        if not out.exists() or out.stat().st_size < 67:     # smaller than a 1x1 PNG → truncated/empty
            return False
        from PIL import Image
        with Image.open(out) as _im:
            _im.load()                                      # FULLY decode the pixels (a solid-colour
        return True                                         # frame is tiny but valid; truncation raises)
    except Exception:
        return False


def _qa_crop_stats(img: Path):
    """From a decoded frame's CENTER region (drops the top/bottom letterbox bars AND the bottom
    caption band) return (dhash256, mean_luma, texture_bits):
      • dhash256   — 256-bit horizontal-gradient hash, grade-tolerant; used to confirm the FINAL
                     aired footage is the SAME scene as the prepared breakout clip;
      • mean_luma  — 0-255 average over the SAME center crop (NOT the full frame), so the black
                     check is never diluted by the letterbox bars and is always available whenever
                     the frame decoded (closing the 'luma unmeasurable -> black check skipped' gap);
      • texture_bits — popcount of the hash. A near-uniform (flat) crop hashes to ~0 bits, which
                     would false-MATCH any other flat crop, so callers treat a low-texture frame as
                     'cannot compare' rather than a match.
    Returns (None, -1.0, 0) if the frame can't be read."""
    try:
        from PIL import Image
        with Image.open(img) as _im:
            im = _im.convert("L")
        w, h = im.size
        box = (int(0.10 * w), int(0.14 * h), int(0.90 * w), int(0.70 * h))  # center, no bars/caption
        crop = im.crop(box)
        px_full = list(crop.getdata())
        mean_luma = (sum(px_full) / len(px_full)) if px_full else -1.0
        sm = crop.resize((17, 16))                         # 17x16 -> 16*16 = 256 horizontal-gradient bits
        px = list(sm.getdata())
        bits = 0
        for r in range(16):
            row = r * 17
            for c in range(16):
                bits = (bits << 1) | (1 if px[row + c] > px[row + c + 1] else 0)
        return bits, mean_luma, bin(bits).count("1")
    except Exception:
        return None, -1.0, 0


def _qa_ham(a, b) -> int:
    return bin(a ^ b).count("1")


def _postrender_breakout_qa(result: Path, caps, work: Path, *, log=None) -> list:
    """POST-RENDER visual gate — the last line before publication. For every accepted breakout,
    sample the FINAL video AND the prepared source breakout clip at the same relative positions and
    verify ALL of (each dimension FAILS CLOSED — an unverifiable probe is a failure, never a pass):
      • the final frames DECODE — a missing/unreadable/truncated probe is a FAILURE, never a silent
        skip; at least MIN_OK of the sampled positions must decode on the final video;
      • the final footage is not sustained black — center-crop mean luma < BLACK (18; limited-range
        black is Y=16). Measured on the SAME center crop as the hash so the letterbox bars never
        dilute it, and it is always available when the frame decoded (no fail-open);
      • the final footage MATCHES the prepared clip — center-crop 256-bit dHash Hamming <= MAXHAM at
        its best position. A bright but WRONG ordinary scene hashes far away and FAILS. A near-uniform
        (flat, low-texture) crop is NOT trusted for matching (it would false-match any other flat
        crop) → treated as 'cannot compare' → fail closed.
    Returns a list of problem DICTS (empty = every breakout shows its correct real scene)."""
    from .config import _i as _cfg_i
    problems = []
    ff = ffmpeg_exe()
    maxham = _cfg_i("VIDLORE_CLIPSTUDIO_BREAKOUT_QA_MAXHAM", 100)
    black_floor = float(_cfg_i("VIDLORE_CLIPSTUDIO_BREAKOUT_QA_BLACK", 18))
    min_texture = 16                                       # hash bits below this = flat/untrustworthy
    rels = (0.25, 0.4, 0.55, 0.7)                          # middle positions (avoid fade in/out edges)
    min_ok = 2
    for c in (caps or []):
        s, d = float(c.get("start", 0.0)), float(c.get("dur", 0.0))
        line = (c.get("line", "") or "")[:44]
        src = c.get("video", "")
        if d <= 0.2:
            problems.append({"breakout": line, "start": round(s, 2),
                             "reason": "zero/degenerate breakout window duration"})
            continue
        lumas, hams, decoded, probe_errs = [], [], 0, []
        for rp in rels:
            ft = s + rp * d
            fimg = work / f"_bkqa_f_{int(ft * 100)}.png"
            simg = work / f"_bkqa_s_{int(ft * 100)}.png"
            if not _qa_extract_frame(ff, result, ft, fimg):
                probe_errs.append(f"final@{ft:.1f}s undecodable")
                continue
            decoded += 1
            fh, fl, ftex = _qa_crop_stats(fimg)            # hash, center-crop luma, texture bits
            lumas.append(fl)
            sh = stex = None
            if src and Path(str(src)).exists() and _qa_extract_frame(ff, Path(str(src)), rp * d, simg):
                sh, _sl, stex = _qa_crop_stats(simg)
            elif src:
                probe_errs.append(f"source@{rp * d:.1f}s undecodable")
            # only a MATCH between two textured crops is trustworthy: a flat crop hashes to ~0 bits
            # and would false-match any other flat crop, so skip low-texture comparisons.
            if (fh is not None and sh is not None
                    and ftex >= min_texture and (stex or 0) >= min_texture):
                hams.append(_qa_ham(fh, sh))
            fimg.unlink(missing_ok=True)
            simg.unlink(missing_ok=True)
        # (a) fail closed — too few final frames decoded to trust anything
        if decoded < min_ok:
            problems.append({"breakout": line, "start": round(s, 2),
                             "probe": True,   # measurement failed; not an editorial verdict
                             "reason": f"only {decoded}/{len(rels)} final frames decoded "
                                       f"(need >= {min_ok}) — cannot verify, failing closed",
                             "probe_errors": probe_errs})
            continue
        # (b) sustained black — luma is always available when a frame decoded, so fail CLOSED if it
        # somehow isn't (rather than the old fail-open skip).
        valid_l = [x for x in lumas if x >= 0]
        if len(valid_l) < min_ok:
            problems.append({"breakout": line, "start": round(s, 2),
                             "probe": True,   # measurement failed; not an editorial verdict
                             "reason": f"could not measure luma on the decoded final frames "
                                       f"({len(valid_l)}/{decoded}) — failing closed",
                             "probe_errors": probe_errs})
            continue
        if max(valid_l) < black_floor:
            problems.append({"breakout": line, "start": round(s, 2),
                             "reason": f"airs BLACK (center-crop luma max {max(valid_l):.1f} < "
                                       f"{black_floor:.0f}) while its audio plays",
                             "lumas": [round(x, 1) for x in valid_l]})
            continue
        # (c) wrong footage — could not compare (fail closed) OR grossly different scene aired
        if not hams:
            problems.append({"breakout": line, "start": round(s, 2),
                             "probe": True,   # measurement failed; not an editorial verdict
                             "reason": "could not verify final footage against the prepared breakout "
                                       "clip (source frame undecodable or too flat to match) — "
                                       "failing closed",
                             "source": str(src), "probe_errors": probe_errs})
            continue
        if min(hams) > maxham:
            problems.append({"breakout": line, "start": round(s, 2),
                             "reason": f"final footage does NOT match the prepared breakout clip "
                                       f"(best dHash distance {min(hams)}/256 > {maxham} — a wrong "
                                       f"scene aired over the breakout audio)",
                             "hamming": hams, "source": str(src)})
    # (d) FINAL-MIX AUDIO INTELLIGIBILITY — a breakout airs the scene's OWN dialogue; verify the
    # final MIX actually carries audible speech there. FAIL-CLOSED: a probe that cannot be taken
    # (extraction/loudness measurement fails) is UNVERIFIED and hard-fails (it is not an automatic
    # pass). Near-silence and no-detectable-speech hard-fail. With ASR we ALSO require meaningful
    # ORDERED coverage of the accepted line unless the window is clearly audible speech (guarding
    # against music-masking false positives). env VIDLORE_CLIPSTUDIO_BREAKOUT_AUDIO_QA=0 disables.
    import os as _os_aq
    if _os_aq.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_AUDIO_QA", "1").strip() not in ("0", "false", "no"):
        _wm = None
        try:
            from faster_whisper import WhisperModel as _WM
            _wm = _WM("base", device="cpu", compute_type="int8")
        except Exception:
            _wm = None
        _sil_floor = float(_cfg_i("VIDLORE_CLIPSTUDIO_BREAKOUT_AUDIO_SILENCE_DB", -50))
        from .config import _f as _cfg_faq
        # DOCUMENTED ASR tolerance: the final-mix ASR runs on MIXED audio (dialogue + ducked score),
        # which whisper transcribes a little less completely than the clean extract used at selection.
        # So the final-mix floor is the candidate's OWN accepted selection floor MINUS this tolerance
        # (verbatim 0.70→~0.50, mined 0.50→~0.30) — a per-candidate threshold, NOT one unrelated 0.34.
        _mix_tol = _cfg_faq("VIDLORE_CLIPSTUDIO_BREAKOUT_AUDIO_ASR_TOLERANCE", 0.20)
        if _wm is None:
            # FAIL CLOSED: breakout QA is enabled but ASR is unavailable — we cannot verify the
            # accepted dialogue survived into the mix, so every breakout is UNVERIFIED (not a pass).
            for c in (caps or []):
                if float(c.get("dur", 0.0)) > 0.4:
                    problems.append({"breakout": (c.get("line", "") or "")[:44],
                                     "probe": True,   # measurement failed; not an editorial verdict
                                     "start": round(float(c.get("start", 0.0)), 2),
                                     "reason": "breakout final-mix ASR unavailable (no Whisper) — "
                                               "UNVERIFIED (failing closed; set BREAKOUT_AUDIO_QA=0 "
                                               "only if intentionally waiving)"})
        for c in (caps or []):
            s, d = float(c.get("start", 0.0)), float(c.get("dur", 0.0))
            line = (c.get("line", "") or "")
            if d <= 0.4 or _wm is None:
                continue
            wav = work / f"_bkaud_{int(s * 100)}.wav"
            try:
                _ext = subprocess.run(
                    [ff, "-y", "-loglevel", "error", "-ss", f"{s:.3f}", "-t", f"{d:.3f}",
                     "-i", str(result), "-vn", "-ar", "16000", "-ac", "1", str(wav)],
                    capture_output=True, timeout=60)
                if _ext.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
                    problems.append({"breakout": line[:44], "start": round(s, 2),
                                     "probe": True,   # measurement failed; not an editorial verdict
                                     "reason": "breakout audio could NOT be extracted from the final "
                                               "video for QA — UNVERIFIED (failing closed)"})
                    continue
                _vd = subprocess.run([ff, "-hide_banner", "-i", str(wav), "-af", "volumedetect",
                                      "-f", "null", "-"], capture_output=True, text=True, timeout=60).stderr
                _mm = re.search(r"mean_volume:\s*(-?[\d.]+) dB", _vd)
                if _mm is None:
                    problems.append({"breakout": line[:44], "start": round(s, 2),
                                     "probe": True,   # measurement failed; not an editorial verdict
                                     "reason": "breakout audio loudness could NOT be measured — "
                                               "UNVERIFIED (failing closed)"})
                    continue
                mean_db = float(_mm.group(1))
                speech_frac, ocov = 0.0, None
                if _wm is not None:
                    _segs, _inf = _wm.transcribe(str(wav), word_timestamps=True, vad_filter=False)
                    # faster-whisper returns a GENERATOR. It must be materialised ONCE: consuming it
                    # in the _wds comprehension left the _dur comprehension iterating an EXHAUSTED
                    # generator, so speech_frac was ALWAYS 0.00 and this gate quarantined every
                    # breakout video as "no detectable speech" — even with the dialogue plainly
                    # audible at normal level. (_asr_wav_words does a single pass for this reason.)
                    _slist = list(_segs)
                    _wds = [str(w.word or "").strip() for _sg in _slist for w in (_sg.words or [])]
                    _dur = [(float(w.start), float(w.end)) for _sg in _slist for w in (_sg.words or [])]
                    if d > 0:
                        speech_frac = min(1.0, sum(e - b for b, e in _dur) / d)
                    ocov = _ordered_coverage(re.findall(r"[a-z']+", line.lower()), _wds)
                if mean_db < _sil_floor:
                    problems.append({"breakout": line[:44], "start": round(s, 2),
                                     "reason": f"breakout airs NEAR-SILENT audio (mean {mean_db:.1f} dB "
                                               f"< {_sil_floor:.0f}) — the scene dialogue is missing",
                                     "mean_db": round(mean_db, 1)})
                elif _wm is not None and speech_frac < 0.05:
                    problems.append({"breakout": line[:44], "start": round(s, 2),
                                     "reason": f"breakout window has essentially NO detectable speech "
                                               f"(speech_frac {speech_frac:.2f}) in the final mix",
                                     "speech_frac": round(speech_frac, 3)})
                elif ocov is not None and ocov < max(0.20, float(
                        c.get("accepted_coverage_floor", 0.50)) - _mix_tol):
                    # per-candidate final-mix floor = the accepted selection floor minus the ASR
                    # tolerance (verbatim >= ~0.50, mined >= ~0.30), not one unrelated 0.34
                    _mix_floor = max(0.20, float(c.get("accepted_coverage_floor", 0.50)) - _mix_tol)
                    problems.append({"breakout": line[:44], "start": round(s, 2),
                                     "reason": f"final-mix ordered coverage of the "
                                               f"{c.get('candidate_type', '?')} line {ocov:.2f} < "
                                               f"{_mix_floor:.2f} (accepted floor "
                                               f"{c.get('accepted_coverage_floor')} − tol {_mix_tol}) "
                                               f"— dialogue masked/missing (speech {speech_frac:.2f})",
                                     "ordered_coverage": round(ocov, 2),
                                     "speech_frac": round(speech_frac, 3)})
                elif log:
                    log(f"build: breakout audio @{s:.1f}s — mean {mean_db:.1f}dB · speech "
                        f"{speech_frac:.0%} · ordered-coverage "
                        f"{('n/a' if ocov is None else f'{ocov:.0%}')}")
            except Exception:
                problems.append({"breakout": line[:44], "start": round(s, 2),
                                 "probe": True,   # measurement failed; not an editorial verdict
                                 "reason": "breakout audio QA probe raised an error — UNVERIFIED "
                                           "(failing closed)"})
            finally:
                wav.unlink(missing_ok=True)
    if log:
        if problems:
            for _p in problems:
                log(f"build: breakout POST-RENDER QA FAIL — @{_p.get('start')}s "
                    f"{_p.get('reason', '')}")
        else:
            log(f"build: breakout post-render QA passed — {len(caps or [])} breakout(s) show their "
                f"correct real footage (decoded, non-black, matched to the prepared clip)")
    return problems


def _norm_bk_line(s: str) -> str:
    """Stable identity for a breakout's spoken line across re-runs (exclusion matching)."""
    import re as _re_nb
    t = _re_nb.sub(r"[^a-z0-9 ]", "", (s or "").lower())
    return _re_nb.sub(r"\s+", " ", t).strip()[:48]


def _persist_breakout_qa_exclusions(work: Path, problems: list, log) -> None:
    """Append post-render-QA-failed breakouts to work/breakout_qa_exclude.json so the NEXT run of
    _select_breakouts recomposes WITHOUT them. A failed insert is not a failed video: the editorial
    answer is to cut the broken insert and keep the rest — but never silently in the SAME artifact
    (the gate below still quarantines this render). The exclusion is by the line's stable identity,
    so re-selection can pick a different, passing candidate for the same beat or none at all; the
    re-run's own post-render QA still applies to whatever airs."""
    import json as _json_px
    try:
        # work.parent (output/), NOT work/: the build WIPES the work dir at startup, so an
        # exclusion stored there died before the very selection it was meant to steer — the same
        # breakout re-aired and re-quarantined. breakout_qa_failures.json lives at output/ for the
        # same reason.
        xf = work.parent / "breakout_qa_exclude.json"
        prev = []
        if xf.exists():
            prev = (_json_px.loads(xf.read_text(encoding="utf-8")) or {}).get("exclude", [])
        have = {_norm_bk_line(e.get("line", "")) for e in prev}
        for p in problems or []:
            ln = str(p.get("breakout") or "")
            if ln and _norm_bk_line(ln) not in have:
                prev.append({"line": ln, "reason": str(p.get("reason", ""))[:160],
                             "at": p.get("start")})
                have.add(_norm_bk_line(ln))
        xf.write_text(_json_px.dumps({"exclude": prev}, indent=1), encoding="utf-8")
        log(f"build: {len(problems)} failing breakout(s) persisted to breakout_qa_exclude.json — "
            f"a RE-RUN recomposes without them (their beats keep normal footage)")
    except Exception:                                      # noqa: BLE001
        pass


def _breakout_qa_gate(result: Path, caps, work: Path, *, log) -> Path:
    """HARD publication gate. Runs the post-render breakout QA on the finished video and stamps the
    verdict into breakout_audit.json (qa_passed). On ANY QA failure it: writes breakout_qa_failures.
    json, QUARANTINES the final video (renames it to *.FAILED_BREAKOUT_QA.* so the normal final.mp4
    no longer exists and cannot be published/downloaded), and RAISES — so the portal/rerender caller
    finishes with ok=false and no output, and 'build: done' is never logged. Returns the (unchanged)
    result path only when every breakout passes."""
    import json as _json_qa
    problems = _postrender_breakout_qa(result, caps, work, log=log)
    try:
        _baud = work / "breakout_audit.json"
        if _baud.exists():
            _bd = _json_qa.loads(_baud.read_text(encoding="utf-8"))
            _bd["qa_passed"] = (not problems)
            _baud.write_text(_json_qa.dumps(_bd, indent=1), encoding="utf-8")
    except Exception:
        pass
    if not problems:
        return result
    try:
        (work.parent / "breakout_qa_failures.json").write_text(
            _json_qa.dumps({"failures": problems, "video": str(result)}, indent=1), encoding="utf-8")
    except Exception:
        pass
    _persist_breakout_qa_exclusions(work, problems, log)
    # QUARANTINE the failed render so nothing downstream can publish it
    _quar = result.with_name(result.stem + ".FAILED_BREAKOUT_QA" + result.suffix)
    try:
        if _quar.exists():
            _quar.unlink()
        result.rename(_quar)
    except Exception:
        _quar = result                                     # rename failed; still refuse to publish
    log(f"build: ⛔ RELEASE-BLOCKED — {len(problems)} breakout(s) failed post-render QA; quarantined "
        f"the final video → {_quar.name} (NOT published). See breakout_qa_failures.json.")
    # The failing video is quarantined either way — nothing here delivers audio over black. What the
    # kind decides is whether the DRIVER may run the recovery this gate was built to feed:
    # `_persist_breakout_qa_exclusions` has just written breakout_qa_exclude.json, and
    # `_select_breakouts` reads exactly that file, so a resume re-composes WITHOUT the offending
    # inserts and this same QA judges whatever airs instead. As a bare RuntimeError that recovery
    # was unreachable — is_content_stop said "crash", the portal offered nothing, and an eight-hour
    # render ended with no file over one bad four-second cutaway.
    #
    # Only per-breakout EDITORIAL verdicts earn it. If any problem is a probe that could not run
    # (marked at the point of record, never re-read from its prose), the honest state is "we do not
    # know what aired" and that stays fatal in every mode — dropping breakouts would be guessing.
    from .verify import NonRetryableBuildError as _NRBE_bq
    _unverified = [p for p in problems if isinstance(p, dict) and p.get("probe")]
    if _unverified:
        raise _NRBE_bq(
            f"breakout post-render QA could not be completed for {len(_unverified)} of "
            f"{len(problems)} breakout(s) — refusing to judge footage nobody could measure "
            f"(quarantined at {_quar.name})", kind="breakout_qa_probe")
    log(f"build: breakout QA verdicts are editorial — a resume will drop these "
        f"{len(problems)} insert(s) via breakout_qa_exclude.json and re-compose. If this recurs "
        f"across renders, suspect a composition regression, not the footage.")
    raise _NRBE_bq(
        f"breakout post-render QA failed for {len(problems)} breakout(s) — refusing to publish a "
        f"video that airs audio over black/wrong footage (quarantined at {_quar.name})",
        kind="breakout_qa")


def _compose_breakouts(proj, segments, scenes, narration, bks, work, captions, *, log):
    """Insert all accepted breakouts (mid-video ones first, then the cold-open via VO word-cut or
    the proven INSERT path) and COMPOSE their four parallel artifacts through EVERY reindexing with
    the single `_compose_breakout_state` mechanism, so clip_map / bidx / captions / audit can never
    drift apart (the release-blocking audio-over-black bug was a silent `.update()` that left the
    mid clip-map keyed by pre-cold-open scene indices). Validates the assembly invariant and rolls
    the WHOLE feature back to the pristine input on failure (a clean breakout-free video beats audio
    over black). Returns (segments, scenes, narration, clip_map, bidx, entries)."""
    import os
    clip_map, bidx, entries = {}, {}, list(bks or [])
    if not bks:
        return segments, scenes, narration, clip_map, bidx, []
    # PRISTINE SNAPSHOT for rollback. `_apply_breakouts`/`_apply_coldopen_vocut` copy the segment
    # and scene objects but REBIND narration.scenes/audio/total/_breakout_caps on the SAME narration
    # object (they never mutate the old lists), so snapshotting these references restores it exactly.
    _pre_segs, _pre_scs = segments, scenes
    _pre_narr = (getattr(narration, "scenes", None), getattr(narration, "audio", None),
                 getattr(narration, "total", None), getattr(narration, "_breakout_caps", None))

    def _rollback():
        narration.scenes, narration.audio = _pre_narr[0], _pre_narr[1]
        try:
            narration.total = _pre_narr[2]
        except Exception:
            pass
        try:
            narration._breakout_caps = []
        except Exception:
            pass
        return _pre_segs, _pre_scs, narration

    try:
        _co = next((b for b in bks if b.get("cold_open")), None)
        _mid = [b for b in bks if not b.get("cold_open")]
        # mid-video breakouts first — _apply_breakouts returns idx_map = {beat -> final_scene_index}
        # for EVERY beat (used to remap proj.selections too), and bmap = {pseudo_index -> video}.
        if _mid:
            segments, scenes, narration, _mid_clip, _mid_bidx = _apply_breakouts(
                proj, segments, scenes, narration, _mid, work, log)
            clip_map, bidx = _compose_breakout_state({}, {}, None, _mid_clip, _mid_bidx, log=log)
        # cold-open: VO word-cut (replace) when the strict hook gate passes; else the INSERT path.
        if _co:
            # seed a full identity beat-map when no mids ran, so the ONE composition helper remaps
            # every beat through the cold-open reindex uniformly whether or not a mid ran first.
            if not bidx:
                bidx = {sg.index: sg.index for sg in segments}
            _vocut = os.environ.get("VIDLORE_CLIPSTUDIO_VO_CUT", "1").strip() \
                not in ("0", "false", "no")
            _done = False
            if _vocut and getattr(narration, "_vo_word_aligned", False):
                _res = _apply_coldopen_vocut(proj, segments, scenes, narration, _co, work, log,
                                             protect_idx=set(clip_map.keys()))
                if _res is not None:
                    segments, scenes, narration, _co_bmap, _co_idx, _drop = _res
                    # remap running mid clip-map KEYS + bidx VALUES through the cold-open reindex,
                    # then merge the cold-open clip (_co_bmap, its pseudo-scene at index 0). bidx maps
                    # ORIGINAL beats -> final REAL scenes for proj.selections only — the cold-open
                    # pseudo-scene is synthetic (its video is in the clip-map, consumed directly), so
                    # it needs NO bidx entry; adding one would overwrite a real beat's footage mapping.
                    clip_map, bidx = _compose_breakout_state(
                        clip_map, bidx, _co_idx, _co_bmap, {}, log=log)
                    _done = True
            if not _done:                                  # fallback / flag-off → normal insert
                # THE FIX: the cold-open insertion reindexes every scene, so the mid clip-map keys
                # (pre-cold-open indices) MUST be remapped through _i2 before merging the cold-open —
                # the old `.update(_b2)` left them stale → the real breakout scene got no video and
                # its audio aired over black. The cold-open clip is in _b2 (clip-map); its "insert
                # before" beat stays a REAL narration beat whose footage mapping _i2 preserves, so
                # pass NO new bidx entry (mapping the cold-open beat here would steal that beat's clip).
                segments, scenes, narration, _b2, _i2 = _apply_breakouts(
                    proj, segments, scenes, narration, [_co], work, log)
                clip_map, bidx = _compose_breakout_state(clip_map, bidx, _i2, _b2, {}, log=log)
        # annotate each caption with its breakout pseudo-scene's FINAL index — keyed off the clip-map
        # by VIDEO PATH (the index that actually paints the clip), not bidx (the following real scene).
        _vid_to_final = {str(v): k for k, v in clip_map.items()}
        _caps_now = list(getattr(narration, "_breakout_caps", None) or [])
        for _c in _caps_now:
            _fi = _vid_to_final.get(str(_c.get("video", "")))
            if _fi is not None:
                _c["final_index"] = _fi
        try:
            narration._breakout_caps = _caps_now
        except Exception:
            pass
        # ASSEMBLY INVARIANT — every breakout atomic (one mapped non-zero video, its audio splice,
        # one caption, unique index, no map key on ordinary footage). Fail → roll the whole feature
        # back rather than publish audio-over-black.
        _ok, _probs = _validate_breakout_assembly(scenes, narration, clip_map, _caps_now,
                                                  captions, log=log)
        if not _ok:
            log(f"build: breakout assembly invariant FAILED ({len(_probs)} problem(s)) — rolling "
                f"breakouts back; publishing a clean breakout-free video")
            _s, _c, _n = _rollback()
            return _s, _c, _n, {}, {}, []
        _val, _acc = _finalize_breakout_audit(work, _caps_now, clip_map, narration, log=log)
        # every finalized audit entry must be validated=true (its clip is in the clip-map). If the
        # audit file was present and any entry failed to validate, the composition is not trustworthy
        # → roll the whole feature back rather than ship a mislabelled/unmapped breakout.
        if _acc and _val != _acc:
            log(f"build: breakout audit validation FAILED ({_val}/{_acc} validated) — rolling "
                f"breakouts back; publishing a clean breakout-free video")
            _s, _c, _n = _rollback()
            return _s, _c, _n, {}, {}, []
        return segments, scenes, narration, clip_map, bidx, entries
    except Exception as e:                                # noqa: BLE001
        log(f"build: breakouts skipped ({str(e)[:80]})")
        _s, _c, _n = _rollback()
        return _s, _c, _n, {}, {}, []


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
    from vidlore.ffmpeg_tool import seeded_noise as _seeded_n
    cmd = [ffmpeg_exe(), "-y",
           "-f", "lavfi", "-i", _seeded_n("anoisesrc=d=0.030:c=pink:a=0.9"),
           "-f", "lavfi", "-i", _seeded_n("anoisesrc=d=0.040:c=brown:a=0.9"),
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
    # Keep the hypothesis index behind every exact anchor.  Leading/trailing script words are
    # commonly substitutions rather than missing speech (for example the authored tail
    # ``liar something`` is heard by Whisper as one token, ``liars``).  The old edge fill threw
    # away that real hypothesis span and assigned every unmatched edge token ``(t, t)``.  One
    # zero-duration word then failed the all-positive validation and replaced an otherwise
    # 98%-aligned authored caption stream with Whisper's entire transcript.
    hyp_anchor: list[int | None] = [None] * len(flat)
    sm = SequenceMatcher(a=s_norm, b=h_norm, autojunk=False)
    for i1, j1, nn in sm.get_matching_blocks():
        for k in range(nn):
            _, st, en = hyp[j1 + k]
            times[i1 + k] = (st, en)
            hyp_anchor[i1 + k] = j1 + k
    n_anchor = sum(1 for t in times if t is not None)
    if n_anchor < max(8, int(0.45 * len(flat))):        # too little matched → unreliable
        return None
    anchors = [(i, t) for i, t in enumerate(times) if t is not None]

    def _edge_has_lexical_evidence(script_edge, hyp_edge) -> bool:
        """True only when every donated edge word is lexically supported by the other stream.

        Timing alone is not evidence that two edge phrases are the same speech.  The previous fix
        would put arbitrary authored words over any unrelated ASR prefix/suffix merely because the
        audio had positive duration.  Accept exact token fusions/splits (``Little finger`` vs
        ``Littlefinger``), or require every token on BOTH sides to have a strong fuzzy counterpart.
        This deliberately rejects the measured ambiguous tail ``liar something`` vs ``liars``:
        ``liar`` is supported, ``something`` is not, and stronger ASR found that the recording cuts
        off at ``some``.  In that case the accurate-ASR fallback must win.
        """
        from difflib import SequenceMatcher as _SM

        def _edge_norm(value):
            return _r.sub(r"[^\w]", "", str(value).lower(), flags=_r.UNICODE)

        authored = [_edge_norm(x) for x in script_edge]
        heard = [_edge_norm(x[0] if isinstance(x, (tuple, list)) else x) for x in hyp_edge]
        authored = [x for x in authored if x]
        heard = [x for x in heard if x]
        if not authored or not heard or len(authored) > 4 or len(heard) > 4:
            return False
        if "".join(authored) == "".join(heard):
            return True

        def _supported(token, others):
            if len(token) < 3:
                return token in others
            return max((_SM(None, token, other).ratio() for other in others), default=0.0) >= 0.72

        return (all(_supported(token, heard) for token in authored)
                and all(_supported(token, authored) for token in heard))

    def _spread_edge(first: int, stop: int, start_t: float, end_t: float) -> bool:
        """Give ``times[first:stop]`` ordered positive spans inside REAL hypothesis time.

        Refuse when the hypothesis has no positive interval to donate.  Synthesising time outside
        the ASR envelope would hide a genuinely unspoken script prefix/tail and weaken the existing
        mismatch contract; returning ``False`` keeps that case on the ASR fallback path.
        """
        import math
        count = stop - first
        if count <= 0:
            return True
        if not (math.isfinite(start_t) and math.isfinite(end_t)) or end_t <= start_t:
            return False
        step = (end_t - start_t) / count
        if not math.isfinite(step) or step <= 0.0:
            return False
        for off in range(count):
            a = start_t + step * off
            b = end_t if off == count - 1 else start_t + step * (off + 1)
            if not (math.isfinite(a) and math.isfinite(b)) or b <= a:
                return False
            times[first + off] = (a, b)
        return True

    fi, ft = anchors[0]
    if fi:
        first_h = hyp_anchor[fi]
        # An unmatched script prefix is safe only when Whisper also heard preceding material.  Its
        # exact spelling/token count may differ, but its real time span can be divided among the
        # authored words without inventing speech before the file begins.
        if first_h is None or first_h <= 0:
            return None
        prefix = hyp[:first_h]
        if not _edge_has_lexical_evidence(flat[:fi], prefix):
            return None
        p0 = min(float(s) for _, s, _ in prefix)
        p1 = min(float(ft[0]), max(float(e) for _, _, e in prefix))
        if not _spread_edge(0, fi, max(0.0, p0), p1):
            return None
    li, lt = anchors[-1]
    if li + 1 < len(times):
        last_h = hyp_anchor[li]
        # Same rule at EOF: borrow only the real suffix that Whisper tokenised differently.  If
        # there is no heard suffix, the authored tail is genuinely unsupported and alignment must
        # fail instead of fabricating positive time past the audio.
        if last_h is None or last_h + 1 >= len(hyp):
            return None
        suffix = hyp[last_h + 1:]
        if not _edge_has_lexical_evidence(flat[li + 1:], suffix):
            return None
        t0 = max(float(lt[1]), min(float(s) for _, s, _ in suffix))
        t1 = max(float(e) for _, _, e in suffix)
        if not _spread_edge(li + 1, len(times), t0, t1):
            return None
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


def _truncated_voiceover_tail_evidence(flat: list, hyp: list, total: float):
    """Identify a near-exact uploaded voiceover that runs into EOF before its script tail.

    A genuinely different recording is allowed to use ASR captions.  This only fires when at
    least 90% of both complete streams are exact anchors, no more than five tokens remain at
    either edge, and decoded speech reaches the final 200 ms.  That combination is evidence of a
    cut-off upload, not permission to invent the missing authored words or publish an ASR guess.
    """
    import math
    import re as _rt
    from difflib import SequenceMatcher as _SM
    if not flat or not hyp or not math.isfinite(total) or total <= 0.0:
        return None

    def _norm(value):
        return _rt.sub(r"[^\w]", "", str(value).lower(), flags=_rt.UNICODE)

    authored = [_norm(word) for word in flat]
    heard = [_norm(word) for word, _, _ in hyp]
    blocks = [block for block in _SM(a=authored, b=heard, autojunk=False).get_matching_blocks()
              if block.size]
    if not blocks:
        return None
    matched = sum(block.size for block in blocks)
    anchor_ratio = matched / max(len(authored), len(heard), 1)
    last = blocks[-1]
    script_tail = authored[last.a + last.size:]
    asr_tail = heard[last.b + last.size:]
    if (anchor_ratio < 0.90 or not script_tail
            or len(script_tail) > 5 or len(asr_tail) > 5):
        return None
    # A different complete final phrase is a legitimate script/recording mismatch and belongs on
    # the accurate-ASR fallback.  Truncation requires the heard tail itself to be only a strict
    # lexical prefix of the authored tail (measured: ``liars`` vs ``liar something``).
    script_joined = "".join(script_tail)
    asr_joined = "".join(asr_tail)
    if not asr_joined or not script_joined.startswith(asr_joined) \
            or len(asr_joined) >= len(script_joined):
        return None
    try:
        speech_end = max(float(end) for _, _, end in hyp)
    except Exception:
        return None
    if not math.isfinite(speech_end) or total - speech_end > 0.20:
        return None
    return {
        "exact_anchor_ratio": round(anchor_ratio, 4),
        "script_tail": script_tail,
        "asr_tail": asr_tail,
        "speech_end_s": round(speech_end, 3),
        "audio_end_s": round(float(total), 3),
    }


def _restore_secure_script_tokens(narration, flat: list, log=None,
                                  protected_terms=None) -> int:
    """Correct high-confidence ASR spellings without rewriting what the recording says.

    This is intentionally narrower than replacing an ASR transcript with the script.  Only a
    one-ASR-token ↔ one-script-token replacement is allowed only for an independently identified
    proper name supplied in ``protected_terms``; an insertion, deletion, split, merge, generic
    fuzzy word or unrelated word is left exactly as heard.  For otherwise equal tokens, authored
    internal apostrophes/hyphens are restored while ASR sentence punctuation remains in place.
    Thus a known ``Stanis`` can become ``Stannis``, but ``horse`` never replaces heard ``house``
    and ambiguous ``liar something`` never replaces ``liars``.  Word count/order/times stay fixed.
    """
    import re as _rs
    from difflib import SequenceMatcher as _SM

    words = [w for sc in (getattr(narration, "scenes", None) or [])
             for w in (getattr(sc, "words", None) or [])]
    if not flat or not words:
        return 0

    token_re = _rs.compile(r"^(\W*)([\w]+(?:[-'’][\w]+)*)(\W*)$", _rs.UNICODE)

    def _parts(raw):
        m = token_re.match(str(raw or "").strip())
        return m.groups() if m else None

    def _norm(raw):
        return _rs.sub(r"[^\w]", "", str(raw or "").lower(), flags=_rs.UNICODE)

    s_norm = [_norm(x) for x in flat]
    h_norm = [_norm(getattr(w, "word", "")) for w in words]
    protected = {_norm(value) for value in (protected_terms or []) if _norm(value)}
    sm = _SM(a=s_norm, b=h_norm, autojunk=False)
    opcodes = sm.get_opcodes()
    exact = sum(i2 - i1 for tag, i1, i2, _j1, _j2 in opcodes if tag == "equal")
    # Script spelling is evidence only when these are overwhelmingly the same narration.  On a
    # genuinely different uploaded recording, even a coincidentally similar word must remain ASR.
    if exact / max(len(s_norm), len(h_norm), 1) < 0.90:
        return 0
    rewrites: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            rewrites.extend((i1 + k, j1 + k) for k in range(i2 - i1))
        elif tag == "replace" and i2 - i1 == j2 - j1:
            # Fuzzy spelling is safe only for a separately identified proper name.  Similarity
            # alone is not semantic proof (horse/house, three/there, trial/trail all look close).
            for off in range(i2 - i1):
                a, b = s_norm[i1 + off], h_norm[j1 + off]
                authored_parts = _parts(flat[i1 + off])
                authored_word = authored_parts[1] if authored_parts else ""
                if authored_word.lower().endswith(("'s", "’s")):
                    authored_word = authored_word[:-2]
                elif authored_word.endswith(("'", "’")):
                    authored_word = authored_word[:-1]
                authored_base = _norm(authored_word)
                if (a in protected or authored_base in protected) \
                        and (b not in protected or b == authored_base) \
                        and min(len(a), len(b)) >= 4 \
                        and _SM(None, a, b).ratio() >= 0.72:
                    rewrites.append((i1 + off, j1 + off))

    fixed = 0
    examples = []
    for si, hi in rewrites:
        authored = _parts(flat[si])
        heard = _parts(getattr(words[hi], "word", ""))
        if authored is None or heard is None:
            continue
        _apre, abody, _apost = authored
        hpre, hbody, hpost = heard
        similar_replace = s_norm[si] != h_norm[hi]
        internal_mark_restore = (s_norm[si] == h_norm[hi]
                                 and any(ch in abody for ch in "-'’")
                                 and abody != hbody)
        if not (similar_replace or internal_mark_restore):
            continue
        new = f"{hpre}{abody}{hpost}"
        old = str(getattr(words[hi], "word", "") or "")
        # Preserve Whisper's leading whitespace convention used by WordTiming streams.
        new = old[:len(old) - len(old.lstrip())] + new + old[len(old.rstrip()):]
        if new == old:
            continue
        words[hi].word = new
        fixed += 1
        if len(examples) < 5:
            examples.append(f"{old.strip()}→{new.strip()}")
    if fixed and log:
        log(f"build: script-guided ASR spelling restored — {fixed} token(s) "
            f"(e.g. {', '.join(examples)})")
    return fixed


def _canonicalize_caption_names(narration, proj, log, script_text: str = "") -> int:
    """Fix ASR-spelled character names in the caption word stream ('Alina Tyrell', 'Owina',
    'James'' for Jaime) against the analysis' canonical cast list. The narration script IS a
    voiceover transcript, so proper nouns arrive with whisper spellings and burn into every
    caption — the very first caption of a measured render misspelled the video's protagonist.

    Conservative by construction: only capitalized tokens are considered; a token already equal
    to any cast token is never touched; a rewrite needs >=0.75 difflib similarity to a cast
    first/last name of >=4 chars. Possessives and edge punctuation are preserved. Word COUNT and
    timings are untouched (1:1 token rewrite), so karaoke sync is unchanged."""
    import difflib as _dl
    import re as _re_cn
    chars = ((getattr(proj, "meta", None) or {}).get("analysis", {}) or {}).get("characters") or []
    canon: dict = {}
    # surname → the character's FIRST name: 'Alina Tyrell' is only 0.55 similar to 'Olenna'
    # (whisper dropped the leading O), but the SURNAME anchors it — a capitalized unknown token
    # immediately before a cast surname is that character's first name with near-certainty, so
    # the bigram rule rewrites at a much lower similarity bar than a lone token ever could.
    surname_first: dict = {}
    for ch in chars:
        parts = [p for p in (str((ch or {}).get("name", "") or "").split()) if p]
        toks = []
        for tok in parts:
            t = _re_cn.sub(r"[^A-Za-z']", "", tok)
            if len(t) >= 4:
                canon[t.lower()] = t[0].upper() + t[1:]
                toks.append(t)
        if len(toks) >= 2:
            surname_first[toks[-1].lower()] = toks[0][0].upper() + toks[0][1:]
    if not canon:
        return 0
    # THE AUTHOR'S OWN WORDS ARE NEVER ASR ERRORS. This pass assumes the narration is a Whisper
    # transcript — true on the TTS/hypothesis path, FALSE when the owner uploads a written script
    # and the captions are word-synced from it (measured on one render: 2173 caption tokens for
    # 2173 script tokens, i.e. not one word came from ASR). On that path every rewrite is damage,
    # and it did damage: a caption read "killed Margaery Tyrell, the head of" over a clear shot of
    # MACE Tyrell, three cues after another card had already said Margaery was killed — two
    # falsehoods, burned in, and shipped in the .srt as the subtitle track too. Margaery is the
    # roster's only Tyrell, so the bigram rule turned ANY word before "Tyrell" into her name
    # ('mace' → 'margaery' scores exactly 0.500 against a 0.40 bar).
    #
    # Keying on the script makes the pass correct on BOTH paths without a mode flag: a token the
    # author typed is skipped, while a Whisper misspelling — which by construction is NOT in the
    # script — is still fixed.
    _script_words = {w.lower() for w in _re_cn.findall(r"[A-Za-z']+", script_text or "")}

    def _inflection(a: str, b: str) -> bool:
        """Is `a` just a plural/possessive of `b` (or vice versa)? 'Tyrells' vs 'Tyrell' scores
        ~0.92 similar, so every plural surname otherwise reads as a misspelling of the singular."""
        x, y = a.lower().rstrip("'"), b.lower().rstrip("'")
        return x != y and (x.rstrip("s") == y.rstrip("s"))

    def _parse(raw):
        m = _re_cn.match(r"^(\W*)([A-Za-z']+)(\W*)$", raw.strip())
        if not m:
            return None
        pre, core, post = m.groups()
        poss, base = "", core
        for sfx in ("'s", "'"):
            if base.lower().endswith(sfx):
                poss, base = base[len(base) - len(sfx):], base[:len(base) - len(sfx)]
                break
        return pre, base, poss, post

    fixed = 0
    examples = []
    for sc in getattr(narration, "scenes", []) or []:
        ws = getattr(sc, "words", []) or []
        for i, w in enumerate(ws):
            raw = str(getattr(w, "word", "") or "")
            p = _parse(raw)
            if p is None:
                continue
            pre, base, poss, post = p
            if not base[:1].isupper() or len(base) < 4:
                continue
            low = base.lower()
            if low in canon:
                continue                                  # already canonical
            if low in _script_words:
                continue                                  # the author typed it — not an ASR error
            target, target_r = None, 0.0
            # bigram rule: the NEXT token is (or canonicalizes to) a cast surname
            if i + 1 < len(ws):
                p2 = _parse(str(getattr(ws[i + 1], "word", "") or ""))
                if p2 is not None:
                    low2 = p2[1].lower()
                    sname = surname_first.get(low2) or surname_first.get(
                        (canon.get(low2) or "").lower())
                    if sname:
                        r = _dl.SequenceMatcher(None, low, sname.lower()).ratio()
                        # MEASURED, and the margin is thin: the false positive this pass shipped,
                        # 'mace' → 'margaery', scores 0.500; the true positives the rule exists for,
                        # 'alina'/'owina' → 'olenna', score 0.545. 0.52 is the only bar between
                        # them, so it is deliberately NOT the main defence — the script guard above
                        # is. This floor only matters when script_text is unavailable, and it is
                        # tuned on two data points; do not read more into it than that.
                        if r >= 0.52:
                            target, target_r = sname, r
            if target is None:
                best, best_r = None, 0.0
                for k, v in canon.items():
                    r = _dl.SequenceMatcher(None, low, k).ratio()
                    if r > best_r:
                        best_r, best = r, v
                if best is not None and best_r >= 0.75:
                    target, target_r = best, best_r
            if target is not None and _inflection(base, target):
                target = None            # 'Tyrells' is not a misspelling of 'Tyrell'
            if target is not None:
                lead = raw[:len(raw) - len(raw.lstrip())]
                trail = raw[len(raw.rstrip()):]
                w.word = f"{lead}{pre}{target}{poss}{post}{trail}"
                fixed += 1
                if len(examples) < 3 and (base, target) not in [e[:2] for e in examples]:
                    examples.append((base, target, round(target_r, 2)))
    if fixed:
        log(f"build: caption names canonicalized — {fixed} token(s) "
            f"(e.g. {', '.join(f'{a}→{b} ({r})' for a, b, r in examples)})")
    return fixed


def _narration_from_hyp(hyp, n_scenes, total, master, workdir):
    """Caption from the voiceover's OWN whisper transcription when the pasted script can't be aligned
    to it (script != voiceover — wrong file / edited draft). The transcription words ARE what is
    spoken, at real timestamps, so captions stay locked to the voice instead of the engine's drifting
    proportional split. Keeps n_scenes contiguous scenes so the scene->footage index mapping is
    unchanged; per-scene audio is sliced from the master exactly like the aligned path."""
    import math as _math
    from vidlore.tts import Narration, NarratedScene, WordTiming, _slice_scene
    ws = [(str(w), float(s), float(e)) for (w, s, e) in (hyp or [])
          if w and _math.isfinite(s) and _math.isfinite(e) and e > s]
    if not ws:
        return None
    n_scenes = max(1, int(n_scenes))
    cut = [round(k * len(ws) / n_scenes) for k in range(n_scenes)] + [len(ws)]
    scenes, prev_end = [], 0.0
    for i in range(n_scenes):
        a, b = cut[i], cut[i + 1]
        if a >= b:
            continue
        start = prev_end
        end = total if (i == n_scenes - 1 or b >= len(ws)) else min(total, max(start + 0.2, ws[b][1]))
        # Reserve a positive slice for every remaining scene.  A ballooned alignment used to
        # consume the whole master here, forcing all later words to ``start == end == total`` and
        # creating the zero-duration cues found in the delivered render.
        _remain = max(0, n_scenes - i - 1)
        _latest = max(start + 0.2, total - 0.2 * _remain)
        end = min(max(end, start + 0.2), total, _latest)
        words = []
        for k in range(a, b):
            wstart = min(max(start, ws[k][1]), end)
            wend = min(max(wstart, ws[k][2]), end)
            if wend <= wstart:
                # The source timestamp fell outside a scene slice reserved after a ballooned
                # neighbour.  Allocate a tiny ordered slot inside this scene instead of emitting
                # a zero-duration word; the publish CPS gate later decides whether such compressed
                # speech is readable enough to ship.
                _cnt = max(1, b - a)
                _slot = max(0.01, (end - start) / _cnt)
                wstart = min(end - 0.01, start + (k - a) * _slot)
                wend = min(end, max(wstart + 0.01, start + (k - a + 1) * _slot))
            words.append(WordTiming(ws[k][0], wstart, wend))
        wav = workdir / f"scene_{i:03d}.wav"
        _slice_scene(master, start, end, total, wav)
        scenes.append(NarratedScene(i, wav, max(0.2, end - start), words))
        prev_end = end
    if not scenes:
        return None
    return Narration(scenes=scenes, audio=master, reused=0)


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
    _align_ok = (bool(aligned) and len(aligned) == len(flat)
                 and all(math.isfinite(s) and math.isfinite(e) for s, e in aligned)
                 and all(e > s for s, e in aligned)
                 and all(aligned[i][0] >= aligned[i - 1][0] - 0.25
                         for i in range(1, len(aligned)))
                 and aligned[-1][1] >= 0.5 * total)
    if not _align_ok:
        _tail_evidence = _truncated_voiceover_tail_evidence(flat, hyp, total)
        if _tail_evidence:
            raise RuntimeError(
                "uploaded voiceover appears cut off at EOF: near-exact script alignment leaves "
                f"unproven final words {_tail_evidence['script_tail']} while ASR ends with "
                f"{_tail_evidence['asr_tail']} at {_tail_evidence['speech_end_s']:.3f}s / "
                f"{_tail_evidence['audio_end_s']:.3f}s; supply a complete voiceover"
            )
        # The pasted script does NOT align to the uploaded voiceover (far too few sequence anchors —
        # the script and the voiceover are different content, or a wrong file was uploaded). DON'T
        # fall through to the engine's proportional split — over a long VO it drifts captions seconds
        # off the voice. Caption from the voiceover's OWN transcription so on-screen text always
        # tracks what is actually spoken (the script is used for captions only when it can be matched).
        if hyp:
            _log("[caption-sync] pasted script does NOT match the uploaded voiceover — captioning "
                 "from the voiceover transcription so captions track the VOICE "
                 "(verify the script and voiceover are the same content)")
            _nar = _narration_from_hyp(hyp, n, total, master, workdir)
            if _nar is not None:
                _restore_secure_script_tokens(_nar, flat, _log)
                try:
                    from vidlore.captions import _caption_schedule, caption_schedule_problems
                    _hp = caption_schedule_problems(
                        _caption_schedule(_nar.all_words()), hard_cps=float("inf"))
                except Exception:
                    _hp = [{"reason": "caption timing validation failed"}]
                if not _hp:
                    return _nar
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
        # Keep enough timeline for every later script scene; otherwise one ballooned alignment
        # consumes the master and clamps all subsequent word spans to zero at EOF.
        _remain = max(0, n - i - 1)
        _latest = max(start + 0.2, total - 0.2 * _remain)
        end = min(max(end, start + 0.2), total, _latest)
        w_times = []
        for k in range(a, b):
            ws = min(max(start, aligned[k][0]), end)
            we = min(max(ws, aligned[k][1]), end)
            if we <= ws:
                _cnt = max(1, b - a)
                _slot = max(0.01, (end - start) / _cnt)
                ws = min(end - 0.01, start + (k - a) * _slot)
                we = min(end, max(ws + 0.01, start + (k - a + 1) * _slot))
            w_times.append(WordTiming(flat[k], ws, we))
        wav = workdir / f"scene_{sc.index:03d}.wav"
        _slice_scene(master, start, end, total, wav)
        scenes.append(NarratedScene(sc.index, wav, max(0.2, end - start), w_times))
        prev_end = end
    _nar = Narration(scenes=scenes, audio=master, reused=0)
    # Scene balloon clamps can still squeeze a late aligned word to zero even when the raw aligner
    # stream was valid.  Validate the constructed viewer-facing stream, then use Whisper's own
    # timings as the only safe fallback; never hand a zero-duration cue to the renderer.
    try:
        from vidlore.captions import _caption_schedule, caption_schedule_problems
        _timing_problems = caption_schedule_problems(
            _caption_schedule(_nar.all_words()), hard_cps=float("inf"))
    except Exception:
        _timing_problems = [{"reason": "caption timing validation failed"}]
    if _timing_problems:
        _log(f"[caption-sync] aligned script produced {len(_timing_problems)} invalid timing "
             f"issue(s); trying the voiceover's own timestamped transcription")
        _hyp_nar = _narration_from_hyp(hyp, n, total, master, workdir) if hyp else None
        if _hyp_nar is not None:
            _restore_secure_script_tokens(_hyp_nar, flat, _log)
            try:
                _hp = caption_schedule_problems(
                    _caption_schedule(_hyp_nar.all_words()), hard_cps=float("inf"))
            except Exception:
                _hp = [{"reason": "caption timing validation failed"}]
            if not _hp:
                return _hyp_nar
        return None
    _log(f"build: captions word-synced to voiceover — {len(flat)} words"
         + (f", {clamped} scene(s) clamped" if clamped else ""))
    return _nar


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


def _sig_scene_tokens(seg) -> set:
    """Meaningful scene tokens for a beat (stopwords + ≤2-char noise removed) drawn from
    scene_query + required_entity + expected_visual. Stopwords carry no scene identity, so they
    must never manufacture overlap between two unrelated beats."""
    import re as _re
    if seg is None:
        return set()
    _txt = " ".join(str(getattr(seg, _a, "") or "") for _a in
                    ("scene_query", "required_entity", "expected_visual"))
    return {w for w in _re.findall(r"[a-z0-9]+", _txt.lower())
            if w not in _STOPE and len(w) > 2}


def _hold_scene_compat(seg_a, seg_b, sel_a, sel_b, *, single_scene, global_era,
                       overlap_min=0.4, char2actor=None):
    """REAL same-scene compatibility test for an editorial hold (R4-3): may a freeze of beat a's
    clean frame legitimately cover rejected beat b? Every gate is evaluated even for single-scene
    videos (never auto-true). Returns (ok: bool, evidence: dict).

    Gates: (1) canonical beat-local season/era must not conflict; (2) meaningful scene/entity tokens
    (stopwords removed) must overlap enough to be the same moment; (3) a beat demanding a NAMED
    person may not be covered by a frame whose Face-ID identity is a DIFFERENT named person —
    `char2actor` (the analysis roster) maps character↔actor first, because Face-ID identities are
    ACTOR names while beats name CHARACTERS (a 'sophie turner' held frame IS 'sansa stark').
    Source-id continuity is recorded as positive evidence, not a gate."""
    import re as _re
    from .verify import _beat_era, _era_conflict
    if seg_a is None or seg_b is None:
        return False, {"reason": "missing beat metadata"}
    ev = {}
    era_a = _beat_era(seg_a, global_era, single_scene)
    era_b = _beat_era(seg_b, global_era, single_scene)
    ev["era_prev"], ev["era_cur"] = era_a or "any", era_b or "any"
    if _era_conflict(era_a, era_b):                 # canonical: 'S04E01' vs 'season 4' is SAME era
        return False, {"reason": f"era mismatch ({era_a} vs {era_b})", **ev}
    ta, tb = _sig_scene_tokens(seg_a), _sig_scene_tokens(seg_b)
    if not (ta and tb):
        return False, {"reason": "no meaningful scene tokens to compare", **ev}
    _shared = ta & tb
    _ov = len(_shared) / max(1, min(len(ta), len(tb)))
    ev["scene_overlap"] = round(_ov, 2)
    ev["shared_tokens"] = sorted(_shared)[:8]
    if _ov < overlap_min:
        return False, {"reason": f"scene tokens differ (overlap {_ov:.2f})", **ev}
    _src_a = getattr(sel_a, "source_id", "") or ""
    _src_b = getattr(sel_b, "source_id", "") or ""
    _src_cont = bool(_src_a and _src_a == _src_b)
    # (2b) In a MULTI-scene video the same character appears in many different moments, so an overlap
    # made up ONLY of the character/entity name (same person, different location/action) is NOT the
    # same scene — a throne-room frame must not freeze over a battlefield beat. Require at least one
    # shared LOCATION/ACTION token beyond either beat's required_entity, unless it's the SAME source
    # shot lineage (genuine continuity) or a single-scene deep-dive (every beat IS the one scene).
    _ent_a = {w for w in _re.findall(r"[a-z0-9]+",
              (getattr(seg_a, "required_entity", "") or "").lower()) if len(w) > 2}
    _ent_b = {w for w in _re.findall(r"[a-z0-9]+",
              (getattr(seg_b, "required_entity", "") or "").lower()) if len(w) > 2}
    _scene_shared = _shared - _ent_a - _ent_b
    if not single_scene and not _scene_shared and not _src_cont:
        return False, {"reason": "same entity, different scene (no shared location/action token)",
                       **ev}
    _id_a = (getattr(sel_a, "identity", "") or "").strip().lower()
    _need_b = (getattr(seg_b, "required_entity", "") or "").strip().lower()
    _kind_b = (getattr(seg_b, "required_kind", "") or "").strip().lower()
    if _id_a and _need_b and _kind_b in ("character", "actor"):
        _need_toks = {w for w in _re.findall(r"[a-z0-9]+", _need_b) if len(w) > 2}
        _id_toks = {w for w in _re.findall(r"[a-z0-9]+", _id_a) if len(w) > 2}
        # widen the beat's accepted names with the roster mapping BOTH ways (character→actor
        # and actor→character) — without this a legitimate same-person hold was rejected as
        # "held frame shows 'sophie turner', beat needs 'sansa stark'"
        _alias = set(_need_toks)
        for _ch, _ac in (char2actor or {}).items():
            _cht = {w for w in _re.findall(r"[a-z0-9]+", str(_ch).lower()) if len(w) > 2}
            _act = {w for w in _re.findall(r"[a-z0-9]+", str(_ac).lower()) if len(w) > 2}
            if _cht and _act and ((_cht & _need_toks) or (_act & _need_toks)):
                _alias |= _cht | _act
        if _need_toks and _id_toks and not (_alias & _id_toks):
            return False, {"reason": f"held frame shows '{_id_a}', beat needs '{_need_b}'", **ev}
    ev["entity_prev"] = _id_a or "n/a"
    ev["source_continuity"] = _src_cont
    return True, ev


def _hold_block_reason(*, clips_present, has_predecessor, compat_ok, compat_reason,
                       consec_holds, hold_cap, beat_hold_dur, hold_total,
                       single_cap, total_cap):
    """The single fail-closed decision (R4-4) for whether a verifier-rejected beat may become a
    bounded editorial hold. Returns a block-reason string (UNRESOLVED → release-block), or None to
    permit the hold. EVERY failure mode is enumerated — none may silently fall through to a black
    or rejected frame."""
    if not clips_present:
        return "beat has no footage at all (empty beat_clips) — unresolved"
    if not has_predecessor:
        return "no clean predecessor"
    if not compat_ok:
        return f"not same scene — {compat_reason}"
    if consec_holds >= hold_cap:
        return f"a consecutive hold already used (cap {hold_cap}) — long repeated freeze"
    if beat_hold_dur > single_cap:
        return (f"hold would freeze {beat_hold_dur:.1f}s > single-hold cap "
                f"{single_cap:.1f}s (frozen frame too long)")
    if hold_total + beat_hold_dur > total_cap:
        return (f"cumulative holds {hold_total + beat_hold_dur:.1f}s > total cap "
                f"{total_cap:.1f}s (too much frozen footage)")
    return None


def preassemble_release_block_reason(proj, segments, analysis=None):
    """FAIL-FAST predictor for the end-of-assembly rejected-footage release gate (the
    '3a-3) REJECTED-FOOTAGE HANDLING' block in build_video). That gate's verdict is fully
    determined by the — now frozen — selections + segment metadata; it reads no encoded pixel. So a
    footage-gap render can learn it will release-block BEFORE assembly, instead of only after the
    ~20-minute per-beat re-encode loop that a doomed render then discards.

    SOUND & FAIL-OPEN by construction. It flags a rejected beat as doomed ONLY when NO preceding
    air-worthy beat anywhere in the timeline is same-scene-compatible with it (durations ignored,
    via the very same `_hold_scene_compat`/`_hold_block_reason` primitives the real gate uses). The
    real gate's only rescue for a verifier-rejected, still-less beat is an editorial hold from its
    nearest compatible clean predecessor — so 'no compatible predecessor exists at all' guarantees
    the real gate blocks it too. It therefore reports a SUBSET of the gate's blocks: it can only
    ever save wasted work, never wrongly kill a render the gate would have let through. The
    authoritative gate at assembly remains the final word.

    Returns a human-readable block reason (mirroring the gate's message), or None to proceed.
    Honors VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE (disabled → None)."""
    import os as _os
    if _os.environ.get("VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE", "1").strip() in ("0", "false", "no"):
        return None
    from .config import _f as _cfg_f, _i as _cfg_i
    a = (proj.meta.get("analysis", {}) or {})
    if analysis is not None:
        a = analysis if isinstance(analysis, dict) else analysis.to_dict()
    single_scene = a.get("video_type", "") == "single_scene"
    global_era = str(a.get("episode_hint", "") or "")
    overlap_min = _cfg_f("VIDLORE_CLIPSTUDIO_HOLD_SCENE_OVERLAP", 0.4)
    hold_cap = int(_cfg_i("VIDLORE_CLIPSTUDIO_MAX_CONSEC_HOLD", 1))
    c2a = {str(c.get("name", "")): str(c.get("actor", ""))
           for c in (a.get("characters") or []) if isinstance(c, dict)}
    sel_by_idx = {s.segment_index: s for s in proj.selections}

    def _rejected(sel):
        return bool(sel is not None
                    and "verifier_failed" in (getattr(sel, "flag_reasons", None) or [])
                    and not getattr(sel, "image_path", ""))

    def _air_worthy(sel):
        # a beat that will show real, non-rejected content — the only legitimate hold source
        if sel is None:
            return False
        if getattr(sel, "image_path", ""):
            return True                                  # a validated still airs
        if _rejected(sel):
            return False                                 # a rejected/held beat can't anchor a hold
        return bool(getattr(sel, "source_id", "") or getattr(sel, "beat_windows", None))

    blocked = []
    for pos, seg in enumerate(segments):
        if not _rejected(sel_by_idx.get(seg.index)):
            continue
        _has_pred = False
        for prev in segments[:pos]:                      # ALL preceding beats (not just the nearest)
            psel = sel_by_idx.get(prev.index)
            if not _air_worthy(psel):
                continue
            compat_ok, ev = _hold_scene_compat(
                prev, seg, psel, sel_by_idx.get(seg.index),
                single_scene=single_scene, global_era=global_era,
                overlap_min=overlap_min, char2actor=c2a)
            if _hold_block_reason(
                    clips_present=True, has_predecessor=True, compat_ok=compat_ok,
                    compat_reason=ev.get("reason", "incompatible"), consec_holds=0,
                    hold_cap=hold_cap, beat_hold_dur=0.0, hold_total=0.0,
                    single_cap=1e9, total_cap=1e9) is None:
                _has_pred = True
                break
        if not _has_pred:
            blocked.append(seg.index)
    if not blocked:
        return None
    # Deliberately worded differently from the authoritative build-stage release gate — this is the
    # EARLY predictor, and distinct wording keeps the two messages independently greppable/testable.
    return (f"pre-assembly footage feasibility: {len(blocked)} verifier-rejected beat(s) have no "
            f"valid same-scene fallback anywhere in the timeline — scene(s) {blocked[:8]}. "
            f"Rediscovery / more footage needed (CONTENT failure: re-running unchanged will not fix it).")


def _resolve_music(music, theme_name: str, total: float, work: Path, log=None):
    """A cinematic background bed. User-supplied path wins; else the engine's theme-aware
    compose_score(); else a deterministic track from the matching music bucket.

    THREE measured defects fixed here after a full render shipped with NO music at all:
      1. the arc cues used {"t": ...} points while compose_score's contract is
         {"start","end"} SEGMENTS — KeyError('end') on every call, so the 'retention arc'
         had never actually run and every render silently used the single-track fallback;
      2. that KeyError was swallowed by a bare `except: pass` (the voiceover-v3 bug class);
      3. the fallback globbed the PACKAGE-relative assets dir, ignoring VIDLORE_MUSIC_DIR —
         empty in a git worktree (mp3s are gitignored), so the fallback also returned None.
    Failures are now LOGGED, the fallback is env-aware (musiclib.library_root), and the
    caller hard-fails on a None result unless VIDLORE_CLIPSTUDIO_ALLOW_NO_MUSIC=1."""
    import os
    if music and Path(music).exists():
        return music
    bucket = _MUSIC_BUCKET.get(theme_name, "historical_epic")
    try:                                              # engine's crossfaded, ducked, arc-aware score
        from vidlore.musiclib import compose_score
        # RETENTION MUSIC ARC as SEGMENTS (compose_score's contract): hook → ease → build →
        # swell into the climax (~80%) → outro soften. env: MUSIC_ARC=0 → flat two segments.
        _t = max(2.0, float(total))
        if os.environ.get("VIDLORE_CLIPSTUDIO_MUSIC_ARC", "1").strip() not in ("0", "false", "no"):
            _pts = [(0.0, 4), (_t * 0.08, 2), (_t * 0.35, 3),
                    (_t * 0.62, 4), (_t * 0.82, 5), (_t * 0.95, 3)]
        else:
            _pts = [(0.0, 3), (max(1.0, _t * 0.6), 4)]
        cues = [{"start": p[0], "end": (_pts[i + 1][0] if i + 1 < len(_pts) else _t),
                 "category": bucket, "intensity": p[1]}
                for i, p in enumerate(_pts)]
        dest = work / "score.wav"
        p = compose_score(cues, _t, dest)
        if p and Path(p).exists():
            if log:
                log(f"build: music — composed {len(cues)}-segment '{bucket}' arc score")
            return str(p)
        if log:
            log(f"build: music — compose_score returned no track for bucket '{bucket}' "
                f"(library empty?) — falling back to a single track")
    except Exception as e:                            # noqa: BLE001 — log it, NEVER swallow it
        if log:
            log(f"build: music — compose_score FAILED ({type(e).__name__}: {str(e)[:80]}) — "
                f"falling back to a single track")
    try:                                              # env-aware fallback: pick a track directly
        import vidlore.musiclib as _ml
        base = _ml.library_root()                     # honors VIDLORE_MUSIC_DIR (worktree-safe)
        tracks = sorted((base / bucket).glob("*.mp3")) or sorted(base.glob("*/*.mp3"))
        if tracks:
            pick = tracks[len(tracks) // 3]
            if log:
                log(f"build: music — single-track fallback: {pick.name}")
            return str(pick)
        if log:
            log(f"build: music — library at {base} has NO tracks")
    except Exception as e:                            # noqa: BLE001
        if log:
            log(f"build: music — fallback failed ({type(e).__name__}: {str(e)[:80]})")
    return None


def _music_envelope_expr(breakout_wins: list, reveal_wins: list, *, dip: float = 0.15,
                         boost: float = 1.15, r_dip: float = 0.4, r_boost: float = 0.6) -> str:
    """A smooth ffmpeg `volume=eval=frame` gain expression: 1.0 under normal narration, a HARD dip
    (dip) across each real-audio breakout so the scene dialogue is clearly the loudest thing, and a
    gentle swell (boost) across reveal/climax beats (heard mainly in the narration gaps, since the
    engine's sidechain still ducks music under the voice). Trapezoid ramps (2-arg min of two clipped
    edges) give NO jumps/pumping. Multiplying the factors means a breakout dip always wins over an
    overlapping reveal boost. Empty windows → constant 1.0 (no-op)."""
    def _tri(a, b, r):                                  # 0->1->0 trapezoid over [a-r, a..b, b+r]
        return (f"min(clip((t-{a - r:.3f})/{r:.3f},0,1),"
                f"clip(({b + r:.3f}-t)/{r:.3f},0,1))")
    terms = ["1.0"]
    for a, b in breakout_wins:
        if b > a:
            terms.append(f"(1-{1 - dip:.3f}*{_tri(a, b, r_dip)})")
    for a, b in reveal_wins:
        if b > a:
            terms.append(f"(1+{boost - 1:.3f}*{_tri(a, b, r_boost)})")
    return "*".join(terms)


def _shape_music_envelope(music_path, total: float, breakout_wins: list, reveal_wins: list,
                          work: Path, *, log=None):
    """Bake a natural dynamics ENVELOPE onto the music track BEFORE it reaches assemble() — a
    clipstudio-side, engine-untouched approach (the shared assemble mix keeps its single static bed
    gain + voice sidechain). Music is strongly ducked during breakouts and swells gently on reveals;
    the uploaded voiceover is NEVER touched. Returns the shaped path (or the original on any failure /
    when disabled). env VIDLORE_CLIPSTUDIO_MUSIC_DYNAMICS=0 disables."""
    import os as _os_m
    if not music_path or not Path(str(music_path)).exists():
        return music_path
    if _os_m.environ.get("VIDLORE_CLIPSTUDIO_MUSIC_DYNAMICS", "1").strip() in ("0", "false", "no"):
        return music_path
    if not breakout_wins and not reveal_wins:
        return music_path
    expr = _music_envelope_expr(breakout_wins, reveal_wins)
    if expr == "1.0":
        return music_path
    dest = work / "score_shaped.wav"
    _t = max(2.0, float(total) + 1.0)
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
           "-stream_loop", "-1", "-i", str(music_path), "-t", f"{_t:.3f}",
           "-af", f"volume=eval=frame:volume='{expr}'", "-ar", "44100", "-ac", "2", str(dest)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=600)
        if p.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            if log:
                log(f"build: music dynamics — {len(breakout_wins)} breakout duck(s) + "
                    f"{len(reveal_wins)} reveal swell(s) baked (engine sidechain unchanged)")
            return str(dest)
        if log:
            log(f"build: music dynamics skipped (ffmpeg rc={p.returncode}) — flat bed kept")
    except Exception as _e:
        if log:
            log(f"build: music dynamics skipped ({type(_e).__name__}) — flat bed kept")
    return music_path


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


# SPEED: per-THREAD RapidOCR engines let the branding / caption-dodge / ad-scan sweeps run
# their pure per-clip probes in the existing bounded pools. An engine instance is state-free
# across calls and two instances built with the identical default config produce bit-identical
# (box, text, conf) output — proven by the 200-frame serial-vs-pool canary (120 text-bearing,
# 0 mismatches) run for the index OCR pool, which uses the same construction. Any construction
# failure returns the caller's shared fallback engine (serial semantics).
import threading as _threading_ocr
_OCR_TL = _threading_ocr.local()


def _ocr_engine_tl(fallback):
    try:
        eng = getattr(_OCR_TL, "eng", None)
        if eng is None:
            from rapidocr_onnxruntime import RapidOCR
            eng = RapidOCR()
            _OCR_TL.eng = eng
        return eng
    except Exception:                                     # noqa: BLE001
        return fallback


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


def _clip_text_corner(clip_path: Path, ocr_engine) -> str:
    """WHERE the burned text of a cut clip sits: 'tl'|'tr'|'bl'|'br' when it is CORNER-LOCALIZED
    (a channel bug / commenter-avatar badge — croppable), '' when it is centered or frame-wide
    (dialogue subtitles / title cards — not croppable, the caller must fall back to caption
    suppression). Same frame offsets as _clip_has_burned_text; corner votes need the text box
    off-center on BOTH axes (the _detect_logo_corner rule), and any full-width strip vetoes."""
    if ocr_engine is None or not Path(clip_path).exists():
        return ""
    import os as _os2
    import re as _re2
    from collections import Counter
    try:
        from PIL import Image
    except Exception:
        return ""
    ff = ffmpeg_exe()
    votes: Counter = Counter()
    strip_seen = False
    for off in (0.3, 1.0, 1.8, 2.6):
        tmp = f"{clip_path}.tcorner_{int(off * 10)}.jpg"
        try:
            subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{off:.2f}",
                            "-i", str(clip_path), "-frames:v", "1", "-vf", "scale=854:-1", tmp],
                           capture_output=True, timeout=20)
            if not Path(tmp).exists():
                continue
            W, H = Image.open(tmp).size
            res, _el = ocr_engine(tmp)
            for box, txt, conf in (res or []):
                if float(conf) < 0.5 or len(_re2.findall(r"[A-Za-z]", str(txt))) < 4:
                    continue
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                bw = (max(xs) - min(xs)) / max(1, W)
                cx = (min(xs) + max(xs)) / 2 / max(1, W)
                cy = (min(ys) + max(ys)) / 2 / max(1, H)
                if bw > 0.45:
                    strip_seen = True                  # frame-wide strip = subs, never croppable
                elif (cx < 0.3 or cx > 0.7) and (cy < 0.3 or cy > 0.7):
                    votes[("b" if cy > 0.5 else "t") + ("r" if cx > 0.5 else "l")] += 1
        except Exception:
            pass
        finally:
            try:
                _os2.remove(tmp)
            except Exception:
                pass
    if strip_seen or not votes:
        return ""
    return votes.most_common(1)[0][0]


def _crop_clip_corner(clip_path: Path, corner: str, log=None) -> bool:
    """REPAIR a corner-badged cut clip in place: the same punch-in crop the watermark path uses
    (_watermark_crop_filter — keeps 16:9, drops the badge corner), re-encoded at the recut CRF.
    Duration-neutral (crop only). False (original untouched) on any failure."""
    src = Path(clip_path)
    if not src.exists() or corner not in ("tl", "tr", "bl", "br"):
        return False
    out = src.with_name(src.stem + ".dodgecrop" + src.suffix)
    # crop ONLY — the clip already got the _CAS detail chain at cut time; a second sharpen
    # pass would halo. Assemble's per-clip normalize handles the rescale to 1920x1080.
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", _watermark_crop_filter(corner),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "copy", str(out)]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=300)
    except Exception:
        out.unlink(missing_ok=True)
        return False
    if p.returncode == 0 and out.exists() and out.stat().st_size > 0:
        out.replace(src)
        return True
    if log:
        log(f"build: caption-dodge crop failed on {src.name} ({(p.stderr or b'')[-120:]!r})")
    out.unlink(missing_ok=True)
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


def _probe_duration(path) -> float:
    """Container duration in seconds (0.0 if unknown). Cheap ffmpeg header parse — no ffprobe dep."""
    try:
        r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True, timeout=20)
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 0.0


def _scan_coverage_reason(n_frames: int, stride: float, dur: float, rc: int = 0):
    """Count-based coverage check for a fps=1/stride extraction. Returns None when covered, else a
    failure string. The LITERAL guarantee: the LAST sampled frame must reach within ONE stride plus a
    tiny rounding epsilon of the probed duration (NOT max(1s, 2 strides), which could miss the final
    ~1s where an outro/ad slate lives). Frame count within ~2 of expected is a secondary check for
    mid-timeline decode gaps — NOT claimed as proof of mid-timeline content (the caller also PTS-checks
    the tail). An unexpected ffmpeg rc is tolerated ONLY when the frames on disk prove the coverage."""
    if dur <= 0:
        return "could not probe final-video duration — cannot confirm full-timeline coverage"
    if n_frames <= 0:
        return "zero decoded scan frames"
    eps = 0.15
    tol = stride + eps
    last_t = (n_frames - 1) * stride
    expected = int(dur / stride) + 1
    if last_t < dur - tol:
        # The last SAMPLE time can fall one fps-grid step short of the timeline purely by rounding
        # when the video is trimmed to an exact frame-length (the timeline-conform does this to lock
        # A/V sync): a 188.233s video yields 376 samples ending at 187.50s, 0.08s past tol, even
        # though every frame decoded. The frame COUNT is the reliable coverage signal — when it is
        # within one of `expected`, coverage is real and this is a grid artifact, not a decode gap.
        # Only when the count ALSO falls short is the tail genuinely missing.
        if n_frames < expected - 1:
            return (f"partial decode — last sampled frame at {last_t:.2f}s is > one stride "
                    f"({tol:.2f}s) short of the {dur:.2f}s timeline AND only {n_frames}/{expected} "
                    f"frames decoded (final frame not sampled)")
    if n_frames < expected - 2:
        return (f"missing scan frames — {n_frames} decoded but ~{expected} expected @{stride}s "
                f"(mid-timeline decode gap)")
    return None


def _final_timestamp_reachable(result, dur: float, ff, work) -> bool:
    """PTS-level tail check: verify a real frame decodes at ~dur-epsilon (the FINAL timestamp is
    actually present), independent of the fps sample count. Returns False on any decode failure."""
    try:
        t = max(0.0, dur - 0.12)
        tmp = Path(work) / "_tailprobe.png"
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(result),
                            "-frames:v", "1", str(tmp)], capture_output=True, timeout=60)
        ok = (r.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0)
        tmp.unlink(missing_ok=True)
        return ok
    except Exception:
        return False


def _branding_probe_offsets(dur: float) -> list:
    """Sample offsets covering the WHOLE clip — head, middle, and (critically) the TAIL, where
    channel outros / streaming end-slates / CTA cards live. The old fixed head-only grid
    (0.2..3.6s) never sampled a clip's last seconds, so an 8s beat that ran into a Max/WarnerMedia
    outro at ~6.3s was invisible to the gate. Now: 0.2s, then ~every 1.3s through the clip, plus
    three explicit tail samples (dur-0.3 / -1.0 / -1.8). Capped so OCR cost stays bounded."""
    if dur <= 0:
        return [0.2, 0.8, 1.6, 2.6, 3.6]                   # unknown length → legacy head grid
    offs = {0.2}
    t = 1.3
    while t < dur - 0.05:
        offs.add(round(t, 2))
        t += 1.3
    for tail in (dur - 0.3, dur - 1.0, dur - 1.8):         # TAIL coverage — end-slates live here
        if tail > 0.1:
            offs.add(round(tail, 2))
    out = sorted(o for o in offs if 0.05 <= o < dur)
    # bound OCR cost on long clips: keep head, tail, and an even spread of the middle (<= 10 probes)
    if len(out) > 10:
        head, tail = out[:2], out[-3:]
        mids = out[2:-3]
        step = max(1, len(mids) // 5)
        out = sorted(set(head + mids[::step][:5] + tail))
    return out


def _clip_branding_text(clip_path: Path, ocr_engine) -> bool:
    """STRICTER than _clip_has_burned_text: True only when the clip shows a channel branding /
    social-links / CTA slate (must be REMOVED), not a mere in-scene dialogue subtitle (kept).
    Probes the ENTIRE clip window (head + middle + TAIL + implicit shot boundaries), so an outro
    slate that only appears in the clip's last seconds is caught (the Max-slate leak)."""
    if ocr_engine is None or not Path(clip_path).exists():
        return False
    import os as _os2
    ff = ffmpeg_exe()
    for off in _branding_probe_offsets(_probe_duration(clip_path)):
        tmp = f"{clip_path}.brand_{int(off * 100)}.jpg"
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


def _kenburns_hold(prev_clip: Path, dest: Path, duration: float) -> Optional[Path]:
    """Editorial hold with MOTION: the previous clean clip's last frame rendered as a Ken-Burns
    push-in instead of a static freeze. Used when a validated same-scene hold is blocked ONLY by
    the frozen-frame duration caps — a slow push-in is the tool's own sanctioned still treatment
    and never reads as a stuck frame, so the caps (which exist against long FROZEN frames) don't
    apply to it."""
    from .ingest import probe
    d = probe(prev_clip).get("duration", 0.0)
    if d <= 0.2:
        return None
    png = dest.with_suffix(".png")
    try:
        p1 = subprocess.run([ffmpeg_exe(), "-y", "-ss", f"{max(0.0, d - 0.08):.3f}",
                             "-i", str(prev_clip), "-frames:v", "1", str(png)],
                            capture_output=True, timeout=60)
        if p1.returncode != 0 or not png.exists():
            return None
        return _image_kenburns_clip(str(png), dest, duration)
    finally:
        try:
            png.unlink(missing_ok=True)
        except OSError:
            pass


# STRONG full-screen promo / outro / CTA card language — deliberately STRICTER than _BRANDING_RX
# (which contains loose words like "follow"/"watch" that a narration caption can legitimately
# contain). These tokens appear on designed promotional cards, not in scene footage or narration:
# prices, subscription plans, streaming CTAs, channel outros, and streamer-brand slates. The
# final-video gate only fires on one of these AND full-screen-CARD geometry (two-factor), so a
# narration caption that happens to read "watch"/"HBO" over live footage never trips it.
_PROMO_RX = re.compile(
    r"\$\s?\d|\d+\.\d{2}\s*/?\s*mo(nth)?|\bper month\b|/month\b|"
    r"plans?\s*start|starting at\b|free trial|\d+[- ]day free|"
    r"now streaming|stream(ing)?\s+(now|all|it|on|only|this)|only on\b|"
    r"watch\s+(it|them|the full|now on|free)|episodes?\s+(now\s+)?(streaming|available|free)|"
    r"binge|sign up\b|start (your|watching|streaming|the free)|the one to watch|"
    r"subscribe|hit (the|that) (like|bell|subscribe)|like (and|&) subscribe|"
    r"don'?t forget to|new videos? every|link in (the )?(bio|description)|all links|"
    r"hbo\s*max|disney\s?\+|\bnetflix\b|\bhulu\b|prime video|paramount\s?\+|peacock|"
    r"discover more|www\.|\.com\b|\.net\b|/watch\b|@[A-Za-z0-9_]{4,}", re.I)


def _frame_card_uniformity(img_path) -> float:
    """0..1 'designed-card' score for a FINAL-video frame: the fraction of the PICTURE CENTER
    occupied by one near-uniform colour. A promotional/outro/CTA slate is a designed graphic with
    a large flat background (~0.5-1.0); a photographic scene is detailed everywhere (~0.05-0.2).
    Measured on a center crop that EXCLUDES the cinematic letterbox bars and the bottom caption
    band, so neither the bars nor our own burned captions inflate the score."""
    try:
        from PIL import Image
        import numpy as _np2
    except Exception:
        return 0.0
    try:
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        # picture center only: drop top/bottom ~28%/30% (letterbox bars + caption band) and outer 10%
        crop = im.crop((int(0.10 * W), int(0.28 * H), int(0.90 * W), int(0.70 * H)))
        crop = crop.resize((48, 32))
        a = (_np2.asarray(crop, dtype=_np2.uint8) >> 4)           # quantize to 4 bits/channel
        keys = (a[..., 0].astype(_np2.int32) << 8) | (a[..., 1].astype(_np2.int32) << 4) | a[..., 2]
        vals, counts = _np2.unique(keys, return_counts=True)
        return float(counts.max()) / float(keys.size) if keys.size else 0.0
    except Exception:
        return 0.0


def _ocr_layout_metrics(res, W: int, H: int):
    """From an OCR result, (n_confident_boxes, text_area_frac, max_box_area_frac). A designed promo
    card — even an IMAGE-BACKED one that isn't flat — carries large and/or many text boxes; an
    in-scene sign or a lone subtitle is 1-2 small boxes."""
    import re as _re3
    area = float(max(1, W * H))
    n, tot, mx = 0, 0.0, 0.0
    for box, txt, conf in (res or []):
        try:
            if float(conf) < 0.30 or len(_re3.findall(r"[A-Za-z0-9]", str(txt))) < 2:
                continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            a = (max(xs) - min(xs)) * (max(ys) - min(ys))
            n += 1
            tot += a
            mx = max(mx, a)
        except Exception:
            continue
    return n, tot / area, mx / area


# ── OWN-CAPTION whitelist for the final-video ad scan ────────────────────────────────────────
# The gate scans the FINISHED video, which legitimately carries OUR OWN burned captions — and a
# narration script may itself end on a CTA ("…subscribe because that's the story coming next").
# That word is the USER'S OWN SCRIPT, not third-party promo material, yet it matches _PROMO_RX
# and a large caption style trips the layout-heavy geometry (observed: a finished 4.3h render
# quarantined on its own outro caption, deterministically on every retry). The precise defence is
# the caption SCHEDULE: a text box is ignored only when its words match what WE burned at that
# very timestamp. A real promo card's text is not in the schedule, so detection is unweakened.

_CAPWORD_RX = re.compile(r"[a-z0-9']+")


def _norm_caption_words(text: str) -> list:
    """Normalized word list for caption-vs-OCR comparison: lowercase, curly quotes/dashes folded,
    everything but [a-z0-9'] dropped (OCR reads '—' as '-', smart quotes as ASCII, etc.)."""
    t = str(text or "").lower()
    t = (t.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"'))
    return _CAPWORD_RX.findall(t)


def _parse_srt_events(path) -> list:
    """[(t0, t1, text)] from an SRT file. Tolerant: returns [] on any read/parse failure —
    the caller then simply has no whitelist (fail-safe: the gate stays as strict as before)."""
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    ts_rx = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
    evs, cur, buf = [], None, []
    for line in txt.splitlines() + [""]:
        m = ts_rx.search(line)
        if m:
            if cur is not None and buf:
                evs.append((cur[0], cur[1], " ".join(buf)))
            g = [int(x) for x in m.groups()]
            cur = (g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0,
                   g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0)
            buf = []
        elif not line.strip():
            if cur is not None and buf:
                evs.append((cur[0], cur[1], " ".join(buf)))
            cur, buf = None, []
        elif cur is not None:
            # every non-blank line after a timestamp is cue TEXT — including digit-only lines
            # (a caption that is just a year, '1942', must stay in the whitelist schedule; SRT
            # index lines only ever appear before a timestamp, where cur is None).
            buf.append(line.strip())
    return evs


def _parse_ass_events(path) -> list:
    """[(t0, t1, text)] from an ASS subtitle file (the breakout word-by-word caption overlay).
    Strips {\\kf..}/{\\fad..} override tags; \\N becomes a space. [] on any failure."""
    try:
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    tag_rx = re.compile(r"\{[^}]*\}")

    def _ts(s):
        try:
            hh, mm, rest = s.strip().split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(rest)
        except Exception:
            return None
    evs = []
    for line in txt.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        t0, t1 = _ts(parts[1]), _ts(parts[2])
        if t0 is None or t1 is None:
            continue
        text = tag_rx.sub("", parts[9]).replace(r"\N", " ").replace(r"\n", " ").strip()
        if text:
            evs.append((t0, t1, text))
    return evs


def _own_caption_schedule(result: Path, work: Path) -> list:
    """[(t0, t1, [words], raw_text)] for every caption WE burned onto the final video:
    the narration captions (final.srt — written by the engine from the same word timings the
    caption burner renders) plus the breakout word-by-word lines (work/breakout_caps.ass — the
    exact burned text). Empty list when neither exists (→ no whitelisting)."""
    evs = []
    try:
        srt = Path(result).with_suffix(".srt")
        if srt.exists():
            evs.extend(_parse_srt_events(srt))
        ass = Path(work) / "breakout_caps.ass"
        if ass.exists():
            evs.extend(_parse_ass_events(ass))
    except Exception:
        return []
    out = []
    for t0, t1, text in evs:
        ws = _norm_caption_words(text)
        if ws:
            out.append((float(t0), float(t1), ws, text))
    return out


def _caption_explained(box_text: str, active_events: list) -> bool:
    """True when an OCR box's text is (fuzzily) covered by ONE caption event we burned in the
    surrounding seconds — and covers enough OF that event to actually BE the caption render (or
    one full row of it), not a promo element that merely SHARES words with it.

    TWO-WAY coverage against each active event's word list (never a flattened union):
      forward — >= 80% (ceil) of the box's words appear in the event (exact, or difflib>=0.8 for
                words of 4+ chars: OCR noise like 'subscrlbe' / 'thats');
      reverse — the box matches at least min(3, len(event_words)) DISTINCT event words.
    The reverse requirement is what stops the subset bypass: a full-screen 'SUBSCRIBE' end-card
    button while our caption reads 'subscribe because that's the story' shares 100% of ITS one
    word but covers only 1 of the event's 7 — never explained. A two-row caption render still
    passes (each row carries >= 3 of its event's words). Residual: a promo element that
    reproduces >= 3 consecutive words of our exact caption line at the exact moment it airs is
    indistinguishable from the caption by text alone — accepted (the flat-card path and the
    upstream clip-stage branding scans still apply to it)."""
    words = _norm_caption_words(box_text)
    if not words:
        return False
    import difflib
    import math
    need_fwd = max(1, math.ceil(0.8 * len(words)))
    for ev_words in (active_events or []):
        if not ev_words:
            continue
        fwd, matched = 0, set()
        for w in words:
            if w in ev_words:
                fwd += 1
                matched.add(w)
            elif len(w) >= 4:
                close = difflib.get_close_matches(w, ev_words, n=1, cutoff=0.8)
                if close:
                    fwd += 1
                    matched.add(close[0])
        if fwd >= need_fwd and len(matched) >= min(3, len(set(ev_words))):
            return True
    return False


def _final_video_ad_scan(result: Path, work: Path, ocr_engine, *, log=None,
                         stride: float = 0.5, own_captions: list = None) -> dict:
    """Scan the FINISHED video every `stride` seconds (0.5s — a 1-2s card cannot fit between probes)
    for full-screen promotional / outro / CTA / streamer-brand slates that must never ship.

    FAIL-CLOSED: returns {"status": clean|blocked|unverified, "hits": [...], "frames", "ocr_errors",
    "reason"}. status 'unverified' means the scan could not run (no OCR engine, zero decoded frames,
    or excessive OCR errors) — the GATE treats that as a block unless an explicit override is set.

    TWO detection paths so an IMAGE-BACKED (non-uniform) promo card is caught too:
      A) flat CARD: a strong promo token (_PROMO_RX) AND a near-uniform designed background.
      B) LAYOUT-HEAVY: a strong promo token AND large/abundant on-screen text (big or many boxes),
         even over a photographic background (pricing/URL/subscribe overlays on a movie still).
    A candidate is CONFIRMED as a hit when it persists across >=2 consecutive samples (a real card
    holds >= ~1s) OR a single frame is an unambiguous flat card (card >= strong). In-scene signs,
    burned subtitles and corner bugs are protected by the two-factor geometry; our OWN burned
    captions are protected by `own_captions` — a text box is dropped only when it two-way-matches
    a caption event WE scheduled at that very timestamp (see _caption_explained), never by
    discarding the bottom band. A transient single frame never confirms unless it is a strong
    flat card."""
    import os as _os2
    scan_dir = work / "_adscan"
    if ocr_engine is None:
        return {"status": "unverified", "hits": [], "frames": 0, "ocr_errors": 0,
                "reason": "no OCR engine available — cannot verify the final video is ad-free"}
    ff = ffmpeg_exe()
    from .config import _f as _cfg_f3
    card_floor = _cfg_f3("VIDLORE_CLIPSTUDIO_AD_CARD_FLOOR", 0.40)
    card_strong = _cfg_f3("VIDLORE_CLIPSTUDIO_AD_CARD_STRONG", 0.55)
    from PIL import Image
    _caps_sched = list(own_captions or [])
    _cap_excluded = [0]                                    # boxes whitelisted as our own captions

    def _active_caption_events(t):
        """Per-EVENT word lists of the captions WE burned that are active at frame-time t
        (±1.0s slack — the fps-filter frame time is only stride-accurate, and caption fades
        straddle event edges). Kept per-event, never flattened: _caption_explained requires the
        box to cover a real fraction of ONE event, which a promo element sharing stray words
        with several events cannot fake."""
        return [ev[2] for ev in _caps_sched if ev[0] - 1.0 <= t <= ev[1] + 1.0]

    import threading as _thr_scan
    _cap_lock = _thr_scan.Lock()

    def _probe_frame(fp, t=None, eng=None):
        """OCR ONE frame → a promo-candidate dict or None. OCRs the FULL frame (promo URLs/prices
        live at the very bottom too) — our own captions are protected by matching each text box
        against the caption SCHEDULE (what we burned at this very timestamp), never by discarding
        the bottom band. Card-uniformity is measured on the picture area. `eng` lets the parallel
        sweep pass a per-thread engine (identical construction → identical output)."""
        card = _frame_card_uniformity(fp)
        im = Image.open(fp).convert("RGB")
        W, H = im.size
        res, _el = (eng if eng is not None else ocr_engine)(str(fp))
        res = list(res or [])
        if _caps_sched and t is not None:
            act = _active_caption_events(t)
            if act:
                kept = []
                for item in res:
                    try:
                        if (float(item[2]) >= 0.30
                                and _caption_explained(str(item[1]), act)):
                            with _cap_lock:
                                _cap_excluded[0] += 1
                            continue
                    except Exception:
                        pass
                    kept.append(item)
                res = kept
        joined = " ".join(str(txt) for _b, txt, conf in res if float(conf) >= 0.30)
        m = _PROMO_RX.search(joined)
        if not m:
            return None
        n_box, area_frac, max_frac = _ocr_layout_metrics(res, W, H)
        is_card = card >= card_floor
        layout_heavy = (area_frac >= 0.06) or (n_box >= 5) or (max_frac >= 0.04)
        if not (is_card or layout_heavy):
            return None
        return {"token": m.group(0)[:40], "text": joined[:160], "card": round(card, 3),
                "n_box": n_box, "text_area": round(area_frac, 3),
                "path": "flat_card" if is_card else "layout_heavy",
                "strong_single": bool(card >= card_strong or (layout_heavy and max_frac >= 0.10))}

    try:
        scan_dir.mkdir(exist_ok=True)
        for _old in scan_dir.glob("*.jpg"):
            _old.unlink(missing_ok=True)
    except Exception:
        pass
    # COMPLETE-DURATION coverage: probe the video length and confirm the 0.5s extraction reaches the
    # final timestamp (a partial decode that yields SOME frames is UNVERIFIED, not clean).
    dur = _probe_duration(result)
    fps = 1.0 / max(0.1, stride)
    _p = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(result),
                         "-vf", f"fps={fps:.4f},scale=854:-1", "-q:v", "4",
                         str(scan_dir / "fr_%05d.jpg")], capture_output=True, timeout=1800)
    frames = sorted(scan_dir.glob("fr_*.jpg"))
    _cov = _scan_coverage_reason(len(frames), stride, dur, _p.returncode)
    if _cov is None and not _final_timestamp_reachable(result, dur, ff, scan_dir):
        _cov = f"final timestamp (~{dur:.1f}s) does not decode — tail not covered (fail-closed)"
    if _cov is not None:
        return {"status": "unverified", "hits": [], "frames": len(frames), "ocr_errors": 0,
                "reason": _cov}
    cand = {}                                          # t -> candidate dict
    ocr_errors = 0
    # SPEED: frames judge in a bounded pool with per-thread engines (bit-identical output —
    # see _ocr_engine_tl). Aggregation is order-independent by construction: `cand` is keyed
    # by t and every consumer walks sorted(cand.items()); ocr_errors sums exactly the frames
    # a serial walk would have counted. VIDLORE_CLIPSTUDIO_ADSCAN_WORKERS=1 restores serial.
    try:
        _aw = int(_os2.environ.get("VIDLORE_CLIPSTUDIO_ADSCAN_WORKERS", "4") or 4)
    except (TypeError, ValueError):
        _aw = 1

    def _judge_one(i, fp):
        t = round(i * stride, 2)
        try:
            c = _probe_frame(fp, t, eng=_ocr_engine_tl(ocr_engine))
            return (t, c, 0)
        except Exception:                              # noqa: BLE001
            return (t, None, 1)

    if _aw > 1 and len(frames) > 16:
        import concurrent.futures as _cf_scan
        with _cf_scan.ThreadPoolExecutor(max_workers=min(_aw, 8)) as _ex_scan:
            for t, c, err in _ex_scan.map(_judge_one, range(len(frames)), frames):
                ocr_errors += err
                if c is not None:
                    c["t"] = t
                    cand[t] = c
    else:
        for i, fp in enumerate(frames):
            t = round(i * stride, 2)
            try:
                c = _probe_frame(fp, t)
                if c is not None:
                    c["t"] = t
                    cand[t] = c
            except Exception:
                ocr_errors += 1
                continue
    try:
        for _f in scan_dir.glob("*.jpg"):
            _f.unlink(missing_ok=True)
    except Exception:
        pass
    if frames and ocr_errors / len(frames) > 0.25:
        return {"status": "unverified", "hits": [], "frames": len(frames), "ocr_errors": ocr_errors,
                "reason": f"excessive OCR errors ({ocr_errors}/{len(frames)}) — cannot verify"}

    def _dense_confirm(t0):
        """A LONE strong candidate is NOT discarded as transient — rescan densely (~0.1s) around it.
        TRI-STATE: 'confirmed' (>=2 dense samples are promo), 'clean' (dense samples decoded but not
        promo), 'unverified' (dense extraction/OCR failed — cannot judge → the whole scan must block)."""
        ddir = scan_dir / "dense"
        try:
            ddir.mkdir(exist_ok=True)
            for _o in ddir.glob("*.jpg"):
                _o.unlink(missing_ok=True)
        except Exception:
            pass
        _r = subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{max(0.0, t0 - 0.5):.2f}",
                             "-t", "1.0", "-i", str(result), "-vf", "fps=10,scale=854:-1", "-q:v", "4",
                             str(ddir / "d_%03d.jpg")], capture_output=True, timeout=120)
        dframes = sorted(ddir.glob("d_*.jpg"))
        if _r.returncode != 0 or len(dframes) < 6:      # ~10 expected over 1.0s; too few → unverified
            for _o in ddir.glob("*.jpg"):
                _o.unlink(missing_ok=True)
            return "unverified"
        hit = derr = 0
        _d0 = max(0.0, t0 - 0.5)
        for _di, dp in enumerate(dframes):
            try:
                if _probe_frame(dp, round(_d0 + _di * 0.1, 2)) is not None:
                    hit += 1
            except Exception:
                derr += 1
        for _o in ddir.glob("*.jpg"):
            _o.unlink(missing_ok=True)
        if derr > len(dframes) * 0.34:
            return "unverified"
        return "confirmed" if hit >= 2 else "clean"

    hits = []
    for t, c in sorted(cand.items()):
        consec = (round(t - stride, 2) in cand) or (round(t + stride, 2) in cand)
        if consec or c["strong_single"]:
            hits.append(c)
        elif c["path"] == "layout_heavy" or c["card"] >= card_floor:
            # a lone strong-ish candidate → dense 0.1s rescan (tri-state)
            _dv = _dense_confirm(t)
            if _dv == "unverified":
                return {"status": "unverified", "hits": hits, "frames": len(frames),
                        "ocr_errors": ocr_errors,
                        "reason": f"dense rescan around a real promo candidate @{t}s failed "
                                  f"(extraction/OCR) — cannot rule it out (fail-closed)"}
            if _dv == "confirmed":
                c["confirmed_by"] = "dense_rescan"
                hits.append(c)
    if log:
        if _cap_excluded[0]:
            log(f"build: final-video ad scan — {_cap_excluded[0]} text box(es) matched OUR OWN "
                f"burned caption schedule (narration/breakout lines) and were whitelisted; "
                f"promo detection ran on the remaining screen text")
        if hits:
            for h in hits[:12]:
                log(f"build: final-video AD SCAN HIT @{h['t']}s token={h['token']!r} "
                    f"path={h['path']} card={h['card']} boxes={h['n_box']}")
        else:
            log(f"build: final-video ad scan clean — {len(frames)} frames @{stride}s "
                f"(full {dur:.0f}s timeline, last @{(len(frames) - 1) * stride:.0f}s), "
                f"{len(cand)} promo candidate(s) (none confirmed)")
    return {"status": ("blocked" if hits else "clean"), "hits": hits,
            "frames": len(frames), "ocr_errors": ocr_errors, "reason": ""}


def _final_video_ad_gate(result: Path, work: Path, ocr_engine, *, log,
                         captions_burned: bool = False) -> Path:
    """HARD, FAIL-CLOSED publication gate against full-screen promo/outro/CTA/streamer-brand cards.
    Auto-repair happens UPSTREAM at the clip stage (_clip_branding_text full-window probe →
    _freeze_replace / clean re-window + Ken-Burns time-neutrality). A survivor here — OR an inability
    to VERIFY the render is clean — quarantines the render (*.FAILED_AD_QA.*) and RAISES. A
    verification failure (no OCR / zero frames / excessive OCR errors) can only be waved through with
    an explicit emergency override (VIDLORE_CLIPSTUDIO_AD_GATE_OVERRIDE=1) that logs a LOUD warning.

    `captions_burned=True` (the caller burned word-synced captions onto this video) enables the
    OWN-CAPTION whitelist: OCR text matching the burned caption schedule (final.srt + breakout ASS)
    at that timestamp is the user's own script, not third-party promo. With captions off, no
    whitelist — screen text then can only come from the footage itself."""
    import json as _json_ad
    import os as _os3
    if _os3.environ.get("VIDLORE_CLIPSTUDIO_FINAL_AD_GATE", "1").strip() in ("0", "false", "no"):
        return result
    own = _own_caption_schedule(result, work) if captions_burned else None
    if own:
        _cta = sorted({m.group(0).strip().lower() for _a, _b, _w, _raw in own
                       for m in [_PROMO_RX.search(_raw)] if m})[:6]
        if _cta:
            log(f"build: ad-gate note — the narration/breakout captions themselves contain "
                f"CTA-like language {_cta} (the user's own script); these caption lines are "
                f"whitelisted by text+time match, NOT by loosening promo detection")
    r = _final_video_ad_scan(result, work, ocr_engine, log=log, own_captions=own)
    status = r.get("status")
    if status == "clean":
        return result
    if status == "unverified":
        if _os3.environ.get("VIDLORE_CLIPSTUDIO_AD_GATE_OVERRIDE", "0").strip() in ("1", "true", "yes"):
            log(f"build: ⚠️⚠️ AD-GATE EMERGENCY OVERRIDE — could not verify the final video is ad-free "
                f"({r.get('reason')}); PUBLISHING ANYWAY by explicit override. This render was NOT "
                f"scanned for promo/outro/CTA material.")
            return result
        # fail closed: cannot verify → do not publish
    try:
        (work.parent / "final_ad_failures.json").write_text(
            _json_ad.dumps(r, indent=1), encoding="utf-8")
    except Exception:
        pass
    _quar = result.with_name(result.stem + ".FAILED_AD_QA" + result.suffix)
    try:
        if _quar.exists():
            _quar.unlink()
        result.rename(_quar)
    except Exception:
        _quar = result
    if status == "blocked":
        log(f"build: ⛔ RELEASE-BLOCKED — {len(r['hits'])} promo/outro/CTA frame(s) survived to the "
            f"final video (first @{r['hits'][0]['t']}s, {r['hits'][0]['token']!r}, "
            f"path={r['hits'][0]['path']}); quarantined → {_quar.name}. See final_ad_failures.json.")
        raise RuntimeError(
            f"final-video ad gate failed — {len(r['hits'])} promo/outro/CTA frame(s) survived "
            f"(quarantined at {_quar.name}); refusing to publish third-party promo material")
    log(f"build: ⛔ RELEASE-BLOCKED (fail-closed) — could NOT verify the final video is ad-free "
        f"({r.get('reason')}); quarantined → {_quar.name}. Set AD_GATE_OVERRIDE=1 to force-publish.")
    raise RuntimeError(
        f"final-video ad gate could not verify the render ({r.get('reason')}) — failing closed and "
        f"refusing to publish (quarantined at {_quar.name}); set AD_GATE_OVERRIDE=1 to override")


def _frame_luma_hi(img_path):
    """~99.5th-percentile luma of a frame (the same 'brightest legible content' signal the index's
    luma_hi uses). A frame whose brightest pixels are still dim has no legible content. Returns None
    on an explicit probe/decode FAILURE (the caller must treat that as unverified, NOT as bright)."""
    try:
        from PIL import Image
        import numpy as _np2
        if not Path(img_path).exists() or Path(img_path).stat().st_size == 0:
            return None
        g = _np2.asarray(Image.open(img_path).convert("L"), dtype=_np2.uint8)
        if g.size == 0:
            return None
        return float(_np2.percentile(g, 99.5))
    except Exception:
        return None                                    # explicit failure → unverified


_IMAGE_MIN_SHORT_EDGE = 720
_IMAGE_MIN_LONG_EDGE = 1280


def _image_pixel_dimensions(path) -> tuple[int, int]:
    """Decode an image and return its real pixel dimensions; metadata/extension is not proof."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
            return int(im.width), int(im.height)
    except Exception:                                    # noqa: BLE001 — caller fails closed
        return 0, 0


def _publishable_still_pixels(width: int, height: int) -> bool:
    """True for a native 1280x720-or-better still in either orientation.

    A narrow 400x1000 image has a nominal 720+ height but still needs a destructive upscale to fill
    a video frame.  Requiring both a 720 short edge and 1280 long edge admits landscape and portrait
    originals symmetrically and never mistakes an enlarged output container for source detail.
    """
    try:
        short, long = sorted((int(width), int(height)))
    except (TypeError, ValueError, OverflowError):
        return False
    return short >= _IMAGE_MIN_SHORT_EDGE and long >= _IMAGE_MIN_LONG_EDGE


def _image_file_sha256(path) -> str:
    import hashlib
    try:
        p = Path(path)
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
    except Exception:                                    # noqa: BLE001 — missing proof fails later
        return ""


def _source_frame_meta(sel) -> tuple[dict, bool]:
    meta = dict(getattr(sel, "image_meta", None) or {})
    return meta, "source-frame" in str(meta.get("source", ""))


def _resolve_indexed_still_owner(proj, sel) -> Optional[dict]:
    """Resolve ``image_meta.src`` + ``shot`` against the persisted index.

    ``None`` means both ownership fields are absent (an older verified-file record).  A partial,
    malformed, or stale ownership claim is an error: it must never quietly become ``sel.source_id``
    or ``sel.in_point``, because those describe the rejected moving selection, not the still.
    """
    import json
    meta, is_source_frame = _source_frame_meta(sel)
    if not is_source_frame:
        return None
    raw_sid, raw_shot = meta.get("src"), meta.get("shot")
    if raw_sid in (None, "") and raw_shot in (None, ""):
        return None
    from .verify import NonRetryableBuildError
    if raw_sid in (None, "") or raw_shot in (None, ""):
        raise NonRetryableBuildError(
            "image-lineage gate: source-frame ownership metadata is partial; both image_meta.src "
            "and image_meta.shot are required", kind="scene_lineage")
    sid = str(raw_sid)
    try:
        shot_index = int(raw_shot)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NonRetryableBuildError(
            f"image-lineage gate: source-frame shot id {raw_shot!r} is malformed",
            kind="scene_lineage") from exc
    index_path = Path(proj.index_dir) / f"{sid}.shots.json"
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        row = next(d for d in data if int(d.get("index", -1)) == shot_index)
        if row.get("source_id") not in (None, "", sid):
            raise ValueError(f"shot row belongs to {row.get('source_id')!r}, not {sid!r}")
        start, end = float(row["start"]), float(row["end"])
    except Exception as exc:                             # noqa: BLE001 — bad claim fails closed
        raise NonRetryableBuildError(
            f"image-lineage gate: image_meta claims {sid!r} shot {shot_index}, but that exact "
            f"shot is absent/unreadable in {index_path.name}", kind="scene_lineage") from exc
    if not (end > start >= 0.0):
        raise NonRetryableBuildError(
            f"image-lineage gate: indexed owner {sid!r} shot {shot_index} has invalid bounds "
            f"{start:.3f}-{end:.3f}", kind="scene_lineage")
    raw_keyframe = str(row.get("keyframe_path") or "")
    keyframe = Path(raw_keyframe).expanduser()
    if raw_keyframe and not keyframe.is_absolute():
        keyframe = index_path.parent / keyframe
    declared = _require_declared_image_path(sel)
    src = proj.source(sid)
    src_path = Path(str(getattr(src, "local_path", "") or "")) if src else Path("")
    if not src or not src_path.is_file() or src_path.stat().st_size <= 0:
        raise NonRetryableBuildError(
            f"image-lineage gate: indexed owner source {sid!r} is missing/empty; refusing to use "
            f"selection source {getattr(sel, 'source_id', '')!r} instead", kind="scene_lineage")
    # The vision/relevance verifier approved the indexed thumbnail.  Re-extracting the claimed
    # source midpoint is only safe when that approved file is demonstrably THIS shot's keyframe;
    # otherwise valid-looking but wrong src/shot metadata could redirect the aired image. A native
    # image already materialized and strictly judged by the pre-assembly recovery lane is the one
    # explicit exception: it carries the original keyframe hash, owner-source fingerprint, exact
    # midpoint and its own semantic SHA, all of which are rechecked here.
    keyframe_hash = _image_file_sha256(keyframe)
    declared_hash = _image_file_sha256(declared)
    same_as_indexed = bool(
        raw_keyframe and keyframe.is_file() and keyframe.stat().st_size > 0
        and (keyframe.resolve() == declared.resolve() or keyframe_hash == declared_hash))
    native_materialized = meta.get("native_semantic_materialized") is True
    native_ok = False
    if not same_as_indexed and native_materialized:
        from .verify import _file_fingerprint
        midpoint = round((start + end) / 2.0, 3)
        try:
            time_ok = abs(float(meta.get("native_owner_time")) - midpoint) <= 0.001
        except (TypeError, ValueError, OverflowError):
            time_ok = False
        source_fp = _file_fingerprint(src_path)
        native_ok = bool(
            keyframe_hash
            and str(meta.get("native_indexed_keyframe_sha256") or "") == keyframe_hash
            and source_fp not in ("", "missing", "unreadable")
            and str(meta.get("native_owner_source_content_fingerprint") or "") == source_fp
            and time_ok and declared_hash
            and str(meta.get("still_image_sha256") or "") == declared_hash
            and str(meta.get("native_semantic_question_fingerprint") or "")
            and str(meta.get("native_semantic_model") or "") not in ("", "none"))
    if not same_as_indexed and not native_ok:
        raise NonRetryableBuildError(
            f"image-lineage gate: declared verified still does not match indexed keyframe for "
            f"{sid!r} shot {shot_index}", kind="scene_lineage")
    # Index keyframes are extracted at the exact shot midpoint (index.py).  Repeating that rule
    # makes the full-resolution frame deterministic and keeps it tied to the verified still.
    return {"source_id": sid, "shot_index": shot_index,
            "time": round((start + end) / 2.0, 3), "start": start, "end": end,
            "source_path": str(src_path.resolve()),
            "keyframe_path": str(keyframe.resolve()),
            "keyframe_sha256": keyframe_hash}


def _probe_image_owner_source(path) -> tuple[int, int]:
    try:
        from .ingest import probe
        info = probe(Path(path)) or {}
        return int(info.get("width") or 0), int(info.get("height") or 0)
    except Exception:                                    # noqa: BLE001 — unknown fails closed
        return 0, 0


def _require_declared_image_path(sel) -> Path:
    """Return a nonempty declared image or block; never fall through to rejected moving footage."""
    raw = str(getattr(sel, "image_path", "") or "")
    if not raw:
        raise ValueError("selection does not declare an image")
    p = Path(raw)
    if not p.is_file() or p.stat().st_size <= 0:
        from .verify import NonRetryableBuildError
        raise NonRetryableBuildError(
            f"image-lineage gate: beat {getattr(sel, 'segment_index', '?')} declares verified "
            f"image {raw!r}, but it is missing/empty; refusing to fall through to moving footage",
            kind="scene_lineage")
    return p


def _still_owner_source_fingerprint(owner: dict) -> str:
    """Return a usable owner-byte identity; sentinel hashes can never bind publication proof."""
    from .verify import NonRetryableBuildError, _file_fingerprint

    source_fp = _file_fingerprint(owner.get("source_path") or "")
    if source_fp in ("", "missing", "unreadable"):
        raise NonRetryableBuildError(
            f"image-lineage gate: indexed owner {owner.get('source_id')!r} has no readable "
            "source-content fingerprint", kind="scene_lineage")
    return source_fp


def _require_unchanged_still_source(owner: dict, expected_fingerprint: str) -> None:
    """Fail if extraction/judging raced a source rewrite or an unreadable owner."""
    from .verify import NonRetryableBuildError

    current = _still_owner_source_fingerprint(owner)
    if not expected_fingerprint or current != expected_fingerprint:
        raise NonRetryableBuildError(
            f"image-lineage gate: indexed owner {owner.get('source_id')!r} changed while its "
            "native still was being materialized or verified", kind="scene_lineage")


def _owned_still_binding(owner: dict, declared: Path, *, source_fingerprint: str = "") -> dict:
    """Immutable inputs that tie an early full-resolution extraction to its indexed owner."""
    source_fp = source_fingerprint or _still_owner_source_fingerprint(owner)
    if source_fp in ("", "missing", "unreadable"):
        # An explicit sentinel must not bypass `_still_owner_source_fingerprint` via the override.
        source_fp = _still_owner_source_fingerprint(owner)

    return {
        "declared_image_sha256": _image_file_sha256(declared),
        "indexed_keyframe_sha256": str(owner.get("keyframe_sha256") or ""),
        "owner_source_content_fingerprint": source_fp,
        "owner_source_id": str(owner.get("source_id") or ""),
        "owner_shot_index": int(owner.get("shot_index", -1)),
        "owner_time": float(owner.get("time", -1.0)),
    }


def _semantic_still_question_fingerprint(proj, seg, owner: dict, *, image_sha256: str,
                                         model: str, source_fingerprint: str = "") \
        -> tuple[str, str]:
    """Fingerprint the exact full-resolution still question and its source bytes."""
    from . import policy as _policy_still
    from .verify import (
        _project_beat_era,
        _project_exact_cast_warning,
        effective_deictic_target,
        verdict_fingerprint,
    )

    source_fp = source_fingerprint or _still_owner_source_fingerprint(owner)
    if source_fp in ("", "missing", "unreadable"):
        source_fp = _still_owner_source_fingerprint(owner)
    era = _project_beat_era(proj, seg)
    is_specific = _policy_still.policy_of(seg) == _policy_still.EXACT
    must_see = effective_deictic_target(seg)
    cast_warning = _project_exact_cast_warning(
        proj, seg, str(owner.get("source_id") or "")) if is_specific else ""
    question_fp = verdict_fingerprint(
        src_hash=source_fp,
        source_id=str(owner.get("source_id") or ""),
        shot_start=float(owner.get("start", 0.0) or 0.0),
        shot_end=float(owner.get("end", 0.0) or 0.0),
        beat_text=getattr(seg, "text", "") or "",
        required_entity=getattr(seg, "required_entity", "") or "",
        required_kind=getattr(seg, "required_kind", "") or "",
        expected_visual=getattr(seg, "expected_visual", "") or "",
        scene_query=getattr(seg, "scene_query", "") or "",
        era=era,
        visual_policy=_policy_still.policy_of(seg),
        is_specific=is_specific,
        faceid_names=(),
        multiframe=False,
        image_id=f"sha256:{image_sha256}",
        model=str(model or ""),
        venue_fallback=False,
        must_see=must_see,
        exact_cast_warning=cast_warning,
    )
    return question_fp, source_fp


def _persisted_native_still_semantic_reason(
        proj, seg, owner: dict, meta: dict, *, image_sha256: str,
        source_fingerprint: str = "") -> str:
    """Why persisted rematerialized pixels no longer prove the current beat, or ``""``.

    Shared by the early relevance contract and the final build invariant so an audit cannot say
    CLEAR for bytes that assembly will later reject.  Original native indexed keyframes do not
    carry ``native_semantic_materialized`` and are governed by their ordinary exact-byte verdict.
    """
    if meta.get("native_semantic_materialized") is not True:
        return ""
    from .relevance_contract import strict_still_evidence_reason

    source_fp = source_fingerprint or _still_owner_source_fingerprint(owner)
    persisted_verdict = dict(meta.get("still_verifier") or {})
    persisted_model = str(meta.get("native_semantic_model") or "")
    expected_qfp, expected_source_fp = _semantic_still_question_fingerprint(
        proj, seg, owner, image_sha256=image_sha256,
        model=persisted_model, source_fingerprint=source_fp)
    stale_reasons = []
    verdict_reason = strict_still_evidence_reason(persisted_verdict, seg)
    if verdict_reason:
        stale_reasons.append(verdict_reason)
    if not persisted_model or persisted_model == "none":
        stale_reasons.append("served model identity is absent")
    if str(persisted_verdict.get("vision_served_by") or "") != persisted_model:
        stale_reasons.append("verdict model does not match persisted model")
    if str(meta.get("native_semantic_question_fingerprint") or "") != expected_qfp:
        stale_reasons.append("beat-question fingerprint changed")
    if str(meta.get("native_owner_source_content_fingerprint") or "") != expected_source_fp:
        stale_reasons.append("owner-source fingerprint changed")
    from .verify import (_cast_warning_resolution_reason, _project_char2actor,
                         _project_exact_cast_warning)
    cast_warning = _project_exact_cast_warning(
        proj, seg, str(owner.get("source_id") or ""))
    cast_resolution = (_cast_warning_resolution_reason(
        persisted_verdict, seg, _project_char2actor(proj)) if cast_warning else "")
    if cast_resolution:
        stale_reasons.append(
            "source-title cast warning is not resolved by the persisted native-pixel verdict "
            f"({cast_resolution})")
    return "; ".join(stale_reasons)


def _strictly_verify_native_still(proj, sel, seg, owner: dict, image_path: Path, eng,
                                  *, source_fingerprint: str,
                                  allow_content_reject: bool = False) -> dict:
    """Freshly judge the exact native pixels; thumbnail verdicts are never transferred."""
    from . import policy as _policy_still
    from . import relevance_contract as _relevance_still
    from . import verify as _verify_still
    from .selfheal import _still_verdict_schema_error
    from .verify import NonRetryableBuildError, VisionBackendError

    era = _verify_still._project_beat_era(proj, seg)
    is_specific = _policy_still.policy_of(seg) == _policy_still.EXACT
    must_see = _verify_still.effective_deictic_target(seg)
    cast_warning = (_verify_still._project_exact_cast_warning(
        proj, seg, str(owner.get("source_id") or "")) if is_specific else "")
    char2actor = _verify_still._project_char2actor(proj)
    _require_unchanged_still_source(owner, source_fingerprint)
    image_hash_before = _image_file_sha256(image_path)
    if not image_hash_before:
        raise NonRetryableBuildError(
            "image-lineage gate: native still bytes vanished before semantic verification",
            kind="scene_lineage")
    try:
        verdict = _verify_still.verify_frame(
            str(image_path), getattr(seg, "text", "") or "",
            getattr(seg, "required_entity", "") or "",
            getattr(seg, "required_kind", "") or "", [], eng,
            getattr(eng, "anthropic_model", ""), is_specific=is_specific,
            expected_visual=getattr(seg, "expected_visual", "") or "",
            scene_query=getattr(seg, "scene_query", "") or "", era_hint=era,
            venue_fallback=False, must_see=must_see,
            exact_cast_warning=cast_warning)
    except (NonRetryableBuildError, VisionBackendError):
        raise
    except Exception as exc:  # noqa: BLE001 — a transport/backend exception is not a verdict
        raise VisionBackendError(
            f"native still verifier failed before judging beat "
            f"{getattr(seg, 'index', '?')}: {type(exc).__name__}", kind="down") from exc
    if not isinstance(verdict, dict):
        raise VisionBackendError(
            f"native still verifier returned no valid verdict for beat "
            f"{getattr(seg, 'index', '?')}; thumbnail evidence was not transferred", kind="down")
    schema_error = _still_verdict_schema_error(
        verdict, seg, require_keep_facts=True)
    if schema_error:
        raise VisionBackendError(
            f"native still verifier returned inconclusive status/schema for beat "
            f"{getattr(seg, 'index', '?')} ({schema_error})", kind="down")
    if (cast_warning and verdict.get("verdict") == "keep"
            and not isinstance(verdict.get("source_title_conflict_resolved"), bool)):
        raise VisionBackendError(
            f"native still verifier omitted source-title conflict resolution for beat "
            f"{getattr(seg, 'index', '?')}", kind="down")
    verdict = dict(verdict)
    verdict["status"] = "ok"
    why = _relevance_still.strict_still_evidence_reason(verdict, seg)
    cast_resolution = (_verify_still._cast_warning_resolution_reason(
        verdict, seg, char2actor) if cast_warning else "")
    if cast_resolution:
        why = ("source-title cast warning was not resolved from the native pixels: "
               f"{cast_warning} ({cast_resolution})")
    if why:
        explicit_negative = (
            verdict.get("verdict") == "replace"
            or verdict.get("matches_narration") is False
            or verdict.get("specific_enough") is False
            or verdict.get("quality_ok") is False
            or verdict.get("correct_subject_visible") is False
            or verdict.get("wrong_subject_visible") is True
            or verdict.get("contradicts_narration") is True
            or verdict.get("era_ok") is False
            or verdict.get("target_visible") is False
            or verdict.get("source_title_conflict_resolved") is False
            or bool(cast_resolution))
        if not explicit_negative:
            raise VisionBackendError(
                f"native still verifier returned incomplete keep evidence for beat "
                f"{getattr(seg, 'index', '?')} ({why})", kind="down")
        if not allow_content_reject:
            raise NonRetryableBuildError(
                f"image semantic gate: native still for beat {getattr(seg, 'index', '?')} failed "
                f"strict verification ({why})", kind="selection_relevance")
    image_hash = _image_file_sha256(image_path)
    _require_unchanged_still_source(owner, source_fingerprint)
    if not image_hash or image_hash != image_hash_before:
        raise NonRetryableBuildError(
            "image-lineage gate: native still pixels changed while the semantic verifier was "
            "judging them", kind="scene_lineage")
    model = str(verdict.get("vision_served_by") or "")
    if not model or model == "none":
        raise VisionBackendError(
            "native still verifier returned a keep verdict without actual served-model identity; "
            "predicted provider identity cannot authorize publication", kind="down")
    question_fp, source_fp = _semantic_still_question_fingerprint(
        proj, seg, owner, image_sha256=image_hash, model=model,
        source_fingerprint=source_fingerprint)
    result = {
        "verdict": verdict,
        "model": model,
        "question_fingerprint": question_fp,
        "source_content_fingerprint": source_fp,
        "image_sha256": image_hash,
        "strict_reason": why,
    }
    return result


def _extract_owned_still_fullres(original: Path, owner: dict) -> tuple[Path, int, int]:
    """Extract the exact indexed midpoint without scaling and require native publication pixels."""
    from .verify import NonRetryableBuildError
    import hashlib

    token = hashlib.sha256(
        f"{owner['source_id']}:{owner['shot_index']}:{owner['time']:.3f}".encode("utf-8")
    ).hexdigest()[:12]
    dest = original.with_name(f"{original.stem}_fullres_{token}.jpg")
    dest.unlink(missing_ok=True)       # never let a stale extraction satisfy this run
    proc = subprocess.run(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-ss", f"{owner['time']:.3f}",
         "-i", owner["source_path"], "-frames:v", "1", "-q:v", "2", str(dest)],
        capture_output=True, timeout=60)
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size <= 0:
        raise NonRetryableBuildError(
            f"image-lineage gate: could not re-extract {owner['source_id']!r} shot "
            f"{owner['shot_index']} at {owner['time']:.3f}s; refusing an unowned fallback",
            kind="scene_lineage")
    iw, ih = _image_pixel_dimensions(dest)
    if not _publishable_still_pixels(iw, ih):
        raise NonRetryableBuildError(
            f"image native-resolution gate: extracted still {dest.name} is {iw}x{ih}; "
            "the aired source-frame itself must retain real 1280x720-or-better pixels",
            kind="native_resolution")
    return dest, iw, ih


def _rescue_still_fullres(proj, sel, img_path: str, log, *, seg=None, eng=None,
                          allow_semantic_reject: bool = False,
                          refresh_semantic_verdict: bool = False) -> dict:
    """Return the aired still plus independently checkable, actual ownership.

    Source-frame stills with complete metadata are re-extracted from the exact indexed ``src`` and
    ``shot`` midpoint.  The selection's moving-video source/time are deliberately never consulted.
    If an older record has no source+shot metadata, its already verified image is preserved byte for
    byte; build then binds lineage to that file hash.  Any contradictory metadata fails closed.
    """
    from .verify import NonRetryableBuildError
    original = _require_declared_image_path(sel)
    if Path(img_path).resolve() != original.resolve():
        raise NonRetryableBuildError(
            "image-lineage gate: rescue input is not the selection's declared verified image",
            kind="scene_lineage")
    meta, _is_source_frame = _source_frame_meta(sel)
    _image_source = str(meta.get("source") or "")
    owned_refresh_candidate = bool(
        refresh_semantic_verdict and "source-frame" in _image_source
        and str(meta.get("src", "") or "") and meta.get("shot") is not None)
    if not (meta.get("still_verified") is True or meta.get("still_semantic_verified") is True
            or _image_source == "web-exact-scene" or owned_refresh_candidate):
        raise NonRetryableBuildError(
            f"image-lineage gate: declared image source {_image_source or '?'} has no explicit "
            f"verifier proof; refusing to label it verified_image", kind="scene_lineage")
    declared_hash = _image_file_sha256(original)
    claimed_semantic_hash = str(meta.get("still_image_sha256") or "")
    semantic_claimed = meta.get("still_semantic_verified") is True
    if semantic_claimed and (
            not claimed_semantic_hash or claimed_semantic_hash != declared_hash):
        raise NonRetryableBuildError(
            "image-lineage gate: semantic still SHA is missing or does not match the "
            "declared judged bytes", kind="scene_lineage")
    owner = _resolve_indexed_still_owner(proj, sel)
    if owner is not None:
        if seg is not None:
            from .era import title_era_conflicts
            from .verify import _project_beat_era
            owner_source = proj.source(owner["source_id"])
            owner_title = ((getattr(owner_source, "title", "") or "") + " "
                           + owner["source_id"])
            beat_era = _project_beat_era(proj, seg)
            if title_era_conflicts(beat_era, owner_title):
                raise NonRetryableBuildError(
                    f"image semantic gate: beat {getattr(seg, 'index', '?')} requires "
                    f"{beat_era or 'its local era'}, but still owner declares another era "
                    f"({owner_title.strip()[:100]})", kind="selection_relevance")
            # Standalone/rerender build must independently enforce the same pixel-level answer as
            # the early relevance audit.  A title mismatch is only a warning, so a focused verdict
            # may resolve it; without that affirmative field an ordinary preserve path cannot air
            # the still.  The refresh lane is allowed through because it asks the corrected native
            # pixel question immediately below.
            from .verify import (_cast_warning_resolution_reason, _project_char2actor,
                                 _project_exact_cast_warning)
            cast_warning = _project_exact_cast_warning(
                proj, seg, str(owner.get("source_id") or ""))
            evidence = meta.get("still_verifier") or meta.get("exact_still_verifier") or {}
            cast_resolution = (_cast_warning_resolution_reason(
                evidence, seg, _project_char2actor(proj)) if cast_warning else "")
            if cast_warning and not refresh_semantic_verdict and cast_resolution:
                raise NonRetryableBuildError(
                    f"image semantic gate: beat {getattr(seg, 'index', '?')} has an unresolved "
                    f"source-title cast warning in its persisted pixel verdict ({cast_warning}; "
                    f"{cast_resolution})",
                    kind="selection_relevance")
        source_fingerprint = _still_owner_source_fingerprint(owner)
        sw, sh = _probe_image_owner_source(owner["source_path"])
        if not _publishable_still_pixels(sw, sh):
            reason = f"{sw}x{sh}" if sw and sh else "unprobeable"
            raise NonRetryableBuildError(
                f"image native-resolution gate: source-frame owner {owner['source_id']!r} is "
                f"{reason}; at least native 1280x720 is required and upscaling cannot create "
                f"detail",
                kind="native_resolution")
        # Recovery may arrive here because old thumbnail evidence is absent, the current beat
        # question changed, or persisted native proof became stale.  Re-establish the contract on
        # exact owner pixels in every case.  This is deliberately separate from normal build,
        # which never spends a fresh semantic call to rescue stale metadata on its own.
        if refresh_semantic_verdict:
            iw, ih = _image_pixel_dimensions(original)
            preserved = _publishable_still_pixels(iw, ih)
            if preserved:
                candidate = original
            else:
                candidate, iw, ih = _extract_owned_still_fullres(original, owner)
            if seg is None or eng is None:
                raise NonRetryableBuildError(
                    "image semantic gate: native still refresh lacks the current beat/verifier",
                    kind="selection_relevance")
            semantic = _strictly_verify_native_still(
                proj, sel, seg, owner, candidate, eng,
                source_fingerprint=source_fingerprint,
                allow_content_reject=allow_semantic_reject)
            _require_unchanged_still_source(owner, source_fingerprint)
            binding = _owned_still_binding(
                owner, original, source_fingerprint=source_fingerprint)
            semantic_reason = str(semantic.get("strict_reason") or "")
            log(f"build: native still semantic proof freshly "
                f"{'rejected' if semantic_reason else 'verified'} for beat "
                f"{getattr(seg, 'index', '?')} from indexed owner {owner['source_id']} shot "
                f"{owner['shot_index']} @{owner['time']:.3f}s ({iw}x{ih})")
            return {
                "path": str(candidate), "ownership_kind": "source_frame",
                "actual_source_id": owner["source_id"],
                "actual_shot_index": owner["shot_index"], "actual_time": owner["time"],
                "source_path": owner["source_path"], "source_width": sw,
                "source_height": sh, "image_width": iw, "image_height": ih,
                "preserved_original": preserved,
                "file_sha256": semantic["image_sha256"],
                "semantic_binding_preserved": not bool(semantic_reason),
                "semantic_image_sha256": semantic["image_sha256"],
                "semantic_rematerialized": not preserved,
                "semantic_verifier": semantic["verdict"],
                "semantic_model": semantic["model"],
                "semantic_question_fingerprint": semantic["question_fingerprint"],
                "source_content_fingerprint": semantic["source_content_fingerprint"],
                "semantic_strict_reason": semantic_reason,
                **binding,
            }
        # Strict semantic evidence is bound to the exact declared bytes. Re-extracting another
        # JPEG from the same owned instant would air pixels the semantic verifier never judged.
        # Preserve native judged bytes when the persisted SHA matches. A low-resolution index
        # thumbnail may only be replaced after a fresh strict verdict on the exact native bytes;
        # its old thumbnail verdict is never transferred.
        strict_semantic_bound = semantic_claimed
        if strict_semantic_bound:
            iw, ih = _image_pixel_dimensions(original)
            # A native image persisted by the pre-assembly recovery lane is no longer byte-equal
            # to the indexed thumbnail, so its semantic authorization must be re-derived from the
            # current beat question before it may take the cheap preserve-original path.  Merely
            # carrying non-empty provenance strings is not proof: a later analyzer/policy edit
            # could otherwise reuse a KEEP that answered a different question.
            if meta.get("native_semantic_materialized") is True:
                if seg is None:
                    raise NonRetryableBuildError(
                        "image semantic gate: persisted native still has no current beat question",
                        kind="selection_relevance")
                stale_reason = _persisted_native_still_semantic_reason(
                    proj, seg, owner, meta, image_sha256=declared_hash,
                    source_fingerprint=source_fingerprint)
                if stale_reason:
                    raise NonRetryableBuildError(
                        "image semantic gate: persisted native still proof is stale/incomplete ("
                        + stale_reason + ")",
                        kind="selection_relevance")
            if not _publishable_still_pixels(iw, ih):
                if seg is None or eng is None:
                    raise NonRetryableBuildError(
                        f"image native-resolution gate: semantically judged still is {iw}x{ih}; "
                        "the exact judged bytes must already be real 1280x720-or-better",
                        kind="native_resolution")
                dest, iw, ih = _extract_owned_still_fullres(original, owner)
                semantic = _strictly_verify_native_still(
                    proj, sel, seg, owner, dest, eng,
                    source_fingerprint=source_fingerprint,
                    allow_content_reject=allow_semantic_reject)
                _require_unchanged_still_source(owner, source_fingerprint)
                binding = _owned_still_binding(
                    owner, original, source_fingerprint=source_fingerprint)
                semantic_reason = str(semantic.get("strict_reason") or "")
                log(f"build: native still freshly "
                    f"{'rejected' if semantic_reason else 'verified'} for beat "
                    f"{getattr(seg, 'index', '?')} from indexed owner {owner['source_id']} shot "
                    f"{owner['shot_index']} @{owner['time']:.3f}s ({sw}x{sh})")
                return {
                    "path": str(dest), "ownership_kind": "source_frame",
                    "actual_source_id": owner["source_id"],
                    "actual_shot_index": owner["shot_index"], "actual_time": owner["time"],
                    "source_path": owner["source_path"], "source_width": sw,
                    "source_height": sh, "image_width": iw, "image_height": ih,
                    "preserved_original": False, "file_sha256": semantic["image_sha256"],
                    "semantic_binding_preserved": not bool(semantic_reason),
                    "semantic_image_sha256": semantic["image_sha256"],
                    "semantic_rematerialized": True,
                    "semantic_verifier": semantic["verdict"],
                    "semantic_model": semantic["model"],
                    "semantic_question_fingerprint": semantic["question_fingerprint"],
                    "source_content_fingerprint": semantic["source_content_fingerprint"],
                    "semantic_strict_reason": semantic_reason,
                    **binding,
                }
            _require_unchanged_still_source(owner, source_fingerprint)
            binding = _owned_still_binding(
                owner, original, source_fingerprint=source_fingerprint)
            return {"path": str(original), "ownership_kind": "source_frame",
                    "actual_source_id": owner["source_id"],
                    "actual_shot_index": owner["shot_index"], "actual_time": owner["time"],
                    "source_path": owner["source_path"], "source_width": sw,
                    "source_height": sh, "image_width": iw, "image_height": ih,
                    "preserved_original": True, "file_sha256": declared_hash,
                    "semantic_binding_preserved": True,
                    "semantic_image_sha256": claimed_semantic_hash,
                    **binding}
        dest, iw, ih = _extract_owned_still_fullres(original, owner)
        _require_unchanged_still_source(owner, source_fingerprint)
        binding = _owned_still_binding(
            owner, original, source_fingerprint=source_fingerprint)
        log(f"build: still re-extracted from indexed owner {owner['source_id']} shot "
            f"{owner['shot_index']} @{owner['time']:.3f}s ({sw}x{sh}) — {dest.name}")
        return {"path": str(dest), "ownership_kind": "source_frame",
                "actual_source_id": owner["source_id"],
                "actual_shot_index": owner["shot_index"], "actual_time": owner["time"],
                "source_path": owner["source_path"], "source_width": sw, "source_height": sh,
                "image_width": iw, "image_height": ih, "preserved_original": False,
                "file_sha256": _image_file_sha256(dest),
                "semantic_binding_preserved": False, "semantic_image_sha256": "",
                **binding}

    # Web images and legacy source-frame records without ownership metadata remain the verified
    # file that was selected.  They may not borrow sel.source_id/time.  Legacy source-frame records
    # need explicit verifier evidence before this preservation path is allowed.
    iw, ih = _image_pixel_dimensions(original)
    if not _publishable_still_pixels(iw, ih):
        raise NonRetryableBuildError(
            f"image native-resolution gate: verified still is {iw}x{ih}; a real 1280x720-or-better "
            f"source image is required and Ken Burns upscaling cannot create detail",
            kind="native_resolution")
    digest = _image_file_sha256(original)
    if not digest:
        raise NonRetryableBuildError(
            f"image-lineage gate: verified still {original} could not be hashed",
            kind="scene_lineage")
    return {"path": str(original), "ownership_kind": "verified_file",
            "actual_source_id": "", "actual_shot_index": None, "actual_time": None,
            "source_path": "", "source_width": 0, "source_height": 0,
            "image_width": iw, "image_height": ih, "preserved_original": True,
            "file_sha256": digest,
            "semantic_binding_preserved": bool(
                meta.get("still_semantic_verified") is True
                and str(meta.get("still_image_sha256") or "") == digest),
            "semantic_image_sha256": (str(meta.get("still_image_sha256") or "")
                                       if meta.get("still_semantic_verified") is True else "")}


def _verified_image_lineage_root(proj, sel, rescue: dict, final_scene: int, *, seg=None) -> dict:
    """Validate expected image ownership against the rescue's actual root and bind it immutably."""
    import hashlib
    import json
    from .verify import NonRetryableBuildError
    meta, _is_source_frame = _source_frame_meta(sel)
    expected = _resolve_indexed_still_owner(proj, sel)
    actual_path = Path(str(rescue.get("path") or ""))
    actual_hash = _image_file_sha256(actual_path)
    actual_dims = _image_pixel_dimensions(actual_path)
    reported_dims = (int(rescue.get("image_width") or 0), int(rescue.get("image_height") or 0))
    if (not actual_hash or actual_hash != str(rescue.get("file_sha256") or "")
            or actual_dims != reported_dims):
        raise NonRetryableBuildError(
            "image-lineage gate: rescued image bytes/dimensions disagree with their ownership "
            "record", kind="scene_lineage")
    if expected is not None:
        source_dims = _probe_image_owner_source(expected["source_path"])
        reported_source_dims = (int(rescue.get("source_width") or 0),
                                int(rescue.get("source_height") or 0))
        if (source_dims != reported_source_dims or not _publishable_still_pixels(*source_dims)
                or not _publishable_still_pixels(*actual_dims)):
            raise NonRetryableBuildError(
                f"image native-resolution gate: source/air dimensions changed or are below "
                f"native 1280x720 "
                f"(source={source_dims}, aired={actual_dims})", kind="native_resolution")
        actual_tuple = (str(rescue.get("actual_source_id") or ""),
                        rescue.get("actual_shot_index"), rescue.get("actual_time"))
        expected_tuple = (expected["source_id"], expected["shot_index"], expected["time"])
        try:
            time_ok = abs(float(actual_tuple[2]) - float(expected_tuple[2])) <= 0.001
        except (TypeError, ValueError):
            time_ok = False
        if (actual_tuple[:2] != expected_tuple[:2] or not time_ok
                or rescue.get("ownership_kind") != "source_frame"):
            raise NonRetryableBuildError(
                f"image-lineage gate: expected owner {expected_tuple!r}, got {actual_tuple!r}",
                kind="scene_lineage")
        declared = _require_declared_image_path(sel)
        # Image rescue runs before narration/breakout work. Re-check every ownership input at the
        # moment the still is consumed so a source/keyframe/selection mutation during that window
        # can never reuse an earlier extraction or verdict.
        current_binding = _owned_still_binding(expected, declared)
        for field in (
                "declared_image_sha256", "indexed_keyframe_sha256",
                "owner_source_content_fingerprint", "owner_source_id", "owner_shot_index"):
            if str(rescue.get(field, "")) != str(current_binding.get(field, "")):
                raise NonRetryableBuildError(
                    f"image-lineage gate: preflight ownership field {field} changed before air",
                    kind="scene_lineage")
        try:
            binding_time_ok = abs(float(rescue.get("owner_time"))
                                  - float(current_binding.get("owner_time"))) <= 0.001
        except (TypeError, ValueError):
            binding_time_ok = False
        if not binding_time_ok:
            raise NonRetryableBuildError(
                "image-lineage gate: indexed still midpoint changed after preflight",
                kind="scene_lineage")

        declared_hash = _image_file_sha256(declared)
        claimed_semantic_hash = str(meta.get("still_image_sha256") or "")
        semantic_claimed = meta.get("still_semantic_verified") is True
        if semantic_claimed and (
                not claimed_semantic_hash or claimed_semantic_hash != declared_hash):
            raise NonRetryableBuildError(
                "image-lineage gate: semantic still SHA is missing or does not match the "
                "declared judged bytes", kind="scene_lineage")
        if semantic_claimed:
            rematerialized = rescue.get("semantic_rematerialized") is True
            if rematerialized:
                if seg is None:
                    raise NonRetryableBuildError(
                        "image-lineage gate: native semantic rescue has no current beat question",
                        kind="scene_lineage")
                from .relevance_contract import strict_still_evidence_reason
                evidence = rescue.get("semantic_verifier") or {}
                reason = strict_still_evidence_reason(evidence, seg)
                model = str(rescue.get("semantic_model") or "")
                expected_qfp, expected_source_fp = _semantic_still_question_fingerprint(
                    proj, seg, expected, image_sha256=actual_hash, model=model)
                if (reason
                        or rescue.get("semantic_binding_preserved") is not True
                        or str(rescue.get("semantic_image_sha256") or "") != actual_hash
                        or str(rescue.get("source_content_fingerprint") or "")
                            != expected_source_fp
                        or str(rescue.get("semantic_question_fingerprint") or "")
                            != expected_qfp
                        or not model or model == "none"
                        or rescue.get("preserved_original") is not False):
                    raise NonRetryableBuildError(
                        "image-lineage gate: native semantic rescue is not bound to the current "
                        "source, pixels, model, and beat question",
                        kind="scene_lineage")
            elif (rescue.get("semantic_binding_preserved") is not True
                    or str(rescue.get("semantic_image_sha256") or "")
                        != claimed_semantic_hash
                    or actual_hash != claimed_semantic_hash
                    or actual_path.resolve() != declared.resolve()
                    or rescue.get("preserved_original") is not True):
                raise NonRetryableBuildError(
                    "image-lineage gate: strict semantic evidence is not bound to the exact aired "
                    "source-frame bytes", kind="scene_lineage")
        owner_payload = {"kind": "source_frame", "source_id": expected["source_id"],
                         "shot_index": expected["shot_index"], "time": expected["time"]}
    else:
        expected_file = _require_declared_image_path(sel)
        expected_hash = _image_file_sha256(expected_file)
        actual_hash = str(rescue.get("file_sha256") or "")
        claimed_semantic_hash = str(meta.get("still_image_sha256") or "")
        semantic_claimed = meta.get("still_semantic_verified") is True
        if semantic_claimed and (
                not claimed_semantic_hash or claimed_semantic_hash != expected_hash
                or rescue.get("semantic_binding_preserved") is not True
                or str(rescue.get("semantic_image_sha256") or "") != expected_hash):
            raise NonRetryableBuildError(
                "image-lineage gate: metadata-free semantic still proof changed before air",
                kind="scene_lineage")
        # No ownership metadata means preservation, never a guessed source-frame extraction.
        if (rescue.get("ownership_kind") != "verified_file" or not expected_hash
                or actual_hash != expected_hash
                or Path(str(rescue.get("path") or "")).resolve() != expected_file.resolve()):
            raise NonRetryableBuildError(
                "image-lineage gate: metadata-free verified image was not preserved byte-for-byte",
                kind="scene_lineage")
        owner_payload = {"kind": "verified_file", "sha256": expected_hash,
                         "source": str(meta.get("source") or "")}
    payload = {"original_beat": int(getattr(sel, "segment_index", final_scene)), **owner_payload}
    binding = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8")).hexdigest()
    iw, ih = int(rescue.get("image_width") or 0), int(rescue.get("image_height") or 0)
    return {
        "kind": "verified_image", "original_beat": payload["original_beat"],
        "owner_beat": payload["original_beat"], "via": "verified_image", "validated": True,
        "root_binding": binding, "image_owner_kind": owner_payload["kind"],
        "expected_image_source_id": (expected or {}).get("source_id", ""),
        "expected_image_shot_index": (expected or {}).get("shot_index"),
        "actual_image_source_id": str(rescue.get("actual_source_id") or ""),
        "actual_image_shot_index": rescue.get("actual_shot_index"),
        "actual_image_time": rescue.get("actual_time"),
        "source_native_width": int(rescue.get("source_width") or 0),
        "source_native_height": int(rescue.get("source_height") or 0),
        "image_width": iw, "image_height": ih,
        "image_sha256": str(rescue.get("file_sha256") or ""),
        "preserved_original": bool(rescue.get("preserved_original")),
        "semantic_binding_preserved": bool(rescue.get("semantic_binding_preserved")),
        "semantic_image_sha256": str(rescue.get("semantic_image_sha256") or ""),
        "semantic_rematerialized": bool(rescue.get("semantic_rematerialized")),
        "semantic_model": str(rescue.get("semantic_model") or ""),
        "semantic_question_fingerprint": str(
            rescue.get("semantic_question_fingerprint") or ""),
        "semantic_source_content_fingerprint": str(
            rescue.get("source_content_fingerprint") or ""),
    }


def _persist_image_lineage_audit(proj, entries: list[dict], failures: list[dict]) -> Path:
    """Atomically persist image ownership and native dimensions; audit failure blocks build."""
    import json
    path = Path(proj.output_dir) / "image_lineage_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "image_lineage/1", "passed": not failures,
               "minimum_source_video_height": 720,
               "minimum_source_video_short_edge": 720,
               "minimum_source_video_long_edge": 1280,
               "minimum_still_short_edge": _IMAGE_MIN_SHORT_EDGE,
               "minimum_still_long_edge": _IMAGE_MIN_LONG_EDGE,
               "entries": entries, "failures": failures}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


_SRCDARK_MEMO: dict = {}


def _source_window_too_dark(src_path, start: float, need: float,
                            floor: float = 50.0, min_dark_run: float = 0.8) -> bool:
    """`_clip_too_dark`'s question, asked of a SOURCE window BEFORE anything is cut.

    Same crop, stride, statistic and thresholds — deliberately not a new rule. It exists so the
    answer can change WHICH already-approved window a beat uses, instead of arriving after the cut,
    when the only remedy left is a freeze.

    WHY THIS AND NOT A SLIDE INSIDE THE SHOT, which is the obvious idea: 175 of 272 beats (64%) air
    LONGER than their selected shot (median 1.03s beyond it), so there is usually no room to slide.
    Replaying an honest slide — the real wqc_render_window with illegibility spans injected, anchor
    overlap and moment-preserve untouched — rescued 1 beat and regressed 1. Net zero, measured.
    Choosing a different window from the beat's OWN relevance-ranked list works instead: of the 28
    beats whose first choice probes dark, 27 have a legible alternate already in that list.

    A source-side probe predicts the encoded-clip verdict: 48 windows stratified across index
    luma_avg 3-95, cut with the production chain (Ken-Burns zoom -> CAS -> libx264 CRF18), gave
    48/48 verdict agreement against `_clip_too_dark` on the resulting clip.

    Unmeasurable -> False, exactly as `_clip_too_dark` does: the final black gate is the backstop."""
    import shutil as _shutil
    import tempfile as _tempfile
    key = (str(src_path), round(float(start), 2), round(float(need), 2))
    if key in _SRCDARK_MEMO:
        return _SRCDARK_MEMO[key]
    ff = ffmpeg_exe()
    _d = Path(_tempfile.mkdtemp(prefix="srcdark_"))
    out = False
    try:
        subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{max(0.0, float(start)):.3f}",
                        "-i", str(src_path), "-t", f"{max(0.2, float(need)):.3f}",
                        "-vf", "crop=iw*0.9:ih*0.84:iw*0.05:ih*0.08,fps=2,scale=320:-1",
                        "-q:v", "5", str(_d / "f_%05d.jpg")], capture_output=True, timeout=60)
        his = [v for v in (_frame_luma_hi(f) for f in sorted(_d.glob("f_*.jpg"))) if v is not None]
        if his:
            if max(his) < floor:
                out = True                     # unreadable THROUGHOUT
            else:
                _run = 0                       # frames are 0.5s apart (fps=2)
                for _h in his:
                    _run = _run + 1 if _h < floor else 0
                    if _run * 0.5 >= min_dark_run:
                        out = True             # a SUSTAINED dark patch the final gate would flag
                        break
    except Exception:
        out = False
    finally:
        _shutil.rmtree(_d, ignore_errors=True)
    _SRCDARK_MEMO[key] = out
    return out


def _clip_too_dark(clip_path, floor: float = 50.0, min_dark_run: float = 0.8) -> bool:
    """True when a CUT clip is unusable: EITHER unreadable throughout, OR it contains a SUSTAINED
    dark RUN of at least `min_dark_run` seconds — the SAME run-based rule _final_video_black_gate
    applies to the finished video.

    This pre-pass previously sampled only 4 points and asked "is it dark THROUGHOUT?" (max < floor).
    A clip that is mostly legible but has a dark PATCH therefore passed here and then tripped the
    final gate, quarantining the whole render (observed: one 1s region at 274.5s failed a 15-minute
    render). The pre-pass must PREVENT exactly what the gate DETECTS, so it now samples densely at
    the gate's own 0.5s stride and looks for a sustained sub-floor run. Measures the PICTURE area
    (centre crop, excluding edges; no letterbox at this stage). Unmeasurable → False (the final
    black gate remains the backstop)."""
    import shutil as _shutil
    import tempfile as _tempfile
    ff = ffmpeg_exe()
    _d = Path(_tempfile.mkdtemp(prefix="clipdark_"))
    try:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(clip_path),
                        "-vf", "crop=iw*0.9:ih*0.84:iw*0.05:ih*0.08,fps=2,scale=320:-1",
                        "-q:v", "5", str(_d / "f_%05d.jpg")], capture_output=True, timeout=90)
        his = [v for v in (_frame_luma_hi(f) for f in sorted(_d.glob("f_*.jpg"))) if v is not None]
        if not his:
            return False                       # unmeasurable → the final gate is the backstop
        if max(his) < floor:
            return True                        # unreadable THROUGHOUT
        _run = 0                               # frames are 0.5s apart (fps=2)
        for _h in his:
            _run = _run + 1 if _h < floor else 0
            if _run * 0.5 >= min_dark_run:
                return True                    # a SUSTAINED dark patch the final gate would flag
        return False
    except Exception:
        return False
    finally:
        _shutil.rmtree(_d, ignore_errors=True)


def _final_video_black_gate(result: Path, work: Path, *, log) -> Path:
    """FAIL-CLOSED release gate against sustained NEAR-BLACK / unusable-dark footage in the finished
    video (distinct from the assemble black-repair, which only fills TRUE-black gaps). Samples every
    0.5s; a run of frames whose brightest content stays below a legibility floor (luma_hi) for longer
    than a short intentional fade is UNUSABLE and blocks publication (quarantine + raise). Short
    dark dips (fades between shots) are allowed. env VIDLORE_CLIPSTUDIO_FINAL_BLACK_GATE=0 disables;
    _FLOOR / _MINDUR tune."""
    import os as _os5
    import json as _json_b
    if _os5.environ.get("VIDLORE_CLIPSTUDIO_FINAL_BLACK_GATE", "1").strip() in ("0", "false", "no"):
        return result
    from .config import _f as _cfg_fb, _i as _cfg_ib
    floor = _cfg_fb("VIDLORE_CLIPSTUDIO_FINAL_BLACK_FLOOR", 50.0)     # luma_hi below this = unusable
    min_dur = _cfg_fb("VIDLORE_CLIPSTUDIO_FINAL_BLACK_MINDUR", 0.8)   # sustained longer than a fade
    max_fail_frac = _cfg_fb("VIDLORE_CLIPSTUDIO_FINAL_BLACK_MAXFAIL", 0.03)   # strict probe-failure cap
    stride = 0.5
    ff = ffmpeg_exe()
    # REVIEW-DRAFT consistency: the footage + unverified-exact gates both WARN (not block) under
    # RELEASE_BLOCK_MODE=warn — the whole point of a review draft is to deliver the imperfect video
    # so a human can SEE and judge it. A short unusable-dark region is exactly such a "see and fix
    # that one beat" defect; withholding a 13-min draft over a 1s dark spot is inconsistent and
    # unhelpful. So in warn mode this gate logs a LOUD warning + records the regions but DELIVERS
    # the video (no quarantine). Production ('block', the default) is unchanged: hard fail-closed —
    # a black region never ships to publication.
    _review_mode = _os5.environ.get("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block") \
        .strip().lower() == "warn"

    def _blk(reason):
        try:
            (work.parent / "final_black_failures.json").write_text(
                _json_b.dumps({"reason": reason, "floor_luma_hi": floor, "min_dur_s": min_dur},
                              indent=1), encoding="utf-8")
        except Exception:
            pass
        if _review_mode:
            log(f"build: ⚠ BLACK-QA (mode=warn, REVIEW BUILD — not for publication) — {reason}. "
                f"Delivered anyway for review; fix the dark beat before publishing. "
                f"See final_black_failures.json.")
            return result
        _q = result.with_name(result.stem + ".FAILED_BLACK_QA" + result.suffix)
        try:
            if _q.exists():
                _q.unlink()
            result.rename(_q)
        except Exception:
            _q = result
        log(f"build: ⛔ RELEASE-BLOCKED — {reason}; quarantined → {_q.name}. See final_black_failures.json.")
        raise RuntimeError(f"final-video black gate: {reason} (quarantined at {_q.name})")

    dur = _probe_duration(result)
    if dur <= 0:
        return _blk("could not probe the final video duration (fail-closed — cannot verify legibility)")
    scan = work / "_blackscan"
    try:
        scan.mkdir(exist_ok=True)
        for _o in scan.glob("*.jpg"):
            _o.unlink(missing_ok=True)
    except Exception:
        pass
    # Crop to the actual PICTURE AREA before scaling: exclude the cinematic letterbox bars AND the
    # bottom caption band (y 0.15-0.70) and the outer 10% — so white captions or bars can never make
    # a black picture pass the gate.
    _p = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(result),
                         "-vf", f"fps={1.0 / stride:.4f},"
                                f"crop=iw*0.80:ih*0.55:iw*0.10:ih*0.15,scale=480:-1",
                         "-q:v", "5", str(scan / "b_%05d.jpg")], capture_output=True, timeout=1800)
    frames = sorted(scan.glob("b_*.jpg"))
    _cov = _scan_coverage_reason(len(frames), stride, dur, _p.returncode)
    if _cov is None and not _final_timestamp_reachable(result, dur, ff, scan):
        _cov = f"final timestamp (~{dur:.1f}s) does not decode — tail not covered"
    if _cov is not None:
        return _blk(f"{_cov} — cannot verify legibility to the final timestamp")
    lows, fails = [], 0
    for i, fp in enumerate(frames):
        v = _frame_luma_hi(fp)
        if v is None:
            fails += 1                                 # explicit probe failure — counted, not 'bright'
        elif v < floor:
            lows.append(i)
    try:
        for _o in scan.glob("*.jpg"):
            _o.unlink(missing_ok=True)
    except Exception:
        pass
    if fails > max(1, int(len(frames) * max_fail_frac)):
        return _blk(f"too many unreadable scan probes ({fails}/{len(frames)} > {max_fail_frac:.0%}) — "
             f"UNVERIFIED (fail-closed)")
    # group consecutive low frames into runs; a run longer than min_dur is an unusable-dark region
    runs, cur = [], []
    for i in lows:
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if cur:
                runs.append(cur)
            cur = [i]
    if cur:
        runs.append(cur)
    bad = [(r[0] * stride, (r[-1] + 1) * stride) for r in runs
           if (r[-1] - r[0] + 1) * stride > min_dur]
    if not bad:
        log(f"build: final-video black/legibility scan clean — {len(frames)} picture-area frames "
            f"@{stride}s ({fails} probe fail), 0 sustained unusable-dark region(s) (short fades allowed)")
        return result
    try:
        (work.parent / "final_black_failures.json").write_text(
            _json_b.dumps({"unusable_dark_regions_s": [[round(a, 2), round(b, 2)] for a, b in bad],
                           "floor_luma_hi": floor, "min_dur_s": min_dur}, indent=1), encoding="utf-8")
    except Exception:
        pass
    return _blk(f"{len(bad)} sustained unusable-dark region(s) (first {bad[0][0]:.1f}-{bad[0][1]:.1f}s, "
         f"picture-area luma_hi < {floor:.0f})")


def _ass_ts(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _correct_breakout_words(words: list, known_line: str, *, log=None, return_meta: bool = False):
    """Repair ASR slips in a breakout caption using the KNOWN dialogue line, keeping ASR timings.

    Breakout captions are re-ASR'd from the extracted audio, so they inherit its mistakes — and a
    burned caption asserts, in the show's own voice, that a character said something. Measured:

        burned : "I'll make sure you understand that, and I've won your war for you"
        actual : "I'll make sure you understand that, WHEN I've won your war for you"

    which inverts a conditional threat into a past-tense boast — over the very line the narration
    then unpacks. ("Come on off" for "Come along" is the same class, less costly.)

    Deliberately narrow, because a caption is a claim about what was SAID:
      * the PHRASE must align to the known line with high confidence. That is the whole safeguard,
        and it is a contextual one: a breakout that drifted onto different dialogue never aligns,
        so it is never overwritten with a line it does not speak. The script proposes; the audio
        disposes.
      * only 1:1 slots the aligner already matched — never inserts, deletes, or re-times;
      * timings stay the ASR's, since they came from the actual audio.

    Note what is deliberately NOT here: a per-word similarity check. "and"/"when" scores 0.29
    character-wise, so a word-level gate would reject the exact repair that matters most — and
    judging one word in isolation is what produced the error in the first place. Context decides.

    return_meta=True additionally yields (words, align, src_ok, opcodes, known_words):
      align   — source-side RECALL (matched source tokens / total source tokens). Recall, not
                SequenceMatcher.ratio(), because a breakout window runs to the end of a complete
                spoken line and often carries extra dialogue past the quote, which sinks the
                symmetric ratio even when every quote word was actually spoken.
      src_ok  — per-ASR-word booleans: True where the source line corroborates that word.
    The caller uses those to KEEP a low-ASR-confidence caption line that the verified source line
    already vouches for, instead of deleting it (see _breakout_caption_ass)."""
    kw = [w for w in re.findall(r"[\w']+", known_line or "")]
    _nometa = (words, 0.0, [False] * len(words or []), [], kw)
    if not words or len(kw) < 3:
        return _nometa if return_meta else words
    from difflib import SequenceMatcher as _SM

    def _n(s):
        return re.sub(r"[^a-z0-9']", "", (s or "").lower())

    aw = [_n(w[0]) for w in words]
    kn = [_n(w) for w in kw]
    sm = _SM(None, aw, kn)
    ratio = sm.ratio()
    if ratio < _BK_CAP_ALIGN_FUZZY:
        # not the same line — leave the audio's own words alone
        return _nometa if return_meta else words
    # TWO TIERS, because context and phonetics corroborate each other:
    #   strong context (>= 0.80) — the surrounding phrase is unmistakably this line, so a lone
    #     differing slot is an ASR slip. This is the tier that fixes "and"/"when", which no word-
    #     level test could ever pass (0.29 character-wise).
    #   weaker context (0.60-0.80) — correct only a slot that ALSO looks phonetically like a slip
    #     ("messence"/"essence" = 0.93). Short lines live here, where one word is a big fraction of
    #     the phrase: "I am the king" vs "…queen" reaches 0.75, and queen/king is 0.44, so the
    #     audio keeps its word. Semantic opposites are exactly what must never be rewritten.
    _strong = ratio >= _BK_CAP_ALIGN_MIN
    out = list(words)
    fixed = 0
    _ops = sm.get_opcodes()
    src_ok = [False] * len(words)
    _matched_src = 0
    for tag, i1, i2, j1, j2 in _ops:
        if tag == "equal":
            # ASR already agrees with the source line across this run
            for oi in range(i1, i2):
                src_ok[oi] = True
            _matched_src += (j2 - j1)
            continue
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue                       # 1:1 slots only — never insert/delete/re-time
        for oi, kj in zip(range(i1, i2), range(j1, j2)):
            if not aw[oi] or not kn[kj] or aw[oi] == kn[kj]:
                continue
            if not _strong and _SM(None, aw[oi], kn[kj]).ratio() < 0.60:
                continue                   # weak context needs phonetic corroboration
            w = out[oi]
            out[oi] = (kw[kj], w[1], w[2], w[3])
            src_ok[oi] = True              # this word is now literally the source's word
            _matched_src += 1
            fixed += 1
    if fixed and log:
        log(f"build: breakout caption — corrected {fixed} ASR slip(s) against the verified source "
            f"line (phrase align {ratio:.2f})")
    if return_meta:
        align = _matched_src / float(len(kn)) if kn else 0.0
        return out, min(1.0, align), src_ok, _ops, kw
    return out


def _breakout_caption_ass(caps: list, out_ass: Path, log=None, *, preset=None) -> Optional[Path]:
    """Build an ASS overlay that captions the SPOKEN dialogue during each real-audio breakout,
    word-by-word (karaoke fill) — distinct from the white narration caption but in the SAME
    selected design family (`preset`), so when the scene's own voice plays the viewer reads exactly
    what's said. Word timings come from whispering each breakout's audio (its own dialogue)."""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    lines = []
    _coverage_rows = []
    _coverage_failed = False
    try:
        m = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        return None
    # The BK Style line comes from the selected preset (same design family as the narration
    # caption). Resolve it UP FRONT so the shared width-aware line layout below can read its real
    # font size / outline / shadow / margins (falls back to professional so a stray call is valid).
    if preset is None:
        from .caption_presets import resolve_style as _rs
        preset = _rs(None)[0]
    # ── shared professional line-layout policy (SAME engine the narration caption uses) ──
    # Breakouts karaoke-FILL (\kf colour sweep); they never \fscx-grow, so peak_extra = 0. We split
    # pathological unbroken words by grapheme and lay each karaoke line into AT MOST two rows that
    # fit the BK safe area — outline+shadow reserved — shrinking \fs only as a bounded last resort.
    from vidlore.captions import _est_px, split_wide_cells, layout_two_lines
    _bk_size = float(preset.bk_size)
    _bk_lines = int(getattr(preset, "max_lines", 2))
    _safe_w = 1920.0 - 120.0 - 120.0                    # BK MarginL + MarginR (breakout_style_line)
    _pad = 2.0 * (float(preset.bk_outline_w) + float(preset.bk_shadow) + 2.0)
    _budget = max(_safe_w, 2.0 * (_safe_w - _pad) * 0.92)   # ~two readable rows' worth of width

    def _grp_w(ws_):
        if not ws_:
            return 0.0
        return (sum(_est_px(w[0], _bk_size) for w in ws_)
                + _est_px(" ", _bk_size) * max(0, len(ws_) - 1))
    for cap in caps:
        try:
            from .breakout_asr import transcribe_breakout_words, caption_coverage
            words, _asr_status = transcribe_breakout_words(
                str(cap["audio"]), model=m, duration=float(cap.get("dur") or 0.0),
                with_status=True)
        except Exception:
            words, _asr_status = [], {"complete": False}
        _spoken_words = list(words)
        # An incomplete transcription must fail this gate exactly as no transcription does.
        # Coverage is captioned/spoken, so words we FAILED TO HEAR raise it — a half-heard breakout
        # scores nearer 1.0 than a fully-heard one. Reading "we did not manage to listen" as
        # "nothing was said there" is the one way this measurement can approve on error.
        if words and not _asr_status.get("complete", False):
            log(f"breakout captions: incomplete ASR for seg {cap.get('seg_index')} "
                f"({_asr_status.get('chunks_decoded')}/{_asr_status.get('chunks_planned')} "
                f"windows) — failing coverage rather than scoring a partial transcript")
            words = []
        if not words:
            _cov0 = {"spoken_words": 0, "captioned_words": 0, "coverage": 0.0,
                     "asr_last_word_s": 0.0, "caption_last_word_s": 0.0,
                     "uncaptioned_tail_s": 0.0, "passed": False}
            cap["caption_coverage"] = _cov0
            _coverage_rows.append({"seg_index": cap.get("seg_index"), **_cov0})
            _coverage_failed = True
            continue
        words, _bk_align, _bk_srcok, _bk_ops, _bk_kw = _correct_breakout_words(
            words, cap.get("line", ""), log=log, return_meta=True)
        base = float(cap["start"])
        dur = float(cap["dur"])
        # WIDTH-AWARE grouping: accumulate words into karaoke lines whose 2-row layout fits the BK
        # safe area (never a clipped third row). Cap at 6 words OR ~two rows' width, whichever first.
        grp, cur = [], []
        _idx, _gidx, _curidx = 0, [], []
        for w in words:
            if cur and (len(cur) >= 6 or _grp_w(cur + [w]) > _budget):
                grp.append(cur); _gidx.append(_curidx); cur = []; _curidx = []
            cur.append(w); _curidx.append(_idx); _idx += 1
        if cur:
            grp.append(cur); _gidx.append(_curidx)
        # ASR-CONFIDENCE floor per line: whisper mishears movie audio occasionally ("...poison
        # your SON" transcribed as "poison your three."), and a wrong word burned on screen
        # reads like a third-party subtitle. A missing caption line beats a wrong one — drop
        # any line whose weakest word is below the floor.
        #
        # SOURCE-BACKED RESCUE. That floor is right when the ASR is the only evidence — but it was
        # deleting lines the audio says perfectly. Measured on job 69d80e9dd4, all four breakouts
        # lost text this way (35-71% of words survived); one showed only "white winds blow, the lone
        # wolf" and never showed the payoff "but the pack survives", while the delivered audio said
        # the full line verbatim (confirmed by re-transcribing the delivered mix). The cause is that
        # "base" int8 whisper mis-hears movie audio and reports low confidence — yet
        # _correct_breakout_words has, three lines earlier, already matched those very words against
        # the VERIFIED source line. A word the source line corroborates is not a guess, whatever the
        # acoustic model's confidence, so keep it. A line with no such corroboration still drops.
        import os as _os_bkc
        try:
            _pfloor = float(_os_bkc.environ.get("VIDLORE_CLIPSTUDIO_BK_CAP_CONF_FLOOR",
                                                "0.45") or 0.45)
        except (TypeError, ValueError):
            _pfloor = 0.45
        _kept = []
        _rescued = 0
        _src_strong = bool(cap.get("line")) and _bk_align >= _BK_CAP_ALIGN_MIN
        # WORD-LEVEL REPAIR (fringe trim + aired corroboration). The all-or-nothing rescue above
        # deleted the essay's own cited quote TWICE on job 5462677f95: the caption line
        # "marry that beast, do you? Well," is 100% source-backed except the trailing "Well,"
        # (the next speaker's reply, past the quote) — one unbacked fringe word must cost the
        # fringe, not the payload. Secondary evidence: the selection-time transcript
        # (aired_transcript in work/breakout_audit.json). It is the same whisper model, so it is
        # corroboration, not proof — an aired-backed word also needs non-hopeless acoustics
        # (conf >= _aconf), which keeps genuinely-garbled lines (min conf 0.01) dropped.
        _repair_on = _os_bkc.environ.get(
            "VIDLORE_CLIPSTUDIO_BK_CAP_REPAIR", "1").lower() not in ("0", "false", "no")
        try:
            _aconf = float(_os_bkc.environ.get("VIDLORE_CLIPSTUDIO_BK_CAP_AIRED_CONF",
                                               "0.30") or 0.30)
        except (TypeError, ValueError):
            _aconf = 0.30
        _aired_toks: set = set()
        if _repair_on:
            try:
                import json as _json_bkc
                _aud = _json_bkc.loads((out_ass.parent / "breakout_audit.json").read_text())
                for _e in (_aud.get("accepted") or []):
                    if int(_e.get("seg_index", -1)) == int(cap.get("seg_index", -2)):
                        _aired_toks = {re.sub(r"[^a-z0-9']", "", t.lower())
                                       for t in re.findall(r"[\w']+",
                                                           _e.get("aired_transcript") or "")}
                        break
            except Exception:                                # noqa: BLE001
                _aired_toks = set()
        _FUNC = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
                 "is", "it", "he", "she", "i", "you", "we", "so", "now"}

        def _word_ok(gi_ids, li, w):
            """Is this word vouched for by evidence stronger than its own ASR confidence?"""
            i = gi_ids[li] if li < len(gi_ids) else -1
            if 0 <= i < len(_bk_srcok) and _bk_srcok[i]:
                return True                       # the verified source line says this word
            _n = re.sub(r"[^a-z0-9']", "", w[0].lower())
            if _n in _aired_toks and w[3] >= _aconf:
                return True                       # heard the same both passes + not hopeless
            return _n in _FUNC and len(_n) <= 3   # semantically inert connective
        for _gi, line in enumerate(grp):
            _minp = min(w[3] for w in line)
            if _minp >= _pfloor:
                _kept.append(line)
                continue
            _ids = _gidx[_gi] if _gi < len(_gidx) else []
            _backed = bool(_ids) and all(
                (i < len(_bk_srcok) and _bk_srcok[i]) for i in _ids)
            _content = sum(1 for w in line if len(re.sub(r"[^\w']", "", w[0])) > 1)
            if _src_strong and _backed and _content >= 2:
                _kept.append(line)
                _rescued += 1
                if log:
                    log(f"build: breakout caption line kept — every word matches the verified "
                        f"source line (align {_bk_align:.2f}, min ASR conf {_minp:.2f}): "
                        f"{' '.join(w[0] for w in line)!r}")
                continue
            if _repair_on and _ids:
                # trim the unbacked FRINGE (words past the quote / before it), then keep the
                # line iff every REMAINING word is source/aired/function-backed
                _oks = [_word_ok(_ids, li, w) for li, w in enumerate(line)]
                lo, hi = 0, len(line)
                while lo < hi and not _oks[lo]:
                    lo += 1
                while hi > lo and not _oks[hi - 1]:
                    hi -= 1
                _rem = line[lo:hi]
                _rcontent = sum(1 for w in _rem if len(re.sub(r"[^\w']", "", w[0])) > 1)
                if (_rem and all(_oks[lo:hi]) and _rcontent >= 2):
                    _kept.append(_rem)
                    _rescued += 1
                    if log:
                        _cut = len(line) - len(_rem)
                        log(f"build: breakout caption line repaired — kept {len(_rem)} "
                            f"backed word(s), trimmed {_cut} unbacked (min ASR conf "
                            f"{_minp:.2f}): {' '.join(w[0] for w in _rem)!r}")
                    continue
            if log:
                log(f"build: breakout caption line dropped (ASR word confidence "
                    f"{_minp:.2f} < {_pfloor}): {' '.join(w[0] for w in line)!r}")
        # ORPHAN CLEANUP: if the survivors caption almost none of what is spoken, a lone fragment
        # ("my son.") floats context-free over a long breakout and reads as a non-sequitur —
        # captioning NOTHING is the cleaner cut (documentaries do not caption screams).
        try:
            _mincov = float(_os_bkc.environ.get("VIDLORE_CLIPSTUDIO_BK_CAP_MIN_COVERAGE",
                                                "0.35") or 0.35)
        except (TypeError, ValueError):
            _mincov = 0.35
        if _repair_on and _kept and words:
            _covw = sum(len(l) for l in _kept) / float(len(words))
            if _covw < _mincov:
                if log:
                    log(f"build: breakout captions suppressed — only {_covw:.0%} of spoken words "
                        f"survived the confidence gate (< {_mincov:.0%}); an orphan fragment "
                        f"reads as a non-sequitur")
                _kept = []
        grp = _kept
        _captioned_words = [w for _line in grp for w in _line]
        _cov = caption_coverage(_spoken_words, _captioned_words)
        cap["caption_coverage"] = _cov
        _coverage_rows.append({"seg_index": cap.get("seg_index"),
                               "final_index": cap.get("final_index"), **_cov})
        if not _cov["passed"]:
            _coverage_failed = True
            if log:
                log(f"build: ⛔ breakout caption coverage — scene {cap.get('seg_index')} "
                    f"{_cov['captioned_words']}/{_cov['spoken_words']} words "
                    f"({_cov['coverage']:.0%}), uncaptioned tail "
                    f"{_cov['uncaptioned_tail_s']:.2f}s; breakout cannot publish")
            continue
        for line in grp:
            ws = base + line[0][1]
            we = min(base + dur, base + line[-1][2] + 0.10)
            if we <= ws:
                continue
            toks = [w[0] for w in line]
            kcs = [max(6, int(round((w[2] - w[1]) * 100))) for w in line]   # karaoke cs per word
            # grapheme-split any pathological unbroken word, then lay the karaoke line into ≤2 rows
            cells, imap = split_wide_cells(toks, _bk_size, _safe_w, peak_extra=0.0, pad=_pad)
            _bidx, _fit, _squeeze = ((None, _bk_size, 100) if _bk_lines < 2 else
                                     layout_two_lines(cells, _bk_size, _safe_w, peak_extra=0.0, pad=_pad))
            _cellcount = {}
            for j in imap:
                _cellcount[j] = _cellcount.get(j, 0) + 1
            parts = []
            for ci, cell in enumerate(cells):
                j = imap[ci]
                # a split word's karaoke fill sweeps evenly across its cells (total sweep == word cs)
                cs = max(3, int(round(kcs[j] / _cellcount[j])))
                safe = cell.replace("{", "(").replace("}", ")").replace("\\", "")
                parts.append(f"{{\\kf{cs}}}{safe}")
            if _bidx is None:
                body = " ".join(parts)
            else:
                body = " ".join(parts[:_bidx]) + "\\N" + " ".join(parts[_bidx:])
            # bounded \fs shrink then, for a pathological unbroken word, a last-resort \fscx squeeze —
            # so the karaoke line always fits its two rows without a clipped third row or truncation
            _fs = f"\\fs{_fit:.0f}" if _fit < _bk_size - 0.5 else ""
            _sq = f"\\fscx{_squeeze}" if _squeeze < 100 else ""
            lines.append(f"Dialogue: 0,{_ass_ts(ws)},{_ass_ts(we)},BK,,0,0,0,,"
                         f"{{\\fad(120,120){_fs}{_sq}}}{body}")
    # Completeness evidence is mandatory and is written even when the gate fails.  A dialogue
    # Breakout with a plausible caption prefix but an uncaptained tail is not publishable.
    try:
        import json as _json_bkcov
        _cov_path = Path(out_ass).parent / "breakout_caption_coverage.json"
        _cov_path.write_text(_json_bkcov.dumps({
            "schema": "breakout_caption_coverage/2", "passed": not _coverage_failed,
            "minimum_word_coverage": 1.0, "maximum_uncaptioned_tail_s": 0.50,
            "breakouts": _coverage_rows,
        }, indent=1), encoding="utf-8")
    except Exception:
        return None
    if _coverage_failed or not lines:
        return None
    # BK Style line from the (already-resolved) preset — karaoke fill sweeps unsung → sung.
    _bk_style = preset.breakout_style_line()
    header = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nPlayResX: 1920\nPlayResY: 1080\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding\n"
        f"{_bk_style}\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    try:
        Path(out_ass).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
        if log:
            log(f"build: breakout captions — {len(lines)} word-by-word line(s)")
        return out_ass
    except Exception:
        return None


def _burn_breakout_captions(video: Path, caps: list, work: Path, log=None, *, preset=None,
                            ass_path: Optional[Path] = None) -> bool:
    """Burn the word-by-word breakout captions onto the final video (engine untouched). `preset`
    (a CaptionPreset) styles the burn to match the selected design family."""
    if not caps:
        return False
    ass = (Path(ass_path) if ass_path and Path(ass_path).exists() else
           _breakout_caption_ass(caps, work / "breakout_caps.ass", log, preset=preset))
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
        ass.unlink(missing_ok=True)                    # nothing burned → keep it OUT of the
        return False                                   # ad-gate's own-caption whitelist
    if p.returncode == 0 and out.exists() and out.stat().st_size > 0:
        Path(out).replace(video)                       # same dir — atomic, no `os` needed
        return True
    if log:
        log(f"build: breakout-caption burn failed ({(p.stderr or b'')[-180:]!r})")
    # the burn FAILED: these lines were never rendered onto the video, so the ASS file must not
    # feed _own_caption_schedule (it would whitelist screen text that can only be the footage's).
    ass.unlink(missing_ok=True)
    return False


def wqc_render_window(shots, start: float, need: float, seg=None, moment=None,
                      src_dur=None):
    """PRODUCTION render-window QC (module-level so tests drive the real path). Validates the
    definitive aired window [start, start+need] against the source's indexed shots; when dirty,
    searches the immediate same-source neighbourhood for a clean span of the full length.

    Moment policy (match.wqc_moment_policy): 'exact' beats pass `moment` — the ORIGINALLY
    SELECTED candidate range [in, out], NOT the padded render window — as `preserve`: a
    shifted window must keep that moment's midpoint or a strong overlap, else the answer is
    kept-dirty (a dirty exact moment beats a clean DIFFERENT moment). With no `moment` (the
    playhead-walk fill — a mechanical start, not a selected moment) and for 'generic' beats,
    any clean span overlapping the original window may air.

    Partial-corner evidence deliberately excluded (candle-sconce FP — see match.py).
    Returns (new_start, action, why, meta); action ∈ ok / shifted / kept-dirty.
    meta audit fields: `candidate` = the originally SELECTED [in, out] (None on walk-fill),
    `render_request` = the padded/full requested aired window [start, start+need],
    `final` = what actually airs, `preserved` = does the FINAL aired window still show the
    candidate moment — ALWAYS computed against the candidate, never the padded request, and
    None (logged n-a) whenever the exact-moment guarantee does not apply (generic policy or
    no candidate). kept-dirty therefore means cleanliness failed, NOT that the moment was
    lost: an unchanged dirty window that still contains its candidate reports preserved=True."""
    from .match import clean_cut_window as _ccw, wqc_moment_policy as _wpol, _moment_kept
    pol = _wpol(seg)
    orig = (float(start), float(start) + float(need))
    cand = tuple(moment) if moment is not None else None
    preserve = cand if (pol == "exact" and cand is not None) else None
    meta = {"policy": pol, "candidate": cand, "render_request": orig, "final": orig,
            "preserved": (_moment_kept(orig[0], orig[1], preserve)
                          if preserve is not None else None)}
    if not shots:
        return start, "ok", "", meta           # nothing indexed → fail-open
    nt0, nt1, act, why = _ccw(shots, start, start + need, need, anchor=orig,
                              preserve=preserve)
    if act == "ok":
        return start, "ok", "", meta
    if act != "shortened":
        # no clean span INSIDE the window — widen the search to the immediate neighbourhood
        # (same source, and the span must still OVERLAP the original window so the aired
        # moment stays in the chosen scene's vicinity). Observed: a beat @93.9 padded into
        # the next shot's Turkish sub while a 4.8s clean span sat directly BEFORE it.
        # The search end is CLAMPED to the indexed extent: clean_cut_window counts
        # un-indexed time as clean, so an unclamped search could "shift" past the last
        # shot (≈ the source's end) and the caller's duration clamp would drag the aired
        # window straight back into the dirt it just dodged — while the log claimed a
        # clean shift. Never clamped below the original window's end: the caller already
        # bounds that to the source duration, and a partially-indexed tail stays fail-open.
        idx_end = max(float(sh.end) for sh in shots)
        nt0, nt1, act, why = _ccw(shots, max(0.0, start - need),
                                  min(start + 2.0 * need, max(idx_end, start + need)),
                                  need, anchor=orig, preserve=preserve)
    if act == "shortened":                     # a clean moment-keeping span ≥ need → SHIFT
        if preserve is not None:
            # the validated clean window [nt0, nt1] can be LONGER than `need`, but only
            # [ns, ns+need] airs — position the aired slice INSIDE the clean window so it
            # still shows the selected moment, and re-check on the aired length
            pm = (preserve[0] + preserve[1]) / 2.0
            ns = max(nt0, min(pm - need / 2.0, nt1 - need))
            if not _moment_kept(ns, ns + need, preserve):
                # shift REFUSED — the ORIGINAL window airs, and meta['preserved'] already
                # says whether THAT window shows the candidate (the refused shift never airs)
                return start, "kept-dirty", f"moment-lost: {why}", meta
        else:
            # generic / walk-fill: the anchor-overlap rule was checked against the FULL
            # clean span — airing the span's raw head could miss the original window
            # entirely. Position the aired slice at the span point nearest the original
            # start so what AIRS still overlaps the window it was cleared against.
            ns = min(max(nt0, orig[0]), nt1 - need)
        # mirror the caller's post-QC duration clamp so `final=` is what actually AIRS:
        # indexed shot ends can exceed the integer-rounded source-duration metadata, so a
        # shift into the index's tail gets dragged back by the caller — the un-mirrored
        # log claimed a clean shift ([ns, ns+need]) that never aired. The mirror is
        # DECISION-NEUTRAL: the caller applies the identical clamp to whatever we return.
        _d = float(src_dur or 0.0)
        if _d > 0 and ns + need > _d:
            ns = max(0.0, _d - need)
            why = f"{why}; duration-clamped"
            if abs(ns - orig[0]) <= 1e-9:
                # the shift evaporated to EXACTLY the original start — the ORIGINAL
                # (dirty) window airs unchanged: that is a kept-dirty in truth, whatever
                # the span search found (exact equality only — the caller returns the
                # original start on kept-dirty, so anything else would change the aired
                # value; near-misses stay 'shifted' with the truthful clamped final)
                if preserve is not None:
                    meta["preserved"] = _moment_kept(orig[0], orig[1], preserve)
                return start, "kept-dirty", why, meta
        if preserve is not None:
            # recompute on the slice that ACTUALLY airs (the clamp may have moved it)
            meta["preserved"] = _moment_kept(ns, ns + need, preserve)
        meta["final"] = (ns, ns + need)
        return ns, "shifted", why, meta
    # kept-dirty: the unchanged original window airs — meta['preserved'] keeps the honest
    # answer computed against the candidate (cleanliness failed ≠ moment lost)
    return start, "kept-dirty", why, meta


def wqc_render_log_line(act: str, meta: dict, why: str) -> str:
    """Build-stage window-QC audit tail with DISTINCT truthful fields:
    candidate= the originally selected [in, out] (none on walk-fill) ·
    render-request= the padded/full requested aired window ·
    final= what actually airs ·
    moment-preserved= yes|no|n-a — computed against the CANDIDATE, never the padded
    request; n-a whenever the exact-moment guarantee does not apply · action=."""
    c, r, f = meta["candidate"], meta["render_request"], meta["final"]
    kept = meta["preserved"]
    return (f"policy={meta['policy']} "
            f"candidate={f'[{c[0]:.1f}-{c[1]:.1f}]' if c is not None else 'none'} "
            f"render-request=[{r[0]:.1f}-{r[1]:.1f}] final=[{f[0]:.1f}-{f[1]:.1f}] "
            f"moment-preserved={'n-a' if kept is None else ('yes' if kept else 'no')} "
            f"action={act} reason={why}")


def build_video(proj: ClipProject, segments: list[ScriptSegment], cfg: ClipConfig, *,
                voice: str = "", captions: Optional[bool] = None, title: str = "",
                theme_name: str = "history", music: Optional[str] = None,
                voiceover: Optional[str] = None, voice_provider: str = "",
                voice_preset: str = "", caption_style: str = "",
                use_tts: bool = True, progress=None) -> Path:
    """Render the matched clips into a final MP4 via the engine's assemble().

    `captions` toggles ALL visible caption burn (narration + breakout); `caption_style` names the
    caption PRESET (professional|minimal|cinematic|documentary|focus — see caption_presets). Both
    fall back safely: env VIDLORE_CLIPSTUDIO_CAPTIONS / _CAPTION_STYLE override, an unknown style
    logs a warning and uses 'professional'. Captions OFF removes every visible caption layer but
    NEVER touches breakout timing / suppression / audit / QA metadata."""
    from vidlore.script_gen import Script, Scene
    from vidlore.footage import FootageItem
    from vidlore.themes import theme as get_theme
    from vidlore.assemble import assemble
    from vidlore.config import load_config

    def log(m):
        if progress:
            progress(m)

    eng = load_config()
    # WHY a review draft must be visible ON THE FILE. In warn mode a release-BLOCKED render is
    # still written out so it can be watched and audited — but it was written to `final.mp4`, the
    # same name a passing render uses, with no marker anywhere in the picture. The only signals
    # were build.log, review.html's badges and the portal's job status, none of which travel with
    # the mp4. One duly reached another machine on a USB stick and was read as a finished render
    # (job 957f56f925: 7 unresolved beats, ~46% relevance). Each warn-mode gate appends its reason
    # here; the file is renamed at the end, and the caller is handed the new path.
    _review_draft: list = []
    _aired_windows: list = []      # what each beat ACTUALLY cut — see the note at the append
    import os
    os.environ.setdefault("VIDLORE_MUSIC_VOLUME", "1.15")   # present cinematic bed under the VO
    work = proj.output_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    out_path = proj.output_dir / "final.mp4"
    sel_by_idx = {s.segment_index: s for s in proj.selections}

    # SEMANTIC PUBLICATION CONTRACT — run before caption setup, narration, breakouts, recuts or
    # encoding. Resume/rerender and --no-verify paths can otherwise consume stale project.json
    # selections whose persisted verifier explicitly says mismatch/insufficient/wrong/unverified.
    # The audit is atomic and the gate never ranks or mutates footage; generic/abstract filler is
    # deliberately outside its strict scope.
    from .relevance_contract import assert_selection_relevance as _assert_selection_relevance
    try:
        _assert_selection_relevance(
            proj, segments, proj.output_dir / "selection_relevance_audit.json", cfg=cfg)
    except RuntimeError as _semantic_preflight_error:
        _review_mode = os.environ.get(
            "VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block").strip().lower() == "warn"
        if (_review_mode
                and getattr(_semantic_preflight_error, "kind", "") == "selection_relevance"):
            _review_draft.append(f"selection relevance: {_semantic_preflight_error}")
            log("build: ⚠ SEMANTIC PREFLIGHT (mode=warn, REVIEW BUILD — not for publication) — "
                f"{_semantic_preflight_error} Continuing only because review mode was explicitly "
                "requested; the delivered filename will be marked REVIEW_DRAFT.")
        else:
            raise

    # IMAGE PUBLICATION PREFLIGHT — index keyframes are intentionally 512px CLIP thumbnails. A
    # source-frame still must therefore be materialized from its indexed owner at native pixels
    # before narration alignment, breakout judging, or encoding. Three strict stills in a 101-beat
    # run otherwise failed sequentially only after 28 minutes of breakout work. A SHA-bound
    # thumbnail gets a fresh verdict on the exact native extraction; its old verdict is never
    # transferred. Keep the rescue in memory so project/checkpoint semantics do not change, then
    # revalidate every source/keyframe/question fingerprint when the image actually enters air.
    _image_lineage_entries: list[dict] = []
    _image_lineage_failures: list[dict] = []
    _preflight_image_rescues: dict[int, dict] = {}
    _semantic_rematerialized = 0
    for _seg_img_pf in segments:
        _sel_img_pf = sel_by_idx.get(_seg_img_pf.index)
        _img_pf = str(getattr(_sel_img_pf, "image_path", "") or "") \
            if _sel_img_pf is not None else ""
        if not _img_pf:
            continue
        try:
            _rescue_pf = _rescue_still_fullres(
                proj, _sel_img_pf, _img_pf, log, seg=_seg_img_pf, eng=eng)
            _preflight_image_rescues[id(_sel_img_pf)] = _rescue_pf
            _semantic_rematerialized += int(
                _rescue_pf.get("semantic_rematerialized") is True)
        except Exception as _img_pf_exc:                 # noqa: BLE001 — persist then fail closed
            _image_lineage_failures.append({
                "final_scene": int(_seg_img_pf.index),
                "original_beat": int(getattr(
                    _sel_img_pf, "segment_index", _seg_img_pf.index)),
                "declared_path": _img_pf,
                "expected_source_id": str((getattr(
                    _sel_img_pf, "image_meta", {}) or {}).get("src", "") or ""),
                "expected_shot_index": (getattr(
                    _sel_img_pf, "image_meta", {}) or {}).get("shot"),
                "preflight": True,
                "reason": str(_img_pf_exc),
            })
            _persist_image_lineage_audit(
                proj, _image_lineage_entries, _image_lineage_failures)
            if not content_defect_is_deliverable(_img_pf_exc):
                raise
            # Quality defect on ONE beat in a review draft: drop the still and let the beat air its
            # moving selection, exactly as if no still had been proposed.
            #
            # This KNOWINGLY relaxes the rule stated at the aired-still site below — "a declared
            # image is a semantic replacement for the moving selection... falling through would air
            # the very moving clip this fallback replaced (often a verifier rejection)". That rule
            # is right for PUBLICATION and is unchanged there. A review draft already airs
            # verifier-rejected footage on the beats the footage gate warned about; refusing to
            # deliver the same draft because one still could not be re-extracted at full
            # resolution is the same inconsistency the black-frame gate note describes. The beat is
            # named in the log, in _review_draft, and in image_lineage_audit.json — never silent.
            log(f"build: \u26a0 REVIEW DRAFT — beat {_seg_img_pf.index} still unusable "
                f"({_img_pf_exc}); dropping the still, the beat keeps its moving clip")
            _review_draft.append(
                f"image still beat {_seg_img_pf.index}: {_img_pf_exc}")
            try:
                _sel_img_pf.image_path = ""
            except Exception:                            # noqa: BLE001
                pass
    if _preflight_image_rescues:
        log(f"build: image preflight — {len(_preflight_image_rescues)} native still(s) "
            f"materialized; {_semantic_rematerialized} low-resolution semantic thumbnail(s) "
            "freshly reverified on exact HD bytes")

    # CAPTION PRESET + ON/OFF — the single resolution point for the whole render (every downstream
    # caption gate reads _cap_on / _cap_preset, never the raw args). Precedence via the centralized
    # resolvers: an EXPLICIT value wins; the env var is a fallback only when nothing was supplied
    # (captions=None); default = captions ON + professional.
    from .caption_presets import (resolve_style as _resolve_cap_style,
                                   captions_enabled as _captions_enabled)
    _cap_on = _captions_enabled(captions)
    _cap_preset, _cap_invalid = _resolve_cap_style(caption_style)
    if _cap_invalid:
        # the offending value may have come from the arg OR the env — report whichever was set
        _bad = caption_style if (caption_style and str(caption_style).strip()) \
            else os.environ.get("VIDLORE_CLIPSTUDIO_CAPTION_STYLE", "")
        log(f"build: invalid caption style {str(_bad)!r} — using {_cap_preset.name}")
    log(f"build: captions={'on' if _cap_on else 'off'} "
        f"style={_cap_preset.name if _cap_on else 'none'}")
    # persist ATOMICALLY to project.json so a rebuild (rerender_project) reproduces the exact design.
    # The orchestrator's last proj.save() runs BEFORE build, so build_video owns this write.
    try:
        proj.meta["caption_settings"] = {"enabled": bool(_cap_on), "style": _cap_preset.name}
        proj.save()
    except Exception as _e:                                # noqa: BLE001
        log(f"build: caption-settings persist skipped ({str(_e)[:60]})")

    # 1) engine Script from our segments (narration text per scene)
    scenes = [Scene(index=seg.index, narration=seg.text,
                    keywords=seg.keywords, visual=seg.expected_visual)
              for seg in segments]
    _assign_editorial(scenes, segments)               # role / intensity / emphasis / shot_type
    script = Script(title=title or proj.name, scenes=scenes)

    # 2) narration — user voiceover (forced-aligned) > edge-tts > silent fallback
    narration = None
    # A caller that NAMES a voiceover wants that voice. A typo'd/missing path used to slide all
    # the way down to the silent fallback and render a full-length video with no narration at all
    # (captions and music only) — an hours-long failure that looks like a success. Fail loudly and
    # immediately instead: the fallbacks below exist for TTS outages, not for caller mistakes.
    if voiceover and not Path(voiceover).exists():
        raise FileNotFoundError(
            f"voiceover file not found: {voiceover} — refusing to render a silent-narration video. "
            f"Pass a real path, or pass voiceover=None to use TTS.")
    if voiceover and Path(voiceover).exists():
        try:
            # caption-sync FIRST: per-scene-tolerant word alignment (the engine's all-or-nothing
            # gate silently drifts long voiceovers). Falls back to the engine path if it can't align.
            narration = _synced_narration_from_file(script, str(Path(voiceover).resolve()), work / "vo", log)
            if narration is not None:
                log(f"build: narration {narration.total:.1f}s (user voiceover, word-synced captions)")
            else:
                from .verify import NonRetryableBuildError
                raise NonRetryableBuildError(
                    "uploaded voiceover could not produce a complete positive-duration word "
                    "alignment; refusing proportional/drifting captions or a substituted TTS "
                    "voice", kind="voiceover_alignment")
            if narration is not None:
                # mark word-aligned uploaded VO — the cold-open VO word-cut needs REAL word
                # boundaries (whisper-aligned), never the proportional estimates of a TTS render.
                try:
                    narration._vo_word_aligned = True
                except Exception:
                    pass
        except Exception as e:                            # noqa: BLE001
            from .verify import NonRetryableBuildError
            if isinstance(e, NonRetryableBuildError):
                raise
            log(f"build: ⛔ voiceover alignment failed ({str(e)[:90]}) — render blocked")
            raise NonRetryableBuildError(
                f"voiceover alignment failed: {e}; refusing to replace the supplied voice or "
                f"publish drifting/invalid captions", kind="voiceover_alignment") from e
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
    if narration is None and voiceover and not use_tts:
        # the voiceover was found but could not be turned into narration, and TTS is off — there is
        # no voice left to fall back to. Same reasoning as the missing-file guard above.
        raise RuntimeError(
            f"voiceover {voiceover} could not be aligned and use_tts=False — refusing to render a "
            f"silent-narration video. See the 'voiceover align failed' line above for the cause.")
    if narration is None:
        narration = _silent_narration(segments, work / "silent", cfg)
        log(f"build: narration {narration.total:.1f}s (silent fallback)")
    # ASR-spelled character names ('Alina Tyrell') would burn into every caption — canonicalize
    # against the analysis cast list (1:1 token rewrite; timings untouched). Env-gated default ON.
    if os.environ.get("VIDLORE_CLIPSTUDIO_CAPTION_NAME_FIX", "1").strip() \
            not in ("0", "false", "no"):
        try:
            _script_caption_tokens = [
                word for scene in (getattr(script, "scenes", []) or [])
                for word in (getattr(scene, "narration", "") or "").split()
            ]
            _proper_caption_terms = set()
            for _seg_term in segments:
                for _entity_term in ((getattr(_seg_term, "entities", None) or [])
                                     + [getattr(_seg_term, "required_entity", "") or ""]):
                    for _term_token in re.findall(r"[A-Za-z][A-Za-z'’-]*", str(_entity_term)):
                        if len(_term_token) >= 4 and _term_token[:1].isupper():
                            _proper_caption_terms.add(_term_token)
            _restore_secure_script_tokens(
                narration, _script_caption_tokens, log,
                protected_terms=_proper_caption_terms)
            _canonicalize_caption_names(
                narration, proj, log,
                script_text=" ".join(_script_caption_tokens))
        except Exception as _e_cn:                        # noqa: BLE001
            log(f"build: caption name canonicalization skipped ({str(_e_cn)[:60]})")

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
    _bidx: dict = {}          # {ORIGINAL beat index -> FINAL post-breakout scene index}; {} = identity
    _breakout_entries: list = []
    if os.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUTS", "1").strip() not in ("0", "false", "no"):
        try:
            _bks = _select_breakouts(
                proj, segments, getattr(narration, "total", 0.0), work, log, cfg=cfg)
        except Exception as e:                            # noqa: BLE001
            log(f"build: breakout selection skipped ({str(e)[:80]})")
            _bks = []
        # ONE tested composition entrypoint (mids → cold-open, artifact composition, invariant,
        # rollback). Everything downstream keys off its return values; nothing composes maps inline.
        segments, scenes, narration, _breakout_clip, _bidx, _breakout_entries = _compose_breakouts(
            proj, segments, scenes, narration, _bks, work, _cap_on, log=log)
        if _bidx:
            sel_by_idx = {_bidx[s.segment_index]: s for s in proj.selections
                          if s.segment_index in _bidx}

    # NATIVE-HD PUBLICATION CONTRACT.  This probes the bytes on disk, not requested/download
    # metadata: a 640x360 fallback inside a 1080p container is still 360p.  Matching may continue
    # to value relevance first, but build may only publish when every selected moving-video root is
    # at least 720p.  A verified full-resolution still has its own rescue/validation path below.
    from .quality_contract import assert_native_hd_selections as _assert_native_hd
    _assert_native_hd(proj, proj.selections, proj.output_dir / "native_resolution_audit.json")

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
    # IMMUTABLE ROOT OWNERSHIP.  Every file that can enter ``beat_clips`` must be registered here,
    # then every derivative inherits the same root.  The final manifest is independently checked
    # before assemble, and assemble additionally fingerprints decoded frames before/after concat.
    # A filename or a truthful-looking audit label is never accepted as provenance.
    from .scene_lineage import (assert_scene_lineage as _assert_scene_lineage,
                                selection_binding as _selection_binding)
    _lineage_roots: dict[str, dict] = {}
    _final_to_orig_lineage = {v: k for k, v in (_bidx or {}).items()}
    _breakout_cap_by_final = {
        int(c["final_index"]): c for c in
        (getattr(narration, "_breakout_caps", None) or [])
        if c.get("final_index") is not None
    }

    def _lineage_key(path) -> str:
        return str(Path(path).expanduser().resolve())

    def _lineage_register(path, root: dict) -> None:
        if path:
            _lineage_roots[_lineage_key(path)] = dict(root)

    def _lineage_derive(path, parent, *, via: str = "selection_derivative",
                        selection_source_compare_filter: str = "",
                        extra: dict | None = None) -> bool:
        """Register ``path`` as a derivative of ``parent``; unknown roots fail later, never infer."""
        inherited = _lineage_roots.get(_lineage_key(parent)) if parent else None
        if not inherited or not path:
            return False
        derived = {**inherited, "via": via}
        if extra:
            # Only a SANCTIONED derivative may add facts of its own (an editorial hold declaring
            # whose frame it froze). The inherited root fields stay authoritative underneath, so a
            # derivative can annotate its provenance but never rewrite it.
            derived.update(extra)
        if selection_source_compare_filter:
            # This is not an exemption from decoded lineage comparison.  It tells the independent
            # canary to apply the exact build-authorized corner crop to the source-window bank too,
            # so removing a channel bug cannot masquerade as a foreign-scene swap.
            derived["selection_source_compare_filter"] = selection_source_compare_filter
        _lineage_register(path, derived)
        return True

    def _selection_root(sel, final_scene: int) -> dict:
        orig = int(getattr(sel, "segment_index", _final_to_orig_lineage.get(
            final_scene, final_scene)))
        selected = [str(getattr(sel, "source_id", "") or ""),
                    round(float(getattr(sel, "in_point", 0.0) or 0.0), 3),
                    round(float(getattr(sel, "out_point", 0.0) or 0.0), 3)]
        binding = _selection_binding(
            orig, selected[0], selected[1], selected[2],
            getattr(sel, "verifier", None) or {})
        _src_owner = proj.source(selected[0]) if selected[0] else None
        _src_owner_path = Path(str(getattr(_src_owner, "local_path", "") or "")) \
            if _src_owner is not None else Path("")
        _src_owner_file = (str(_src_owner_path.expanduser().resolve())
                           if _src_owner_path.is_file() and _src_owner_path.stat().st_size > 0
                           else "")
        return {
            "kind": "selection_video", "original_beat": orig, "owner_beat": orig,
            "selected_source_id": selected[0], "actual_source_id": selected[0],
            "selected_window": selected, "actual_window": selected,
            # The derivative path alone is not proof: a stale beat_034_owned.mp4
            # can carry foreign bytes under the expected filename.  The renderer
            # independently compares it with this exact source window at bind time.
            "selection_source_path": _src_owner_file,
            "selection_binding": binding, "root_binding": binding,
            "via": "selection", "validated": True,
        }
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
    from .match import _ocr_text_heavy as _txt_heavy, _shot_unreadable as _dark_b
    _TGATE = os.environ.get("VIDLORE_CLIPSTUDIO_TEXT_GATE", "1").strip() not in ("0", "false", "no", "")

    def _shot_at(sid, t):
        for _sh in _shots_b(sid):
            if _sh.start <= float(t) < _sh.end:
                return _sh
        return None

    def _win_unreadable(sid, t, out=None):
        """Is the aired window unreadable? Validate the ENTIRE rendered source window [t, out], not
        only its in-point shot — the padded window can walk from a legible in-point into an
        unreadable adjacent shot (the murk that aired mid-beat)."""
        t = float(t)
        hi = float(out) if (out is not None and float(out) > t) else t + 0.01
        for _sh in _shots_b(sid):
            if _sh.end > t + 0.03 and _sh.start < hi - 0.03 and _dark_b(_sh):
                return True
        _sh0 = _shot_at(sid, t)                         # also the in-point shot itself
        return bool(_sh0 is not None and _dark_b(_sh0))

    # CUT-WINDOW FLAG VALIDATION at render time — the playhead walk / lead-in snapping computes
    # the DEFINITIVE aired window [start, start+need] here, after every earlier stage. Shift the
    # start into a clean same-scene span when the window overlaps a flagged shot (persisted
    # multi-frame flags: burned subs / unreadable murk / partial corner-logo evidence).
    _WQC = os.environ.get("VIDLORE_CLIPSTUDIO_WINDOW_QC", "1").strip() not in ("0", "false", "no")
    _wqc_render_stats = {"shifted": 0, "kept-dirty": 0}

    def _wqc_render_start(sid, start, need, seg, moment=None):
        if not _WQC or need <= 0:
            return start
        shots = _shots_b(sid)
        if not shots:
            return start
        _sdur = float(getattr(proj.source(sid), "duration", 0) or 0)
        nstart, act, why, meta = wqc_render_window(shots, start, need, seg, moment, _sdur)
        if act == "ok":
            return start
        seg_idx = getattr(seg, "index", "?")
        if act == "shifted":
            _wqc_render_stats["shifted"] += 1
            log(f"window-qc: shifted beat={seg_idx} src={sid[:28]} "
                f"{wqc_render_log_line(act, meta, why)}")
            return nstart
        _wqc_render_stats["kept-dirty"] += 1
        log(f"window-qc: kept-dirty beat={seg_idx} src={sid[:28]} "
            f"{wqc_render_log_line(act, meta, why)}")
        return start

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

    # WHOLE-TIMELINE look cap: the near-term guard above only looks back _HASH_WIN beats, so the
    # SAME picture re-aired from a DIFFERENT source-rip 78 / 28 beats apart (the burning-field b-roll
    # aired 3×). Count how many times a look has aired across the ENTIRE video (ahash cluster,
    # hamming <= 8) and prefer never-/less-aired looks, capping any one look at _LOOK_CAP airings when
    # an alternative exists (never a hard drop of the last option — coverage/moment-lock preserved).
    _LOOK_CAP = int(_cfg_i("VIDLORE_CLIPSTUDIO_LOOK_CAP", 2))

    def _look_aired_count(h):
        if h is None:
            return 0
        n = 0
        for prev in _aired_hashes:
            try:
                if prev is not None and int(_np.sum(prev != h)) <= 8:
                    n += 1
            except Exception:
                pass
        return n

    def _next_distinct_shot(sid, after_t, need: float = 0.0):
        # SHOT-AWARE walk: the next detected shot boundary at/after `after_t` whose LOOK is not
        # a near-term repeat — raw seconds-walking inside one static take produced visually
        # identical consecutive beats (a council room doesn't change in 3 seconds; the camera
        # CUT does). Returns a start time, falling back to after_t.
        #
        # `need` (the beat's cut length) enables the live legibility probe below. Of the freeze
        # replacements in the measured render, roughly 38% came through this walk rather than
        # through a chosen beat_window, so the window-level check alone leaves them uncovered.
        _lg_walk = os.environ.get("VIDLORE_CLIPSTUDIO_WINDOW_LEGIBILITY", "1").strip() \
            not in ("0", "false", "no")
        _lg_left = 6                    # bounded: 6 live probes, then today's behaviour verbatim
        try:
            src_obj = proj.source(sid)
            sp = src_obj.local_path if src_obj else None
            for sh in _shots_b(sid):
                if sh.end - sh.start < 1.2 or sh.start < after_t - 0.25:
                    continue
                if _TGATE and _txt_heavy(sh):
                    continue                           # burned-in text never airs
                if _dark_b(sh):
                    continue                           # near-black/unreadable never airs (Gap 3)
                try:
                    if int(getattr(sh, "graphics_flag", -1) or -1) >= 2:
                        continue                       # designed graphics/illustration never airs
                except (TypeError, ValueError):
                    pass
                if sp and _looks_recent(_frame_hash(sp, min(sh.start + 1.2, sh.end - 0.2))):
                    continue
                if _TGATE and sp and (
                        _frame_has_burned_text(
                            sp, min(sh.start + 1.0, max(sh.start, sh.end - 0.4)))
                        or _frame_has_burned_text(
                            sp, min(sh.start + 2.2, max(sh.start, sh.end - 0.4)))):
                    continue                           # air-time probe (keyframe can miss)
                # LIVE darkness probe of the span this walk would actually air. `_dark_b` above
                # reads the persisted SHOT AGGREGATE (3 samples over up to 18s), so it is blind by
                # construction to a sustained dark run inside an otherwise-bright shot — which is
                # the run the post-cut sweep then freezes over.
                if _lg_walk and sp and need > 0 and _lg_left > 0:
                    _lg_left -= 1
                    if _source_window_too_dark(sp, max(after_t, sh.start), need):
                        continue
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

    # ---- EARLY RELEASE-GATE DRY-RUN (fail-fast; PROVABLY a subset of the authoritative gate) ----
    # The authoritative rejected-footage gate below runs only AFTER the full beat-encode loop
    # (~11-22 min), yet a doomed render can be known NOW: a rejected beat whose every possible
    # hold predecessor fails _hold_scene_compat can never be resolved by any encode outcome —
    # compat failure forces _hold_block_reason regardless of durations/consec state, and encode
    # failures only REMOVE predecessor candidates (they can never turn an all-fail set into a
    # pass). Beats where ANY candidate passes are left to the real gate (which stays the final
    # word, unchanged). Honors the same kill-switch and warn mode as the real gate.
    import os as _os_eg
    if (_os_eg.environ.get("VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE", "1").strip()
            not in ("0", "false", "no")
            and _os_eg.environ.get("VIDLORE_CLIPSTUDIO_EARLY_RF_GATE", "1").strip()
            not in ("0", "false", "no")):
        _an_e = (proj.meta.get("analysis", {}) or {})
        _sing_e = _an_e.get("video_type", "") == "single_scene"
        _era_e = str(_an_e.get("episode_hint", "") or "")
        _ovl_e = _cfg_f("VIDLORE_CLIPSTUDIO_HOLD_SCENE_OVERLAP", 0.4)
        _c2a_e = {str(c.get("name", "")): str(c.get("actor", ""))
                  for c in (_an_e.get("characters") or []) if isinstance(c, dict)}
        _seg_by_idx_e = {s.index: s for s in segments}
        _fin2orig_e = {v: k for k, v in (_bidx or {}).items()}
        _blk_e = []
        _preds_e: list = []                 # candidate predecessor indexes seen so far, in order
        for seg in segments:
            _sel_e = sel_by_idx.get(seg.index)
            _rej_e = bool(_sel_e is not None
                          and "verifier_failed" in (getattr(_sel_e, "flag_reasons", None) or [])
                          and not getattr(_sel_e, "image_path", ""))
            if seg.index in _breakout_clip or not _rej_e:
                # SUPERSET of the real gate's predecessor rule (which additionally requires the
                # beat to have produced clips): more candidates here → fewer early blocks → safe
                if not _rej_e and seg.index not in _breakout_clip:
                    _preds_e.append(seg.index)
                continue
            _any_ok = False
            for _p_e in reversed(_preds_e):
                try:
                    _ok_e, _ev_e = _hold_scene_compat(
                        _seg_by_idx_e.get(_p_e), _seg_by_idx_e.get(seg.index),
                        sel_by_idx.get(_p_e), sel_by_idx.get(seg.index),
                        single_scene=_sing_e, global_era=_era_e,
                        overlap_min=_ovl_e, char2actor=_c2a_e)
                except Exception:           # noqa: BLE001 — evaluation failure ≠ proof of doom
                    _ok_e = True
                if _ok_e:
                    _any_ok = True
                    break
            if not _any_ok:
                _blk_e.append({"seg_index": _fin2orig_e.get(seg.index, seg.index),
                               "final_index": seg.index,
                               "reason": ("no clean predecessor" if not _preds_e else
                                          "not same scene (every candidate predecessor fails)"),
                               "early_gate": True})
        if _blk_e:
            _mode_e = _os_eg.environ.get("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE",
                                         "block").strip().lower()
            _msg_e = (f"{len(_blk_e)} verifier-rejected beat(s) have NO possible same-scene hold "
                      f"(early gate, before assembly) — scene(s) "
                      f"{[b['seg_index'] for b in _blk_e[:8]]}.")
            if _mode_e == "warn":
                _review_draft.append(_msg_e)
                log(f"build: ⚠ EARLY RELEASE-BLOCK (mode=warn, REVIEW BUILD) — {_msg_e} "
                    f"Continuing; the authoritative gate reports after assembly.")
            else:
                try:
                    import json as _json_e
                    (proj.output_dir / "rejected_footage_audit.json").write_text(_json_e.dumps(
                        {"editorial_holds": [], "unresolved_release_block": _blk_e,
                         "early_gate": True}, indent=1), encoding="utf-8")
                except Exception:
                    pass
                log(f"build: ⛔ RELEASE-BLOCKED EARLY — {_msg_e} Skipping the doomed "
                    f"assembly (~20 min) — heal/rediscovery needed.")
                from .verify import NonRetryableBuildError
                raise NonRetryableBuildError(
                    f"rejected-footage gate: {len(_blk_e)} beat(s) unresolved (no valid editorial "
                    f"hold or contextual fallback) — rediscovery needed for scene(s) "
                    f"{[b['seg_index'] for b in _blk_e[:8]]}. This is a CONTENT failure: "
                    f"re-running the same render will not fix it.",
                    kind="rejected_footage")

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
            _bcap = _breakout_cap_by_final.get(int(seg.index), {})
            _bowner = int(_bcap.get("seg_index", seg.index))
            _lineage_register(_bclip, {
                "kind": "breakout", "original_beat": _bowner, "owner_beat": _bowner,
                "via": "breakout", "validated": True,
                "root_binding": (f"breakout:{_bcap.get('source_id', '')}:"
                                 f"{_bcap.get('source_t', '')}:{_bcap.get('line', '')}"),
            })
            # register the breakout's look in the air-guard so the SAME shot can't re-air
            # as ordinary footage a few beats later (breakouts bypass the windows walk)
            _aired_hashes.append(_frame_hash(str(_bclip), 1.0))
            clips_for_scene = (_split_clip_sequential(_bclip, _lens, proj.clips_dir, seg.index)
                               if k > 1 else [_bclip]) or [_bclip]
            for _bp in clips_for_scene:
                if _lineage_key(_bp) != _lineage_key(_bclip):
                    _lineage_derive(_bp, _bclip, via="breakout")
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
        # A CLIP KEYFRAME IS A THUMBNAIL, NOT A FRAME. index.py writes every keyframe at
        # `scale=512:-2` for the embedding model, and the still passes hand that same 512x288 file
        # to Ken-Burns, which upscales it 3.75x to 1080p and then zooms a further 1.08-1.10x on top.
        # Measured against a full-res re-extract of the identical instant, the shipped version loses
        # ~8.7x power at 0.30 Nyquist and ~29x at 0.42 — 18 of 21 aired stills on one render.
        # The exact owner source + indexed shot are still on disk, so the repair is one deterministic
        # midpoint extraction per still: no re-download or API spend.  A low-resolution/unprobeable
        # owner now blocks publication; silently retaining a 512px thumbnail would only disguise the
        # defect inside a 1080p container.
        # A declared image is a semantic replacement for the moving selection.  Missing/empty or
        # unverifiable image bytes must therefore BLOCK; falling through would air the very moving
        # clip this fallback replaced (often a verifier rejection).
        if _img:
            try:
                _declared_img = _require_declared_image_path(sel)
                _img_rescue = _preflight_image_rescues.get(id(sel))
                if _img_rescue is None:
                    # Defensive only: a post-preflight image mutation/addition must still take the
                    # strict path, never fall through to moving footage or an unverified thumbnail.
                    _img_rescue = _rescue_still_fullres(
                        proj, sel, str(_declared_img), log, seg=seg, eng=eng)
                _img = str(_img_rescue["path"])
                _img_root = _verified_image_lineage_root(
                    proj, sel, _img_rescue, seg.index, seg=seg)
                _img_audit_row = {
                    **_img_root, "final_scene": int(seg.index),
                    "declared_path": str(_declared_img.resolve()),
                    "aired_image_path": _lineage_key(_img),
                }
                _image_lineage_entries.append(_img_audit_row)
                _persist_image_lineage_audit(
                    proj, _image_lineage_entries, _image_lineage_failures)
            except Exception as _img_exc:                 # noqa: BLE001 — persist then fail closed
                _image_lineage_failures.append({
                    "final_scene": int(seg.index),
                    "original_beat": int(getattr(sel, "segment_index", seg.index)),
                    "declared_path": str(_img or ""),
                    "expected_source_id": str((getattr(sel, "image_meta", {}) or {}).get(
                        "src", "") or ""),
                    "expected_shot_index": (getattr(sel, "image_meta", {}) or {}).get("shot"),
                    "reason": str(_img_exc),
                })
                _persist_image_lineage_audit(
                    proj, _image_lineage_entries, _image_lineage_failures)
                raise
            _lineage_register(_img, _img_root)
            clips_for_scene = []
            for m in range(k):
                per_beat = max(cfg.min_clip_sec, _lens[m]) + 0.5
                _kc = proj.clips_dir / f"beat_{seg.index:03d}_{m}_img.mp4"
                _z = 1.08 + 0.02 * (m % 2)            # alternate push-in so multi-beat stills vary
                got = _image_kenburns_clip(_img, _kc, per_beat, zoom_to=_z)
                if got:
                    _lineage_derive(got, _img, via="verified_image")
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
            from .verify import NonRetryableBuildError
            raise NonRetryableBuildError(
                f"scene-lineage gate: verified image derivative failed for beat "
                f"{getattr(sel, 'segment_index', seg.index)}; refusing to fall through to an "
                f"unchecked moving-video or placeholder root", kind="scene_lineage")

        # VERIFIED-SELECTION LOCK.  From this point ordinary footage is derived ONLY from the
        # already-cut ``seg_NNN.mp4`` that belongs to this selection.  The legacy alternate-window
        # sorter / playhead walk remains below solely so old audit tests can describe the defect;
        # this fail-closed path always continues before it and has no opt-out switch.
        if sel is None:
            from .verify import NonRetryableBuildError
            raise NonRetryableBuildError(
                f"scene-lineage gate: final scene {seg.index} has no ClipSelection; refusing an "
                f"unowned placeholder/neighbor frame", kind="scene_lineage")
        # A blank clip_path is MISSING, not the current directory. `Path("")` is `Path(".")`, which
        # exists and stats non-empty, so this guard used to wave it through and the render died 30
        # lines later on "could not make a complete owned derivative" — a message about the fitter,
        # for a selection that simply had no file. Job f840b0cb49 lost a build to it on beat 57.
        _declared_clip = str(getattr(sel, "clip_path", "") or "").strip()
        _selected_clip = Path(_declared_clip) if _declared_clip else None
        if _selected_clip is None or not _selected_clip.is_file() \
                or _selected_clip.stat().st_size <= 0:
            # …and "re-cut the selection before build" had nobody to do it. verify's late
            # replacement and the recovery pass both run AFTER the cut stage, so a selection they
            # install owns a source, a shot and a window but never gets a file. Materialise it the
            # one legitimate way: the cut stage's own cutter, on this selection's own declared
            # source and window. That is the identical call cut_all would have made — no alternate
            # window, no neighbour, no placeholder — so ownership and every downstream lineage
            # check are unchanged. Only a genuine failure to produce it is fatal.
            from .cut import cut_selection as _cut_one
            _recut = None
            try:
                _recut = _cut_one(proj, sel, cfg, resume=True)
            except Exception as _recut_exc:               # noqa: BLE001 — report, then fail closed
                log(f"build: beat {sel.segment_index} clip re-cut raised "
                    f"{type(_recut_exc).__name__}: {str(_recut_exc)[:120]}")
            if _recut is not None and Path(_recut).is_file() and Path(_recut).stat().st_size > 0:
                # setattr, not attribute assignment: the selection lock forbids naming clip_path
                # directly in this block so no future edit can quietly read one in.
                setattr(sel, "clip_path", str(_recut))
                _selected_clip = Path(_recut)
                log(f"build: beat {sel.segment_index} had no cut clip (installed after the cut "
                    f"stage) — re-cut from its own source window {sel.in_point:.2f}-"
                    f"{sel.out_point:.2f}")
            else:
                from .verify import NonRetryableBuildError
                raise NonRetryableBuildError(
                    f"scene-lineage gate: beat {sel.segment_index} has no cut clip and re-cutting "
                    f"its own source window failed"
                    + (f" (declared {_declared_clip})" if _declared_clip else " (none declared)")
                    + "; refusing an unowned placeholder/neighbour frame", kind="scene_lineage")
        _root = _selection_root(sel, seg.index)
        _lineage_register(_selected_clip, _root)
        _owned_lens = [max(cfg.min_clip_sec, float(L)) + 0.5 for L in _lens]
        _owned_full = proj.clips_dir / f"beat_{seg.index:03d}_owned.mp4"
        _owned_crop = (_watermark_crop_filter(wm_corners[sel.source_id])
                       if sel.source_id in wm_corners else "")
        # assemble() adds the final editorial camera drift.  A second pre-zoom here both crops the
        # selected picture twice and can make an otherwise correct derivative fail the independent
        # source-window lineage canary.  Keep this ownership derivative spatially neutral; the
        # watermark crop above remains allowed and the renderer supplies the visible motion once.
        _owned_zoom = 1.0
        from .cut import _CUT_CONTRACT as _CUTC
        _owned = _fit_verified_selection_clip(
            _selected_clip, _owned_full, sum(_owned_lens),
            crop_filter=_owned_crop, zoom_to=_owned_zoom,
            frame_exact=(getattr(sel, "cut_contract", "") == _CUTC))
        if _owned is None or not _lineage_derive(
                _owned, _selected_clip,
                selection_source_compare_filter=_owned_crop):
            from .verify import NonRetryableBuildError
            raise NonRetryableBuildError(
                f"scene-lineage gate: could not make a complete owned derivative for beat "
                f"{sel.segment_index}; no alternate/walk/placeholder fallback is permitted",
                kind="scene_lineage")
        if k > 1:
            clips_for_scene = _split_clip_sequential(
                _owned, _owned_lens, proj.clips_dir, seg.index, suffix="sel")
            if len(clips_for_scene) != k:
                from .verify import NonRetryableBuildError
                raise NonRetryableBuildError(
                    f"scene-lineage gate: owned derivative split failed for beat "
                    f"{sel.segment_index} ({len(clips_for_scene)}/{k}); refusing a repeated or "
                    f"foreign fill", kind="scene_lineage")
            for _part in clips_for_scene:
                if not _lineage_derive(_part, _owned):
                    from .verify import NonRetryableBuildError
                    raise NonRetryableBuildError(
                        f"scene-lineage gate: unregistered split derivative {_part.name}",
                        kind="scene_lineage")
        else:
            clips_for_scene = [Path(_owned)]
        total_clips += len(clips_for_scene)
        footage.append(FootageItem(index=seg.index, path=clips_for_scene[0], is_video=True))
        beat_clips[seg.index] = clips_for_scene
        for m, (_cp, _need) in enumerate(zip(clips_for_scene, _owned_lens)):
            _aired_windows.append({
                "beat": seg.index, "original_beat": int(sel.segment_index), "clip": m,
                "file": Path(_cp).name, "root_file": _selected_clip.name,
                "source_id": sel.source_id,
                "source_title": ((proj.source(sel.source_id).title or "")[:120]
                                 if proj.source(sel.source_id) else ""),
                "in": round(float(sel.in_point), 3), "need": round(float(_need), 3),
                "via": "selection", "ok": True,
                "selection_binding": _root["selection_binding"],
            })
        gbeat += k
        continue

        windows_avail = list(getattr(sel, "beat_windows", []) or [])
        # snapshot the VERIFIED first choice: windows_avail is consumed as windows air, so by the
        # time provenance is recorded its [0] is no longer the beat's top-ranked window
        _w0_ref = list(windows_avail[0]) if windows_avail else None
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
                    if _win_unreadable(w[0], float(w[1]), float(w[2]) if len(w) > 2 else None):
                        continue                       # near-black/unreadable NEVER airs (Gap 3) — no
                        # 'least-dark last resort'; a dark-only beat falls to the still/freeze fallback
                        # below (image_fallback still or a freeze of the previous clean clip), never dark
                    _wsrc = proj.source(w[0])
                    _h = _frame_hash(_wsrc.local_path, float(w[1]) + 1.2) if (
                        _wsrc and _wsrc.local_path) else None
                    if _looks_recent(_h):
                        continue
                    _fresh.append((w, _h))
                if _fresh:
                    # prefer a look that has NOT already aired _LOOK_CAP times across the whole video;
                    # only fall back to an over-cap look if every fresh window is over-cap (never drops
                    # the last option → coverage preserved). Within a tier, prefer not-semantically-
                    # near-the-last-few-clips (_near_recent, previously dead code), then least-recent.
                    _under = [wh for wh in _fresh if _look_aired_count(wh[1]) < _LOOK_CAP]
                    _pool = _under if _under else _fresh
                    _pool.sort(key=lambda wh: (
                        _look_aired_count(wh[1]),
                        1 if _near_recent(_win_embed(wh[0][0], float(wh[0][1]))) else 0,
                        used_at.get(_wkey(wh[0][0], wh[0][1]), -1)))
                    _fresh = _pool
                    # AIRED-WINDOW LEGIBILITY (pass 1). Ask, BEFORE cutting, whether the window a
                    # beat is about to use is legible — and if it is not, take the next window from
                    # this beat's OWN relevance-ranked list. Measured on a 272-beat render: 28
                    # first-choice windows probe dark, and 27 of them have a legible alternate
                    # already in their list (depth 1 for 20 of them). The other 244 beats issue one
                    # probe and change nothing.
                    #
                    # Why it matters more than it sounds: today that darkness is discovered AFTER
                    # the cut, when the only remedy left is a freeze — and that render froze 31
                    # clips, 30 of them onto a donor from a DIFFERENT SCENE. The frames the audit
                    # called "an unreadable blue burst" and "a murky interior with a blob" were not
                    # dark cuts at all; they were those donors.
                    #
                    # Pass 2 is the old loop verbatim, so the option set is a STRICT SUPERSET of
                    # today's: nothing leaves any pool, and a beat whose every window is dark keeps
                    # exactly the window it has now.
                    _legib_on = os.environ.get(
                        "VIDLORE_CLIPSTUDIO_WINDOW_LEGIBILITY", "1").strip() \
                        not in ("0", "false", "no")
                    # ENTITY MUST SURVIVE A SUBSTITUTION. Only beat_windows[0] is the window the
                    # verifier confirmed; the rest are match-ranked alternates. A beat gets pushed
                    # off [0] more often than it looks — 31% of windows appear in more than one
                    # beat's list and a window airs only ONCE, so on a measured render 22 of 181
                    # beats (12%) were forced onto an alternate before the look-variety sort even
                    # ran. Usually that is harmless: 14 of those 22 substitutes still came from a
                    # source whose title names the beat's required entity (often the same scene from
                    # a different upload, which is exactly what the pool is for).
                    #
                    # The damage is the other kind: a beat requiring "Qyburn" landing on a Gregor
                    # Clegane character study, or one requiring "The Mountain" on the Sept
                    # explosion. Measured, that is 3 of 181 beats — narrow, and worth catching,
                    # because the caption names the person the picture does not show.
                    #
                    # Keyed on the source TITLE, not on Face-ID: the identities of these very
                    # windows are empty ([], [''], ['','']), so an identity test would fail open on
                    # exactly the beats that need it. Pass 2 ignores this, so nothing is lost.
                    _req_ent = (getattr(seg, "required_entity", "") or "").strip()
                    _ent_toks = {t for t in re.findall(r"[a-z']+", _req_ent.lower())
                                 if len(t) > 2 and t not in ("the", "of", "and")}

                    def _title_toks(_sid):
                        _s0 = proj.source(_sid)
                        return {t for t in re.findall(
                            r"[a-z']+", ((_s0.title if _s0 else "") or "").lower()) if len(t) > 2}
                    # only defend an entity the VERIFIED window actually carried
                    _w0_holds = bool(_ent_toks) and bool(
                        _ent_toks & _title_toks(windows_avail[0][0])) if windows_avail else False
                    for _lpass in (1, 2):
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
                            if (_lpass == 1 and _legib_on and _wsp
                                    and _source_window_too_dark(_wsp, float(_wh[0][1]), per_beat)):
                                continue
                            if (_lpass == 1 and _w0_holds
                                    and _wh[0][0] != windows_avail[0][0]
                                    and not (_ent_toks & _title_toks(_wh[0][0]))):
                                continue           # substitute drops the beat's named subject
                            chosen_w, _ch_hash = _wh
                            break
                        if chosen_w is not None:
                            if _lpass == 2 and _legib_on:
                                log(f"build: beat {seg.index} clip {m} — every candidate window "
                                    f"probes dark; keeping the ranked first choice (the "
                                    f"black-repair sweep remains the backstop)")
                            break
                        if not _legib_on:
                            break                # pass 2 would be identical — do not repeat it
                    if chosen_w is not None:
                        windows_avail.remove(chosen_w)
                # else: every window aired/looks recent → fall through to the shot-aware walk
                # below, which finds the next DIFFERENT-looking shot instead of a replay
            _wqc_moment = None       # the ORIGINALLY SELECTED range this beat must keep airing
            _aired_via = "walk"
            if chosen_w is not None:
                _aired_via = ("window[0]" if (_w0_ref
                              and chosen_w[0] == _w0_ref[0]
                              and abs(float(chosen_w[1]) - float(_w0_ref[1])) < 0.01)
                              else "window[alt]")
                sid, in_p = chosen_w[0], float(chosen_w[1])
                src = proj.source(sid)
                # lead-in must NOT cross the shot boundary backwards: 0.2s of the PREVIOUS
                # (unrelated) shot flashing at the beat opening is how a cartoon/tweet-card
                # frame aired at a scene start — snap to the containing shot's own start
                _csh = next((s9 for s9 in _shots_b(sid)
                             if s9.start <= in_p < s9.end), None)
                start = max(0.0, in_p - 0.2, (float(_csh.start) if _csh else 0.0))
                # window-QC moment = the chosen window's OWN [in, out] — the padding that
                # stretches it to src_need is NOT part of the selected moment, so a shift
                # only has to keep [in, out] on screen (observed: a 1.4s chosen moment
                # padded to 4.4s refused a clean shift because the PADDING was treated as
                # the moment)
                _wqc_moment = (in_p, float(chosen_w[2]))
                used_at[_wkey(sid, in_p)] = gbeat
                _air_ct[_wkey(sid, in_p)] = _air_ct.get(_wkey(sid, in_p), 0) + 1
                _aired_hashes.append(_ch_hash)
                _e = _win_embed(sid, in_p)
                if _e is not None:
                    recent_emb.append(_e)
            elif sel and sel.source_id:
                # fill: SHOT-AWARE walk — continue the chosen source from its playhead, snapped
                # to the next shot whose look is not a near-term repeat. The walk start is
                # MECHANICAL (not a selected moment), so no _wqc_moment: any clean span
                # overlapping the window is as relevant as the window itself.
                src = proj.source(sel.source_id)
                sid = sel.source_id
                start = _next_distinct_shot(sid, max(_playhead.get(sid, 0.0),
                                                     sel.in_point - 0.2), need=per_beat)
                if src:
                    used_at[_wkey(sid, start)] = gbeat
                    # count this airing too — the walk used to write only `used_at`, so a
                    # walk-aired window stayed invisible to the "a window airs ONCE, ever" gate
                    # and a later beat could legally re-air the very same window.
                    _air_ct[_wkey(sid, start)] = _air_ct.get(_wkey(sid, start), 0) + 1
                    _aired_hashes.append(_frame_hash(src.local_path, start + 1.2)
                                         if src.local_path else None)
            gbeat += 1
            if not (src and src.local_path and Path(src.local_path).exists()):
                continue
            factor = 1.30 if (is_peak and m == k - 1 and k >= 2) else 1.0   # held "land" = slow-mo
            src_need = per_beat / factor
            if src.duration and start + src_need > src.duration:
                start = max(0.0, src.duration - src_need)
            start = _wqc_render_start(sid, start, src_need, seg, _wqc_moment)
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
            # WHAT ACTUALLY AIRED. `ledger.jsonl` records the MATCH stage's pick; build then
            # re-selects (a window airs once, look-variety reorders, probes skip candidates, the
            # shot-walk fills) and nothing recorded the outcome. An audit of the delivered video
            # therefore read the ledger, named a source, and described a scene that is not on
            # screen at that timecode — twice, on different beats. This is the record that makes
            # the next audit trustworthy: source, in-point, length, and where it came from.
            _aired_windows.append({
                "beat": seg.index, "clip": m, "file": Path(dest).name,
                "source_id": src.id, "source_title": (src.title or "")[:120],
                "in": round(float(start), 3), "need": round(float(src_need), 3),
                "via": _aired_via, "ok": bool(rc),
            })
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
    # STAGED-REPLAY PARALLEL QA SWEEPS. The branding and dark sweeps below each pay one
    # expensive per-clip probe (_clip_branding_text: up to ~10 frame decodes + OCR;
    # _clip_too_dark: a full-clip decode — measured 418s serial over 231 clips). The tested
    # set is FROZEN before each loop runs (each loop iterates a snapshot; its replacements
    # are never re-tested), so the verdicts are precomputed in a bounded thread pool and the
    # UNCHANGED serial loops replay them — _last_clean ordering, freeze-replace decisions and
    # log lines are byte-identical to the serial pass. Verdicts are pure functions of the
    # clip file (+ fixed floor), so precomputation cannot change any outcome. A missing
    # verdict falls back to the direct call. VIDLORE_CLIPSTUDIO_QA_SWEEP_WORKERS=1 restores
    # the fully-serial path.
    def _precompute_clip_verdicts(probe, paths, label):
        try:
            _w = int(_os.environ.get("VIDLORE_CLIPSTUDIO_QA_SWEEP_WORKERS",
                                     str(max(2, ((_os.cpu_count() or 8) // 2)))) or 1)
        except (TypeError, ValueError):
            _w = 1
        out: dict = {}
        uniq = []
        seen = set()
        for p in paths:
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                uniq.append(sp)
        if _w <= 1 or len(uniq) <= 1:
            return out                                   # serial loops probe directly
        import concurrent.futures as _cfq
        from . import perf_metrics as _pmq
        with _pmq.timed(f"build.qa_sweep.{label}"):
            with _cfq.ThreadPoolExecutor(max_workers=_w) as _exq:
                futs = {_exq.submit(probe, sp): sp for sp in uniq}
                for fu in _cfq.as_completed(futs):
                    try:
                        out[futs[fu]] = bool(fu.result())
                    except Exception:                    # noqa: BLE001 — replay probes directly
                        pass
        _pmq.incr(f"build.qa_sweep.{label}.precomputed", len(out))
        return out

    def _sweep_paths():
        _ps = []
        for _sg in segments:
            if _sg.index in _breakout_clip:
                continue
            _ps.extend(beat_clips.get(_sg.index) or [])
        return _ps

    # (the BRANDING sweep's probes now precompute in the same bounded pool as the dark sweep:
    # the per-thread-engine identity proof exists — _ocr_engine_tl builds instances with the
    # identical default config, canary-proven bit-identical output — and the serial walk below
    # replays verdicts in its original order, so _last_clean/donor decisions are unchanged.)
    if _ocr_eng is not None and _os.environ.get("VIDLORE_CLIPSTUDIO_BRANDING_GATE", "1").strip() \
            not in ("0", "false", "no", ""):
        _brand_verdicts = _precompute_clip_verdicts(
            lambda sp: _clip_branding_text(Path(sp), _ocr_engine_tl(_ocr_eng)),
            _sweep_paths(), "branding")
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
            # SAME-SCENE donor first (mirrors the near-black pass): freezing from _last_clean
            # propagates the PREVIOUS scene's content across the cut. Probe the scene's own
            # clips for a clean donor before reaching backwards.
            _brand_flags = [(_brand_verdicts[str(cp)] if str(cp) in _brand_verdicts
                             else _clip_branding_text(Path(cp), _ocr_eng)) for cp in clips]
            _own_ok = [cp for cp, _bf in zip(clips, _brand_flags) if not _bf]
            for m, cp in enumerate(list(clips)):
                if _brand_flags[m]:
                    _d = (_ls[m] if m < len(_ls) and _ls[m] > 0 else 3.0) + 0.5
                    _donor = _own_ok[0] if _own_ok else _last_clean
                    if _donor is not None:
                        _fr = proj.clips_dir / f"beat_{seg.index:03d}_{m}_nobrand.mp4"
                        _got = _freeze_replace(Path(_donor), _fr, _d)
                        if _got:
                            _lineage_derive(_got, _donor)
                            clips[m] = Path(_got)
                            _replaced += 1
                            continue
                    # no clean donor → drop to a black placeholder rather than air the card
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

    # 3a-2) UNREADABLE-CLIP REMOVAL (Gap 3) — a cut clip that is near-black/illegible THROUGHOUT must
    #       never air (distinct from a legitimate low-RESOLUTION but legible clip). Measured on the
    #       actual rendered clip's picture area; replaced with a freeze of the previous CLEAN clip
    #       (timing preserved). No 'least-dark last resort' airs. env FINAL_BLACK_GATE also gates this.
    if _os.environ.get("VIDLORE_CLIPSTUDIO_DARK_CLIP_GATE", "1").strip() not in ("0", "false", "no"):
        _dfloor = _cfg_f("VIDLORE_CLIPSTUDIO_FINAL_BLACK_FLOOR", 50.0)
        _dlens = {segments[_p].index: list(_ls) for _p, _ls in _lens_by_pos.items()}
        _dark_verdicts = _precompute_clip_verdicts(
            lambda sp: _clip_too_dark(Path(sp), floor=_dfloor), _sweep_paths(), "dark")
        _last_clean_d = None
        _drep = 0
        for seg in segments:
            if seg.index in _breakout_clip:
                clips0 = beat_clips.get(seg.index) or []
                if clips0:
                    _last_clean_d = clips0[-1]
                continue
            clips = beat_clips.get(seg.index) or []
            _ls = _dlens.get(seg.index) or []
            # SAME-SCENE donor first: freezing from the PREVIOUS scene's clip propagates that
            # scene's content across the cut (observed: a news-CGI frame from the neighbouring
            # beat froze into a legit S05E08 beat, airing broadcast graphics over the wrong
            # narration). A cross-scene donor stays the last resort and is logged as such.
            # SYNTHESIZED clips (branding-pass _nobrand freezes, earlier _nodark freezes, black
            # placeholders) are never donors AND never become _last_clean_d: a _nobrand freeze IS
            # the previous scene's frame, so donating from it would silently re-create the exact
            # cross-scene propagation this pass exists to stop — while logging "same-scene".
            def _is_synth(_p):
                _s = Path(_p).stem
                # `_img` is a Ken-Burns STILL, not footage. The comment on the donor rule says the
                # donor "must be REAL footage" but nothing checked it, so a beat could freeze onto
                # an image that is itself already a held frame — on one render two adjacent beats
                # then spent 10.7 consecutive seconds on a single web JPEG.
                return ("_nobrand" in _s or "_nodark" in _s or "placeholder" in _s
                        or _s.endswith("_img"))
            # (probe served from the staged-replay parallel precompute when available — the
            # verdict is a pure function of the clip file + fixed floor, so this changes
            # nothing about the donor selection below)
            _dark_flags = [(_dark_verdicts[str(cp)] if str(cp) in _dark_verdicts
                            else _clip_too_dark(Path(cp), floor=_dfloor)) for cp in clips]
            _own_clean = [cp for cp, _dk in zip(clips, _dark_flags)
                          if not _dk and not _is_synth(cp)]
            for m, cp in enumerate(list(clips)):
                if not _dark_flags[m]:
                    if not _is_synth(cp):
                        _last_clean_d = cp
                    continue
                _d = (_ls[m] if m < len(_ls) and _ls[m] > 0 else 3.0) + 0.5
                _donor = _own_clean[0] if _own_clean else _last_clean_d
                if _donor is not None:
                    _fr = proj.clips_dir / f"beat_{seg.index:03d}_{m}_nodark.mp4"
                    _got = _freeze_replace(Path(_donor), _fr, _d)
                    if _got:
                        _lineage_derive(_got, _donor)
                        clips[m] = Path(_got)
                        _drep += 1
                        log(f"build: unreadable-clip removal — scene {seg.index} clip {m} "
                            f"near-black, freeze-replaced with a "
                            f"{'same-scene' if _own_clean else 'PREVIOUS-scene'} clean frame"
                            + ("" if _own_clean else
                               " (cross-scene donor — content is the neighbour beat's)"))
                        continue
                # no clean donor to freeze — leave it for the final black gate to BLOCK
                # (never silently air dark; never substitute a black placeholder either)
                log(f"build: ⚠ scene {seg.index} clip {m} near-black with no clean predecessor "
                    f"— final black gate will block this render (footage gap needs rediscovery)")
            beat_clips[seg.index] = clips
            if clips:
                for _fi in footage:
                    if _fi.index == seg.index:
                        _fi.path = clips[0]
                        break
        if _drep:
            log(f"build: unreadable-clip removal — {_drep} near-black clip(s) freeze-replaced")

    # 3a-3) REJECTED-FOOTAGE HANDLING (R3-3) — a beat whose moving clip the verifier REJECTED
    #       (verifier_failed) with no validated still must NEVER air the rejected footage. It becomes
    #       a validated EDITORIAL HOLD (a short freeze of the previous clean clip) ONLY when that clip
    #       is same-canonical-scene, clean, and the hold is capped to ONE consecutive beat; otherwise
    #       the beat is UNRESOLVED and the render is RELEASE-BLOCKED (never a repeated freeze, never a
    #       black placeholder, never the rejected clip). The aired replacement is recorded honestly.
    if _os.environ.get("VIDLORE_CLIPSTUDIO_REJECTED_FOOTAGE_GATE", "1").strip() not in ("0", "false", "no"):
        _analysis_rf = (proj.meta.get("analysis", {}) or {})
        _single_scene_rf = _analysis_rf.get("video_type", "") == "single_scene"
        _global_era_rf = str(_analysis_rf.get("episode_hint", "") or "")
        _hold_cap = int(_cfg_i("VIDLORE_CLIPSTUDIO_MAX_CONSEC_HOLD", 1))
        # R4-4 duration bounds: a single editorial hold may not freeze a frame longer than
        # _hold_single_cap, and the CUMULATIVE hold time across the whole video may not exceed
        # _hold_total_cap. Holds are a last resort — a video that needs seconds of frozen frames
        # has a discovery gap and must recover or release-block, not paper it over with freezes.
        _hold_single_cap = _cfg_f("VIDLORE_CLIPSTUDIO_MAX_HOLD_SINGLE_SEC", 2.5)
        _hold_total_cap = _cfg_f("VIDLORE_CLIPSTUDIO_MAX_HOLD_TOTAL_SEC", 3.0)
        _seg_by_idx_rf = {s.index: s for s in segments}
        _hold_overlap_min = _cfg_f("VIDLORE_CLIPSTUDIO_HOLD_SCENE_OVERLAP", 0.4)
        # roster mapping so the Face-ID identity check recognises the SAME person across
        # actor/character naming (Face-ID says 'sophie turner', the beat says 'sansa stark')
        _c2a_rf = {str(c.get("name", "")): str(c.get("actor", ""))
                   for c in (_analysis_rf.get("characters") or []) if isinstance(c, dict)}

        def _scene_compat(a_idx, b_idx):
            """REAL same-scene compatibility (R4-3) via the module-level pure decision — a freeze of
            beat a's clean frame may cover rejected beat b ONLY when they are genuinely the same
            moment. Never auto-true for a single-scene video. Returns (ok, evidence)."""
            return _hold_scene_compat(
                _seg_by_idx_rf.get(a_idx), _seg_by_idx_rf.get(b_idx),
                sel_by_idx.get(a_idx), sel_by_idx.get(b_idx),
                single_scene=_single_scene_rf, global_era=_global_era_rf,
                overlap_min=_hold_overlap_min, char2actor=_c2a_rf)

        _rlens = {segments[_p].index: list(_ls) for _p, _ls in _lens_by_pos.items()}
        # The block loop runs in FINAL (post-breakout-reindex) index space, which is CORRECT for the
        # gate. But every other surface — project.json, match/verify logs, orchestrate recovery —
        # uses ORIGINAL beat indices. Relabel the AUDIT/report to original space so "scene N" names
        # the same beat the rest of the pipeline does (identity map when there are no breakouts).
        _final_to_orig = {v: k for k, v in (_bidx or {}).items()}
        _orig = lambda i: _final_to_orig.get(i, i)      # noqa: E731
        _last_clean_r, _last_clean_idx = None, None
        _consec_holds, _rrep, _hold_total = 0, 0, 0.0
        _rf_audit, _rf_block = [], []
        for seg in segments:
            _sel_r = sel_by_idx.get(seg.index)
            _rejected = bool(_sel_r is not None
                             and "verifier_failed" in (getattr(_sel_r, "flag_reasons", None) or [])
                             and not getattr(_sel_r, "image_path", ""))
            if seg.index in _breakout_clip or not _rejected:
                clips0 = beat_clips.get(seg.index) or []
                # A clean predecessor for holds must be REAL footage that itself verified — never a
                # breakout, a rejected clip, or (implicitly) a prior hold (holds don't update this).
                if clips0 and not _rejected and seg.index not in _breakout_clip:
                    _last_clean_r, _last_clean_idx, _consec_holds = clips0[-1], seg.index, 0
                continue
            clips = beat_clips.get(seg.index) or []
            _ls = _rlens.get(seg.index) or []
            # this beat's TOTAL frozen duration. The loop below freezes EVERY sub-clip window of the
            # beat to the SAME predecessor still, and the sub-clips play SEQUENTIALLY, so the viewer
            # sees a frozen frame for the SUM of the windows (each + its 0.5s tail pad), not the max.
            # Using max would under-count a multi-sub-clip hold and let it slip past the R4-4 caps
            # (e.g. 3×2.3s = 6.9s of frozen frame recorded as 2.3s).
            _beat_hold_dur = (sum((_ls[m] if m < len(_ls) and _ls[m] > 0 else 3.0) + 0.5
                                  for m in range(len(clips)))) if clips else 0.0
            _compat_ok, _compat_ev = (
                _scene_compat(_last_clean_idx, seg.index) if _last_clean_idx is not None
                else (False, {"reason": "no clean predecessor"}))
            # Enumerate EVERY failure mode explicitly (module-level pure decision) — none may
            # silently pass to a black/rejected frame.
            _reason = _hold_block_reason(
                clips_present=bool(clips), has_predecessor=(_last_clean_r is not None),
                compat_ok=_compat_ok, compat_reason=_compat_ev.get("reason", "incompatible"),
                consec_holds=_consec_holds, hold_cap=_hold_cap,
                beat_hold_dur=_beat_hold_dur, hold_total=_hold_total,
                single_cap=_hold_single_cap, total_cap=_hold_total_cap)
            _motion_hold = False
            if _reason is not None:
                # Is DURATION the ONLY blocker? (re-evaluate the same pure decision with zeroed
                # durations — compat/predecessor/consec must all still pass.) The freeze caps
                # exist against long FROZEN frames looking broken; a validated same-scene frame
                # aired as a Ken-Burns MOTION still is the tool's own sanctioned still treatment
                # (observed: a 4.6s beat with a PASSING same-scene hold release-blocked purely
                # on the 2.5s freeze cap).
                if _hold_block_reason(
                        clips_present=bool(clips), has_predecessor=(_last_clean_r is not None),
                        compat_ok=_compat_ok, compat_reason=_compat_ev.get("reason", "incompatible"),
                        consec_holds=_consec_holds, hold_cap=_hold_cap,
                        beat_hold_dur=0.0, hold_total=0.0,
                        single_cap=_hold_single_cap, total_cap=_hold_total_cap) is None:
                    _motion_hold = True
                    log(f"build: rejected-footage — beat {_orig(seg.index)} hold exceeds the freeze "
                        f"caps ({_beat_hold_dur:.1f}s) → same-scene Ken-Burns MOTION hold instead")
                else:
                    # UNRESOLVED — release-block (recovery is attempted upstream in orchestrate; by
                    # here a still-rejected beat with no valid bounded hold must never air
                    # rejected/black frames).
                    _rf_block.append({"seg_index": _orig(seg.index), "final_index": seg.index,
                                      "reason": _reason, "hold_dur_s": round(_beat_hold_dur, 2),
                                      "evidence": _compat_ev})
                    continue
            _held_ok = True
            for m, cp in enumerate(list(clips)):
                _d = (_ls[m] if m < len(_ls) and _ls[m] > 0 else 3.0) + 0.5
                _fr = proj.clips_dir / f"beat_{seg.index:03d}_{m}_hold.mp4"
                _got = (_kenburns_hold(Path(_last_clean_r), _fr, _d) if _motion_hold
                        else _freeze_replace(Path(_last_clean_r), _fr, _d))
                if _got:
                    # An editorial hold AIRS the donor beat's verified frame on this beat — that
                    # owner change is the whole point of the feature, and it is separately proven
                    # (same-scene compat + the freeze caps above) and audited in
                    # rejected_footage_audit.json. Declare it as its own lineage kind so the
                    # provenance contract can hold it to the hold rules instead of reading a
                    # sanctioned donation as a silent owner swap and killing the finished render.
                    _lineage_derive(
                        _got, _last_clean_r, via="editorial_hold",
                        extra={
                            "kind": "scene_hold",
                            "hold_of_beat": int(_last_clean_idx),
                            "hold_kind": ("editorial_hold_kenburns" if _motion_hold
                                          else "editorial_hold"),
                            "hold_duration_s": round(float(_d), 3),
                            "hold_compat_evidence": dict(_compat_ev or {}),
                        })
                    clips[m] = Path(_got)
                    _rrep += 1
                    _rf_audit.append({"seg_index": _orig(seg.index), "final_index": seg.index,
                                      "replacement": ("editorial_hold_kenburns" if _motion_hold
                                                      else "editorial_hold"),
                                      "held_from_beat": _orig(_last_clean_idx), "duration_s": round(_d, 2),
                                      "validation": ("same_scene_kenburns_hold" if _motion_hold
                                                     else "same_scene_clean_hold"), "clip": m,
                                      "compat_evidence": _compat_ev})
                else:
                    _held_ok = False                      # freeze GENERATION FAILURE → fail closed
                    _rf_block.append({"seg_index": _orig(seg.index), "final_index": seg.index,
                                      "reason": "editorial-hold freeze generation FAILED"})
                    break
            if _held_ok:
                _consec_holds += 1
                # EVERY held second counts. Exempting the Ken-Burns variant let 6.68s of hold ship
                # while the audit reported total_hold_seconds 2.38, and two adjacent beats spent
                # 10.7 CONSECUTIVE seconds on one web JPEG — 2.8x the median beat and 25% longer
                # than the longest real shot in the film. A gentle push-in does not make a held
                # frame stop being a held frame; it only makes the caps blind to it.
                _hold_total += _beat_hold_dur
                beat_clips[seg.index] = clips
                if clips:
                    for _fi in footage:
                        if _fi.index == seg.index:
                            _fi.path = clips[0]
                            break
        try:
            import json as _json_rf
            (proj.output_dir / "rejected_footage_audit.json").write_text(_json_rf.dumps(
                {"editorial_holds": _rf_audit, "unresolved_release_block": _rf_block,
                 "total_hold_seconds": round(_hold_total, 2),
                 "caps": {"consecutive": _hold_cap, "single_sec": _hold_single_cap,
                          "total_sec": _hold_total_cap}}, indent=1),
                encoding="utf-8")
        except Exception:
            pass
        if _rrep:
            log(f"build: rejected-footage — {_rrep} verifier-rejected clip(s) replaced with a validated "
                f"same-scene editorial HOLD ({_hold_total:.1f}s total, caps {_hold_single_cap:.1f}s/"
                f"{_hold_total_cap:.1f}s; never the rejected footage)")
        if _rf_block:
            # RELEASE-BLOCK MODE (default 'block' = fail-closed, the production behaviour). 'warn'
            # is an explicit REVIEW/acceptance opt-in: the gate still evaluates every beat and writes
            # the same honest rejected_footage_audit.json, but the render COMPLETES so the video can
            # actually be watched and audited, with the weak beats reported loudly instead of the
            # whole render being thrown away. It never makes a beat's footage any less validated —
            # it only chooses reporting over aborting.
            _blk_mode = _os.environ.get("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block").strip().lower()
            _msg = (f"{len(_rf_block)} verifier-rejected beat(s) have NO valid fallback (first scene "
                    f"{_rf_block[0]['seg_index']}: {_rf_block[0]['reason']}) — scene(s) "
                    f"{[b['seg_index'] for b in _rf_block[:8]]}. See rejected_footage_audit.json.")
            if _blk_mode == "warn":
                _review_draft.append(_msg)
                log(f"build: ⚠ RELEASE-BLOCK (mode=warn, REVIEW BUILD — not for publication) — {_msg}")
            else:
                _quar = out_path.with_name(out_path.stem + ".FAILED_REJECTED_FOOTAGE" + out_path.suffix)
                log(f"build: ⛔ RELEASE-BLOCKED — {_msg} Refusing to air rejected/repeated-freeze/"
                    f"black footage.")
                # NON-RETRYABLE: this is a judgment about the CONTENT, not a transient fault.
                # Raised as a bare RuntimeError it was indistinguishable from a network blip, so a
                # driver restarted the whole pipeline 8 times; the 8th "passed" only because the
                # vision API had died by then. Re-running an unchanged render cannot resolve a
                # rejected beat — only rediscovery/re-matching can.
                from .verify import NonRetryableBuildError
                raise NonRetryableBuildError(
                    f"rejected-footage gate: {len(_rf_block)} beat(s) unresolved (no valid editorial "
                    f"hold or contextual fallback) — rediscovery needed for scene(s) "
                    f"{[b['seg_index'] for b in _rf_block[:8]]}. This is a CONTENT failure: "
                    f"re-running the same render will not fix it.",
                    kind="rejected_footage")

    # ---- UNVERIFIED-EXACT GATE -------------------------------------------------------------
    # A separate gate from the rejected-footage one above, deliberately. That gate handles footage
    # PROVEN wrong (freeze-replace it with a clean neighbour). This one handles footage nobody could
    # check — which is not the same thing and must not be freeze-replaced, because we have no
    # evidence it is wrong. It only has to stop the render from claiming it was verified.
    #
    # Without this, a vision outage was INDISTINGUISHABLE from a clean pass: 229 errored beats
    # produced 0 rejections, the gate above found nothing to block, and the render shipped with 178
    # exact_scene beats whose relevance_class was literally 'unverified'.
    from . import policy as _policy_g
    _unver_block = []
    for seg in segments:
        _s = sel_by_idx.get(seg.index)
        if _s is None or not _policy_g.verify_strict(seg):
            continue
        if getattr(_s, "image_path", ""):
            continue                       # a validated still already covers this beat
        if str((getattr(_s, "verifier", None) or {}).get("status", "")) in ("error", "unavailable"):
            _unver_block.append(seg.index)
    if _unver_block:
        _um = (f"{len(_unver_block)} exact_scene beat(s) were never verified (vision backend "
               f"error/unavailable) — scene(s) {_unver_block[:8]}")
        if _os.environ.get("VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block").strip().lower() == "warn":
            log(f"build: ⚠ UNVERIFIED-EXACT (mode=warn, REVIEW BUILD — not for publication) — {_um}")
        else:
            log(f"build: ⛔ RELEASE-BLOCKED — {_um}. An unverifiable beat is unresolved, not "
                f"accepted; refusing to publish footage nobody could check.")
            from .verify import NonRetryableBuildError
            # kind, so the portal's auto-review can see what the branch above already decided: in
            # warn mode these beats ship flagged. Untyped, the driver read this as a crash and the
            # render ended with no file at all — the exact defect release_policy exists to end.
            raise NonRetryableBuildError(
                f"unverified-exact gate: {_um}. Restore the vision backend and re-verify — "
                f"re-running with a dead verifier will only make this pass silently.",
                kind="unverified_exact")

    suppress_wins = []
    if _cap_on and _suppress_on:
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
        # SPEED: precompute the FIRST-PASS burned-text probes over the exact clip set the walk
        # below visits (snapshot taken NOW — after the branding/dark/black sweeps mutated
        # beat_clips, so verdicts bind to the paths actually probed). The post-crop RECHECK
        # stays a direct fresh probe: _crop_clip_corner rewrites the clip file in place, so a
        # cached pre-crop verdict would wrongly re-flag a repaired clip.
        _dodge_paths = []
        for _sg_d in segments:
            _cl_d = beat_clips.get(_sg_d.index) or []
            if _cl_d and span_by_idx.get(_sg_d.index):
                _dodge_paths.extend(_cl_d)
        _dodge_verdicts = _precompute_clip_verdicts(
            lambda sp: _clip_has_burned_text(Path(sp), _ocr_engine_tl(_ocr_eng)),
            _dodge_paths, "dodge")
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
                if (_dodge_verdicts[str(cp)] if str(cp) in _dodge_verdicts
                        else _clip_has_burned_text(Path(cp), _ocr_eng)):
                    # REPAIR-FIRST: corner-localized text (channel bug / commenter-avatar badge)
                    # is CROPPED out of the clip — the viewer keeps both clean footage AND the
                    # caption. Suppression is the last resort for non-croppable text (dialogue
                    # subs / frame-wide cards) and is now logged per-beat: a silently vanishing
                    # caption reads as a render bug (observed: 3.3s caption dropout over an
                    # avatar badge the earlier gates missed).
                    _corner = _clip_text_corner(Path(cp), _ocr_eng)
                    if _corner and _crop_clip_corner(Path(cp), _corner, log=log) \
                            and not _clip_has_burned_text(Path(cp), _ocr_eng):
                        log(f"build: caption-dodge REPAIR — beat {seg.index} clip {m} "
                            f"corner-cropped ({_corner}); caption kept")
                    else:
                        suppress_wins.append((round(max(a, t - 0.3), 2),
                                              round(min(b, t2 + 0.3), 2)))
                        log(f"build: caption-dodge SUPPRESS — beat {seg.index} "
                            f"[{max(a, t - 0.3):.1f}-{min(b, t2 + 0.3):.1f}s] caption hidden "
                            f"over source text (not corner-croppable)")
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
    if _cap_on:
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

    # FINAL BUILD-SIDE LINEAGE MANIFEST.  Construct this only after every crop/freeze/QA repair has
    # mutated ``beat_clips``; the exact paths below are the inputs assemble will read.  The expected
    # owner comes from the scene's ClipSelection, while the actual owner comes from the immutable
    # derivative registry.  That asymmetry is intentional: a previous-scene donor cannot make
    # itself look valid merely by carrying its own internally consistent metadata.
    _scene_lineage: list[dict] = []
    _aired_by_key = {(int(r.get("beat", -1)), int(r.get("clip", -1))): dict(r)
                     for r in _aired_windows}
    _aired_final: list[dict] = []
    for _seg_l in segments:
        _sel_l = sel_by_idx.get(_seg_l.index)
        _expected_orig = (int(getattr(_sel_l, "segment_index")) if _sel_l is not None
                          else _final_to_orig_lineage.get(_seg_l.index, _seg_l.index))
        for _m_l, _cp_l in enumerate(beat_clips.get(_seg_l.index) or []):
            _root_l = _lineage_roots.get(_lineage_key(_cp_l))
            if _root_l is None:
                _root_l = {
                    "kind": "unknown", "owner_beat": None, "via": "untracked",
                    "validated": False, "root_binding": "",
                }
            _kind_l = str(_root_l.get("kind") or "unknown")
            if _kind_l == "selection_video":
                _kind_l = "selection_derivative"
            # A breakout is an explicit pseudo-scene anchored to the original beat it evidences;
            # it is not a substitute for an ordinary narration scene.  Its own stable owner is the
            # expected owner.  Ordinary/image scenes must agree with their ClipSelection owner.
            _row_orig = (int(_root_l.get("owner_beat"))
                         if _kind_l == "breakout" and _root_l.get("owner_beat") is not None
                         else int(_expected_orig))
            _row_l = {
                **_root_l, "kind": _kind_l, "final_scene": int(_seg_l.index),
                "original_beat": _row_orig, "clip": int(_m_l),
                "file": _lineage_key(_cp_l), "media_kind": "video",
            }
            _scene_lineage.append(_row_l)
            _aw_l = _aired_by_key.get((int(_seg_l.index), int(_m_l)), {})
            _aw_l.update({
                "beat": int(_seg_l.index), "original_beat": _row_orig, "clip": int(_m_l),
                "file": Path(_cp_l).name, "lineage_kind": _kind_l,
                "root_owner_beat": _root_l.get("owner_beat"),
                "via": str(_root_l.get("via") or "untracked"),
                "lineage_validated": bool(_root_l.get("validated")),
            })
            _aired_final.append(_aw_l)
    _aired_windows = _aired_final
    # Persistence is part of both invariants.  If either audit cannot be written, publication is
    # blocked rather than silently proceeding with unverifiable output.
    import json as _json_lineage
    (proj.output_dir / "aired_windows.json").write_text(
        _json_lineage.dumps({"schema": "aired_windows/2", "clips": _aired_windows}, indent=1),
        encoding="utf-8")
    _assert_scene_lineage(
        _scene_lineage, proj.output_dir / "scene_lineage_manifest.json")
    log(f"build: scene-lineage manifest PASS — {len(_scene_lineage)} aired input(s), "
        f"all owned by their own verified selection")

    # Breakout captions are validated BEFORE the expensive assembly.  The same ASS is reused by
    # the post-render burn, so the exact 100%-coverage / 0.5s-tail evidence is what actually airs.
    _breakout_ass_preflight = None
    _breakout_caption_burn_ok = True
    _breakout_caps_enabled = (_cap_on and os.environ.get(
        "VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1").strip() not in ("0", "false", "no"))
    _caps_pre = list(getattr(narration, "_breakout_caps", None) or [])
    if _breakout_caps_enabled and _caps_pre:
        _breakout_ass_preflight = _breakout_caption_ass(
            _caps_pre, work / "breakout_caps.ass", log, preset=_cap_preset)
        if _breakout_ass_preflight is None:
            # TWO CORRECT RULES, IN DIRECT CONTRADICTION — AND THE VIDEO PAID FOR IT.
            #
            # A breakout's caption may only show words the ASR is confident about: "a missing
            # caption line beats a wrong one", because a mis-transcribed line burned on screen
            # reads as somebody else's subtitle. And a breakout may only air if it is captioned
            # through its final spoken word. When the ASR garbles the tail, the first rule
            # guarantees the second one fails — so ANY breakout with a garbled tail killed the
            # whole render. Measured on job 229233891e scene 110: line one captioned perfectly
            # (align 1.00), line two dropped at ASR confidence 0.09 against the 0.45 floor
            # ('that men shall tremble. from both'), coverage 8/14 words = 57%, build dead.
            #
            # The contradiction is only irreconcilable at RENDER scope. At BREAKOUT scope it
            # resolves itself: a breakout that cannot be fully captioned does not air. Neither rule
            # is bent — no partial caption is ever burned, and no unverified line is shown — and
            # the other 145 beats keep their video.
            _dropped = len(_caps_pre)
            log(f"build: ⛔ {_dropped} breakout(s) cannot be captioned through their final spoken "
                f"word (see breakout_caption_coverage.json) — DROPPING the breakout caption pass "
                f"rather than the render; no partial caption is burned")
            _breakout_caps_enabled = False
            _caps_pre = []
            # The _caps metadata itself STAYS: it is what keeps the main narration caption off the
            # breakout's real-audio window. Only the burn is skipped.
            _breakout_caption_burn_ok = False

    # 4) theme + 5) render
    th = get_theme(theme_name)
    # CAPTION PRESET — the selected preset OWNS the narration caption look (font / size / weight /
    # primary + active-word colour / outline / shadow / backplate / vertical margin). Override the
    # theme's caption dict, and set a caption-SCOPED `caption_accent` for the active word — NOT the
    # theme's global `accent` (which colours title/graphic overlays and the key-phrase stabs, so a
    # caption-style choice must never recolour those; the stabs even fire only when captions are
    # OFF). The cinematic letterbox lift below still raises margin_v on top (all presets clear bars).
    _cap_dict, _cap_accent = _cap_preset.theme_caption()
    th = {**th, "caption": {**th.get("caption", {}), **_cap_dict}, "caption_accent": _cap_accent}
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
    # MUSIC DYNAMICS (Stage 3) — bake a natural envelope onto the bed BEFORE assemble: strong duck
    # across real-audio breakouts (dialogue is clearly loudest) + gentle swell on reveal/climax beats
    # (heard in narration gaps; the engine sidechain still ducks under the voice, so music is never
    # louder than narration). Engine mix untouched; the uploaded voiceover is never altered.
    _bk_wins = [(float(c["start"]), round(float(c["start"]) + float(c["dur"]), 2))
                for c in (getattr(narration, "_breakout_caps", None) or [])]
    _reveal_wins = []
    _nsc = {ns.index: ns for ns in getattr(narration, "scenes", [])}
    for _pos, _sc in enumerate(scenes):
        _role = (getattr(_sc, "role", "") or "").lower()
        _inten = int(getattr(_sc, "intensity", 0) or 0)
        if _role in ("reveal", "climax") or _inten >= 5:
            _seg = segments[_pos] if _pos < len(segments) else None
            _ns = _nsc.get(getattr(_seg, "index", -1)) if _seg is not None else None
            _ws = (getattr(_ns, "words", None) or []) if _ns is not None else []
            if _ws:
                _reveal_wins.append((min(float(w.start) for w in _ws),
                                     max(float(w.end) for w in _ws)))
    _reveal_wins = _reveal_wins[:12]                    # bound the volume-expression length
    _resolved_music = _resolve_music(music, theme_name,
                                     getattr(narration, "total", 0.0) + 1.0, work, log=log)
    # NEVER a silent-music render by accident (the voiceover-never-silent doctrine, applied to
    # music): a full 19-minute video shipped with no bed because two silent fallthroughs stacked.
    # A music-less render now requires the explicit env override.
    import os as _os_mus
    if not _resolved_music and _os_mus.environ.get(
            "VIDLORE_CLIPSTUDIO_ALLOW_NO_MUSIC", "0").strip().lower() not in ("1", "true", "yes"):
        raise RuntimeError(
            "music resolution failed (no composed score AND no fallback track) — refusing a "
            "silent-music render. Check VIDLORE_MUSIC_DIR / vidlore/assets/music, or set "
            "VIDLORE_CLIPSTUDIO_ALLOW_NO_MUSIC=1 to render without a bed on purpose.")
    _music_track = _shape_music_envelope(
        _resolved_music,
        getattr(narration, "total", 0.0), _bk_wins, _reveal_wins, work, log=log)
    # assemble() internally does len() on these per-scene lists, so pass them explicitly as
    # the engine's own pipeline does (its None defaults are a latent bug that never fires there).
    result = assemble(
        footage, narration, th, work, out_path,
        beat_clips=beat_clips, captions=_cap_on,
        # music: a real cinematic bed with a baked dynamics envelope (see above). The engine's full
        # sidechain-duck chain fires on top when a valid track PATH is passed.
        music=_music_track,
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
        # a black span inside a breakout window is a hard failure, never an intentional fade:
        # tell the black-frame repair to freeze-fill (not preserve) any it finds there.
        breakout_windows=[(float(c["start"]), round(float(c["start"]) + float(c["dur"]), 2))
                          for c in (getattr(narration, "_breakout_caps", None) or [])],
        # Independent renderer invariant: bind every encode-plan beat to the exact owned file,
        # fingerprint decoded encoded segments, then verify those beats remain in order after
        # concat/conform.  Construction metadata alone is not accepted as proof.
        scene_lineage={"entries": _scene_lineage},
        # Same review-draft contract the footage / unverified-exact / black-frame gates already
        # follow: a cue that only reads FASTER than the ceiling is reported in
        # caption_readability_audit.json instead of throwing away the finished video. When the
        # narration itself outruns the ceiling no regrouping can help — reading speed over a span
        # is chars/duration, which splitting and merging leave unchanged. Publication ('block',
        # the default) is untouched, and a structurally broken cue still fails in both modes.
        caption_readability=(
            "warn" if _os.environ.get(
                "VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE", "block").strip().lower() == "warn"
            else "block"),
    )
    result = Path(result)
    if cinematic:
        if _apply_cinematic_letterbox(result, _bar_h):
            log(f"build: cinematic bars baked onto {result.name}")
        else:
            log("build: cinematic letterbox post-pass failed — kept flat frame")
    # WORD-BY-WORD BREAKOUT CAPTIONS — during real-audio breakouts the narration caption is
    # silent, so caption the SCENE's own spoken line (karaoke fill, styled to match the SAME
    # selected preset family). Engine untouched; burned as a clipstudio post-pass. Skipped when
    # captions are OFF — but the breakout _caps metadata (suppression/audit/QA) always stays.
    if _cap_on and os.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_CAPS", "1").strip() \
            not in ("0", "false", "no"):
        _caps = list(getattr(narration, "_breakout_caps", None) or [])
        if _caps and not _breakout_caption_burn_ok:
            log("build: breakout caption burn skipped — the preflight could not caption every "
                "spoken word; the breakout airs with its real audio and no caption, and the main "
                "narration caption stays suppressed over its window")
            _caps = []
        if _caps:
            _bk_burn_ok = _burn_breakout_captions(
                result, _caps, work, log, preset=_cap_preset,
                ass_path=_breakout_ass_preflight)
            if not _bk_burn_ok:
                _qbk = result.with_name(
                    result.stem + ".FAILED_BREAKOUT_CAPTION" + result.suffix)
                try:
                    result.replace(_qbk)
                except Exception:
                    _qbk = result
                from .verify import NonRetryableBuildError
                raise NonRetryableBuildError(
                    f"breakout-caption burn failed; incomplete/absent dialogue captions cannot "
                    f"publish (quarantined at {_qbk.name})", kind="breakout_caption")
            log(f"build: word-by-word breakout captions burned ({len(_caps)} scene-line(s), "
                f"style={_cap_preset.name})")
    # POST-RENDER BREAKOUT VISUAL QA — the HARD publication gate (see _breakout_qa_gate).
    _final_caps = list(getattr(narration, "_breakout_caps", None) or [])
    if _final_caps and os.environ.get("VIDLORE_CLIPSTUDIO_BREAKOUT_QA", "1").strip() \
            not in ("0", "false", "no"):
        result = _breakout_qa_gate(result, _final_caps, work, log=log)
    # FINAL-VIDEO AD/BRANDING RELEASE GATE — scan the finished video every 0.5s for full-screen
    # promo / outro / CTA / streamer-brand slates (the Max/WarnerMedia end-slate class). Clip-stage
    # removal + Ken-Burns time-neutrality are the primary fix; this is the last-line block so a
    # promo frame can never SILENTLY ship (quarantine + raise on any survivor).
    result = _final_video_ad_gate(result, work, _ocr_eng, log=log, captions_burned=_cap_on)
    # FINAL-VIDEO SUSTAINED-BLACK / LEGIBILITY GATE — no near-black/unusable-dark footage may ship
    # (distinct from the assemble true-black repair). Short fades are allowed; sustained illegible
    # regions block publication.
    result = _final_video_black_gate(result, work, log=log)

    # LAST MUTATING-PASS PROVENANCE CHECK.  assemble() proved its own mux, but
    # ClipStudio subsequently bakes letterbox bars, Breakout captions and final
    # QA filters.  Re-run the persisted bind-time canaries on the exact artifact
    # returned to the portal so a post-pass swap/reorder cannot inherit an old
    # PASS sidecar.
    from ..scene_lineage_canary import verify_delivered_output as _verify_delivered_lineage
    _verify_delivered_lineage(
        result, proj.output_dir / "scene_lineage_audit.json", stage="delivered_output")
    log(f"build: delivered scene-lineage canary PASS — {Path(result).name}")

    # DELIVERED A/V SYNC — measured on the artifact the viewer actually receives, after everything
    # above has re-encoded it: letterbox bake, caption burn, breakout QA, ad + black gates, mux.
    # The pre-mux check in assemble() proves the concat matched the composed audio; it cannot speak
    # for what any of those later passes did. Checks per-stream duration AND first/last PTS —
    # two streams can share a duration and still not start together, which is silent lip-sync error.
    # persist the aired-window record before the final gates, so it survives even a gate raise
    try:
        import json as _json_aw
        (proj.output_dir / "aired_windows.json").write_text(
            _json_aw.dumps({"schema": "aired_windows/2", "clips": _aired_windows}, indent=1),
            encoding="utf-8")
        _n_alt = sum(1 for a in _aired_windows if a.get("via") == "window[alt]")
        _n_walk = sum(1 for a in _aired_windows if a.get("via") == "walk")
        _n_owned = sum(1 for a in _aired_windows
                       if a.get("via") in ("selection", "selection_derivative"))
        log(f"build: aired-window record — {len(_aired_windows)} clip(s) "
            f"({_n_owned} owned selection derivative(s), {_n_alt} alternate, {_n_walk} walk) "
            f"→ aired_windows.json")
    except Exception as _e_aw:                            # noqa: BLE001
        log(f"build: aired-window record skipped ({type(_e_aw).__name__})")

    from ..assemble import assert_delivered_av_sync as _avsync
    _sync = _avsync(result)
    log(f"build: delivered A/V sync OK — video {_sync['video'][0]:.3f}s "
        f"[{_sync['video'][1]:.3f}→{_sync['video'][2]:.3f}] · audio {_sync['audio'][0]:.3f}s "
        f"[{_sync['audio'][1]:.3f}→{_sync['audio'][2]:.3f}] (tol {_sync['tol_s']*1000:.0f}ms)")
    _af = (_sync.get("audio_frames") or {})
    if _af.get("warning"):
        log(f"build: ⚠ {_af['warning']}")
    elif _af:
        log(f"build: audio timeline clean — {_af.get('frames')} frame(s), "
            f"{_af.get('anomalies', 0)} anomal(y/ies), true media {_af.get('true_media_s')}s")

    # A TOTAL HD COLLAPSE is a release block of its own. Every beat is legitimate and every gate
    # passes — the footage is simply 360p upscaled onto a 1080p canvas for the whole runtime, which
    # no per-beat check can see. MEASURED 2026-08-02: a 12-minute render shipped with hd_path_ok
    # 0/72 and 1% of sources at >=720p; the download stage said so, and the file still called
    # itself final. The threshold is total collapse, not degradation, so a render that merely lost
    # some sources to genuinely-HD-less uploads is never held back.
    try:
        _da = (proj.meta or {}).get("download_audit") or {}
        _yt_n = int(_da.get("youtube_sources") or 0)
        if _yt_n >= 5 and int(_da.get("hd_path_ok") or 0) == 0:
            _msg_hd = (f"HD path collapsed: 0 of {_yt_n} YouTube source(s) downloaded via the HD "
                       f"path — the entire video is built from legacy ~360p footage upscaled to "
                       f"the 1080p canvas. Reason: "
                       f"{str(_da.get('top_fallback_reason') or '?')[:160]}")
            _review_draft.append(_msg_hd)
            log(f"build: ⚠ {_msg_hd}")
    except Exception as _e_hd:                           # noqa: BLE001 — never block on the check
        log(f"build: hd-collapse check skipped ({type(_e_hd).__name__}: {_e_hd})")

    # REVIEW DRAFT — rename so the FILE itself says what it is. Done after the A/V gate so that
    # gate still measures the artifact it was written to measure, and the new path is returned so
    # the portal's download link (which follows res["output"]) keeps working.
    if _review_draft:
        try:
            _draft = result.with_name(result.stem + ".REVIEW_DRAFT" + result.suffix)
            result.replace(_draft)
            for _side in (".srt",):                      # keep the sidecar next to its video
                _s_old = result.with_suffix(_side)
                if _s_old.exists():
                    _s_old.replace(_draft.with_suffix(_side))
            result = _draft
            log(f"build: ⚠ REVIEW DRAFT — NOT FOR PUBLICATION. Renamed → {result.name} "
                f"({len(_review_draft)} release-block report(s); see rejected_footage_audit.json). "
                f"Rerun after rediscovery before publishing.")
        except Exception as _e_rd:                       # a rename must never lose a finished render
            log(f"build: ⚠ REVIEW DRAFT (could not rename: {type(_e_rd).__name__}) — "
                f"{result.name} is NOT for publication")

    # assemble() fingerprints the MP4 at its own renderer boundary, but ClipStudio subsequently
    # re-encodes it for cinematic bars and breakout captions, and a review build may rename it.
    # Refresh only after every successful byte/path mutation so this sidecar describes the exact
    # artifact returned to the portal.  Keep the writer best-effort, matching assemble()'s contract.
    try:
        from ..assemble import FPS as _EXPORT_FPS
        from ..assemble import _write_export_metrics as _write_final_export_metrics
        _xm = _write_final_export_metrics(result, _EXPORT_FPS)
        log(f"build: delivered export metrics refreshed — {result.name} · "
            f"dur={_xm.get('duration_s')}s · sha256={(_xm.get('sha256') or '')[:12]}…")
    except Exception as _e_xm:                           # noqa: BLE001 — metadata never kills render
        log(f"build: delivered export metrics refresh skipped ({type(_e_xm).__name__}: {_e_xm})")
    log(f"build: done → {result}")
    return result
