"""Step 3: captions. Word timings -> grouped caption cues -> .ass subtitle
file (styled per theme). Vidlore ships captions off by default; here they
are on by default and toggled from the brief.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .tts import WordTiming

# A caption cue must NEVER visually HOLD across a long silence. A real-audio
# breakout (and any multi-second narration pause) leaves a gap in the word
# stream — the previous scene's last word ends, then several seconds later the
# next scene's first word starts. Without a cut here the cue spanning that gap
# freezes on screen for the WHOLE breakout (10-13s observed) AND swallows the
# next scene's first word ("...evil. This"). Breaking the cue at the gap lets the
# breakout's own-dialogue caption own that window and keeps every later caption
# locked to the voice. The threshold sits well above speech-rhythm / breath gaps
# (<=~0.8s, where holding reads cinematic) and well below a breakout (>=~3s).
try:
    _CUE_GAP_BREAK = float(os.environ.get("VIDLORE_CAPTION_GAP_BREAK_S", "1.0"))
except Exception:                                              # noqa: BLE001
    _CUE_GAP_BREAK = 1.0


def _norm(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


def _channel_caption_pace() -> tuple[int | None, float | None]:
    """Read Look-DNA `captions.max_words` and `captions.max_dur_s`
    overrides.  Returns (None, None) when no channel — `_group()`
    then keeps its legacy defaults (6 words / 3.4s).

    The channel cadence is the single biggest editorial signature
    a viewer reads from captions:

      • Atlas (dense explainer): max_words 3-4, max_dur 2.0-2.4
        → captions change every 1.5-2 seconds, feel "always-on
        editor explaining rapidly".
      • Amber (slow contemplative): max_words 9-10, max_dur 5-6
        → captions hold long across the breath, feel cinematic.
      • Midnight (premium investigative): defaults (6 / 3.4).
    """
    try:
        from .look_dna import current as _ld_current, look_get
        if _ld_current() is None:
            return None, None
        mw = look_get("captions.max_words")
        md = look_get("captions.max_dur_s")
        return (int(mw) if mw else None,
                float(md) if md else None)
    except Exception:                                       # noqa: BLE001
        return None, None


def _group(words: list[WordTiming], max_words: int = 6, max_dur: float = 3.4):
    """Group word-timings into subtitle CUES.

    Latin/RTL languages chunk on word count (default 6 words per cue).
    CJK has no spaces so each Whisper "word" can be a whole sentence
    fragment -- left to the default rule, a 30-character Japanese line
    flashes for 3 seconds and is unreadable.  We detect the script and
    relax `max_words` to 12 for CJK (each "word" is shorter / a kanji
    phrase) and tighten `max_dur` to 2.6 s so dense scripts get more
    cuts.  Latin behaviour is byte-identical to before.

    LOOK DNA OVERRIDE (P-subtitle):
      When an active channel declares `captions.max_words` /
      `captions.max_dur_s`, those override the caller's defaults so
      Atlas literally cuts captions faster and Amber holds them
      longer — same WordTiming source, different cadence."""
    if not words:
        return []
    # Channel override on cadence (P-subtitle).
    _ch_mw, _ch_md = _channel_caption_pace()
    if _ch_mw is not None:
        max_words = _ch_mw
    if _ch_md is not None:
        max_dur = _ch_md
    # Sample first few tokens to detect script (cheap heuristic)
    sample = "".join((getattr(w, "word", "") or "")[:4] for w in words[:8])
    try:
        from . import lang as _lang
        script = _lang.detect_script(sample)
    except Exception:                                          # noqa: BLE001
        script = "latin"
    if script in ("jp", "kr", "cjk"):
        # CJK: more "words" per cue (each Whisper token is shorter),
        # tighter max duration, and a soft cut after CJK punctuation.
        max_words = max(max_words, 12)
        max_dur = min(max_dur, 2.6)
        cut_punct = set("、。・！？；：")
        cues, buf = [], []
        for w in words:
            if buf and (float(w.start) - float(buf[-1].end)) >= _CUE_GAP_BREAK:
                cues.append(buf)            # hard cut on a breakout / long pause
                buf = []
            buf.append(w)
            span = buf[-1].end - buf[0].start
            last_ch = (getattr(w, "word", "") or "")[-1:]
            if (len(buf) >= max_words or span >= max_dur
                    or last_ch in cut_punct):
                cues.append(buf)
                buf = []
        if buf:
            cues.append(buf)
        return cues
    # Latin / RTL: original rule + a HARD CUT on long silence (a breakout or
    # multi-second pause): close the cue BEFORE the gapped word so it starts a
    # fresh cue instead of the current one freezing across the gap.
    cues, buf = [], []
    for w in words:
        if buf and (float(w.start) - float(buf[-1].end)) >= _CUE_GAP_BREAK:
            cues.append(buf)
            buf = []
        buf.append(w)
        span = buf[-1].end - buf[0].start
        if len(buf) >= max_words or span >= max_dur:
            cues.append(buf)
            buf = []
    if buf:
        cues.append(buf)
    return cues


def _ts(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    if cs == 100:
        cs, s = 0, s + 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_color(rgb: tuple) -> str:
    r, g, b = (int(c) & 0xFF for c in rgb)
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _esc(s: str) -> str:
    return s.replace("{", "(").replace("}", ")").replace("\n", " ").strip()


# SUBTITLE STYLE FAMILY (REC subtitle_style).  A per-video, RESTRAINED
# variation so two videos don't carry identical captions — readability-first,
# always bottom-positioned, never karaoke.  Tuple = (size_mult, margin_v_mult,
# emphasis_intensity, bounce_on_punch, font_kind).  emphasis_intensity scales
# how far above 100% the spoken / key / punch words pop (1.0 = legacy).
_SUBTITLE_STYLE = {
    "minimal":     (0.95, 1.06, 0.45, False, None),    # history/mystery — small, higher, gentle colour-lift, no bounce
    "clean_lower": (1.00, 1.00, 0.85, True,  None),    # neutral premium
    "bold_lower":  (1.06, 0.95, 1.15, True,  "sans"),  # explainer/business — bigger, tighter, punchier
    "serif_lower": (0.98, 1.00, 0.70, False, "serif"), # spy/disaster — serif, restrained
    "mono_lower":  (1.00, 0.94, 0.80, False, "mono"),  # true_crime — mono, tight, cold
}
_SUB_FONT = {"serif": "Georgia", "mono": "Courier New", "sans": "Arial"}


def write_ass(
    words: list[WordTiming],
    out_path: Path,
    *,
    style: dict,
    accent: tuple = (255, 210, 90),
    emphasis_words: set[str] | None = None,
    play_w: int = 1920,
    play_h: int = 1080,
) -> Path:
    """Kinetic captions. The word currently being spoken pops in the
    theme accent colour (bigger + bold). Words the LLM flagged as the
    scene's emotional punch (``emphasis_words``) stay accent-tinted
    across the whole phrase and hit hardest the instant they are spoken
    — Vidlore-style dynamic subtitles that drive retention. style keys:
    font, size, primary, outline, back, bold, border_style, outline_w,
    shadow, margin_v."""
    emphasis_words = emphasis_words or set()
    # Channel subtitle styling (P-subtitle).  Pulls font_size_mult,
    # fade timing, margin position, and emphasis-shake intensity
    # from Look DNA so Atlas reads tight + snappy, Amber reads
    # cinematic + soft, Midnight reads premium restrained.  All
    # fields optional — missing = legacy behaviour.
    _ch_size_mult = 1.0
    _ch_fade_in_ms = 140
    _ch_fade_out_ms = 140
    _ch_margin_v_mult = 1.0
    _ch_font_chain: list = []
    try:
        from .look_dna import current as _ld_current, look_get
        if _ld_current() is not None:
            _ch_size_mult     = float(look_get("captions.size_mult", 1.0) or 1.0)
            _ch_fade_in_ms    = int(look_get("captions.fade_in_ms", 140) or 140)
            _ch_fade_out_ms   = int(look_get("captions.fade_out_ms", 140) or 140)
            _ch_margin_v_mult = float(look_get("captions.margin_v_mult", 1.0) or 1.0)
            _cf = look_get("captions.font_family", []) or []
            if isinstance(_cf, list):
                _ch_font_chain = [str(f).strip() for f in _cf if str(f).strip()]
    except Exception:                                       # noqa: BLE001
        pass
    # REC subtitle_style — per-video restrained family (size / vertical
    # position / emphasis intensity / font), recorded-but-unconsumed before.
    # env VIDLORE_SUBTITLE_STYLE=0 → legacy.
    _ss_size, _ss_margin, _ss_emph, _ss_bounce, _ss_font = 1.0, 1.0, 1.0, True, None
    try:
        import os as _os
        if _os.environ.get("VIDLORE_SUBTITLE_STYLE", "1").strip().lower() \
                not in ("0", "false", "no", "off"):
            from .look_dna import look_get as _lg_ss
            _p = _SUBTITLE_STYLE.get((_lg_ss("subtitle_style") or "").strip().lower())
            if _p:
                _ss_size, _ss_margin, _ss_emph, _ss_bounce, _ss_font = _p
    except Exception:                                       # noqa: BLE001
        pass
    _base_size = int(style["size"] * 1.06)
    big = max(18, int(_base_size * _ch_size_mult * _ss_size))
    hi = _ass_color(accent)
    # Channel font override — use the first family from the chain
    # that ASS can resolve via the OS font registry.  When unset,
    # keep the theme-supplied font.
    _ass_font = style["font"]
    if _ch_font_chain:
        _ass_font = _ch_font_chain[0]
    elif _ss_font and _ss_font in _SUB_FONT:    # subtitle_style font (no channel override)
        _ass_font = _SUB_FONT[_ss_font]
    # RTL-aware subtitle alignment: Arabic / Hebrew / Urdu captions
    # natively right-align (Alignment=3 in ASS = bottom-right) so the
    # text reads from the entry side.  Latin / CJK keep bottom-center
    # (Alignment=2) -- the existing behaviour.
    _ass_align = 2
    _is_rtl_caption = False
    try:
        from . import lang as _lang
        _sample_cap = "".join((getattr(w, "word", "") or "")[:6]
                               for w in words[:6])
        if _lang.is_rtl(_sample_cap):
            _ass_align = 3
            _is_rtl_caption = True
    except Exception:                                          # noqa: BLE001
        pass
    # ── RTL margin: right-anchored captions want a slightly looser
    # right margin (so the punctuation / period at the visual line-end
    # doesn't crowd the bezel) and a tighter left margin (the line is
    # free to extend to the left).  ASS Alignment=3 anchors at the
    # bottom-right corner, so MarginR pushes IN from the right edge.
    if _is_rtl_caption:
        _ml, _mr = 60, 140
    else:
        _ml, _mr = 90, 90
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{_ass_font},{big},{style['primary']},{style['outline']},{style['back']},{1 if style['bold'] else 0},{style['border_style']},{style['outline_w']},{style['shadow']},{_ass_align},{_ml},{_mr},{int(style['margin_v'] * _ch_margin_v_mult * _ss_margin)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    # subtitle_style emphasis intensity — scale how far above 100% the
    # spoken / key / punch words pop (1.0 = legacy 116/110/138-150).
    def _sc(v: int) -> int:
        return int(round(100 + (v - 100) * _ss_emph))
    _e_p0, _e_pk = _sc(138), _sc(150)          # punch rest, peak
    _e_spoken, _e_key = _sc(116), _sc(110)
    if _ss_bounce:
        _punch = (f"\\b1\\c{hi}\\fscx{_e_p0}\\fscy{_e_p0}"
                  f"\\t(0,120,\\fscx{_e_pk}\\fscy{_e_pk})"
                  f"\\t(120,260,\\fscx{_e_p0}\\fscy{_e_p0})")
    else:
        _punch = f"\\b1\\c{hi}\\fscx{_e_p0}\\fscy{_e_p0}"
    _spoke = f"\\b1\\c{hi}\\fscx{_e_spoken}\\fscy{_e_spoken}"
    _keyw = f"\\b1\\c{hi}\\fscx{_e_key}\\fscy{_e_key}"
    lines = [header]
    cues = _group(words)
    for cue in cues:
        toks = [_esc(w.word) for w in cue]
        n = len(cue)
        for k, w in enumerate(cue):
            ws = w.start
            we = max(w.end, ws + 0.06)
            if k == n - 1:                       # hold last word to cue end
                we = max(we, cue[-1].end)
            parts = []
            for j, tk in enumerate(toks):
                is_emph = _norm(cue[j].word) in emphasis_words
                if j == k and is_emph:           # the punch word, on beat
                    parts.append(f"{{{_punch}}}{tk}{{\\r}}")
                elif j == k:                     # the spoken word: pop
                    parts.append(f"{{{_spoke}}}{tk}{{\\r}}")
                elif is_emph:                    # key word, always lifted
                    parts.append(f"{{{_keyw}}}{tk}{{\\r}}")
                else:
                    parts.append(tk)
            txt = " ".join(parts)
            fade = ""
            if k == 0:
                fade = "{\\fad(%d,0)}" % _ch_fade_in_ms
            elif k == n - 1:
                fade = "{\\fad(0,%d)}" % _ch_fade_out_ms
            lines.append(
                f"Dialogue: 0,{_ts(ws)},{_ts(we)},Main,,0,0,0,,{fade}{txt}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
