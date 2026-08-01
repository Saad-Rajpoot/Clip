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


def _char_w_factor(ch: str) -> float:
    """Rough per-character advance as a fraction of the font size (proportional-font estimate;
    good enough to decide line breaks + a fit-scale, never pixel-exact). CJK/Hangul/kana are
    treated as ~1em wide."""
    o = ord(ch)
    if o < 0x20:
        return 0.0
    if ch == " ":
        return 0.32
    if (0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3
            or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE4F or 0xFF00 <= o <= 0xFF60
            or 0xFFE0 <= o <= 0xFFE6 or 0x3040 <= o <= 0x30FF):
        return 1.0
    if ch in "iIl.,:;'!|":
        return 0.30
    if ch in "mwMW@":
        return 0.86
    if ch.isupper():
        return 0.62
    return 0.52


def _est_px(text: str, fs: float) -> float:
    return sum(_char_w_factor(c) for c in text) * fs


import unicodedata as _ud


def _graphemes(s: str) -> list:
    """Split into grapheme-ish clusters: a base char plus any trailing combining marks stays one
    unit, so a break never lands in the middle of an accented / composed character."""
    out: list = []
    for ch in s:
        if out and _ud.category(ch) in ("Mn", "Mc", "Me"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def _pack_graphemes(text: str, fs: float, max_w: float) -> list:
    """Grapheme-aware pack of `text` into the fewest runs whose each estimated width <= max_w."""
    runs, cur, cw = [], "", 0.0
    for g in _graphemes(text):
        gw = _est_px(g, fs)
        if cur and cw + gw > max_w:
            runs.append(cur)
            cur, cw = "", 0.0
        cur += g
        cw += gw
    if cur:
        runs.append(cur)
    return runs or [text]


def split_wide_cells(tokens: list, base_fs: float, safe_w: float, *,
                     peak_extra: float = 0.5, pad: float = 0.0, min_fs_frac: float = 0.5):
    """Turn caption `tokens` into display CELLS that each fit one line — grapheme-splitting any
    pathological unbroken token (a 60/100-char run) that is wider than the safe area even at the
    minimum font. Returns (cells, index_map): index_map[i] = the original token index of cell i, so
    callers can map word emphasis / karaoke timing back after a split. Spoken text is never dropped.
    Only a genuinely un-wrappable token splits (a real long word like 'antidisestablishmentarianism'
    is left whole); when one does, it is cut into FINE runs so the two-line layout can balance them
    closely (less horizontal compression, more readable) rather than into a few coarse halves."""
    safe = max(120.0, float(safe_w) - float(pad))
    fs_floor = base_fs * min_fs_frac
    cell_budget = safe / (1.0 + peak_extra)                # decision: fits one line even at peak?
    run_budget = max(safe * 0.10, cell_budget / 3.0)       # when splitting, use fine runs to balance
    cells, imap = [], []
    for i, t in enumerate(tokens):
        if _est_px(t, fs_floor) > cell_budget and len(_graphemes(t)) > 1:
            for run in _pack_graphemes(t, fs_floor, run_budget):
                cells.append(run)
                imap.append(i)
        else:
            cells.append(t)
            imap.append(i)
    return cells, imap


def layout_two_lines(cells: list, base_fs: float, safe_w: float, *,
                     peak_extra: float = 0.5, pad: float = 0.0, min_fs_frac: float = 0.5):
    """GUARANTEED ≤2-line, no-clip layout of pre-sized `cells` within (safe_w - pad). Returns
    (break_index, fit_fs, squeeze): break_index = cells on line 1 (None = one line); fit_fs = font
    size (<= base_fs, floored); squeeze = horizontal \\fscx percent (100 = none).

    Normal path: the layout is evaluated at the active-word PEAK (the popped word grows by
    peak_extra, e.g. 0.5 = 150%) and fit_fs is shrunk so even the peak line fits — squeeze stays 100
    and the caller keeps its per-word emphasis animation. Pathological path (a 60/100-char unbroken
    token where even the floored font on two lines still overflows): the caller is signalled — via
    squeeze < 100 — to DROP the emphasis pop for that cue and apply a bounded horizontal compression
    so the text still fits exactly. Spoken text is never truncated and a third line never appears."""
    safe = max(120.0, float(safe_w) - float(pad))
    fs_floor = base_fs * min_fs_frac

    def line_w(seg, fs, pk):                                # width of a line of cells at scale `pk`
        if not seg:
            return 0.0
        wsum = sum(_est_px(c, fs) for c in seg)
        spc = _est_px(" ", fs) * max(0, len(seg) - 1)
        return wsum + spc + pk * max(_est_px(c, fs) for c in seg)

    # ONE line whenever the whole cue fits (peak reserved) — a break is only introduced when needed.
    if len(cells) <= 1 or line_w(cells, base_fs, peak_extra) <= safe:
        w1 = line_w(cells, base_fs, peak_extra)
        if w1 <= safe:
            return None, base_fs, 100
        # single un-splittable cell too wide even alone: shrink, then compress as a last resort
        fit1 = max(fs_floor, base_fs * safe / max(w1, 1.0))
        if line_w(cells, fit1, peak_extra) <= safe + 0.5:
            return None, fit1, 100
        fit1f = max(fs_floor, base_fs * safe / max(line_w(cells, base_fs, 0.0), 1.0))
        sq1 = int(safe * 0.98 / max(line_w(cells, fit1f, 0.0), 1.0) * 100)
        return None, fit1f, max(1, min(100, sq1))
    # TWO lines: pick the break that minimises the wider line (peak-aware), tie → most balanced
    best = None
    for b in range(1, len(cells)):
        w1, w2 = line_w(cells[:b], base_fs, peak_extra), line_w(cells[b:], base_fs, peak_extra)
        key = (round(max(w1, w2), 2), round(abs(w1 - w2), 2))
        if best is None or key < best[0]:
            best = (key, b)
    bidx = best[1]

    def widest(fs, pk):
        return max(line_w(cells[:bidx], fs, pk), line_w(cells[bidx:], fs, pk))

    # 1) fits at full size with the peak reserved → nothing to do
    if widest(base_fs, peak_extra) <= safe:
        return bidx, base_fs, 100
    # 2) shrink the font (floored) so the PEAK line fits → emphasis stays on, no compression
    fit_pk = max(fs_floor, base_fs * safe / max(widest(base_fs, peak_extra), 1.0))
    if widest(fit_pk, peak_extra) <= safe + 0.5:
        return bidx, fit_pk, 100
    # 3) pathological: even the floored font can't hold the peak on two lines. Drop the peak (caller
    #    turns off the emphasis pop) and compress horizontally the last, bounded amount to fit.
    fit_flat = max(fs_floor, base_fs * safe / max(widest(base_fs, 0.0), 1.0))
    w_flat = widest(fit_flat, 0.0)
    squeeze = int(safe * 0.98 / max(w_flat, 1.0) * 100)    # 0.98 guards rounding; always makes it fit
    return bidx, fit_flat, max(1, min(100, squeeze))


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
    # PRESET LOCK — when the caller declares an exact caption preset (ClipStudio's caption-preset
    # system sets style['preset_locked']=True), the per-channel Look-DNA overrides and the legacy
    # per-video subtitle_style family must NOT silently replace its font / size / margin / weight /
    # emphasis personality. Other engine callers omit the flag → existing Look-DNA behaviour intact.
    _locked = bool(style.get("preset_locked"))
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
        if not _locked and _ld_current() is not None:
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
        if not _locked and _os.environ.get("VIDLORE_SUBTITLE_STYLE", "1").strip().lower() \
                not in ("0", "false", "no", "off"):
            from .look_dna import look_get as _lg_ss
            _p = _SUBTITLE_STYLE.get((_lg_ss("subtitle_style") or "").strip().lower())
            if _p:
                _ss_size, _ss_margin, _ss_emph, _ss_bounce, _ss_font = _p
    except Exception:                                       # noqa: BLE001
        pass
    # PER-PRESET WORD MOTION — a caption preset declares its own active-word emphasis intensity +
    # bounce ('motion' in the style dict). It OVERRIDES the (skipped-when-locked) subtitle_style
    # motion so the preset owns the pop: minimal/cinematic/documentary stay restrained & un-bounced,
    # professional is a controlled premium lift, focus is the strongest word-synced emphasis. Because
    # locked presets never load subtitle_style, this is the SOLE motion source for them → locked.
    _motion = style.get("motion")
    if isinstance(_motion, dict):
        _ss_emph = float(_motion.get("emphasis", _ss_emph))
        _ss_bounce = bool(_motion.get("bounce", _ss_bounce))
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
    # emphasis intensity — scale how far above 100% the spoken / key / punch words pop, driven by
    # the preset's motion (_ss_emph). Minimal ~0.30 barely lifts; focus ~1.15 hits hardest.
    def _sc(v: int) -> int:
        return max(100, int(round(100 + (v - 100) * _ss_emph)))
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
    # LAYOUT — a cue is laid into AT MOST two lines that fit the horizontal SAFE area even at the
    # active-word PEAK. The safe width reserves the L/R margins PLUS the outline+shadow bleed; the
    # peak factor is the widest state any word reaches (punch peak, e.g. 150% → 0.5). A single
    # over-wide token is grapheme-split (split_wide_cells) so it wraps within the two lines; the
    # widest peak line is then shrunk with a bounded per-cue \fs — never a third line, never a clip,
    # never truncated. Break + fit are computed ONCE per cue so every karaoke frame shares the
    # identical layout; \r resets re-apply the fit \fs so it survives.
    _max_lines = int(style.get("max_lines", 2) or 2)
    _peak_extra = max(0.0, (max(_e_pk, _e_spoken, _e_key) - 100) / 100.0)
    _pad = 2.0 * (float(style.get("outline_w", 2) or 0) + float(style.get("shadow", 1) or 0) + 2.0)
    _safe_w = max(200.0, float(play_w) - _ml - _mr)
    lines = [header]
    cues = _group(words)
    # NEXT-EVENT START, across cue boundaries. The aligner can hand back words whose spans OVERLAP
    # (measured on a delivered render: word 797 starts at 251.260 while word 796 still ends at
    # 251.280). At a cue's last word there is no in-cue successor to bridge to, so that overlap
    # reaches the ASS — and libass, given two live events, stacks the newer one ABOVE the older,
    # displacing the caption a full line height. On this render it did that for 0.87s on a static
    # shot, preceded by a frame of the same sentence printed twice.
    # Used only to CLAMP an end down, never to extend one, so the deliberate gaps between cues
    # (sentence pauses) are untouched.
    _flat = [w for c in cues for w in c]
    _next_start = {}
    for _i, _w in enumerate(_flat[:-1]):
        _next_start[id(_w)] = _flat[_i + 1].start
    for cue in cues:
        toks = [_esc(w.word) for w in cue]
        n = len(cue)
        # display cells (over-wide words grapheme-split) + map back to the source word index
        cells, imap = split_wide_cells(toks, float(big), _safe_w, peak_extra=_peak_extra, pad=_pad)
        # a word that had to be grapheme-split spans multiple cells (possibly across the line break):
        # never pop it, because all its cells would scale at once and blow past the single-cell peak
        # the layout reserved. Such tokens are pathological anyway — they render calm and plain.
        _split_src = {j for j in set(imap) if imap.count(j) > 1}
        _bidx, _fit, _squeeze = (
            layout_two_lines(cells, float(big), _safe_w, peak_extra=_peak_extra, pad=_pad)
            if _max_lines >= 2 else (None, float(big), 100))
        # squeeze < 100 only on a pathological cue (a 60/100-char unbroken token): drop the emphasis
        # pop for THIS cue and apply a bounded horizontal \fscx compression so it fits without a
        # third line or truncation. \fscy stays 100 so text height (readability) is preserved.
        _emph_on = _squeeze >= 100
        _scaled = _fit < float(big) - 0.5
        _cue = (f"\\fs{int(round(_fit))}" if _scaled else "")
        if not _emph_on:
            _cue += f"\\fscx{_squeeze}"
        _reset = "\\r" + _cue
        _prefix = "{%s}" % _cue if _cue else ""
        for k, w in enumerate(cue):
            ws = w.start
            # BRIDGE TO THE NEXT WORD. One Dialogue event is emitted per word, each carrying the
            # WHOLE line with that word highlighted — so when an event ends at its own word's end,
            # the entire caption disappears until the next word begins. Measured on a 12-minute
            # render: 246 mid-sentence blackouts, 108.7s (15% of runtime), a hard caption pop every
            # ~2.9s. It reads as a broken render rather than a style.
            #
            # Ending each event where the NEXT word starts makes the line continuous through the
            # word's own trailing silence. Gaps BETWEEN cues survive untouched — those are the
            # sentence pauses (57 of them here, 29.9s) where the band is supposed to be empty.
            #
            # This also supersedes the old `ws + 0.06` minimum: that floor pushed an event's end
            # PAST the next event's start and produced 4 overlapping pairs, one of which displaced
            # the caption a full line height for 0.87s and printed the same sentence twice.
            we = max(w.end, cue[k + 1].start) if k < n - 1 else max(w.end, cue[-1].end)
            _nxt = _next_start.get(id(w))
            if _nxt is not None:
                we = min(we, _nxt)           # never outlive the next event — see the note above
            if we <= ws + 1e-4:
                # degenerate alignment — the aligner gave two tokens the same start (seen on an
                # em-dash followed by a word). A zero-length event renders as a flash; drop it and
                # let the previous event's bridge cover the span.
                continue
            parts = []
            for ci, tk in enumerate(cells):
                j = imap[ci]                     # source word index of this cell
                is_emph = _emph_on and (_norm(cue[j].word) in emphasis_words)
                if not _emph_on or j in _split_src:   # pathological cue / split word: plain, no pop
                    parts.append(tk)
                elif j == k and is_emph:         # the punch word, on beat
                    parts.append(f"{{{_punch}}}{tk}{{{_reset}}}")
                elif j == k:                     # the spoken word: pop
                    parts.append(f"{{{_spoke}}}{tk}{{{_reset}}}")
                elif is_emph:                    # key word, always lifted
                    parts.append(f"{{{_keyw}}}{tk}{{{_reset}}}")
                else:
                    parts.append(tk)
            if _bidx is None:
                txt = " ".join(parts)
            else:                                # exactly ONE line break at the chosen boundary
                txt = " ".join(parts[:_bidx]) + "\\N" + " ".join(parts[_bidx:])
            fade = ""
            if k == 0:
                fade = "{\\fad(%d,0)}" % _ch_fade_in_ms
            elif k == n - 1:
                fade = "{\\fad(0,%d)}" % _ch_fade_out_ms
            lines.append(
                f"Dialogue: 0,{_ts(ws)},{_ts(we)},Main,,0,0,0,,{fade}{_prefix}{txt}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
