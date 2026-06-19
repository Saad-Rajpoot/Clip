"""PROFESSIONAL SOUND-DESIGN LIBRARY (procedural, royalty-free).

A categorised palette of 100+ documentary sound-effect *presets* — all
SYNTHESISED on the fly with ffmpeg lavfi (sine tones + colored noise +
filters + envelopes), so there are zero licensing concerns and the pack can
later be swapped for premium files behind the same API.

Design goals (MagnatesMedia / Netflix / Johnny Harris sound design):
  • every important on-screen MOTION gets a subtle, MATCHING micro-sound,
  • nothing is loud, cheesy or trailer-like (documentary-grade, restrained),
  • the SAME sound is never spammed — a matcher rotates variants and applies
    seeded pitch / volume / timing jitter + per-category cooldowns,
  • selection is intelligent: visual-event + template + intensity aware.

Public API:
  pick(event_kind, *, intensity, seed, history) -> (preset_name, spec)
  build_event_bed(events, duration, dest, cache_dir=None) -> Path
      events = [(time_s, event_kind, intensity0..1), ...]

`event_kind` is a high-level visual event (e.g. 'map_pin', 'stamp',
'doc_highlight', 'timeline_tick', 'process_step', 'stat_tick',
'node_connect', 'reveal', 'impact', 'transition', 'radar_ping', ...). The
matcher resolves it to a concrete preset in the right CATEGORY.
"""
from __future__ import annotations

from pathlib import Path

from .cache import scene_key
from .ffmpeg_tool import run

SR = 44100

# ====================================================================== #
#  Preset registry — every preset is a synth spec:
#    {cat, src(lavfi source), chain(post-filters), vol(base 0..1), dur(s)}
#  Volumes are deliberately LOW (documentary). The master bed is attenuated
#  again at mix time and ducked under narration.
# ====================================================================== #
PRESETS: dict[str, dict] = {}
_POOL: dict[str, list[str]] = {}      # category -> [preset names]


def _voice(cat: str, *, freq: float | None = None, color: str | None = None,
           dur: float = 0.12, hp: int | None = None, lp: int | None = None,
           fin: float = 0.004, fout: float | None = None, vol: float = 0.4,
           swell: bool = False, echo: bool = False) -> dict:
    """Build one preset spec. `freq` -> tonal (sine); else colored noise.
    `swell` ramps in then fades (pulses/risers); otherwise a transient with
    a fast attack + decay. `echo` adds a short ping tail (radar/UI)."""
    if freq is not None:
        src = f"sine=frequency={freq}:sample_rate={SR}:duration={dur:.2f}"
    else:
        src = (f"anoisesrc=color={color or 'white'}:sample_rate={SR}:"
               f"amplitude=0.8:d={dur:.2f}")
    ch: list[str] = []
    if hp:
        ch.append(f"highpass=f={hp}")
    if lp:
        ch.append(f"lowpass=f={lp}")
    fo = fout if fout is not None else max(0.02, dur * 0.6)
    fo_st = max(0.01, dur - fo)
    if swell:
        ramp = max(0.05, dur * 0.5)
        ch.append(f"volume='min(1\\,t/{ramp:.2f})':eval=frame")
        ch.append(f"afade=t=out:st={fo_st:.2f}:d={fo:.2f}")
    else:
        ch.append(f"afade=t=in:d={fin:.3f}")
        ch.append(f"afade=t=out:st={fo_st:.2f}:d={fo:.2f}")
    if echo:
        ch.append("aecho=0.8:0.7:45:0.35")
    return {"cat": cat, "src": src, "chain": ",".join(ch),
            "vol": round(vol, 3), "dur": round(dur, 2)}


def _add(name: str, spec: dict) -> None:
    PRESETS[name] = spec
    _POOL.setdefault(spec["cat"], []).append(name)


def _series(cat: str, base: str, n: int, **kw) -> None:
    """Register `n` seeded variants of a preset family (suffix a,b,c...).
    Each variant nudges freq / filter band / duration so rotation never
    sounds stamped, while staying inside the family's character."""
    for i in range(n):
        s = dict(kw)
        f = (i - (n - 1) / 2)
        if s.get("freq"):
            s["freq"] = round(s["freq"] * (1.0 + 0.05 * f), 1)
        if s.get("hp"):
            s["hp"] = int(s["hp"] * (1.0 + 0.07 * f))
        if s.get("lp"):
            s["lp"] = int(s["lp"] * (1.0 + 0.06 * f))
        if s.get("dur"):
            s["dur"] = round(s["dur"] * (1.0 + 0.08 * f), 3)
        _add(f"{base}_{chr(97 + i)}", _voice(cat, **s))


# ---- 1) TEXT REVEAL -------------------------------------------------- #
_series("text_reveal", "tr_whoosh", 4, color="brown", dur=0.46, hp=440,
        lp=3600, vol=0.30)
_series("text_reveal", "tr_paper_swipe", 3, color="white", dur=0.18, hp=1500,
        lp=6500, vol=0.22)
_series("text_reveal", "tr_breath", 3, color="pink", dur=0.5, hp=220, lp=1900,
        vol=0.18, swell=True)
_add("tr_keyword_tap", _voice("text_reveal", freq=2600, dur=0.04, vol=0.22))

# ---- 2) IMPACT HITS -------------------------------------------------- #
_series("impact", "im_boom", 4, freq=54, dur=1.15, lp=170, vol=0.62,
        fout=0.9)
_series("impact", "im_soft_hit", 2, freq=82, dur=0.6, lp=320, vol=0.42,
        fout=0.5)
_add("im_dark", _voice("impact", freq=46, dur=1.3, lp=130, vol=0.66,
                       fout=1.05))
_add("im_reveal_thud", _voice("impact", freq=60, dur=0.9, lp=200, vol=0.55,
                              fout=0.75))
_add("im_evidence", _voice("impact", freq=72, dur=0.7, lp=260, vol=0.5,
                           fout=0.55))
# softer sibling so the "evidence" diegetic hit rotates (no back-to-back
# identical thud when two evidence beats land close) — prefix-matched by
# EVENT_PREFER["evidence"]="im_evidence".
_add("im_evidence_soft", _voice("impact", freq=88, dur=0.5, lp=300, vol=0.4,
                                fout=0.4))
_add("im_verdict", _voice("impact", freq=50, dur=1.2, lp=150, vol=0.64,
                          fout=0.95))

# ---- 3) MAP SFX ------------------------------------------------------ #
_series("map", "mp_pin", 3, freq=330, dur=0.12, lp=900, vol=0.42, fout=0.1)
_series("map", "mp_pulse", 3, freq=520, dur=0.5, vol=0.24, swell=True)
_series("map", "mp_route_draw", 3, color="brown", dur=0.6, hp=520, lp=3400,
        vol=0.46, swell=True)
_add("mp_location_lock", _voice("map", freq=760, dur=0.12, vol=0.3))
_series("map", "mp_radar", 2, freq=1400, dur=0.3, vol=0.22, echo=True)
_add("mp_region_swell", _voice("map", color="brown", dur=1.0, lp=2400,
                               vol=0.3, swell=True))
_add("mp_tick", _voice("map", freq=2200, dur=0.03, vol=0.18))

# ---- 4) UI / HUD ----------------------------------------------------- #
_series("ui", "ui_blip", 3, freq=1800, dur=0.05, vol=0.24)
_series("ui", "ui_beep", 3, freq=2200, dur=0.08, vol=0.22)
_add("ui_target_lock", _voice("ui", freq=1300, dur=0.16, vol=0.26, echo=True))
_series("ui", "ui_tick", 2, freq=3000, dur=0.03, vol=0.18)
_add("ui_chirp", _voice("ui", freq=2600, dur=0.06, vol=0.2))
_add("ui_pulse", _voice("ui", freq=900, dur=0.3, vol=0.2, swell=True))
# terminal / keyboard typing — short clicky mechanical key ticks. For the
# text_on_black cipher / intercepted-message card, which should TYPE on
# screen, never whoosh. Soft texture; NOT in the throttled whoosh family.
_series("ui", "kb_key", 4, color="white", dur=0.025, hp=1500, lp=6000, vol=0.16)

# ---- 5) DOCUMENT / ARCHIVE ------------------------------------------- #
_series("document", "doc_paper_slide", 3, color="white", dur=0.22, hp=1200,
        lp=7000, vol=0.22)
_add("doc_folder_open", _voice("document", color="pink", dur=0.3, hp=600,
                               lp=4000, vol=0.2))
_add("doc_page_flip", _voice("document", color="white", dur=0.16, hp=2000,
                             lp=8000, vol=0.2))
_series("document", "doc_stamp", 3, freq=120, dur=0.2, lp=420, vol=0.52,
        fout=0.16)
_series("document", "doc_highlighter", 3, color="pink", dur=0.32, hp=800,
        lp=4200, vol=0.22)
_add("doc_tape_click", _voice("document", freq=1500, dur=0.04, vol=0.24))
_add("doc_scanner_hum", _voice("document", color="brown", dur=0.8, lp=1400,
                               vol=0.16, swell=True))

# ---- 6) TIMELINE ----------------------------------------------------- #
_series("timeline", "tl_marker_tick", 3, freq=2600, dur=0.04, vol=0.22)
_series("timeline", "tl_date_lock", 3, freq=700, dur=0.12, vol=0.3, fout=0.1)
_add("tl_line_draw", _voice("timeline", color="brown", dur=0.5, hp=600,
                            lp=3000, vol=0.44, swell=True))
_add("tl_reveal_hit", _voice("timeline", freq=88, dur=0.6, lp=300, vol=0.42,
                             fout=0.5))
_add("tl_time_pulse", _voice("timeline", freq=480, dur=0.35, vol=0.22,
                             swell=True))

# ---- 7) PROCESS / EXPLAINER ------------------------------------------ #
_series("process", "pr_step_pop", 4, freq=480, dur=0.09, vol=0.3, fout=0.07)
_add("pr_arrow_draw", _voice("process", color="white", dur=0.18, hp=1500,
                             lp=6000, vol=0.18))
_series("process", "pr_node_connect", 3, freq=900, dur=0.07, vol=0.26)
_add("pr_flow_pulse", _voice("process", freq=620, dur=0.3, vol=0.22,
                             swell=True))
_add("pr_diagram_build", _voice("process", color="pink", dur=0.4, hp=700,
                                lp=3800, vol=0.18, swell=True))
_add("pr_system_lock", _voice("process", freq=300, dur=0.22, lp=900,
                              vol=0.34, fout=0.16))

# ---- 8) DATA / STATISTIC --------------------------------------------- #
_series("data", "da_count_tick", 3, freq=2400, dur=0.03, vol=0.28)
_series("data", "da_settle", 3, freq=520, dur=0.26, vol=0.46, fout=0.2)
_add("da_bar_grow", _voice("data", color="brown", dur=0.4, lp=2600, vol=0.4,
                           swell=True))
_add("da_chart_reveal", _voice("data", color="pink", dur=0.45, hp=500,
                               lp=3600, vol=0.4, swell=True))
_add("da_graph_pulse", _voice("data", freq=820, dur=0.3, vol=0.34, swell=True))
_add("da_percent_hit", _voice("data", freq=96, dur=0.55, lp=320, vol=0.42,
                              fout=0.45))
# money / financial tick — a soft, bright, very short metallic register tick
# for currency stats (low-level diegetic texture, NOT a cartoon cha-ching).
_series("data", "da_money_tick", 3, freq=1720, dur=0.05, vol=0.44, fout=0.04)

# ---- 9) SURVEILLANCE / INVESTIGATION --------------------------------- #
_add("sv_shutter", _voice("surveillance", color="white", dur=0.08, hp=2200,
                          vol=0.3))
_series("surveillance", "sv_cctv_beep", 2, freq=1900, dur=0.1, vol=0.24)
_add("sv_rec_click", _voice("surveillance", freq=1200, dur=0.05, vol=0.26))
_add("sv_static", _voice("surveillance", color="white", dur=0.25, hp=1000,
                         lp=6000, vol=0.2))
_add("sv_radio_crackle", _voice("surveillance", color="pink", dur=0.3,
                                hp=1400, lp=4200, vol=0.2))
_add("sv_tag_snap", _voice("surveillance", freq=200, dur=0.08, lp=700,
                           vol=0.36, fout=0.06))

# ---- 10) TRANSITION -------------------------------------------------- #
_series("transition", "tx_whoosh", 4, color="brown", dur=0.5, hp=400,
        lp=3800, vol=0.32)
_add("tx_air_pass", _voice("transition", color="pink", dur=0.45, hp=600,
                           lp=4400, vol=0.26))
_add("tx_dark_riser", _voice("transition", color="brown", dur=1.4, lp=2500,
                             vol=0.3, swell=True))
_add("tx_tension_swell", _voice("transition", color="brown", dur=1.6, lp=1800,
                                vol=0.3, swell=True))
_add("tx_hard_cut_hit", _voice("transition", freq=58, dur=0.4, lp=150,
                               vol=0.46, fout=0.32))
_add("tx_dissolve_breath", _voice("transition", color="pink", dur=0.7,
                                  hp=250, lp=2000, vol=0.2, swell=True))
_add("tx_flashback_wash", _voice("transition", color="white", dur=0.9,
                                 hp=300, lp=5000, vol=0.22, swell=True))

# ---- 11) KINETIC TYPOGRAPHY ------------------------------------------ #
_series("kinetic", "kt_word_pop", 4, freq=600, dur=0.06, vol=0.24, fout=0.05)
_add("kt_emphasis_hit", _voice("kinetic", freq=90, dur=0.5, lp=300, vol=0.42,
                               fout=0.42))
_add("kt_text_slam", _voice("kinetic", freq=70, dur=0.6, lp=200, vol=0.5,
                            fout=0.5))
_add("kt_masked_sweep", _voice("kinetic", color="white", dur=0.3, hp=900,
                               lp=5200, vol=0.22, swell=True))
_add("kt_phrase_pulse", _voice("kinetic", freq=520, dur=0.28, vol=0.22,
                               swell=True))
_add("kt_final_settle", _voice("kinetic", freq=160, dur=0.5, lp=600,
                               vol=0.34, fout=0.42))


# ====================================================================== #
#  Event-kind -> category routing (what visual moment sounds like what)
# ====================================================================== #
EVENT_CATEGORY = {
    "reveal": "text_reveal", "text_reveal": "text_reveal",
    "phrase": "text_reveal",
    "impact": "impact", "boom": "impact", "land": "impact",
    "map_pin": "map", "map_pulse": "map", "map_route": "map",
    "map_region": "map", "radar_ping": "map", "location_lock": "map",
    "ui": "ui", "blip": "ui", "beep": "ui", "target_lock": "ui",
    "keyboard": "ui", "kb_key": "ui", "terminal": "ui",
    "doc_slide": "document", "doc_highlight": "document", "stamp": "document",
    "page_flip": "document", "folder_open": "document",
    "timeline_tick": "timeline", "date_lock": "timeline",
    "timeline_draw": "timeline",
    "process_step": "process", "node_connect": "process",
    "arrow_draw": "process", "system_lock": "process",
    "stat_tick": "data", "stat_settle": "data", "bar_grow": "data",
    "percent_hit": "data", "chart_reveal": "data", "money_tick": "data",
    "evidence": "impact",
    "surveillance": "surveillance", "shutter": "surveillance",
    "rec": "surveillance", "tag_snap": "surveillance",
    "transition": "transition", "whoosh": "transition",
    "riser": "transition", "dissolve": "transition",
    "word_pop": "kinetic", "emphasis": "kinetic", "text_slam": "kinetic",
}

# Preferred preset families per event kind (the matcher rotates within).
EVENT_PREFER = {
    "map_pin": "mp_pin", "map_pulse": "mp_pulse", "map_route": "mp_route_draw",
    "map_region": "mp_region_swell", "radar_ping": "mp_radar",
    "location_lock": "mp_location_lock",
    "stamp": "doc_stamp", "doc_highlight": "doc_highlighter",
    "doc_slide": "doc_paper_slide", "page_flip": "doc_page_flip",
    "timeline_tick": "tl_marker_tick", "date_lock": "tl_date_lock",
    "timeline_draw": "tl_line_draw",
    "process_step": "pr_step_pop", "node_connect": "pr_node_connect",
    "arrow_draw": "pr_arrow_draw", "system_lock": "pr_system_lock",
    "stat_tick": "da_count_tick", "stat_settle": "da_settle",
    "bar_grow": "da_bar_grow", "percent_hit": "da_percent_hit",
    "chart_reveal": "da_chart_reveal", "money_tick": "da_money_tick",
    "evidence": "im_evidence",
    "shutter": "sv_shutter", "rec": "sv_rec_click", "tag_snap": "sv_tag_snap",
    "keyboard": "kb_key", "terminal": "kb_key",
    "reveal": "tr_whoosh", "phrase": "tr_breath", "impact": "im_boom",
    "transition": "tx_whoosh", "whoosh": "tx_whoosh",
    "dissolve": "tx_dissolve_breath", "riser": "tx_dark_riser",
    "word_pop": "kt_word_pop", "emphasis": "kt_emphasis_hit",
    "text_slam": "kt_text_slam",
}


# ====================================================================== #
#  TIER GAIN — documentary sound-design dynamics (2026-06-01).
#  Prior builds were "subtle by design" then attenuated again at mix, so
#  EVERY cue measured -24..-47 dB eff peak — inaudible under a -16 LUFS
#  voice (measured: 0/7 audible). A professional doc is TIERED: a MAJOR
#  reveal clearly punches, a MID moment supports, a MINOR tick stays subtle
#  but present. We scale each event's synth vol by its tier so a single
#  sensible mix gain makes majors audible without making ticks spammy.
# ====================================================================== #
_TIER_GAIN = {"major": 2.8, "mid": 2.05, "minor": 1.45}
_EVENT_TIER = {
    # MAJOR — the beat owns the moment
    "impact": "major", "boom": "major", "land": "major", "evidence": "major",
    "percent_hit": "major", "stamp": "major", "text_slam": "major",
    "emphasis": "major", "reveal_thud": "major", "verdict": "major",
    "money_hit": "major", "hard_cut": "major",
    # MINOR — supporting ticks / blips / texture (present, never dominate)
    "timeline_tick": "minor", "stat_tick": "minor", "date_lock": "minor",
    "count_tick": "minor", "money_tick": "minor", "blip": "minor",
    "beep": "minor", "tick": "minor", "kb_key": "minor", "keyboard": "minor",
    "terminal": "minor", "node_connect": "minor", "ui": "minor",
    "map_pulse": "minor", "radar_ping": "minor", "word_pop": "minor",
}


def _tier_of(kind: str, q: float) -> str:
    t = _EVENT_TIER.get(kind)
    if t:
        return t
    return "major" if q >= 0.82 else "mid"


def tier_gain(kind: str, q: float) -> float:
    return _TIER_GAIN[_tier_of(kind, q)]


def _seed(t: float, salt: int = 0) -> int:
    return (int(round(t * 1000)) * 2654435761 + salt * 40503) & 0x7FFFFFFF


def pick(event_kind: str, *, intensity: float = 0.6, seed: float = 0.0,
         history: dict | None = None) -> tuple:
    """Resolve a visual event to a concrete preset, ROTATING variants and
    honouring a per-category cooldown so the same sound is never spammed.
    `history` (mutable dict) tracks recently-used names per category."""
    history = history if history is not None else {}
    cat = EVENT_CATEGORY.get(event_kind, "text_reveal")
    # candidate pool: prefer the family that matches the event kind, but
    # allow the whole category so rotation has room to vary.
    fam = EVENT_PREFER.get(event_kind)
    # rotate WITHIN the matched family first (e.g. mp_pin_a/b/c); fall back
    # to the whole category, then the whole library.
    pool = [n for n in _POOL.get(cat, []) if fam and n.startswith(fam)]
    if not pool:
        pool = list(_POOL.get(cat, [])) or list(PRESETS.keys())
    recent = history.get(cat, [])
    avail = [n for n in pool if n not in recent] or pool
    idx = _seed(seed, 7) % len(avail)
    name = avail[idx]
    recent = (recent + [name])[-3:]               # cooldown window = 3
    history[cat] = recent
    return name, PRESETS[name]


def _build_event_bed_single(
    norm: list, dur: float, dest: Path, *, batch_offset: int = 0
) -> Path:
    """Render a SINGLE batch of normalized events into one bed.

    Internal helper for `build_event_bed`. `norm` must already be the
    per-event tuples `(t, k, q)`. `dur` is the bed length. `batch_offset`
    is used for the seed jitter so the same event in different batches
    of a re-run still picks the same variant.
    """
    history: dict = {}
    inputs: list[str] = []
    parts: list[str] = []
    mix = "[base]"
    parts.append(f"[0:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS[base]")
    vi = 1
    for j, (t, k, q) in enumerate(norm):
        name, spec = pick(k, intensity=q, seed=t, history=history)
        sd = _seed(t, j + 1 + batch_offset)
        pitch = 0.94 + ((sd >> 3) % 130) / 1000.0
        volj = 0.85 + ((sd >> 9) % 260) / 1000.0
        tj = (((sd >> 5) % 60) - 30) / 1000.0
        vol = max(0.05, spec["vol"] * volj * (0.6 + 0.6 * q) * tier_gain(k, q))
        lead = spec["dur"] if ("swell" in spec["chain"]
                               or "riser" in name or "tension" in name
                               or "draw" in name or "region" in name) else 0.0
        start = max(0.0, t - lead + tj)
        ms = int(start * 1000)
        inputs += ["-f", "lavfi", "-i", spec["src"]]
        ch = spec["chain"]
        pitch_f = (f"asetrate={int(SR * pitch)},aresample={SR},"
                   if abs(pitch - 1.0) > 0.005 else "")
        parts.append(
            f"[{vi}:a]asetpts=PTS-STARTPTS,{pitch_f}{ch},"
            f"volume={vol:.3f},adelay={ms}|{ms}[v{j}]")
        mix += f"[v{j}]"
        vi += 1
        # v14.1 (forensic tune vs Vidlore AI): MAJOR reveals get a low-end WEIGHT
        # layer. Measured gap — Vidlore reveals had ~0.2% sub-20-80 Hz energy vs
        # Vidlore impacts ~1.8%. A short 52 Hz sine thump (lowpassed, fast attack +
        # decay, well under the voice) gives the reveal cinematic "chest" the
        # broadband whoosh lacks — without making the mix boomy or louder.
        if _tier_of(k, q) == "major":
            _sd = 0.45
            inputs += ["-f", "lavfi", "-i",
                       f"sine=frequency=52:sample_rate={SR}:duration={_sd}"]
            _sv = round(0.26 * (0.6 + 0.5 * q), 3)
            parts.append(
                f"[{vi}:a]lowpass=f=120,afade=t=in:d=0.008,"
                f"afade=t=out:st=0.10:d={_sd - 0.10:.2f},"
                f"volume={_sv:.3f},adelay={ms}|{ms}[w{j}]")
            mix += f"[w{j}]"
            vi += 1
    k = mix.count("[")
    parts.append(
        f"{mix}amix=inputs={k}:duration=first:normalize=0,"
        f"alimiter=limit=0.9,"
        f"aformat=sample_fmts=s16:sample_rates={SR}:"
        "channel_layouts=stereo[out]")
    run(["-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo"] + inputs
        + ["-filter_complex", ";".join(parts),
           "-map", "[out]", "-t", f"{dur:.2f}", str(dest)])
    return dest


def build_event_bed(events: list, duration: float, dest: Path, *,
                    cache_dir: Path | None = None) -> Path:
    """Render all SFX events into ONE bed. `events` = [(t, event_kind, q)].

    Each event is matched to a preset, then given seeded PITCH / VOLUME /
    TIMING jitter so repeats never sound identical, and mixed at its time.
    Subtle by design — the master is attenuated again at the final mix.

    BATCHED v13.1 (2026-05-28): on long renders (~25 min documentaries
    accumulate 150-250 reveal events), feeding every event as a separate
    `-f lavfi -i ...` input into ONE ffmpeg call spawns N×3 decoder
    pthreads, hitting the macOS `pthread_create()` limit at ~180+
    events. ("Resource temporarily unavailable" on the full Mossad
    render.) When `len(norm) > BATCH`, render in chunks of BATCH events
    each → per-chunk WAV → final pass that mixes the chunk WAVs (a much
    smaller graph of at most `ceil(N/BATCH)` inputs).
    """
    dur = max(1.0, round(float(duration), 2))

    norm = []
    for e in (events or []):
        if len(e) >= 3:
            t, k, q = float(e[0]), str(e[1]), float(e[2])
        else:
            t, k, q = float(e[0]), str(e[1]), 0.6
        if 0.0 <= t < dur:
            norm.append((round(t, 2), k, max(0.0, min(1.0, q))))
    norm.sort()

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = scene_key("sfxlib1", int(round(dur)), tuple(norm))
        cached = cache_dir / f"{key}.wav"
        if cached.exists():
            import shutil as _sh
            _sh.copyfile(cached, dest)
            return dest

    if not norm:
        run(["-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo",
             "-t", f"{dur:.2f}", str(dest)])
        if cache_dir is not None:
            import shutil as _sh
            _sh.copyfile(dest, cached)
        return dest

    # Pick a batch size that keeps a single ffmpeg under ~150 pthreads
    # (40 events × ~3 decoder threads each = ~120, plus output threads).
    BATCH = 40
    if len(norm) <= BATCH:
        _build_event_bed_single(norm, dur, dest, batch_offset=0)
    else:
        # Render each chunk to its own WAV, then mix the chunks.
        import tempfile as _tf
        chunks: list[Path] = []
        with _tf.TemporaryDirectory(prefix="sfxbed_") as _td:
            tdir = Path(_td)
            for ci, start in enumerate(range(0, len(norm), BATCH)):
                cnorm = norm[start:start + BATCH]
                cwav = tdir / f"chunk_{ci:03d}.wav"
                _build_event_bed_single(
                    cnorm, dur, cwav, batch_offset=start,
                )
                chunks.append(cwav)

            # Final pass: mix N chunk-beds (small ffmpeg graph).
            mix_inputs: list[str] = []
            mix_legs: list[str] = []
            label = ""
            for ci, cwav in enumerate(chunks):
                mix_inputs += ["-i", str(cwav)]
                mix_legs.append(f"[{ci}:a]")
                label += f"[{ci}:a]"
            mix_fc = (
                f"{label}amix=inputs={len(chunks)}:duration=first:"
                f"normalize=0,"
                f"alimiter=limit=0.9,"
                f"aformat=sample_fmts=s16:sample_rates={SR}:"
                "channel_layouts=stereo[out]"
            )
            run(mix_inputs + [
                "-filter_complex", mix_fc,
                "-map", "[out]", "-t", f"{dur:.2f}", str(dest),
            ])
        print(
            f"  [sfx] event-bed batched: {len(norm)} events → "
            f"{len(chunks)} chunks of <= {BATCH} (pthread-safe)",
            flush=True,
        )

    if cache_dir is not None:
        import shutil as _sh
        _sh.copyfile(dest, cached)
    return dest


def preset_count() -> int:
    return len(PRESETS)


def category_summary() -> dict:
    return {c: len(v) for c, v in _POOL.items()}
