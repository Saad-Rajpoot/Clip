"""Local visual-relevance scorer — pixel-aware candidate validation.

The footage ladder (see research/visual_relevance/VISUAL_RELEVANCE_AUDIT.md)
accepts a clip on QUERY/SLUG text overlap + a blank-frame check — it never looks
at the actual pixels, so a clip whose metadata *sounds* right (e.g. "aerial drone
over farmland road") is accepted even when the visible content is an interstate
interchange on an Amish-farm beat. This module adds the missing layer.

Engine: **local ONNX CLIP** (Qdrant ViT-B/32 vision+text via onnxruntime
CoreML/CPU; CLIP BPE via `tokenizers`) — no torch, no API. Semantic relevance =
how much more the frame matches the scene's EXPECTED subject prompts than a fixed
set of generic / off-subject distractor prompts (zero-shot softmax). Complementary
classical-CV signals (cv2/numpy): clarity, darkness/information, period-risk,
repetition. Everything is cached by content hash and fully defensive — any load
or inference failure degrades to "accept" so a render is never blocked.

Public API:
    available() -> bool
    score_asset(path, is_video, *, expected, objects=(), place="", period="",
                modern_risk=False, seen_hashes=None) -> dict
    accept(path, is_video, *, expected, ..., min_score=None) -> (ok, scores, reason)

Env flags:
    VIDLORE_VISUAL_RELEVANCE=1        enable (off => available() False)
    VIDLORE_VISUAL_RELEVANCE_MIN_SCORE  accept floor (default 0.42)
    VIDLORE_VISUAL_RELEVANCE_CACHE=1  persist embedding/score cache to disk
    VIDLORE_CLIP_DIR                  model dir (default ~/.cache/vidlore_clip)
    VIDLORE_CLIP_PROVIDER=auto|directml|cpu   inference EP (default auto; Windows
                                     auto = DirectML GPU when available, else CPU)
    VIDLORE_DML_DEVICE_ID=<n>         DirectML/DXGI adapter index (default: auto-
                                     detect the NVIDIA/RTX card, else 0)
    VIDLORE_VR_ACCELERATOR=auto|cpu|coreml   legacy macOS knob (still honoured)
"""
from __future__ import annotations

import hashlib
import math
import os
import platform
import re
import subprocess
import tempfile
import threading
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
_CLIP_DIR = Path(os.environ.get("VIDLORE_CLIP_DIR",
                                os.path.expanduser("~/.cache/vidlore_clip")))
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CTX = 77                      # CLIP context length
_LOGIT_SCALE = 100.0           # CLIP temperature (exp(0.07^-1) ≈ 100)

# Fixed distractor prompts — "what off-subject / generic / wrong footage looks
# like". Relevance = does the frame match the EXPECTED subject more than these?
_DISTRACTORS = (
    "a generic stock background", "an empty road and traffic",
    "a modern city street", "a plain office interior",
    "an abstract blurry texture", "a random unrelated landscape",
    "a generic nature b-roll clip", "a corporate business building",
)
_MODERN_PROMPTS = (
    "a modern car", "a highway with traffic", "a contemporary city skyline",
    "a smartphone", "modern industrial machinery", "a present-day street scene",
)
# HARD "wrong dominant concept" prompts (V1.3). If a candidate frame matches one
# of these MORE than the expected subject, the frame is ABOUT the wrong thing
# (a war/riot crowd on a 'cheap metals' beat; a preacher/portrait collage on an
# agriculture beat) — a hard reject regardless of the (possibly mid) relevance
# softmax. This is the gate the war-footage / random-people / preacher-collage
# failures needed. Topic-specific negatives can be appended per scene.
_STRONG_NEG = (
    "war and soldiers", "a violent crowd or riot", "armed conflict or a battle",
    "military weapons and tanks", "an angry protest mob", "a political rally",
    "a religious preacher giving a lecture to a crowd",
    "a collage of people's portrait photos", "a crowd of random people posing",
    "a modern city street full of cars", "a corporate office meeting",
    "a sports stadium crowd",
)

# Dedicated PEOPLE / CROWD probe. The Haar face gate misses a DISTANT or B&W
# crowd (small, side-lit, low-contrast faces), and a faded archival war-crowd /
# soldier clip only weakly clears the generic distractor-dominance gate (its CLIP
# cosines are compressed). This set asks, specifically, "is this frame dominantly
# a group of PEOPLE?" — which a war crowd / marching soldiers / a procession match
# strongly but a field, forest, soil macro or copper strip do not. On a scene
# whose subject is NOT a person it is the precise signal that separates the
# residual 0:18 war-crowd / 1:30 soldier class from legitimate nature footage.
_PEOPLE_NEG = (
    "a large crowd of people", "a group of people standing together",
    "soldiers marching in uniform", "a procession of many people",
    "a black and white photograph of a crowd", "a gathering of people outdoors",
)

# Dedicated WAR / MILITARY probe. War / soldiers / weapons / battle footage is
# the single most damaging off-topic class (a Napoleonic reenactor on a gardening
# beat), and on faded archival footage the general distractor-dominance sits
# right on the gate. This narrow, always-on probe (gated only by `crowd_ok`, i.e.
# OFF for genuine war/history scenes) asks specifically "is this military / war
# footage?" — which soldiers / uniforms / tanks / battles match strongly and a
# garden / lab / office / city does not — so a faded soldier clip is caught on
# EVERY non-war niche with a tight margin. Niche-general by construction.
_WAR_NEG = (
    "soldiers in military uniform", "an army marching to war",
    "armed soldiers carrying weapons", "a historical battle or war scene",
    "military tanks and artillery", "men in old military uniforms",
    "a war reenactment with soldiers",
)

# Dedicated VEHICLE / GENERIC-AMERICANA probe. The residual leak (a vintage
# home-movie of two men posing by a 1960s car on an "Amish farmer / garden"
# beat) is period-appropriate (the video IS about the 1960s, so the period guard
# doesn't fire) and only loosely off-subject ("vintage people outdoors" weakly
# matches "farmer outdoors", so visual_relevance sat at ~0.23, above the floor).
# What makes it WRONG is the CAR + generic-snapshot framing — there is no copper,
# soil, garden, plant, or pest in it. This narrow, always-on probe asks "is this
# frame dominated by a car / vehicle / people posing by a car / a casual vintage
# snapshot?" — which a garden / soil-macro / copper-strip / plant frame does not
# match. Gated by `vehicle_ok` (OFF unless the scene is genuinely about cars /
# driving / roads / transport), so it fires on every garden / nature / science /
# product niche. Amish footage in particular should never contain a car.
_VEHICLE_NEG = (
    "a parked car or automobile", "people posing by a car",
    "an old vintage automobile", "a vehicle parked outdoors",
    "a casual snapshot of people standing by a car",
    "a family posing for a photo next to an automobile",
    "a vintage home movie of people and a car",
)

# Dedicated DESIGNED-GRAPHIC probe (the weak-keyword sweep gap, 2026-06-03). A
# documentary beat must be a real photograph / footage frame — NEVER an
# infographic, chart, diagram, logo, clip-art, cartoon, poster, screenshot,
# slide, or text sign. These leak in on scenes with weak/empty keywords (the
# scorer's expected subject is too vague to reject them, and an off-topic
# infographic or party-logo clip-art weakly matches a generic subject), and the
# war/crowd/vehicle/face probes don't fire on them. This probe is KEYWORD-
# INDEPENDENT: instead of comparing the frame to the (vague) expected subject,
# it asks "does this frame look more like a designed GRAPHIC than a real
# PHOTOGRAPH?" — `graphic_dom = max(graphic_sim) - max(realphoto_sim)`. A
# documentary photo / film still / candid scene scores this strongly NEGATIVE; an
# infographic / logo / text image scores it positive. It fires on EVERY niche and
# on both concrete and abstract (guard-only) beats — a chart is wrong footage for
# any documentary, whatever the keywords. Calibrated against the seven-niche
# sweep renders: party-logo clip-art 0.062, a "POLYSEXUAL" text image 0.046, a
# modern mortgage-rates infographic 0.056, a "Politics" text sign 0.037 — vs every
# good asset measured <= -0.007 (real archival photo -0.007, fal portraits/film
# stills -0.06, real soldiers/field clips <= 0.02). Aggregated as the MEAN over
# sampled frames (NOT max): a real clip with one incidentally-flat frame is not a
# graphic, but a true infographic/logo is graphic in every frame.
_GRAPHIC_NEG = (
    "an infographic with charts and statistics", "a bar chart or pie chart",
    "a diagram with labels and arrows", "a logo or emblem", "clip art graphics",
    "a cartoon or digital illustration", "a poster with large title text",
    "a screenshot of a website or document", "a presentation slide with text",
    "a table of numbers and percentages", "a sign with printed words",
    "a political party logo", "a flat vector graphic", "a meme with text",
    # RC5.1 — game / software UI prompts. A strategy-game / missile-command /
    # tactical HUD / software-dashboard frame is a DESIGNED graphic (panels,
    # buttons, on-screen readouts), not footage — so it must score graphic_dom
    # positive. Phrased as INTERFACE chrome ("user interface", "control panel",
    # "on-screen buttons") rather than "a map", so a real cartographic frame or an
    # engine-rendered map-animation card (clean, no UI chrome) is NOT pulled in.
    "a screenshot of a strategy video game", "a video game user interface",
    "a video game heads-up display with on-screen icons",
    "a software dashboard with panels and buttons",
    "a control panel with gauges and readouts",
    "a tactical map interface from a war simulation game",
)
_REALPHOTO_POS = (
    "a realistic photograph", "a cinematic film still",
    "a documentary photograph of a real scene", "a candid photo of a real place",
    "a color photograph of people", "a natural outdoor photograph",
)
# graphic_dom above this ⇒ the frame is a designed graphic, not footage. Tuned to
# clear every good asset (worst good = real archival at -0.007) with margin while
# catching the sweep's graphic leaks (lowest = a text sign at 0.037). Env-tunable.
_DEFAULT_GRAPHIC_MAX = 0.036

# Calibrated against real failure vs good frames (CLIP cosines are compressed,
# so the zero-shot softmax sits in ~0.10–0.40, not 0–1). Floor 0.10 is an
# EGREGIOUS-off-subject net that clears every good frame measured (lowest good
# = a blurry copper macro at 0.10); the named failures are caught by the
# stronger, orthogonal signals (period-risk for modern-on-historical, clarity
# for the dim shot, face-mismatch for the archival group) so the relevance
# floor never has to over-reach and false-reject good footage.
_DEFAULT_MIN = 0.10

# ── HARD-REJECT metadata classifier (RC5) ─────────────────────────────────────
# A cheap, pixel-free pre-filter that flags an asset as junk-for-documentary from
# its SOURCE METADATA — filename, URL slug, page title, snippet, and the search
# query that found it. This is the first line of defence the three RC5 leaks (an
# anime/video-game DVD cover, a strategy-game UI screenshot, a generic
# multilingual sign) needed: every one carried a give-away token (game / cover /
# anime / poster / UI / sign-shaped query) that this catches before download.
#
# IMPORTANT (per the RC5 brief): keyword evidence is NECESSARY but, on its own,
# NOT sufficient to keep a borderline asset — the caller combines this verdict
# with the CLIP graphic_dom + text-density signal. But a HARD junk token (game,
# anime, dvd, ui, poster, screenshot, infographic, logo, meme, wallpaper, console
# brand, …) IS sufficient to REJECT unless the narration explicitly names that
# title (e.g. a documentary that is literally about "Bomberman" or "Photoshop").
#
# Returns (is_junk: bool, reason: str, hits: list[str]).
_JUNK_TOKENS = {
    # game / interactive
    "game", "gameplay", "videogame", "video game", "gaming", "simulation",
    "simulator", "emulator", "rom", "console", "playstation", "ps2", "ps3",
    "ps4", "ps5", "nintendo", "xbox", "steam", "hud", "speedrun", "mod",
    "walkthrough", "boss fight", "level up",
    # RC5.1 — strategy / grand-strategy / simulation-game UI. The missed game UI
    # (a Hearts-of-Iron / Paradox-style tactical map + missile-command HUD on an
    # Iran-Iraq war beat) reads as a "map" to CLIP, so its give-away is the TITLE
    # / engine name + the in-game / control-panel framing. These catch it from
    # metadata before download (and are NOT in the narration of a real war doc).
    "strategy game", "grand strategy", "hearts of iron", "paradox",
    "paradox interactive", "europa universalis", "crusader kings",
    "civilization", "total war", "wargame", "war game", "missile command",
    "command and conquer", "real time strategy", "rts", "turn based strategy",
    "4x game", "in-game", "in game", "tactical map", "tactical game",
    "control panel", "minimap", "mini map", "tech tree", "skill tree",
    "game ui", "game hud", "game map", "game screenshot",
    # animation / illustration
    "anime", "manga", "cartoon", "comic", "chibi", "fanart", "fan art",
    "render", "3d render", "digital art", "vector art", "clip art", "clipart",
    "illustration",
    # packaging / product / merch
    "dvd", "blu-ray", "bluray", "boxart", "box art", "cover art", "album cover",
    "dvd cover", "book cover", "poster", "merchandise", "merch", "tshirt",
    "t-shirt", "mug", "sticker", "keychain", "figurine", "action figure",
    "advertisement", "advert", "wallpaper", "desktop wallpaper",
    # UI / screenshots / graphics
    "ui", "user interface", "interface", "screenshot", "screen shot",
    "screen grab", "screengrab", "app screen", "dashboard ui", "infographic",
    "info graphic", "template", "powerpoint", "slide template", "logo",
    "emblem", "icon", "favicon", "meme", "stock chart", "chart", "diagram",
    "flowchart", "tutorial", "how to", "how-to",
    # RC5.1 — software dashboard / control-panel / generic interface chrome.
    "dashboard", "control panel", "admin panel", "software ui", "app ui",
    "web app", "settings menu", "toolbar", "gui",
    # Holiday / greeting-card / postcard / scrapbook ephemera. This is real
    # FOOTAGE (a filmed wall of vintage Christmas cards + a reindeer-sleigh
    # ornament), so the pixel graphic_dom probe never flags it — yet it is almost
    # never on-topic for a serious documentary. It leaked onto a water-plant beat
    # via the abstract word "gifts" (VO: "diplomatic gifts"), pulling a greeting-
    # card montage with an ugly grey wall. Reject on metadata unless the narration
    # itself is about cards / the holiday / philately (the _JUNK_AS_PHRASES +
    # narration-override below keeps a genuine Christmas/stamp doc safe).
    "greeting card", "greeting cards", "christmas card", "christmas cards",
    "holiday card", "holiday cards", "postcard", "postcards", "christmas",
    "xmas", "santa claus", "reindeer", "sleigh", "nativity", "scrapbook",
    "stamp album", "stamp collection", "philately", "greeting-card",
}
# Tokens that, when the SAME token also appears in the narration/subject, mean the
# asset is legitimately on-topic (a doc literally about that game/software/anime)
# — so we must NOT hard-reject on the keyword alone. The caller passes narration.
_JUNK_AS_PHRASES = (
    "video game", "clip art", "fan art", "box art", "cover art", "album cover",
    "dvd cover", "book cover", "info graphic", "stock chart", "how to",
    "user interface", "screen shot", "screen grab", "app screen",
    "desktop wallpaper", "action figure", "slide template", "boss fight",
    "level up", "t-shirt",
)


def _norm_text(*parts) -> str:
    try:
        return re.sub(r"[\W_]+", " ", " ".join(
            str(p or "") for p in parts)).lower().strip()
    except Exception:                                          # noqa: BLE001
        return ""


def classify_junk_metadata(*, title="", slug="", url="", query="", snippet="",
                           narration="", provider="") -> tuple:
    """Keyword-level hard-reject pre-filter for a candidate asset (RC5).

    Inspects all available source metadata + the search query and returns
    (is_junk, reason, hits). `is_junk` is True when a designed-graphic / game /
    anime / UI / poster / merch token appears AND that token is NOT also present
    in the narration/subject (which would make the asset genuinely on-topic).

    Pure string logic, never raises, no network/CLIP. The caller is expected to
    COMBINE this with the pixel-level graphic_dom + text-density gate — a clean
    (non-junk) verdict here does NOT by itself accept an asset."""
    try:
        meta = _norm_text(title, slug, url, query, snippet, provider)
        narr = _norm_text(narration)
        if not meta:
            return False, "", []
        meta_padded = f" {meta} "
        narr_padded = f" {narr} "
        hits = []
        for tok in _JUNK_TOKENS:
            # phrase tokens (with a space) match as substrings; single words
            # match on word boundaries so 'game' doesn't fire inside 'gameel'.
            if " " in tok or "-" in tok:
                present = tok.replace("-", " ") in meta_padded
            else:
                present = f" {tok} " in meta_padded
            if not present:
                continue
            # ON-TOPIC EXEMPTION: the narration explicitly names this token (a
            # documentary about that very game/software/anime/poster) → keep.
            tnorm = tok.replace("-", " ")
            if (" " in tnorm and tnorm in narr_padded) or \
                    (" " not in tnorm and f" {tnorm} " in narr_padded):
                continue
            hits.append(tok)
        if hits:
            return True, "junk-metadata:" + ",".join(sorted(set(hits))[:6]), \
                sorted(set(hits))
        return False, "", []
    except Exception:                                          # noqa: BLE001
        return False, "", []


def graphic_signal(path, is_video=False) -> dict:
    """Pixel-level designed-graphic probe for ONE asset (RC5 web-image gate).

    Runs the CLIP graphic_dom + text-density scorer on a downloaded image/clip
    and returns {graphic_dom, graphic_dom_base, ui_geom, looks_designed,
    looks_ui_screenshot, engine}. `looks_designed` is True when graphic_dom
    exceeds the env-tunable VIDLORE_VR_GRAPHIC_MAX (the same keyword-independent
    gate that rejects an infographic/chart/logo/cartoon/poster/screenshot/
    text-sign) OR when the UI-geometry signal (RC5.1) marks the frame a game/
    software interface screenshot (`looks_ui_screenshot`). Defensive: returns
    engine='skipped' (NOT designed) when the scorer is unavailable, so the
    caller's metadata + the VR post-pass remain the backstops — it never
    fabricates a 'designed' verdict."""
    try:
        if not available():
            return {"graphic_dom": None, "looks_designed": False,
                    "engine": "skipped"}
        s = score_asset(path, bool(is_video), expected="a documentary scene")
        if s.get("engine") not in ("clip-onnx", "cache"):
            return {"graphic_dom": s.get("graphic_dom"),
                    "looks_designed": False, "engine": s.get("engine")}
        try:
            gmax = float(os.environ.get("VIDLORE_VR_GRAPHIC_MAX",
                                        _DEFAULT_GRAPHIC_MAX))
        except (TypeError, ValueError):
            gmax = _DEFAULT_GRAPHIC_MAX
        gd = s.get("graphic_dom", -9)
        ui_geom = s.get("ui_geom", 0.0)
        # RC5.1 — a strongly UI-geometry frame (game HUD / dashboard / control
        # panel) is a designed INTERFACE even if its (already UI-bumped) graphic_dom
        # sits a hair under the gate, so surface a hard ui-screenshot verdict that
        # the post-render sweep turns into a loud game-UI flag. Env-tunable floor.
        try:
            ui_hard = float(os.environ.get("VIDLORE_VR_UI_GEOM_HARD", "0.62"))
        except (TypeError, ValueError):
            ui_hard = 0.62
        looks_ui = bool((ui_geom or 0.0) >= ui_hard)
        return {"graphic_dom": gd, "graphic_dom_base": s.get("graphic_dom_base"),
                "ui_geom": ui_geom, "looks_ui_screenshot": looks_ui,
                "looks_designed": bool(gd > gmax) or looks_ui,
                "engine": s.get("engine")}
    except Exception:                                          # noqa: BLE001
        return {"graphic_dom": None, "looks_designed": False, "engine": "error"}


# ── P1.6 — footage SPECIFICITY signal ────────────────────────────────────────
# A pure, deterministic re-ranking signal (no CLIP, no I/O). It does NOT accept
# or reject anything — `accept()` already decided that. It only produces a small
# additive bonus/penalty so the footage selector in footage.py can prefer the
# MORE-SPECIFIC of two already-ACCEPTED candidates (the water-hyacinth render
# leaned on generic "dark water" / "abstract mood" footage instead of the named,
# concrete subject — a lab, a harvester, a hearing, a scientist). Conservative:
# the magnitude is tiny (±~0.12) and capped so it can only re-order, never
# overturn a relevance verdict.
_SPECIFIC_CONCRETE = set(
    """laboratory lab harvester harvest factory machine machinery engine pump
    digester reactor turbine generator scientist researcher technician engineer
    microscope sample specimen document report ledger map chart blueprint
    diagram hearing courtroom committee senate parliament podium witness
    testimony signature stamp seal dam reservoir canal pipeline pipe valve
    boat barge vessel crane excavator tractor crop field root stem leaf flower
    bloom petal pollen seed plant weed vine wetland marsh lake river pond
    shoreline bank fisherman farmer worker villager laborer operator official
    skeleton fossil artifact ruin temple monument statue archive newspaper
    headline telegram letter manuscript instrument device apparatus tank
    container barrel vat beaker flask gauge dial meter""".split()
)
# Generic mood / atmospheric tokens — present a DOWN-RANK (never a reject) when a
# concrete scene grabbed this kind of filler. Mirrors footage_strength's filler
# lexicon but is the SPECIFICITY view of it.
_GENERIC_MOOD = set(
    """abstract mood moody atmosphere atmospheric ambient ambience bokeh blur
    blurry texture smoke haze fog mist particles glow shimmer ripple flowing
    fabric silk cloth gradient pattern aesthetic dreamy ethereal soft dark
    darkness shadow shadows void abstraction motion-blur defocus light-leak
    backdrop wallpaper minimal minimalist generic""".split()
)
# Archival / period markers — REWARD on a historical beat.
_ARCHIVAL_MARK = set(
    """archival archive vintage retro antique historic historical old aged
    sepia monochrome black-and-white grainy newsreel footage-archive
    nineteenth twentieth century 1800s 1900s 1920s 1930s 1940s wartime
    period-correct""".split()
)
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_SPEC_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")


def specificity_signal(text: str, *, person_beat: bool = False,
                       historical_beat: bool = False) -> float:
    """Additive SPECIFICITY bonus/penalty in roughly [-0.12, +0.12] for an
    asset, derived from its descriptive `text` (slug / caption / query / kind).

    REWARD (+): named entities / proper nouns, concrete documentary nouns
    (laboratory, harvester, hearing, scientist, document, map, …), human
    presence on a people-appropriate beat, archival markers on a historical
    beat. DOWN-RANK (−): generic mood/atmospheric footage (dark water, abstract,
    smoke, bokeh, texture, flowing fabric, mood) on a concrete scene.

    This is a pure RE-RANK signal: it is read AFTER accept() has already passed
    the candidate, so it never accepts a rejected asset and never changes the
    fail-closed floor. Deterministic, no model, never raises."""
    try:
        raw = (text or "").strip()
        if not raw:
            return 0.0
        low = raw.lower()
        toks = set(_SPEC_TOKEN_RE.findall(low))
        bonus = 0.0
        # proper nouns / named entities (use ORIGINAL case, ignore the first
        # word so a sentence-initial cap isn't mistaken for a name).
        propers = _PROPER_RE.findall(raw)
        if len(propers) >= 2 or any(len(p) >= 4 for p in propers[1:]):
            bonus += 0.05
        # concrete documentary nouns
        n_concrete = len(toks & _SPECIFIC_CONCRETE)
        if n_concrete >= 1:
            bonus += min(0.07, 0.04 + 0.015 * n_concrete)
        # archival markers on a historical beat
        if historical_beat and (toks & _ARCHIVAL_MARK):
            bonus += 0.04
        # human presence on a people-appropriate beat
        if person_beat and (toks & {
                "person", "people", "man", "woman", "men", "women", "worker",
                "scientist", "farmer", "villager", "child", "family", "crowd",
                "portrait", "face", "official", "witness", "researcher"}):
            bonus += 0.03
        # generic mood / atmospheric DOWN-RANK (only meaningful when the asset
        # text is dominated by filler and offers no concrete subject).
        n_mood = len(toks & _GENERIC_MOOD)
        if n_mood >= 1 and n_concrete == 0:
            bonus -= min(0.12, 0.05 + 0.025 * n_mood)
        # clamp to the small re-rank band.
        return round(max(-0.12, min(0.12, bonus)), 4)
    except Exception:                                          # noqa: BLE001
        return 0.0


def startup_log(prefix="  [visual-relevance]") -> str:
    """One-line gate-status line for render startup (RC5 observability).

    Returns (and prints) whether the fail-closed gate is ACTIVE (enabled + model
    loaded) or why it is not, so an operator can see at a glance that the gate is
    running. Never raises."""
    try:
        en = _enabled()
        if not en:
            msg = (f"{prefix} gate DISABLED (VIDLORE_VISUAL_RELEVANCE=0) — "
                   f"fail-OPEN; assets pass on metadata only")
        elif _try_load():
            msg = (f"{prefix} gate ACTIVE (fail-closed) — CLIP model loaded "
                   f"[{_CLIP_DIR}]")
        else:
            msg = (f"{prefix} gate ENABLED but model UNAVAILABLE at {_CLIP_DIR} "
                   f"— metadata hard-reject still applies; pixel gate inactive")
        print(msg, flush=True)
        return msg
    except Exception:                                          # noqa: BLE001
        return ""


# ── lazy model state ─────────────────────────────────────────────────────────
_vis_sess = None
_txt_sess = None
_tok = None
_vis_in = _vis_out = None
_txt_in = _txt_out = None
_load_tried = False
_load_ok = False
# ── CoreML/ONNX safe-fallback state (V3.2.2) ──────────────────────────────
_vr_backend = "uninit"          # active execution provider tier
_vr_retries = 0                 # cumulative inference retries this process
_vr_degraded = False            # True once we fell back / gave up
_vr_fallback_reason = ""        # human-readable last degradation cause
_text_emb_cache: dict = {}      # prompt -> np.ndarray (unit)
_asset_cache: dict = {}         # content-hash -> scores dict

# ── DirectML (Windows RTX) state ───────────────────────────────────────────
# DirectML sessions are NOT safe for concurrent Run() calls, so every inference
# on a DML-backed session is serialised through this lock (no-op cost on the
# CPU/CoreML path, which leaves _vr_is_dml False). Selecting/validating DirectML
# never disables the relevance gate — the worst case is a safe CPU fallback.
_vr_is_dml = False              # True only while the active EP is DirectML
_vr_device = None               # DXGI adapter index in use (DML), else None
_vr_gpu_name = ""               # best-effort adapter name (logging only)
_dml_lock = threading.Lock()    # serialise Run() on the shared DML session


def _enumerate_video_adapters() -> list:
    """Best-effort list of GPU adapter names (Windows only) for logging + an
    auto heuristic to find the NVIDIA card. Returns [] off-Windows or on any
    failure. NOTE: Win32_VideoController order is not guaranteed to equal the
    DXGI adapter index DirectML uses — treat as a HINT and validate by run."""
    if platform.system() != "Windows":
        return []
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=12)
        if r.returncode == 0:
            return [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:                                          # noqa: BLE001
        pass
    return []


def _clip_log(provider: str, **kw) -> None:
    extra = " ".join(f"{k}={v}" for k, v in kw.items() if v not in (None, ""))
    print(f"  [visual-relevance] provider={provider}"
          + (f" {extra}" if extra else ""), flush=True)


def _pick_clip_providers(ort):
    """Decide the ONNX execution-provider list for the CLIP sessions per the
    Windows RTX provider policy. Returns
        (providers, session_opts_mutator_or_None, label, device_id, gpu_name).

    Policy (NEVER disables the gate — worst case is CPU):
      VIDLORE_CLIP_PROVIDER = auto | directml | cpu     (default auto)
      VIDLORE_DML_DEVICE_ID = <dxgi index>              (default: auto-detect NVIDIA, else 0)

      cpu                            -> CPUExecutionProvider
      directml (forced)              -> DirectML if available, else CPU (logged)
      auto + Windows + DML available -> DirectML (validated at build), else CPU
      auto + macOS                   -> CoreML->CPU (UNCHANGED legacy behaviour)
      auto + Linux/other             -> CPU (unchanged)
    The actual DirectML build is validated in _try_load(); if it fails to
    initialise we rebuild on CPU there — this only expresses the preference."""
    provs = ort.get_available_providers()
    pref = os.environ.get("VIDLORE_CLIP_PROVIDER", "auto").strip().lower()
    acc = os.environ.get("VIDLORE_VR_ACCELERATOR", "auto").strip().lower()
    is_win = platform.system() == "Windows"
    dml_ok = "DmlExecutionProvider" in provs

    # explicit CPU (either knob) always wins
    if pref == "cpu" or acc == "cpu":
        _clip_log("CPUExecutionProvider", reason="forced_cpu")
        return ["CPUExecutionProvider"], None, "CPUExecutionProvider", None, ""

    want_dml = pref in ("directml", "dml") or (pref == "auto" and is_win)
    if want_dml:
        if not dml_ok:
            reason = "dml_unavailable" if pref != "auto" else "dml_unavailable_or_failed"
            _clip_log("CPUExecutionProvider", reason=reason)
            return ["CPUExecutionProvider"], None, "CPUExecutionProvider", None, ""
        adapters = _enumerate_video_adapters()
        dev_env = os.environ.get("VIDLORE_DML_DEVICE_ID", "").strip()
        gpu_name = ""
        if dev_env.lstrip("-").isdigit():
            device_id = max(0, int(dev_env))
            if 0 <= device_id < len(adapters):
                gpu_name = adapters[device_id]
        else:
            device_id = 0                       # never ASSUME adapter 0 is the RTX
            for i, nm in enumerate(adapters):
                if any(k in nm.lower() for k in
                       ("nvidia", "geforce", "rtx", "quadro")):
                    device_id, gpu_name = i, nm
                    break
            if not gpu_name and adapters and device_id < len(adapters):
                gpu_name = adapters[device_id]

        def _mut(so):
            # DirectML requirements: disable mem-pattern + sequential execution.
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = [("DmlExecutionProvider", {"device_id": device_id}),
                     "CPUExecutionProvider"]
        _clip_log("DmlExecutionProvider", device=device_id,
                  gpu=gpu_name or "unknown")
        return providers, _mut, "DmlExecutionProvider", device_id, gpu_name

    # non-Windows auto (or coreml): preserve legacy CoreML->CPU behaviour
    if acc == "coreml" or (acc == "auto" and "CoreMLExecutionProvider" in provs):
        use = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
               if p in provs] or ["CPUExecutionProvider"]
        return use, None, use[0], None, ""
    return ["CPUExecutionProvider"], None, "CPUExecutionProvider", None, ""


def _enabled() -> bool:
    # RC5 FAIL-CLOSED FIX: the gate is now ON BY DEFAULT. Previously this read
    # default "0" while footage.py read default "1" — the var is set nowhere, so
    # available() returned False and EVERY beat went unscored (the anime DVD
    # cover / game-UI / multilingual-sign leaks). Both sides now default "1".
    # VIDLORE_VISUAL_RELEVANCE=0 still disables it (the tests' opt-out).
    return os.environ.get("VIDLORE_VISUAL_RELEVANCE", "1").strip().lower() \
        in ("1", "true", "yes", "on")


def _try_load() -> bool:
    """Load ONNX sessions + tokenizer once. Returns False (never raises) if the
    model files are missing or onnxruntime/tokenizers unavailable."""
    global _vis_sess, _txt_sess, _tok, _vis_in, _vis_out, _txt_in, _txt_out
    global _load_tried, _load_ok, _vr_backend, _vr_is_dml, _vr_device, _vr_gpu_name
    if _load_tried:
        return _load_ok
    _load_tried = True
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        vis_p = _CLIP_DIR / "clip_vision.onnx"
        txt_p = _CLIP_DIR / "clip_text.onnx"
        tok_p = _CLIP_DIR / "tokenizer.json"
        if not (vis_p.exists() and txt_p.exists() and tok_p.exists()):
            return False

        # DETERMINISTIC MODE (VIDLORE_VR_DETERMINISTIC=1) — the QA regression
        # gate forces stable single-thread sequential CPU so a borderline CLIP
        # margin (~0.03 vs a 0.05 gate) doesn't flip run-to-run. This takes
        # precedence over ANY accelerator preference (CoreML / DirectML).
        _deterministic = os.environ.get("VIDLORE_VR_DETERMINISTIC", "") \
            in ("1", "true", "yes")

        def _build(providers, mutator):
            so = ort.SessionOptions()
            if _deterministic:
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            else:
                so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 2)
            if mutator:
                mutator(so)
            v = ort.InferenceSession(str(vis_p), sess_options=so, providers=providers)
            t = ort.InferenceSession(str(txt_p), sess_options=so, providers=providers)
            return v, t

        if _deterministic:
            _clip_log("CPUExecutionProvider", reason="deterministic_mode")
            providers, mutator, label, dev, gpu = \
                ["CPUExecutionProvider"], None, "CPUExecutionProvider", None, ""
        else:
            # VIDLORE_CLIP_PROVIDER=auto|directml|cpu — Windows auto picks the
            # DirectML GPU when available; macOS auto keeps CoreML->CPU; the gate
            # is never disabled (worst case CPU). Logged inside the helper.
            providers, mutator, label, dev, gpu = _pick_clip_providers(ort)

        try:
            _vis_sess, _txt_sess = _build(providers, mutator)
            _vr_is_dml = (label == "DmlExecutionProvider")
            _active = (_vis_sess.get_providers() or [label])[0]
            if _vr_is_dml and "Dml" not in _active:
                # Asked for DirectML but ORT silently used something else — treat
                # as a failed accelerator init and drop to CPU below.
                raise RuntimeError(f"DirectML inactive (active={_active})")
        except Exception as _accel_err:                        # noqa: BLE001
            if label == "CPUExecutionProvider":
                raise
            # Accelerator (DirectML) failed to initialise → SAFE CPU rebuild.
            # The relevance gate is NEVER disabled; only the EP changes.
            _clip_log("CPUExecutionProvider", reason="accel_init_failed",
                      detail=type(_accel_err).__name__)
            _vis_sess, _txt_sess = _build(["CPUExecutionProvider"], None)
            _vr_is_dml = False
            label, dev, gpu = "CPUExecutionProvider(accel_fallback)", None, ""

        _vr_device, _vr_gpu_name = dev, gpu
        try:
            _vr_backend = label if "fallback" in label \
                else (_vis_sess.get_providers() or [label])[0]
        except Exception:                                      # noqa: BLE001
            _vr_backend = label or "unknown"
        _vis_in = _vis_sess.get_inputs()[0].name
        _vis_out = _vis_sess.get_outputs()[0].name
        _txt_in = [i.name for i in _txt_sess.get_inputs()]
        _txt_out = _txt_sess.get_outputs()[0].name
        _tok = Tokenizer.from_file(str(tok_p))
        try:
            _tok.enable_truncation(max_length=_CTX)
            _tok.enable_padding(length=_CTX, pad_id=0, pad_token="<|endoftext|>")
        except Exception:                                      # noqa: BLE001
            pass
        _load_ok = True
        return True
    except Exception:                                          # noqa: BLE001
        _load_ok = False
        return False


def _reload_cpu_only() -> bool:
    """Rebuild both ONNX sessions on the CPU EP (V3.2.2) after a CoreML/ONNX —
    or DirectML (Windows RTX pass) — context crash. Never raises; returns True
    on success. Clears the DirectML flag so inference stops taking the DML lock."""
    global _vis_sess, _txt_sess, _vis_in, _vis_out, _txt_in, _txt_out, _vr_backend
    global _vr_is_dml
    try:
        import onnxruntime as ort
        vis_p = _CLIP_DIR / "clip_vision.onnx"
        txt_p = _CLIP_DIR / "clip_text.onnx"
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 2)
        _vis_sess = ort.InferenceSession(str(vis_p), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        _txt_sess = ort.InferenceSession(str(txt_p), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        _vis_in = _vis_sess.get_inputs()[0].name
        _vis_out = _vis_sess.get_outputs()[0].name
        _txt_in = [i.name for i in _txt_sess.get_inputs()]
        _txt_out = _txt_sess.get_outputs()[0].name
        _vr_backend = "CPUExecutionProvider(fallback)"
        _vr_is_dml = False
        return True
    except Exception:                                          # noqa: BLE001
        return False


def _session_run(s, outs, feed):
    """sess.run, SERIALISED when the active EP is DirectML — DML inference
    sessions are not safe for concurrent Run() calls. Zero lock cost on the
    CPU / CoreML path (the common case)."""
    if _vr_is_dml:
        with _dml_lock:
            return s.run(outs, feed)
    return s.run(outs, feed)


def _vr_run(which, sess, outs, feed):
    """Run an ONNX inference with a CoreML→CPU safe fallback (V3.2.2).
    On a model-context crash: retry; if it persists and CPU fallback is enabled
    (`VIDLORE_VR_CPU_FALLBACK=1`, default on), rebuild the sessions on CPU and
    retry up to `VIDLORE_VR_MAX_RETRIES` (default 2). On total failure RAISES so
    the caller degrades CONSERVATIVELY (unscored — never a fake relevance pass).
    `which`/`sess` let the retry re-fetch the freshly-reloaded global session."""
    global _vr_retries, _vr_degraded, _vr_fallback_reason
    try:
        return _session_run(sess, outs, feed)
    except Exception as e:                                      # noqa: BLE001
        cpu_fb = os.environ.get("VIDLORE_VR_CPU_FALLBACK", "1").strip().lower() \
            in ("1", "true", "yes", "on")
        try:
            mx = int(os.environ.get("VIDLORE_VR_MAX_RETRIES", "2"))
        except ValueError:
            mx = 2
        _vr_fallback_reason = f"{type(e).__name__}: {str(e)[:80]}"
        already_cpu = "CPU" in (_vr_backend or "") and "fallback" in (_vr_backend or "")
        for attempt in range(max(1, mx)):
            _vr_retries += 1
            if cpu_fb and not already_cpu:
                if _reload_cpu_only():
                    already_cpu = True
                    sess = _vis_sess if which == "vis" else _txt_sess
                    outs = [_vis_out] if which == "vis" else [_txt_out]
            try:
                r = _session_run(sess or (_vis_sess if which == "vis" else _txt_sess),
                                 outs, feed)
                if attempt > 0 or already_cpu:
                    _vr_degraded = True
                    print(f"  [visual-relevance] ⚠ CoreML/ONNX context failure → "
                          f"recovered on {_vr_backend} (retries={_vr_retries}, "
                          f"reason={_vr_fallback_reason})", flush=True)
                return r
            except Exception:                                  # noqa: BLE001
                continue
        _vr_degraded = True
        print(f"  [visual-relevance] ⚠ inference UNAVAILABLE after {_vr_retries} "
              f"retries ({_vr_fallback_reason}); degrading to UNSCORED (NOT a "
              f"relevance pass) — other backstops decide", flush=True)
        raise


def vr_status() -> dict:
    """Active inference backend + degradation telemetry (V3.2.2; Windows RTX
    pass adds DirectML device fields)."""
    return {"backend": _vr_backend, "retries": _vr_retries,
            "degraded": _vr_degraded, "fallback_reason": _vr_fallback_reason,
            "is_directml": _vr_is_dml, "device_id": _vr_device,
            "gpu_name": _vr_gpu_name}


def available() -> bool:
    """True only when enabled AND the local model actually loaded."""
    return _enabled() and _try_load()


_IDENT_DIM = 0


def model_identity() -> str:
    """Stable identity of the ACTIVE image-embedding model, for embedding-manifest
    validation: vision-onnx file name + size + mtime + output dimension. Any model swap,
    re-export, or re-download changes it; a persisted embedding row may only be trusted
    when the manifest's identity equals this string. '' when the model is unavailable."""
    global _IDENT_DIM
    try:
        if not available():
            return ""
        p = _CLIP_DIR / "clip_vision.onnx"
        st = p.stat()
        if not _IDENT_DIM:
            import numpy as _np_mi
            from PIL import Image as _Im_mi
            _IDENT_DIM = int(_np_mi.asarray(
                _img_embed(_Im_mi.new("RGB", (8, 8)))).shape[-1])
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}:{_IDENT_DIM}"
    except Exception:
        return ""


# ── embeddings ───────────────────────────────────────────────────────────────
def _perf_incr(name: str) -> None:
    """Decision-neutral call counter (never raises, never affects behavior)."""
    try:
        from vidlore.clipstudio import perf_metrics as _pm
        _pm.incr(name)
    except Exception:
        pass


def _img_embed(pil_img):
    import numpy as np
    _perf_incr("clip.img_embed")
    im = pil_img.convert("RGB")
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, int(w * s + 0.5)), max(224, int(h * s + 0.5))))
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    im = im.crop((l, t, l + 224, t + 224))
    a = np.asarray(im, dtype="float32") / 255.0
    a = (a - np.array(_CLIP_MEAN, "float32")) / np.array(_CLIP_STD, "float32")
    a = a.transpose(2, 0, 1)[None]                            # [1,3,224,224]
    out = _vr_run("vis", _vis_sess, [_vis_out], {_vis_in: a})[0][0]
    n = np.linalg.norm(out) + 1e-8
    return out / n


def _txt_embed(prompt: str):
    import numpy as np
    if prompt in _text_emb_cache:
        _perf_incr("clip.txt_embed.cache_hit")
        return _text_emb_cache[prompt]
    _perf_incr("clip.txt_embed")
    enc = _tok.encode(prompt)
    ids = np.array([enc.ids[:_CTX] + [0] * max(0, _CTX - len(enc.ids))],
                   dtype="int64")
    feed = {_txt_in[0]: ids}
    if len(_txt_in) > 1:                                       # attention_mask
        feed[_txt_in[1]] = (ids != 0).astype("int64")
    out = _vr_run("txt", _txt_sess, [_txt_out], feed)[0][0]
    n = np.linalg.norm(out) + 1e-8
    v = out / n
    _text_emb_cache[prompt] = v
    return v


# ── classical-CV signals (cv2/numpy) ─────────────────────────────────────────
def _cv_signals(pil_img):
    """clarity (0..1 sharpness), darkness_info (0..1), phash (64-bit int)."""
    import numpy as np
    try:
        import cv2
        g = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(g, cv2.CV_64F).var()
        clarity = max(0.0, min(1.0, lap / 600.0))             # ~600 = crisp
    except Exception:                                          # noqa: BLE001
        g = np.asarray(pil_img.convert("L"), dtype="float32")
        clarity = max(0.0, min(1.0, float(g.std()) / 70.0))
    gf = np.asarray(pil_img.convert("L").resize((64, 64)), dtype="float32")
    mean = float(gf.mean())
    hist, _ = np.histogram(gf, bins=32, range=(0, 255))
    p = hist / (hist.sum() + 1e-8)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum()) / 5.0  # /log2(32)
    darkness_info = max(0.0, min(1.0, (mean / 90.0) * 0.5 + ent * 0.5))
    # perceptual hash (DCT-free average hash on 8x8)
    small = np.asarray(pil_img.convert("L").resize((8, 8)), dtype="float32")
    bits = (small > small.mean()).flatten()
    ph = 0
    for b in bits:
        ph = (ph << 1) | int(b)
    return clarity, darkness_info, ph


def _phash_dist(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


_face_cascade = None
_face_tried = False


def _face_frac(pil_img) -> float:
    """Fraction of frame area covered by the largest detected face (0..1).
    A strong face on a scene whose subject is NOT a person is a clear mismatch
    (e.g. an archival group photo on a 'two cheap metals' beat)."""
    global _face_cascade, _face_tried
    try:
        import cv2
        import numpy as np
        if not _face_tried:
            _face_tried = True
            xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _face_cascade = cv2.CascadeClassifier(xml)
        if _face_cascade is None or _face_cascade.empty():
            return 0.0
        g = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        H, W = g.shape[:2]
        faces = _face_cascade.detectMultiScale(g, 1.2, 5,
                                               minSize=(int(W * 0.06),
                                                        int(H * 0.06)))
        if len(faces) == 0:
            return 0.0
        area = max(w * h for (x, y, w, h) in faces)
        return min(1.0, area / float(W * H))
    except Exception:                                          # noqa: BLE001
        return 0.0


def _ui_geom_signal(pil_img) -> float:
    """UI-GEOMETRY score 0..1 (RC5.1) — how much a frame looks like a software /
    game INTERFACE screenshot rather than a photograph or an engine-rendered map.

    A UI screenshot (a strategy-game HUD, a missile-command console, a software
    dashboard, a control panel) has a tell-tale geometry that a real photo and a
    clean cartographic frame do NOT: long, pixel-straight HORIZONTAL and VERTICAL
    edges forming rectangular panels/bars, repeated at high frequency, packing the
    frame edges (toolbars, side panels, button rows, mini-maps). We measure this
    cheaply with cv2:

      * Canny edges → fraction of edge pixels that lie on AXIS-ALIGNED runs (a
        photo's edges are mostly diffuse/curved; a UI's are ruler-straight).
      * the count of LONG straight horizontal/vertical lines (panel/bar borders).
      * a border-density bias: UI chrome clusters around the frame perimeter.

    Returns ~0 for photos / footage / clean maps and rises toward 1 for a dense
    rectangular-panel interface. Conservative by construction (it needs BOTH a high
    axis-aligned-edge fraction AND several long ruled lines), so a real map (curved
    coastlines, few straight rules) and ordinary footage stay near 0. Never raises;
    returns 0.0 when cv2 is unavailable so it can only ever ADD evidence, not block.
    """
    try:
        import cv2
        import numpy as np
        g = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        H, W = g.shape[:2]
        if H < 24 or W < 24:
            return 0.0
        edges = cv2.Canny(g, 60, 160)
        total_edges = int((edges > 0).sum())
        if total_edges < (H * W) * 0.004:        # almost no structure → not UI
            return 0.0
        # (1) axis-aligned edge fraction: count edge pixels that sit on a run of
        # >=3 consecutive edge pixels along a row (horizontal) or column (vertical).
        e = edges > 0
        horiz = e[:, :-2] & e[:, 1:-1] & e[:, 2:]          # 3-in-a-row across cols
        vert = e[:-2, :] & e[1:-1, :] & e[2:, :]           # 3-in-a-row down rows
        axis_run_px = int(horiz.sum()) + int(vert.sum())
        axis_frac = axis_run_px / float(total_edges + 1)
        # (2) long straight lines (panel / bar / button borders). Probabilistic
        # Hough; keep only near-horizontal / near-vertical segments, length-gated.
        n_long = 0
        try:
            minlen = int(min(H, W) * 0.33)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80,
                                    minLineLength=max(40, minlen), maxLineGap=6)
            if lines is not None:
                for ln in lines[:400]:
                    x1, y1, x2, y2 = ln[0]
                    dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
                    if dy <= 2 or dx <= 2:        # axis-aligned only
                        n_long += 1
        except Exception:                                      # noqa: BLE001
            n_long = 0
        # (3) perimeter bias — UI chrome hugs the frame edge. Fraction of edge
        # pixels in the outer 14% border band vs the whole frame.
        by, bx = max(2, int(H * 0.14)), max(2, int(W * 0.14))
        border = np.zeros_like(e)
        border[:by, :] = True
        border[-by:, :] = True
        border[:, :bx] = True
        border[:, -bx:] = True
        border_edges = int((e & border).sum())
        border_frac = border_edges / float(total_edges + 1)
        # Combine. UI needs strongly axis-aligned edges AND several long ruled
        # lines; the perimeter bias is a gentle multiplier (chrome at the edges).
        # Calibrated so photos/maps (axis_frac ~0.2-0.45, n_long 0-3) stay < ~0.25
        # while a panelled interface (axis_frac > 0.55, n_long >= 6) clears 0.5.
        line_term = min(1.0, n_long / 14.0)
        axis_term = max(0.0, (axis_frac - 0.45) / 0.45)        # 0 below 0.45 → 1 at 0.90
        perim_term = min(1.0, max(0.0, (border_frac - 0.40) / 0.40))
        score = (0.55 * axis_term + 0.45 * line_term) * (0.75 + 0.25 * perim_term)
        return float(max(0.0, min(1.0, score)))
    except Exception:                                          # noqa: BLE001
        return 0.0


# ── frame sampling ───────────────────────────────────────────────────────────
def _probe_duration(path, ff) -> float:
    """Best-effort clip duration in seconds (0.0 if unknown). Parses the
    `Duration:` line from `ffmpeg -i` stderr (no ffprobe dependency). Never
    raises — a 0.0 makes the caller fall back to a wide fixed-time spread."""
    try:
        o = subprocess.run([ff, "-i", str(path)], capture_output=True,
                           text=True, timeout=15)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", o.stderr or "")
        if m:
            return (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                    + float(m.group(3)))
    except Exception:                                          # noqa: BLE001
        pass
    return 0.0


def _sample_frames(path: Path, is_video: bool, n: int = 3):
    """Sample representative frames. For a STILL, the one image. For a VIDEO,
    frames spread across the WHOLE usable clip — NOT only the first ~3 s.

    Root cause of the residual war-crowd / vintage-soldier leaks: a wrong-concept
    shot frequently sits in the MIDDLE or TAIL of a stock clip, but the old
    sampler only read 0.6/1.6/3.0 s, so the dominant wrong content was never
    seen and the max-over-frames distractor-dominance gate had nothing to fire
    on. We now probe the real duration and sample evenly (start / ~22% / mid /
    ~72% / end), with extra interior probes on longer clips, so a clip is
    rejected if ANY part of it is dominantly about the wrong concept."""
    from PIL import Image
    if not is_video:
        try:
            return [Image.open(path)]
        except Exception:                                      # noqa: BLE001
            return []
    try:
        from .ffmpeg_tool import ffmpeg_exe
        ff = ffmpeg_exe()
    except Exception:                                          # noqa: BLE001
        ff = "ffmpeg"
    dur = _probe_duration(path, ff)
    if dur and dur > 1.2:
        usable = max(0.4, dur - 0.25)                  # avoid a black tail frame
        if dur >= 12.0:
            fracs = (0.05, 0.17, 0.29, 0.41, 0.53, 0.65, 0.77, 0.90)  # 8 (long)
        elif dur >= 6.0:
            fracs = (0.06, 0.20, 0.34, 0.48, 0.62, 0.78, 0.92)        # 7 (med)
        elif dur >= 3.0:
            fracs = (0.08, 0.26, 0.46, 0.68, 0.90)                    # 5 (short)
        else:
            fracs = (0.12, 0.45, 0.80)                                # 3 (tiny)
        times = [round(min(usable, max(0.12, dur * f)), 2) for f in fracs]
    else:
        times = [0.4, 1.5, 3.0, 5.0, 7.0]            # unknown dur -> spread wide
    frames = []
    for t in times:
        fd, png = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            subprocess.run([ff, "-y", "-hide_banner", "-loglevel", "error",
                            "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                            "-vf", "scale=320:-1", png], check=True, timeout=20)
            from PIL import Image as _I
            frames.append(_I.open(png).copy())
        except Exception:                                      # noqa: BLE001
            pass
        finally:
            try:
                os.unlink(png)
            except Exception:                                  # noqa: BLE001
                pass
    return frames


def _content_key(path: Path, expected: str) -> str:
    try:
        st = path.stat()
        h = hashlib.sha1(f"{path.name}:{st.st_size}:{int(st.st_mtime)}:"
                         f"{expected}".encode()).hexdigest()
        return h
    except Exception:                                          # noqa: BLE001
        return hashlib.sha1(f"{path}:{expected}".encode()).hexdigest()


# ── P1.2 — within-video asset de-dup registry ────────────────────────────────
# A small, self-contained registry of the visual CONTENT actually USED in one
# render. The existing dedup (seen_links claim + the VR post-pass cross-scene
# phash log) covers FETCHED clips that get pixel-scored, but NOT engine-rendered
# LABEL/PHOTO cards, AI stills, or web images deduped by visual content. This
# closes that gap: a water-hyacinth render repeated an engine "WATER HYACINTH"
# label card 8+ times, the same dark-water clip 3×, a fountain 2×.
#
# Three signature kinds, all order-independent and cheap:
#   • file stem + content hash  — footage clips / AI stills / web images
#   • image-bytes content hash  — card background photos / stills
#   • label signature           — (graphic_kind, normalized text, bg-hash) for
#                                 engine-rendered text/label cards
# Plus an optional 64-bit perceptual hash for NEAR-duplicate detection.
#
# FAIL-SAFE by contract: the caller asks `should_use()` and, if it would leave
# nothing, keeps the asset and logs `kept-no-alternative`. The registry NEVER
# downloads, renders, decodes blindly, or raises — every public method is
# wrapped so a dedup decision can never cause a black frame, crash, or empty
# scene.
def phash_dist(a, b) -> int:
    """Public Hamming distance between two 64-bit perceptual hashes (None-safe;
    returns 64 = 'maximally different' when either is missing)."""
    try:
        if a is None or b is None:
            return 64
        return bin(int(a) ^ int(b)).count("1")
    except Exception:                                          # noqa: BLE001
        return 64


def _norm_card_text(text: str) -> str:
    """Normalize label/card text for an exact-repeat signature: lowercase,
    collapse whitespace, strip surrounding punctuation. 'WATER HYACINTH' and
    ' Water Hyacinth. ' collapse to the same key so the 2nd render is blocked."""
    t = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return re.sub(r"^[\W_]+|[\W_]+$", "", t)


def image_bytes_sig(path) -> str:
    """Content hash of an image/asset's bytes (first 1 MiB is plenty to separate
    distinct stock/AI/web assets while staying cheap). Falls back to a
    name+size+mtime signature if the bytes can't be read. Never raises."""
    try:
        p = Path(path)
        with open(p, "rb") as fh:
            return hashlib.sha1(fh.read(1 << 20)).hexdigest()
    except Exception:                                          # noqa: BLE001
        try:
            st = Path(path).stat()
            return hashlib.sha1(
                f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}".encode()
            ).hexdigest()
        except Exception:                                      # noqa: BLE001
            return hashlib.sha1(str(path).encode()).hexdigest()


def label_card_sig(kind: str, text: str, bg_hash: str = "") -> str:
    """Stable signature for an engine-rendered LABEL card:
    (graphic_kind + normalized lowercased text + optional background-image-hash).
    Two cards with the same kind+text collapse to one key (the HARD anti-repeat
    rule), while the same text over a DIFFERENT background photo stays distinct."""
    return hashlib.sha1(
        f"{(kind or '').strip().lower()}|{_norm_card_text(text)}|{bg_hash or ''}"
        .encode()).hexdigest()


def label_text_sig(text: str) -> str:
    """Background- AND kind-INDEPENDENT signature for a full-screen label card,
    keyed purely on the normalized visible text. Used (P2.1) to block the same
    on-screen label (e.g. 'WATER HYACINTH') from rendering as a full-screen card
    more than RenderDedup.LABEL_MAX times per video — even when the card kind or
    the background photo changes (the case label_card_sig's bg_hash missed)."""
    return hashlib.sha1(_norm_card_text(text).encode()).hexdigest()


class RenderDedup:
    """Within-video used-asset registry (one instance per render).

    Tracks an EXACT content signature and an optional perceptual hash for every
    visual actually used. Callers:
        rd = RenderDedup(out_dir)                # out_dir may be None
        ok, reason = rd.should_use(sig, scene=i, phash=ph, kind="footage")
        if ok: rd.commit(sig, phash=ph)          # mark as used
        rd.dump()                                # write render_dedup_log.json

    `should_use` returns (True, "") when the asset is new enough to use, or
    (False, reason) when it EXACTLY matches or is a NEAR-duplicate of a used
    asset — the caller then tries an alternative, and only if none exists keeps
    it and calls `note_kept()`. Label kinds enforce the HARD rule: a 2nd render
    of the same normalized text is always blocked."""

    NEAR_DUP_BITS = 10        # Hamming <=10 on 64-bit phash ⇒ near-duplicate
    LABEL_MAX = 1             # same full-screen label text renders at most N×/video

    def __init__(self, out_dir=None):
        self.out_dir = out_dir
        self._sigs: set[str] = set()
        self._phashes: list[int] = []
        self._labels: set[str] = set()       # label signatures (hard 1×/video)
        self._label_counts: dict[str, int] = {}   # text-sig → times rendered
        self.log: list[dict] = []

    def should_use(self, sig: str, *, scene=None, phash=None,
                   kind: str = "asset", is_label: bool = False):
        """Decide whether `sig` (an exact content signature) may be used.
        Returns (ok: bool, reason: str). Never raises — any error ⇒ allow."""
        try:
            if is_label:
                # HARD RULE: the same full-screen label (same normalized text)
                # renders at most LABEL_MAX times per video — INDEPENDENT of the
                # card kind AND the background photo. The 2nd+ is demoted so
                # assemble falls back to footage / a lower-third for that scene.
                if self._label_counts.get(sig, 0) >= self.LABEL_MAX:
                    self._add_log(scene, "label-repeat", kind, sig)
                    return False, "label-repeat"
                return True, ""
            if sig and sig in self._sigs:
                self._add_log(scene, "exact", kind, sig)
                return False, "exact"
            if phash is not None:
                for h in self._phashes:
                    if phash_dist(phash, h) <= self.NEAR_DUP_BITS:
                        self._add_log(scene, "near", kind, sig, phash=phash)
                        return False, "near"
            return True, ""
        except Exception:                                      # noqa: BLE001
            return True, ""

    def commit(self, sig: str, *, phash=None, is_label: bool = False) -> None:
        """Record an asset as USED so later scenes dedup against it."""
        try:
            if is_label:
                if sig:
                    self._label_counts[sig] = self._label_counts.get(sig, 0) + 1
                    self._labels.add(sig)
                return
            if sig:
                self._sigs.add(sig)
            if phash is not None:
                self._phashes.append(int(phash))
        except Exception:                                      # noqa: BLE001
            pass

    def note_kept(self, *, scene=None, kind: str = "asset",
                  reason: str = "kept-no-alternative") -> None:
        """Log that a rejected asset was KEPT because no alternative existed
        (fail-safe — never leave a scene empty)."""
        self._add_log(scene, reason, kind, sig=None)

    def _add_log(self, scene, reason, kind, sig, phash=None) -> None:
        try:
            print(f"[dedup] reject scene={scene} reason={reason} "
                  f"kind={kind}", flush=True)
        except Exception:                                      # noqa: BLE001
            pass
        try:
            self.log.append({
                "scene": scene, "reason": reason, "kind": kind,
                "sig": (sig[:16] if isinstance(sig, str) else sig),
                "phash": int(phash) if phash is not None else None})
        except Exception:                                      # noqa: BLE001
            pass

    def dump(self, out_dir=None) -> None:
        """Append the dedup decisions to render_dedup_log.json in the run output
        dir. Guarded with try/except: if the dir is unknown or unwritable it is
        silently skipped (never crash a render on logging)."""
        try:
            import json as _json
            d = out_dir if out_dir is not None else self.out_dir
            if d is None or not self.log:
                return
            p = Path(d) / "render_dedup_log.json"
            # MERGE with any prior write to the same dir (the footage pass and the
            # graphic-card pass each carry their own registry but share the run
            # dir), so one render_dedup_log.json holds every decision.
            events = list(self.log)
            try:
                if p.exists():
                    prev = _json.loads(p.read_text())
                    if isinstance(prev, dict) and isinstance(
                            prev.get("events"), list):
                        events = prev["events"] + events
            except Exception:                                  # noqa: BLE001
                events = list(self.log)
            payload = {
                "summary": {
                    "rejections": len(events),
                    "exact": sum(1 for e in events if e.get("reason") == "exact"),
                    "near": sum(1 for e in events if e.get("reason") == "near"),
                    "label_repeat": sum(1 for e in events
                                        if e.get("reason") == "label-repeat"),
                    "kept_no_alternative": sum(
                        1 for e in events
                        if e.get("reason") == "kept-no-alternative"),
                },
                "rule": "within-video asset de-dup: exact content-sig match or "
                        "phash Hamming<=10 is a near-duplicate; an engine label "
                        "card with the same normalized text renders at most once "
                        "per video; fail-safe keeps the asset (kept-no-alternative)"
                        " when rejecting would leave a scene empty.",
                "events": events,
            }
            p.write_text(_json.dumps(payload, indent=1, default=str))
        except Exception:                                      # noqa: BLE001
            pass


# ── main scorer ──────────────────────────────────────────────────────────────
def score_asset(path, is_video: bool, *, expected: str, objects=(), place: str = "",
                action: str = "", period: str = "", modern_risk: bool = False,
                negatives=(), seen_hashes=None) -> dict:
    """Score a candidate visual. Returns a dict with:
        visual_relevance (0..1, softmax pos vs distractors)
        pos_sim, distractor_sim (raw cosines)
        clarity, darkness_info (0..1)
        period_risk (0..1; >0 only when modern_risk)
        repetition (0..1; 1 = near-duplicate of a seen asset)
        phash, ok (None here — decided by accept())
    Defensive: returns a permissive {visual_relevance: 1.0, ...} on any failure."""
    import numpy as np
    blank = {"visual_relevance": 1.0, "pos_sim": 0.0, "distractor_sim": 0.0,
             "clarity": 1.0, "darkness_info": 1.0, "period_risk": 0.0,
             "repetition": 0.0, "phash": 0, "engine": "skipped"}
    if not available():
        return blank
    path = Path(path)
    ck = _content_key(path, expected)
    if ck in _asset_cache:
        d = dict(_asset_cache[ck])
        d["engine"] = "cache"
        return d
    try:
        frames = _sample_frames(path, is_video)
        if not frames:
            return blank
        # build prompt sets
        subj = (expected or "").strip() or "the subject"
        pos = [f"a photo of {subj}"]
        if objects:
            pos.append("a photo of " + ", ".join(list(objects)[:3]))
        if place:
            pos.append(f"a photo of {subj} at {place}")
        if action:
            pos.append(f"a photo of {action}")
        pos_e = [_txt_embed(p) for p in pos]
        dis_e = [_txt_embed(p) for p in _DISTRACTORS]
        mod_e = [_txt_embed(p) for p in _MODERN_PROMPTS] if modern_risk else []
        # strong "wrong dominant concept" set (+ any per-scene topic negatives)
        neg_e = [_txt_embed(p) for p in
                 (tuple(_STRONG_NEG) + tuple(negatives or ()))]
        ppl_e = [_txt_embed(p) for p in _PEOPLE_NEG]    # dedicated crowd probe
        war_e = [_txt_embed(p) for p in _WAR_NEG]        # dedicated war probe
        veh_e = [_txt_embed(p) for p in _VEHICLE_NEG]   # dedicated vehicle probe
        gfx_e = [_txt_embed(p) for p in _GRAPHIC_NEG]    # designed-graphic probe
        rph_e = [_txt_embed(p) for p in _REALPHOTO_POS]  # real-photo anchor

        rel_probs, pos_sims, dis_sims, clar, dki, phs, mod_sims, faces, doms, \
            ppl_doms, war_doms, veh_doms, gfx_doms, ui_geoms = (
                [], [], [], [], [], [], [], [], [], [], [], [], [], [])
        for fr in frames:
            ie = _img_embed(fr)
            ps = max(float(ie @ pe) for pe in pos_e)
            ds = max(float(ie @ de) for de in dis_e)
            # PEOPLE dominance: how much more this frame is "a crowd of people"
            # than the expected (non-person) subject. Catches the distant/B&W
            # crowd the Haar face gate can't see.
            pps = max(float(ie @ pe) for pe in ppl_e)
            ppl_doms.append(pps - ps)
            # WAR / MILITARY dominance — narrow always-on probe (see _WAR_NEG).
            wps = max(float(ie @ pe) for pe in war_e)
            war_doms.append(wps - ps)
            # VEHICLE / generic-Americana dominance (see _VEHICLE_NEG) — catches
            # the vintage people-by-a-car snapshot that is period-OK but
            # subject-wrong on a garden / copper / Amish beat.
            vps = max(float(ie @ pe) for pe in veh_e)
            veh_doms.append(vps - ps)
            # DESIGNED-GRAPHIC dominance (see _GRAPHIC_NEG) — keyword-INDEPENDENT:
            # how much more this frame looks like a graphic (infographic / chart /
            # logo / clip-art / poster / text image) than a REAL photograph. >0 ⇒
            # it's a designed graphic, not footage. Compared to the real-photo
            # anchor, NOT the expected subject, so it fires regardless of how weak
            # the scene's keywords are.
            gps = max(float(ie @ pe) for pe in gfx_e)
            rps = max(float(ie @ pe) for pe in rph_e)
            gfx_doms.append(gps - rps)
            # zero-shot softmax: P(pos) over [best pos, each distractor]
            logits = np.array([ps] + [float(ie @ de) for de in dis_e]) * \
                (_LOGIT_SCALE / 100.0 * 10.0)
            p = np.exp(logits - logits.max())
            rel_probs.append(float((p / p.sum())[0]))
            pos_sims.append(ps)
            dis_sims.append(ds)
            # distractor DOMINANCE: how much more this frame matches a known
            # WRONG concept (war/crowd/preacher/portrait-collage/…) than the
            # expected subject. >0 ⇒ the frame is ABOUT the wrong thing.
            ns = max(float(ie @ ne) for ne in neg_e)
            doms.append(ns - ps)
            if mod_e:
                mod_sims.append(max(float(ie @ me) for me in mod_e))
            c, di, ph = _cv_signals(fr)
            clar.append(c)
            dki.append(di)
            phs.append(ph)
            faces.append(_face_frac(fr))
            # UI-GEOMETRY (RC5.1) — does this frame have software/game-interface
            # geometry (dense axis-aligned rectangular panels / bars / button
            # rows)? Near-0 for photos, footage, and clean maps; high for a HUD /
            # dashboard / strategy-game UI screenshot. Used below to BUMP
            # graphic_dom so a UI frame is caught even when its CLIP graphic
            # cosine is borderline (a tactical-game map reads partly as "a map").
            ui_geoms.append(_ui_geom_signal(fr))
        # repetition vs seen
        rep = 0.0
        rep_ph = phs[len(phs) // 2]
        if seen_hashes:
            mind = min((_phash_dist(rep_ph, h) for h in seen_hashes), default=64)
            rep = max(0.0, 1.0 - mind / 12.0)                 # <12 bits ⇒ dup-ish
        period_risk = 0.0
        if mod_sims:
            # modern content scoring higher than the (historical) subject
            period_risk = max(0.0, min(1.0, (max(mod_sims) -
                                             float(np.mean(pos_sims))) * 6.0 + 0.3))
        # UI-GEOMETRY → graphic_dom BUMP (RC5.1). A strategy-game / missile-command
        # / dashboard HUD partly reads as "a map" to CLIP, so its semantic
        # graphic_dom can sit just under the gate. The pixel UI-geometry signal
        # (dense axis-aligned panels) is the orthogonal evidence that it is an
        # INTERFACE, not footage — so when UI geometry is strong we ADD a bump to
        # graphic_dom (scaled by how UI-like the frame is, above a dead-band that
        # photos/maps never reach). MAX over frames: a HUD is on-screen throughout,
        # so one strongly-UI frame is decisive. The bump only ever RAISES the score
        # (never lowers it), and the dead-band (>0.5) + cap keep real maps/footage
        # untouched. Env-tunable so it can be dialed or disabled without code edits.
        ui_geom = float(np.max(ui_geoms)) if ui_geoms else 0.0
        try:
            ui_floor = float(os.environ.get("VIDLORE_VR_UI_GEOM_MIN", "0.5"))
        except (TypeError, ValueError):
            ui_floor = 0.5
        try:
            ui_bump_max = float(os.environ.get("VIDLORE_VR_UI_GEOM_BUMP", "0.12"))
        except (TypeError, ValueError):
            ui_bump_max = 0.12
        graphic_dom_base = float(np.mean(gfx_doms))
        ui_bump = 0.0
        if ui_geom > ui_floor:
            # linear ramp from ui_floor..1.0 → 0..ui_bump_max
            ui_bump = ui_bump_max * min(1.0, (ui_geom - ui_floor) /
                                        max(1e-6, (1.0 - ui_floor)))
        graphic_dom = graphic_dom_base + ui_bump
        # Aggregate per-frame -> clip score with MAX (best frame): a clip is
        # relevant/clear/bright if ANY sampled frame is, so one weak frame never
        # false-rejects a good clip; a clip wrong in EVERY frame still scores low.
        out = {
            "visual_relevance": round(float(np.max(rel_probs)), 3),
            "pos_sim": round(float(np.max(pos_sims)), 3),
            "distractor_sim": round(float(np.mean(dis_sims)), 3),
            "margin": round(float(np.max(pos_sims) - np.mean(dis_sims)), 3),
            "clarity": round(float(np.max(clar)), 3),
            "darkness_info": round(float(np.max(dki)), 3),
            "face_frac": round(float(np.max(faces)) if faces else 0.0, 3),
            # MAX over frames (not mean): a clip that is strongly about a WRONG
            # concept in even one sampled frame (a war crowd / soldiers / a
            # preacher collage) is wrong footage — the mean diluted it below
            # the gate and let the war/soldier clips through in the live render.
            "distractor_dom": round(float(np.max(doms)), 3),
            "people_dom": round(float(np.max(ppl_doms)), 3),
            "war_dom": round(float(np.max(war_doms)), 3),
            "vehicle_dom": round(float(np.max(veh_doms)), 3),
            # MEAN over frames (not max): a true graphic is graphic in every
            # frame, while a real clip with one incidentally-flat frame is not —
            # the mean keeps good footage safe while still catching infographics.
            # graphic_dom is the MEAN semantic signal PLUS the RC5.1 UI-geometry
            # bump (graphic_dom_base = pre-bump). ui_geom is the raw 0..1 signal.
            "graphic_dom": round(float(graphic_dom), 3),
            "graphic_dom_base": round(float(graphic_dom_base), 3),
            "ui_geom": round(float(ui_geom), 3),
            "period_risk": round(period_risk, 3),
            "repetition": round(rep, 3),
            "phash": int(rep_ph),
            "engine": "clip-onnx",
        }
        _asset_cache[ck] = out
        return out
    except Exception:                                          # noqa: BLE001
        return blank


def accept(path, is_video: bool, *, expected: str, objects=(), place: str = "",
           action: str = "", period: str = "", modern_risk: bool = False,
           person_expected: bool = False, concrete: bool = True,
           negatives=(), seen_hashes=None, min_score: float = None,
           guard_only: bool = False, crowd_ok: bool = False,
           vehicle_ok: bool = False):
    """Decide whether to accept a candidate visual for a CONCRETE scene.
    Returns (ok: bool, scores: dict, reason: str). Permissive when the scorer is
    unavailable, when the scene is abstract (concrete=False), or on any error —
    so it NEVER blocks a render, only filters clear pixel-mismatches.

    `guard_only` (V1.3.1) — for an ABSTRACT scene (concrete=False), do NOT skip
    the scorer outright; run it but apply ONLY the wrong-dominant-concept gate
    (a people/portrait collage, a preacher, a war/riot crowd, an office/stadium
    crowd). The relevance/clarity/period/darkness floors are NOT applied, so a
    genuinely atmospheric abstract shot is still never over-filtered — but a
    blatantly off-topic web image (the Cornell-'96%' Ahmed-Deedat collage class:
    a statistic scene that reads 'abstract', whose footage tier then grabbed a
    random face-collage) is finally caught instead of sailing through unscored.

    Each gate is calibrated to clear every GOOD frame measured (high precision):
    a render only loses a clip when the evidence is strong, and the ladder then
    escalates to a more-relevant tier (ultimately a subject-true AI still)."""
    if not available():
        return True, {"engine": "skipped"}, "scorer-off"
    if not concrete and not guard_only:
        return True, {"engine": "skipped"}, "abstract"
    try:
        floor = float(min_score if min_score is not None else
                      os.environ.get("VIDLORE_VISUAL_RELEVANCE_MIN_SCORE",
                                     _DEFAULT_MIN))
    except (TypeError, ValueError):
        floor = _DEFAULT_MIN
    try:
        graphic_max = float(os.environ.get("VIDLORE_VR_GRAPHIC_MAX",
                                           _DEFAULT_GRAPHIC_MAX))
    except (TypeError, ValueError):
        graphic_max = _DEFAULT_GRAPHIC_MAX
    # F2 calibration (2026-06-04): distractor gates are env-tunable so thresholds
    # can be A/B-calibrated per niche WITHOUT code edits (and disabled for
    # rollback). Defaults = the validated current values -> no behavior change
    # unless overridden. CRITICAL distractors (war) stay tight by default; only
    # loosen with measured fail-closed proof (0 wrong-subject returns).
    def _vrf(_n, _d):
        try:
            return float(os.environ.get(_n, _d))
        except (TypeError, ValueError):
            return _d
    dist_max = _vrf("VIDLORE_VR_DISTRACTOR_MAX", 0.05)
    people_max = _vrf("VIDLORE_VR_PEOPLE_MAX", 0.045)
    war_max = _vrf("VIDLORE_VR_WAR_MAX", 0.03)
    vehicle_max = _vrf("VIDLORE_VR_VEHICLE_MAX", 0.05)
    s = score_asset(path, is_video, expected=expected, objects=objects,
                    place=place, action=action, period=period,
                    modern_risk=modern_risk, negatives=negatives,
                    seen_hashes=seen_hashes)
    if s.get("engine") in ("skipped", "cache-skip"):
        return True, s, "scorer-skipped"
    # reject rules — orthogonal signals, each conservative (any one ⇒ reject):
    # (0) DISTRACTOR DOMINANCE (V1.3): the frame matches a known WRONG concept
    # (war/riot crowd, preacher, portrait-collage, sports/office crowd, modern
    # street) MORE than the expected subject -> it is ABOUT the wrong thing.
    # This is the fail-closed gate the war-footage / random-people failures
    # needed; it fires independently of the (compressed) relevance softmax.
    if s.get("distractor_dom", -9) > dist_max:
        return False, s, f"wrong-dominant-concept (dom={s['distractor_dom']})"
    # (0b) DESIGNED-GRAPHIC (2026-06-03): the frame is a designed graphic
    # (infographic / chart / diagram / logo / clip-art / cartoon / poster /
    # screenshot / slide / text sign) rather than a real photograph or footage
    # frame — wrong for ANY documentary beat. This is the keyword-INDEPENDENT
    # gate the weak-keyword sweep needed (party-logo clip-art, a "POLYSEXUAL"
    # text image, a modern mortgage-rates infographic): it compares the frame to
    # a real-photo anchor, NOT the (vague) expected subject, so it rejects a
    # chart/logo even when the scene's keywords are empty. Placed BEFORE the
    # guard-only early-return so it fires on abstract/stat beats too (those are
    # exactly where an off-topic infographic lands as an MG-card background).
    if s.get("graphic_dom", -9) > graphic_max:
        return False, s, f"designed-graphic-not-footage (graphic_dom={s['graphic_dom']})"
    # CROWD / SOLDIERS — the dedicated people probe catches the faded/distant
    # B&W war-crowd & marching-soldier class that the Haar face gate (frontal
    # only) and the compressed distractor-dominance gate both miss. This gate is
    # gated by `crowd_ok`, NOT by `person_expected`: a single-person scene (a
    # name-reveal portrait) legitimately has ONE face but is NEVER a crowd, so a
    # war crowd / soldier montage must still be rejected there. A crowd/army is
    # accepted ONLY when the scene is genuinely ABOUT a crowd/army/protest/battle
    # (history / geopolitics / war topics set crowd_ok=True from the subject) —
    # which is why this is niche-general: soldiers pass on a WWII scene and are
    # rejected on agriculture / business / science / biography scenes alike.
    # Calibrated over 6 live renders: war crowd / soldiers score people_dom
    # 0.045-0.096; nature footage (field/forest/soil/copper/beetle) and a single
    # portrait stay <=0.043.
    if (not crowd_ok) and s.get("people_dom", -9) > people_max:
        return False, s, f"crowd-on-non-crowd-scene (people_dom={s['people_dom']})"
    # WAR / MILITARY on a non-war scene — narrow always-on probe with a tight
    # margin (war footage is the most damaging off-topic class and must reject on
    # EVERY niche except genuine war/history scenes, which set crowd_ok=True).
    if (not crowd_ok) and s.get("war_dom", -9) > war_max:
        return False, s, f"war-footage-on-non-war-scene (war_dom={s['war_dom']})"

    # VEHICLE / generic-Americana dominance — a car / parked vehicle / people
    # posing by a car / casual vintage snapshot, on a scene that is NOT about
    # cars or transport, is subject-wrong even when it is period-correct (the
    # 1960s home-movie-by-a-car on an Amish-garden beat). Always on except when
    # the scene is genuinely about vehicles/driving/roads (vehicle_ok=True).
    if (not vehicle_ok) and s.get("vehicle_dom", -9) > vehicle_max:
        return False, s, (f"vehicle-on-non-vehicle-scene "
                          f"(vehicle_dom={s['vehicle_dom']})")
    if guard_only:
        # Abstract scene: the wrong-dominant-concept gate above is the ONLY one
        # we apply. A clean atmospheric abstract shot has dom<=0.06 and is kept;
        # a people/portrait collage or crowd is rejected. No relevance/clarity/
        # period floor here — those would over-filter legitimate abstract b-roll.
        return True, s, "accepted-guard"
    if modern_risk and s.get("period_risk", 0) > 0.55:
        return False, s, f"period-conflict (modern on historical, risk={s['period_risk']})"
    if (not person_expected) and s.get("face_frac", 0) > 0.05:
        return False, s, f"person-on-non-person-scene (face={s['face_frac']})"
    if s["clarity"] < 0.06:
        return False, s, f"too-unclear (clarity={s['clarity']})"
    if s["darkness_info"] < 0.14:
        return False, s, f"too-dark/low-info (dki={s['darkness_info']})"
    if s["visual_relevance"] < floor:
        return False, s, f"subject-absent (rel={s['visual_relevance']}<{floor})"
    if s.get("repetition", 0) > 0.85:
        return False, s, f"repetition (rep={s['repetition']})"
    # P1.6 — attach a SPECIFICITY re-rank bonus to the accepted verdict so the
    # selector can prefer the more-specific of two acceptable candidates. Pure
    # text signal; never affects accept/reject (we're already past every gate).
    try:
        _spec_txt = " ".join(str(x) for x in (
            expected, " ".join(map(str, objects or ())), place, action) if x)
        s["specificity_bonus"] = specificity_signal(
            _spec_txt, person_beat=bool(person_expected),
            historical_beat=bool(modern_risk or period))
    except Exception:                                          # noqa: BLE001
        s["specificity_bonus"] = 0.0
    return True, s, "accepted"
