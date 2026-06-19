"""MotionGraphicManifest — forward-compat JSON describing every major
card / motion graphic this engine emits.

PURPOSE
-------
Today the engine renders motion graphics with PIL + FFmpeg.  That's
cheap (~$0.05 per 10-min doc in compute), reliable, and produces a
final MP4.  No change to that pipeline.

TOMORROW we may want a web editor (live scrub preview, click-to-edit,
React-based timeline UI) or a Remotion premium-tier renderer for a
specific advanced motion graphic.  When that day comes we DON'T want
to reverse-engineer 100 PIL renderers to learn what each one drew.

This manifest is the answer.  Every major renderer optionally emits a
small JSON file next to its PNG describing the editorial decisions in
a structured, render-agnostic form:

    document.png         <- the actual frame
    document.png.json    <- this manifest

A future React preview component can read the manifest and reproduce
the card in the browser without invoking PIL.  A future Remotion
component can read the manifest and animate the card with browser-
grade motion primitives.  Today, nothing reads the manifest — it's
write-only and zero-cost.

DESIGN INVARIANTS
-----------------
1. **Write-only today.**  Nothing in the render path reads the
   manifest.  If emission fails, the render still succeeds.
2. **Render-agnostic.**  No PIL-specific or FFmpeg-specific fields.
   Colors are sRGB tuples, positions are pixel coords on the canonical
   1920×1080 frame, timings are seconds, fonts are family-name chains
   (not file paths).  A React renderer could consume it without
   knowing what PIL or FFmpeg are.
3. **Channel-traced.**  Every manifest captures the Look DNA values
   that drove its visual decisions so a designer can audit "why did
   Atlas's quote card look like that?"  This is also the seed of
   future per-channel style debugging.
4. **Small and stable.**  Schema versioned (`schema_version`), keys
   namespaced, optional fields default to empty.  A v1 reader should
   handle a v2 manifest by ignoring unknown keys.
5. **No external deps.**  Pure stdlib — `dataclasses` + `json`.

SCHEMA (v1)
-----------
Every manifest is a single JSON object with these top-level keys:

    schema_version : int             (always 1 for now)
    type           : str             ("document", "quote_highlight",
                                       "name_reveal", "lower_third",
                                       "stat_dashboard", "timeline",
                                       "map_reveal", "comparison",
                                       "process_diagram", "bullet_list",
                                       ...)
    channel        : str | null      (active Look DNA channel name)
    scene_index    : int             (the scene this card belongs to)
    intensity      : int             (1-5, scene emotional energy)
    role           : str             (showrunner role, e.g. "evidence")

    canvas         : { w: int, h: int }
                                     (always 1920x1080 today)

    text_elements  : [ { id, text, role, bbox, font_family,
                         font_size, color, align, weight, casing } ]
                                     (every visible text block — id is
                                      stable per renderer so a future
                                      preview can map back to source)

    layout         : { name: { x, y, w, h }, ... }
                                     (named regions — "paper", "panel",
                                      "marker", "portrait", "headline",
                                      etc.  Renderer-defined.)

    colors         : { paper_rgb, ink_rgb, mute_rgb, highlight_rgb,
                       accent_rgb, panel_bg_rgb, panel_text_rgb, ... }
                                     (only the colors actually used.
                                      All are [r, g, b] 0-255 lists.)

    fonts          : { display_family_chain: [...],
                       body_family_chain: [...],
                       caps_tracking: float,
                       weight_bias: str,
                       title_case_mode: str }

    timing         : { reveal_in_s, hold_s, tail_out_s,
                       reveal_curve, hold_curve, tail_curve }
                                     (intent only — actual FFmpeg
                                      composite times live in the
                                      assembler's manifest.txt)

    animations     : [ { layer, kind, from, to, duration_s, easing,
                         delay_s, params } ]
                                     (declarative motion intent.
                                      Layer = a key from `layout` or
                                      `text_elements[].id`.  Kind =
                                      "wipe_lr" / "fade_in" /
                                      "slide_from_right" / "scale_in"
                                      etc.)

    assets         : { plain_png, focus_png, hl_png, portrait_png,
                       ... }
                                     (paths RELATIVE to the directory
                                      this manifest sits in)

    look_dna_used  : { bg_mode, accent_style, card_radius,
                       card_padding_mult, rule_weight_px, tone,
                       paper_grain, vignette_strength, ... }
                                     (the specific Look DNA knobs that
                                      drove this card's appearance —
                                      reproducibility + audit trail)

    extras         : { ... }         (renderer-specific overflow.
                                      Reader must ignore unknown keys.)


USAGE
-----
At the END of a card renderer:

    from .motion_graphic_manifest import MotionGraphicManifest, write_manifest

    mf = MotionGraphicManifest.for_card(
        type_="document",
        scene=sc,
        theme=theme,
        pal=pal,
        canvas=(W, H),
    )
    mf.add_text("title", title, role="title",
                bbox=(LX, top - 50, RX, top - 12),
                font_family=pal["display_font"],
                font_size=46, color=pal["ink_rgb"])
    mf.add_layout("paper", 0, 0, W, H)
    mf.add_layout("marker", HL_X, HL_Y, BW, BH)
    mf.add_animation("marker", kind="wipe_lr", duration_s=0.85,
                     easing="ease_out_cubic")
    mf.add_asset("plain_png", plain.name)
    mf.add_asset("focus_png", focus.name)
    write_manifest(focus, mf)        # writes focus.png.json next to focus.png

Renderers may skip emission entirely without breaking anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA_VERSION = 1


def _rgb(value) -> Optional[list[int]]:
    """Coerce anything color-shaped into a [r, g, b] list of ints,
    clamped 0-255.  Returns None for unsupported input."""
    if value is None:
        return None
    try:
        r, g, b = list(value)[:3]
    except (TypeError, ValueError):
        return None
    return [
        max(0, min(255, int(r))),
        max(0, min(255, int(g))),
        max(0, min(255, int(b))),
    ]


@dataclass
class TextElement:
    """A single visible text block on the card."""
    id: str
    text: str
    role: str = ""                    # "title" | "body" | "source" | "label" | "value" | ...
    bbox: Optional[list[int]] = None  # [x0, y0, x1, y1] in canvas px
    font_family: list[str] = field(default_factory=list)
    font_size: int = 0
    color: Optional[list[int]] = None
    align: str = "left"               # "left" | "center" | "right"
    weight: str = "regular"           # "regular" | "bold" | "black"
    casing: str = "as_is"             # "as_is" | "upper" | "title" | "lower"


@dataclass
class Animation:
    """A declarative animation intent for one layer."""
    layer: str
    kind: str                          # "wipe_lr" | "fade_in" | "scale_in" | ...
    from_: float = 0.0
    to: float = 1.0
    duration_s: float = 0.5
    easing: str = "ease_out_cubic"
    delay_s: float = 0.0
    params: dict = field(default_factory=dict)


@dataclass
class MotionGraphicManifest:
    """The structured description of one motion graphic card.

    Build with `MotionGraphicManifest.for_card(...)` then attach
    text elements / layout regions / animations / assets via the
    `add_*` helpers.  Serialise with `to_dict()` or write next to
    the PNG with `write_manifest(png_path, mf)`."""
    schema_version: int = SCHEMA_VERSION
    type: str = ""
    channel: Optional[str] = None
    scene_index: int = 0
    intensity: int = 0
    role: str = ""

    canvas: dict = field(default_factory=lambda: {"w": 1920, "h": 1080})

    text_elements: list[TextElement] = field(default_factory=list)
    layout: dict = field(default_factory=dict)
    colors: dict = field(default_factory=dict)
    fonts: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    animations: list[Animation] = field(default_factory=list)
    assets: dict = field(default_factory=dict)
    look_dna_used: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    # ── factory ──────────────────────────────────────────────────
    @classmethod
    def for_card(cls, type_: str, *, scene=None, theme=None, pal=None,
                 canvas: tuple[int, int] = (1920, 1080)) -> "MotionGraphicManifest":
        """Convenience constructor that pre-fills the channel /
        scene / Look-DNA fields from the standard render context.

        `scene` may be any object with `index` / `intensity` /
        `role` attributes (the Scene dataclass qualifies).  `pal`
        may be the dict returned by `footage._look_card_palette()`.
        All args are optional — missing fields default sensibly."""
        m = cls(type=type_)
        m.canvas = {"w": int(canvas[0]), "h": int(canvas[1])}
        if scene is not None:
            m.scene_index = int(getattr(scene, "index", 0) or 0)
            m.intensity   = int(getattr(scene, "intensity", 0) or 0)
            m.role        = str(getattr(scene, "role", "") or "")
        if pal is not None:
            m.channel = pal.get("channel_id")
        # Fallback channel detection — read the active ResolvedLook
        # directly so manifest channel field is reliable even when
        # the renderer didn't populate pal["channel_id"].
        if not m.channel:
            try:
                from .look_dna import current as _ld_current
                _cur = _ld_current()
                if _cur is not None:
                    m.channel = getattr(_cur, "name", None) or m.channel
            except Exception:                              # noqa: BLE001
                pass
            # colors that actually exist on the palette
            for k in ("paper_rgb", "ink_rgb", "mute_rgb", "highlight",
                      "panel_bg", "panel_text", "panel_text2",
                      "accent"):
                v = _rgb(pal.get(k))
                if v is not None:
                    m.colors[k if k != "highlight" else "highlight_rgb"] = v
                    if k == "accent":
                        m.colors["accent_rgb"] = v
            # fonts
            for k in ("display_font", "body_font"):
                if pal.get(k):
                    m.fonts[k.replace("_font", "_family_chain")] = \
                        list(pal[k])
            for k in ("caps_tracking", "weight_bias", "title_case_mode"):
                if pal.get(k) is not None:
                    m.fonts[k] = pal[k]
            # Look DNA knobs that drove this card
            for k in (
                "bg_mode", "accent_style", "accent_motif",
                "card_radius", "card_padding_mult", "gutter_mult",
                "rule_weight_px", "panel_alpha", "panel_border_alpha",
                "tone", "paper_grain", "vignette_strength",
                "chart_style", "map_style", "lt_variant",
                "timeline_style",
            ):
                if pal.get(k) is not None:
                    m.look_dna_used[k] = pal[k]
        return m

    # ── builders ─────────────────────────────────────────────────
    def add_text(self, id_: str, text: str, *, role: str = "",
                 bbox=None, font_family=None, font_size: int = 0,
                 color=None, align: str = "left", weight: str = "regular",
                 casing: str = "as_is") -> "MotionGraphicManifest":
        """Append a TextElement.  bbox is (x0,y0,x1,y1) in canvas px."""
        te = TextElement(
            id=id_,
            text=str(text or ""),
            role=role,
            bbox=[int(v) for v in bbox] if bbox else None,
            font_family=list(font_family or []),
            font_size=int(font_size or 0),
            color=_rgb(color),
            align=align,
            weight=weight,
            casing=casing,
        )
        self.text_elements.append(te)
        return self

    def add_layout(self, name: str, x: int, y: int,
                   w: int, h: int) -> "MotionGraphicManifest":
        """Register a named layout region (px on canvas)."""
        self.layout[name] = {
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        }
        return self

    def add_color(self, name: str, rgb) -> "MotionGraphicManifest":
        """Record a colour the renderer actually used."""
        v = _rgb(rgb)
        if v is not None:
            self.colors[name] = v
        return self

    def add_animation(self, layer: str, *, kind: str,
                      from_: float = 0.0, to: float = 1.0,
                      duration_s: float = 0.5,
                      easing: str = "ease_out_cubic",
                      delay_s: float = 0.0,
                      params: Optional[dict] = None) -> "MotionGraphicManifest":
        self.animations.append(Animation(
            layer=layer, kind=kind, from_=float(from_), to=float(to),
            duration_s=float(duration_s), easing=easing,
            delay_s=float(delay_s), params=dict(params or {}),
        ))
        return self

    def add_asset(self, role: str, path) -> "MotionGraphicManifest":
        """Record a source asset (path or filename, RELATIVE to the
        manifest's directory)."""
        if path:
            self.assets[role] = str(path)
        return self

    def set_timing(self, *, reveal_in_s: float = 0.0,
                   hold_s: float = 0.0, tail_out_s: float = 0.0,
                   reveal_curve: str = "ease_out_cubic",
                   hold_curve: str = "linear",
                   tail_curve: str = "ease_in_cubic") -> "MotionGraphicManifest":
        self.timing = {
            "reveal_in_s": float(reveal_in_s),
            "hold_s":      float(hold_s),
            "tail_out_s":  float(tail_out_s),
            "reveal_curve": reveal_curve,
            "hold_curve":   hold_curve,
            "tail_curve":   tail_curve,
        }
        return self

    def add_extras(self, **kwargs) -> "MotionGraphicManifest":
        """Stash renderer-specific overflow data.  Reader must
        ignore unknown keys."""
        self.extras.update(kwargs)
        return self

    # ── serialisation ────────────────────────────────────────────
    def to_dict(self) -> dict:
        out = asdict(self)
        # rename `from_` -> `from` in animations (Python keyword)
        out["animations"] = [
            {("from" if k == "from_" else k): v for k, v in a.items()}
            for a in out["animations"]
        ]
        return out

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent,
                          ensure_ascii=False, sort_keys=False)


def write_manifest(png_path, manifest: MotionGraphicManifest,
                   *, suffix: str = ".json") -> Optional[Path]:
    """Write the manifest next to its PNG.  Best-effort: any failure
    is swallowed so a write error never breaks a render.

    For `document.png` this produces `document.png.json` by default.
    Override `suffix` to control the file extension."""
    try:
        p = Path(png_path)
        out = p.with_name(p.name + suffix)
        out.write_text(manifest.to_json(), encoding="utf-8")
        return out
    except Exception:                                       # noqa: BLE001
        return None


def safe_emit(png_path, type_: str, *, scene=None, theme=None,
              pal=None, canvas: tuple[int, int] = (1920, 1080),
              configure=None) -> Optional[Path]:
    """Single-call convenience helper.

    `configure` is an optional callable that receives the freshly-
    built MotionGraphicManifest and lets the renderer add text /
    layout / animations.  Returns the path of the written .json
    (or None on failure).  Never raises.

    Example:

        safe_emit(focus, "document", scene=sc, theme=theme, pal=pal,
                  configure=lambda m: (
                      m.add_layout("paper", 0, 0, W, H)
                       .add_layout("marker", HL_X, HL_Y, BW, BH)
                       .add_text("body", body, role="body",
                                 font_family=pal["body_font"],
                                 font_size=43, color=pal["ink_rgb"])
                       .add_animation("marker", kind="wipe_lr",
                                      duration_s=0.85)
                       .add_asset("plain_png", plain.name)
                       .add_asset("focus_png", focus.name)))
    """
    try:
        mf = MotionGraphicManifest.for_card(
            type_, scene=scene, theme=theme, pal=pal, canvas=canvas)
        if configure is not None:
            configure(mf)
        return write_manifest(png_path, mf)
    except Exception:                                       # noqa: BLE001
        return None


__all__ = [
    "SCHEMA_VERSION",
    "TextElement",
    "Animation",
    "MotionGraphicManifest",
    "write_manifest",
    "safe_emit",
]
