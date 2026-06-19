"""P0 PROOF — render the same SCENE-TYPE samples through 3 channels
side-by-side, then montage a contact sheet so we can compare them.

This bypasses the full pipeline so we get a fast visual proof.  It
exercises ONLY the wired card renderers (document, quote_highlight,
lower_third, stat_dashboard, name_reveal) — the surfaces that drive
the strongest visible differentiation.

Output:  output/p0_proof/<channel>/<card>.png
         output/p0_proof/contact_sheet.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from PIL import Image

import vidlore.footage as fg                                  # noqa: E402
from vidlore.look_dna import (                                # noqa: E402
    set_current, clear_current, resolve_look,
)
from vidlore.brief import Brief                              # noqa: E402


CHANNELS = ["midnight_pacific", "atlas_explained", "amber_chronicles"]
OUT = ROOT / "output" / "p0_proof"
OUT.mkdir(parents=True, exist_ok=True)


def _load_look(name: str):
    """Load a look preset directly via Brief.look_preset."""
    brief = Brief(
        title="P0 proof",
        prompt="proof",
        fmt="documentary",
        duration="6-8",
        theme="history",
        look_preset=name,
    )
    return resolve_look(brief)


def _theme():
    """Return the standard documentary theme (assembler's `mode.theme`)."""
    return {
        "accent": (210, 170, 90),
        "ink":    (235, 235, 235),
        "panel":  (10, 14, 22),
        "shadow": (0, 0, 0),
    }


def _render_one_card(channel: str, kind: str, slug: str):
    """Render one card with the given channel look active."""
    look = _load_look(channel)
    set_current(look)
    try:
        sub = OUT / channel
        sub.mkdir(exist_ok=True)
        dest = sub / f"{slug}.png"
        theme = _theme()
        # accent override (pipeline.py does this normally)
        acc = look.get("accent_rgb", [210, 170, 90])
        theme["accent"] = (int(acc[0]), int(acc[1]), int(acc[2]))

        if kind == "document":
            plain  = sub / f"{slug}_plain.png"
            focus  = sub / f"{slug}.png"
            meta   = sub / f"{slug}.meta.json"
            ok = fg._render_document(
                "Reuters, October 14 2008",
                "The decision to nationalize three major banks marked the "
                "single biggest peacetime intervention in the British "
                "economy. Within forty-eight hours, the Treasury committed "
                "£37 billion in emergency capital — roughly 2.5% of GDP — "
                "to halt a collapse that had begun a month earlier with "
                "Lehman Brothers.",
                plain, focus, meta)
            return ok, focus

        if kind == "quote":
            class _MockSc:
                emphasis = "collapse"
            ok = fg._render_quote_highlight_card(
                _MockSc(), theme, dest,
                "The fundamentals of our economy are strong.",
                "Henry Paulson, US Treasury Secretary")
            return ok, dest

        if kind == "lower_third":
            class _MockSc2:
                pass
            ok = fg._render_lower_third_card(
                _MockSc2(), theme, dest,
                "Henry Paulson",
                "U.S. Treasury Secretary, 2008",
                variant=0)
            return ok, dest

        if kind == "stat_dashboard":
            class _MockSc3:
                pass
            ok = fg._render_stat_dashboard_card(
                _MockSc3(), theme, dest,
                "By The Numbers",
                "BANKS BAILED OUT|17;TOTAL COMMITTED|$700B;PEAK UNEMPLOY|10.0%;"
                "RECOVERY|6 YRS")
            return ok, dest

        if kind == "name_reveal":
            class _MockSc4:
                pass
            ok = fg._render_name_reveal_card(
                _MockSc4(), theme, dest,
                name="HENRY PAULSON",
                role="U.S. TREASURY SECRETARY")
            return ok, dest

        if kind == "timeline":
            class _MockSc5:
                emphasis = ""
            ok = fg._render_timeline_card(
                _MockSc5(), theme, dest,
                "Collapse Sequence",
                "Sep 7, 2008|Fannie & Freddie seized;"
                "Sep 15|Lehman files Chapter 11;"
                "Sep 16|AIG rescued ($85B);"
                "Oct 3|TARP signed ($700B);"
                "Oct 14|UK nationalises 3 banks")
            return ok, dest

        if kind == "map":
            class _MockSc6:
                pass
            ok = fg._render_real_map_reveal_card(
                _MockSc6(), theme, dest,
                "Lower Manhattan",
                coords="40.7074,-74.0113",
                style="satellite")
            return ok, dest

        if kind == "comparison":
            class _MockSc7:
                emphasis = ""
            ok = fg._render_comparison_card(
                _MockSc7(), theme, dest,
                "Before / After",
                "2007|4.2% CAPITAL RATIO;;"
                "2024|13.1% CAPITAL RATIO")
            return ok, dest

        if kind == "process":
            class _MockSc8:
                pass
            ok = fg._render_process_diagram_card(
                _MockSc8(), theme, dest,
                "Bailout Mechanics",
                "Run on bank|Depositors withdraw faster than reserves;"
                "Fed line opens|Treasury injects emergency capital;"
                "Stabilisation|Confidence restored, withdrawals slow;"
                "Recapitalisation|Bank rebuilds reserves over months")
            return ok, dest

        if kind == "bullets":
            class _MockSc9:
                index = 0
            ok = fg._render_bullet_list_card(
                _MockSc9(), theme, dest,
                "Key Vulnerabilities",
                "Over-leveraged investment banks;"
                "Opaque derivatives exposure;"
                "Inadequate capital reserves;"
                "Regulatory blind spots")
            return ok, dest

        return False, dest
    finally:
        clear_current()


def main():
    cards = ["document", "name_reveal", "quote", "lower_third",
             "stat_dashboard", "timeline", "map", "comparison",
             "process", "bullets"]
    grid = {}    # channel -> card -> path
    for ch in CHANNELS:
        grid[ch] = {}
        for k in cards:
            ok, p = _render_one_card(ch, k, k)
            grid[ch][k] = (ok, p)
            status = "OK" if ok else "FAIL"
            print(f"  [{ch:>18}] {k:>15}  {status}  {p}")

    # ---- montage a single contact sheet -------------------------------
    THUMB_W = 600
    THUMB_H = int(THUMB_W * 9 / 16)
    GAP = 14
    LABEL_H = 32
    rows = cards
    cols = CHANNELS
    sheet_w = LABEL_H + GAP + len(cols) * (THUMB_W + GAP) + GAP
    sheet_h = LABEL_H + GAP + len(rows) * (THUMB_H + GAP) + GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    # column headers
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except Exception:
        font = ImageFont.load_default()
    for i, ch in enumerate(cols):
        x = LABEL_H + GAP + i * (THUMB_W + GAP)
        d.text((x + 6, 6), ch, fill=(230, 230, 230), font=font)
    for j, kind in enumerate(rows):
        y = LABEL_H + GAP + j * (THUMB_H + GAP)
        d.text((6, y + THUMB_H // 2 - 8), kind, fill=(180, 180, 180), font=font)
        for i, ch in enumerate(cols):
            ok, p = grid[ch][kind]
            x = LABEL_H + GAP + i * (THUMB_W + GAP)
            if ok and p.exists():
                im = Image.open(p).convert("RGB").resize((THUMB_W, THUMB_H),
                                                          Image.LANCZOS)
                sheet.paste(im, (x, y))
            else:
                d.rectangle([x, y, x + THUMB_W, y + THUMB_H],
                             fill=(60, 30, 30))

    out = OUT / "contact_sheet.png"
    sheet.save(out)
    print(f"\nCONTACT SHEET -> {out}")


if __name__ == "__main__":
    main()
