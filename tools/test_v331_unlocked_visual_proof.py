#!/usr/bin/env python3
"""V3.3.1 STEP 4 — visual proof for every V3.3-unlocked primitive.

For each unlocked kind: parse a valid structured body through the REAL validator
(_parse_extra) → route the kind+assets through the REAL director → micro-render the
selected primitive → extract a proof frame → production black-QA → record a proof
manifest. Then tile a desktop + mobile contact sheet. Inspect the actual frames
(this script writes them; a human/agent reads them visually).

  python tools/test_v331_unlocked_visual_proof.py
"""
from __future__ import annotations
import json, subprocess, sys, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore import script_gen as SG                          # noqa: E402
from vidlore.motion_graphics import director as D             # noqa: E402
from vidlore.motion_graphics import registry as R             # noqa: E402
from vidlore.ffmpeg_tool import ffmpeg_exe                    # noqa: E402

FF = ffmpeg_exe()
OUT = ROOT / "research/motion_graphics_expansion/emission_unlock/v331_visual_proof"
OUT.mkdir(parents=True, exist_ok=True)
W, H, FPS, DUR = 960, 540, 30, 3.0

# (graphic_kind, valid graphic_body, expected primitive, render kwargs from the
#  adapter-shaped real data — these mirror exactly what the pipeline adapter produces)
CASES = [
    ("bar_chart", "bars=Pipelines:40|Rail:25|Refineries:35;suffix=%", "statistic_bar_reveal",
     dict(bars=[{"label": "Pipelines", "value": 40}, {"label": "Rail", "value": 25}, {"label": "Refineries", "value": 35}], title="SHARE OF FLOW", suffix="%")),
    ("versus", "pair=Standard Oil|The Independents;values=90|10;suffix=%", "comparison_split",
     dict(left="Standard Oil", right="The Independents", leftval=90, rightval=10, suffix="%")),
    ("balance", "pair=Monopoly|Competition;values=90|10", "vs_balance_scale",
     dict(left="Monopoly", right="Competition", leftval=90, rightval=10, title="THE BALANCE")),
    ("composition", "segments=Crude:50|Refined:30|Export:20;suffix=%", "composition_stack",
     dict(segments=[{"label": "Crude", "value": 50}, {"label": "Refined", "value": 30}, {"label": "Export", "value": 20}], title="WHERE IT WENT", suffix="%")),
    ("process", "steps=Survey|Acquire|Integrate|Dominate", "process_flow_steps",
     dict(steps=["Survey", "Acquire", "Integrate", "Dominate"], title="THE PLAYBOOK")),
    ("hierarchy", "children=Domestic|Export|Pipelines", "org_hierarchy_tree",
     dict(root="Standard Oil Trust", children=[{"label": "Domestic"}, {"label": "Export"}, {"label": "Pipelines"}], title="STRUCTURE")),
    ("before_after", "before=1865;after=1882", "before_after_slider",
     dict(before_label="1865", after_label="1882", caption="The refinery district")),
    ("decision", "yes=Absorbed|no=Driven out|chosen=yes", "flowchart_decision",
     dict(question="Sell or be crushed?", yes="Absorbed at his price", no="Driven out", chosen="yes", title="THE ULTIMATUM")),
    ("sankey", "branches=Reinvest:50|Dividends:30|Reserves:20", "sankey_flow",
     dict(source="Revenue", branches=[{"label": "Reinvest", "value": 50}, {"label": "Dividends", "value": 30}, {"label": "Reserves", "value": 20}], total=100, title="WHERE THE MONEY WENT")),
    ("eras", "eras=Rise:1865-1882|Peak:1882-1904|Fall:1904-1911", "era_band_timeline",
     dict(eras=[{"label": "Rise", "from": "1865", "to": "1882"}, {"label": "Peak", "from": "1882", "to": "1904"}, {"label": "Fall", "from": "1904", "to": "1911"}], title="THREE ERAS")),
    ("headlines", "headlines=Trust on trial|Court orders breakup|Oil king's empire", "headline_montage",
     dict(headlines=["TRUST ON TRIAL", "COURT ORDERS BREAKUP", "OIL KING'S EMPIRE"], title="THE HEADLINES")),
    ("gauge", "value=72;bands=LOW|HIGH;readout=72 dB", "spectrum_meter",
     dict(value=72, label="SIGNAL", bands=["LOW", "GUARDED", "ELEVATED", "HIGH"], readout="72 dB", title="INTERCEPT")),
    ("scale_compare", "items=Carrier:333:333 m|Bus:12:12 m", "silhouette_scale_compare",
     dict(title="TRUE SCALE", items=[{"label": "Carrier", "size": 333, "note": "333 m"}, {"label": "Bus", "size": 12, "note": "12 m"}])),
    ("route_trace", "points=0.2:0.7:Start|0.5:0.45|0.8:0.3:End", "footage_route_trace",
     dict(title="THE PIPELINE", points=[[0.2, 0.7], [0.5, 0.45], [0.8, 0.3]])),
]


def _blackqa(mp4):
    from PIL import Image
    try:
        from vidlore import editorial_qa as EQ
        dl, ds, stats = EQ.DEAD_LUMA, EQ.DEAD_STD, EQ.luma_stats
    except Exception:
        return -1
    td = Path(tempfile.mkdtemp())
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp4),
                    "-vf", "fps=4", str(td / "q%03d.png")], capture_output=True)
    dead = 0
    for i, p in enumerate(sorted(td.glob("q*.png"))):
        try:
            m, s = stats(Image.open(p))
            if m < dl and s < ds and i / 4.0 > 0.5:
                dead += 1
        except Exception:
            pass
    shutil.rmtree(td, ignore_errors=True)
    return dead


def main():
    frames_dir = OUT / "frames"; frames_dir.mkdir(exist_ok=True)
    records, card_mp4s = [], []
    for kind, body, prim, kw in CASES:
        rec = {"graphic_kind": kind, "expected_primitive": prim}
        # 1) validator
        _, gk, gt, gb = SG._parse_extra({"graphic": {"kind": kind, "text": prim, "body": body}})
        rec["validator_accepted"] = (gk == kind and bool(gb))
        # 2) director routing (kind + adapter-shaped assets)
        mg = [{"index": 0, "role": "proof", "graphic_kind": kind, "intensity": 3,
               "narration": "x", "emphasis": "", "assets": kw}]
        sel = [d.primitive for d in D.plan(mg, niche="business", seed=7, density=0.7) if d.primitive]
        rec["director_selected"] = prim in sel
        # 3) render the primitive directly with the same data
        e = R.REGISTRY.get(prim)
        mp4 = OUT / f"{prim}.mp4"
        try:
            import inspect
            params = set(inspect.signature(e["render"]).parameters)
            res = e["render"](str(mp4), dur=DUR, fps=FPS, w=W, h=H, seed=7,
                              **{k: v for k, v in kw.items() if k in params})
            rec["render_ok"] = bool(res.get("ok"))
            rec["fallback"] = bool(res.get("fallback", False))
        except Exception as ex:                                # noqa: BLE001
            rec["render_ok"] = False
            rec["err"] = f"{type(ex).__name__}: {ex}"[:120]
        if rec.get("render_ok") and mp4.exists():
            rec["dead_black_post_entrance"] = _blackqa(mp4)
            subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-ss", "2.1",
                            "-i", str(mp4), "-frames:v", "1", str(frames_dir / f"{prim}.png")], capture_output=True)
            card_mp4s.append(mp4)
        records.append(rec)
        print(f"  {kind:14s} -> {prim:24s} valid={rec.get('validator_accepted')} "
              f"route={rec.get('director_selected')} render={rec.get('render_ok')} "
              f"fb={rec.get('fallback')} dead={rec.get('dead_black_post_entrance','-')} {rec.get('err','')[:40]}")
    # reel + contact sheets
    if card_mp4s:
        lst = OUT / "_concat.txt"; lst.write_text("\n".join(f"file '{m}'" for m in card_mp4s))
        subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy", str(OUT / "v331_unlocked_reel.mp4")], capture_output=True)
        lst.unlink(missing_ok=True)
        for m in card_mp4s:
            m.unlink(missing_ok=True)
    _sheet(frames_dir, OUT / "sheet_desktop.png", 960)
    _sheet(frames_dir, OUT / "sheet_mobile.png", 360)
    (OUT / "v331_proof_manifest.json").write_text(json.dumps(records, indent=1))
    ok = sum(1 for r in records if r.get("render_ok") and r.get("validator_accepted") and r.get("director_selected"))
    dead = sum(1 for r in records if r.get("dead_black_post_entrance", 0))
    fb = sum(1 for r in records if r.get("fallback"))
    print(f"\n  full-chain ok: {ok}/{len(CASES)} | fallback>0: {fb} | dead-black-post-entrance: {dead}")
    print("  artifacts ->", OUT)
    return ok == len(CASES) and dead == 0 and fb == 0


def _sheet(frames_dir, out_png, card_w):
    from PIL import Image
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        return
    cols = 3 if card_w >= 900 else 4
    cards = []
    for p in pngs:
        try:
            im = Image.open(p).convert("RGB")
            cards.append(im.resize((card_w, int(im.height * card_w / im.width)), Image.LANCZOS))
        except Exception:
            pass
    if not cards:
        return
    cw, chh = cards[0].size
    rows = (len(cards) + cols - 1) // cols
    pad = max(6, card_w // 80)
    sheet = Image.new("RGB", (cols * cw + (cols + 1) * pad, rows * chh + (rows + 1) * pad), (12, 12, 14))
    for i, c in enumerate(cards):
        r, cc = divmod(i, cols)
        sheet.paste(c, (pad + cc * (cw + pad), pad + r * (chh + pad)))
    sheet.save(out_png)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
