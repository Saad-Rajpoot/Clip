#!/usr/bin/env python3
"""MG FAMILY DEMO REELS (V3.2.1 master-validation STEP 1 + 2).

Renders every registered primitive with CORRECT, curated input shapes, grouped
into 8 family reels. Per family: a concatenated reel MP4 (960x540), a desktop
contact sheet, a mobile-size contact sheet (cards @360px wide), proof frames,
and a per-card record (ok / fallback / black / render_s). Disk-frugal: per-card
MP4s are blackdetect-probed then DELETED after the reel is concatenated; only the
reel MP4 + frames + sheets survive.

Run all:      python tools/_mg_family_reels.py
Run one reel: python tools/_mg_family_reels.py maps
"""
from __future__ import annotations
import inspect, json, subprocess, sys, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidlore.motion_graphics import registry as R          # noqa: E402
from vidlore.ffmpeg_tool import ffmpeg_exe                  # noqa: E402

FF = ffmpeg_exe()
OUT = ROOT / "research/motion_graphics_expansion/master_audit/demo_reels"
OUT.mkdir(parents=True, exist_ok=True)
W, H, FPS, DUR = 960, 540, 30, 3.0

# ── 8 family reels — EVERY primitive appears in exactly one reel (71 total) ──
REELS = {
    "maps": ["diplomatic_link", "location_establish_card", "map_badge_node",
             "map_heat_spread", "map_region_highlight", "map_route_spread",
             "map_status_banner", "parchment_war_map", "portrait_name_over_map",
             "supply_route_dashes", "territory_advance_arrows",
             "velocity_route_map", "world_map_arc"],
    "spy_investigation": ["classified_stamp_reveal", "evidence_connection_board",
             "investigation_location_map", "route_comparison",
             "sightline_trajectory", "suspect_profile_card",
             "witness_testimony_card", "redacted_document"],
    "documents_evidence": ["headline_document_reveal", "framed_evidence_spotlight",
             "headline_montage", "era_band_timeline", "era_stamp_overlay",
             "annotated_detail_callout"],
    "business_charts": ["composition_stack", "growth_curve_chart",
             "money_flow_empire", "pictograph_scale", "proportion_ring",
             "ranked_list_countdown", "sankey_flow", "statistic_bar_reveal",
             "wealth_arc_counter", "acquisition_timeline", "supply_chain_network",
             "gold_number_callout", "comparison_split", "vs_balance_scale",
             "countdown_clock", "spectrum_meter"],
    "tech_systems_hybrid": ["exploit_chain", "packet_path_trace",
             "system_planview_flow", "footage_fact_overlay",
             "footage_object_callout", "footage_route_trace"],
    "biography_character": ["relationship_roster", "cinematic_portrait_hold",
             "portrait_legend_reveal", "pull_quote_portrait", "quote_stream",
             "verdict_duality_card"],
    "science_engineering": ["labeled_cross_section", "measurement_callout",
             "silhouette_scale_compare"],
    "typography_timeline_process": ["act_chapter_card", "kinetic_keyword",
             "chronology_timeline", "life_milestone_spine", "cause_effect_chain",
             "connection_web", "flowchart_decision", "org_hierarchy_tree",
             "process_flow_steps", "definition_card", "statement_card",
             "before_after_slider", "spotlight_object_hold"],
}

# ── curated, valid demo inputs (real shapes from impl signatures/docstrings) ──
DEMO = {
    # maps
    "diplomatic_link": dict(a="USSR", b="Egypt", outcome="ARMS PACT · 1955", title="THE ALIGNMENT"),
    "location_establish_card": dict(place="Stalingrad", sub="Eastern Front", coords="48.7°N 44.5°E"),
    "map_badge_node": dict(place="Berlin", label="HEADQUARTERS", at=[0.5, 0.42]),
    "map_heat_spread": dict(title="THE SPREAD", hotspots=[{"at": [0.32, 0.40], "intensity": 0.9},
                            {"at": [0.58, 0.52], "intensity": 0.6}, {"at": [0.70, 0.35], "intensity": 0.4}]),
    "map_region_highlight": dict(region="Bavaria", pos=[0.52, 0.50], sub="Industrial heartland"),
    "map_route_spread": dict(title="SUPPLY LINE", stops=[["Baku", [0.72, 0.62]],
                             ["Rostov", [0.52, 0.44]], ["Berlin", [0.30, 0.30]]]),
    "map_status_banner": dict(year="1943", event="SOVIET COUNTER-OFFENSIVE", place="STALINGRAD", ticks=["0.46,0.50", "0.52,0.44"]),
    "parchment_war_map": dict(title="THE EASTERN FRONT", side_a="AXIS", side_b="SOVIET", front_x=0.5),
    "portrait_name_over_map": dict(name="Erwin Rommel", place="NORTH AFRICA", side="left"),
    "supply_route_dashes": dict(source=["Moscow", [0.50, 0.30]],
                                dests=[["Stalingrad", [0.62, 0.60]], ["Kiev", [0.32, 0.50]]]),
    "territory_advance_arrows": dict(label="OPERATION BARBAROSSA", event="ADVANCE", year="1941",
                                origin_region="Poland", target_region="Ukraine", region="europe"),
    "velocity_route_map": dict(title="THE PUSH", stops=["Baku", "Rostov", "Stalingrad", "Berlin"]),
    "world_map_arc": dict(from_place="London", to_place="New York", title="THE CROSSING"),
    # spy / investigation
    "classified_stamp_reveal": dict(header="CASE FILE 0447", body="Subject linked to three intercepts in East Berlin.",
                                reveal="CONFIRMED", stamp_text="CLASSIFIED"),
    "evidence_connection_board": dict(title="THE NETWORK", center="The Handler",
                                nodes=[{"label": "Courier"}, {"label": "Embassy"}, {"label": "Dead drop"}],
                                links=[[0, 1], [1, 2], [0, 2]]),
    "investigation_location_map": dict(title="SIGHTINGS", place="EAST BERLIN",
                                pins=[{"label": "Alley", "at": [0.32, 0.60]}, {"label": "River", "at": [0.70, 0.42]}]),
    "route_comparison": dict(title="TWO ROUTES", start="Embassy", end="Safehouse",
                                route_a=[[0.20, 0.70], [0.50, 0.50], [0.80, 0.30]],
                                route_b=[[0.20, 0.70], [0.45, 0.62], [0.80, 0.30]]),
    "sightline_trajectory": dict(title="THE SHOT", origin="6th-floor window", target="Motorcade",
                                bearing="NE 042°", distance="81 m", elevation="27 m"),
    "suspect_profile_card": dict(name="Viktor Lindqvist", alias="The Cartographer", status="AT LARGE",
                                fields=[{"k": "Age", "v": "47"}, {"k": "Origin", "v": "Riga"}], case_no="0447"),
    "witness_testimony_card": dict(name="Anna Vogel", role="Embassy clerk", case_no="0447",
                                quote="He came every Thursday, always paid in cash."),
    "redacted_document": dict(title="SENATE REPORT", stamp="CLASSIFIED",
                                reveal="The committee found the company systematically crushed its competitors."),
    # documents / evidence / historical
    "headline_document_reveal": dict(headline="STANDARD OIL DECLARED A MONOPOLY", source="The New York Times · 1911", highlight="MONOPOLY"),
    "framed_evidence_spotlight": dict(caption="Ledger seized in the 1906 raid", tag="EXHIBIT A"),
    "headline_montage": dict(title="THE HEADLINES", headlines=["OIL KING'S EMPIRE", "TRUST ON TRIAL", "COURT ORDERS BREAKUP"]),
    "era_band_timeline": dict(title="THREE ERAS", eras=[{"label": "Rise", "from": "1865", "to": "1882"},
                                {"label": "Dominance", "from": "1882", "to": "1904"}, {"label": "Fall", "from": "1904", "to": "1911"}]),
    "era_stamp_overlay": dict(year="1911", place="NEW YORK", unit_label="REFINERIES", unit_count="40+"),
    "annotated_detail_callout": dict(label="Cut-glass seal", focus=[0.55, 0.45], tag="DETAIL"),
    # business / charts / numbers
    "composition_stack": dict(title="CONTROL OF FLOW", suffix="%", segments=[{"label": "Pipelines", "value": 40},
                                {"label": "Rail", "value": 25}, {"label": "Refineries", "value": 35}]),
    "growth_curve_chart": dict(title="REVENUE", prefix="$", suffix="M", points=[["1865", 1], ["1870", 10], ["1875", 40], ["1882", 90]]),
    "money_flow_empire": dict(center="Standard Oil", branches=[{"label": "Kerosene", "value": 50},
                                {"label": "Pipelines", "value": 30}, {"label": "Rail rebates", "value": 20}]),
    "pictograph_scale": dict(count=9, total=10, label="Refineries controlled"),
    "proportion_ring": dict(share=90, label="MARKET SHARE", suffix="%", center_sub="1880"),
    "ranked_list_countdown": dict(title="MARKET SHARE", suffix="%", items=[["Standard Oil", 90], ["Shell", 6], ["Others", 4]]),
    "sankey_flow": dict(title="WHERE THE MONEY WENT", total=100, source="Revenue",
                                branches=[{"label": "Reinvest", "value": 50}, {"label": "Dividends", "value": 30}, {"label": "Reserves", "value": 20}]),
    "statistic_bar_reveal": dict(title="SHARE OF FLOW", suffix="%", bars=[{"label": "Pipelines", "value": 40},
                                {"label": "Rail", "value": 25}, {"label": "Refineries", "value": 35}]),
    "wealth_arc_counter": dict(value=1_000_000_000, label="PERSONAL FORTUNE", prefix="$"),
    "acquisition_timeline": dict(parent="Standard Oil", tally_label="ABSORBED", tally_dollars="40+ firms",
                                targets=[{"name": "Acme", "year": "1872"}, {"name": "Vacuum", "year": "1879"}, {"name": "Galena", "year": "1886"}]),
    "supply_chain_network": dict(title="REFINING EMPIRE", stages=[{"label": "Crude"}, {"label": "Pipeline"}, {"label": "Refinery"}, {"label": "Market"}]),
    "gold_number_callout": dict(value=72500, label="STOLEN PER YEAR", prefix="$"),   # STEP 1 fixed shape
    "comparison_split": dict(left="Standard Oil", right="The Independents", leftval=90, rightval=10, suffix="%"),  # STEP 1 fixed shape
    "vs_balance_scale": dict(left="Monopoly", right="Competition", leftval=90, rightval=10, title="THE BALANCE OF POWER"),
    "countdown_clock": dict(value=9, unit="MINUTES", label="TIME TO ESCAPE"),
    "spectrum_meter": dict(value=72, label="SIGNAL", bands=["LOW", "GUARDED", "ELEVATED", "HIGH", "SEVERE"], readout="72 dB", title="INTERCEPT"),
    # tech / systems / hybrid
    "exploit_chain": dict(stages=["Recon", "Payload", "Foothold", "Exfiltrate"]),
    "packet_path_trace": dict(hops=["Client", "CDN", "Gateway", "Database"]),
    "system_planview_flow": dict(title="THE DATACENTRE", regions=[{"label": "Ingest"}, {"label": "Compute"}, {"label": "Store"}]),
    "footage_fact_overlay": dict(fact="90% of America's oil flowed through one company.", kicker="1880"),
    "footage_object_callout": dict(label="The hidden valve", point=[0.60, 0.50], note="Controlled the entire flow"),
    "footage_route_trace": dict(title="THE PIPELINE", points=[[0.20, 0.70], [0.50, 0.42], [0.80, 0.30]]),
    # biography / character
    "relationship_roster": dict(kicker="THE INNER CIRCLE", people=[{"name": "Henry Flagler", "role": "Partner"},
                                {"name": "Tom Scott", "role": "Rival"}, {"name": "William Rockefeller", "role": "Brother"}]),
    "cinematic_portrait_hold": dict(name="John D. Rockefeller", sub="1839 – 1937", side="left"),
    "portrait_legend_reveal": dict(name="John D. Rockefeller", kicker="THE TITAN"),
    "pull_quote_portrait": dict(name="John D. Rockefeller", quote="I would rather earn one percent off a hundred people's efforts."),
    "quote_stream": dict(title="IN HIS WORDS", quotes=["Competition is a sin.",
                                "The way to make money is to buy low.", "I had no ambition to make a fortune."]),
    "verdict_duality_card": dict(verdict="Both a ruthless monster and a generous genius.", pole_a="MONSTER", pole_b="GENIUS"),
    # science / engineering
    "labeled_cross_section": dict(title="ELECTRON MICROSCOPE", parts=["electron gun", "magnetic lenses", "specimen stage", "detector"]),
    "measurement_callout": dict(value="333 m", label="LENGTH", p0=[0.20, 0.62], p1=[0.80, 0.62]),
    "silhouette_scale_compare": dict(title="TRUE SCALE", items=[{"label": "Carrier", "size": 333, "note": "333 m"},
                                {"label": "Bus", "size": 12, "note": "12 m"}]),
    # typography / timeline / process
    "act_chapter_card": dict(title="THE TRUST", kicker="ACT II"),
    "kinetic_keyword": dict(keyword="MONOPOLY", context="One company. Total control."),
    "chronology_timeline": dict(title="STANDARD OIL", events=[["1865", "First refinery"],
                                ["1870", "Standard Oil founded"], ["1882", "The Trust"], ["1911", "Broken up"]]),
    "life_milestone_spine": dict(milestones=[{"year": "1839", "label": "Born"}, {"year": "1859", "label": "First business"},
                                {"year": "1870", "label": "Standard Oil"}, {"year": "1911", "label": "Dissolution"}]),
    "cause_effect_chain": dict(title="THE METHOD", steps=["Undercut price", "Starve the rival", "Force a sale", "Absorb the firm"]),
    "connection_web": dict(title="THE TRUST", nodes=[{"label": "Rockefeller"}, {"label": "Flagler"},
                                {"label": "Andrews"}, {"label": "Harkness"}], links=[[0, 1], [0, 2], [0, 3]]),
    "flowchart_decision": dict(title="THE ULTIMATUM", question="Sell or be crushed?",
                                yes="Absorbed at his price", no="Driven out of business", chosen="yes"),
    "org_hierarchy_tree": dict(title="STRUCTURE", root="Standard Oil Trust",
                                children=[{"label": "Domestic"}, {"label": "Export"}, {"label": "Pipelines"}]),
    "process_flow_steps": dict(title="THE PLAYBOOK", steps=["Survey", "Acquire", "Integrate", "Dominate"]),
    "definition_card": dict(term="MONOPOLY", definition="Exclusive control of a commodity or service in a market."),
    "statement_card": dict(text="He controlled ninety percent of a nation's oil.", emphasis="ninety percent", source="Senate · 1911"),
    "before_after_slider": dict(before_label="1865", after_label="1882", caption="The refinery district"),
    "spotlight_object_hold": dict(subject="The Ledger", kicker="EVIDENCE", sub="Seized 1906"),
}


def _filter_kwargs(fn, kw):
    """Keep only kwargs the render() actually accepts (prevents TypeErrors)."""
    params = set(inspect.signature(fn).parameters)
    return {k: v for k, v in kw.items() if k in params}


def _blackdetect(mp4):
    """Raw luma-only span count (over-sensitive on intentionally dark cards)."""
    r = subprocess.run([FF, "-i", str(mp4), "-vf", "blackdetect=d=0.05:pix_th=0.10",
                        "-an", "-f", "null", "-"], capture_output=True, text=True)
    return r.stderr.count("black_start")


def _black_qa(mp4, dur=DUR):
    """PRODUCTION-grade dead-black QA = editorial_qa's (mean<DEAD_LUMA AND
    std<DEAD_STD) — a frame is dead only when near-black AND featureless. Returns
    (dead_total, dead_after_entrance) where entrance = first 0.5s (masked by the
    assembler's incoming dissolve in real videos). dead_after_entrance>0 is the
    only real concern; entrance-window dips are house-style fade-ins."""
    import numpy as np                                          # noqa: F401
    from PIL import Image
    try:
        from vidlore import editorial_qa as EQ
        dl, ds = EQ.DEAD_LUMA, EQ.DEAD_STD
        stats = EQ.luma_stats
    except Exception:                                           # noqa: BLE001
        return (-1, -1)
    td = Path(tempfile.mkdtemp())
    dead_total = dead_after = 0
    # SINGLE ffmpeg pass: extract ~4 fps of frames at once (no per-frame spawn)
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp4),
                    "-vf", "fps=4", str(td / "q%03d.png")], capture_output=True)
    pngs = sorted(td.glob("q*.png"))
    for idx, png in enumerate(pngs):
        t = idx / 4.0                                           # fps=4 → 0.25s steps
        try:
            mean, std = stats(Image.open(png))
        except Exception:                                       # noqa: BLE001
            continue
        if mean < dl and std < ds:
            dead_total += 1
            if t > 0.5:
                dead_after += 1
    shutil.rmtree(td, ignore_errors=True)
    return (dead_total, dead_after)


def _frame(mp4, png, at=0.7):
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{DUR*at:.2f}",
                    "-i", str(mp4), "-frames:v", "1", str(png)], capture_output=True)


def render_reel(fam):
    pids = REELS[fam]
    reel_dir = OUT / fam
    reel_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = reel_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    records, card_mp4s = [], []
    for pid in pids:
        e = R.REGISTRY.get(pid)
        if not e:
            records.append({"pid": pid, "ok": False, "err": "not in registry"})
            continue
        fn = e["render"]
        kw = _filter_kwargs(fn, DEMO.get(pid, {}))
        mp4 = reel_dir / f"{pid}.mp4"
        rec = {"pid": pid, "family": e.get("family", "?")}
        try:
            res = fn(str(mp4), dur=DUR, fps=FPS, w=W, h=H, seed=7, **kw)
            rec["ok"] = bool(res.get("ok"))
            rec["fallback"] = bool(res.get("fallback", False))
            rec["render_s"] = res.get("render_s")
            rec["err"] = res.get("err", "")[:120]
        except Exception as ex:                                  # noqa: BLE001
            rec["ok"] = False
            rec["err"] = f"{type(ex).__name__}: {ex}"[:160]
        if rec.get("ok") and mp4.exists():
            rec["black_spans_raw"] = _blackdetect(mp4)          # luma-only (sensitive)
            dt, da = _black_qa(mp4)                             # production metric
            rec["dead_black_total"] = dt
            rec["dead_black_after_entrance"] = da               # >0 = real concern
            _frame(mp4, frames_dir / f"{pid}.png")
            card_mp4s.append(mp4)
        records.append(rec)
        print(f"  {fam}/{pid}: ok={rec['ok']} raw_black={rec.get('black_spans_raw','-')} "
              f"dead_post_entrance={rec.get('dead_black_after_entrance','-')} {rec.get('err','')[:50]}")
    # concat per-card mp4s -> single reel mp4, then DELETE per-card mp4s (disk)
    reel_mp4 = OUT / f"reel_{fam}.mp4"
    if card_mp4s:
        lst = reel_dir / "_concat.txt"
        lst.write_text("\n".join(f"file '{m}'" for m in card_mp4s))
        subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", str(lst), "-c", "copy", str(reel_mp4)], capture_output=True)
        lst.unlink(missing_ok=True)
        for m in card_mp4s:
            m.unlink(missing_ok=True)            # keep only the concatenated reel
    # contact sheets (desktop full-res frames; mobile = 360px-wide cards)
    _contact_sheet(frames_dir, OUT / f"sheet_{fam}_desktop.png", card_w=W)
    _contact_sheet(frames_dir, OUT / f"sheet_{fam}_mobile.png", card_w=360)
    (reel_dir / "records.json").write_text(json.dumps(records, indent=1))
    return records


def _contact_sheet(frames_dir, out_png, card_w):
    from PIL import Image
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        return
    cols = 3 if card_w >= 900 else (4 if card_w >= 350 else 5)
    cards = []
    for p in pngs:
        try:
            im = Image.open(p).convert("RGB")
            ch = int(im.height * card_w / im.width)
            cards.append(im.resize((card_w, ch), Image.LANCZOS))
        except Exception:                                       # noqa: BLE001
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
    fams = sys.argv[1:] if len(sys.argv) > 1 else list(REELS)
    allrec = {}
    for fam in fams:
        print(f"\n=== REEL: {fam} ({len(REELS[fam])} primitives) ===")
        allrec[fam] = render_reel(fam)
    (OUT / "all_records.json").write_text(json.dumps(allrec, indent=1))
    # summary
    flat = [r for rs in allrec.values() for r in rs]
    ok = sum(1 for r in flat if r.get("ok"))
    fb = sum(1 for r in flat if r.get("fallback"))
    dead = sum(1 for r in flat if r.get("dead_black_after_entrance", 0))   # real concern only
    raw = sum(1 for r in flat if r.get("black_spans_raw", 0))
    print(f"\n=== SUMMARY: {ok}/{len(flat)} ok | fallback={fb} | "
          f"dead-black-post-entrance={dead} | raw-luma-spans={raw} ===")
    for r in flat:
        if not r.get("ok") or r.get("dead_black_after_entrance", 0):
            print(f"  ATTN {r['pid']}: ok={r.get('ok')} dead_post_entrance="
                  f"{r.get('dead_black_after_entrance')} {r.get('err','')[:80]}")
