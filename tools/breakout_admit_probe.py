#!/usr/bin/env python3
"""Run the REAL breakout admission gate over a finished job's breakout candidates. No render.

    python3 tools/breakout_admit_probe.py <job_dir> [--fixture path.json]

The gate that ships is `build._breakout_window_admissible`. This calls it with exactly the inputs
the render would supply — the aired window's ASR text, the beat's narration, the beat's subject, the
beat's promised quote, and whether that quote actually LOCATED in the chosen source — and prints
what would air.

The pass condition is the last line: `SAFETY: no irrelevant breakout admitted`.

Why this exists: job benjen_v2 aired four Season-1 Cersei/Ned breakouts (Robert drinking, Jaime,
the Iron Throne) inside a Benjen Stark essay. They were genuine in-character dialogue, so the
dialogue-vs-narration gate passed them; nothing asked whether they belonged at their beat.
Costs ~3 text calls per candidate (~$0.007 for ten). Keep it OUT of CI — a provider outage must
never turn the suite red.
"""
import argparse, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".clipstudio_libs"))
sys.path.insert(0, str(ROOT))

from vidlore.config import _load_dotenv                          # noqa: E402
_load_dotenv(ROOT / ".env")

# beats whose aired audio a human judged OFF-TOPIC for this essay (benjen_v2 ground truth)
KNOWN_BAD = {44, 48, 96, 112, 156}
KNOWN_GOOD = {41, 70}


def build_fixture(job: Path) -> list:
    """Reconstruct each built breakout's (beat, aired text, beat narration) from the job itself."""
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio import index as I
    proj = ClipProject.load(job)
    segs = {s.index: s for s in proj.segments}
    aud = {}
    p = job / "output" / "work" / "breakout_audit.json"
    if p.exists():
        for e in (json.loads(p.read_text()).get("accepted") or []):
            aud[int(e.get("seg_index", -1))] = e
    rows = []
    for f in sorted((job / "output" / "work").glob("breakout_*.mp4")):
        idx = int(re.sub(r"\D", "", f.stem))
        seg = segs.get(idx)
        if seg is None:
            continue
        e = aud.get(idx) or {}
        sid = e.get("source_id") or ""
        quote = (getattr(seg, "quote", "") or "").strip()
        span = None
        if quote and sid:
            try:
                span = I.find_quote_span(I.load_words(proj, sid), quote)
            except Exception:                                    # noqa: BLE001
                span = None
        rows.append({
            "beat": idx, "sid": sid,
            "wtxt": e.get("aired_transcript") or "",
            "beat_text": getattr(seg, "text", "") or "",
            "required_kind": getattr(seg, "required_kind", "") or "",
            "required_entity": getattr(seg, "required_entity", "") or "",
            "beat_quote": quote,
            "quote_located": bool(span),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--fixture", default="")
    a = ap.parse_args()
    job = Path(a.job)

    if a.fixture and Path(a.fixture).exists():
        rows = json.loads(Path(a.fixture).read_text())
        for r in rows:
            r.setdefault("quote_located", bool(r.get("beat_quote")) and r.get("origin") == "verbatim_quote")
    else:
        rows = build_fixture(job)
    if not rows:
        raise SystemExit("no breakout candidates found — nothing to probe")

    from vidlore.clipstudio import build as B
    from vidlore.clipstudio import llm as L
    L.reset_usage()

    movie = "Game of Thrones"
    try:
        movie = (json.loads((job / "project.json").read_text())
                 .get("meta", {}).get("analysis", {}).get("movie_title") or movie)
    except Exception:                                            # noqa: BLE001
        pass

    admitted, rejected = [], []
    print(f"{'beat':>5}  {'quote?':>7}  {'verdict':<8} reason")
    print("-" * 100)
    for r in rows:
        wtxt = (r.get("wtxt") or "").strip()
        if not wtxt:
            print(f"{r['beat']:5d}  {'-':>7}  SKIP     no aired transcript recorded")
            continue
        ok, why, _ = B._breakout_window_admissible(
            wtxt, movie,
            beat_text=r.get("beat_text", ""),
            beat_subject=(f"{r.get('required_kind','')}: {r.get('required_entity','')}"
                          if r.get("required_entity") else ""),
            promised_quote=r.get("beat_quote", ""),
            quote_authored=bool(r.get("quote_located")))
        (admitted if ok else rejected).append(r["beat"])
        print(f"{r['beat']:5d}  {'located' if r.get('quote_located') else '-':>7}  "
              f"{'ADMIT' if ok else 'REJECT':<8} {why[:78]}")

    u = L.usage_summary()
    print(f"\nADMIT : {sorted(admitted)}")
    print(f"REJECT: {sorted(rejected)}")
    print(f"cost  : ${u['usd']:.4f} over {u['calls']} call(s)")

    bad_in = sorted(set(admitted) & KNOWN_BAD)
    good_out = sorted(KNOWN_GOOD - set(admitted))
    if bad_in:
        print(f"\nFAIL: irrelevant breakout(s) admitted: {bad_in}")
        return 1
    print("\nSAFETY: no irrelevant breakout admitted")
    if good_out:
        print(f"note: known-good beat(s) also dropped: {good_out} "
              f"(fewer breakouts is acceptable; zero irrelevant ones is not)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
