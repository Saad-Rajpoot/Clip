#!/usr/bin/env python3
"""A/B the INTRO of a job — match + verify per arm, then score the picked footage with vision.

Built for the Night-King failure: job 5cab63d801's opening beats narrate the Night King catching
Arya by the throat, and the render showed an S7E4 reunion instead. Measuring that needs three
things this harness keeps together — the SAME pipeline stages in both arms (three earlier
comparisons in this project were invalidated by comparing match-only against verified), the SAME
beats, and a judge that looks at the frames the viewer would actually see rather than at a proxy
like "did the pick change".

Verify runs on the intro beats only: their picks have no earlier beats to inherit anti-reuse from,
so the subset is faithful, and a full 288-beat verify would cost ~$1 per arm to answer a question
about beat 0.

    python3 intro_ab.py <job> <out> <off-env-spec> <on-env-spec> [--beats N]

Emits <out>/{off,on}/ frame trees, scores.json, and a per-beat delta table.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAIN = "/Users/hussnain/Desktop/vidlore-clipstudio"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, f"{MAIN}/.clipstudio_libs")
sys.path.insert(0, MAIN)

from vidlore.config import _load_dotenv                                  # noqa: E402
_load_dotenv(Path(MAIN) / ".env")

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


# The editor does not ask the same thing of every beat, so neither can the judge. Grading a
# connective line ("It issued four of them") by "is the exact moment on screen" scored a perfectly
# reasonable snowy-mountain cutaway 0/10 three votes out of three, which measures the rubric, not
# the pipeline. Each beat carries the policy match chose it under; grade against that contract.
JUDGE_SYS = (
    "You grade a video essay's footage against its narration. You see 3 frames sampled across the "
    "clip that airs under one line, plus the EDITORIAL CONTRACT that clip was chosen under.\n\n"
    "exact_scene — the line describes a specific moment:\n"
    "  9-10 that exact moment is on screen   7-8 right scene, not the precise instant\n"
    "  5-6  right character or era, wrong scene   3-4 same show, unrelated scene\n"
    "character_specific — the line is about a named person:\n"
    "  9-10 that person clearly on screen, fitting moment   7-8 that person on screen\n"
    "  5-6  right show, person unclear or absent   3-4 a different named character\n"
    "abstract_effect / generic_filler — a connective or thematic line with no scene to show:\n"
    "  9-10 apt, watchable footage from the right show that suits the line's mood or subject\n"
    "  7-8  reasonable on-show footage that does not fight the line\n"
    "  5-6  on-show but jarring or contradicts what is being said\n"
    "  3-4  off-topic enough to confuse a viewer\n"
    "ALL contracts, overriding the bands above: 0-2 for wrong show, a graphic or text card, a "
    "talking-head reactor, black, a flat featureless frame, or anything unreadable.\n"
    "Judge ONLY what is visible; do not reward or punish resolution. Reply with JSON: "
    '{"score": <int>, "shows": "<=12 words on what is actually on screen", '
    '"why": "<=20 words"}'
)


def judge(row: dict, votes: int = 3) -> dict:
    """Median of N independent reads.

    A single read is too noisy to tune on: re-judging IDENTICAL frames measured mean |Δ| 0.50 with
    swings up to 3 points, which is the same size as the effects under test. The median of three
    kills the outlier read without pretending the judge is deterministic."""
    from vidlore.clipstudio import llm
    frames = [f for f in row.get("frames") or [] if os.path.exists(f)]
    if not frames:
        return {"beat": row["beat"], "score": 0, "shows": "no footage", "why": "no frames"}
    content = []
    for f in frames:
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(Path(f).read_bytes()).decode()}})
    want = row.get("expected_visual") or ""
    content.append({"type": "text", "text":
                    f"EDITORIAL CONTRACT: {row.get('visual_policy') or 'generic_filler'}\n"
                    f"NARRATION heard over this clip: {row.get('narration','')!r}\n"
                    + (f"The script's own expected visual: {want!r}\n" if want else "")
                    + "Score the footage."})
    got, shows = [], ""
    for _ in range(votes):
        try:
            txt = llm.complete(system=JUDGE_SYS, messages=[{"role": "user", "content": content}],
                               max_tokens=300)
            d = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])
            got.append(int(d.get("score", 0)))
            shows = shows or str(d.get("shows", ""))[:70]
        except Exception as e:
            shows = shows or f"judge error: {e}"[:70]
    if not got:
        return {"beat": row["beat"], "score": -1, "shows": shows, "why": "all reads failed"}
    got.sort()
    return {"beat": row["beat"], "score": got[len(got) // 2], "shows": shows,
            "why": f"votes {got}", "spread": got[-1] - got[0]}


def run_arm(job: Path, spec: str, nbeats: int, cache0: Path = None) -> dict:
    """match ALL beats (CPU only), verify the intro subset, return {beat: (sid, in, out)}.

    The verdict cache is restored to the SAME baseline first. Without that the second arm reads
    verdicts the first arm just paid for, which changes which shots the prefetch waves reach — an
    arm-order confound, not a property of the change under test."""
    vc = job / "verdict_cache.json"
    if cache0 is not None:
        if cache0.exists():
            shutil.copy2(cache0, vc)
        elif vc.exists():
            vc.unlink()
    for kv in spec.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k.strip()] = v.strip()
    for m in [m for m in list(sys.modules) if m.startswith("vidlore")]:
        del sys.modules[m]
    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio.config import ClipConfig, engine_config
    from vidlore.clipstudio.match import match_segments
    from vidlore.clipstudio import verify as V
    from vidlore.clipstudio import llm

    llm.reset_usage()
    proj = ClipProject.load(job)
    cfg, eng = ClipConfig(), engine_config()
    segs = list(proj.segments)
    proj.selections = match_segments(proj, segs, cfg, progress=None)
    intro = [s for s in segs if s.index < nbeats]
    V.verify_and_repair(proj, intro, cfg, eng, progress=None)
    proj.save()
    u = llm.usage_summary()
    log(f"  arm [{spec}] → ${u['usd']:.2f} / {u['calls']} call(s)")
    return {s.segment_index: (s.source_id, round(s.in_point, 2), round(s.out_point, 2))
            for s in proj.selections if s.segment_index < nbeats}


def prep(job: Path, dest: Path, nbeats: int) -> list:
    r = subprocess.run([sys.executable, str(HERE / "prep_eval.py"), "--job", str(job),
                        "--out", str(dest), "--beats", f"0-{nbeats-1}"],
                       capture_output=True, text=True)
    if r.returncode:
        log(f"prep FAILED: {r.stdout[-300:]}{r.stderr[-300:]}")
        sys.exit(2)
    return json.loads((dest / "eval_manifest.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("out")
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("--beats", type=int, default=20)
    a = ap.parse_args()
    job, out = Path(a.job), Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    saved = out / "project_original.json"
    shutil.copy2(job / "project.json", saved)
    cache0 = out / "verdict_cache_baseline.json"
    if (job / "verdict_cache.json").exists():
        shutil.copy2(job / "verdict_cache.json", cache0)

    arms = {}
    try:
        for tag, spec in (("off", a.off), ("on", a.on)):
            log(f"run {tag.upper()} ({spec})")
            arms[tag] = run_arm(job, spec, a.beats, cache0)
            shutil.copy2(job / "project.json", out / f"project_{tag}.json")

        changed = sorted(i for i in arms["off"] if arms["off"][i] != arms["on"].get(i))
        log(f"intro beats changed: {len(changed)}/{a.beats} → {changed}")

        rows = {}
        for tag in ("off", "on"):
            shutil.copy2(out / f"project_{tag}.json", job / "project.json")
            rows[tag] = {r["beat"]: r for r in prep(job, out / tag, a.beats)}
            log(f"{tag}: {len(rows[tag])} beat(s) framed")
    finally:
        shutil.copy2(saved, job / "project.json")

    scores = {}
    for tag in ("off", "on"):
        with ThreadPoolExecutor(max_workers=6) as ex:
            got = list(ex.map(judge, [rows[tag][b] for b in sorted(rows[tag])]))
        scores[tag] = {g["beat"]: g for g in got}
        ok = [g["score"] for g in got if g["score"] >= 0]
        log(f"{tag}: mean {sum(ok)/max(1,len(ok)):.2f} over {len(ok)} beat(s)")

    (out / "scores.json").write_text(json.dumps(
        {"changed": changed, "off": scores["off"], "on": scores["on"]}, indent=1))

    print(f"\n{'beat':>4} {'off':>4} {'on':>4} {'Δ':>4}  on-shows")
    for b in sorted(rows["on"]):
        o, n = scores["off"].get(b, {}), scores["on"].get(b, {})
        mark = "*" if b in changed else " "
        print(f"{b:>4}{mark}{o.get('score',0):>4}{n.get('score',0):>4}"
              f"{n.get('score',0)-o.get('score',0):>4}  {n.get('shows','')[:52]}")
    for tag in ("off", "on"):
        ok = [scores[tag][b]["score"] for b in changed if scores[tag].get(b, {}).get("score", -1) >= 0]
        if ok:
            log(f"CHANGED-only {tag}: mean {sum(ok)/len(ok):.2f} over {len(ok)} beat(s)")
    pol = {b: (rows["on"][b].get("visual_policy") or "generic_filler") for b in rows["on"]}
    print()
    for tag in ("off", "on"):
        by = {}
        for b, g in scores[tag].items():
            if g["score"] >= 0:
                by.setdefault(pol.get(b, "?"), []).append(g["score"])
        parts = [f"{k} {sum(v)/len(v):.1f} ({sum(1 for s in v if s >= 7)}/{len(v)})"
                 for k, v in sorted(by.items())]
        log(f"{tag} by policy: " + " | ".join(parts))
    for tag in ("off", "on"):
        allx = [g["score"] for g in scores[tag].values() if g["score"] >= 0]
        ex = [scores[tag][b]["score"] for b in scores[tag]
              if pol.get(b) == "exact_scene" and scores[tag][b]["score"] >= 0]
        log(f"{tag}: INTRO acc {sum(1 for s in allx if s>=7)}/{len(allx)} = "
            f"{100*sum(1 for s in allx if s>=7)/max(1,len(allx)):.0f}%  |  "
            f"EXACT-SCENE acc {sum(1 for s in ex if s>=7)}/{len(ex)} = "
            f"{100*sum(1 for s in ex if s>=7)/max(1,len(ex)):.0f}% (mean {sum(ex)/max(1,len(ex)):.2f})")


if __name__ == "__main__":
    main()
