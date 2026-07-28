#!/usr/bin/env python3
"""Is a bad beat a PICKING failure or a POOL HOLE?

The audit tagged 135 beats `wrong_scene`. Two completely different fixes hide under that one label:
  - the right footage is in the pool and match chose wrong  -> better locator (Fix A)
  - the right footage was never downloaded                  -> targeted discovery (Fix B)
Building the locator first without knowing the split would be guessing again, so measure it.

For each beat we take its DISTINCTIVE terms (proper nouns and scene objects from expected_visual /
entities / required_entity, minus terms that are common across the whole pool) and ask whether any
source's title or ASR stream carries them. Title and speech are independent evidence, so either one
counts as "the pool plausibly has this".

    python3 pool_coverage.py --job <dir> [--audit <workflow output json>]
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/Users/hussnain/Desktop/vidlore-clipstudio/.clipstudio_libs")
sys.path.insert(0, "/Users/hussnain/Desktop/vidlore-clipstudio/.claude/worktrees/clipstudio-handover-review-113723")

STOP = set("""a an the and or but if then than that this these those of in on at to for with from by
as is are was were be been being it its his her their our your my he she they we you i him them us
who whom which what when where why how not no nor so such only own same too very can will just
about into over after before under again further once here there all any both each few more most
other some he's she's it's don't doesn't didn't isn't aren't wasn't weren't""".split())


def toks(s):
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']+", s or "")
            if len(w) > 2 and w.lower() not in STOP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--audit", default="")
    ap.add_argument("--max-beats", type=int, default=0)
    a = ap.parse_args()

    from vidlore.clipstudio.models import ClipProject
    from vidlore.clipstudio import index as I
    proj = ClipProject.load(Path(a.job))
    segs = {s.index: s for s in proj.segments}
    ok_srcs = [s for s in proj.sources if s.status == "ok"]

    # ---- pool term index: term -> {sid}, from titles and full word streams
    title_terms, asr_terms = defaultdict(set), defaultdict(set)
    for s in ok_srcs:
        for t in set(toks(s.title)):
            title_terms[t].add(s.id)
        try:
            w = I.load_words(proj, s.id)
        except Exception:
            w = []
        for t in {x.lower() for _, _, x in w if len(x) > 2}:
            t = re.sub(r"[^a-z']", "", t)
            if t and t not in STOP:
                asr_terms[t].add(s.id)
    n_src = len(ok_srcs)
    print(f"pool: {n_src} sources · {len(title_terms)} title terms · {len(asr_terms)} spoken terms\n")

    # a term carried by most of the pool localises nothing
    def distinctive(t):
        return len(title_terms.get(t, set())) + len(asr_terms.get(t, set())) <= max(3, n_src // 4)

    audit = {}
    if a.audit and os.path.exists(a.audit):
        d = json.load(open(a.audit))
        r = d.get("result", d)
        for b in (r.get("worst_beats") or []) + (r.get("upheld_criticals") or []):
            i = b.get("beat")
            if i is not None:
                audit[i] = b

    rows = []
    for i, seg in sorted(segs.items()):
        want = " ".join([getattr(seg, "expected_visual", "") or "",
                         getattr(seg, "required_entity", "") or "",
                         " ".join(getattr(seg, "entities", []) or [])])
        terms = [t for t in set(toks(want)) if distinctive(t)]
        if not terms:
            rows.append((i, "no-distinctive-terms", 0, 0, []))
            continue
        hit_src = Counter()
        for t in terms:
            for sid in title_terms.get(t, set()) | asr_terms.get(t, set()):
                hit_src[sid] += 1
        best = hit_src.most_common(3)
        cover = (best[0][1] / len(terms)) if best else 0.0
        status = ("pool-has-it" if cover >= 0.5 else
                  "weak" if cover > 0 else "POOL HOLE")
        rows.append((i, status, len(terms), round(cover, 2), best))

    cnt = Counter(r[1] for r in rows)
    print("ALL 268 BEATS")
    for k, v in cnt.most_common():
        print(f"  {v:4d}  {k}")

    if audit:
        bad = [r for r in rows if r[0] in audit]
        print(f"\nTHE {len(bad)} BEATS THE AUDIT FLAGGED WORST")
        c2 = Counter(r[1] for r in bad)
        for k, v in c2.most_common():
            print(f"  {v:4d}  {k}")
        print("\n  beat  status              terms  cover  best source")
        for i, st, nt, cov, best in sorted(bad, key=lambda r: (r[1], r[0]))[:40]:
            b = best[0][0][:34] if best else "-"
            print(f"  {i:4d}  {st:18s}  {nt:4d}  {cov:5.2f}  {b}")

    json.dump([{"beat": r[0], "status": r[1], "terms": r[2], "cover": r[3],
                "best": [[s, n] for s, n in r[4]]} for r in rows],
              open(os.path.join(os.path.dirname(a.audit or ".") or ".", "pool_coverage.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
