# ClipStudio — Relevance Work Handover

**Session:** 2026-07-26 → 2026-07-28
**Repo:** `/Users/hussnain/Desktop/vidlore-clipstudio`
**Status:** all work **committed and merged to `main`** (`f5e9f99`). Nothing is left uncommitted.
**Owner:** Hussnain. Communicates in Hinglish/Urdu; reply in the same.

Read this before touching anything. Roughly half of what looked like an obvious fix in this session
turned out, on measurement, to be wrong or already implemented. The measurements are the valuable
part of this document — not the code.

---

## 1. What the work was about

ClipStudio auto-builds YouTube video essays from downloaded YouTube footage. The owner's complaint,
repeated across the whole session and the thing to optimise for:

> "Jab VO mein dagger (khanjar) dekhne ko kaha jata hai, toh screen par wohi aa raha hai kya? Jab
> chehre ke expressions dekhne ko kahe jayein, to wo hi aa raha ho."

And his rule for everything else:

> "Agar koi exact scene dastyab na ho, to wo uski zid na kare aur kisi miltay jultay (generic) scene
> par guzara kar le. Exact scene ka screenshot bhi use kar sakta ho — but zyada nahi."

Working job throughout: **`69d80e9dd4`** — a 22-minute Game of Thrones essay about Petyr Baelish's
trial ("Petyr Baelish won his trial - watch it as a lawyer"), 268 beats, 84–87 sources.

---

## 2. The single most important thing in this document

**Do not decide to re-render on proxy metrics.** Three full renders (v2, v3, v4) were launched in
this session on signals like "the right source was picked", "moment_bonus fired", "shot repeats
dropped". None of those measure whether the moment the voiceover describes is on screen. The
frame-level audit then scored v4 at **5.18/10** and the owner was — correctly — angry.

The replacement is a **no-render relevance eval** built during this session:

- `scratchpad/eval/prep_eval.py` extracts 3 frames (15%/50%/85%) from each beat's **selected window**
  straight out of the source file, mirroring build's watermark crop so it judges what the viewer
  actually sees.
- A workflow scores them with vision against that beat's own narration.
- **Validated: 4.96 against the full frame audit's 5.18, in 8 minutes instead of 1.7 hours.**

Scratchpad path (session-scoped, copy it somewhere permanent if you want to keep it):
`/private/tmp/claude-501/-Users-hussnain-Desktop-vidlore-clipstudio--claude-worktrees-video-audit-quality-569ca6/f88cb459-7abc-4c52-874a-2e9d5cb71b98/scratchpad/eval/`

| harness | what it does |
|---|---|
| `prep_eval.py` | builds the eval frames + slices for a job (crop-aware) |
| `gate_ab.py` | **A/B any env-flag set** — match+verify twice, same job, evaluates only the beats whose pick differs |
| `bench_ab.py` | the earlier, deep-bench-specific version of the same idea |
| `rematch.py` | re-run match only |
| `match_verify.py` | run match+verify and report deep-bench rescues + cost |
| `reindex_faceid.py` | re-index specific sources WITH Face-ID |
| `pool_coverage.py` | is a bad beat a picking failure or a pool hole? |
| `try_backfill.py` | run the backfill pass standalone against a real pool |

### Three measurement mistakes that were made and caught — do not repeat them

1. **Comparing different pipeline stages.** v5 (match only) scored 4.45 against v4 (match + verify +
   recover) 4.96 and read as a regression. The difference was verify, not the change under test.
   Verify is worth **+0.72** on its own. Always build a same-stage control.
2. **Workflow `args` arriving as a JSON string.** The eval script silently fell back to its default
   directory and **re-scored the old job under a new label**. The script now throws without an
   explicit `dir`.
3. **Judging pre-crop frames.** The eval read raw source frames and reported watermarks on beats
   where build already crops the logo away — 18 flags against the rendered audit's 8. Fixed;
   `prep_eval.py` now mirrors `build._watermark_crop_filter`.

Scorer run-to-run variance is about **±0.05 on a group mean and ±3 on a single beat**. Never claim a
result from one beat.

---

## 3. Renders that exist on disk

`/Users/hussnain/Desktop/clipstudio_output/portal/`

| job | duration | size | state |
|---|---|---|---|
| `69d80e9dd4` | 22:11 | 507 MB | original, audited **4.6/10** |
| `69d80e9dd4_v2` | 22:23 | 541 MB | first fix pass, has narration + no music |
| `69d80e9dd4_v3` | 20:18 | 513 MB | **BROKEN — no voiceover at all**, silent-narration fallback |
| `69d80e9dd4_v4` | 22:05 | 521 MB | last good render; audited **4.5/10 (mean relevance 5.18)** |
| `69d80e9dd4_v5` | — | — | scratch job used for A/B experiments (match+verify only, never built) |
| `69d80e9dd4_v4m` | — | — | match-only control for the same purpose |

`_v5` and `_v4m` are experiment scratch, not deliverables. Their `sources/` and `index/` are
symlinks into `_v2` — do not delete `_v2`.

---

## 4. What was committed (all on `main`, `f5e9f99`)

Suites on main: **295 pytest · 834 `tools/test_clipstudio_fixes.py` · 41 graphics-gate · 10 source-ban — all green, zero failures.**
(Inside the worktree those 7 cold-open tests fail; that is worktree-specific, they pass on main.)

### 4.1 Backfill for gate-rejected sources — `orchestrate._backfill_rejected_sources`, stage 5b

**Problem found by measurement, not assumed.** 11 of 84 sources were dropped at match time by the
pool gates. Frames were inspected by hand: **every rejection was correct** — 7 burned-caption
re-uploads, 2 promo-card compilations, 1 STAR India screener with "FOR INTERNAL VIEWING ONLY" and a
burned timecode, 1 talking-head. Among the casualties were the most on-topic upload in the whole
pool ("The Trial of Petyr Baelish", 1080p, 23 min, 215 shots) and the only clip of the dagger
handover. 207 beats then asked for the trial and got scene packs of neighbouring scenes.

The defect was that **nothing replaced them**. Discovery spends its budget once, then match silently
subtracts 13% of the pool.

Now: rejections carry a **reason** (`proj.meta['auto_rejected_reasons']`), only *quality* rejects
(subtitled / watermarked / promo) are chased — not *content* rejects (interview, reaction, wrong
show) — the lost upload's own title is the search query, replacements must prove a **usable shot
yield** (`match.usable_shot_yield`), and the pass refuses to run without Face-ID refs.

**Measured: +0.16 video-wide** — real but near the scorer's noise (41 beats better, 36 worse).

Live run found 3 clean sources including "Littlefinger gives Catspaw dagger to Bran Stark" — which
then turned out to be **another screener** with burned text, 0 of 11 shots usable. That is what
prompted the yield check.

Kill switch: `VIDLORE_CLIPSTUDIO_BACKFILL_REJECTED=0`, rounds `..._BACKFILL_ROUNDS`.

### 4.2 Deep bench — `match.py` + `verify.py`

**The strongest measured finding of the session.** Cross-referencing the v4 render log against the
frame eval:

```
verify REPLACED a pick        38 beats   5.92   <- what actually works
verify left it alone         117 beats   5.25
verify DOWNGRADED to
  contextual_fallback        113 beats   4.34   <- 42% of the video, the worst group
```

Verify is worth **+0.72** over raw match — more than any match-side fix attempted. The leak is the
downgrade path: "the required subject is on screen, so keep the clip and relabel it
contextual_fallback". It only ever saw **6 alternates out of a ~4000-shot pool**.

`ClipSelection.deep_alternates` now carries ranked candidates 7–20; the verifier tries them at the
**strict** bar before settling.

**Measured, confound-free A/B on the 10 beats it changes: 4.00 → 6.10 (+2.10).** Deictic misses
1 → 0. Video-wide ≈ +0.078.

Two gotchas each cost a full measurement round:
- The ranked list was truncated at **build** time, so the bench was always empty — `deep bench: mean
  0.0 candidates/beat`, 0 rescues over a full 268-beat verify. The fix and its regression test are
  in `test_deep_bench_rescue.py`.
- The bench must sit down the `elif not _deep_bench():` branch. Trying it on a pick that already
  passes `_contextual_subject_ok` swapped good clips for strict-passing worse ones (two beats
  scoring 9 fell to 4 and 5).

Kill switch: `VIDLORE_CLIPSTUDIO_DEEP_BENCH=0`, depth `..._DEEP_ALTERNATES` (default 20).

### 4.3 Look gate — `policy.deictic_target` + `verify.py`

Directly answers the owner's repeated ask. `policy.is_deictic` already existed but answers a
different question (it spots "that table" and only decides policy); it cannot say **what** to look
for.

`deictic_target(seg)` extracts the noun phrase — on the real script it finds:

| beat | line | target |
|---|---|---|
| 13 | "Keep your eye on the dagger in that room" | `the dagger` |
| 14 | "keep your eye on Bran's face while his sister reads" | `Bran's face` |
| 31 | "watch the trial the way Bran watched it" | `the trial` |
| 164 | "Count the chairs" | `the chairs` |
| 246 | "Watch Bran's face while his sister reads out the charges" | `Bran's face` |

and correctly rejects "notice the division of labour", "watch his strategy", "that is the tragedy".

**This gate went through three failed iterations. Read this part before changing it.**

- v1 blocked the contextual downgrade unconditionally when the target was missing. **Measured
  −4.00**: beats scoring 8/9/9 fell to 4/2/3, and the deictic target ended up *less* visible. The
  replacement was worse than what it discarded.
- v2 tried "search, then settle". **No effect** — the ON picks were byte-identical, caught by
  comparing picks before spending eval tokens.
- v3 kept a usable pick instead of gambling it. **Still no effect**, same reason.
- The actual mechanism was found last: `must_see` was passed on **every** verification, including
  alternates, so the *strict promotion* — which runs before all the edited code — accepted a
  different, worse alternate. It is now scoped to the current pick only (`_look_scope`).

**Final state: 0 beats changed — the gate no longer touches footage at all.** Its entire remaining
value is that a missed target flags `look_target_missing`, which routes the beat to the **still
pass** — a frame of that moment instead of moving footage of the wrong one. That still-routing is
wired and unit-tested but its relevance benefit is **unmeasured**, because the still pass runs in
build and is not in the A/B harness.

The lesson, which is also the owner's own rule: **when the footage does not exist, searching harder
only finds a worse clip.**

Kill switch: `VIDLORE_CLIPSTUDIO_LOOK_GATE=0`.

### 4.4 Era on the contextual fallback — `verify._contextual_subject_ok`

The verifier is told a clearly different season is wrong even when the character matches, but that
only drove `verdict`; the fallback then re-admitted the clip because the subject was visible. That
is how **season-1 child Bran shipped under season-8 Dragonpit lines** (16–18 `wrong_era` beats).

Now reads `era_ok is not False`, so verdicts cached before the field existed still pass and only
fresh ones tighten — avoids invalidating a whole render's verdict cache (~$1 of vision calls).

**Not independently measured** — a warm verdict cache carries no `era_ok`, so it cannot show its
effect without a cold re-verify.

Kill switch: `VIDLORE_CLIPSTUDIO_ERA_FALLBACK_GATE=0`.

### 4.5 Voiceover can never silently vanish — `build.py`

**v3 shipped 20 minutes of video with captions and music and zero narration.** Chain of silent
skips: `rerender_v2` pointed at the original job's voiceover without copying it; `rerender_v3` copied
from v2 behind an `if exists` guard that no-opped; `build_video` fell through to `narration N s
(silent fallback)`. The log said so; it was not read.

`build_video` now raises `FileNotFoundError` on a named-but-missing voiceover and `RuntimeError`
when it cannot be aligned and `use_tts=False`. The silent fallback survives only for the genuine
`voiceover=None` TTS-outage case. Tests: `tests/test_voiceover_never_silent.py`.

### 4.6 Cost accounting — `llm.py` + `orchestrate.py`

Hundreds of vision calls per render were recorded nowhere. Every Gemini/Claude/DeepSeek call now
books its tokens; a render prints a `cost:` line and writes `output/cost_report.json`.

**Measured: 878 in / 98 out per verifier call = $0.0005. v4 (warm cache) 512 calls = $0.26. A cold
render ≈ $1. Vision runs on `gemini-2.5-flash`.** Prices are env-overridable
(`VIDLORE_CLIPSTUDIO_PRICE_<MODEL>_IN/_OUT`) and are estimates, not an invoice.

---

## 5. Things measured and DELIBERATELY NOT SHIPPED

Do not re-implement these without new evidence.

### 5.1 Uncroppable-overlay / centre-frame watermark detector — two calibrations failed

Three sources carry furniture no crop can remove: a fan music-video edit with a translucent "TM" at
frame centre (`petyr_baelish_littlefi_30b5ed70`, "|| Power over me"), a scene-finder upload with
"1X03 / 19:48" top-left **and** "scene seekers" bottom-right (opposing corners), and a Kingslayer
upload. `watermark_mode` defaults to `crop`, which assumes one corner.

- Attempt 1 (edge-persistence on an 8×5 grid): **0/3 caught, 9 false positives.**
- Attempt 2 (temporal per-pixel std + structure in the mean image): localises correctly — TM at
  cx 0.52 / cy 0.91, the scene-finder marker at cx 0.12 / cy 0.13 — but at 320×180 keyframe
  resolution the magnitude does not separate from clean sources (guilty ranked 8th, 9th, last).

The code was **removed rather than shipped broken**. If you retry: use full-resolution keyframes
with connected-component analysis on the frozen-pixel mask, require the same component box across
shots rather than a global percentage, and calibrate against the 87-source pool. Bar: 3/3 with 0
false positives.

**Important scope reduction:** corner watermarks are **already handled** — build punch-in-crops them
and 12 of 84 sources were cropped in v4. The owner's instruction ("corner ma watermark hai to blur ya
crop kar do") is already the shipped behaviour. Only centre-frame marks remain, and those would need
a blur path, which does not exist.

### 5.2 Blanket luma floor — rejected by data

The audit said "87 of 268 picks under luma 20, beyond what the candle-lit trial justifies". Measured
the pool: **median luma 31.9, and 31.1% of its 4688 shots sit under luma 20.** The picks (32%)
mirror the pool exactly. A floor at 20 would delete a third of the pool and push beats *away* from
the trial the essay is about. Darkness that actually hides the subject is already covered by
`_shot_unreadable`, the moment-lock legibility damper, and the look gate.

Pinned by `test_no_blanket_luma_floor_was_added` so nobody adds it back on plausibility.

### 5.3 `_anchor_echo` — proved dead by a 19-agent adversarial sweep

The anchor-line fallback for moment-lock resolves **0 of the 183 quote-less beats** it was written
for, and a with/without differential across 22,780 (beat × source) pairs found it changed **zero**
decisions. Its gate (`hits >= 2 and cov >= 0.6`) is unreachable: the best coverage any quote-less
beat achieves is 0.286. Left in place (harmless) but do not count on it.

### 5.4 Release-block — a usage rule, not a bug

The 17 beats that shipped with `RELEASE_BLOCK_verifier_rejected_no_fallback` did so because the
render scripts set `VIDLORE_CLIPSTUDIO_RELEASE_BLOCK_MODE=warn`. Correct for a review draft, wrong
for a final render. **Final renders must not set that override.**

---

## 6. The conclusion the evidence keeps pointing at

Every measured fix lands between +0.08 and +0.16 video-wide. v4 sits at 5.18; publishable is ~7.
**That gap will not close with more picker cleverness.**

Three independent investigations reached the same place:

1. Backfill: the best sources were downloaded and correctly rejected as unusable; replacements were
   themselves screeners.
2. Deep bench: 20 candidates deep, only 11 of ~101 downgrade beats found an exact-scene rescue.
3. Look gate: searching harder for the named target produced strictly worse clips.

**The ceiling on this job is footage availability.** 207 of 272 beats are `exact_scene` — the script
promises a specific moment three-quarters of the time. The pool has no clean copy of:
Bloodraven's cave / Three-Eyed Raven (beats 196–212, the worst continuous 90 seconds), the Dragonpit
council, King's Landing burning (beats 265–270, the ending), and Baelish's dagger at Ned's throat.

The frame audit's own ranking put **targeted acquisition of those 4 clusters at ≈ +0.47** — more
than everything shipped in this session combined. That is the recommended next move, not more
matching heuristics.

The second lever is upstream: stop the analyzer writing `exact_scene` cheques the footage cannot
cash. A beat that demands an unobtainable moment scores 1–3; the same beat as `generic_filler`
scores 5–6.

---

## 7. State of the task list

Completed: eval loop, backfill, deep bench, look gate, era gate, cost accounting, voiceover guard,
luma-floor decision.

Still open:
- **Fix A** — moment-lock for the 183 quote-less beats. The biggest structural gap. `_anchor_echo`
  was supposed to cover it and is dead (§5.3). Would need locking on narration entity+action rather
  than a quotable line.
- **Fix E** — confidence calibration. Mean match confidence is **0.849** against measured relevance
  **0.518**. The scorer is confidently wrong, which is exactly why it masked three bad renders.
  Confidence cannot currently be trusted as a release signal.
- **Fix F** — centre-frame watermark (§5.1).
- **Full validation → short render → full render.** No render has been made since v4; none of the
  committed changes has been seen end-to-end in a video.

---

## 8. Operational notes for whoever continues

- **Restart the portal** after pulling `main`, or it keeps running the old code. It was not running
  at handover time.
- **Never render from the git worktree.** Music mp3s are gitignored, so a worktree render ships with
  **no music bed** and says nothing about it — v2, v3 and v4 all shipped silent-music for this
  reason. Symlink the 118 files in first and verify `musiclib.scan()` returns 11 categories / 118
  tracks.
- **Always probe the finished file's audio** before calling a render done. Grep the build log for
  `silent fallback`, `align failed`, `degraded`.
- HD downloads need `.hdvenv` (Python 3.11) + the `.pot` server; from a worktree, set
  `VIDLORE_HD_PYTHON` / `VIDLORE_HD_POT_DIR`. The portal resolves these itself from the main
  checkout.
- Portal `max_sources` cap is 96. Long essays benefit from a high budget.
- Cost of this session: pipeline side ≈ $2 of vision calls. The audits and evals cost roughly
  **19M Opus tokens** — that is the real expense. Scope evals to affected beats (`gate_ab.py` does
  this) and keep the full 268-beat run for pre-render decisions only.

## 9. Memory files written (auto-loaded next session)

`~/.claude/projects/-Users-hussnain-Desktop-vidlore-clipstudio/memory/`

- `clipstudio-relevance-eval-loop.md` — measure before rendering; the three confound mistakes
- `clipstudio-verify-deep-bench.md` — verify is the strongest lever; the bench/look/era changes
- `clipstudio-render-output-verification.md` — always probe the output's audio
- `clipstudio-worktree-renders-lose-media.md` — worktree renders lose music
