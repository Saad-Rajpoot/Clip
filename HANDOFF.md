# ClipStudio — Complete Handoff

> **Hussnain ke liye (Urdu/Hinglish):** yeh file naye Claude account ko denay ke liye hai. Ismein
> woh sab kuch hai jo tool ke baare mein jaanna zaroori hai — code ki structure, humne ab tak kya
> kya fix kiya, kaunsi cheezein naap kar radd ki gayin (taake koi dobara na azmaye), machine par kya
> setup chahiye jo git nahi le kar ja sakta, aur abhi kya kaam baqi hai. Naye account mein bas kaho:
> *"read HANDOFF.md first"*.

**Written:** 2026-08-02 · **Repo:** `/Users/hussnain/Desktop/vidlore-clipstudio` · **Branch:** `main`
(clean, 209 commits) · **Suite:** 675 passed / 6 skipped · **Owner:** Hussnain — writes in
Hinglish/Urdu, **reply in the same language**.

---

## 1. What this is

ClipStudio auto-builds long-form YouTube video essays. You give it a **script** (or a topic) and a
**voiceover**; it discovers source videos on YouTube, downloads them, indexes them (shots, CLIP
embeddings, ASR, Face-ID, OCR), matches every script beat to a shot, verifies the match with a
vision model, cuts, and assembles a captioned, scored 1080p video.

**Run it:** the owner starts the web portal himself (`ClipStudio-Portal.command` on macOS,
`run-windows.bat` on Windows). **Do not start the portal yourself** — this is a standing
instruction. To render without the portal, write a small driver in `tools/` (see
`tools/rerender_benjen.py` for the canonical pattern) and run it directly.

**Pipeline stages** (`vidlore/clipstudio/orchestrate.py::produce_auto`, 9 stages):
`analyze → discover → download → face-ID refs → index → match → cut → verify+repair → build`.
Checkpointed: `produce_auto(resume=True)` skips completed stages (`meta["pipeline"]` v2).

---

## 2. Machine setup that git CANNOT carry — read this before debugging anything

These are `.gitignore`d, so a fresh clone is NOT runnable:

| Thing | Where | Why it matters |
|---|---|---|
| `.env` | repo root | real API keys (DeepSeek, Gemini, Anthropic). **Never paste its contents anywhere.** |
| CLIP models | `models/clip` (exact name) | no CLIP → no matching |
| Face-ID models | see `hd_download`/Face-ID setup | no Face-ID → identity gates go blind |
| 118 music mp3s | `vidlore/assets/music/` | `musiclib.scan()` must return **11 categories / 118 tracks** or the build raises at the end |
| `.hdvenv` | repo root | Python-3.11 venv holding a RECENT yt-dlp; the engine runs 3.9 |
| `.pot/server` | repo root | PO-token server (needs `deno`) |

**Renders from a git worktree lose media** — the gitignored mp3s are absent, `_resolve_music`
returns None silently. Always render from the MAIN repo, or symlink the 118 files in first and
verify `musiclib.scan()`.

---

## 3. Code map

| File | Lines | What |
|---|---|---|
| `vidlore/footage.py` | 28.8k | legacy footage engine (older pipeline) |
| `vidlore/assemble.py` | 10.5k | final assembly, A/V sync gate, audio timeline gate |
| `vidlore/clipstudio/build.py` | 7.3k | **the build stage** — window selection, cuts, breakouts, stills, all final gates |
| `vidlore/clipstudio/match.py` | 3.1k | **beat → shot matching**, scoring, window-QC (`clean_cut_window`) |
| `vidlore/clipstudio/orchestrate.py` | 2.2k | stage orchestration, `produce_auto`, bounded recovery |
| `vidlore/clipstudio/verify.py` | 1.8k | vision verifier (prompt v8), repair/promotion |
| `vidlore/clipstudio/discover.py` | — | YouTube discovery, anchor/key-scene coverage, non-show title gates |
| `vidlore/clipstudio/download.py` | — | download orchestration + `download_audit` |
| `vidlore/clipstudio/hd_download.py` | — | **HD path**: yt-dlp + PO-token + client ladder |
| `vidlore/clipstudio/index.py` | — | shots, CLIP embeds, flags (luma, subs, graphics, corner masks) |
| `vidlore/captions.py` | — | ASS/SRT caption generation, word-level karaoke |
| `vidlore/musiclib.py` | — | music selection + usage ledger |
| `tests/` | 61 files | run with `python3 -m pytest tests/ -q` |

**Models:** default brain is `deepseek-v4-pro → v4-flash → Gemini → Claude` (last). Vision QC runs
on `gemini-2.5-flash` via `GEMINI_API_KEY` (~10× cheaper than Claude vision; DeepSeek cannot see
images, so the order is gemini → claude).

**Cost:** a cold render is **$1.26–1.88**, ~95% of it gemini-flash vision calls.

---

## 4. What changed in the 2026-08-02 session (8 commits)

The trigger: a finished 12-minute render (`portal/409e284b60`) was audited frame-by-frame and found
to have shipped **entirely at 360p**.

### `207a26b` — HD: an unknown yt-dlp flag can no longer take a render to 0% HD
**The bug was mine.** `--remote-components ejs:github` (added in `70f56eb` to fetch the JS-challenge
solver) was sent unconditionally. When the resolved yt-dlp does not know the option, yt-dlp exits
**in argument parsing**, before any network call — so every source, every client rung and every
retry failed identically, `_classify_dl_err` read it as `"other"` (not recoverable, so the sweep
never re-attempted), and each per-source log line read exactly like *"this video has no HD copy"*.
Result: `hd_path_ok 0/72`, 1% of sources ≥720p, and the render completed and called itself final.

Three independent defences:
1. **support probe** — `_flag_supported()` asks `--help` once per process, cached; fail-open if the
   probe itself cannot run.
2. **runtime retry** — a rejection is classified `"badflag"` FIRST (argparse death cannot coexist
   with a real response), drops the flag process-wide via `_disable_unsupported_flag()`, and retries
   the **same client rung** (a rung is for 403s, not our own broken command line). Now recoverable,
   so the sweep re-attempts.
3. **release block** — `hd_path_ok == 0` over ≥5 YouTube sources marks the output a REVIEW DRAFT.
   No per-beat gate can see this: every beat is correct and merely 360p.

Also: a FAILED `pip install -U` of the HD stack logged nothing yet still stamped the weekly marker,
so seven days of renders could inherit a stale yt-dlp silently. It now says so.
Tests: `tests/test_hd_flag_rejection.py` (26).

### `0529d09` — recovery: rotate beats between rounds
Bounded recovery ranked unresolved beats by policy class then script order — a deterministic key
with no memory — so **every round re-took the same head of the same list**. Measured: round 1 took
`[90,110,166,76,89,91,12,13]`, round 2 took `[90,110,76,79,89,12,13,19]` (six re-attempts) and
reported `candidates_found 21, new_candidates 0`, because re-issuing a query YouTube already
answered returns the sources already on disk. Beats naming Arya-kills-the-Night-King, the Children
of the Forest, the Dragonpit wight were **never searched once**.

Within its class, a beat never searched now outranks one already tried. Re-attempts are kept, just
last. Class priority still wins over freshness (CHARACTER beats block releases). Ranking extracted
to `orchestrate.recovery_pick()` so it is testable. The turn is recorded BEFORE the search, so a
round that dies mid-way still spends the beat's turn.
Replay on the real 32-beat list: round-2 overlap **8/8 → 2/8**; three rounds reach **20 distinct
beats instead of 8**. Same cap, same cost. Tests: `tests/test_recovery_rotation.py` (11).

### `a3bb036` — watermark: the badge on the SIDE border
A media-player badge (orange square, white `m`, a running timer) was burned into the right border of
four re-uploads and aired on four beats. `_source_corner_logo` could not see it for two geometric
reasons: it sits at **y 46–56% of frame height** (corner patches are the outer 18%×12% boxes), and
its digits re-roll every second (any whole-patch-static test fails).

`match._source_edge_logo()` keeps the same statistic — an overlay's edges land on the same pixels
while a scene's edges move — over the outer **9% of width, full height**, plus one discriminator:
**compactness** (y-extent ≤ 25% of height), which is what separates a badge from a pillarbox seam.
Calibrated on 69 indexed sources: fires on 5, all real (`SPHINX TV`, `FAVORITE FLASHBACKS FRENZY`,
a bottom-edge mark, and the two player badges), **0 false positives**. Consulted only after the
corner detector, and it **crops** rather than rejects.
**Known limit:** two badge sources show the overlay on only 25% of keyframes (the player UI
auto-hides) — no source-level detector can reach them. Left to the per-shot rule on purpose.
Tests: `tests/test_edge_badge_gate.py` (12). Kill switch: `VIDLORE_CLIPSTUDIO_EDGE_LOGO_GATE=0`.

### `f7420ca` — window-QC: the right shot can still air the wrong seconds
Two beats chose a shot carrying the RIGHT actor (faceid 1.0, `Joseph Mawle` for a Benjen beat,
`Kit Harington` for a Jon beat) and still aired seconds showing somebody else. By the time a window
is cut the scoring is over, so the shot-level `wrongface` penalty cannot reach it.

`clean_cut_window(..., face_guard=)` now treats seconds showing a **confidently-named different
main-cast member** as dirty spans. Three safety properties:
- **three states** — wrong needs a CONFIDENT name, that name in the main cast, and the entity FULLY
  resolved. Unknown is never wrong. Both the character and actor spellings are targets.
- **strict shorten-only** — the window is computed twice (`_clean_cut_window_inner`), once with the
  identity spans and once through the identical path without them; the identity pass is used only
  when it does not reject. **Proven exhaustively:** no window the old path accepted is rejected.
- moment rules untouched.

`match.face_guard_for()` is the single builder. Kill switch:
`VIDLORE_CLIPSTUDIO_WRONGFACE_WINDOW_GATE=0`. Tests: `tests/test_window_face_guard.py` (18).
**HONEST NOTE:** measured **inert on all 7** wrong-character cases of that render — Lyanna is not on
the roster and Cersei has 3 confident shots pool-wide, so nothing was confidently misnamed. Kept
because it is safe and free, not because it fixed those.

### `123778b` — bare `os` in `match_segments`, and the guard widened
The face-guard commit introduced `os.environ.get(...)` inside `match_segments`. `match.py` has **no
module-level `import os`** — every function takes a local alias — so it raised NameError the first
time the match stage ran. A smoke render caught it before a long render did.

**Third occurrence of this exact bug.** The static guard written after the first one only parsed
`orchestrate.py`; it now walks **every module** in `vidlore/clipstudio/`. Verified by reintroducing
the bug and watching the guard name it. See `tests/test_recovery_stage_alive.py`.

### `43662da` — `tools/rerender_benjen.py`
A render driver with a **pre-flight that can abort before money is spent**: asserts the flag is
advertised, probes two real pool URLs for ≥720p, checks `musiclib.scan()` = 11/118. Copy this
pattern for any future render driver.

### `f58b465` — audit tooling: correct the scene→beat join, and refuse when it drifts
See §7. **This fixed a mistake I made and shipped**: the first audit joined scenes to
`aired_windows` clip order by duration; 23 of its 24 findings named a beat 1–4 places off.

---

## 5. Accumulated subsystem knowledge (earlier sessions)

Each of these is a real defect that was found, root-caused and fixed. Do not undo them.

**Footage leaks / source screening**
- Reactor facecams leaked as stills — `\b(...)\b` trailing boundary made singular junk words miss
  plurals (`Reactions`/`Reactors`). Fixed at 4 layers.
- Stylized corner watermark (`BLACK TRVLLS`) OCR'd as garbage → pixel static-corner detector v3
  (7/7 true bugs, 0 FP) + clean-copy arbitration + script-agnostic subtitle-band gate.
- Commenter-avatar badge beat every text rule (name ≠ junk keyword, 2 words < heavy floor,
  intermittent < 25% corner presence, 51s-shot sampling gap) → per-shot overlay-badge rule
  (0 FP on 1957 shots) + caption-dodge crop-REPAIR-first + duration-scaled sampling.
- Non-show leaks (news CGI, game-UI parody, cartoon, fan art) → tiered CLIP `graphic_dom` on
  PERSISTED embeds (hard 0.036 / band+art, ≥3-hard arming, source drop ≥20%; 0 FP on 1957 shots).
- Cross-show: House of the Dragon aired inside a GoT cold-open — `_wrong_installment` only matched
  sibling SHOW names, missing character-titled HotD clips.
- Promo-overlay source gate (tail-aware), listicle/static gates, crossover-mashup title arm.

**Relevance / matching**
- **Scene identity vs character presence**: the correct Night-King source ranked #1 and lost by
  0.0084 because character presence scored 0.50 (faceid + entity double-count) vs scene identity
  0.12. Fixed with ITF title affinity + flat-frame gate + era agreement (`f62f225`).
  exact_scene 64–73% → 82% on that measure.
- **Face-ID identity gate** (`ec4154a`): `face_ids & main_cast` READ like a wrong-character test and
  was bit-identical to `face_ids != {}` — 0 of 625 Face-ID names fall outside the roster. It
  rejected 87% of shots as "wrong". Three states (right/wrong/unknown) + a real entity resolver
  (`resolve_face_targets`, 80 → 102 of 107 beats) removed 1,295 false −0.50 penalties on the
  CORRECT person; 525 fake rejections became 71 real ones.
- Era poison: `S04E01 ≠ S03E10` purged 354 shots.
- Per-window anti-reuse; `_load_pool` rejections promoted to the shared ban-list (the BTS leak path).

**Window / cut quality**
- **QC the FINAL rendered window, not the shot's own frames** (`65d0ccc`) — `clean_cut_window` with
  anchor-overlap, wired at match/verify/cut/build/breakouts.
- Freeze-onto-ANOTHER-SCENE was the real damage (31 freezes, 30 cross-scene), not aired darkness
  (`d2b06ad`) — fixed by probing the source window BEFORE the cut.
- Exact-moment pass: moment-locked beats never slide (`preserve` / `_moment_kept`).

**Audio / captions / delivery**
- **ALWAYS probe the rendered file's audio before calling it final.** v3 once shipped 20 minutes
  with captions + music and ZERO voiceover (chained silent skips). The build now RAISES on a
  named-but-missing voiceover.
- Cold-open caption desync: `_apply_breakouts` shifted words in place then raised on a failed t=0
  splice, leaving captions shifted but audio un-spliced. Made ATOMIC.
- Caption flicker: word events ended at `w.end` with nothing covering inter-word gaps → bridge to
  the next word's start (gaps 303 → 57).
- Delivered audio must not lie: per-frame `stts` check, fatal above `VIDLORE_AUDIO_GAP_FATAL_MS`.
- Breakout atomic composition (`_compose_breakout_state`, invariant + rollback, post-render QA).
- 5 selectable caption presets (professional default / minimal / cinematic / documentary / focus).

**Infrastructure**
- HD path has broken in **four** distinct ways now: PO-token/SABR (error 152), Windows cookie-DB
  lock (yt-dlp issue 7271 + DPAPI), the missing JS-challenge solver script, and the unknown-flag
  collapse. Each has its own classifier + recovery.
- Speed pass (`0830b56`, all parity-proven): flags mono-decode 2–6×, OCR pool, download↔index
  prewarm, verify rung prefetch, early RF-gate 213s → 8s.
- Persisted-embed stills >100×, all-rung verdict caching (warm 25 → 0 calls).
- Resume checkpoints + fail-fast pre-assembly feasibility gate.
- Ad-gate own-caption whitelist — an end-of-render FATAL was our OWN "subscribe" outro caption.
- **`fail-open` catches hide bugs**: `recovery: skipped (NameError)` looked benign but R4-5 recovery
  was DEAD every render for months. `_log_stage_skip` now shouts on code faults.

**Windows parity** — works, but git can't carry it: CLIP models, 118 mp3s, Face-ID models and the
real `.env` are all gitignored blockers. Launch only via `run-windows.bat` (sets `PYTHONUTF8`).

---

## 6. MEASURED AND REJECTED — do not retry without new evidence

Every one of these looked right and was killed by measurement. Re-proposing them wastes a cycle.

| Idea | Why rejected |
|---|---|
| Face-ID at native res instead of the 512px keyframe | names 23 more per 300 but **5 of 23 are WRONG (21.7% vs 4.8% baseline)**; score does not separate right from wrong |
| Quality-scaled title bonus | net negative, 7.25 → 6.50 |
| "Full" legibility damper | broke the godswood beat 10 → 5; switched to HARD-only |
| Slide-in-shot instead of choosing another window | rescued 1 beat, regressed 1. Net zero. 175/272 beats air LONGER than their shot, so there is no room |
| New luma thresholds for the darkness gate | see below — no luma statistic separates the classes |
| Dark-fraction threshold (<25 luma) | confirmed-unreadable 0.527–0.898 vs refuted-readable 0.227–0.601 — **overlapping** |
| Structure inside the dark mass | separates the labelled sets but fires on 2 clearly-GOOD frames (a Valyrian-steel-dagger shot). Measuring structure inside the dark region excludes the subject by construction |
| Any luma statistic at all | scene 20 (unusable) and scene 82 (excellent) have mean luma **14.3 vs 14.4**, dark fraction 0.898 vs 0.865 |
| Re-probe the FINAL aired start instead of the candidate in-point | mechanically real (~19% of clips are relocated after the probe) but an adversarial check probed all 38 shifted clips at both starts — the verdict flips on **ZERO** |
| `best_name` as evidence | tautological — 0 of 4,586 shots have a `best_name` outside the cast |
| Downscale / batch / flash-lite-for-exact (cost) | rejected with reasons in the cost audit |

---

## 7. How to audit a render (do it this way)

**`research/eval/scene_audit_dataset.py <job_dir> <out_dir>`** — extracts one frame per timeline
scene and pairs it with the narration **heard at that timestamp** (from the delivered `.srt`), never
from the pipeline's own bookkeeping.

**The join is the trap.** Every beat occupies exactly one scene (footage or image still); a breakout
inserts an EXTRA scene belonging to no beat, and a breakout scene is identifiable because it carries
source audio so no voiceover caption overlaps it. Therefore:

```
beat = scene_index − (breakouts before it)
```

Do **NOT** join on `aired_windows` clip order — it records only footage clips, so its index runs
ahead by every still and breakout. Duration-matching looks right (it consumes the correct number)
and drifts silently up to five beats. The script now self-checks against the ledger's own narration
and **returns 3 rather than emitting** if under 90% agree (validated: 169/170 and 166/167).

**`research/eval/caption_sync_probe.py <job_dir>`** — transcribes windows of the DELIVERED file and
scores them against the `.srt` at shifts −6…+6s. A healthy render peaks at 0. Note: windows landing
on a breakout score lower because ASR hears show dialogue the VO captions never contained.

**Other useful artifacts in every job:** `output/build.log`, `rejected_footage_audit.json`,
`recovery_audit.json`, `aired_windows.json`, `ledger.jsonl` (keyed by `segment_index` — the same
index the join resolves to), `cost_report.json`.

**Read `rejected_footage_audit.json` FIRST** when a render release-blocks.

---

## 8. Current state

**Renders on disk**
- `output/portal/409e284b60/` — the 360p one. 24 confirmed defects. Superseded.
- `output/portal/benjen_v2/output/final.REVIEW_DRAFT.mp4` — the rerender. 12:55, 2.55h.

**benjen_v2 vs 409e284b60** (same audit rubric, same agent count — comparable):

| | old | new |
|---|---|---|
| HD path | 0/72 | **101/101** |
| sub-480p sources | 72 | 13 |
| frame sharpness (median edge-var) | 7.0 | **9.9 (×1.42)** |
| repeated windows | 11.8% | **6.6%** |
| distinct sources aired | 48 | 54 |
| release-blocked beats | 11 | 8 |
| **confirmed defects** | **24** | **24** |
| — illegible/dark | 4 | **0** |
| — wrong character | 7 | **2** |
| — watermark | 4 | **1** |
| — non-show | 1 | **0** |
| — **named scene not delivered** | 8 | **16** |

**Read this honestly:** the four classes that were targeted went **16 → 3**. The total is unchanged
because named-scene misses doubled. There is a confound that could not be removed — the old frames
were 360p mush and the new ones are HD, so some of the +8 is auditors now being ABLE to identify
which scene is on screen. It is not a pure regression and it is not harmless.

**Still true of both:** captions ↔ delivered audio in sync (no shift better than 0), beat↔narration
alignment 166/167, all final gates clean (ad scan, black/legibility, A/V sync, audio timeline).

---

## 9. Open work, highest value first

**#34 — Beat 90 airs the wrong scene while the exact source sits unused** *(selection, not footage)*
Beat 90 hears *"Winterfell, the godswood, Arya and a dagger"* and airs *"6x05 Bran meets the Night's
King"*. The pool holds **two exact-titled** Arya-kills-the-Night-King sources — one 1080p used on
another beat, one 720p used on **zero** beats. The pipeline labelled the beat `exact_scene_missing`,
so it knows it failed. Start at `match()`: why does a title carrying every scene token lose? Check
ITF title affinity, whether those sources were gated, and whether the `[Spoilers]` prefix or the
S8E3 era check demoted them.
Same family: beat 47 holds a generic wide **7.62s** under *"a bear comes out of the mist already
burning"* while the real Wight-Bear source airs one beat earlier for **1.96s**. Right footage, wrong
beat, inverted durations. **Both are testable offline with a `match_segments` replay — no render
cost.**

**#32 — Dark aired windows: the STATISTIC misses them, not the call site**
Root cause known: `_frame_luma_hi` is a 99.5th-percentile test, so any bright sliver rescues an
80–90%-black frame. Four candidate fixes measured and rejected (§6). What would actually work is a
**subject-presence** check, not a brightness check — closer to image understanding. The pipeline
already runs gemini-flash for footage QC, so ~1 extra call per suspect window; needs its own cost
design and 0-FP calibration.

**#21** beat-class release-block: abstract beats have no same-scene fallback.
**#24** freeze donor: constrain cross-scene donors, soften the darkness trigger.
**#26** breakout audio: 3.5s dead air in a 5s window; caption 1.1s before any sound.
**#27** Windows: NVENC never probed (renders on the Intel iGPU); perf telemetry blind.
**#30** remaining `fc41397ea5` items: HBO-Max watermark on image stills; "must CONTAIN required
entity" strengthening; planner demoting emotional beats to `generic_filler`.

---

## 10. Working agreements with the owner

- **Reply in Hinglish/Urdu.**
- **Never starve footage.** Standing instruction: *"yeh na ho footage bht limited ho jaye, rejection
  bht zyada na ho jaye."* Every new gate must be shorten-only / crop-first, or prove that rejection
  cannot grow (the two-pass shape: pass 1 = new rule, pass 2 = old loop verbatim).
- **Renders cost money and hours.** Test on 90s–2min clips (`tools/render_intro_test.py`) or offline
  replays. Do not rerender repeatedly; the owner asks for exactly one.
- **Do not start the portal** — he starts it himself.
- **The USB bundle / `.env` hold his real API keys.** Never share or print them.
- **Measure before claiming.** Proxy metrics let three renders ship flat. Validate any new eval
  against a render with a known score before trusting it — a quick 26-beat eval once scored a 5.43
  render as 7.50.
- **Never trust a mid-flight artifact.** `project.json` is rewritten as stages run; sampling it
  during a render produced a wrong conclusion in this very session.
- A permanently-red test is never acceptable; a fail-open catch that swallows a code fault must
  shout.

---

## 11. First 10 minutes in a new session

```bash
cd /Users/hussnain/Desktop/vidlore-clipstudio
git log --oneline -10
python3 -m pytest tests/ -q          # expect 675 passed, 6 skipped
python3 -c "import sys;sys.path.insert(0,'.');from vidlore import musiclib;c=musiclib.scan();print(len(c),sum(len(v) for v in c.values()))"   # expect 11 118
python3 -c "import sys;sys.path.insert(0,'.');from vidlore.clipstudio import hd_download as H;print(H.available(),H._flag_supported('--remote-components'))"
```

Then read this file's §8 and §9 and pick up **#34**.
