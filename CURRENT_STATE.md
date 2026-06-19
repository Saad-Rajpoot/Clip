# CURRENT STATE — Vidlore

## R1 Cinematic Weight — **SHIPPED** (2026-06-04, snapshot `AudioEngine_V1.3_CinematicWeight`)

A subtle, scene-aware **sustained low-end "weight" floor** added to the EXISTING atmosphere bed
(`build_atmosphere_bed` in `music.py`) — closes the measured Vidlore↔Vidlore low-mid-tilt gap on
the heavy beats without touching the stable audio architecture. **Minimal additive extension:** no
new mix chain, no new ducking, no assets, no license risk (fully synthesized). Studied the engine
first (`research/audio_engine/cinematic_weight/EXISTING_AUDIO_ENGINE_ARCHITECTURE_AUDIT.md`) — the
atmosphere bed + scene-role classification + limiter already existed; only a sustained 20–80 Hz
component was missing. **Design:** synthesized sub sine pair (42/63 Hz) lowpassed <90 Hz (below the
300–3400 Hz voice band → no narration masking, no mud), gated to the **top-3 heaviest beats only**
(pre-ranked by energy+role; heaviest=strong, next 1–2=light; windows split on weight change so it's
never a constant floor), gain tuned **gentle after the user's listening review** (v1 muddy → v5
approved). Flags `VIDLORE_CINEMATIC_WEIGHT` (**default-ON**, `=0` rollback) +
`VIDLORE_CINEMATIC_WEIGHT_LEVEL=light|balanced|strong`. **Validated:** Edison A/B (low-mid tilt
lifted on the climax; **voice band Δ 0.0**, true-peak −1.4 dBTP, LUFS −16.0, 0 black, 2 weighted
windows) + B(concrete)/C(modern) full renders default-on (0 black, safe LUFS/peak, 2–3 capped
windows). Regression `test_audio_director` 55/11 — **the 11 are PRE-EXISTING** (parallel audio
session's intro-profile/safety-rail tuning; identical with my flag on/off → not R1). dist 0-drift
(`music.py` + `assemble.py`). Audio **limiter (0.85) untouched** (already synced). Docs +
A/B clips: `research/audio_engine/cinematic_weight/`. **Stabilization pass (2026-06-04,
`R1_STABILIZATION_{TEST_AUDIT,REPORT}.md`):** confirmed R1 fully clean — engine_guards 106/0, 0 .py
drift, 0 USE-ONLY in dist, 0 temp leak, limiter 0.85 untouched. The **11 `test_audio_director`
failures are the PARALLEL audio session's intro-profile retuning** (`intro_profiles.json` 12:08:
start_mult→≤2.8, recede_s→26, `default` niche no longer a no-op), NOT R1 (identical with my flag
on/off). **RESOLVED — `AudioEngine_V1.3.1_CinematicWeightStabilized`** (`INTRO_PROFILE_STABILIZATION_REPORT.md`):
named-niche v14 hold-then-recede is intentional → 2 stale rails widened to the validated ceilings
(start_mult ≤2.85, recede_s ≤30.5); the `default` non-no-op was **accidental** (contradicted its own
"neutral no-op intro" comment) → restored `_DEFAULT_INTRO` → `_BODY_BED` (true no-op) + regression
guard. Suite **67/0**, engine_guards 106/0, full `.py` 0-drift (`music_director.py` synced), smoke
clean (default no-op intro + weight on 2 moments, 0 black, −16 LUFS, −1.9 dBTP). R1 layer UNCHANGED;
`AudioEngine_V1.3_CinematicWeight` preserved.

## Vidlore Quality Forensic — **Phase 1 read-only complete** (2026-06-03, `research/vidlore_forensic/`)

Evidence-driven forensic to find *why* Vidlore docs still feel more professional than Vidlore.
**Read-only phase only — NO code changed; renders deferred** (3 bio renders running, disk 93%).
Built on the substantial prior art (`final_forensic_compare/two_cheap_metals/`,
`audio_engine/vidlore_ai_comparison/`, `visual_relevance/`) rather than redoing it.

- **3 Vidlore samples** (Two Cheap Metals 20:34/720p, Copper 15:11/720p, **$12 Tesla Antenna
  14:35/1080p** = newly analyzed). Tool: `research/vidlore_forensic/_forensic.py` (pacing,
  low-end audio, contact sheets; bundled ffmpeg, no ffprobe).
- **Root causes ranked** (`ROOT_CAUSE_RANKING.md`): R1 sustained 20–80 Hz low-end "weight" floor
  (VR −5.2 vs VF −25.2 dB — #1 perceptual gap, audio/**risky**); R2 subject-wrong-but-plausible
  footage (query-text not pixel match); R3 static cards held ~24 s on reveal beats (Vidlore shots
  **never > 12.3 s**, median 3.53 s on Tesla); R4 per-VIDEO grade + uniform `_CINEMA_FINISH` on
  every scene (Vidlore varies grade by asset type). Already-shipped: subject-floor, prose-stat→
  animated-card, black-frame repair, CLIP distractor probes.
- **Hypotheses tested, not assumed:** #3 (one overly-strong grade) is **philosophy-level
  confirmed** (over-unification) but prior per-beat metric rates Vidlore grade *quality* high →
  R4 is delicate, **A/B on frames before keeping**; "Vidlore 720p-soft" is now **stale** (newest
  sample is 1080p).
- **Plan** (`IMPLEMENTATION_PLAN.md`, per user's choices): Batch 1 safe/additive/flag-gated —
  F1 anti-static reveals (R3), F2 subject-presence live-validate (R2), F3 conservative grade
  restraint (R4, keep `_VINTAGE` as-is per prior instruction); Batch 2 audio sub-drone + QA gate
  (separate sign-off); Batch 3 architectural. Every batch render+frame-validated, QA-gated,
  Mac/Win 0-drift, snapshot only on pass. Docs: `VIDLORE_SAMPLE_INVENTORY.md`,
  `VIDLORE_EDITORIAL_GRAMMAR.md`, `ROOT_CAUSE_RANKING.md`, `IMPLEMENTATION_PLAN.md`.
- **F3 grade restraint LANDED (2026-06-04):** benchmark `edison_tesla` measured Vidlore **luma 61,
  30% murky frames vs Vidlore 116/4%** (~2× too dark — confirms hypothesis #3). Root cause: uniform
  finish vignette `PI/5.0` + grade crushing footage. Fix `_grade_restraint()` in `assemble.py`
  (soften vignette `PI/5.0→PI/6.8` + gentle `gamma=1.14:brightness=0.028` lift on non-archival
  footage only; `_VINTAGE` untouched; default-ON, `VIDLORE_GRADE_RESTRAINT=0` disables). A/B
  (same footage): **luma 61→76 (+24%), murky 30%→11%, 0 black frames**, still cinematic. dist
  0-drift (`assemble.py` `c4568a052f`). Remaining Batch 1: F1 anti-static (R3, 18.1 s hold vs ≤12.3),
  F2 subject-vision (R2); Batch 2 audio low-end (R1, tilt −32.7 vs −9.0 dB; separate sign-off).
- **F2 Footage Taste Intelligence SHIPPED (2026-06-04, snapshot `Vidlore_V1.5_FootageTasteIntelligence`):**
  the local frame-level CLIP relevance scorer was **default-OFF** (`VIDLORE_VISUAL_RELEVANCE`) — default
  renders accepted footage on query/slug TEXT overlap only. Calibrated across 3 niches + shipped **default-ON
  for concrete scenes** (`footage.py` both scorer sites `0→1`; strict-concrete / guard-only-abstract / fully
  defensive). Distractor thresholds made **env-tunable** in `visual_relevance.py` (defaults byte-identical →
  behavior-neutral; `VIDLORE_VR_{WAR,VEHICLE,PEOPLE,DISTRACTOR}_MAX`; war stays tight 0.03 fail-closed).
  **3-benchmark A/B:** A Edison 1880s (21 wrong-subject→2 stock+19 AI), B Industrial 1700-1800s (15→6+9),
  C **modern** startup (7→**6 real stock + 1 AI**) — **0 wrong-subject kept, 0 black frames** all three;
  AI escalation tracks ERA not over-rejection (modern keeps real stock). Gates pass. Regression:
  graphic_gate 13/13, engine_guards 106/0, multiniche 10/13 (3 **pre-existing** CLIP-margin FNs, not
  regressions). dist 0-drift on the 2 F2 files. Rollback: `VIDLORE_VISUAL_RELEVANCE=0`. Docs:
  `research/quality_intelligence/F2_*`. **F1 deferred** (justified hook/name-reveal hold; ready guard documented).
  **Audio engine UNTOUCHED** (cinematic low-end pass awaits explicit approval). ⚠️ external parallel-session
  `alimiter 0.89→0.85` drift in SOURCE `assemble.py` only — left for the audio session to sync.

## Creation Dashboard — **Premium beginner-first rebuild** (2026-06-03)

Rebuilt the main video-creation form (`GET /new`, the `_FORM` template in `vidlore/web.py`)
from a dense 7-section developer form into a beginner-first flow with progressive disclosure,
driven by a full backend audit (`research/creation_dashboard/`). **Only the `_FORM` template
changed**; pipeline, script handling, voice/TTS, footage ladder, Look-DNA, motion graphics,
music/SFX, subtitles, repair, Review Editor, post-render dashboard untouched; **AI video OFF**;
not deployed. Snapshot `Vidlore_CreationDashboard_V1.0_PremiumBeginnerUX`.

- **Audit first (required).** `CREATION_DASHBOARD_BACKEND_AUDIT.md` mapped every control to the
  real `Brief`/`extra` surface. Findings: `fmt=top10` is **dead** (removed); `duration` drives
  the **AI-script word target** not final length (relabelled "Target script length", shown only
  under "Write one for me"); **Look-DNA** (`look_preset`, with **Auto** niche-detect) is the
  real style system (promoted); legacy **Style Mode** (`style`) is overridden by it (pinned
  `auto`, hidden); `theme`/`background` are no-Look fallbacks (→ Advanced); Homestead + Style
  Modes are intentional (kept). No real preset removed.
- **Removed:** HeyGen dashboard banner (standalone `/heygen-broll` tool + route KEPT), Format
  selector (hidden `fmt=documentary`), top-level Duration, mandatory creative brief.
- **Beginner flow:** Step 1 content (title · "I have a script"/"Write one for me" segmented
  toggle · paste-script *or* optional direction+target-length · voiceover upload · load-sample)
  → Step 2 documentary style (Look-DNA cards, **Auto detect = Recommended**) → Step 3 narration
  voice (Basic/Premium, auto-noted when a voiceover is uploaded) → **Create documentary →**.
  **Advanced** (collapsed `<details>`, auto-opens on error): Visual sourcing (`shutterstock` ·
  `wi_mix` · `wf_mix` experimental), Editing (`music`/`transitions`/`overlays` ON ·
  `sfx`/`captions` OFF), Voice details (`tts_model`/`tts_voice`/`voice`), Appearance
  (`theme`/`background`). **Every `name=` preserved** with backend-matching defaults so
  `_brief_from` wires unchanged; self-contained dark theme + tooltip engine + responsive grid.
- **Verified:** `tools/test_creation_dashboard.py` **81/81** (removals, 22 field names,
  defaults, anchors, Advanced collapse/error-open, toggle defaults, JS, `_brief_from` mirror).
  **Real end-to-end render** through the new fields: QA **PASS**, **0 black frames**,
  1920×1080@30, −16.1 LUFS, 3 scenes, AI video off; fed post-render summary cleanly. Headless-
  Chrome screenshots of both UI states. Regression: post-render **36/36**, editor **124/124**.
  dist **0-drift** (`web.py` src=Mac=Win). Preview sandbox can't read the venv → ran live from
  `dist/Vidlore-Mac` + headless Chrome for visuals.
- **Recommended (not shipped):** re-add "▶ Preview voice" (`/voice-preview`) + Premium-readiness
  chip (`/voice-status`) in Advanced; expand the style gallery (4 of 7 Look presets surfaced).

## Relevance Gate — **Designed-Graphic rejection (keyword-independent)** (2026-06-03)

Closed the seven-niche sweep's relevance-gate gap: on scenes with weak/empty
`keywords` the visual-relevance gate ACCEPTED off-topic **designed graphics**
(party-logo clip-art, a "POLYSEXUAL" text image, a modern mortgage-rates
infographic) because the expected subject was too vague to reject them and no
gate asked "is this a graphic instead of footage?". Changed
`vidlore/visual_relevance.py` + `vidlore/footage.py` only; AI-provider path
(fal primary) byte-identical.

- **Graphic probe (the fix).** `visual_relevance.score_asset` now emits
  `graphic_dom` = MEAN-over-frames of `max(graphic_sim) − max(realphoto_sim)`
  (`_GRAPHIC_NEG` = infographic/chart/diagram/logo/clip-art/cartoon/poster/
  screenshot/slide/text-sign vs a real-photo anchor). `accept()` rejects when
  `graphic_dom > 0.036` (env `VIDLORE_VR_GRAPHIC_MAX`), placed **before** the
  guard-only early-return so it also fires on abstract/stat beats — i.e. footage
  used as a **motion-graphics card background**. Keyword-INDEPENDENT by design
  (compares to a real-photo anchor, not the scene subject), so it rejects a
  chart/logo/text image even when keywords are empty.
- **Calibration** (sweep assets): party-logo 0.062, "POLYSEXUAL" 0.046, mortgage
  infographic 0.056, "Politics" sign 0.037 — vs every good asset ≤ −0.007 (real
  archival photo, fal portraits, real soldiers/field clips). MEAN agg so a real
  clip with one incidentally-flat frame is never rejected.
- **Validation.** New `tools/test_graphic_gate.py` **13/13** (pins the exact
  sweep assets, incl. guard-only mode); `tools/test_multiniche_relevance.py`
  unaffected (additive — identical with the gate on/off). Re-ran the real
  pipeline: **spy** party-logos+polysexual+Politics-sign → rejected→fal stills
  (`research/editorial_qa/fal_niche_sweep/CONTACT_spy_FIXED.jpg`); **history**
  mortgage infographic → rejected→fal still (`CONTACT_history_FIXED.jpg`),
  slides **8.8%→0.0%** (escalations went to AI stills, not fail-closed slides ⇒
  zero good-footage false-rejects).
- **Also (safe):** era-derived modern NEGATIVES in `_vr_judge` for pre-1945
  scenes (feed `distractor_dom`; verified not to false-reject good period
  footage), and `graphic_dom` is logged in `ASSET_DECISION_MANIFEST.json`.
- **Residual (honest, deferred):** the modern **COVID-masked crowd** /
  **modern-Moscow** (wrong-*era*, history) are NOT fixed. A period-gated CLIP
  anachronism probe was calibrated and **rejected as unsafe** — a good period
  clip (0.042) scored higher than the wrong masked crowd (0.040); no threshold
  separates modern from historical crowds without false-rejecting good footage.
  Wrong-era belongs on the search side (period_guard query bias) or a future
  frame-level era classifier — NOT the pixel relevance gate. Do not chase it by
  strengthening the expected subject: that drops good period-neutral footage
  (snowy soldiers, open field) below the relevance floor.

## Review Editor V1.3 — **Real-Browser-Verified Clean UX** (2026-06-02, snapshot `Vidlore_ReviewEditor_V1.3_RealBrowserVerifiedCleanUX`)

Fixed the 3 USER-REPORTED real-browser bugs that prior passes had wrongly marked
"working" (I'd relied on JS `.click()` / synthetic events / API 200s — which prove
wiring, not real mouse behavior). This pass used **Chrome MCP real CDP mouse input**
(`computer` tool, screenshot-pixel coords). **Only `vidlore/web.py` changed**; render
pipeline + `editor_manifest` byte-identical to V1.2. QA: `research/review_editor/
REAL_BROWSER_CONTROL_BY_CONTROL_QA.md`, `FINAL_CLEAN_DASHBOARD_AUDIT.md`.

- **Bug 2 (timeline playhead not draggable) — FIXED + real-mouse-verified.** Root
  cause: `#edph` was a passive `<div>` with no handler. Added click-to-seek + drag-
  scrub on `#edtlb` using **mouse** events (the Chrome tool emits mouse, not pointer)
  with document-level move/up + bound clamp + a grabbable `.phknob`. Verified by a
  real drag (2:00→1:00 → currentTime 62) and a real ruler click (→121).
- **Bug 1 (Replace looked broken) — FIXED.** The replace DATA path always worked
  (thumbnail/override/badge/persistence/final-MP4 — proven by the V1.1 magenta
  render). The gap was UX: the main preview is the OLD render until re-render and
  there was no feedback. Added a toast on every replace + an **instant corner pip**
  (`#edrepov`) showing the new visual on the selected scene with a "re-render to bake
  it in" note. Verified: real search-pick on footage scene 3 → ladybug pip shows.
- **Bug 3 (timeline crowded) — FIXED.** Width-tiered block labels: narrow = scene
  number, medium = number + keyword, wide = number + narration; full text in tooltip.
- **Validation:** regression `tools/test_review_editor_repairs.py` **27/27** (incl.
  seek/clamp math); real footage-scene re-render proof (scene 3 = ladybug, not a card
  scene). dist 0-drift (`web.py` synced Mac+Win); AI video OFF. V1.0/V1.1/V1.2
  snapshots preserved.
- **Honest:** the native file picker can't be driven by Chrome MCP (upload path proven
  via the prior magenta render + the search-pick path — both reach the renderer via
  `visual_override`→`locked_visuals`).

## Review Editor V1.2 — **Final Dashboard Polish** (2026-06-02, snapshot `Vidlore_ReviewEditor_V1.2_FinalDashboardPolish`)

Final polish + deep-validation pass on V1.1. Changed `vidlore/web.py` +
`vidlore/editor_manifest.py` only; render pipeline byte-identical. Audit:
`research/review_editor/CAPCUT_UX_FINAL_AUDIT.md`; manual QA:
`CAPCUT_UX_FINAL_MANUAL_QA.md`.

- **P4 inspector declutter** — removed the action-bar/section REDUNDANCY: now a
  compact 3-group bar (Visual: Replace/Generate · Card: Edit/Preview · Scene:
  Up/Down/**••• More**) + 4 collapsed sections (Visual, Card, Scene details,
  Advanced). Re-voice / Reset / Delete / Use-original-visual live in the ••• More
  menu (`__edRowMenu`, reused from the scene-row menu).
- **P1 timeline direct drag-reorder** — visual blocks `draggable`; X-axis insertion
  line (`.tldropl/.tldropr`), grab/grabbing cursors, **disabled during render**;
  routes the drop through the SAME validated override reorder endpoint via
  `__edReorderApply` (33 draggable blocks verified; reorder backend confirmed).
- **P2 first-open loading** — per-project `_build_lock` serializes the parallel
  manifest+timeline build; `_atomic_write` (tmp+replace); `extract_scene_thumbs`
  regenerates a thumb only if stale (older than the mp4) — fixes stale posters
  after a re-render; per-thumb cache reuse retained.
- **P3 deep upload validation** — `_probe_media` (ffmpeg readability + stream +
  dimensions) gates `save_visual_override`: corrupt / empty / unsupported rejected
  with beginner messages; the scene's ORIGINAL visual is preserved (override only
  recorded after the probe passes).
- **Tests** `tools/test_review_editor_repairs.py` **22/22** (P2/P3 added). Chrome
  MCP: Menu/popover/•••/timeline/select/preview all verified, **0 console errors**.
  Real re-render through the new UI validated (reorder + valid replace + rejected
  invalid upload + regen + globals). dist 0-drift; AI video OFF. P5/P6 Menu+popover
  were polished in V1.1 (re-verified; no redundant items added).

## Review Editor V1.1 — **CapCut-Clean UX** (2026-06-02, snapshot `Vidlore_ReviewEditor_V1.1_CapCutCleanUX`)

UX-simplification pass (CapCut-*inspired*, not copied). **Only `vidlore/web.py`
changed** (editor frontend); render pipeline byte-identical to V1.0. Audit:
`research/review_editor/CAPCUT_UX_AUDIT.md` (6-area read-only audit; the drag-drop
agent ran away and was stopped — that area was lead-specced). Manual QA:
`CAPCUT_UX_MANUAL_QA.md`.

- **Toolbar decluttered** — now just `Undo · Apply & re-render · ☰ Menu` (was 3
  buttons + an always-on 6-control audiobar).
- **☰ Menu dropdown** — grouped Project / Video settings / Advanced; reusable
  `.edpop` popup primitive; opens on click, closes on outside-click + Escape; only
  genuinely-supported actions (Reset-all moved here from the toolbar).
- **Whole-video settings popover** — the music/volume/captions/look controls moved
  off the always-visible surface into a centered "Whole video settings" popover
  (compact summary line under the preview opens it). `paintAudio` fill logic
  unchanged (IDs preserved) → zero regression.
- **Scene-row ••• overflow menu** — secondary actions (Re-voice / Reset / Preview-
  or-Restore-card / Delete) tucked behind a hover ••• button; does not select the row.
- **Smooth drag-drop** — reorder DISABLED while a render runs (`__edDragStart`
  guards on `#edexport.disabled`); grab/grabbing cursors + accent insertion line
  retained.
- **Timeline** — scene-number-primary block labels (`.blknum`, narration appended
  when wide, full narration in tooltip).
- **Wording** — "Music level"→"Music volume", "Fit to width"→"Fit whole video".
- **Validation** — Chrome MCP: Menu/popover/••• open+close (outside+Escape),
  scene-select→seek, timeline-click→seek, preview play/mute, 33 timeline number
  chips; **0 console errors**. dist 0-drift (`web.py` synced Mac+Win). Real
  re-render through the new UI confirms the render trigger still works (backend
  untouched). AI video OFF.

## Review Editor V1.0 — **CapCut Beginner UX** (2026-06-02, snapshot `Vidlore_ReviewEditor_V1.0_CapCutBeginnerUX`)

The editor (`/e/<slug>` in `web.py` + `editor_manifest.py`) was **unfrozen** and
fully repaired across two passes. A 15-agent read-only audit
(`research/review_editor/EDITOR_FUNCTIONAL_AUDIT.md`, 84 controls / 212 findings)
drove sequential fixes; full writeup in
`research/review_editor/REVIEW_EDITOR_V1_FINAL_REPORT.md`.

**Pass 2 (P1-P6), all validated by a real comprehensive re-render + snapshotted:**
- **P1 Responsive** — fluid `clamp(18vw)`/`clamp(22vw)` panel defaults + stack
  breakpoint 1140→1000; 3-panel & preview-dominant from 1024→2560 (no overflow;
  laptops/zoom keep 3 panels); timeline hover tooltips.
- **P2 Scene sync** — `assemble.py` emits exact `scene_starts`/`scene_durations`
  into `render_meta.json`; `build_manifest` uses them (`duration_source=render_meta`,
  word-proportional fallback). No more off-by-one seek/highlight.
- **P3 QA metrics** — re-render writes `render_metrics.json` (real black-frame +
  loudness → PASS/WARN/FAIL), "Checking final output" stage, honest beginner
  wording; never writes stale/fake on failure.
- **P4 Slug robustness** — `render_from_script(run_dir=…)`: a re-render targets the
  editor's EXACT dir, never recomputes from `_slug(title)` (fixes the quote-title
  mismatch permanently).
- **P5 Upload safety** — 300 MB server cap (measured pre-read) + friendly
  type/size/missing-file errors.
- **P6 Friendly 404** — branded "project could not be found" + Back to dashboard.
- Tests: `tools/test_review_editor_repairs.py` **16/16**. Real re-render proof:
  magenta upload rendered into scene 5 (RGB 164,14,162), QA verdict PASS, 0 black,
  −16.1 LUFS, 0 temp leak, dist 0-drift, `_after7` reference intact. Snapshot
  builder `tools/build_review_editor_snapshot.py`.

- **Headline fix — Replace-Visual now reaches the render.** `apply_overrides`
  writes `edits/locked_visuals.json` (final-scene-index → uploaded file);
  `pipeline.py` overrides `fr.items`/`fr.beat_clips` right after `fetch_footage`.
  **`footage.py`/`assemble.py` untouched.** Proven in a real MP4 (a magenta test
  upload rendered into scene 5; scene card preserved on top).
- Also fixed: stale preview after re-render (`?v=video_mtime` cache-bust);
  **pending clears on success** + **regen no longer double-bumps** (new
  `mark_rendered()` records a rendered-signature + clears one-shot flags, called
  from `_run_job` on success; `pending_summary.rendered_clean`); export
  duplicate-click guard; **delete empty-project guard** (UI + server); Volume→
  "Music level" + "WHOLE VIDEO" scope chip; Reset-all danger styling; honest
  metrics-missing wording; **card Restore ≠ Reset scene** (`restore_card`);
  card-body preserve for map/default kinds; reset-scene restores poster;
  first-open timeline build uses `thumbs=False`.
- **Tests:** `tools/test_review_editor_repairs.py` (13/13). Real re-render: 0 black
  frames, −valid A/V, 0 temp leak, `_after7` reference untouched. **dist 0-drift**
  (web.py / editor_manifest.py / pipeline.py sha-verified src=Mac=Win).
- **Gotcha (durable):** re-render recomputes its dir as `output/_slug(brief.title)`,
  which can differ from the editor's dir if the title has odd chars (this project's
  brief.title has embedded quotes → 49-char slug vs a 50-char dir). Normal projects
  are fine; deferred robustness fix = pass the editor run_dir through.
- **Still deferred (minor):** a true file-lock to dedupe the parallel
  manifest+timeline first-open build (P1 `thumbs=False` already halves the work);
  deep upload codec validation (currently relies on assemble's render-time slate
  fallback for unreadable clips); a full timeline-label rebuild (mitigated by
  hover tooltips + zoom). AI video stays OFF.

## Vidlore V1.2 — **Local Visual-Relevance Scorer** (2026-06-02, snapshot `Vidlore_V1.2_VisualRelevance`)

Pixel-aware footage validation — the footage ladder no longer accepts a clip
whose metadata/slug *sounds* right but whose *pixels* are wrong. Parent
`Vidlore_V1.1_FinalVisualPolish` (untouched). **Opt-in + flag-gated:** with
`VIDLORE_VISUAL_RELEVANCE` off (default), the render path is **byte-identical to
V1.1** — zero regression risk.

- **New module `vidlore/visual_relevance.py`** — local **ONNX CLIP** (Qdrant
  ViT-B/32 vision+text via `onnxruntime` CoreML/CPU; CLIP BPE via `tokenizers`).
  No torch, no API; ~600 MB one-time model cache in `~/.cache/vidlore_clip/`
  (downloaded on first use, **not** bundled in dist). Signals: CLIP relevance
  (zero-shot vs distractors), clarity (Laplacian var), darkness/info, period-risk
  (CLIP modern-element prompts on pre-1945 scenes), face-mismatch (Haar),
  repetition (avg-hash). ~0.3 s/frame.
- **Integration** (`footage.py` post-selection pass, flag-gated) — pixel-checks
  each concrete scene's beats; a reject escalates: retry stock → better real
  footage → subject-true AI still. Repetition only across scene beat-0.
- **Validation** — unit: **9/10, perfect precision** (catches 0:32 aerial road via
  period-risk + 1:35 dim person via clarity; zero good-footage false-rejects;
  0:18 archival is the honest residual). Live re-render (`output/_after7`, same
  33-scene script): **70 beats → 3 pixel-reject → 3 better real stock, 0 AI,
  0 kept**; black=0; −16.1 LUFS; subs intact. Three live-pipeline bugs found+fixed
  (era dict→year [period gate was dead], intra-scene repetition, beat-0 coverage).
- **Honest residual** — 0:18 archival group still passes (ViT-B/32 limit; larger
  model would help); period floor 0.55 is borderline for the aerial road (improved
  here via reshuffle, not a direct reject — recommend ~0.50 for historical
  niches). Net delta on this agriculture video = 3 clear fixes + neutral reshuffle;
  live value is larger on niches that reach the generic stock tier.
- **Package**: `research/visual_relevance/final_manual_review/` (V1.1 BEFORE vs
  V1.2 AFTER, `V11_vs_V12_compare.png`, rejected-candidate log, report).
- **Rollback**: `VIDLORE_VISUAL_RELEVANCE=0` (instant → V1.1 behaviour).

## Vidlore V1.1 — **Final Visual Polish** (2026-06-01, snapshot `Vidlore_V1.1_FinalVisualPolish`)

Evidence-validated engine fixes on an EXACT same-script re-render (`output/_after6`,
`sha_match=True`, 200.73 s, AI **video OFF**). Parent `Vidlore_V1.0_FinalForensicTune`
(untouched). Changed modules synced Mac+Win + sha-verified: `assemble.py`,
`footage.py`, `pipeline.py`.

- **Black-frame ELIMINATED (the #1 blocker)** — root cause was a near-black/empty
  *source clip* (beat 4) whose dark TAIL `blackdetect` (pic_th=0.98) couldn't see,
  so the freeze-fill missed it and the grade crushed it to black — while metadata
  read "clean". Fix: concat **re-encode** (was `-c copy`) + new **`_detect_dark_spans`**
  (signalstats; YAVG<26 **AND** YMAX<130 → empty, not just pure-black) + **iterate-
  until-clean** driver. DIRECT audit on final mp4 = **0** accidental pure-black gaps;
  the failed beat is now bright; a legit dark scene (104–106 s, YMAX≈253) is correctly
  **preserved** (calibrated against real frames — no over-rejection).
- **Injected-stat SFX** — director-injected `gold_number_callout` etc. now voiced
  (pipeline forwards `{scene:primitive}` → assemble schedules a ducked reveal→settle,
  throttle+cooldown-gated). The 2:23 number beat is no longer silent.
- **Readability** — `footage._slide` headline alpha-28 (~11 %) → high-contrast
  off-white + shadow scrim. **Restraint + niche-aware clutter guard** (collage/montage
  on calm niches; premium cards always survive). **Dark-empty arm** added to the
  footage blank-clip guard.
- **QA: 15/16 gates PASS.** Partial: **footage relevance** — 3 beats (32 s aerial road,
  18 s archival, 95 s dim person) are on-theme but subject-wrong, the "topically-
  plausible / pixel-wrong" class the **query-based** matcher can't catch. NOT a
  regression; needs a CLIP/vision relevance scorer (recommended next). AI video=0,
  slides=0, LUFS −16.1/peak −1.9, subs present, temp leak=0, USE-ONLY in dist=0.
- Package: `research/final_forensic_compare/two_cheap_metals/final_manual_review/`
  (AFTER_V11 mp4, contact sheet, black-fix before/after clips, `QA_GATES_REPORT.md`).
- **Preserved**: AudioEngine_V1.2 mix files untouched (SFX change only *schedules* an
  existing cue); editor UI frozen; no AI video.

## Audio Engine V1.0 — **Stable** (Music Director + SFX Director + cue sheets + QA gate + cross-video anti-rep)

A permanent, reusable **professional documentary music + sound-design engine**,
built ON TOP of the mature audio core (preserved verbatim) and **additive +
import-guarded** (a failure reverts to legacy behaviour). New package
`vidlore/audio_director/` (`music_director` · `sfx_director` ·
`audio_usage_history`) + library `vidlore/audio_library/` (manifests + the
adversarially-verified taxonomy/intro/mix/SFX-policy specs).

- **Library manifests** — `music_manifest.json` (118 tracks, full tag + license
  schema, measured LUFS) + `sfx_manifest.json` (123 synth presets). License gate:
  **91 bundle-OK CC-BY + synth; 27 Mixkit USE-ONLY** excluded from dist
  (`dist_exclude.txt`). Built by `tools/build_audio_manifests.py`.
- **Music director** — niche INTRO intelligence (louder-then-recede: spy restrained
  pulse, business confident swell, mystery silence-heavy), per-niche reveal-duck
  character, cross-video category biasing, `music_cue_sheet.json`.
- **SFX director** — per-primitive restraint (silence-default cards stay silent;
  intensity caps; niche density), `sfx_cue_sheet.json`. Validated: business 0.46
  SFX/min (footage-led, sparse), spy 4.13/min with motivated **foley_doc/data/
  timeline** (not whooshes).
- **Cross-video anti-rep** — category + SFX-family cooldowns + deterministic
  per-video seeding (`audio_usage_history.json`), on top of the existing track-level
  persistence.
- **QA gate** — `tools/audio_quality_audit.py` → `audio_quality_report.json`
  (LUFS, true-peak/clip, intro-vs-body, mud, dead-air, dialogue/duck, SFX density,
  repetition, **license completeness + provenance**) PASS/WARN/FAIL.
- **Preserved** — musiclib selection/scoring/reveal-tiers, sfx synthesis, assemble
  two-stage mux + the **empirically-tuned base sidechain duck** (NOT replaced); the
  per-niche bed character already in look DNA (NOT double-applied).
- **Tests** — `tools/test_audio_director.py` **66/66** · `test_engine_guards.py`
  **87/87**. Validation: 5 niche renders (spy/crime/business/history/geopolitics),
  QA gate green, 0 black frames, ~−16 LUFS. Reports:
  `research/audio_engine/AUDIO_ENGINE_V1_REPORT.md`, `CROSS_NICHE_AUDIO_REPORT.md`.

- **YTAL library expansion (USE-ONLY, license-clean)** — license-aware ingester
  (`tools/ingest_ytal.py` + `merge_ytal_music.py` + `curate_ytal_sfx.py` +
  `recover_ytal_credits.py` + `ingest_local_music.py`). Ingested **72 CC BY 4.0
  documentary music tracks (72/72 with verified verbatim credit** — Scott Buckley,
  Savfk, Nakarada, Kevin MacLeod …) + **37 documentary SFX** (official YTAL,
  no-attribution). Selectable music library **118 → 190 (+61%)**. Raw files are
  **USE-ONLY**: git-ignored `vidlore/audio_library/ytal_cache/`, merged into
  `musiclib.scan()` at render time only (`VIDLORE_YTAL_USE_ONLY`, default on),
  **never bundled in dist**. Per-video `MUSIC_CREDITS.txt` auto-written for any
  attribution-required track (proven on the spy render). Official no-attribution
  *music* (login-gated) → safe local-folder step, no credentials. See
  `research/audio_engine/YTAL_INGESTION_REPORT.md`.

Snapshot: `snapshots/AudioEngine_V1.0_MusicSFXDirector/` (sibling of V2.3, which is
untouched; builder `tools/build_audioengine_snapshot.py`). No new MG primitives;
editor UI frozen; not deployed.

## Motion Graphics Cluster V2.3 — **Stable** (Asset guards: pre-photo portrait / period footage / niche palette / dark card / QA · 39 primitives)

The motion-graphics engine is integrated, validated, and frozen at **V2.3**. **Thirty-nine**
primitives, director, registry, dispatcher, cache, manifest, black-frame repair,
subtitles, local voices, AI-image fallback, stock footage, web-image filters
(incl. a near-white **blank-bright clip reject** guard, symmetric to the
black-frame repair), Mac+Win sync, legacy flag-off safety — all in place. V2.3 adds
five **permanent reusable asset guards** (below). **Editor UI is frozen.**

Snapshot: `snapshots/MG_Cluster_V2.3_AssetGuards/` (source copies · `HASHES.txt` ·
`SNAPSHOT_MANIFEST.json` · builder `tools/build_v23_snapshot.py` · 68 files · 5 new
+ 3 changed · all_zero_drift src=mac=win). The prior
`MG_Cluster_V2.2.1_DispatchKwargGuard/` (parent, untouched),
`MG_Cluster_V2.2_SpotlightDecisionArc/`,
`MG_Cluster_V2.1_RegionCauseGauge/`,
`MG_Cluster_V2.0_CountdownWebQuotes/`, `MG_Cluster_V1.9_RankedSankeyEra/`,
`MG_Cluster_V1.8_HeadlinesHeatRedacted/`,
`MG_Cluster_V1.7_DefinitionBalanceBeforeAfter/`,
`V1.6_StatementPictographComposition/`, `V1.5_ProportionProcessHierarchy/`,
`V1.4_GrowthAnnotationRoute/`, `V1.3_EvidenceDataLocation/`,
`V1.2.1_TempDirCleanup/`, `V1.2_StorytellingBeats/`,
`V1.1.2_EncodePoolReliability/`, `V1.1.1_PortraitFix/` and `V1.1_Stable/` are
**kept untouched** as rollback points.

### V2.3 — Permanent reusable asset guards (engine-level, from multi-niche validation)
Five **reusable, unit-tested** modules that automatically prevent four issue
classes for ALL future videos/topics (not just the validation samples), plus a QA
layer. Pure-logic cores (`tools/test_engine_guards.py` — **87 assertions**: 8
portraits, 6 eras, palette weighting/variation/anti-repeat, dark-niche cards, QA
checks); all hooks are defensive (try/except → legacy behaviour on any failure):
- **`portrait_intel.py`** — detects PRE-PHOTOGRAPHIC people (Napoleon †1821, Caesar,
  Washington…) and prefers a verified painting/engraving/PD-illustration over a
  modern photo or AI face; strict name match; provenance; an AI fallback for such
  people requests a period OIL PAINTING, never a photo. Wired into footage.py
  `_real_person_image` (Wikimedia artwork queries) + the name_reveal AI prompt.
- **`period_guard.py`** — era detection + period-risk + modern-marker rejection
  (cars/glass/skylines/highways) + era-biased stock queries + safe fallback order;
  ambiguous nationality words ("Egyptian-born") no longer false-trigger antiquity,
  and explicit modern years override. Wired into footage.py `_pexels_queries` +
  `fetch_footage` (per-video era hint).
- **`niche_palette.py`** — niche-aware WEIGHTED palette (true-crime → ember_red /
  cold / muted-dark, never warm business gold) + deterministic per-video variation +
  cross-video anti-repeat + reason logging. Wired into director `video_palette` (the
  single palette chokepoint) + `video_palette_reason`.
- **`card_style_guard.py`** — dark niches (spy/mystery/crime/intelligence/covert)
  route a bright full-screen "statement" card to the dark mono variant unless an
  authorised RARE contrast beat. Wired into footage.py's statement-card dispatch.
- **`asset_qa.py`** (+ `tools/render_asset_qa.py`) — reusable QA layer that flags
  mismatched / pre-photo-modern-face portraits, modern footage in period scenes,
  palette-niche mismatch, bright card in dark niche, uncertain provenance — low
  confidence → a safer fallback was already chosen by the guards; the warning is
  recorded into `motion_graphics_manifest.json` (pipeline) so nothing doubtful is
  silently accepted.

**Regression-validated** by re-rendering Napoleon (history) / Al Capone (crime) /
Eli Cohen (spy): Napoleon → period **oil-painting** portrait + period-neutral/vintage
footage (the modern-town aerial is gone); crime palette → **ember_red** (on-genre,
variation preserved); spy statement card → **dark** (luma ~30, was ~230). Gates held
(0 black · −16.1/−16.2 LUFS · peak ≤ −1.8 dBFS · 0 temp · 0 fallbacks). No new MG
primitives (39 unchanged); `assemble.py`/`look.py` byte-identical. Limitation: the
period guard is TEXT-based (era ↔ query/title) — true frame-level modern-object
detection is a future enhancement; the fallback order keeps results safe meanwhile.
See `research/motion_graphics_qa/engine_guards_report.md`.

### Multi-Niche Real Validation (V2.2 → V2.2.1)
Five original short documentaries rendered through the **real** pipeline
(`VIDLORE_MOTION_GRAPHICS=1`, real footage, web images, PD portraits, AI fallback,
local voice, subtitles, music, restrained SFX), one per niche — **spy** (Eli Cohen),
**true-crime** (Al Capone), **business** (Andrew Carnegie), **history** (Napoleon
1812), **geopolitics** (Cuban Missile Crisis). Graphics were NOT forced; the Director
chose naturally at density 0.35 (≈5 graphics per 13–15 scenes). All five passed the
hard gates — **0 sustained black · ~−16 LUFS · true-peak ≤ −1.9 dBFS · 0 temp leak ·
0 fallbacks · 17 distinct primitives · no repeated primitive *sequence* across
niches**. The 3 newest primitives all fired premium in real renders (world_map_arc
in spy+geopolitics, flowchart_decision in crime+geopolitics, spotlight_object_hold in
history). Strongest niche: **geopolitics**; weakest: **history** (portrait + footage
misses below). Harness: `tools/multiniche_validate.py` + `tools/multiniche_audit.py`;
reports under `research/motion_graphics_qa/multiniche/` (per-sample QA + cross-niche).

**One real bug found + fixed → V2.2.1:** `render_dispatch.dispatch()` now filters
render kwargs to each primitive's declared signature. A `name_reveal` scene folds a
`place=` hint into its assets (for the sibling `portrait_name_over_map`); when the
director instead picked `cinematic_portrait_hold` (which has no `place` param), the
stray kwarg crashed the render → the **subject portrait silently fell back** (hit
Capone/Carnegie/Napoleon). Filtered now; the 3 affected samples re-rendered with 0
fallbacks and portraits restored. Only `render_dispatch.py` changed (footage.py /
assemble.py / look.py / director.py / registry.py / pipeline.py byte-identical to
V2.2). **Recommendations left for review** (touch stable subsystems, not fixed):
palette repetition / crime→amber-not-ember; Napoleon (pre-photographic) portrait
likeness; anachronistic modern footage in period docs; a light kinetic-text card in
dark-palette videos. **No missing storytelling capability found — no new primitive
warranted.**

### V2.2 — Batch 11: 3 spotlight/decision/world-arc primitives (36 → 39)
From a MagnatesMedia re-analysis: **spotlight_object_hold** (`reveals/` — a text-led
"behold" reveal: a near-black stage; a warm-gold radial spotlight pool sweeps in
from an edge (easeOutExpo) and settles centre, lifting a bold serif SUBJECT out of
darkness as the beam reaches it; KICKER above, sub below, title on a hairline — NO
frame, NO photo, distinct from `framed_evidence_spotlight`), **flowchart_decision**
(`diagrams/` — a single yes/no DECISION FORK: a gold-outlined diamond question node
at top, two diverging connectors to outcome cards with YES/NO chips; `chosen`
ignites the taken path to gold and dims the other — a Y-fork, not a row, distinct
from `cause_effect_chain`), **world_map_arc** (`maps/` — ONE great-circle ARC bows
from an origin city to a distant destination over a purpose-built antique world
chart, a comet head tracing it in, pulsing pins + gold city labels; two points, one
link, distinct from `map_route_spread`). Built + micro-rendered via a **3-agent
parallel workflow**, then each frame human-QA'd premium (`world_map_arc` polished
after QA: its own `_world_bed` — clean lat/long graticule + even aged-paper
mottling, no blobs — and a frame-spanning composition). All reuse the `look.py`
layer; director `_REVEAL`/`_DECISION`/`_ARC` cues + affinity + scoring; additive
pipeline-adapter branches parsing `subject` (via `_gt`) + `kicker=`/`sub=` /
`question` + `yes=`/`no=`/`chosen=` / `from_place` (via `_gt`) + `to=`/`from_pos=`/
`to_pos=` hints. `flowchart_decision` + `world_map_arc` are `intensity_range [2,4]`;
`spotlight_object_hold` `[3,5]`. The legacy footage.py `spotlight` card builder only
builds an image (never mutates the kind) so it stays out of the way; validation uses
collision-free kinds `reveal`/`decision`/`world_arc`. No stable renderer files
modified this batch (footage.py / assemble.py / look.py byte-identical);
`location_establish_card` byte-identical (`world_map_arc` carries its own
`_world_bed`). Validated on a fresh Petrov-1983 (nuclear false-alarm) render
exercising all 3 + chronology + spectrum: manifest 5 graphics rendered · 0
fallbacks · 0 sustained black frames (blackdetect pix_th=0.10) · −16.1 LUFS · 83s ·
all 3 new cards frame-verified premium · all five graphic scenes different families
(no same-family clash, spaced 2 apart) · ~$0.015. Each self-cleans its PNG temp dir
(V1.2.1 fix preserved). See `research/motion_graphics_qa/batch11_render/`.

### V2.1 — Batch 10: 3 region/cause-effect/spectrum-meter primitives (33 → 36)
From a MagnatesMedia re-analysis: **map_region_highlight** (`maps/` — a graded
period map with ONE region singled out by a soft gold glow + an irregular
hand-drawn boundary + a pin + name/sub, the rest dimmed), **cause_effect_chain**
(`diagrams/` — 2-4 cause cards in a domino row linked by bold gold chevrons,
revealed left→right, the final OUTCOME card crowned in gold), **spectrum_meter**
(`meters/`, NEW family — a qualitative gauge: a cool→warm gradient band track with
labelled bands + a gold needle that sweeps to the value and settles, the landed
band brightening, a big serif readout above). Built + micro-rendered via a
**3-agent parallel workflow**, then each frame human-QA'd premium. All reuse the
`look.py` layer; director `_REGION`/`_CAUSAL`/`_GAUGE` cues + affinity + scoring;
additive pipeline-adapter branches parsing `region` (via `_gt`) + `pos=`/`sub=` /
`steps=` / `value=`/`bands=`/`readout=` hints. `spectrum_meter` requires `{value}`
(low score without a gauge kind/cue, so it never steals plain-number scenes); the
legacy footage.py `map_region`/`cause_effect` card builders only build images
(never mutate the kind) so they stay out of the way — no stable renderer files
modified this batch (footage.py / assemble.py / look.py byte-identical). Validated
on a fresh Road-to-War (appeasement) render exercising all 3 + portrait +
chronology: manifest 5 graphics rendered · 0 fallbacks · 0 sustained black frames
(one 0.5s dissolve dip) · −16.2 LUFS · 75s · all 3 cards frame-verified premium ·
all five graphic scenes different families (no same-family clash) · ~$0.015. Each
self-cleans its PNG temp dir (V1.2.1 fix preserved). See
`research/motion_graphics_qa/appeasement_batch10/`.

### V2.0 — Batch 9: 3 countdown/connection-web/quote-stream primitives (30 → 33)
From a MagnatesMedia re-analysis: **countdown_clock** (`clocks/`, NEW family — a
circular dial whose depleting gold arc + ticking serif count run down toward zero,
warming to ember as time runs out; a gold head marker rides the arc's leading
edge), **connection_web** (`diagrams/` — named nodes ring the frame and thin
gold/ember lines draw between the linked ones, a non-hierarchical web of
who-knew-who with a brighter hub), **quote_stream** (`quotes/` — two-to-four short
quotations cascade in and stack, each with a gold quote-mark and an attribution: a
chorus of voices). All reuse the `look.py` layer; director `_COUNTDOWN`/`_NETWORK`/
`_CHORUS` cues + affinity + scoring; additive pipeline-adapter branches parsing
`value=`/`to=`/`unit=` (+ label via `_gt`) / `nodes=`/`links=` / `quotes=`
(`text:attrib|…`). `countdown_clock` requires `{value}` and `quote_stream` requires
`{quotes}`, so the legacy footage.py `countdown`/`quote_stream` card builders (which
only build images, never mutate the kind) stay out of the way when no MG inputs are
supplied — no stable renderer files were modified this batch (footage.py /
assemble.py / look.py byte-identical). Validated on a fresh Teapot-Dome render
exercising all 3 + portrait + chronology: manifest 5 graphics rendered · 0
fallbacks · 0 sustained black frames (7s/45s flags = dark-but-detailed courtroom
footage, 77 KB) · −16.1 LUFS · 85s · all 3 cards frame-verified premium · all five
graphic scenes different families (no same-family clash) · ~$0.015. Each
self-cleans its PNG temp dir (V1.2.1 fix preserved). See
`research/motion_graphics_qa/teapot_dome_batch9/`.

### V1.9 — Batch 8: 3 ranked-list/sankey-flow/era-band primitives (27 → 30)
From a MagnatesMedia re-analysis: **ranked_list_countdown** (`charts/` — a Top-N
leaderboard; rows drop in bottom-rank UP to #1, each with a rank numeral, label,
proportional bar and value; #1 crowned in gold), **sankey_flow** (`charts/` — a
source column splits into proportional-width gold bezier ribbons flowing to
labelled branches, thin dark outlines keeping adjacent ribbons distinct),
**era_band_timeline** (`timelines/` — a horizontal time axis divided into labelled
era bands whose WIDTH is their span, wiping in left→right with a dim→bright tonal
ramp + year markers). All reuse the `look.py` layer; director `_RANK`/`_FLOW`/
`_ERA` cues + affinity + scoring; additive pipeline-adapter branches parsing
`items=` / `branches=` / `eras=` (+ `source` / `prefix` / `suffix` / `title`
hints). The new `ranked_list_countdown` requires `{items}`, so the legacy
footage.py `ranking` carousel (per-scene, footage-backed) is untouched when no
item list is supplied — no stable renderer files were modified this batch
(footage.py / assemble.py / look.py byte-identical). Validated on a fresh
Robber-Barons render exercising all 3 + portrait + location: manifest 5 graphics
rendered · 0 fallbacks · 0 sustained black frames · −16.1 LUFS
· 85s · all 3 cards frame-verified premium · the two charts-family cards (ranked@3,
sankey@7) sat 4 scenes apart so the same-family guard held · ~$0.015. Each
self-cleans its PNG temp dir (V1.2.1 fix preserved). See
`research/motion_graphics_qa/robber_barons_batch8/`.

### V1.8 — Batch 7: 3 headlines/heat-spread/redacted primitives (24 → 27)
From a MagnatesMedia re-analysis: **headline_montage** (`media/`, NEW family —
three-to-five period headlines cascade in as aged-newsprint clippings, each
rotated and overlapping, latest centred + upright on top: a press storm),
**map_heat_spread** (`maps/` — warm ember→amber heat blooms ignite in sequence
and spread across a graded antique map; soft pins + labels; dimmed-additive glow
so the map shows through, no white blow-out), **redacted_document**
(`documents/` — a classified page where black redaction bars sweep across typed
lines while ONE line stays legible, and a red TOP-SECRET/DECODED stamp lands askew
low over the bars). All reuse the `look.py` layer; director `_PRESS`/`_HEAT`/
`_SECRET` cues + affinity + scoring; additive pipeline-adapter branches parsing
`headlines=` / `hotspots=` / `reveal` (via `_gt`) / `title=` / `stamp=` hints
(+ `_gimg`→`map_image` for the heat bed). The `redacted`/`spread` affinities were
retargeted to the new primitives.

**Two integration-collision fixes were required (both general, not Batch-7-only):**
(1) the weak-footage AI-explainer (`pipeline.py`) clobbered any deliberately-
tagged graphic beat whose kind wasn't in a hard-coded allow-list — replaced with
`director.known_graphic_kinds()` so EVERY batch's kinds are auto-protected;
(2) `footage.py` rerouted `classified`/`redacted` → a flat `document` card AND
mutated `sc.graphic_kind`, hiding the tag from the MG director — now gated so when
the MG engine is on those kinds pass through to `redacted_document` (`case_file`
still reroutes; MG-off path byte-identical). Validated on a fresh Zimmermann-
Telegram render exercising all 3 + portrait + timeline: manifest 6 graphics
rendered · 0 fallbacks · 0 sustained black frames (32–36s flag = dark-but-detailed
Room 40 footage; one 0.1s dissolve dip) · −16.1 LUFS · 85s · all 3 cards frame-
verified premium · ~$0.015 (fal images, mostly cache-hit). Each self-cleans its
PNG temp dir (V1.2.1 fix preserved). See
`research/motion_graphics_qa/zimmermann_batch7/`.

### V1.7 — Batch 6: 3 definition/balance/before-after primitives (21 → 24)
From a MagnatesMedia re-analysis: **definition_card** (`statements/` — a
dictionary-style TERM in gold serif + part-of-speech tag + rule + definition
beneath), **vs_balance_scale** (`scales/`, NEW family — two labelled forces hang
from a beam that tips toward the heavier side and settles with a small wobble),
**before_after_slider** (`reveals/`, NEW family — a lit vertical seam wipes an
"after" over a "before" of the same frame, graded vs degraded; warm/cool then-now
panels when no image resolves). All reuse the `look.py` layer; director `_DEFINE`/
`_BALANCE`/`_TRANSFORM` cues + affinity + scoring; additive pipeline-adapter
branches parsing `definition=` / `pos=` / `pair=` / `values=` / `before=` /
`after=` hints (+ `_gimg` resolution for the wipe image). Validated on a fresh
Industrial-Revolution render exercising all 3 + portrait + location: manifest 5
graphics rendered · 0 fallbacks · 0 sustained black frames · −16.2 LUFS · 76s ·
no_template_feel 10.0 · ~$0.018. Each self-cleans its PNG temp dir (V1.2.1 fix
preserved). See `research/motion_graphics_qa/industrial_rev_batch6/`.

### V1.6 — Batch 5: 3 statement/pictograph/composition primitives (18 → 21)
From a MagnatesMedia re-analysis: **statement_card** (`statements/`, NEW family —
one bold editorial CLAIM in large serif, lines rising in, a key phrase
gold-underscored, optional source tag; the narrator's thesis), **pictograph_scale**
(`charts/` — a grid of figure icons, first N lit gold + rest muted, making a ratio
countable like "3 IN 10"), **composition_stack** (`charts/` — one 100% horizontal
bar split into labelled segments on a gold→muted ramp, wiping in left→right, e.g.
"where every dollar went"). All reuse the `look.py` layer; director `_CLAIM`/
`_PICTO`/`_COMPOSITION` cues + `_RATIO_RE` ("N in M") derivation + affinity +
scoring; additive pipeline-adapter branches parsing `emphasis=` / `source=` /
`count=` / `total=` / `segments=` / `suffix=` hints; the two chart primitives are
held >2 scenes apart by the same-family guard. Validated on a fresh Startup-Failure
render exercising all 3 (statement ×2, pictograph, composition): manifest 4
graphics rendered · 0 fallbacks · −16.1 LUFS · 76s · text_restraint 10.0 ·
no_template_feel 10.0 · ~$0.015. Each self-cleans its PNG temp dir (V1.2.1 fix
preserved). See `research/motion_graphics_qa/startup_fail_batch5/`.

### V1.5 — Batch 4: 3 proportion/process/hierarchy primitives (15 → 18)
From a MagnatesMedia re-analysis: **proportion_ring** (`charts/` — a parts-of-a-
whole share: a gold arc sweeps from 12 o'clock filling to the %, with a centre
count-up numeral + a label naming what the share is OF, e.g. 90%),
**process_flow_steps** (`diagrams/`, NEW family — 2-5 numbered nodes left→right
joined by arrows that draw in one after another with short labels: an ordered
mechanism / scheme), **org_hierarchy_tree** (`diagrams/` — a root over 2-4 child
nodes joined by clean elbow connectors, revealed top-down: a power / corporate
structure). All reuse the `look.py` layer; director `_SHARE`/`_PROCESS`/
`_HIERARCHY` cues + affinity + scoring + share-derivation; additive pipeline-
adapter branches parsing `share=` / `label=` / `sub=` / `steps=` / `children=` /
`title=` hints; the two node-diagram primitives are held >2 scenes apart by
`INCOMPATIBLE_ADJACENT`. Validated on a fresh Standard-Oil render exercising all 3
+ portrait + location: manifest 5 graphics rendered · 0 fallbacks · **0 black
frames** · −16.1 LUFS · 78s · no_template_feel 10.0 · ~$0.015. Each self-cleans
its PNG temp dir (V1.2.1 fix preserved). See
`research/motion_graphics_qa/standard_oil_batch4/`.

### V1.4 — Batch 3: 3 growth/annotation/route primitives (12 → 15)
From a MagnatesMedia re-analysis: **growth_curve_chart** (`charts/` — a smooth
Catmull-Rom time-series that draws in with a glowing plot-head + serif count-up +
gridlines + x-labels lighting as the head passes; a continuous trend, e.g. 4 → 55
→ 310), **annotated_detail_callout** (`annotations/`, NEW family — points INTO a
photo: local spotlight crushes everything but a disc around a focus point, a
bright ring + cardinal ticks + a leader line to a label chip; premium aged
archival plate when no photo resolves), **map_route_spread** (`maps/` — a route
draws across a graded antique map: ghost path → bright traced line + comet-head +
origin/destination pins + waypoint names lighting as the head passes; an
expansion / journey). All reuse the `look.py` layer; director `_GROWTH`/`_ROUTE`/
`_DETAIL` cues + affinity + scoring; additive pipeline-adapter branches parsing
`points=` / `focus=` / `tag=` / `stops=` / `suffix=` / `prefix=` hints. Validated
on a fresh Transcontinental-Railroad render exercising all 3 + location: manifest
4 graphics rendered · 0 fallbacks · −16.1 LUFS · 80s · ~$0.012. Each self-cleans
its PNG temp dir (V1.2.1 fix preserved). Plus a near-white **blank-bright clip
reject** guard in `footage.py`/`assemble.py` (rejects very-bright + near-uniform
library clips that would render as a pale blank flash; gates on LOW variance so
detailed bright shots survive). See `research/motion_graphics_qa/railroad_batch3/`.

### V1.3 — Batch 2: 3 evidence/data/location primitives (9 → 12)
From a MagnatesMedia re-analysis: **framed_evidence_spotlight** (`evidence/` —
artifact in a gold frame under a warm spotlight + EVIDENCE/EXHIBIT tag chip +
naming caption; no-photo path renders a premium aged-paper exhibit document, the
caption typeset as the title with a printed keyline, faded typed body and an ink
seal — never a hollow card), **statistic_bar_reveal** (`charts/` — 1-4 columns
rising from a baseline with count-up numerals + labels + title, e.g. 1 → 12 →
200), **location_establish_card** (`maps/` — place + era over a graded antique
map with corner coordinate ticks, a pulsing pin and a slow establishing push).
All reuse the `look.py` layer; director `_ARTIFACT`/place/bignum cues + affinity +
scoring; additive pipeline-adapter branches parsing `tag=` / `bars=` / `suffix=` /
`prefix=` / `place=` / `coords=` hints. Validated on a fresh Edison-vs-Tesla render
that exercises all 3 alongside the 3 Batch-1 beats: manifest 8 graphics rendered ·
0 fallbacks · 0 black frames · −16.0 LUFS · 134s · ~$0.018. Each self-cleans its
PNG temp dir (V1.2.1 fix preserved). See
`research/motion_graphics_qa/edison_tesla_batch2/` (this render).

### V1.2 — Batch 1: 3 storytelling-beat primitives (6 → 9)
From a MagnatesMedia re-analysis: **chronology_timeline** (`timelines/` — editorial
year-spine / era band), **pull_quote_portrait** (`quotes/` — sourced quote + face-safe
portrait), **comparison_split** (`comparison/` — A-vs-B with VS medallion + value
bars). All reuse the `look.py` layer; director affinity + narration derivation +
additive pipeline-adapter branches. Validated on a fresh Edison-vs-Tesla render
(cold-steel palette): all 3 fire premium, 0 pure-black, −16 LUFS, ~$0.018. See
`research/motion_graphics_qa/edison_tesla_batch1/BATCH1_REPORT.md`. The 6 frozen
primitives are unchanged except for the **V1.2.1** temp-dir cleanup (below).

### V1.2.1 — Temp-dir leak cleanup (maintenance over V1.2)
The 6 original primitives each staged a 1080p PNG frame sequence via
`tempfile.mkdtemp()` for the ffmpeg encode but **never removed it**, leaking
gigabytes of PNGs to the temp dir over long / repeated renders (one session
leaked ~40 GB and nearly hit a disk-full stop). Each now calls
`shutil.rmtree(td, ignore_errors=True)` immediately after the `subprocess.run(...)`
encode and before `return` — the **same self-clean the 3 new V1.2 primitives
already shipped with**. The cleanup runs *after* ffmpeg has read the PNGs, so the
encoded mp4 bytes are **output-identical**; only the leak is fixed.

Failure-path + safety net: `look.cleanup_frames(td)` (marker-guarded — refuses to
delete anything that isn't a frame dir, so it can never touch an output mp4) and
`look.sweep_stale_frames(max_age_s=3600)` (removes orphaned frame dirs left by a
hard crash, but **never** an in-progress render — an age cutoff protects live
work). The pipeline auto-sweeps stale orphans at the start of each MG render, and
a standalone utility **`tools/clean_stale_frames.py`** (`--dry-run` / `--max-age`)
frees temp space on demand. Verified: all 6 `py_compile` OK; each micro-renders to
a valid (ffmpeg-decodable) mp4; **0 frame dirs leaked** across all 6 (count before
== after); the sweep removes stale dirs yet **preserves fresh/active** ones and any
mp4; Mac+Win re-synced, **0-drift**. Snapshot + old→new hashes:
`snapshots/MG_Cluster_V1.2.1_TempDirCleanup/` (builder `tools/build_v121_snapshot.py`).

### V1.1.2 — Encode-pool reliability fix
The default 4-worker parallel encode pool could crash when a scene's footage clip
was momentarily unreadable (partial download / zero-byte / corrupt). **Root cause:**
the tier-4 graded slate was the one unguarded `run()` in `_scene_video`, so under
4-worker fork pressure its subprocess spawn raised `OSError EAGAIN` and `pool.map`
abandoned the render (reproduced deterministically). **Fix (additive, 5 layers):**
`run()` retries transient spawn failures; `_clip_ready()` validates a clip before
the encode tiers; `_safe_slate()` can no longer raise; the pool uses
`submit`/`as_completed` with per-beat retry + emergency slate (one bad beat can't
abort the render); stock downloads are atomic (`.part` → `os.replace`). Validated:
fault-injection render at default 4 workers (the exact original crash conditions) +
3× clean 4-worker + 1× single-worker, all 0-crash / 0-black / 6-MG / $0; portrait
crop intact. See
`research/motion_graphics_qa/encode_pool_reliability/ENCODE_POOL_RELIABILITY_REPORT.md`.

### V1.1.1 — Portrait head-crop / face-safe framing fix
The portrait inside `cinematic_portrait_hold` / `portrait_name_over_map` was
clipping the top of the head. **Root cause:** `footage._cover_to_canvas`
cover-cropped the tall Wikipedia portrait (1495×2048) into 16:9, removing ~775 px
off the top (forehead + hair). **Fix (additive, 3 layers):** face-aware crop
(`look.face_safe_crop` + cv2 Haar on luma, B&W/sepia-tolerant) in both primitives;
a face-safe blur-fill canvas (`footage._portrait_canvas`) replacing
`_cover_to_canvas` at the source; and a real-render adapter preference for the
full-head canvas. **8/8 figures pass**; re-render shows the **whole head** in both
primitives, 0 black frames, −16 LUFS, 6/6 primitives, $0. See
`research/motion_graphics_qa/portrait_crop_fix/PORTRAIT_CROP_FIX_REPORT.md`.

### Validated by
`research/motion_graphics_qa/rockefeller_business_validation_v1_1/final_sample.mp4`
— real Rockefeller portrait · **full head (face-safe crop)** · **0 black frames** ·
all 6 primitives · 0 fallbacks · −16.0 LUFS · 2:11.70 · **$0**.

### Six primitives
`gold_number_callout` · `cinematic_portrait_hold` · `headline_document_reveal` ·
`portrait_name_over_map` · `kinetic_keyword` · `money_flow_empire`.

### Files in the V1.1.2 set (all 0-drift: source = Mac = Win)
- `vidlore/ffmpeg_tool.py` — `run()` + **transient spawn-failure retry**
  (EAGAIN/EMFILE/ENFILE). *[V1.1.2]*
- `vidlore/footage.py` — portrait sourcing (Wikipedia-lead-first, B&W/sepia
  validator, name-match gate, provenance) + **`_portrait_canvas`** (face-safe
  full-head blur-fill) *[V1.1.1]* + **`_stream_download_atomic`** (.part →
  os.replace for stock clips). *[V1.1.2]*
- `vidlore/pipeline.py` — MG hook + adapter (place/branches/label/name),
  per-primitive duration windows, **stable sha1 seed**, density env, protections,
  + **full-head realperson-canvas preference** for name/map cards. *[V1.1.1]*
- `vidlore/assemble.py` — MG slice integration, window clearing, seek clamp,
  black-frame repair *(preserved)* + **`_clip_ready` / `_safe_slate` +
  hardened `submit`/`as_completed` encode pool** (per-beat retry + emergency
  slate). *[V1.1.2]*
- `vidlore/motion_graphics/look.py` — shared look layer + **`_detect_primary_face`
  / `face_safe_crop` / `portrait_blurfill_canvas`** (face-aware, B&W/sepia). *[V1.1.1]*
- `vidlore/motion_graphics/portraits/cinematic_portrait_hold.py` &
  `maps/portrait_name_over_map.py` — now crop via `face_safe_crop`. *[V1.1.1]*
- `vidlore/motion_graphics/` — `director.py`, `registry.py`,
  `render_dispatch.py` (cache `VERSION = mg-0.2.0`, `useful_dur`) + the 6 primitives.
- `vidlore/assets/MGSerif.ttf` (+ OFL/NOTICE).

### Environment flags
| flag | value | effect |
|---|---|---|
| `VIDLORE_MOTION_GRAPHICS` | `1` | enable MG. **Default OFF → byte-identical legacy.** |
| `VIDLORE_MG_DENSITY` | float 0–0.6 | graphics density override (optional) |
| `VIDLORE_REAL_PERSON` | `1` (default) | real-portrait lookup; `0` disables |
| `VIDLORE_AIMG` | `1` | AI-image fallback |
| `VIDLORE_TTS_BACKEND` | `legacy` | free edge-tts |
| `VIDLORE_REUSE_SCRIPT_JSON` | `force` | skip LLM, reuse `script.json` |
| `VIDLORE_ENCODE_WORKERS` | int (default `4`) | parallel encode workers; multi-worker is crash-safe as of V1.1.2 |

### Cache
`vidlore/motion_graphics/render_dispatch.py` → `VERSION = "mg-0.2.0"`. Bump to
invalidate all cached MG clips. MG clips are content-hash cached and now reuse
across renders (stable sha1 seed fix).

### Known limits (DEFERRED — do not implement without approval)
- **Beat-split:** mid-scene footage-return fires only for **multi-beat** graphic
  scenes. A graphic scene rendered as one long beat holds the graphic (a *bright*,
  clean frame — no black, no faded tail) for the scene. The fix (beat-split at the
  window) is a risky assembly-core change; a segment-internal composite was
  trialled and reverted (0.13s boundary artefact). See
  `research/motion_graphics_qa/BEAT_SPLIT_DESIGN_NOTE.md`.
- ~~Encode-pool race (footage path)~~ — **FIXED in V1.1.2.** The default
  multi-worker encode pool now validates clips before encoding, retries
  transient spawn failures, never lets the slate raise, and guards each beat
  (retry → emergency slate) so one bad clip can't abort the render.

### Rollback
See `snapshots/MG_Cluster_V1.1.2_EncodePoolReliability/SNAPSHOT_MANIFEST.json` →
`rollback_steps` (restores pre-EPR V1.1.1). The untouched
`MG_Cluster_V1.1.1_PortraitFix/` and `MG_Cluster_V1.1_Stable/` remain deeper
rollback points. Reliability-only workaround (no rollback):
`VIDLORE_ENCODE_WORKERS=1`. Fastest MG disable: unset `VIDLORE_MOTION_GRAPHICS`
→ legacy path is byte-identical.

### Do NOT (standing constraints)
Editor UI frozen · no new primitives until manual review · no beat-split/composite
until approved · no paid APIs for validation · keep Mac+Win 0-drift.
