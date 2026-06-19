# ClipStudio Relevance + Repetition Fix — Bronn/Hound tavern (2026-06-11)

User report: the portal video (`portal/5b0d55b7d1`) still showed irrelevant footage and repetition
despite the earlier bug-fix passes. Those passes fixed MECHANICAL bugs; relevance is a SEMANTIC
problem that needed its own root-cause work. This pass closes it.

## Root cause (diagnosed from the portal project + competitor video)
1. **The exact scene was NEVER downloaded.** Discovery's anchor query was one long prose string
   ("…Bronn sings Rains of Castamere tavern The Hound confrontation"). Raw-scene uploads aren't
   titled that way, so it surfaced song clips / battle clips / the Red Wedding instead. The
   tavern scene exists on YouTube mainly *inside* essay/breakdown videos — which were
   title-REJECTED as "annotated".
2. **False anchors by title.** "The Red Wedding — The Rains of Castamere plays" shares the song
   words, so it scored as an anchor and AIRED 35× as the video's dominant source — pure wrong-scene.
3. **Repetition.** beat_windows share alternates across scenes; with a thin/wrong pool one window
   replayed everywhere.

Competitor (XywbvWmExdU) for reference: **~87% of screen time is the exact tavern scene**, ~3.5s
avg shot, captions + B&W accents.

## Fixes
- **Multi-variant anchor search** (`discover.anchor_queries`): character-pair, scene-name, episode
  code (`S02E09`), and **verbatim quote** variants — the way clip uploaders actually title.
- **Episode-code split** (`analyze`): "Show (Season 2, Episode 9)" → clean search title + `S02E09`
  hint (the parenthetical was poisoning every query).
- **Content-level dialogue verification** (`discover.verify_anchor_candidates`): downloads each
  likely candidate's English captions (via the HD/PO-token yt-dlp path — plain clients get none)
  and marks it `anchor_verified` if it SPEAKS the scene's lines. A verified essay/breakdown
  **earns its way back in** past the title-reject and resolution gates, and leads the pool.
- **LLM now returns the scene's verbatim DIALOGUE** (`analyze` prompt), explicitly *not* song
  lyrics (lyrics appear in every cover and caused false verifies on run 1).
- **Music-only demotion + verify-pool exclusion** (`_MUSIC_ONLY_RX`): lyric/cover/“extended/best
  version” uploads can't crowd out or false-verify the scene.
- **Content-based anchoring in match.py**: a source is an anchor if it's `anchor_verified` OR its
  own ASR speaks a scene line OR (title fallback) ≥2 scene tokens incl. a character/actor name —
  so the Red Wedding song clip is no longer a false anchor.
- **Repetition cap** (`build`): the same exact window airs at most twice across the whole video;
  beyond that, stride a different part of the source instead of a third replay.

## Result — same script, fresh render (`output/clipstudio_test_bronn`, 100s, 25/25 QA pass)
- Discovery dialogue-verified exactly **1** anchor (the real scene-carrier), **0 false positives**
  (run 1 had 2 music-video false verifies — fixed).
- `match: 1 anchor source · dark_scene=True` — the candlelit tavern is correctly treated as a
  night/interior scene, so bright daytime battle shots are penalized.
- **Every beat's primary footage is the verified exact-scene source** (was: Red Wedding clip 35×).
  Red Wedding dropped 35× → 6×.
- Frame-by-frame: tavern standoff, gold cloaks, Bronn + Hound, the side-by-side comparison on the
  "he's just like the Hound" beat, candlelit two-shot on the climax beat. dark-scene-correct.
- −16.0 LUFS, −1.9 dBTP, 0 black frames, beat_windows/alternates contracts intact.

## Honest caveat
The strongest available source for this scene was the competitor **essay itself** (dialogue-
verified), so some frames inherit ITS burned-in graphics (a "SOMEONE" title card, a Bronn/Hound
split graphic). Our own narration captions render correctly above the letterbox; the essay's
graphics are a side effect of the raw full-length scene not existing standalone on YouTube (only a
37s singing excerpt does). For scenes whose raw footage IS uploaded standalone, the pipeline
prefers that. This is the DESIGN's ~85%-automation reality; the relevance + repetition complaints
are resolved.

Suite: `tools/test_clipstudio_fixes.py` 95/95. Engine guards 106/0.

---

## Round 2 — Tywin "Small Council chair test" (2026-06-11, portal job 2fe62e3cef)

User tested a 2nd topic (script taken from competitor HGRZTx_cLVY) and got irrelevant footage
again — "the narration is about Peter/Littlefinger but Littlefinger never appears."

**Diagnosis (sec-by-sec vs the competitor):**
- Competitor uses ~90% of ONE scene: GoT **S3E3 "Walk of Punishment" — the Small Council meeting**
  where Littlefinger drags a chair to sit nearest Tywin.
- Our render: the LLM's anchor query was a vague THEME ("Tywin Lannister chairs Littlefinger
  reveal") — no canonical scene name, no episode. So the raw S3E3 council scene was NEVER found;
  discovery downloaded 6 Tywin COMPILATIONS ("All Scenes", "Best scenes") that scatter footage
  across the whole show. Nothing dialogue-verified (LLM gave no dialogue), so no decisive anchor.
- "Peter Beish" in the script = ASR mis-transcription of "Petyr Baelish"; the LLM analysis
  correctly resolved it to Littlefinger, but with no Littlefinger-scene clip downloaded he couldn't
  appear.
- Empirically: the raw scene IS on YouTube ("S03E03 Chair scraping scene", "Tyrion Tywin council
  meeting") — but only surfaces when you search the canonical name ("Small Council" / "Walk of
  Punishment" / "S03E03"), which the anchor query lacked.

**Fixes (this round):**
- analyze prompt now demands the CANONICAL scene name + an `episode` field (SxxExx) + verbatim
  dialogue — not a thematic description. (Fresh run now yields "Small Council Meeting / S03E03 Walk
  of Punishment" and query "Small Council scene S03E03 Tywin Littlefinger chairs".)
- `episode_hint` also parsed from the anchor `episode` field.
- `_dominant_scene_phrases`: data-driven canonical-scene mining from the beats' own scene_queries
  ("small council", "council table") — finds the scene even when the LLM anchor is weak.
- Single-scene COMPILATION demotion (`all scenes/best scenes/best of/moments/supercut/...`, −0.30)
  so the raw single-scene upload outranks character montages (mirrors the existing trailer demotion).
- Episode-code relevance boost (+0.25) and episode-code anchoring in match.py (a "Chair scraping
  scene S03E03" upload IS the scene even without a character name in its title).

**Dry-run result:** the exact **"S03E03 Chair scraping scene"** (the Littlefinger chair-drag, 127s)
now ranks rel=1.00 at the top of the pool and downloads — previously never found.
Suite: 100/100.

**Render result (`output/clipstudio_test_tywin`, 106s, 25/25 QA):**
- `match: 4 anchor source(s)` (portal run had 0) — the S3E3 council clips anchored.
- Usage now dominated by the EXACT scene: Small Council S03E03 (20×), chair-scraping S03E03
  (19+17×), Tyrion/Tywin council meeting (19×), Ned's council (14×) — vs the portal run's 6
  scattered Tywin compilations.
- **Littlefinger (the user's "Peter") now appears at his beats.** Real beat-9 clip = Littlefinger
  + Varys + Pycelle at the council; beat-11 = the full S3E3 chair-test chamber (Tywin seated,
  Littlefinger at a chair, the lined-up chairs) — the exact shots the competitor uses.
- (Note: the even-division QA thumbnail for scene_09 sampled an adjacent clip and looked off; the
  REAL beat clips — extracted at beat boundaries — show the council scene with Littlefinger.)

---

## Round 3 — Pacing (cinematic hold + Ken Burns) + quality (2026-06-11)

User: "video too fast — hold the main moment a few seconds with a slow zoom, like the competitor"
and "quality is low again."

**Pacing / Ken Burns hold:**
- Main-moment scenes (dramatic peaks + the top-third highest-confidence exact-scene beats, capped
  at ~1/3 of scenes) now HOLD — their energy is forced to 1 so plan_beats gives one shot for the
  whole beat — and that shot gets a slow Ken Burns push-in (zoom 1.0→1.10 via zoompan, pre-upscaled
  with lanczos so the zoom never reveals soft pixels). Result: 6 hold+zoom scenes, 46 beat-clips
  (was 64) — slower, more cinematic dwell. Verified: a held beat's first frame is a wide shot, its
  last frame is visibly zoomed in.

**Quality:**
- Root cause: the dominant exact-scene source was a **480p** upload (used 20×); assemble bilinear-
  upscaled the 640×480 clip → soft. Now SD sources (<1280w) are **lanczos + unsharp** upscaled to
  1080 at CUT time (proven sharper in a beat-9 before/after), and clips encode at **CRF 18** (was
  20) since they're re-encoded twice downstream.
- HD selection preference (+0.06·height/1080) nudges the matcher toward 1080p clips of the same
  scene; the 1080p S03E03 council clips now carry most beats, the 480p one is sharpened.
- Honest limit: the exact-scene upload is genuinely 480p, so it can't be native-1080 sharp — the
  lanczos+unsharp pre-upscale is the mitigation; the 1080p council clips cover the rest.

Suite: 107/107. Engine guards 106/0.

---

## Round 4 — Measured repetition + quality, calibrated against the competitor (2026-06-11)

User: "still no improvement — quality low, repetition back." Stopped trusting spot-checks; built a
beat-level forensic audit (mid-frame ahash near-dup pairs split into HOLD-dups [consecutive — a
held shot, a feature] vs DISTANT-dups [true repetition] + Laplacian sharpness), and CALIBRATED it
on the competitor's own first 130s.

**Calibration findings (changed the whole diagnosis):**
- The competitor's footage sharpness is LOW too (median lapvar 5 — same soft YouTube sources);
  our v2 was already sharper (8). Sharpness was never the real gap.
- The competitor's own edit shows 14 ahash dup-pairs in 130s — but almost all CONSECUTIVE (their
  holds). Only ~2 DISTANT re-airs (spaced ~50s). So: holds are fine, near-term re-airs are the sin.
- The "1080p" sources are fake-HD (lapvar 2-18); the 480p upload had the most real detail.

**What shipped (after two failed intermediate designs — raw playhead walk and a CLIP-embed global
guard BOTH increased visual dups, because CLIP judges semantics, not pixels):**
- PIXEL-hash spaced repeat guard: a window airs only if its mid-clip 8x8 ahash differs (hamming>8)
  from the last 18 aired beats; same guard on the shot-aware fill walk (next DETECTED shot whose
  look isn't a near-term repeat). Distant tasteful re-airs stay allowed (competitor-style).
- Detail-enhance chain on every cut: hqdn3d light denoise → unsharp 5:5:0.9 → cas 0.5
  (empirically 3.3× lapvar on the soft S03E03 source, halo-checked).

**Final numbers (same script/sources):**
| | beats | hold-dups | DISTANT-dups | sharp-med |
|---|---|---|---|---|
| v2 (user watched) | 46 | 1 | **14** | 8 |
| v6 (final) | 46 | 7 | **5** | **16** |
| competitor | 56 | 12 | 2 | 5 |

Distant repetition down 65%; remaining 5 reflect the pool's limited distinct angles of one static
council scene. Footage is now 3× sharper than the competitor's own.

---

## Round 5 — Competitor-level editing intelligence (2026-06-11, v8)

User compared screenshots: competitor frames clean/neutral + B&W freeze "screenshot" moments on
analytical punchlines; ours green-murky with no such treatment. Studied the competitor 0-5min
WITH audio (Gemini): 12 B&W freezes in 5min, each with a camera-shutter click, slow push-ins on
nearly every shot, music ducked under narration, base footage NEUTRAL (no tint).

**Shipped (clipstudio-side only):**
1. **Neutral footage grade** — engine themes tint footage (history: warm/green colorbalance +
   desat + paper grain = the murk in the user's screenshots). build now overrides the theme dict:
   `grade = eq=contrast=1.05:saturation=1.04`, `overlay_effects=[]`
   (`VIDLORE_CLIPSTUDIO_NEUTRAL_GRADE=0` reverts). Theme captions/cards/music untouched.
2. **Analytical B&W FREEZE** (`_freeze_punchline`): held main-moment beats play live to ~40%,
   then HOLD the exact frame — B&W + contrast lift + luma grain — under the key line.
   Gotchas found: the bundled ffmpeg's `tpad stop_mode=clone` silently no-ops (use `loop`),
   an input label can't feed two filter chains (explicit `split`), and `noise=alls` adds CHROMA
   noise that re-colorizes B&W (use `noise=c0s` luma-only).
3. **Shutter click SFX** (`_click_wav` + `_mix_clicks`): synthesized tick mixed into
   narration.audio at each freeze moment (assemble muxes narration.audio — no engine change).
4. **Push-in on every beat** (1.055 gentle; 1.10 held) — competitor applies one to nearly every
   static shot.

**Verified in the final render (tywin8):** 6 freezes + clicks; t=2.2s frame is a grain B&W
frozen council wide with caption — visually equivalent to the competitor's signature move; the
color footage is clean neutral (no green wash). 25/25 QA · suite 108 · engine guards 106.

### Round 5b — hold length + click loudness (user feedback on v8)
- **Holds extended to the FULL scene** (`_freeze_continuation`): a held scene's later beats now
  continue the same frozen frame (living grain), so freezes run 4.6–7.7s like the competitor's
  4–8s (v8's froze only the first beat ≈1.7s).
- **Shutter click rebuilt + measure-calibrated**: pink-tick + delayed brown-thunk ("ka-chak"),
  gain tuned by volumedetect to **−22 dB peak** (v8's white-noise click peaked −1.9 dB — far too
  loud; the first rework was −70 dB — inaudible. Gains are measured, never guessed).
- Verification gotchas recorded: cv2 VideoCapture time-seeks snap to keyframes (scans missed
  existing B&W frames — decode sequentially or extract via ffmpeg accurate seek), and HSV
  saturation is unreliable on dark pixels (a warm-graded near-black frame measures S≈40 while
  looking gray — use ffmpeg-extracted frames + eyes, or chroma-difference metrics).
Final reference render: `output/clipstudio_test_tywin10/output/final.mp4` (25/25 QA, suite 108).

---

## Round 6 — REAL-AUDIO BREAKOUTS (2026-06-11, v12)

User request: when the narration discusses a specific moment, PAUSE the narration and play the
actual scene WITH ITS REAL VOICE for 4-5s, then resume — like the competitor's cold-open hook +
"evidence" moments (studied jsbXnZgYf78: 6 breakouts + a 12s cold open in 4 min; music stays,
ducked; letterboxed; placed right after the claim they validate). 1-3 per video, NATURAL only.

**Implementation (clipstudio-only):**
- `_select_breakouts` — three-stage intelligence: (1) locate the beat's QUOTE in every source's
  own ASR (per-shot transcripts; LLM quotes are often paraphrases, so) (2) also try the analysis'
  verbatim anchor-dialogue lines mapped to their best-overlap beat, and (3) fallback: MINE the
  exact-episode sources' dialogue-rich shots and attach each to the beat its words overlap —
  tiered (episode-coded/verified sources need no overlap: the whole scene IS the topic;
  scene-titled-only sources need ≥2 content-word overlap so another episode's council can't leak).
  Cap 1-3 by video length, ≥4-scene spacing — naturally skips when nothing真 relates (tywin11
  fired 0 with the strict matcher; the mined fallback found the REAL line).
- `_apply_breakouts` — inserts a pseudo-scene before the beat: reindexes segments/scenes/
  narration, shifts later WordTimings (+dur), splices the breakout audio INTO narration.audio
  (`_splice_audio` atrim+concat) — assemble muxes that file, so narration literally pauses and
  the scene speaks. Breakout video = enhanced 1080p30 cut (`_extract_breakout`, loudnorm I=-17 +
  fades on audio); no captions over it (empty words); excluded from hold/freeze; split
  sequentially if the plan wants multiple beats.

**End-to-end verification (tywin12, 25/25 QA):** 1 breakout fired — Tywin's real line
"Conveniently close to your own quarters, I like it" (the actual S3E3 chair moment) inserted
before the "that arrangement is no accident" beat; final duration 106.3→109.9s; whisper on the
FINAL audio confirms narration → real scene voice (28.1–31.8s) → narration resume; breakout frame
is in-scene (Tyrion at council) with no caption overlay. Suite: 116/116.
