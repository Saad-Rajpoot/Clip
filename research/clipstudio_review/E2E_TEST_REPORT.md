# ClipStudio End-to-End Deep Test — "The Interrogation Scene That Broke Batman" (2026-06-10)

Full auto pipeline (topic + 12-line script → finished video), single-scene deep-dive on the
The Dark Knight interrogation scene. Project: `output/clipstudio_test_joker/`
(final video: `output/final.mp4`, 74.8s, 74 MB). Ran THREE times: run 1 exposed a missing
dependency, run 2 exposed a real HD-path bug (both fixed), run 3 = clean HD render.

## Verdict
**PASS — 25/25 automated QA checks, 0 warn, 0 fail.** Visually: a coherent, cinematic,
caption-clean single-scene essay with ~16/19 beats on the exact interrogation scene.
Two genuinely new bugs were found BY the test and fixed (that was the point of testing deeply).

## Bugs the test caught (now fixed + pinned in the suite)
1. **yt-dlp missing from the runtime** — discovery crashed (`ModuleNotFoundError`). Installed
   into `.clipstudio_libs` (isolated); `vidlore/clipstudio/__init__.py` now registers the libs
   dir for EVERY entrypoint (was: only modules that imported llm/ocr). scenedetect same.
2. **HD path silently dead for ALL downloads** (`hd_download.py`): `download_hd` ran yt-dlp with
   `cwd=<stem dir>` + a RELATIVE `-o` template → output nested into
   `sources/output/.../sources/<sid>.mp4`, produced-lookup found nothing, every source fell back
   to 360p legacy with zero log evidence. Fixes: stem resolved absolute; HD fallback now logged +
   recorded in `sv.extra["hd_fallback"]`; `download_hd` probes the produced file's REAL dimensions
   (info.json mislabeled pre-existing files). Suite: 85/85.

## Automated QA (tools/qa_clipstudio_render.py)
1920x1080@30 · 74.8s · audio ok · 0 sustained black frames · **-16.3 LUFS / -1.9 dBTP** ·
13/13 sources ok (9 HD ≥720p; 4 are letterboxed cinemascope ~544-800px tall, full-width HD) ·
0 DASH fragments / partials left · 19/19 selections (conf 0.52–0.92, mean 0.73) ·
beat_windows lead = chosen pick (19/19) · alternates best-first (19/19) · verifier ran on 19,
**8 repaired**, 6 unrepaired → all flagged · ledger + review_queue + review.html written.

## Fix-pass features observed live in the logs
- `match: 12 anchor source(s) · bonus=0.45 · dark_scene=False` — dark-scene guard did NOT
  false-fire on the "The **Dark** Knight" title (the round-2 fix).
- Anchor/coverage force-includes downloaded (13 > max_sources 6).
- `visual budget — lowered scene energies … (no first-clip replay)`; 33 beat-clips, 0 padded.
- `caption-dodge on 8 text-bearing window(s)` — per-beat, real plan_beats lengths.
- Crime theme → dark_investigation music bucket.
- Cinematic letterbox 132px + captions lifted to 158px.

## Visual relevance (frame-by-frame inspection of 19 midpoint frames)
- Quote beats carry the TOP confidences via dialogue-lock: "Nothing to threaten me with" 0.92,
  "you complete me" 0.86, "You have nothing" 0.85 — and their frames are the right moments.
- Anchor continuity holds: the cut stays inside the interrogation scene for ~16/19 beats
  (Joker/Batman/Gordon at the table, the slam, the wall confrontation, the reveal).
- Caption-dodge verified in frames: scenes with the source's own burned text show NO stacked
  caption; clean scenes show the premium caption (white+red keyword) above the letterbox bar.
- Honest misses (the ~15% the DESIGN doc predicts): beat 16 ("improvised the slow clap") shows
  the hospital-nurse Joker (the actual slow-clap clip wasn't in any downloaded source); beat 8's
  source is a fan-edit with a stylized text overlay (passes the keyword junk-gate); beat 0 uses a
  behind-the-scenes shot of the interrogation set (defensible for a "how it was made" essay, and
  beat 15/18 class beats are flagged for human review as designed).

## Operational notes
- Policy: `approved_testing` (the user's standing testing assertion; ledger records provenance).
- Runtime ~25 min end-to-end on this machine (downloads + ASR dominate).
- `VIDLORE_CLIPSTUDIO_DISCOVER_MAX_SEC=600` used to cap source length for the test.
