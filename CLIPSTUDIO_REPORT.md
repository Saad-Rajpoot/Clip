# ClipStudio — Complete Project Report

> A reference document describing what ClipStudio is, how it works end-to-end, every major
> feature, the tech stack, the project's history, and where it's going. Written to be
> self-contained — a new engineer (or an AI assistant like ChatGPT) should be able to understand
> the entire project from this document alone.

---

## 1. Executive summary

**ClipStudio is an automated "faceless" video editor.** You give it a **topic + a narration
script** (and optionally your own recorded voiceover), and it produces a **finished, captioned
YouTube-ready video** in which **real movie/TV footage is matched scene-by-scene to what the
narration is saying** — then graded, cut to a human-feeling rhythm, captioned word-by-word, and
rendered.

The flagship use case is **Game-of-Thrones–style video essays** (e.g. *"How Daenerys Became the
Villain Nobody Saw Coming"*, *"This One Scene Explains the Entire Tyrion & Tywin Relationship"*):
a narrator explains a character/scene while the **exact** footage being described plays on screen.

**One-sentence pitch:** *Script in → the precise on-screen footage discovered, matched, and edited
for you → finished essay video out.*

### Vision (in the owner's words)
ClipStudio is fundamentally a **video compilation tool** — like a human editor joining clips into a
finished video, it **downloads clips from multiple YouTube videos and matches them to the script**.
Where the voiceover describes something, the **matching scene plays on screen**; **images and effects**
fill in between; and the editing is done so the result has **no copyright problems**. Today it starts
in the **movie niche**, but the goal is to make it smart enough to produce **expert-level compilation
videos for *every* niche** — i.e. an **intelligent auto-editor** that edits like a professional human
compilation editor, on any topic.

---

## 2. The problem it solves (and how it's different)

Generic AI-video tools generate or paste **loosely-relevant stock/AI footage** under a narration.
ClipStudio's entire reason to exist is the opposite, non-negotiable principle:

> **Exact for specific, filler for generic** — the relevance bar is *contextual*, not blanket:
> - When the narration makes a **specific scene claim** ("Tyrion shoots Tywin with a crossbow"), the
>   video must show **that exact scene** — the exact clip, screenshot, or footage of that moment.
> - When the narration is **generic** ("and everything was about to change", "sit with that"), there
>   is no exact scene to show, so a **thematically-relevant filler clip is fine** — it just has to fit
>   the mood/topic, not be a precise match.
>
> *"jin ke baare mein baat ho rahi hai (specific), screen par wahi chale; generic baat par milta-julta
> filler chal sakta hai."* Engagement is further boosted with **breakout clips** (real-audio movie
> moments) sprinkled through the video.

This distinction is **built into the data model** (`is_specific_claim` per beat) and now drives the
**AI verifier**: it is **strict** on specific beats (demands the exact scene) and **lenient** on generic
beats (accepts an on-topic, right-subject filler — only replaces if the footage is off-topic, jarring,
or shows the wrong character/era). This stops the verifier from falsely flagging generic narration as
"wrong" and matches how a human editor actually works.

To honor that, ClipStudio doesn't *generate* footage — it **discovers and downloads the real
clips from YouTube**, **indexes** them (transcribes the dialogue, detects shots, embeds keyframes,
recognizes faces), and **matches** each narration beat to the single best real shot. It then layers
the narration, captions, music, and editorial polish on top.

Key differentiators:
- **Footage is real and specific**, located by visual + dialogue + face evidence — not generated.
- **An AI vision verifier** double-checks every chosen shot ("does this frame actually match what
  the narration is saying?") and swaps bad picks.
- **Competitor-grade editing**: word-by-word kinetic captions, real-audio "breakout" moments,
  freeze/zoom/B&W treatments, motivated transitions, dynamic music.

---

## 3. Relationship to the parent "vidlore" engine

ClipStudio is a **self-contained product that lives inside a larger repo** at
`~/Desktop/vidlore-clipstudio`:

- **`vidlore/` (the parent engine)** — an older "AI documentary render" engine (its own web portal
  on **port 5000**) that generates AI images and assembles documentary videos. It owns the heavy
  **renderer**: `assemble()` (grading, captions, music, transitions, encode), the **TTS** stack
  (`narrate`, `narrate_premium`, `narrate_from_file`), **forced alignment** (`align.py`), themes,
  and ffmpeg tooling.
- **`vidlore/clipstudio/` (this product)** — the movie-clip editor. **Web portal on port 5151.**
  It implements its own discovery/download/index/match/cut/verify logic and **reuses the engine's
  renderer** for the final stitch.

**Hard rule:** the parent engine's files (`vidlore/*.py` outside `clipstudio/`) **must not be
modified.** A guard suite (`tools/test_engine_guards.py`, **106 checks**) protects them. Every
ClipStudio improvement is done on the ClipStudio side, even when it works *around* engine behavior
(e.g. the caption word-sync fix re-implements alignment in `build.py` rather than touching the
engine's aligner).

---

## 4. End-to-end architecture — the 9-stage pipeline

The whole run is one call: `produce_auto(...)` (auto-discovery mode) or `produce(...)` (manual
sources), in `vidlore/clipstudio/orchestrate.py`. Every stage saves `project.json` so a run is
**resumable** and every decision is inspectable.

| # | Stage | What happens | Key tech |
|---|-------|--------------|----------|
| 1 | **Analyze** | LLM reads the topic+script → `ScriptAnalysis` (movie, year, actors, characters→actor map, **anchor scenes**, key scenes, `video_type` = single_scene/multi_scene). Then per-beat enrichment (scene query, required entity, quote, est. duration, emotion, shot intent). | DeepSeek/Gemini/Claude, robust JSON parsing + flash fallback |
| 2 | **Discover** | Build targeted YouTube search queries (anchor-scene variants + **per-beat scene queries** + dominant scene phrases, era-anchored) and rank candidate sources. Junk titles are gated out here. | yt-dlp `ytsearch`, title gates |
| 3 | **Download** | Download the chosen sources in **HD** under a permission policy. | yt-dlp (`.hdvenv`, bgutil **PO-token**, deno, `cookies-from-browser`), concurrency |
| 4 | **Face-ID refs** | Build face-recognition references for each named character (mapped to its actor). | insightface |
| 5 | **Index** | For every source: transcribe dialogue (word-level), detect shots, extract a keyframe per shot, embed it (CLIP), measure face fraction, read on-screen text (OCR), perceptual-hash, score quality. Cached/resumable. | faster-whisper, PySceneDetect, CLIP ONNX, OCR |
| 6 | **Match** | For each narration beat, score every candidate shot and pick the best (with anti-repetition + wrong-character/era penalties). Junk sources are dropped again here. | numpy scoring |
| 7 | **Cut** | Trim each selected shot to its planned in/out as a standalone clip. | ffmpeg `libx264` (parallel) |
| 8 | **Verify** | An AI **vision** model looks at each chosen frame and judges keep/replace vs the narration; bad picks are swapped for the best passing alternate. | Gemini/Claude vision |
| 9 | **Build** | Narration (your voiceover or TTS) + word-by-word captions + breakouts + editorial polish + music → final MP4 via the engine's `assemble()`. | engine `assemble()`, VideoToolbox encode |

Outputs land under the project dir: `output/final.mp4`, `output/final.srt`, render metadata,
a QC `review.html` + `review_queue.json`, and a `ledger.jsonl` (provenance of every decision).

---

## 5. The AI "brain" (LLM provider abstraction)

All reasoning (script analysis, per-beat enrichment, the vision verifier) runs through
`vidlore/clipstudio/llm.py`, a **pluggable provider with automatic fallback**:

**Default chain:** `deepseek-v4-pro → deepseek-v4-flash → Gemini → Claude (last resort)`

- **DeepSeek V4 Pro** is the default brain (reasoning model, best quality). If it's empty/unavailable,
  **V4 Flash** (fast) serves; only if all DeepSeek fails does it fall to **Gemini**, then **Claude**.
- DeepSeek is **text-only** — for **vision** calls (the verifier looks at image frames) it
  automatically skips DeepSeek and uses **Gemini** (Google Vertex creds present) → **Claude**.
- This key's endpoint serves **only** `deepseek-v4-pro` and `deepseek-v4-flash` (there is no
  `deepseek-chat`). DeepSeek is called over stdlib `urllib` (OpenAI-compatible API), no SDK dep.
- Configured in `.env`: `VIDLORE_CLIPSTUDIO_LLM_PROVIDER=deepseek`, `..._DEEPSEEK_MODEL=deepseek-v4-pro`,
  `..._DEEPSEEK_FAST_MODEL=deepseek-v4-flash`, plus `DEEPSEEK_API_KEY`.
- The **web portal exposes a brain selector** (DeepSeek V4 Pro = default, V4 Flash, Claude, Gemini),
  validated against an allow-list and applied per render job.

---

## 6. Feature catalogue

### 6.0 Unified beat visual-policy (the "brain" that drives every stage)
Every script beat is classified once (in the Analyze stage, by the LLM with a deterministic heuristic
fallback — `policy.py`) into ONE treatment that **all** downstream stages obey:
- **exact_scene** — a precise scene/quote/action/event → STRICT matching; ONLY real footage or an
  exact source-frame may air (never a web/AI image or loose filler); if none is found the beat is
  flagged **`exact_scene_missing`** → manual review (never silently filled); aggressive discovery.
- **character_specific** — a named person/thing in general → the right subject (Face-ID), any clean
  shot; medium discovery.
- **generic_filler** — generic narration → any relevant clip; source **variety actively maximised**;
  low discovery / reuse the pool.
- **abstract_effect** — abstract/emotional, no literal visual → prefer image/effect/freeze/B-roll.

Plus an orthogonal **breakout_candidate** flag (a quoted iconic moment). This single policy is the
source of truth for discovery budget, match strictness/variety, the verifier, image/effect fallback,
and the QC ledger — so the whole pipeline decides like one editor: exact where it must be, filler
where it can be, image/effect where there's nothing literal, and *honest* (flagged) where the exact
scene simply doesn't exist.

### 6.1 Relevance / exact-scene matching
- **Per-beat discovery queries** — every narration line's `scene_query` is searched, so the *specific*
  scenes (not just generic character clips) get downloaded.
- **Footage-budget scaling** (`_scaled_source_budget`) — a long multi-scene essay automatically pulls
  far more sources (~1 per 5 beats, capped) so the specific scenes exist in the pool; single-scene
  deep-dives are *not* scaled (they want one scene's raw footage).
- **Match scoring** = CLIP visual similarity (base) + transcript overlap + **dialogue-lock** (when a
  beat's iconic quote is literally spoken in the clip's ASR) + **Face-ID** bonus + object signal,
  minus penalties (reuse, burned-subtitle, **wrong-face**, period/era), plus an **anchor bonus**.
- **Face-ID with character→actor mapping** — beats name characters ("Daenerys"); the index recognizes
  actors ("Emilia Clarke"); the analysis's `characters` map bridges them so identity actually scores.
- **AI vision verifier** — per-beat keep/replace judgment on the actual frame; replaces failures.

### 6.2 Guards against irrelevant / junk footage
Title-based **source-quality gates** (applied at discovery *and* re-applied at match-time, so junk
already on disk is still excluded — no re-download needed):
- `_REJECT_TITLE` — reactions, reviews, commentary, interviews, **video essays** ("the real reason",
  "why X ruins … character arc", "explained", "breakdown", "deep dive", "analysis"), **BTS / making-of /
  "anatomy of a scene"** featurettes (which show film crew + equipment), parody/joke re-edits
  ("fart edition", crack/YTP), fake future seasons, press junkets.
- `_REACTION_TITLE` — facecam/reaction videos (a person on a couch over a tiny show inset).
- `_NONSHOW_TITLE` — strategy-game battles, AMVs, animated/lego/claymation, AI-generated.
- `_wrong_installment` / `_WRONGSHOW_SIBLINGS` — wrong show/prequel (e.g. House of the Dragon on a GoT topic).

### 6.3 Anti-repetition
A shot **and its visual near-duplicates** are penalized if reused within a recency window (decaying
penalty), two shots from the same source within `scene_gap_sec` count as the same scene, source-recency
discourages clustering, and there are hard `max_reuse_per_source` / `max_reuse_per_shot` caps.

### 6.4 Captions (word-by-word, synced)
- **Kinetic word-by-word captions** burned by the engine, with **caption-dodge** (captions are
  suppressed over windows where the footage already carries burned-in subtitles/logos).
- **Caption word-sync for uploaded voiceovers** — the user's own narration is **force-aligned** to the
  script. A single whisper pass over a long (16-min+) file *drifts*, so ClipStudio uses **chunked
  alignment** (`_synced_narration_from_file` + `_chunked_whisper_words` + `_align_words_to_hyp`):
  transcribe in ~90-second overlapping windows (locally accurate), then sequence-align the script to
  that dense stream. Per-scene tolerant — one odd scene is clamped, not a reason to discard all.

### 6.5 Real-audio breakouts
The narration **pauses** and a movie clip plays with **its own dialogue** (cold-open / "the scene
proves the point in its own voice") — a competitor-grade flourish.
- Selection (`_select_breakouts`): locate a beat's verbatim quote in a source's ASR, or **mine
  evidence** (a dialogue-rich shot whose words overlap the beat's narration by **≥2 content words**).
- **Luma gate** — too-dark/illegible clips are skipped.
- **Era-coherence gate** — sources whose title declares *only* later seasons than the core scene are
  barred (no bearded S7 Tyrion over an S4 scene); earlier seasons stay (the script narrates backstory).
- **Breakout captions** — the spoken dialogue is captioned word-by-word during the breakout.
- **Default OFF for uploaded voiceovers** — because a breakout grows the timeline past the audio and can
  desync the main caption track; caption sync to the user's own voice wins (env can force them on).

### 6.6 Editorial polish
Beat planning into human-uneven shot lengths (energies/roles), freeze frames, black-&-white treatment
on key moments, hold + punch-in zoom on the "money" shot, motivated transitions (dissolve/whip), SFX
cues, dynamic music arc, watermark-crop (punch-in to hide a rival channel's logo), branding/CTA-card
freeze-replace, and black-frame repair.

### 6.7 Image fallbacks (when no clip fits) — strict "real images only" policy
When footage is weak/missing/repeated, beats are covered by **real downloaded source-video frames**
(Ken-Burns stills), never AI:
- **Pass 1 — source-frame stills (preferred):** a real keyframe from the downloaded/indexed pool, for
  abstract beats (prefer a still/effect), filler repeats (variety), weak/unconfirmed picks, exact
  beats whose footage failed (recovery), and no-clip beats (`pick_pool_still` CLIP-ranks the pool).
- **Pass 2 — web-exact-scene (last resort, gated):** a real **live-action** web still, ONLY for
  exact_scene / character_specific beats, and ONLY if it passes strict validation — CLIP relevance +
  Face-ID + rejects watermark/burned-text/collage/poster/seam + **AI-source/AI-art blacklist**
  (`is_ai_generated_source`, `_photographic_ok`). Never used as generic-filler decoration.
- **Pass 3 — exact_scene_missing:** an exact beat with no confirmed footage and no validated still is
  **flagged for manual review**, never silently covered by weak filler.
- **AI images are globally banned** (`allows_ai_image()` is always false); **NOW-photos / recent-actor
  photos are disabled.** `image_meta.source` is only `source-frame`, `source-frame-recovery`, or
  `web-exact-scene` (recorded in the ledger) — never `ai`/`now-photo`/`web`.

### 6.8 Performance (CPU / turbo)
Worker counts **auto-scale to the machine's cores** (`_workers`/`_NCPU`): cut workers, whisper ASR
`cpu_threads`, download concurrency. A master switch **`VIDLORE_CLIPSTUDIO_MAX_CPU=1` (Turbo)** saturates
every core. (The final encode runs on Apple **VideoToolbox** — hardware-accelerated — so the CPU levers
speed up cut/index/download, not the encode.) The deep-index stage is deliberately **serial** (the
whisper/Face-ID/CLIP/OCR models aren't thread-safe to share).

### 6.9 Web portal UX (port 5151)
Paste a **topic + script**, optionally **upload a voiceover MP3**, pick the **AI brain** (DeepSeek V4
Pro default), toggle **Turbo (all cores)**, choose **AI voice** + character + theme + AI-verify on/off,
click **Create**. The job runs in a **background thread** with **live progress** and a **download**
link. Renders are **serialized** through a lock (so one job's brain/turbo env can't clobble another's).
**No authentication by design** (local single-user tool).

---

## 7. Tech stack

- **Language:** Python 3.
- **LLM:** DeepSeek (urllib, OpenAI-compatible) primary; Google Gemini (Vertex, `google-genai`) and
  Anthropic Claude as fallbacks / vision.
- **ASR / alignment:** faster-whisper (`base`, int8) for indexing + voiceover word-alignment.
- **Vision:** CLIP (ONNX runtime) keyframe embeddings; insightface face recognition; OCR for on-screen text.
- **Shot detection:** PySceneDetect.
- **Download:** yt-dlp in an isolated `.hdvenv`, with bgutil PO-token provider, deno, browser cookies.
- **Render/encode:** the parent engine's `assemble()` + ffmpeg (imageio-ffmpeg binary) with **VideoToolbox** (Apple Silicon HW encode).
- **Web:** Flask (portal, port 5151).
- **Isolated deps** live in `.clipstudio_libs` to keep the engine venv pristine.

---

## 8. Hard constraints & design principles

1. **Parent engine files are never modified** (106 guard checks enforce this).
2. **Exact for specific, relevant filler for generic** — a *specific* scene claim must show the exact
   scene (the right scene always beats a prettier wrong one); *generic* narration may use a
   thematically-relevant filler. Engagement is lifted with periodic real-audio breakouts.
3. **Portal stays unauthenticated** (intentional, local use).
4. **Repo is NOT a git repository** — changes are made directly; tests are the safety net.
5. Renders use the standing **`approved_testing`** permission policy (user's own rights assertion;
   ClipStudio records provenance but does not certify copyright).
6. **Captions sometimes "don't touch"** — at times the user explicitly froze caption behavior.
7. **Resilience:** the pipeline never hard-fails on the LLM (provider fallback), alignment (proportional
   fallback), or footage (image fallback) — it degrades gracefully.

---

## 9. Project history / evolution

- **Deep review fix-pass (June 2026):** ~75 correctness/quality fixes across the pipeline; a regression
  suite grown from **82 → 302 tests** (`tools/test_clipstudio_fixes.py`), plus the 106-check engine guard.
- **Competitor-level editing:** iterated a portal video to ~98% relevance; added freeze/B&W/SFX/clean-look
  editorial, word-by-word captions, and real-audio breakouts.
- **Face-ID for all named characters**, image-search ported from a sibling project, image-relevance
  validation + AI-image leak fixes, source-frame stills + NOW-photos for then-and-now / compilation content.
- **DeepSeek migration:** made **DeepSeek V4 Pro** the default brain (was Claude); Claude moved to
  last-resort; portal brain selector added.
- **Performance:** CPU auto-scaling + `MAX_CPU` Turbo (and a portal Turbo toggle).
- **Long-form fixes (this iteration):**
  - **Caption word-sync** via chunked alignment (fixed drift on a 16-min uploaded voiceover).
  - **Footage-budget scaling** for long essays (a 181-beat video went from 14 → ~45–50 sources).
  - **Source-quality gating** of video-essays/BTS/parody (an essay source had aired 37×; "Anatomy of a
    Scene" BTS had leaked film-equipment shots).
  - **Breakout era-gate** (no wrong-season breakouts) + **≥2-word overlap** (no tangential breakout lines).
  - **Specific-vs-generic verifier** — the AI verifier is now strict (exact scene) on specific-claim
    beats and lenient (relevant filler OK) on generic narration, so generic beats are no longer
    false-flagged as "wrong".
- **Videos produced:** Robert/Cersei (S1E5), Tywin "stag"/Arya, Tyrion trial monologue, Peaky Blinders
  (Tommy/Alfie), a 16-min **Daenerys** essay, and the current 24-min **Tyrion & Tywin** essay (iterated v1 → v2 → v3).

---

## 10. Current state & known limitations

**Current state:** Tests **305/305 pass**, engine guards **106/106**. A **v3** render of the Tyrion &
Tywin essay is in progress (essay-source repetition + breakout polish applied; the specific-vs-generic
verifier leniency applies to the next render).

**Known limitations:**
- **Long-form relevance isn't 100%** — for a 20-min essay referencing dozens of specific scenes, some
  beats have no exact clip on YouTube; the matcher then uses the right character but not the exact
  moment. The AI verifier flags these (~30–40% on the hardest essays), but it can only swap for what's
  in the pool. Abstract narration ("sit with that") has no literal visual by nature.
- **Repetition** can concentrate on a dominant source after junk is removed (mitigated by gating +
  reuse caps; a stronger cap is on the roadmap).
- **A few breakout lines** can still be thematically loose (tightened, not perfected).

---

## 11. Roadmap / future goals

- **✅ DONE — unified beat visual-policy** (see §6.0): every beat classified into exact_scene /
  character_specific / generic_filler / abstract_effect, obeyed by all stages. This was the
  immediate next step and is now implemented + tested.
- **The north star — an intelligent auto-editor for EVERY niche.** Today ClipStudio is tuned for the
  movie/TV niche; the goal is to generalize the same expert-level compilation editing to *any* niche
  (sports, history, tech, true-crime, etc.) via a **niche-adapter interface** (the current movie logic
  — Face-ID actor refs, season/era gates, show-sibling guards, dialogue-lock — becomes the "movie
  adapter"). This is the next major roadmap item.
- **Every content format** — robustly handle **then-and-now**, **compilation**, and multi-scene essays
  (in progress; image fallbacks + NOW-photos already support then-and-now).
- **Near-perfect long-form relevance** — smarter per-beat specific-scene discovery + matching so 20-min
  essays approach scene-exact on every *specific* beat (generic beats intentionally use relevant filler).
- **Breakout line-relevance** — tighter dialogue-to-narration matching.
- **Repetition caps** scaled to video length / scene count.
- **Copyright-safety** — keep refining transformation (cuts, captions, effects, grading, breakouts) so
  finished videos avoid copyright strikes.
- Continue iterating each delivered video to flawless before sign-off.

---

## 12. Glossary

- **Beat / segment** — one narration clause/sentence; the unit that gets its own footage.
- **Anchor scene** — the specific iconic scene(s) the essay is centered on; discovery + matching prioritize it.
- **Exact-scene priority** — the rule that the footage must show the precise moment the narration describes.
- **single_scene vs multi_scene** — a deep-dive on one scene vs an essay spanning many; controls source-budget scaling, era/purity gates.
- **Breakout** — a moment where narration pauses and a movie clip plays with its own real audio.
- **Era-gate** — bars footage/breakouts from seasons later than the core scene (continuity coherence).
- **Dialogue-lock** — strong match signal when a beat's quote is literally spoken in a clip's transcript.
- **Face-ID** — actor face recognition used to confirm the right character is on screen.
- **Caption-dodge** — suppressing burned captions over footage that already has on-screen text.
- **NOW-photo** — a recent real photo of an actor, used for "where are they now" beats.
- **Turbo (MAX_CPU)** — switch that uses every CPU core for cut/index/download.

---

*Stack: Python · DeepSeek/Gemini/Claude · faster-whisper · CLIP(ONNX) · insightface · PySceneDetect ·
yt-dlp · ffmpeg/VideoToolbox · Flask. Portal: http://127.0.0.1:5151. Engine portal: :5000 (separate).*
