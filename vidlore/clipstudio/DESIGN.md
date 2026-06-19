# ClipStudio — intelligent movie-clip editing module for Vidlore

> Takes a narration **script** + a list of **legally-permitted source videos** (URLs or local
> files) and assembles a 10–15 min faceless movie-niche video by finding, for each line of
> narration, the source clip that best shows *what the line is actually about* — then renders
> through Vidlore's existing `assemble()` engine.
>
> This module is **purely additive**. It lives in `vidlore/clipstudio/` and only *imports* the
> rest of the engine. It does **not** modify any existing engine file. It was built in an
> isolated copy of Vidlore (`vidlore-clipstudio/`) so the production tool cannot break.

---

## 0. Honest scope (read before trusting any output)

This is an **~85%-automation + fast-human-finish** system, not a "100% exact, guaranteed-safe"
system. Two things the module will **never** claim:

1. **Frame-perfect semantic exactness.** Matching "show exactly what the narration discusses"
   is *probabilistic*. State-of-the-art temporal grounding scores ~20% tIoU on hard benchmarks;
   retrieval models ~0.47–0.75 NDCG@10. We use these to *rank candidates*, not to ship
   unattended. Abstract lines ("his career was never the same") have no literal visual — the
   module picks the best mood/topic match and **flags it for review**.
2. **Copyright / fair-use safety.** The ledger records provenance and flags risk. It does
   **not** certify fair use or copyright safety. Downloading from a platform can violate that
   platform's ToS regardless of any fair-use argument about the final cut. Permission is the
   user's assertion, per source, and is gated (see §5).

Manual-review checkpoints are **structural**, not optional — see §7.

---

## 1. Pipeline

```
script.txt + sources[]            (URLs and/or local files, each with a permission flag)
        │
        ▼
[1 INGEST]    ingest.py    yt-dlp / local copy → project/sources/ + source manifest
        │                   permission gate · concurrency cap · retry/backoff
        ▼
[2 INDEX]     index.py     per source:
        │                    · faster-whisper  → transcript + word timestamps
        │                    · PySceneDetect   → shot list (in/out per shot)
        │                    · keyframe/shot   → CLIP image embed (vidlore.visual_relevance)
        │                    · OpenCV faces / object tags (optional)
        ▼
[3 SEGMENT]   segment.py   narration script → ordered segments (~2–4 s of speech each)
        │                    + expected-visual hint + keywords per segment
        ▼
[4 MATCH]     match.py     per segment → rank candidate shots by weighted score
        │                    (CLIP text↔image · transcript text · face · object)
        │                    + anti-reuse · source-diversity · pacing constraints
        │                  → 1 ClipSelection (+ alternates) + confidence per segment
        ▼
[5 CUT]       cut.py       ffmpeg-trim each chosen [in,out] → its own .mp4 (clips/)
        │                    snapped to shot boundaries. Renderer plays it from frame 0,
        │                    so NO edit to assemble.py is required.
        ▼
[6 LEDGER]    ledger.py    compliance JSONL: source URL · in/out · duration · reuse count
        │                    · confidence · signals fired · flagged?  → review queue
        ▼
[7 BUILD]     build.py     FootageItem[] + beat_clips{idx:[clip]} + Narration(TTS of script)
        │                    → vidlore.assemble.assemble() → final MP4
        ▼
[8 REVIEW]    web (later)  low-confidence picks surfaced for approve/swap (models the
                           existing /e/<slug> editor + _guard_manual_replacement gate)
```

---

## 2. What we reuse from Vidlore (the leverage)

| Need | Reused from engine | Notes |
|---|---|---|
| Semantic match (text↔image) | `vidlore.visual_relevance` (local ONNX CLIP ViT-B/32) | `score_asset()`, `accept()`, `_img_embed()`, `_txt_embed()`. No torch, no API. Model at `~/.cache/vidlore_clip/`. |
| Source transcripts | `faster_whisper` (already a dep; used in `vidlore.align`) | word-level timestamps |
| Scene detection | `scenedetect` (PySceneDetect, already installed) | content-aware shot boundaries |
| ffmpeg path | `vidlore.ffmpeg_tool.ffmpeg_exe()` | resolves imageio-ffmpeg or `VIDLORE_FFMPEG` |
| Render to MP4 | `vidlore.assemble.assemble()` | the render contract — see §3 |
| TTS narration + word timing | `vidlore.tts` | builds the `Narration` the renderer needs |
| Cards / portrait blurred-fill | `assemble()` `graphics` + `graphic_assets` + `archival` | name/title cards at segment boundaries |
| Manual-replacement gate + audit | `web._guard_manual_replacement`, `replacement_audit.jsonl` | model the review surface + ledger on these |
| Config / API keys | `vidlore.config.load_config()` | ANTHROPIC_API_KEY, etc. |

Everything above is **already installed** in the engine venv. Nothing new to pip-install.

---

## 3. Render contract (how a clip reaches the screen)

The renderer is **not** driven by a single manifest dict. `assemble()` takes parallel
per-scene arrays + two dicts keyed by `scene.index`:

```python
vidlore.assemble.assemble(
    footage,        # list[FootageItem(index, path, is_video)]  — one base asset per scene
    narration,      # Narration(scenes=[NarratedScene(index, audio, duration, words)], audio=<full.mp3>)
    theme, workdir, out_path,
    beat_clips={scene.index: [Path, ...]},        # the REAL per-beat clip timeline
    graphics=[(graphic_kind, graphic_text, graphic_body), ...],   # per scene; "" = pure footage
    graphic_assets={scene.index: "card.png"},     # optional card / portrait art
    energies=[...], emphasis=[...], shot_types=[...], roles=[...], chapters=[...],
    captions=True, music=<path|None>, transitions=True, title="...",
) -> Path
```

- A scene's **length** = `NarratedScene.duration` (derived from its narration audio).
- `FootageItem` has **only** `(index, path, is_video)` — *no in/out trim field*. The engine
  plays a video with `-stream_loop -1 -i <path>` from frame 0 for `round(duration*30)` frames.
- **Therefore** ClipStudio pre-cuts each chosen `[in,out]` to its own `.mp4` (stage 5) and
  injects the path via `beat_clips` — exactly the mechanism `locked_visuals.json` uses
  (`pipeline.py:1233-1262`). Zero renderer edits.
- Canvas is hard-coded **1920×1080 @ 30 fps**. Letterboxed/period sources get the
  "blurred copy behind" treatment automatically when `shot_types[i] == "archival"`.

Integration reference points (in the original engine, for grounding):
`FootageItem` footage.py:46 · `assemble()` assemble.py:8170 · canonical call pipeline.py:3147 ·
`_scene_video` (no in/out seek; add `-ss` here only if we ever stop pre-cutting) assemble.py:7187 ·
`locked_visuals.json` injection pipeline.py:1233 · `build_graphic_images` footage.py:24016 ·
`_guard_manual_replacement` web.py:4879.

---

## 4. Matching model (stage 4)

For a segment `S` and a candidate shot `C`, the confidence is a weighted blend of orthogonal
signals (each 0..1), reusing the engine's CLIP where possible:

```
score(S, C) =  w_clip   * clip_text_image_sim(S.expected_visual, C.keyframe)   # vidlore.visual_relevance
             + w_trans  * text_sim(S.text, C.transcript)                        # spoken-word overlap
             + w_face   * face_match(S.entities, C.faces)                       # if a named person is expected
             + w_obj    * object_match(S.keywords, C.tags)                      # open-vocab / YOLO tags (advanced)
             - p_reuse  * reuse_penalty(C.source_id, C.shot_index)             # anti-repetition
             - p_period * period_risk(S, C)                                     # anachronism guard (vidlore.period_guard)
```

Constraints applied during selection (not just scoring):
- **Anti-reuse:** cap uses per source and per individual shot; rising penalty per reuse.
- **Source diversity:** discourage long runs from a single source; target visual variety.
- **Pacing:** target ~2.5 s average clip (measured from the reference), vary 1–4 s, allow
  longer holds on cards.
- **Confidence floor:** anything below `min_confidence` (default 0.42, matching the engine's
  still-relevance gate) is **flagged** and routed to review rather than shipped silently.

The matcher emits, per segment: the chosen `ClipSelection`, its confidence + signal breakdown,
and the top-N alternates (for the review UI / re-pick).

---

## 5. Permission gate (stage 1)

Each source carries a `permission` value. The ingest stage **blocks** any source whose
permission is `unverified` (the default) unless the user explicitly sets one of:
`owner` · `licensed` · `public_domain` · `cc` · `fair_use_claim`. The chosen value and an
optional note are recorded verbatim in the ledger. The module makes **no legal determination** —
it records the user's assertion and proceeds or blocks accordingly.

Politeness: bounded concurrency, retry with backoff, and per-host pacing on download. The
module respects `yt-dlp`'s own rate-limit handling and does not attempt to bypass platform
protections.

---

## 6. Data model (models.py)

`SourceVideo` · `Shot` · `ScriptSegment` · `ClipCandidate` · `ClipSelection` · `ClipProject`.
All JSON-serializable; embeddings stored out-of-band (`.npy`) to keep manifests readable.
See `models.py` for exact fields.

Project layout on disk:
```
<project>/
  project.json            # ClipProject manifest
  script.txt              # input narration
  sources/                # ingested source videos
    <source_id>.mp4
  index/
    <source_id>.shots.json    # Shot[] (in/out, transcript, keyframe ref, face/obj signals)
    <source_id>.embeds.npy    # CLIP keyframe embeddings
    <source_id>/keyframes/*.jpg
  clips/                  # pre-cut trimmed sub-clips (what the renderer plays)
    seg_000.mp4 ...
  ledger.jsonl            # compliance ledger (one line per selection)
  review_queue.json       # flagged selections awaiting human approval
  output/                 # final MP4 + assemble workdir
```

---

## 7. Manual-review checkpoints (structural)

1. **Permission/licensing gate** — per source; `unverified` blocks. Software certifies nothing.
2. **Low-confidence match queue** — anything under `min_confidence`.
3. **Identity/face mismatch** — wrong-actor risk; confirm when a named person is expected.
4. **Specific-claim shots** — lines asserting a precise visual fact ("the scene where X dies")
   require human confirm.
5. **Reuse/variety audit** — if one source dominates the final cut.
6. **Final pacing/flow pass** — before the render is published.

---

## 8. Config flags (config.py)

`VIDLORE_CLIPSTUDIO_*` namespace (full list in `config.py`). Key ones:
`*_MIN_CONFIDENCE` (default 0.42), `*_TARGET_CLIP_SEC` (2.5), `*_MAX_REUSE_PER_SOURCE`,
`*_MAX_REUSE_PER_SHOT`, `*_WHISPER_MODEL` (default `base`), `*_SCENE_THRESHOLD` (27.0),
`*_CONCURRENCY` (download), and the matcher weights `*_W_CLIP/_W_TRANS/_W_FACE/_W_OBJ`.
Engine flags reused: `VIDLORE_CLIP_DIR`, `VIDLORE_VISUAL_RELEVANCE`, `ANTHROPIC_API_KEY`.

---

## 9. Status

Built stage-by-stage; see the session task list. Each stage is independently runnable and
writes its artifacts under `<project>/` so a run can resume and a human can inspect every
intermediate decision.
