# ClipStudio Speed Audit — 2026-07-21

Goal: major wall-clock reduction with ZERO quality change (identical decisions, verdicts, pixels).
Method: quantitative log analysis (runs `24e12f7d2f` ×5 attempts, live `18e1fb9293` ×2 passes) + code reading of every hot subsystem + adversarial per-fix verification (would the fix change ANY output?). 22 agents, all findings code-cited.

## Where the time actually goes (measured, live 18-min-video run)

| Bucket | Wall time | Share | Note |
|---|---|---|---|
| verify + repair machinery | ~63% | verify serial loop 2735s + image-fallback 3564s + recovery 860s + venue fallbacks 416s | Vision API itself is ~2s/call (221 verdicts warm in 108s) — the cost is SERIAL ORCHESTRATION around cached/fast calls |
| deep index | 2352s | 21% | ASR 1459s (62% of index); strictly serial per source; ~3 decode passes/source + thousands of ffmpeg keyframe spawns |
| discover | 784s | 7% | 200 search queries strictly serial @ ~3s |
| download | 641s | 6% | already parallel |
| encode+mux | 162s | ~1% | VideoToolbox is NOT the problem |

Repeated work measured: same still frame vision-judged **55×** across beats (580 judgments over 109 unique frames); recovery rediscovery invoked **7×** across runs (2825s, 5/7 found nothing, same beat lists); canary project executed **full verify 5×** on identical 228 beats (~85 min); image-fallback per-beat setup ~25s × 102 visits = 2591s before any vision call; finished 1074s video gets **5 full decode/encode passes** at the tail (encode → bars bake → breakout-caption burn → ad scan → black scan).

## Verified optimizations (all quality-neutral under listed conditions)

### P0 — biggest wins
1. **OPT-1 Image-fallback setup (~35–40 min)** — `image_fallback.py:142-156,382-439`, `orchestrate.py:251-260,343-372`. Per-beat setup re-runs CLIP image forward passes per (beat×call×shot) (up to ~45k passes) + re-parses shots.json per candidate. Fix: use persisted index embeddings (bit-identical dot product), one top-N scan per beat, memoize by (path,mtime). CONDITION: replay the *sliding* scan_cap semantics exactly (`used_keys` skip happens before `scanned+=1` — naive top-N changes picks).
2. **OPT-3 Cache/prefetch the verify fallback chain (~10–25 min)** — `verify.py:1022-1189`. Fallback rungs (promote/venue/lenient re-ask) bypass verdict_cache entirely — only primaries are cached; this is why re-runs pay ~800–1200s each. Fix: fingerprint+cache all rungs (venue_fallback flag must join the fingerprint), phase-2 prefetch rung candidates for segs whose cached primary is `replace`, save cache incrementally. This ALSO kills most of the cross-attempt repetition (canary's 5× verify).
3. **OPT-4 Parallel discovery + recovery fan-out (~12 min + 4–7 min/recovery)** — `discover.py:826-830,768-770,793-809`. 200 serial network queries → 8-worker pool with ORDER-PRESERVING aggregation (dedupe at :833-842 is first-occurrence-wins; concatenate per-query results in original query order). Cache `_fetch_subs_text`/probe results per URL in project dir so recovery rounds are cache hits.
4. **OPT-5 Still-layer verdict memoization + small pool (~8–14 min)** — `orchestrate.py:397-564,300-320`. 580 serial ~2.1s judgments over 109 unique frames. Fix: memoize verify_frame by full verdict inputs (needs OPT-3's venue-variant fingerprint field — without it, still-layer and verify-stage questions collide on identical fingerprints), freeze per-beat candidate list, judge in a 2–3-worker pool, install serially in ranked order.
5. **OPT-2 Verify verdict-prefetch everywhere (~5–10 min/pass)** — `verify.py:842-923`. Prefetch is proven (221 verdicts/108s @4 workers) but canary's 5 passes all ran fully serial (env `VIDLORE_CLIPSTUDIO_VERIFY_WORKERS` referenced only at verify.py:856-859, set nowhere: not web.py `_run_job`, not cli, not Portal.command, not .env). Pin at the MEASURED 4 workers (Gemini rate-limit safety; 429s would change fallback behavior).

### P1
6. **OPT-6 Pipeline deep index (~10–13 min)** — `index.py:667-675`. Overlap decode-only work (scenedetect/keyframes/flags/OCR) of source N+1 with main-thread model calls (whisper/CLIP) of source N. Model thread-safety invariant intact (models only on main thread). CONDITION: `_band_ocr_hit` input must be byte-identical (same crop coords/resize/quality).
7. **OPT-8 Overlap independent stages (~8–13 min)** — `download.py:185-235`, orchestrate stage chain. Feed the serial indexer per-completed-download (after checksum-dedup, SOURCE_OK only); start narration bundle + Face-ID refs early behind a turbo flag.
8. **OPT-7 Batch build-phase frame probes (~5–10 min)** — `build.py:1917-1949,5096-5107`. Hundreds of cold per-call ffmpeg/VideoCapture seeks into multi-GB sources during beat-prep (900s bucket). Pre-extract the full probe grid per source in one pass, pre-populate `_FRAME_TXT_CACHE` + a frame-hash cache. CONDITION: hash cache keys on EXACT t (existing `_FRAME_TXT_CACHE` rounds keys but seeks unrounded t — replicate exactly; 0.1s rounding can straddle a cut and flip decisions).
9. **OPT-9 Parallel post-cut QA sweeps (~5–9 min)** — `build.py:5528-5609,5840-5862`. Near-black scan = 418s serial full decode of 231 clips; branding/dark/dodge ≈ 16 spawns/clip ≈ 4500 serial spawns. One decode per clip (union of timestamps), verdicts in a pool, replacement loops replay serially on cached verdicts. CONDITION: staged detection per sweep (later sweeps see replaced clip paths like `_nobrand.mp4`).
10. **OPT-10 Final-video tail (~3.5–4.5 min)** — `build.py:4064-4214`. Ad-scan's 224s is single-instance serial OCR (decode is only ~8s — black scan proves it). Parallel OCR workers; share one decode between ad+black gates via filter_complex split; batch breakout-QA seeks. Optional (+2.5 min): fold bars-bake + caption-burn re-encodes into one pass (changes encode count — needs explicit approval since output bytes differ, visually identical but not bit-identical).

### P2
11. **OPT-11 Index micro-dedup (~2–6 min)** — batch CLIP ONNX Run (batch outputs verified `np.array_equal` to singles on this machine), single monotonic decode for shot flags. CONDITION: don't feed cv2 arrays to RapidOCR (it decodes via PIL — decoder identity matters).
12. **OPT-12 Breakout pipeline (~0.5–2.5 min)** — batch pick-loop probes; overlap encode with loudnorm measure. Probes fail OPEN — keep timeouts generous, bound concurrency.
13. **OPT-13 Web-exact fallback cache (plausible, ~0.5–3 min)** — cached rejections MUST still increment `tried` (budget cutoff parity).
14. **OPT-14 Outage fast-fail (0 healthy; 2–5+ min on outages)** — safe subset only: shared outage Event across prefetch workers + stop retrying billing errors. Do NOT add global 1-attempt mode (changes verdicts under transient 429 storms).

## Totals
Conservative sum for a fallback-heavy render like the live one: **~100–140 min saved on a ~3.2 h render → roughly halves**, without touching any gate, threshold, model, or sample count. Systemic multiplier: OPT-3's rung caching + OPT-4's recovery caching also collapse the cross-attempt repetition (canary spent 5.7 h compute across 5 attempts for one video).

## Determinism ground rules for implementation
- Every parallel fan-out aggregates in original serial order before any downstream consumer.
- Every cache key must include EVERY prompt/probe input (venue flag, exact t, crop params, model, PROMPT_VERSION).
- Vision concurrency stays at measured-safe 4 (429s alter fallback paths).
- Replay quirks bit-exactly (sliding scan_cap; first-occurrence dedupe; fail-open probes keep their timeouts).
- A/B acceptance: one real render per fix; diff project.json selections, all audit JSONs, and final.mp4 hash (except OPT-10d which changes bytes by design).

Full agent outputs: session scratchpad `tasks/wqjbxujdg.output` (this session) — per-fix conditions and line-level citations.
