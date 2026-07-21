# Speed P0/P1 Implementation — 2026-07-21 (branch `worktree-perf-speed-p0`)

Implements the P0 + P1-discovery items of `SPEED_AUDIT_2026-07-21.md` under the hard rule:
**faster must mean identical editorial decisions** — no relevance rule, threshold, budget,
model, candidate count/order, quality floor, fallback policy, breakout/caption behavior, or
release gate was modified.

## Commits
1. `Perf: decision-neutral instrumentation` — perf_metrics module (counters/timers/stage
   durations/ffmpeg-spawn audit hook, `VIDLORE_PERF=1` → `output/perf_report.json`).
2. `P0-1: still-pool relevance from persisted index embeddings` — `pick_pool_still` scores
   via `embeds.npy[Shot.embed_row]` (identity proven: 47/47 stored-vs-fresh vectors
   byte-equal on a real project index, incl. CoreML EP; 47/47 relevance scores equal);
   live-path fallback on missing/stale rows; shots.json/embeds/relevance memoized per
   render; sliding scan_cap / used_keys / ordering untouched.
3. `P0-2/P0-4: one verdict-cache layer for every verifier rung + still-verdict dedup` —
   strict promotion, exact→contextual, venue-contextual, lenient generic-filler re-ask, and
   the still layer's venue verdicts all route through verdict_cache.json with the complete
   question fingerprint (fingerprint gains `venue_fallback`, appended only when True so
   every pre-existing cache key stays valid; still reuse is NEVER by image path alone).
   Schema-valid successes only; errors/timeouts/breaker outcomes never cached; incremental
   atomic saves.
4. `P0-3: enable the proven 4-worker verify-verdict prefetch in portal + CLI` — the env
   (`VIDLORE_CLIPSTUDIO_VERIFY_WORKERS`) was set by no production runner; portal sets it
   symmetrically per job (operator process-start value wins), CLI uses setdefault; library
   default stays 1 (breaker suites unchanged).
5. `Tests: adversarial decision-parity suite` (9 checks) + `P1 discovery parity suite`
   (3 checks).
6. `P1-6: bounded parallel discovery fan-out` — per-query buckets, original-order concat
   before first-occurrence dedupe (candidate-identical, proven under adversarial completion
   jitter), 4 workers default, serial in-order retry of errored buckets, positive-only
   per-URL subtitle/HD-probe caches (probe key includes the height cap).

## P0-5 finding (review-draft retry re-ran verify + image-fallback)
Root cause, from `orchestrate.py:1152-1161`: the stage checkpoints' signature chain hashes
the usable-source set; the bounded recovery pass in pass-1 **downloads new sources**, so on
the auto-review resume `_sig_index` (hence match/cut/verify/recover sigs) legitimately
changes and those stages re-run. This is semantically required — the re-match lets every
beat consider the recovered source, and the measured pass-2 decisions did change — so the
pass is NOT skipped. The waste inside it is instead removed by commit 3: every unchanged
question now replays from the verdict cache (canary: 25 fallback calls cold → 0 calls /
25 cache-hits warm), so a repeated pass pays only for genuinely new questions.

## Validation
- Full suite: 16/18 suites pass; the 2 failures are the pre-existing, documented baseline
  failures (`tests/BASELINE_FAILURES.md`) failing identically on `main`.
- New adversarial suites: 9/9 + 3/3 (cache-key fields incl. golden legacy derivation,
  corrupt-cache tolerance, zero-call cached replay with byte-identical decisions,
  strict/lenient key separation, never-cache-errors, workers-parity, pool-abort-to-serial
  with breaker contract, persisted-vs-live still equality + stale-row fallback,
  discovery order-preservation under jitter, positive-only URL caches).
- Frozen canary (accept_mini, 43 beats, deterministic fake vision oracle, recovery no-op,
  web-images off — identical for both code versions; ran `nice -19`):
  * decisions: **identical across all four runs** (baseline cold/warm, optimized cold/warm)
    — selections, flags, verdicts, downgrade labels, relevance classes, stills, audits,
    final.srt;
  * decoded video frames: **bit-identical** (5935/5935 frames, same md5) baseline vs
    optimized;
  * decoded PCM: differs run-to-run **on unmodified baseline as well** (base_cold vs
    base_warm md5s differ; video frames stable) — pre-existing audio-bed nondeterminism in
    the untouched build stage, not introduced by this branch. Likewise the breakout
    candidate count wobbles ±1 between two identical-code runs (accepted breakouts always
    identical); both noted as pre-existing issues worth their own fix.
  * timings (contended machine — the live render ran concurrently; counts are the honest
    metric): stills stage 69.3s/133.2s (baseline cold/warm) → **0.5s/0.3s**; verify
    7.4s → 4.0s cold / 1.3s warm; warm fallback rung calls 25 → **0**.
  * counters (optimized cold): 1450 relevance memo hits + 354 persisted-embedding dots vs
    162 live CLIP embeds — the baseline pays a full CLIP forward pass for each of those.

## Deliberately NOT implemented (need separate proof — can alter frames/scheduling/pixels)
Smaller candidate pools; single-pass bars/caption encode; deep-index pipelining;
download/index overlap; batched exact-frame probes; shared-decode OCR/QA; the audit's
"small pool" for still verdicts (P0-4 was implemented as pure dedup instead, per spec).

## Expected production effect (from the audit's measured profiles, non-overlapping)
- Image-fallback per-beat setup (measured 25s/beat × ~100 beat-visits on fallback-heavy
  renders): collapses to dot products + memo hits (canary: >100×).
- Repeated passes (review-draft retry, resume, rerender): verify fallback ladder and still
  verdicts replay from cache — the measured ~800-1500s per repeated verify pass and the
  still-layer's repeated judgments stop being re-paid.
- Serial verify warm: prefetch pool ON in production (was never enabled) — measured 221
  verdicts in 108s vs ~1200s serial on cold passes.
- Discovery: ~200 serial queries × ~3s → 4-way overlap; recovery rounds' subtitle/probe
  refetches become cache hits.
