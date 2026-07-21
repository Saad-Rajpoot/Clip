# Speed/Reliability Hardening — 2026-07-21 evening pass (on top of merge `a1ac742`)

Autonomous continuation: audit + harden the P0/P1 work, finish every suite, remove
nondeterminism, and land the remaining provable speed items. Same hard rule throughout:
identical editorial decisions; no gate/threshold/model/candidate change.

## Commits (branch `worktree-perf-speed-p0`, on a1ac742's content)
1. **P1.1** typed discovery statuses (`ok/empty/throttled/timeout/transport`) +
   bounded serial retry of ONLY failed/throttled provider buckets, partial results
   preserved, order/dedupe/ranking unchanged. 6 tests (429/timeout classification incl.
   yt-dlp message forms, retry convergence, one-provider-down partials, empty-is-an-answer,
   serial==parallel under failures).
2. **P1.2** provider-ACTUAL verdict caching: `llm.complete_ex` reports the branch that
   really served (canonical vision_config format); verify_frame attaches
   `vision_served_by`; every cache writer keys by the ACTUAL server; lookups reject
   provider-mismatched entries. Proven: gemini-down pass stores Claude-keyed entries;
   recovered-gemini pass re-asks fresh (zero cross-provider reuse); same-provider warm
   pass = pure hits. Legacy-seam shim: a monkeypatched `llm.complete` is honored.
3. **P1.3** certified embedding manifests (schema + model identity + dim + rows +
   per-row shot/keyframe/content-md5); stored vectors used ONLY when every identity
   matches; stale/reordered/re-extracted rows fall to the live path per-row; legacy
   indexes always live-path; verified backfill tool. 9 tests.
4. **P1.4** per-render metrics lifecycle (start_run/end_run, run id, monotonic timing,
   atomic reports) + missing counters (pool scans, cache misses, provider errors, serial
   retries, ffmpeg/ffprobe spawns armed per run). 4 tests.
5. **P1.5** the two documented-failure suites repaired to assert ACTUAL behavior
   (vidlore app: sfx ON + deliberate captions-off-by-default with its own UI copy;
   ClipStudio portal: captions ON + style validated against the preset registry;
   registry invariants instead of the obsolete count 70). **Full sweep: 24/24 suites.**
6. **P1.6** determinism: every `anoisesrc` spec now carries a stable crc32-derived seed
   (atmos/archival beds, risers, whooshes, clicks, sfx, shutter ticks) — loudness/
   duration/mix untouched. **Proof (two fresh identical-code canary runs, quiet
   machine, deterministic LLM stubs): decoded PCM identical, decoded frames identical,
   captions identical, FULL breakout audit identical (candidates 33==33, every rejection
   and log line), container sha256 identical — bit-identical MP4s.**
   Breakout ±1 finding: candidate enumeration is PROVEN stable under frozen inputs — no
   unstable iteration exists to fix. The earlier ±1 wobble was input-driven: the
   semantic dialogue-vs-narration gate is a real LLM judgment (fresh by design,
   fail-closed) and heavy machine contention can flip fail-open probe timeouts. Both are
   now bounded: repeated identical questions replay from the provider-actual cache, and
   the gates remain fail-closed/fail-open exactly as designed.
7. **P2 (landed)** staged-replay parallel DARK sweep: the tested clip set is frozen
   before the loop, `_clip_too_dark` verdicts precompute in a bounded thread pool (pure
   ffmpeg subprocess), the unchanged serial loop replays them byte-identically
   (measured 418s serial on the audit's production profile). Branding sweep kept serial
   deliberately (shared RapidOCR engine, concurrent-Run safety unproven — a flipped OCR
   verdict would be editorial).

## P2 items explicitly REJECTED this pass (with reasons)
- **Shared-decode / parallel-OCR ad+black scans**: another active session is rewriting
  the ad gate in main right now; implementing against the pre-rewrite base guarantees
  conflicts and double work. Follow-up after their commit lands.
- **Batched exact-frame probes** (window-qc/beat-prep): large surface across build.py's
  beat-prep, same collision risk, and needs its own frame-exact A/B (the probe grid must
  key on EXACT timestamps; a rounded key can straddle a cut and flip a decision).
- **Deep-index pipelining + download/index overlap**: requires thread-safety proofs
  around whisper/CLIP/OCR engines and byte-identical `_band_ocr_hit` inputs, plus a
  quiet-machine benchmark to demonstrate the win; index.py is also under concurrent
  edit. Deserves a dedicated pass with its own parity fixtures.
- **Smaller candidate pools / visually-equivalent single-pass re-encode**: excluded by
  the standing directive (quality/output-bytes change).

## Validation matrix (this pass)
- 24/24 repository suites green (no documented-failure carve-outs remain).
- Frozen canary (accept_mini, 43 beats): decisions/audits/captions/frames/PCM/sha256
  bit-identical across two same-code runs; earlier cross-code runs (baseline vs P0)
  already proved decision parity + frame identity.
- Cold vs warm (optimized): warm fallback rung calls 25 → 0 (all cache hits);
  stills 0.3–0.5s; verify 1.3–4.0s.
