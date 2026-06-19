# ClipStudio Fix Pass — Applied 2026-06-10

Two rounds, all inside `vidlore/clipstudio/` (parent engine untouched — engine_guards 106/0).
Full unified diff vs pre-fix: `FIXES_APPLIED.diff` (pre-fix backup: `/tmp/clipstudio_backup_pre_fixes`).
Regression suite: `tools/test_clipstudio_fixes.py` — **82/82**. Web portal: NO authentication added
(per user instruction); only error-handling/eviction hygiene.

## Round 1 — from the 60-finding deep review (`REVIEW_FINDINGS.md`)

**HIGH**
- verify.py — promoting an alternate now rewrites `beat_windows` (rejected pick no longer airs).
- build.py — visual-budget loop seeds from the stage-3 fallback; plan_beats failure can no longer
  zero the beat counts (ZeroDivisionError dead-fallback fixed); final `nbeats` floor 1.
- ingest.py — `ingest_sources` merges instead of REPLACING `proj.sources` (incremental --add safe).
- discover/orchestrate — coverage+anchor force-includes survive download (cfg copied via
  `dataclasses.replace`, head = budget, `dl_limit` covers the returned list).
- hd_download/download/ingest — produced-file pickup requires a clean exit, prefers the merge
  target, never accepts a video-only `.fNNN` DASH fragment / merge temp / zero-byte file
  (`find_produced_video`); probe-validated.

**MEDIUM/LOW (clusters)**
- Pipeline: `PipelineError` replaces SystemExit (web worker + CLI boundary); produce() usable-gate
  requires `status == ok`; analyze beat-index/batch-JSON guards; review.py null-confidence guard;
  CLI flag forwarding (--voiceover manual, --force-index auto).
- LLM: keyless-Claude fast-fail (no 5.6s retry burn); Claude↔Gemini model-id cross-mapping;
  GOOGLE_APPLICATION_CREDENTIALS hard override of stale env; ANTHROPIC_MODEL read call-time.
- Index: resume cache checks capabilities (faceid/ocr/roster sidecar) + corrupt-cache tolerance;
  OCR no longer needs a roster (junk/watermark gates live in manual mode); atomic shots/meta/
  project.json writes; 0-shot media not cached.
- Build/ffmpeg: watermark crop no longer hard-scales 1280x720; logo-corner vote requires a true
  corner + skips subtitle strips; caption-dodge per-beat; energy-floor desync removed (build keeps
  assemble's k); slow-mo/cut/recut return-code checks; theme→music buckets cover real engine
  themes; tolerant env parses.
- Web/download: BaseException job handler + finished-anchored TTL eviction; traceback only to
  server console; portal policy env-overridable; duplicate-sid submission dedup (download+ingest);
  permission strings validated against the enum.
- Lows: shot-merge floor = max(min_shot, min_clip); mistyped-local-path detection (case-insensitive
  schemes, host:port); HD_ENABLED parsing; pot-server lock + tempdir log + closed fd; bounded
  _BRIGHT_CACHE; ffprobe candidate order (env first, expanduser'd dev path kept — only ffprobe on
  this machine); discover_prefer_height knob wired (monotonic tiers).

## Round 2 — adversarial review of the Round-1 diff (16 confirmed + 13 self-verified)

- **HIGH** match.py — alternates ordering had stripped the anchor bonus; restored via
  `signals["anchor_bonus"]` in the sort key (reported confidence stays pure quality).
- verify.py — windows of alternates the verifier explicitly FAILED are dropped too (av=None
  transport errors deliberately not treated as judgments).
- build.py — FINAL plan recompute from the final energies (loop-exhaustion desync); beat clips cut
  to plan_beats' REAL non-uniform lengths (hold beat no longer stream-loops); caption-dodge windows
  use those same lengths; slow-mo failure re-cuts to a fresh path (atomic replace).
- match.py — dark-scene title-token removal blanks in place (no spliced false adjacency) and only
  meaningful title words; prefix matching bounded to short suffixes ("dragon"→"dragons", never
  "jon"→"jonquil") in match + discover (coverage + anchor); _BRIGHT_CACHE partial FIFO eviction.
- ingest/download — probe gate falls back to yt-dlp metadata (valid .ts captures pass); unreadable
  artifacts unlinked so re-runs actually re-download; sid dedup pre-submission; merge displaces
  same-url/same-checksum stale entries.
- hd_download — failed run sweeps its fragments/partials/info.json.
- index.py — roster recorded AND checked as a capability; embeds.npy atomic (tmp+replace, before
  shots.json); non-dict meta tolerated; forced re-index of unreadable media drops the stale cache.
- llm.py — provider/model/location knobs all read call-time; hard claude fallback if eng_cfg's own
  model id is non-Claude.
- web.py — TTL anchors on job completion (finished), not start.
- analyze/segment — empty-`[]` enrich reply spends the retry; segment enrich guards non-numeric
  "i" + non-dict elements.
- config.py — ffprobe: env override first; dev-machine path via expanduser (verified the ONLY
  ffprobe here); discover `_hd_bonus` knob now monotonic (default tiers byte-identical).
- orchestrate.py — `dl_limit = len(candidates)` (discover's own construction bounds it; any prefix
  cap sliced the anchors appended last).

## Acknowledged, by design
- In auto mode, `max_sources` (user budget) overrides `VIDLORE_CLIPSTUDIO_DISCOVER_TARGET`
  (the env knob still governs direct `discover_sources()` callers) — documented in code.
- Portal remains unauthenticated per user instruction; `VIDLORE_CLIPSTUDIO_PORTAL_POLICY` can
  override the hard-coded `approved_testing` policy.
