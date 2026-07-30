# Cost A/B protocol review — 2026-07-30 (result: all three items NOT shipped)

The three "needs_proof" cost items from `clipstudio-api-cost-audit-2026-07-30` were put through a
30-agent adversarial protocol review **before** any API spend. 25 findings, all confirmed, none
refuted. Every load-bearing claim below was then re-verified by hand against this repo.

Owner's bar: **"if quality drops even 1%, do not apply."** That bar is what killed all three.

## Why the planned A/B could not have answered the question

1. **n=120 can only ever certify "< 2.47%".** With zero dangerous flips observed, the exact
   (Clopper-Pearson) one-sided 95% upper bound on the dangerous-flip rate is 2.47% — 2.5× the
   owner's bar. One flip → 3.89%. The rule of three (3/n) forbids a sub-1% claim at this size, at
   any confidence level. To *reject* `p_dangerous ≥ 1%` needs n=299 (0 flips) … 913 (4 flips).
2. **The endpoint is a DIFFERENCE, not an absolute rate**, because flash-vs-flash has its own
   nonzero flip rate. Non-inferiority sizing on a difference (margin 1%, α .05, power 80%) needs
   ≈1,224 / 2,424 / 5,874 / 11,129 questions **per arm** for a 1% / 2% / 5% / 10% noise floor.
3. **The noise-floor arm would have measured a false 0%** — arm A1 vs A2 asks byte-identical
   questions, so A2 is served from `verdict_cache.json`, not from the model.
4. **Multiplicity**: 7 decision fields × 3 arms = 21 uncorrected comparisons → 65.9% chance of at
   least one spurious rejection when the arms are truly identical.

## Why each item is rejected on its own merits (no A/B needed)

### Item 2 — cap the verdict `reason` (est. saving $0.12/render): **NET NEGATIVE**
`PROMPT_VERSION` is hashed into `verdict_fingerprint` (verify.py:344). Any prompt-wording change
bumps it and **invalidates every cached verdict**. Measured on disk: 33,295 cached verdicts across
jobs ≈ **$18.81** to re-buy — about 157 renders' worth of the item's own saving. Rejected on
arithmetic; the A/B was never the binding question.

### Item 1 — route lenient classes to flash-lite (est. $0.18/render): **MECHANISM BROKEN**
- The named mechanism (`VIDLORE_CLIPSTUDIO_GEMINI_MODEL`, llm.py:200) is **process-global env
  state**, and the lenient callers are exactly the concurrent ones (verify's 4-worker prefetch,
  self-heal's 4–6 worker waves). Per-class routing via env is a race, not a design.
- `vision_served_by` is derived from the same env (`vision_config()`, llm.py:648), so a lite
  verdict would be filed under a **flash-keyed** fingerprint — precisely the cross-provider reuse
  `_hit_provider_ok` exists to prevent.
- Fixing provenance *without* re-keying gives a 100% cache miss on the lenient population, so
  **cost goes UP**. Shipping this needs a real per-call `model` parameter plumbed through
  `verify_frame` → `complete_ex` → the fingerprint. That is a substantial change, and it would
  *still* face the sizing problem above.

### Item 3 — churn-proof `image_id` (est. $0.35 on re-index cycles): **WOULD COLLIDE TODAY**
The proposal (derive `kf:` like the sheet id, from `(src_hash, shot span, KF_VERSION)`) collapses
the **venue layer's** key: `selfheal._venue_fp` has only a keyframe path, so it passes
`shot_start=0.0, shot_end=0.0` (selfheal.py:169). A span-based image_id would therefore give
**every candidate frame of a source the same key** — one frame's verdict served for another.
Also: a sentinel/empty `src_hash` turns the derived id into a wildcard (today the JPEG hash is an
independent second check), the persisted span is a lossy projection of the frame actually
extracted, and `KF_VERSION` does not exist yet.

Deterministic evidence gathered before rejecting (this part was sound):
`extract_keyframe` **is** byte-reproducible for identical (source bytes, timestamp) — 6/6 real
sources. The one keyframe that did not match its indexed copy had a **changed media file**
(recorded checksum `5811c988` vs actual `88475b33`; the 403 HD sweep replaced the bytes), so
`src_hash` would have invalidated it correctly. The idea is salvageable — the *derivation* is not.

## The finding worth more than all three savings

**Vision calls never set `temperature`.** `_gemini_complete` (llm.py:431) sets only
`max_output_tokens`, `system_instruction` and `thinking_config`; DeepSeek text gets
`temperature=0.3` (llm.py:313) but the verifier gets nothing → **Gemini's 2.5 default of 1.0**.
Every footage verdict is a sample from a distribution rather than a stable judgment. That is why
the noise floor is large, why re-running a render can move borderline beats, and why an
equivalence test is expensive to run here at all.

Pinning `temperature=0` would make verdicts reproducible and shrink every future A/B by an order
of magnitude — but it **changes editorial outcomes**, so under the owner's bar it is its own
decision, not a free win. Not applied. Recommended as the next thing to consider, on quality
grounds rather than cost.

## Reusable harness (committed, unused so far)
`tools/vision_ab_corpus.py` reconstructs real verifier questions from a finished job, stratified
across the venue / contextual / strict prompt shapes. Before it is used for a shipping decision it
needs: recorded (not reconstructed) payloads, a cache-bypassed noise-floor arm, a pinned provider,
gate-predicate outcomes rather than raw fields, and the sizing from §1.
