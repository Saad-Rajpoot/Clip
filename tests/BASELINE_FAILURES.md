# Baseline-failing suites (fail identically on `main`, before this branch)

Two suites in `tests/` fail on a clean `main` checkout. They are recorded here with exact
commands and signatures so they are never silently absorbed into this branch's results.

Verified on `main @ c2c3c2e` and on `fix/relevance-verifier-timeline-breakouts`:
**same test, same line number, same assertion, same message.**

Run everything from the repo root with `PYTHONPATH=.` (these two suites do not insert `sys.path`
themselves, unlike `tests/test_relevance_fix_pass.py` and `tools/test_clipstudio_fixes.py`).

---

## 1. `tests/test_rc4_defaults_and_metrics.py`

```
PYTHONPATH=. python3 tests/test_rc4_defaults_and_metrics.py
```

```
File "tests/test_rc4_defaults_and_metrics.py", line 51, in test_form_template_defaults_on
File "tests/test_rc4_defaults_and_metrics.py", line 38, in check
AssertionError: FAIL: template captions default ON  (f.get('captions','1'))
```

**What it asserts** (line 48-51): that the literal source strings `f.get('sfx','1')` and
`f.get('captions','1')` appear inside `vidlore.clipstudio.web._FORM`.

**Why it fails:** neither literal exists in `_FORM` any more. Measured:

| probe | present in `_FORM` |
|---|---|
| `"f.get('sfx','1')"` | False |
| `"f.get('captions','1')"` | False |
| `"f.get('captions'"` | False |
| `"captions"` | True |

The portal form still handles captions; it no longer spells the default that way. This is a stale
**source-string** assertion against portal HTML, last touched in the initial commit (`23e2dcf`).

## 2. `tests/test_rc4_director_gating.py`

```
PYTHONPATH=. python3 tests/test_rc4_director_gating.py
```

```
File "tests/test_rc4_director_gating.py", line 143, in <module>
File "tests/test_rc4_director_gating.py", line 137, in test_registry_length_unchanged
AssertionError
```

**What it asserts** (line 137): `len(registry.REGISTRY) == 70`.

**Why it fails:** `len(vidlore.motion_graphics.registry.REGISTRY)` is **71**. A motion-graphics
template was added after the test pinned the count.

---

## Relation to this branch

None. Neither suite imports or exercises any module this branch changes
(`verify.py`, `era.py`, `policy.py`, `faceid.py`, `assemble.py`, `index.py`, `match.py`,
`build.py`, `analyze.py`, `orchestrate.py`). One asserts a string inside the portal's HTML form;
the other counts motion-graphics templates.

**Deliberately not fixed here.** Both encode product decisions someone else made — whether the
portal form should carry an explicit caption default, and whether the 71st template belongs in the
production registry. Changing either to make a number green would be guessing at intent from a
test file, inside an unrelated branch. They are surfaced instead so the decision is a real one.

## Reproduction

```
git checkout main
PYTHONPATH=. python3 tests/test_rc4_defaults_and_metrics.py   # AssertionError line 51
PYTHONPATH=. python3 tests/test_rc4_director_gating.py        # AssertionError line 137
```
