"""Which native stack is this render about to run on? — decided before any work, not after a crash.

Two renders have now been killed by the same uncatchable abort: CTranslate2's thread pool calling
`std::condition_variable::wait`, which libc++ declares `_NOEXCEPT`, so the throw is an immediate
`std::terminate` — SIGABRT, no Python traceback, and in the portal's case the render, the portal and
the in-memory job registry all die together.

Both crash reports name the same library build: `libctranslate2.4.7.1.dylib`, loaded from the user
site-packages of a system interpreter. The repo's own `.venv` holds **4.8.0**. The two renders never
ran on the environment this project installs, because starting the portal as

    python3 -m vidlore.clipstudio.web        # instead of ./ClipStudio-Portal.command

silently swaps the whole native stack — ctranslate2, onnxruntime, cv2, numpy — for whatever the
system interpreter happens to have. `run_state` already RECORDED the version that ran (that is how
this was attributed at all), but recording is not checking, and nothing checked.

So the environment is decided at STARTUP, where it costs a second, instead of being discovered two
hours into an index pass. That is the whole point: a render this expensive must fail in the first
line of its log or not at all.

This does NOT claim 4.8.0 fixes the abort — that is unproven, and if a 4.8.0 render aborts the same
way the next step is isolating the ASR in a child process. It claims something narrower and certain:
a render must run on the environment the project installs and tests, and when it cannot, it must say
so before it burns the hours.

Escape hatch: VIDLORE_CLIPSTUDIO_ALLOW_FOREIGN_ENV=1 downgrades the refusal to a loud warning, for a
machine where the venv is genuinely not the right answer (a Windows checkout, a packaged copy on a
USB stick — this project has both). It is deliberately explicit: nothing infers it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The libraries whose thread pools can take the process down without a traceback, plus the two that
# decide what a frame looks like. Same list run_state records, for the same reason.
NATIVE = ("ctranslate2", "faster_whisper", "onnxruntime", "cv2", "numpy")

OVERRIDE_ENV = "VIDLORE_CLIPSTUDIO_ALLOW_FOREIGN_ENV"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def project_venv(root=None):
    """The venv this project installs into, or None when the checkout has no venv at all."""
    root = Path(root or project_root())
    venv = root / ".venv"
    return venv if (venv / "bin" / "python").exists() or (venv / "Scripts" / "python.exe").exists() \
        else None


def _same(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:                                      # noqa: BLE001 — a path we cannot resolve
        return False                                       # is not a match


def running_in_project_venv(root=None) -> bool:
    venv = project_venv(root)
    return bool(venv) and _same(sys.prefix, venv)


def native_versions() -> dict:
    out = {}
    for mod in NATIVE:
        try:
            out[mod] = str(getattr(__import__(mod), "__version__", "?"))
        except Exception:                                  # noqa: BLE001 — absence is a fact too
            out[mod] = "absent"
    return out


def native_origin(mod: str = "ctranslate2") -> str:
    """Where the loaded copy actually came from — the user site-packages path is the tell."""
    try:
        return str(getattr(__import__(mod), "__file__", "") or "")
    except Exception:                                      # noqa: BLE001
        return ""


def summary() -> str:
    """One line for the top of a build.log. A crash you can attribute is a crash you fix once."""
    v = native_versions()
    return (f"py={sys.version.split()[0]} prefix={sys.prefix} "
            + " ".join(f"{k}={v[k]}" for k in NATIVE))


def foreign_reason(root=None) -> str:
    """Why this interpreter is not the project's own, or "" when it is (or when there is no venv)."""
    venv = project_venv(root)
    if venv is None or running_in_project_venv(root):
        return ""
    ct2 = native_versions().get("ctranslate2", "absent")
    origin = native_origin("ctranslate2")
    where = " (user site-packages)" if "Library/Python" in origin or "/.local/" in origin else ""
    return (f"this interpreter is not the project environment.\n"
            f"    running : {sys.executable}\n"
            f"    loaded  : ctranslate2 {ct2}{where}\n"
            f"    project : {venv}\n"
            f"  Two renders have been killed by a CTranslate2 abort on a stack that was NOT this "
            f"project's; a render must run on the environment the project installs.\n"
            f"  Start it with ./ClipStudio-Portal.command, or run {venv}/bin/python directly.\n"
            f"  Set {OVERRIDE_ENV}=1 to proceed anyway (it will be recorded in the run).")


def override_requested() -> bool:
    return os.environ.get(OVERRIDE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def enforce(root=None, *, driver: str, log=None) -> str:
    """Refuse to start a render on a foreign environment. Returns the summary line for the log.

    Raises RuntimeError before any work happens. With the override set it warns instead — loudly,
    and the warning is returned so the caller can put it in build.log where the next investigation
    will look.
    """
    emit = log or (lambda m: print(m, flush=True))
    reason = foreign_reason(root)
    if not reason:
        return summary()
    banner = f"ENVIRONMENT ({driver}): {reason}"
    if not override_requested():
        raise RuntimeError(banner)
    emit(f"⚠ {banner}")
    emit(f"⚠ proceeding because {OVERRIDE_ENV}=1 — {summary()}")
    return summary() + f" [{OVERRIDE_ENV}=1]"
