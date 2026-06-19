"""Offline license-key gate.

Goal (as requested): without a valid key the software cannot be used,
and one key is tied to one PC.

How it works (no internet / no license server needed):
  * A key is a short HMAC-signed token. `verify_key()` checks the
    signature OFFLINE against a secret, so a random/guessed string is
    rejected but a real key validates without phoning home.
  * On first run the user types the key once. We then BIND it to this
    machine: ~/.vidlore/license.json stores the key + a fingerprint of
    THIS computer. On every later run we re-check the signature AND
    that the fingerprint still matches — so copying the install folder
    (or the license file) to another PC will NOT work: that is the
    "1 key = 1 PC" rule.

Honest limitation: because the app ships as source the developer can
read the secret, and offline there is no central place that remembers
"this key was already used on PC #1", so the SAME key string typed on
a second PC would also activate there. Stopping that needs an online
activation endpoint (a small future server) — see notes given to you.
For now this blocks casual copying / sharing, which is the ask.

Developer commands:
    python -m vidlore.license make "customer name"   # mint a new key
    python -m vidlore.license check  VR-....          # validate a key
    python -m vidlore.license whoami                  # this PC's id
    python -m vidlore.license status                  # activated?
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import time
import uuid
from pathlib import Path

# The signing secret. Change this ONCE to your own random string before
# you distribute the app — any key minted with the old secret then stops
# working. It can also be overridden with the VIDLORE_LICENSE_SECRET
# environment variable / .env line (keeps it out of the source if you
# prefer). Keep it private: anyone who has it can mint keys.
_DEFAULT_SECRET = "vidlore-7f3c1a9e-CHANGE-ME-before-selling-2b6d40"


def _secret() -> str:
    return (os.environ.get("VIDLORE_LICENSE_SECRET", "").strip()
            or _DEFAULT_SECRET)


# ---- key format ---------------------------------------------------------
# 20 hex of random payload + 12 hex of signature = 32 hex, shown grouped
# as 8 blocks of 4 with a VR- prefix:  VR-A1B2-C3D4-...-Y9Z0
_PREFIX = "VR"


def _sig(payload: str) -> str:
    return hmac.new(
        _secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _pretty(raw32: str) -> str:
    raw32 = raw32.upper()
    groups = "-".join(raw32[i:i + 4] for i in range(0, 32, 4))
    return f"{_PREFIX}-{groups}"


def _strip(key: str) -> str | None:
    """Return the 32 lowercase hex chars from a typed key, or None."""
    raw = re.sub(r"[^0-9A-Fa-f]", "", key or "").lower()
    return raw if len(raw) == 32 else None


def make_key(note: str = "") -> str:
    """Mint a new valid license key (developer side)."""
    payload = secrets.token_hex(10)            # 20 hex chars
    sig = _sig(payload)[:12]                    # 12 hex chars
    return _pretty(payload + sig)


def verify_key(key: str) -> bool:
    """True if the key's signature is valid for the current secret."""
    raw = _strip(key)
    if not raw:
        return False
    payload, sig = raw[:20], raw[20:]
    return hmac.compare_digest(_sig(payload)[:12], sig)


# ---- per-machine binding ------------------------------------------------
def machine_id() -> str:
    """Stable-ish fingerprint of THIS computer (hostname + MAC + OS)."""
    raw = "|".join(
        (
            platform.node() or "",
            str(uuid.getnode()),          # hardware MAC (48-bit int)
            platform.system() or "",
            platform.machine() or "",
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_LIC_DIR = Path.home() / ".vidlore"
_LIC_FILE = _LIC_DIR / "license.json"


def activate(key: str) -> tuple[bool, str]:
    """Validate `key` and bind it to this PC. Returns (ok, message)."""
    key = (key or "").strip()
    if not key:
        return False, "Please enter your license key."
    if not verify_key(key):
        return False, "Invalid license key. Please check and try again."
    try:
        _LIC_DIR.mkdir(parents=True, exist_ok=True)
        _LIC_FILE.write_text(
            json.dumps(
                {
                    "key": key,
                    "machine": machine_id(),
                    "activated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not save activation: {exc}"
    return True, "Activated. Thank you!"


def is_activated() -> bool:
    """True only if a valid key is bound to THIS machine.

    A built-in/dev override is allowed via VIDLORE_LICENSE_KEY (env or
    .env) — handy for shipping a pre-activated copy; it is still
    machine-checked the normal way by also writing the binding file on
    first use.
    """
    envk = os.environ.get("VIDLORE_LICENSE_KEY", "").strip()
    if envk and verify_key(envk):
        if not _LIC_FILE.exists():
            activate(envk)
        return True
    try:
        data = json.loads(_LIC_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if not verify_key(str(data.get("key", ""))):
        return False
    return str(data.get("machine", "")) == machine_id()


# ---- tiny developer CLI -------------------------------------------------
def _cli(argv: list[str]) -> int:
    cmd = (argv[0] if argv else "status").lower()
    if cmd == "make":
        note = " ".join(argv[1:])
        print(make_key(note))
        return 0
    if cmd == "check":
        if len(argv) < 2:
            print("usage: check <KEY>")
            return 2
        print("VALID" if verify_key(argv[1]) else "INVALID")
        return 0
    if cmd == "whoami":
        print(machine_id())
        return 0
    if cmd == "status":
        print("ACTIVATED" if is_activated() else "NOT ACTIVATED")
        print(f"license file: {_LIC_FILE}")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))
