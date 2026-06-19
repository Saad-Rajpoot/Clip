#!/usr/bin/env python3
"""Safe stale-temp cleanup utility (V1.2.1).

Removes orphaned motion-graphics PNG frame directories left in the system temp
dir by a crashed / interrupted render — without ever touching an in-progress
render (an age cutoff protects live work) or any output mp4 / cached asset (only
dirs holding the `f00000.png` frame marker are considered).

Each primitive already self-cleans its own frame dir after a successful (or
ffmpeg-failed) encode; this utility is the safety net for the rare hard-crash
orphan, and a manual "free my temp space now" tool.

  python3 tools/clean_stale_frames.py                 # remove dirs older than 1h
  python3 tools/clean_stale_frames.py --max-age 0     # remove ALL frame dirs now
  python3 tools/clean_stale_frames.py --dry-run       # report only, delete nothing
  python3 tools/clean_stale_frames.py --root /tmp/x   # a specific temp root
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Remove stale MG temp frame dirs.")
    ap.add_argument("--max-age", type=float, default=3600.0,
                    help="only remove frame dirs older than this many seconds "
                         "(default 3600; an active render is never deleted)")
    ap.add_argument("--root", default=None,
                    help="temp root to scan (default: system temp dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what WOULD be removed, delete nothing")
    args = ap.parse_args()

    base = Path(args.root) if args.root else Path(tempfile.gettempdir())
    if args.dry_run:
        now = time.time()
        hits, mb = [], 0.0
        for d in base.glob("tmp*"):
            try:
                mk = d / "f00000.png"
                if not (d.is_dir() and mk.exists()):
                    continue
                age = now - max(d.stat().st_mtime, mk.stat().st_mtime)
                if age < args.max_age:
                    continue
                sz = sum(f.stat().st_size for f in d.glob("*.png")) / 1e6
                mb += sz
                hits.append((d.name, round(age), round(sz, 1)))
            except OSError:
                continue
        print(f"[dry-run] {len(hits)} stale frame dir(s), ~{mb:.1f} MB under {base}")
        for name, age, sz in hits:
            print(f"  would remove {name}  age={age}s  {sz}MB")
        return 0

    from vidlore.motion_graphics import look
    res = look.sweep_stale_frames(root=args.root, max_age_s=args.max_age)
    print(f"removed {res['removed']} stale frame dir(s), freed ~{res['freed_mb']} MB "
          f"(under {base}, older than {args.max_age:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
