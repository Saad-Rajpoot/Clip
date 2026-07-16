#!/bin/bash
# ClipStudio web portal — paste a script (+ optional voiceover) → finished video.
# Double-click to launch; it opens the browser automatically. Close this window (or Ctrl+C) to stop.
#
# PORTABLE: this script works from ANY location (internal disk, USB stick, a different Mac). It
# resolves everything relative to ITS OWN folder — never a hardcoded /Users/<name>/... path, which is
# exactly what broke it when the folder was copied to a USB drive and opened on another laptop.

# The folder this script lives in (resolves symlinks; safe for spaces in the path).
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$DIR" || { echo "ClipStudio folder not found at: $DIR"; read -r; exit 1; }

export VIDLORE_CLIPSTUDIO_PORT="${VIDLORE_CLIPSTUDIO_PORT:-5151}"
# the app itself + its bundled dependency folder
export PYTHONPATH="$DIR:$DIR/.clipstudio_libs:$PYTHONPATH"

# ── Portable assets shipped NEXT TO this script (used only when present, so a normal dev checkout
#    keeps using its venv/system copies and this stays a no-op there).
# ffmpeg: an exFAT/removable drive cannot store the executable bit — restore it before use.
if [ -f "$DIR/bin/ffmpeg" ]; then
  [ -x "$DIR/bin/ffmpeg" ] || chmod +x "$DIR/bin/ffmpeg" 2>/dev/null
  export VIDLORE_FFMPEG="$DIR/bin/ffmpeg"
fi
# CLIP models: the pipeline REFUSES to run without them (it will not ship visually unverified
# footage) and they are NOT auto-downloaded — a copied folder must carry them.
[ -d "$DIR/models/vidlore_clip" ] && export VIDLORE_CLIP_DIR="$DIR/models/vidlore_clip"
# Whisper/HF models (optional — auto-download when absent and online).
[ -d "$DIR/models/huggingface" ] && export HF_HOME="$DIR/models/huggingface"

# Interpreter: prefer a local venv when present, else the system python3.
PY="/usr/bin/python3"
[ -x "$DIR/.venv/bin/python" ] && PY="$DIR/.venv/bin/python"

# Renders are large and must never land on a removable drive by default — keep them on this
# machine's own Desktop unless explicitly overridden.
export VIDLORE_CLIPSTUDIO_PORTAL_DIR="${VIDLORE_CLIPSTUDIO_PORTAL_DIR:-$HOME/Desktop/clipstudio_output/portal}"

# stop any portal already running on this port (avoids "address in use")
pkill -f "vidlore.clipstudio.web" 2>/dev/null
sleep 1

echo "=============================================="
echo "  ClipStudio portal -> http://127.0.0.1:${VIDLORE_CLIPSTUDIO_PORT}"
echo "  folder : $DIR"
echo "  output : $VIDLORE_CLIPSTUDIO_PORTAL_DIR"
echo "  (opening your browser... keep this window open)"
echo "  Press Ctrl+C or close this window to stop."
echo "=============================================="

# open the browser once the server is up
( for i in $(seq 1 30); do
    curl -s -m1 "http://127.0.0.1:${VIDLORE_CLIPSTUDIO_PORT}/" >/dev/null 2>&1 && { open "http://127.0.0.1:${VIDLORE_CLIPSTUDIO_PORT}"; break; }
    sleep 1
  done ) >/dev/null 2>&1 &

exec "$PY" -m vidlore.clipstudio.web
