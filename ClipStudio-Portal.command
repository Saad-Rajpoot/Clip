#!/bin/bash
# ClipStudio web portal — paste a script (+ optional voiceover) → finished video.
# Double-click to launch; it opens the browser automatically. Close this window (or Ctrl+C) to stop.

cd /Users/hussnain/Desktop/vidlore-clipstudio || { echo "ClipStudio folder not found"; read; exit 1; }
export VIDLORE_CLIPSTUDIO_PORT=5151
export PYTHONPATH="/Users/hussnain/Desktop/vidlore-clipstudio:$PYTHONPATH"

# stop any portal already running on this port (avoids "address in use")
pkill -f "vidlore.clipstudio.web" 2>/dev/null
sleep 1

echo "=============================================="
echo "  ClipStudio portal -> http://127.0.0.1:5151"
echo "  (opening your browser... keep this window open)"
echo "  Press Ctrl+C or close this window to stop."
echo "=============================================="

# open the browser once the server is up
( for i in $(seq 1 20); do
    curl -s -m1 http://127.0.0.1:5151/ >/dev/null 2>&1 && { open "http://127.0.0.1:5151"; break; }
    sleep 1
  done ) >/dev/null 2>&1 &

exec /usr/bin/python3 -m vidlore.clipstudio.web
