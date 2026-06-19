#!/usr/bin/env bash
# =============================================================================
# Vidlore — Continuous Documentary Intelligence Improvement Loop
# =============================================================================
# Launches the autonomous research → improve → report loop and keeps it running
# in the background. It studies premium documentary references, updates the
# DNA library, picks the highest-priority pending improvement, implements it
# safely (backup → test render → before/after score → keep or revert), writes
# a daily report, then sleeps and repeats — forever.
#
# It STOPS ONLY for critical blockers:
#   - research/STOP sentinel present   (touch it to halt gracefully)
#   - disk < 2 GB free
#   - ANTHROPIC_API_KEY missing         (degrades to cached/metadata analysis)
#
# Usage:
#   ./run_continuous_improvement_loop.sh            # background (nohup) + tail
#   ./run_continuous_improvement_loop.sh --fg       # run in the foreground
#   ./run_continuous_improvement_loop.sh --stop     # touch research/STOP
#   ./run_continuous_improvement_loop.sh --status   # show loop state + tail log
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
LOG="research/continuous_loop.log"
PIDFILE="research/continuous_loop.pid"
STOPFILE="research/STOP"
STATEFILE="research/loop_state.json"

mkdir -p research

# ── Subcommands ──────────────────────────────────────────────────────────────
case "${1:-}" in
  --stop)
    touch "$STOPFILE"
    echo "Touched $STOPFILE — the loop will halt gracefully after its current cycle."
    echo "(Delete $STOPFILE before relaunching to resume.)"
    exit 0
    ;;
  --status)
    echo "── loop state ──────────────────────────────────────────────"
    [ -f "$STATEFILE" ] && "$PY" -m json.tool "$STATEFILE" || echo "(no state file yet — loop has not run)"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "── loop is RUNNING (pid $(cat "$PIDFILE")) ─────────────────"
    else
      echo "── loop is NOT running ─────────────────────────────────────"
    fi
    echo "── last 25 log lines ───────────────────────────────────────"
    [ -f "$LOG" ] && tail -n 25 "$LOG" || echo "(no log yet)"
    exit 0
    ;;
esac

# ── Cost controls (override by exporting before calling, or edit here) ───────
export RESEARCH_MAX_VIDEOS_PER_CYCLE="${RESEARCH_MAX_VIDEOS_PER_CYCLE:-3}"
export RESEARCH_MAX_IMPROVEMENTS_PER_CYCLE="${RESEARCH_MAX_IMPROVEMENTS_PER_CYCLE:-1}"
export RESEARCH_MAX_LLM_CALLS_PER_CYCLE="${RESEARCH_MAX_LLM_CALLS_PER_CYCLE:-10}"
export RESEARCH_SLEEP_BETWEEN_CYCLES_MIN="${RESEARCH_SLEEP_BETWEEN_CYCLES_MIN:-10}"
export RESEARCH_DAILY_BUDGET_MODE="${RESEARCH_DAILY_BUDGET_MODE:-safe}"
export RESEARCH_CONTINUOUS=1

# A stale STOP file would halt us immediately — clear it on a fresh launch.
if [ -f "$STOPFILE" ]; then
  echo "Removing stale $STOPFILE before launch ..."
  rm -f "$STOPFILE"
fi

CMD=("$PY" tools/autonomous_research_loop.py --mode continuous)

echo "============================================================"
echo "  Vidlore Continuous Improvement Loop"
echo "  budget: $RESEARCH_DAILY_BUDGET_MODE | "\
"$RESEARCH_MAX_VIDEOS_PER_CYCLE vids/cycle | "\
"$RESEARCH_MAX_IMPROVEMENTS_PER_CYCLE improve/cycle | "\
"sleep ${RESEARCH_SLEEP_BETWEEN_CYCLES_MIN}min"
echo "  stop:   ./run_continuous_improvement_loop.sh --stop"
echo "  status: ./run_continuous_improvement_loop.sh --status"
echo "============================================================"

if [ "${1:-}" = "--fg" ]; then
  exec "${CMD[@]}"
fi

# Background launch with nohup; record PID; tail the log so the user sees life.
nohup "${CMD[@]}" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "Loop started in background (pid $(cat "$PIDFILE")). Logging to $LOG"
echo "Tailing log (Ctrl-C to stop watching — the loop keeps running):"
echo "------------------------------------------------------------"
sleep 1
tail -n 20 -f "$LOG"
