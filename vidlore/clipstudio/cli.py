"""Command-line entry for ClipStudio.

  python -m vidlore.clipstudio -p PROJECT_DIR -s script.txt --sources sources.json
  python -m vidlore.clipstudio -p PROJECT_DIR -s script.txt \
         --add "/path/movie.mp4" --permission fair_use_claim --note "owned copy"

Runs ingest → index → segment → match → cut → ledger → build and prints the compliance summary.
Use --no-build to stop before rendering (just produce selections + ledger for review).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .ingest import SourceSpec, load_specs
from .orchestrate import produce, produce_auto, PipelineError
from .models import PERMISSIONS
from .download import POLICIES


def _progress(m):
    print(m, flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m vidlore.clipstudio",
        description="ClipStudio — intelligent movie-clip editor for Vidlore.",
    )
    ap.add_argument("-p", "--project", required=True, help="project directory (created if absent)")
    ap.add_argument("-s", "--script", help="path to the narration script (.txt)")
    ap.add_argument("--sources", help="JSON/YAML list of sources: {ref|url|file, permission, note, title}")
    ap.add_argument("--add", action="append", default=[], metavar="REF",
                    help="add one source (URL or local path); repeatable")
    ap.add_argument("--permission", default="unverified", choices=PERMISSIONS,
                    help="permission for --add sources (unverified is BLOCKED)")
    ap.add_argument("--note", default="", help="permission note for --add sources")
    ap.add_argument("--name", default="", help="project name")
    ap.add_argument("--title", default="", help="video title")
    ap.add_argument("--theme", default="history",
                    help="render theme: crime|history|modern|minimalist|standard")
    ap.add_argument("--voice", default="", help="TTS voice (default edge en-US-GuyNeural)")
    ap.add_argument("--voice-provider", default="",
                    help="AI voice: kokoro|chatterbox (local neural) | elevenlabs (cloud) | edge "
                         "(free). Default kokoro.")
    ap.add_argument("--voice-preset", default="",
                    help="neural voice character (see voice_presets.py): deep_male_documentary, "
                         "warm_male_explainer, dark_investigative_narrator, "
                         "mature_female_documentary, calm_female_storyteller")
    ap.add_argument("--no-captions", action="store_true", help="disable burned-in captions")
    ap.add_argument("--no-tts", action="store_true", help="silent narration (skip edge-tts)")
    ap.add_argument("--no-build", action="store_true", help="stop after ledger; do not render")
    ap.add_argument("--no-enrich", action="store_true", help="skip the Claude visual-director pass")
    ap.add_argument("--force-index", action="store_true", help="re-index sources even if cached")
    # --- AUTO mode (topic + script only; no --sources/--add) ---
    ap.add_argument("--topic", default="", help="video topic — enables AUTOMATIC source discovery")
    ap.add_argument("--movie", default="", help="movie name hint (optional; inferred otherwise)")
    ap.add_argument("--policy", default="block", choices=list(POLICIES),
                    help="download policy for auto-discovered sources (block|open_only|approved_testing)")
    ap.add_argument("--max-sources", type=int, default=6, help="max sources to download in auto mode")
    ap.add_argument("--no-verify", action="store_true", help="skip the AI visual verifier pass")
    ap.add_argument("--voiceover", default="", help="path to your OWN voiceover audio (mp3/wav) — used instead of TTS, forced-aligned to the script")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    specs: list[SourceSpec] = []
    if args.sources:
        specs += load_specs(Path(args.sources))
    for ref in args.add:
        specs.append(SourceSpec(ref=ref, permission=args.permission, permission_note=args.note))

    try:
        if specs:
            # MANUAL mode: user supplied sources
            res = produce(
                args.project, script_path=args.script, source_specs=specs,
                name=(args.name or None), theme=args.theme, voice=args.voice, title=args.title,
                captions=not args.no_captions, use_tts=not args.no_tts,
                voiceover=(args.voiceover or None),
                do_build=not args.no_build, enrich=not args.no_enrich,
                force_index=args.force_index, progress=_progress,
            )
        else:
            # AUTO mode: discover + download sources from topic + script
            if args.no_enrich:
                print("note: --no-enrich applies to manual mode only (auto mode's analyze "
                      "stage IS the enrichment)", flush=True)
            res = produce_auto(
                args.project, topic=args.topic, script_path=args.script, movie_hint=args.movie,
                policy=args.policy, max_sources=args.max_sources, theme=args.theme, voice=args.voice,
                title=args.title, captions=not args.no_captions, use_tts=not args.no_tts,
                voiceover=(args.voiceover or None),
                voice_provider=args.voice_provider, voice_preset=args.voice_preset,
                verify=not args.no_verify, do_build=not args.no_build,
                force_index=args.force_index, progress=_progress,
            )
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2

    print("\n=== COMPLIANCE SUMMARY ===")
    print(json.dumps(res["summary"], indent=2))
    print("\nproject:      ", res["project"])
    print("output:       ", res["output"])
    print("ledger:       ", res["ledger"])
    print("review queue: ", res["review_queue"])
    print("review (html):", res.get("review_html"))
    s = res["summary"]
    if s.get("flagged_for_review"):
        print(f"\n⚠  {s['flagged_for_review']} selection(s) need human review — see review_queue.json")
    if s.get("permission_unverified_sources"):
        print(f"⚠  unverified-permission sources were blocked: {s['permission_unverified_sources']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
