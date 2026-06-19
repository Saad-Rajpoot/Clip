"""Command-line entry.

Two-stage flow (the differentiator vs Vidlore):

  1. python -m vidlore --brief brief.yaml
       -> writes script.txt for you to review/edit, then STOPS
  2. python -m vidlore --brief brief.yaml --resume
       -> renders the reviewed script into the final video

One-shot (no review pause): add --auto.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .brief import Brief
from .config import load_config
from .pipeline import produce, render_from_script, write_script


def _channel_subcmd(argv: list[str]) -> int:
    """`vidlore channel ...` sub-CLI for Look DNA management.

    Supported subcommands:
        list                          # list channels
        presets                       # list bundled presets
        show <name>                   # print a channel's resolved YAML path
        init <name> [--from <preset>] # create a new channel YAML
        history <name> [--last N]     # tail history.jsonl
    """
    from . import look_dna as _ld
    if not argv or argv[0] in ("-h", "--help"):
        print(_channel_subcmd.__doc__)
        return 0
    cmd, *rest = argv
    if cmd == "list":
        chs = _ld.list_channels()
        if not chs:
            print("(no channels yet — try `vidlore channel init <name>`)")
        else:
            for c in chs:
                print(c)
        return 0
    if cmd == "presets":
        for n in _ld.list_presets():
            print(n)
        return 0
    if cmd == "show":
        if not rest:
            print("usage: vidlore channel show <name>", file=sys.stderr)
            return 2
        p = _ld.channel_path(rest[0])
        print(p)
        if p.exists():
            print()
            print(p.read_text())
        else:
            print("(does not exist — `vidlore channel init` to create)")
        return 0
    if cmd == "init":
        if not rest:
            print("usage: vidlore channel init <name> [--from <preset>] "
                  "[--force]", file=sys.stderr)
            return 2
        name = rest[0]
        inherits = "standard"
        force = False
        i = 1
        while i < len(rest):
            if rest[i] == "--from" and i + 1 < len(rest):
                inherits = rest[i + 1]; i += 2
            elif rest[i] == "--force":
                force = True; i += 1
            else:
                print(f"unknown arg: {rest[i]}", file=sys.stderr); return 2
        try:
            p = _ld.init_channel(name, inherits=inherits, force=force)
        except FileExistsError as e:
            print(str(e), file=sys.stderr); return 1
        except FileNotFoundError as e:
            print(f"unknown preset '{inherits}'. Available: "
                  f"{', '.join(_ld.list_presets())}", file=sys.stderr); return 1
        print(f"created {p}")
        return 0
    if cmd == "history":
        if not rest:
            print("usage: vidlore channel history <name> [--last N]",
                  file=sys.stderr); return 2
        name = rest[0]
        n = 10
        if len(rest) >= 3 and rest[1] == "--last":
            n = int(rest[2])
        import json
        hist = _ld.read_recent_history(name, n=n)
        if not hist:
            print("(no history yet)")
            return 0
        for entry in hist:
            print(json.dumps(entry, ensure_ascii=False))
        return 0
    print(f"unknown channel subcommand: {cmd}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    # Detect the `channel` sub-CLI BEFORE argparse so we keep its
    # surface small and don't pollute the main flags help text.
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "channel":
        return _channel_subcmd(args[1:])

    p = argparse.ArgumentParser(
        prog="vidlore",
        description="prompt -> reviewable script -> faceless documentary video",
    )
    p.add_argument("--brief", required=True, help="path to a brief .yaml")
    p.add_argument("--script", help="optional ready-made narration .txt "
                                    "(skips LLM scripting)")
    p.add_argument("--out", default="output", help="output directory")
    p.add_argument("--voice", help="override edge-tts voice")
    p.add_argument("--voiceover", help="use YOUR own narration audio file "
                                       "(mp3/wav) instead of TTS")
    p.add_argument("--no-music", action="store_true",
                   help="disable the auto background music bed")
    p.add_argument("--no-transitions", action="store_true",
                   help="hard cuts instead of scene crossfades")
    p.add_argument("--no-overlays", action="store_true",
                   help="disable the title card + subscribe CTA overlays")
    p.add_argument("--resume", action="store_true",
                   help="render using the reviewed/edited script.txt from a "
                        "previous run")
    p.add_argument("--auto", action="store_true",
                   help="one-shot: generate the script AND render without "
                        "pausing for review")
    p.add_argument("--keep-work", action="store_true",
                   help="keep intermediate files")
    # ── LOOK DNA flags ────────────────────────────────────────────
    p.add_argument("--channel", help="apply this channel's Look DNA "
                                     "(~/.vidlore/channels/<name>.yaml)")
    p.add_argument("--look-preset", help="apply a built-in preset "
                                          "(see `vidlore channel presets`)")
    a = p.parse_args(args)

    cfg = load_config()
    if a.no_music:
        cfg.music_enabled = False
    if a.no_transitions:
        cfg.transitions_enabled = False
    if a.no_overlays:
        cfg.overlays_enabled = False
    brief = Brief.from_yaml(a.brief)
    if a.voice:
        brief.voice = a.voice
    if a.voiceover:
        brief.voiceover = a.voiceover
    # CLI > brief for Look DNA fields
    if a.channel:
        brief.channel = a.channel
    if a.look_preset:
        brief.look_preset = a.look_preset

    print("vidrush-clone — providers:")
    print(cfg.describe())
    print(f"\nBrief: {brief.title!r}  [{brief.fmt} | {brief.duration}min "
          f"| {brief.theme} theme]\n")

    try:
        if a.resume:
            r = render_from_script(brief, cfg, Path(a.out),
                                   keep_work=a.keep_work)
        elif a.auto:
            r = produce(brief, cfg, Path(a.out),
                        script_file=a.script, keep_work=a.keep_work)
        else:
            st = write_script(brief, cfg, Path(a.out),
                              script_file=a.script)
            print("\nSCRIPT READY FOR REVIEW")
            print(f"  edit : {st.script_txt}")
            print(f"  ({len(st.script.scenes)} scenes, "
                  f"~{st.script.word_count} words)")
            print("\nReview/edit the script (line 1 = title; keep ONE blank "
                  "line between scenes — add/remove blank lines to split or "
                  "merge scenes). When happy, render it with:")
            print(f"  python -m vidlore --brief {a.brief} --resume")
            print("\nSkip the review pause next time with --auto.")
            return 0
    except Exception as e:  # noqa: BLE001
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    _vs = getattr(r, "video_seconds", 0.0) or 0.0
    _vlen = (" · video %d:%02d" % (int(_vs // 60), int(_vs % 60))) if _vs else ""
    print("\nDONE in %0.1fs render%s" % (r.seconds, _vlen))
    print(f"  video     : {r.video}")
    print(f"  script    : {r.script_txt}")
    print(f"  captions  : {r.srt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
