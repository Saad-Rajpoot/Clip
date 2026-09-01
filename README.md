# vidrush-clone

A functional clone of the **Vidlore AI** pipeline: one brief → a fully
produced, faceless, narrated documentary video (script, voiceover,
B-roll, themed grade, burned captions, thumbnail) — produced locally.

This is the **core pipeline** (Vidlore's "generate" button), not the web
dashboard/editor. Same architecture, runs on your machine.

## How it maps to Vidlore

| Vidlore step              | Here                                              |
|---------------------------|---------------------------------------------------|
| Format / length / mode    | `brief.yaml` (`fmt`, `duration`, 4-Pillars prompt)|
| Script (LLM, TTS-ready)   | `script_gen.py` — Anthropic, or your own `.txt`   |
| Voice (ElevenLabs)        | `tts.py` — ElevenLabs on key, else free edge-tts  |
| Theme grade               | `themes.py` — Crime/History/Modern/Minimalist/Std |
| Footage Agent (B-roll)    | `footage.py` — Pexels → AI image → slide ladder   |
| Music licensing           | `music.py` — synthesized theme bed, auto-ducked    |
| Captions (off by default) | `captions.py` — on by default, ASS burned in      |
| Thumbnail (AI)            | `thumbnail.py` — title/theme composite            |
| Transitions               | `assemble.py` — scene crossfades (xfade)          |
| Title card + CTA          | `assemble.py` — fade-in title, SUBSCRIBE lower-3rd|
| Chapter strips            | `assemble.py` — brief per-scene chapter label     |
| Visual Director           | `script_gen.py` — LLM writes a precise cinematic shot per sentence (with a key); narration-derived AI image otherwise |
| Footage Agent (relevance) | `footage.py` — AI image generated from that shot/narration (keyless, on-topic) |
| Kinetic captions          | `captions.py` — word highlight + LLM-chosen emphasis word punched on beat |
| Sound design + film look  | `music.py`/`assemble.py` — ambient bed + *motivated* boom/riser on intense beats + vignette |
| Motion graphics           | `assemble.py` — energy-scaled Ken Burns + slide-in chapters |
| Cloud render ~50-60 min   | `assemble.py` — local ffmpeg, minutes             |

## Hybrid keys (free out of the box, upgrade when ready)

Copy `.env.example` → `.env`:

- *(nothing)* → bundled sample script + free edge-tts voice + a free
  **Pollinations AI image generated from each scene's narration** (so
  the picture always matches what's being said) + ambient music bed.
  **$0, no keys, works immediately.**
- `ANTHROPIC_API_KEY` → scripts written automatically; also yields much
  better per-scene visual prompts.
- `PEXELS_API_KEY` (free at pexels.com/api) → real stock B-roll *video*
  preferred when it matches; AI image is the always-on-topic default.
- `ELEVENLABS_API_KEY` → narration uses ElevenLabs (Vidlore's real voice
  engine); a per-scene failure auto-falls back to free edge-tts.
- Music: a theme-matched ambient bed is generated and ducked under the
  narration automatically. `--no-music` (or `VIDLORE_MUSIC=0`) turns it
  off; a `music:` path in the brief overrides it with your own track.

## Setup

```bash
cd vidrush-clone
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# (ffmpeg is bundled via imageio-ffmpeg — no system install needed)
```

## Web wizard (browser)

The CLI flow, in a browser: brief → **review/edit the script** → render
with a live progress bar → preview + download.

```bash
python -m vidlore.web        # open http://127.0.0.1:5000
```

No keys needed — click **Load sample** to try it instantly. Same
pipeline, providers, cache, and differentiators as the CLI.

Opens on a **My Videos** dashboard: every finished render
(`output/<slug>/`) persists on disk and is listed there with a
thumbnail (survives restarts — no database); in-progress renders show
a live status. `+ New video` starts the wizard.

Open any finished video to see its **scene list** with a **↻ Regenerate**
button per scene: don't like one scene's visual? Regenerate just that
one — its variant counter bumps so a *different* clip/image is chosen,
while every other scene and all narration are restored from cache, so
it re-renders in seconds. (Vidlore can't do this — one change forces a
full ~50–60 min re-render.) Each scene also has a **🔊 Re-voice**
button — a fresh narration take for just that scene (fixes Vidlore's
~10% audio-glitch weakness, where you otherwise can't regenerate a
single voice segment). Visual and voice counters are independent. Each
video also has a **Delete** button (with confirm) that removes its
`output/<slug>/` folder.

The detail page also has an **✎ Edit script & re-render** panel: tweak
the wording or scene breaks of an already-finished video and re-render
— the hash-diff + per-scene cache mean only the scenes you actually
changed are recomputed. The script-review differentiator works both at
creation time and on any finished video, forever.

## Run — CLI (two stages — review the script before the long render)

Vidlore's #1 user complaint is that it renders for ~50–60 min with **no
script preview**. Here you review/edit the script first.

**Stage 1 — generate the script, then stop:**

```bash
# zero keys (bundled sample script):
python -m vidlore --brief examples/sample_brief.yaml \
                  --script examples/sample_script.txt
# or auto-written (needs ANTHROPIC_API_KEY):
python -m vidlore --brief examples/sample_brief.yaml
```

This writes `output/<title-slug>/script.txt` and pauses. Open it and
edit freely: line 1 is the title, then **one blank line between
scenes** — add or remove blank lines to split/merge scenes.

**Stage 2 — render the reviewed script:**

```bash
python -m vidlore --brief examples/sample_brief.yaml --resume
```

If you didn't touch `script.txt`, the original LLM scene split +
keywords are reused; if you edited it, scenes/keywords are re-derived
from your text.

**Incremental re-render:** rendering keeps a per-scene cache in
`output/<slug>/cache/`. Re-run `--resume` after editing and only the
scenes you actually changed are re-narrated / re-sourced — every
unchanged scene is restored from cache (the log shows `reused/total`).
Editing one line no longer means waiting for the whole video again
(Vidlore's #1 limitation). Delete the `cache/` folder to force a clean
rebuild.

**One-shot (skip the review pause):**

```bash
python -m vidlore --brief examples/sample_brief.yaml --auto
```

Outputs land in `output/<title-slug>/`: the `.mp4`, `thumbnail.jpg`,
`script.txt`, `script.json`, and a `.srt`.

## Brief format

See `examples/sample_brief.yaml`. Key fields: `title`, `fmt`
(`documentary` | `top10`), `duration` (`6-8` | `10-12` | `18-20` |
`30-40`), `theme`, `captions`, optional `music` (local file path), and
the 4-Pillars `prompt`.

## Roadmap (next, if you want)

- Web dashboard + 9-step wizard + queue + timeline editor (Vidlore UI)
- ElevenLabs voice provider slot, AI image/video B-roll for "Mini" mode
- Built-in music library + auto ducking, Top-10 listicle templating
