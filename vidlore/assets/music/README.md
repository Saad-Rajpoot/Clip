# Vidlore Music Library

Drop royalty-free music files into the category folder that matches their
mood. The engine (`vidlore/musiclib.py`) scans this folder, reads metadata,
and scores each documentary by selecting + crossfading tracks per story
section. **If a category is empty the engine falls back to a related
category, and if the whole library is empty it uses the procedural bed.**

## Layout
```
vidlore/assets/music/<category>/<track>.mp3      (or .wav .m4a .ogg .flac)
vidlore/assets/music/<category>/<track>.json     (optional metadata override)
```
Categories: `suspense  mystery  dark_investigation  emotional_piano  ambient
historical_epic  military_tension  tech_cyber  financial  survival_urgency
slow_reveal  climax_build  aftermath  neutral  archive_texture`

Filename tip: include `_120bpm` in the name to record BPM automatically.
Optional sidecar `<track>.json`: `{"bpm":120,"tags":["dark","pulse"],
"license":"CC0","source":"Pixabay"}`.

## Recommended FREE / licence-safe sources (no copyright risk)

| Source | License | API / access needed |
|---|---|---|
| **Pixabay Music** (pixabay.com/music) | Pixabay Content License — free, **no attribution**, commercial OK | Manual download (best). Pixabay's REST API is images/video only — music is download-from-site. *(You already have a Pixabay key in `.env`, but it does NOT cover music downloads.)* |
| **YouTube Audio Library** (studio.youtube.com) | Free; some "no attribution", some "attribution required" | Manual download (no API) |
| **Incompetech – Kevin MacLeod** | **CC-BY 4.0** (attribution required) | Manual download; put the credit line in `LICENSES.md` |
| **Free Music Archive** (freemusicarchive.org) | Per-track (CC0 / CC-BY / CC-BY-NC — check each) | Manual download; API exists but is limited |
| **Mixkit** (mixkit.co/free-stock-music) | Mixkit Free License — free, no attribution | Manual download |
| **Uppbeat** (uppbeat.io) | Free tier needs credit; paid removes it | Account; manual download |
| **Internet Archive** (archive.org) | Public-domain / CC where marked — **verify each** | Manual download |

**My recommendation:** start with **Pixabay Music** + **Mixkit** (both
free, commercial, *no attribution*) so there's zero licensing burden. Add
**Incompetech** for cinematic cues (just keep the CC-BY credit). ~15-20
tracks per category gets you to ~250 tracks.

**No new API key is required** — this is a manual-curation folder by design
(music APIs for these sources are absent/limited, and manual curation lets
you keep quality high). Just drop files in the right folders and the engine
ingests them on the next render.

## Licensing log
Record every track's source + license in `LICENSES.md` (create it here).
The engine reads optional per-track `license`/`source` metadata too.

## Automated music pipeline — `music_extract`

Four-stage system, each stage building on the last. **One command** at the
top of any session keeps the library growing on its own:

```bash
python -m vidlore.music_extract --auto --reindex
```

### Stage 1 — safe automated crawling
`--auto` walks every trusted royalty-free source registered under
`vidlore/music_sources/` in round-robin so no one source dominates:

| Source | License model | Module |
|---|---|---|
| **YouTube Audio Library** (community @audiolibrary mirrors) | YT AL Free To Use (attribution captured when required) | `youtube_al.py` |
| **Incompetech** (Kevin MacLeod) | CC BY 4.0 — attribution required, auto-written | `incompetech.py` |
| **Free Music Archive** (filtered to CC0 / CC BY / CC BY-SA) | per-track, NC + ND skipped | `fma.py` |
| **Mixkit** (free section) | Mixkit Free License, commercial OK, no attribution | `mixkit.py` |
| **Pixabay Music** | Pixabay Content License, commercial OK, no attribution | `pixabay.py` |

A candidate with no resolvable license is **dropped before download**.
Each landed track gets a sidecar `<track>.json` with `title / channel /
source_url / license / attribution / classify.{features,confidence}` and
its license line is appended to `LICENSES.md`.

### Stage 2 — smart auto-classification
After download, every track passes through `vidlore/music_classify.py`:

1. **Title keywords** — first guess (free, instant).
2. **Source hint** — pages like Incompetech `?feel=Dark` carry a confirmed bucket.
3. **Audio features** — ffmpeg `volumedetect` + `silencedetect` + a numpy FFT pass on a centered 45 s slice extracts `mean / max dB`, silence ratio, spectral centroid, spectral flatness, and an **energy arc** (rising / falling / arc / steady). Heuristic flags (`is_sparse`, `is_bright`, `is_dynamic`) tie-break the routing.

The final category, confidence, and which passes voted for it are written
into the sidecar (`classify.voted_by`).

### Stage 3 — editor-quality selection
`musiclib.select()` now scores each candidate track against the cue:

- Energy fit (high-energy cues prefer dynamic tracks, calm cues prefer sparse).
- Arc fit (swell cues prefer rising arcs; aftermath cues prefer falling).
- Brightness fit (dark categories prefer non-bright tracks).
- Duration fit (penalises tracks that would need >2× loop).
- Cross-render usage decay (Stage 4 — least-used tracks get +0.15, frequent reuse pays a -0.04 / play penalty up to -0.20).

The existing per-render `_used` 6-track cooldown still applies as a hard
filter on top of the score.

### Stage 4 — large-library upkeep
The orchestrator persists:

- `_history.json` — fingerprint set (sha256 of first 256 KB), seen URLs,
  yt_ids, and per-source last-run dates. Tracks deduped across sources
  even when re-encoded slightly differently in name.
- `_usage.json` — per-track play counter maintained by `compose_score()`.
  Feeds Stage 3 scoring so the same 10 tracks don't dominate every video.

Re-runs are idempotent: known URLs / yt_ids / fingerprints are skipped, so
`--auto` can be cron'd weekly and the library grows fresh tracks without
duplicates.

### Useful flags
```bash
python -m vidlore.music_extract --auto --max 5 --per-cat-cap 6 --reindex
python -m vidlore.music_extract --auto --only incompetech,fma --dry-run
python -m vidlore.music_extract --auto --category suspense --max 4
python -m vidlore.music_extract --search "no copyright dark documentary" \
                                --category dark_investigation --max 5
python -m vidlore.music_extract --no-audio-classify   # skip Stage 2 audio
```

- `--max N`           candidate cap per source (default 8)
- `--per-cat-cap N`   max NEW tracks per category per run (default 8)
- `--min-sec / --max-sec`  duration window (defaults 45 / 420)
- `--dry-run`         list everything, fetch nothing
- `--reindex`         refresh `musiclib._index.json` on exit

The legacy `_sources.yaml` manual mode still works (no `--auto`); useful
for adding your own curated playlist URLs alongside the automatic crawl.
