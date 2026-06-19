"""Smart music classification (Stage 2 of the auto-music pipeline).

Three layered passes, each cheaper than the next:

    1. **Title keywords**          (free, instant)            -> bucket guess
    2. **Tag / source hints**      (free, from Candidate)     -> bucket override
    3. **Audio feature analysis**  (ffmpeg + numpy on PCM)    -> tie-breaker

The audio pass is optional: if ffmpeg can't decode a sample, or numpy isn't
present, we fall back to whatever the title+hint pass produced. Nothing in
the pipeline depends on the audio pass succeeding.

The classifier returns a single :class:`Classification` -- a chosen
``category`` plus the underlying features so downstream selection (Stage 3)
can score per-track within a category (BPM-match, energy-match, etc.).
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path

from .ffmpeg_tool import ffmpeg_exe
from .musiclib import CATEGORIES

# --------------------------------------------------------------------------- #
# 1.  Title-keyword classifier  (also used standalone by music_extract.py)
# --------------------------------------------------------------------------- #
_CLASSIFY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dark_investigation", (
        "noir", "investigation", "detective", "crime", "covert",
        "espionage", "spy", "interrogation", "evidence", "stakeout",
        "intrigue", "underworld",
    )),
    ("suspense", (
        "suspense", "tension", "thriller", "tense", "stalking",
        "uneasy", "unsettling", "creeping", "ominous", "lurking",
        "edge", "anxiety",
    )),
    ("mystery", (
        "mystery", "mysterious", "enigma", "unknown", "hidden",
        "secret", "puzzle", "whisper", "shadow",
    )),
    ("military_tension", (
        "military", " war ", "battle", " march", "drum corps",
        "combat", "operation", "soldier", "tactical", "battlefield",
        "warfare",
    )),
    ("historical_epic", (
        "epic", "cinematic", "orchestral", "trailer", "heroic",
        "majestic", "medieval", "ancient", "saga", "legendary",
        "monumental",
    )),
    ("tech_cyber", (
        "cyber", " tech", "synth", "digital", "hacker", "code",
        "futuristic", "techno", "electronic", "cyberpunk", "glitch",
        "synthwave",
    )),
    ("financial", (
        "corporate", "business", "finance", "money", "market",
        "wall street", "boardroom",
    )),
    ("survival_urgency", (
        "survival", "urgent", "urgency", "chase", "escape",
        " race ", "disaster", "alarm", "panic",
    )),
    ("emotional_piano", (
        "piano", "emotional", " sad ", "melancholy", "reflective",
        "lonely", "tearful", "heartfelt", "tender", "intimate",
    )),
    ("slow_reveal", (
        "reveal", "discovery", " rising", "swell", "build slow",
        "unveil", "awakening", "dawn",
    )),
    ("climax_build", (
        "climax", "build up", "buildup", "rising action",
        "crescendo", "peak", "powerful", "anthem", "triumphant",
    )),
    ("aftermath", (
        "aftermath", "ending", "resolution", "calm after",
        "peaceful", "closing", "outro", "denouement", "release",
    )),
    ("archive_texture", (
        "archive", "vinyl", "tape", " static", "vintage",
        "lo-fi crackle", "radio static", "shortwave",
    )),
    ("ambient", (
        "ambient", "drone", "atmosphere", "atmospheric",
        "soundscape", "pad ", "underscore",
    )),
)


def classify_title(title: str, fallback: str = "neutral") -> str:
    t = " " + title.lower() + " "
    for cat, words in _CLASSIFY:
        if any(w in t for w in words):
            return cat
    return fallback


# --------------------------------------------------------------------------- #
# 2.  Audio feature analysis (ffmpeg + numpy; degrades cleanly)
# --------------------------------------------------------------------------- #
@dataclass
class AudioFeatures:
    """Cheap audio-descriptor bundle (all optional / may be 0.0)."""
    duration: float = 0.0
    mean_volume_db: float = 0.0          # ffmpeg volumedetect: mean_volume
    max_volume_db: float = 0.0           # ffmpeg volumedetect: max_volume
    silence_ratio: float = 0.0           # 0..1
    spectral_centroid_hz: float = 0.0    # numpy FFT on 30s slice
    spectral_flatness: float = 0.0       # noisiness: 1.0 = white noise
    energy_arc: str = "steady"           # rising / steady / falling / arc
    bpm: int = 0                         # autocorrelation on onset envelope
    # Heuristic deductions
    is_sparse: bool = False              # silence_ratio > 0.25
    is_bright: bool = False              # centroid > 3.5 kHz
    is_dynamic: bool = False             # peak - mean > 12 dB
    # Composite editorial scores (0..1) -- "how X does this track FEEL"
    tension: float = 0.0                 # high = oppressive / stalking
    darkness: float = 0.0                # low spectrum + low brightness
    orchestral_density: float = 0.0      # busy/full ensemble vs sparse pad
    rhythmic_aggression: float = 0.0     # strong tempo + dynamic + percussive


def _ffmpeg_volumedetect(path: Path) -> tuple[float, float]:
    """(mean_db, max_db). 0.0/0.0 on failure."""
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-nostats",
             "-i", str(path), "-vn",
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=45)
        s = r.stderr
        mean = re.search(r"mean_volume:\s*(-?\d+\.?\d*)", s)
        peak = re.search(r"max_volume:\s*(-?\d+\.?\d*)", s)
        return (float(mean.group(1)) if mean else 0.0,
                float(peak.group(1)) if peak else 0.0)
    except Exception:                                       # noqa: BLE001
        return 0.0, 0.0


def _ffmpeg_silence_ratio(path: Path, dur: float) -> float:
    """Fraction of the track below -30 dB for >= 0.5 s."""
    if dur <= 0:
        return 0.0
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-nostats",
             "-i", str(path), "-vn",
             "-af", "silencedetect=noise=-30dB:d=0.5",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        s = r.stderr
        # silence_duration: X
        total = 0.0
        for m in re.finditer(r"silence_duration:\s*(\d+\.?\d*)", s):
            total += float(m.group(1))
        return max(0.0, min(1.0, total / dur))
    except Exception:                                       # noqa: BLE001
        return 0.0


def _ffmpeg_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
            capture_output=True, text=True, timeout=20)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr)
        if m:
            return (int(m.group(1)) * 3600
                    + int(m.group(2)) * 60
                    + float(m.group(3)))
    except Exception:                                       # noqa: BLE001
        pass
    return 0.0


def _ffmpeg_pcm_slice(path: Path, start: float, length: float,
                      sr: int = 22050) -> bytes:
    """Mono float32 PCM bytes of [start, start+length] of ``path``."""
    try:
        r = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-nostats", "-loglevel", "error",
             "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
             "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
             "-f", "f32le", "-"],
            capture_output=True, timeout=60)
        return r.stdout or b""
    except Exception:                                       # noqa: BLE001
        return b""


def _spectral(features_path: Path, dur: float) -> tuple[float, float, str]:
    """Return (centroid_hz, flatness, energy_arc).
    Energy arc compares mean RMS over thirds of the slice."""
    try:
        import numpy as np
    except Exception:                                       # noqa: BLE001
        return 0.0, 0.0, "steady"
    if dur < 10.0:
        return 0.0, 0.0, "steady"
    sr = 22050
    slice_len = min(45.0, max(15.0, dur * 0.6))
    start = min(max(0.0, (dur - slice_len) / 2.0), max(0.0, dur - slice_len))
    raw = _ffmpeg_pcm_slice(features_path, start, slice_len, sr=sr)
    if not raw:
        return 0.0, 0.0, "steady"
    try:
        x = np.frombuffer(raw, dtype=np.float32)
    except Exception:                                       # noqa: BLE001
        return 0.0, 0.0, "steady"
    if x.size < sr:
        return 0.0, 0.0, "steady"
    # FFT over a few centered seconds for spectral descriptors
    n = 1 << 15                                             # 32768 samples
    mid = x.size // 2
    seg = x[max(0, mid - n // 2): mid + n // 2]
    if seg.size < n:
        return 0.0, 0.0, "steady"
    seg = seg - float(seg.mean())
    win = np.hanning(seg.size).astype(np.float32)
    spec = np.abs(np.fft.rfft(seg * win)) + 1e-9
    freqs = np.fft.rfftfreq(seg.size, 1.0 / sr)
    centroid = float((freqs * spec).sum() / spec.sum())
    # spectral flatness = geo mean / arith mean
    log_spec = np.log(spec)
    flatness = float(np.exp(log_spec.mean()) / (spec.mean()))
    # energy arc over thirds
    third = x.size // 3
    rms = lambda v: float(np.sqrt(np.mean(v.astype(np.float64) ** 2)) + 1e-9)
    a, b, c = rms(x[:third]), rms(x[third:2 * third]), rms(x[2 * third:])
    if c > a * 1.25 and c > b:
        arc = "rising"
    elif a > c * 1.25 and a > b:
        arc = "falling"
    elif b > a * 1.15 and b > c * 1.15:
        arc = "arc"
    else:
        arc = "steady"
    return centroid, flatness, arc


def _bpm_from_pcm(raw: bytes, sr: int = 22050) -> int:
    """Estimate BPM via short-time energy onset autocorrelation.

    Resilient enough for steady-tempo orchestral / electronic / percussive
    tracks (60..180 BPM range). Returns 0 on failure or very ambient music
    that has no clear pulse -- which is informative on its own."""
    try:
        import numpy as np
    except Exception:                                       # noqa: BLE001
        return 0
    if not raw or len(raw) < sr * 4 * 4:        # need ~4s of float32
        return 0
    try:
        x = np.frombuffer(raw, dtype=np.float32)
    except Exception:                                       # noqa: BLE001
        return 0
    hop, win = 512, 1024
    n = (x.size - win) // hop
    if n < 100:
        return 0
    # short-time energy per frame
    energy = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = x[i * hop: i * hop + win].astype(np.float64)
        energy[i] = float(np.dot(s, s))
    # half-wave-rectified energy diff = onset strength
    onset = np.diff(energy)
    onset = np.maximum(onset, 0.0)
    if onset.std() < 1e-9:
        return 0
    onset = (onset - onset.mean()) / (onset.std() + 1e-9)
    # autocorrelation (positive lags)
    ac = np.correlate(onset, onset, mode="full")
    ac = ac[len(ac) // 2:]
    fps = sr / hop                              # frames per second
    min_lag = int(round(fps * 60.0 / 180.0))    # 180 BPM cap
    max_lag = int(round(fps * 60.0 / 60.0))     # 60 BPM floor
    if max_lag >= ac.size or min_lag <= 0:
        return 0
    window = ac[min_lag:max_lag + 1]
    if window.size == 0:
        return 0
    peak_lag = min_lag + int(np.argmax(window))
    if peak_lag <= 0:
        return 0
    bpm = 60.0 * fps / peak_lag
    return int(round(bpm))


def _composite_scores(f: AudioFeatures) -> dict:
    """Derive editorial composite scores from the cheap descriptors.
    All outputs are 0..1.  Calibrated against the real spread of
    documentary-music features (centroid 500-2500 Hz, flatness 0.004-0.40,
    silence_ratio 0.0-0.30, dyn 12-25 dB, BPM 60-160) so the four scores
    actually DISCRIMINATE -- ambient/aftermath tracks land low on
    density/aggression, climax/military tracks land high."""
    # Brightness: where the spectrum sits.  600 Hz = sub-bass-dominant
    # (dark), 2400 Hz = percussive/bright.
    bright = max(0.0, min(1.0,
                          (f.spectral_centroid_hz - 600.0) / 1800.0))
    # Sparseness: 0 = wall-of-sound, 1 = many breath moments.  Saturates at
    # 30% silence (anything more is basically a sparse atmospheric).
    sparse = max(0.0, min(1.0, f.silence_ratio / 0.30))
    # Noisiness: high spectral flatness = noise / drums / drone; low =
    # pitched (piano / strings / brass).
    noisy = max(0.0, min(1.0, f.spectral_flatness / 0.40))
    # Dynamics: peak vs mean dB delta.  12 dB = compressed pad, 25 dB =
    # full orchestral swell.
    dyn = max(0.0, min(1.0, (f.max_volume_db - f.mean_volume_db - 12.0)
                       / 12.0))
    # Fast/slow tempo on the 60-140 BPM useful band
    fast = (max(0.0, min(1.0, (f.bpm - 60.0) / 80.0))
            if 60 <= f.bpm <= 175 else 0.0)

    # Darkness: low spectrum dominated.  Pure spectral-centroid signal.
    darkness = (0.70 * (1.0 - bright)
                + 0.20 * (1.0 - sparse)
                + 0.10 * (1.0 - noisy))

    # Tension: dark + busy + dynamic
    tension = (0.40 * (1.0 - bright)
               + 0.25 * (1.0 - sparse)
               + 0.20 * dyn
               + 0.15 * (1.0 - noisy))

    # Orchestral density: BUSY (non-silent) + TONAL (pitched, not noise).
    # Solo piano = low density. Wall-of-orchestra pad = high.  Drum solo
    # is NOT orchestral density (low here, high in rhythmic_aggression).
    orchestral_density = (0.55 * (1.0 - sparse)
                          + 0.45 * (1.0 - noisy))

    # Rhythmic aggression: PERCUSSIVE + FAST + DYNAMIC.  Sparse drums =
    # high; pad = low.
    rhythmic_aggression = (0.45 * noisy + 0.35 * fast + 0.20 * dyn)

    return {
        "tension": round(tension, 3),
        "darkness": round(darkness, 3),
        "orchestral_density": round(orchestral_density, 3),
        "rhythmic_aggression": round(rhythmic_aggression, 3),
    }


def extract_features(path: Path) -> AudioFeatures:
    """Best-effort audio descriptor. Never raises."""
    feat = AudioFeatures()
    try:
        feat.duration = _ffmpeg_duration(path)
        feat.mean_volume_db, feat.max_volume_db = _ffmpeg_volumedetect(path)
        feat.silence_ratio = _ffmpeg_silence_ratio(path, feat.duration)
        feat.spectral_centroid_hz, feat.spectral_flatness, feat.energy_arc \
            = _spectral(path, feat.duration)
        feat.is_sparse = feat.silence_ratio > 0.25
        feat.is_bright = feat.spectral_centroid_hz > 3500.0
        feat.is_dynamic = ((feat.max_volume_db - feat.mean_volume_db) > 12.0)
        # BPM via autocorrelation on a 30s centered slice
        if feat.duration >= 12.0:
            sr = 22050
            slice_len = min(30.0, max(8.0, feat.duration * 0.6))
            start = max(0.0, (feat.duration - slice_len) / 2.0)
            raw = _ffmpeg_pcm_slice(path, start, slice_len, sr=sr)
            feat.bpm = _bpm_from_pcm(raw, sr=sr)
        # Composite editorial scores
        comp = _composite_scores(feat)
        feat.tension = comp["tension"]
        feat.darkness = comp["darkness"]
        feat.orchestral_density = comp["orchestral_density"]
        feat.rhythmic_aggression = comp["rhythmic_aggression"]
    except Exception:                                       # noqa: BLE001
        pass
    return feat


# --------------------------------------------------------------------------- #
# 3.  Combined classification  (title + hint + features)
# --------------------------------------------------------------------------- #
@dataclass
class Classification:
    category: str
    confidence: float                     # 0..1
    sources: list[str] = field(default_factory=list)   # which pass voted
    features: AudioFeatures = field(default_factory=AudioFeatures)

    def to_meta(self) -> dict:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 2),
            "voted_by": self.sources,
            "features": asdict(self.features),
        }


def _audio_to_category(f: AudioFeatures, lean: str) -> tuple[str, float]:
    """Map audio features -> category + confidence.  ``lean`` is the
    category we'd choose absent any audio signal (title/hint); it nudges
    ties so audio refinement doesn't fight the metadata."""
    # Sparse + dark spectrum + sustained = ambient-family
    if f.is_sparse and not f.is_bright:
        if lean in ("dark_investigation", "mystery", "suspense"):
            return lean, 0.7
        return "ambient", 0.65
    # Rising arc + dynamic = climax_build
    if f.energy_arc == "rising" and f.is_dynamic:
        return "climax_build", 0.7
    # Falling arc + sparse = aftermath
    if f.energy_arc == "falling" and f.is_sparse:
        return "aftermath", 0.65
    # Very dark + sustained = dark_investigation
    if (not f.is_bright) and (not f.is_dynamic) and f.energy_arc == "steady":
        if lean in ("dark_investigation", "mystery"):
            return lean, 0.7
        return "ambient", 0.55
    # Very bright + dynamic = either tech_cyber or historical_epic
    if f.is_bright and f.is_dynamic:
        return ("historical_epic" if lean in ("historical_epic",
                                              "climax_build", "neutral")
                else lean), 0.6
    return lean or "neutral", 0.5


def classify(*, title: str, hint: str = "auto", tags: list[str] | None = None,
             audio: Path | None = None) -> Classification:
    """One-shot classifier used by the orchestrator post-download."""
    sources: list[str] = []
    chosen = ""
    conf = 0.0

    # Pass 1: title keywords
    by_title = classify_title(title)
    if by_title != "neutral":
        chosen, conf = by_title, 0.55
        sources.append("title")

    # Pass 2: explicit hint from the source module (highest authority for
    # cases where the source page is a specific feel/genre filter)
    if hint and hint != "auto" and hint in CATEGORIES:
        chosen, conf = hint, max(conf, 0.75)
        sources.append("source_hint")

    # Pass 2b: tags (cheap)
    for tag in tags or []:
        t = tag.lower()
        for cat, words in _CLASSIFY:
            if any(w.strip() in t for w in words):
                if not chosen:
                    chosen, conf = cat, 0.5
                    sources.append("tags")
                break

    # Pass 3: audio features (optional refinement)
    feats = AudioFeatures()
    if audio and audio.exists():
        feats = extract_features(audio)
        audio_cat, audio_conf = _audio_to_category(
            feats, chosen or "neutral")
        # if we had no opinion yet, take audio outright
        if not chosen:
            chosen, conf = audio_cat, audio_conf
            sources.append("audio")
        # if audio strongly disagrees AND title was weak, switch
        elif (audio_cat != chosen and conf < 0.7
              and audio_conf >= 0.65):
            chosen, conf = audio_cat, audio_conf
            sources.append("audio_override")
        else:
            sources.append("audio_agree")
            conf = min(1.0, conf + 0.1)

    if not chosen or chosen not in CATEGORIES:
        chosen = "neutral"
        conf = max(conf, 0.3)
    return Classification(category=chosen, confidence=conf,
                          sources=sources, features=feats)
