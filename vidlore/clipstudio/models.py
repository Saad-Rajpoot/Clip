"""ClipStudio data model.

Plain dataclasses, all JSON-serializable. Large numeric data (CLIP embeddings) is kept
out-of-band as .npy files so the JSON manifests stay human-readable and diff-able — every
intermediate decision is meant to be inspectable by a reviewer.

Nothing here imports the rest of the engine; this is the shared vocabulary the stages speak.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

# Permission assertions a user can attach to a source. `UNVERIFIED` is the default and is
# BLOCKED at ingest — the module never guesses rights. (See DESIGN.md §5.)
PERMISSIONS = ("owner", "licensed", "public_domain", "cc", "fair_use_claim", "unverified")
PERMISSION_BLOCKS = {"unverified"}

SOURCE_OK = "ok"
SOURCE_BLOCKED = "blocked_no_permission"
SOURCE_FAILED = "download_failed"

# Why a selection was routed to manual review.
FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_IDENTITY = "identity_unconfirmed"
FLAG_SPECIFIC_CLAIM = "specific_claim"
FLAG_OVERUSED_SOURCE = "source_overused"
FLAG_NO_CANDIDATE = "no_candidate"
FLAG_VERIFIER_FAILED = "verifier_failed"
FLAG_VERIFIER_UNVERIFIED = "verifier_unverified"   # verifier errored (transient) — not confirmed
FLAG_EXACT_MISSING = "exact_scene_missing"   # an exact_scene beat had NO real footage/source-frame —
#                                              must go to manual review, NEVER silent weak filler


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_str(p: Optional[Path | str]) -> str:
    return "" if p is None else str(p)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@dataclass
class SourceVideo:
    """One ingested source video (a permitted URL or a local file)."""
    id: str                       # stable short id (slug or sha1[:12])
    url: str                      # original URL, or "local:<abspath>" for a local import
    local_path: str = ""          # ingested file under <project>/sources/
    title: str = ""
    duration: float = 0.0         # seconds
    width: int = 0
    height: int = 0
    fps: float = 0.0
    checksum: str = ""            # sha1 of the file (provenance)
    permission: str = "unverified"
    permission_note: str = ""
    status: str = SOURCE_OK
    error: str = ""
    ingested_at: str = field(default_factory=_utcnow)
    extra: dict = field(default_factory=dict)   # platform metadata (uploader, license tag, ...)

    @property
    def is_blocked(self) -> bool:
        return self.status == SOURCE_BLOCKED or self.permission in PERMISSION_BLOCKS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SourceVideo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Shots (one detected shot within a source)
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    source_id: str
    index: int                    # shot number within its source
    start: float                  # in-point within the source (s)
    end: float                    # out-point within the source (s)
    keyframe_path: str = ""       # representative frame extracted for this shot
    transcript: str = ""          # spoken words during this shot (from ASR)
    embed_row: int = -1           # row index into <source_id>.embeds.npy (-1 = none)
    faces: int = 0                # face count in the keyframe (OpenCV)
    face_ids: list[str] = field(default_factory=list)   # matched reference-actor ids (advanced)
    identities: list[dict] = field(default_factory=list)  # Face-ID: [{name,score,confident,area,bbox}]
    ocr_names: list[str] = field(default_factory=list)  # roster names read off the frame (name cards)
    ocr_text: str = ""            # raw OCR text on the keyframe
    tags: list[str] = field(default_factory=list)       # object/scene tags (advanced)
    quality: float = 1.0          # 0..1 shot quality (sharpness / brightness / resolution)
    phash: str = ""               # perceptual hash (near-duplicate detection)
    dup_of: int = -1              # shot index this duplicates within the source (-1 = unique)
    motion: float = 0.0           # mean frame motion (0..1) — static stills score low
    scores: dict = field(default_factory=dict)          # cached relevance sub-scores
    # MULTI-FRAME flags (index-time, 3 samples/shot: start/mid/end) — the keyframe alone misses
    # intermittent burned subs / corner bugs / mid-shot darkness. Sentinels (-1) mark an OLD index
    # where flags were never computed; gates fall back to keyframe heuristics then.
    subs_flag: int = -1           # 1 = subtitle band on ANY sampled frame, 0 = clean, -1 = unknown
    text_conf: float = 0.0        # subtitle-band confidence (max band/mid edge-density ratio)
    luma_avg: float = -1.0        # mean of sample mean-lumas (0..255); -1 = not computed
    luma_hi: float = -1.0         # max over samples of the ~99.8th-percentile pixel — the
                              # "brightest real content" level that separates a readable
                              # candlelit scene (lit face/candle ≥120) from true murk (<90)
    luma_min: float = -1.0        # DARKEST sample's mean luma. luma_avg/luma_hi are shot-wide, so a
                                  # shot that goes black for part of its span but carries a torch
                                  # reads as fine (measured: avg 39.4 / hi 255 on a shot whose
                                  # delivered frames sat at luma 2.5). This is the intra-shot floor.
    luma_min_black_frac: float = -1.0   # share of pixels <16 in that darkest sample — separates a
                                  # low-key lit frame from an actually unusable one
    corner_masks: dict = field(default_factory=dict)    # per-corner hex edge-bitmask (11x29 grid)
    graphics_flag: int = -1       # 2 = HARD designed graphics (news CGI/game UI/cartoon/fan art —
                                  # never airs), 1 = band-art (gated only in a source with hard
                                  # evidence), 0 = photographic, -1 = not computed

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Shot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Script segments (one narration line / beat)
# ---------------------------------------------------------------------------

@dataclass
class ScriptSegment:
    index: int
    text: str                     # the narration line
    expected_visual: str = ""     # short phrase describing what should be on screen
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)   # named people/things mentioned
    required_entity: str = ""     # the SPECIFIC actor/character/object/scene this line demands
    required_kind: str = ""       # actor | character | object | scene | event | location | ""
    scene_query: str = ""         # a PRECISE search string for the EXACT moment (drives discovery)
    quote: str = ""               # iconic line/dialogue spoken IN that moment (locks onto it via ASR)
    shot_intent: str = ""         # establishing|action|emotional_closeup|reaction|symbolic|montage
    emotion: str = ""             # emotional tone (drives pacing/selection)
    is_specific_claim: bool = False     # asserts a precise visual fact → force review
    # UNIFIED VISUAL POLICY (see policy.py) — the single per-beat treatment every stage obeys:
    # exact_scene | character_specific | generic_filler | abstract_effect  ('' = classify lazily)
    visual_policy: str = ""
    breakout_candidate: bool = False    # orthogonal flag: a quoted iconic moment → real-audio breakout
    est_duration: float = 0.0     # estimated screen time before TTS (s)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ScriptSegment":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Candidates + selections
# ---------------------------------------------------------------------------

@dataclass
class ClipCandidate:
    """A scored (segment, shot) pairing. Many produced per segment; one becomes the pick."""
    segment_index: int
    source_id: str
    shot_index: int
    score: float = 0.0            # combined confidence 0..1
    in_point: float = 0.0         # chosen trim within the source (s)
    out_point: float = 0.0
    signals: dict = field(default_factory=dict)   # {clip, transcript, face, object, ...}

    @property
    def duration(self) -> float:
        return max(0.0, self.out_point - self.in_point)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClipCandidate":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ClipSelection:
    """The chosen clip for one segment, plus everything the ledger and review need."""
    segment_index: int
    source_id: str
    shot_index: int
    in_point: float
    out_point: float
    confidence: float
    clip_path: str = ""           # the pre-cut trimmed .mp4 (what the renderer plays)
    signals: dict = field(default_factory=dict)
    identity: str = ""            # Face-ID name on the chosen shot ("" = none/uncertain)
    identity_score: float = 0.0
    verifier: dict = field(default_factory=dict)   # AI-verifier verdict (Phase 6)
    # Multi-clip-per-scene (Phase 9): ordered list of distinct [source_id, in, out] windows so the
    # engine's intra-scene sub-beats each show a DIFFERENT relevant clip instead of one looped clip.
    beat_windows: list = field(default_factory=list)
    reuse_count: int = 0          # how many times this source was already used before this pick
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)
    approved: bool = False        # set by a human in the review surface
    alternates: list[ClipCandidate] = field(default_factory=list)
    # The verifier's DEEP BENCH — ranked candidates beyond `alternates`, read only when the verifier
    # is about to give up on an exact beat and keep a contextual fallback. Measured on job
    # 69d80e9dd4_v4: that give-up path took 113 of 268 beats and they score 4.34 on the frame eval,
    # against 5.92 for beats where a replacement was found. Six alternates out of a ~4000-shot pool
    # was simply too shallow a bench to find the exact moment.
    deep_alternates: list[ClipCandidate] = field(default_factory=list)
    source_url: str = ""          # denormalized for the ledger
    # IMAGE FALLBACK (Phase 11): when no YouTube clip was relevant enough, an exact-scene
    # still fetched + face/CLIP-verified from the open web. build.py renders it as a
    # Ken-Burns motion still. Empty = normal video footage was used.
    image_path: str = ""
    image_meta: dict = field(default_factory=dict)   # {source, score, clip, face, query}

    @property
    def duration(self) -> float:
        return max(0.0, self.out_point - self.in_point)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("alternates", "deep_alternates"):
            d[k] = [a.to_dict() if isinstance(a, ClipCandidate) else a
                    for a in (getattr(self, k) or [])]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ClipSelection":
        d = dict(d)
        for k in ("alternates", "deep_alternates"):
            d[k] = [ClipCandidate.from_dict(a) for a in d.get(k, [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Project (top-level container + on-disk layout)
# ---------------------------------------------------------------------------

@dataclass
class ClipProject:
    name: str
    root: str                     # absolute path to the project folder
    script_path: str = ""
    created_at: str = field(default_factory=_utcnow)
    sources: list[SourceVideo] = field(default_factory=list)
    segments: list[ScriptSegment] = field(default_factory=list)
    selections: list[ClipSelection] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # --- path helpers (single source of truth for the on-disk layout) ---
    @property
    def root_path(self) -> Path:
        return Path(self.root)

    @property
    def sources_dir(self) -> Path:
        return self.root_path / "sources"

    @property
    def index_dir(self) -> Path:
        return self.root_path / "index"

    @property
    def clips_dir(self) -> Path:
        return self.root_path / "clips"

    @property
    def output_dir(self) -> Path:
        return self.root_path / "output"

    @property
    def manifest_path(self) -> Path:
        return self.root_path / "project.json"

    @property
    def ledger_path(self) -> Path:
        return self.root_path / "ledger.jsonl"

    @property
    def review_queue_path(self) -> Path:
        return self.root_path / "review_queue.json"

    def shots_path(self, source_id: str) -> Path:
        return self.index_dir / f"{source_id}.shots.json"

    def embeds_path(self, source_id: str) -> Path:
        return self.index_dir / f"{source_id}.embeds.npy"

    def keyframes_dir(self, source_id: str) -> Path:
        return self.index_dir / source_id / "keyframes"

    def ensure_dirs(self) -> None:
        for d in (self.sources_dir, self.index_dir, self.clips_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)

    def source(self, source_id: str) -> Optional[SourceVideo]:
        return next((s for s in self.sources if s.id == source_id), None)

    # --- serialization ---
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "root": self.root,
            "script_path": self.script_path,
            "created_at": self.created_at,
            "sources": [s.to_dict() for s in self.sources],
            "segments": [s.to_dict() for s in self.segments],
            "selections": [s.to_dict() for s in self.selections],
            "meta": self.meta,
        }

    def save(self) -> Path:
        # atomic tmp+replace — save() runs after EVERY stage; an interrupted write must not
        # corrupt the manifest the whole project resumes from
        self.root_path.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)
        return self.manifest_path

    @classmethod
    def from_dict(cls, d: dict) -> "ClipProject":
        return cls(
            name=d.get("name", ""),
            root=d.get("root", ""),
            script_path=d.get("script_path", ""),
            created_at=d.get("created_at", _utcnow()),
            sources=[SourceVideo.from_dict(x) for x in d.get("sources", [])],
            segments=[ScriptSegment.from_dict(x) for x in d.get("segments", [])],
            selections=[ClipSelection.from_dict(x) for x in d.get("selections", [])],
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, root: Path | str) -> "ClipProject":
        root = Path(root)
        manifest = root / "project.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        proj = cls.from_dict(data)
        proj.root = str(root)        # honor the actual on-disk location
        return proj
