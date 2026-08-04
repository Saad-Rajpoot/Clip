"""Focused regressions for the beat-56 absence/ship failure.

These tests deliberately isolate prompt construction and native-still evidence binding.  They do
not exercise a provider, encode media, or depend on the large real-job fixture.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

from vidlore.clipstudio import build as B
from vidlore.clipstudio import llm as L
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.models import ScriptSegment


def _keep_reply() -> str:
    return (
        '{"matches_narration":true,"correct_subject_visible":true,'
        '"wrong_subject_visible":false,"contradicts_narration":false,'
        '"era_ok":true,"specific_enough":true,"quality_ok":true,'
        '"confidence":0.9,"verdict":"keep","reason":"correct scene elsewhere"}'
    )


def _absence_question(**overrides) -> dict:
    question = {
        "narration": "and Baelish is not even in the room.",
        "required_entity": "Petyr Baelish",
        "required_kind": "character",
        "faceid_names": ["Aidan Gillen"],
        "eng_cfg": NS(),
        "model": "vision-test",
        "is_specific": True,
        "expected_visual": (
            "Petyr Baelish aboard a ship, looking toward King's Landing, not at the wedding"
        ),
        "scene_query": "Game of Thrones S4E4 Petyr Baelish ship after Purple Wedding",
        "era_hint": "season 4",
        "multiframe": True,
    }
    question.update(overrides)
    return question


def test_absence_elsewhere_prompt_replaces_unconditional_presence_contradiction(
        monkeypatch, tmp_path):
    frame = tmp_path / "sheet.jpg"
    frame.write_bytes(b"three-frame contact sheet")
    prompts: list[str] = []

    def fake_complete_ex(**kwargs):
        prompts.append(kwargs["messages"][0]["content"][1]["text"])
        return _keep_reply(), {"served": "vision-test"}

    monkeypatch.setattr(L, "complete_ex", fake_complete_ex)

    V.verify_frame(frame, **_absence_question())
    # Same narration, but the storyboard names the excluded room rather than a distinct location.
    # This must retain the ordinary direct-contradiction rule.
    V.verify_frame(
        frame,
        **_absence_question(
            expected_visual="Petyr Baelish standing in the council room.",
            scene_query="Game of Thrones Petyr Baelish council room",
        ),
    )

    assert len(prompts) == 2
    elsewhere, ordinary = prompts
    assert "ABSENCE-ELSEWHERE CONTEXT" in elsewhere
    assert "narrow ABSENCE-ELSEWHERE exception" in elsewhere
    assert "Set false when the named subject is visibly at the storyboard's distinct location" \
        in elsewhere
    assert "The exact storyboard scene/location is still required" in elsewhere
    assert "the line says a named person is absent but that person is visibly present" \
        not in elsewhere

    assert "ABSENCE-ELSEWHERE CONTEXT" not in ordinary
    assert "narrow ABSENCE-ELSEWHERE exception" not in ordinary
    assert "the line says a named person is absent but that person is visibly present" \
        in ordinary


def test_absence_elsewhere_v2_fingerprint_cold_misses_v1_only(monkeypatch):
    """The corrected conditional prompt cannot reuse the real v1 answer to another question."""
    real_sha256 = hashlib.sha256
    chunks: list[bytes] = []

    class RecordingHash:
        def __init__(self, *args, **kwargs):
            self.inner = real_sha256(*args, **kwargs)

        def update(self, value):
            chunks.append(value)
            return self.inner.update(value)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(hashlib, "sha256", RecordingHash)
    base = {
        "src_hash": "source-content",
        "source_id": "ship-scene",
        "shot_start": 25.32,
        "shot_end": 29.0,
        "beat_text": "and Baelish is not even in the room.",
        "required_entity": "Petyr Baelish",
        "required_kind": "character",
        "expected_visual": "Petyr Baelish aboard a ship after the Purple Wedding",
        "scene_query": "Game of Thrones S4E4 Petyr Baelish ship",
        "era": "season 4",
        "visual_policy": "exact_scene",
        "is_specific": True,
        "faceid_names": ["Aidan Gillen"],
        "multiframe": True,
        "image_id": "sheet:source-content:25.320-29.000",
        "model": "vision-test",
    }

    current = V.verdict_fingerprint(**base)
    assert b"absence-elsewhere-v2" in chunks
    assert b"absence-elsewhere-v1" not in chunks

    legacy = real_sha256()
    for chunk in chunks:
        legacy.update(
            b"absence-elsewhere-v1"
            if chunk == b"absence-elsewhere-v2"
            else chunk
        )
    assert current != legacy.hexdigest()[:32]

    chunks.clear()
    V.verdict_fingerprint(
        **{
            **base,
            "beat_text": "Baelish is not the killer.",
        }
    )
    assert b"absence-elsewhere-v2" not in chunks
    assert b"absence-elsewhere-v1" not in chunks


def test_native_still_verdict_and_fingerprint_both_use_beat_local_era(
        monkeypatch, tmp_path):
    """An unverified global S1 hint must not contaminate a multi-scene beat authored as S4E4."""
    source_path = tmp_path / "ship.mp4"
    source_path.write_bytes(b"native source bytes")
    image_path = tmp_path / "ship-frame.jpg"
    image_path.write_bytes(b"native frame bytes")

    proj = NS(
        meta={
            "analysis": {
                "video_type": "multi_scene",
                "episode_hint": "S01E04",
                "episode_hint_verified": False,
            }
        }
    )
    seg = ScriptSegment(
        index=56,
        text="and Baelish is not even in the room.",
        expected_visual="Petyr Baelish aboard a ship after the Purple Wedding",
        required_entity="Petyr Baelish",
        required_kind="character",
        scene_query="Game of Thrones S4E4 Petyr Baelish ship after Purple Wedding",
        is_specific_claim=True,
        visual_policy="exact_scene",
    )
    owner = {
        "source_id": "ship-scene",
        "shot_index": 5,
        "start": 23.857,
        "end": 30.464,
        "source_path": str(source_path),
    }
    source_fingerprint = V._file_fingerprint(source_path)
    seen: dict[str, str] = {}

    def fake_verify_frame(*_args, **kwargs):
        seen["verifier_era"] = kwargs["era_hint"]
        return {
            "status": "ok",
            "matches_narration": True,
            "correct_subject_visible": True,
            "wrong_subject_visible": False,
            "contradicts_narration": False,
            "era_ok": True,
            "specific_enough": True,
            "quality_ok": True,
            "confidence": 0.9,
            "verdict": "keep",
            "reason": "correct ship scene",
            "vision_served_by": "vision-test",
        }

    def fake_verdict_fingerprint(**kwargs):
        seen["fingerprint_era"] = kwargs["era"]
        return "beat-local-question"

    monkeypatch.setattr(V, "verify_frame", fake_verify_frame)
    monkeypatch.setattr(V, "verdict_fingerprint", fake_verdict_fingerprint)

    result = B._strictly_verify_native_still(
        proj,
        NS(segment_index=56),
        seg,
        owner,
        image_path,
        NS(anthropic_model="configured-text-model"),
        source_fingerprint=source_fingerprint,
    )

    assert V._season_num(seen["verifier_era"]) == 4
    assert V._season_num(seen["fingerprint_era"]) == 4
    assert seen["verifier_era"] != "S01E04"
    assert seen["fingerprint_era"] != "S01E04"
    assert result["question_fingerprint"] == "beat-local-question"

