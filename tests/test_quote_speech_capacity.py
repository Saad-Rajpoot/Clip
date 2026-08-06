"""A prompted decode that regurgitates its own prompt over silence must not block publication.

Job 0ca9dc4c2f died at the end of a 2-hour render because beats 88/89 shared the authored line
"One of the stones was not a stone." The prompted retrieval "found" it at 92.23-104.61s of a
105.4s clip — a stretch that is digitally silent apart from one 0.1s transient. The unprompted
confirmation could not confirm it (no words) and could not reject it either, because a silent
segment fails the decoder's own confidence floor, and a low-confidence segment is never treated as
proof of absence. The result was a permanently indeterminate quote branch: a hard technical blocker
that no retry, no recovery pass and no review-draft policy could ever clear.

Audible time settles it. These tests pin that the measurement separates silence from speech, that
absence is only ever concluded when the phrase physically cannot fit, and that every other path
still fails closed exactly as before.
"""
from __future__ import annotations

import json
import math
import struct
import wave

import pytest

from vidlore.clipstudio import index as IX
from vidlore.clipstudio import policy as P
from vidlore.clipstudio import relevance_contract as R
from vidlore.clipstudio import verify as V
from vidlore.clipstudio.config import load_clip_config
from vidlore.clipstudio.models import ClipProject, ClipSelection, ScriptSegment, Shot, SourceVideo


QUOTE = "One of the stones was not a stone."
RATE = 16000


def _write_wav(path, plan, *, rate=RATE):
    """Render a real decodable wav from [(seconds, amplitude), ...] spans."""
    frames = bytearray()
    for seconds, amplitude in plan:
        for n in range(int(round(seconds * rate))):
            value = amplitude * math.sin(2.0 * math.pi * 220.0 * (n / rate))
            frames += struct.pack("<h", max(-32767, min(32767, int(value * 32767.0))))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return path


def _project(tmp_path, media, *, quote=QUOTE, duration=20.0):
    proj = ClipProject(name="capacity", root=str(tmp_path))
    proj.ensure_dirs()
    proj.sources = [SourceVideo(id="s1", url="u", title="Game of Thrones Purple Wedding",
                                permission="owner", status="ok", local_path=str(media),
                                duration=duration)]
    frame = tmp_path / "shot_0000.jpg"
    frame.write_bytes(b"representative-frame")
    shot = Shot(source_id="s1", index=0, start=0.0, end=2.0, keyframe_path=str(frame))
    proj.shots_path("s1").write_text(json.dumps([shot.to_dict()]))
    proj.meta["analysis"] = {"video_type": "multi_scene", "characters": [], "actors": []}
    (proj.index_dir / "s1.index.meta.json").write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA,
        "words": True,
        "asr_prompt_fingerprint": IX.asr_semantic_fingerprint(proj, load_clip_config()),
    }))
    seg = ScriptSegment(index=0, text="The poison rode in on her hairnet.",
                        required_entity="Sansa Stark", required_kind="character",
                        visual_policy=P.EXACT, is_specific_claim=True, quote=quote)
    verdict = {"status": "ok", "verdict": "keep", "matches_narration": True,
               "specific_enough": True, "correct_subject_visible": True,
               "wrong_subject_visible": False, "contradicts_narration": False,
               "quality_ok": True}
    sel = ClipSelection(segment_index=0, source_id="s1", shot_index=0,
                        in_point=0.0, out_point=2.0, confidence=0.8,
                        verifier=dict(verdict), signals={"dialogue": 1.0})
    V.bind_selection_verifier_evidence(
        proj, sel, seg, verdict, shot=shot, model="vision-test",
        is_specific=True, multiframe=True, faceid_names=[],
        era=V._project_beat_era(proj, seg), must_see=P.deictic_target(seg))
    proj.selections = [sel]
    proj.segments = [seg]
    return proj, seg, sel


# --- the measurement itself ------------------------------------------------------------------

def test_measurement_separates_a_silent_outro_from_real_dialogue(tmp_path):
    """The production window's shape: 15s of silence carrying one brief transient."""
    media = _write_wav(tmp_path / "tail.wav", [(7.0, 0.0), (0.2, 0.5), (7.8, 0.0)])

    silent = IX.measure_window_speech_capacity(media, 0.0, 15.0)
    assert silent["status"] == "ok"
    assert silent["measured_seconds"] == pytest.approx(15.0, abs=0.05)
    # Only the transient is audible — nowhere near enough to carry a sentence.
    assert silent["audible_seconds"] == pytest.approx(0.2, abs=0.05)

    speech = IX.measure_window_speech_capacity(
        _write_wav(tmp_path / "speech.wav", [(15.0, 0.3)]), 0.0, 15.0)
    assert speech["status"] == "ok"
    # Continuously audible media is measured as fully capable, so the gate never fires on it.
    assert speech["audible_seconds"] == pytest.approx(speech["measured_seconds"], abs=0.05)
    assert speech["audible_seconds"] > 70.0 * silent["audible_seconds"]


def test_measurement_counts_every_audible_instant_once(tmp_path):
    """Frames must not overlap: an inflated duration would let a gate under-reject."""
    media = _write_wav(tmp_path / "half.wav", [(5.0, 0.4), (5.0, 0.0)])
    measured = IX.measure_window_speech_capacity(media, 0.0, 10.0)
    assert measured["audible_seconds"] == pytest.approx(5.0, abs=0.1)
    assert measured["audible_seconds"] <= measured["measured_seconds"]


@pytest.mark.parametrize(("path", "start", "end"), [
    ("does-not-exist.wav", 0.0, 5.0),
    ("tail.wav", 5.0, 5.0),
    ("tail.wav", -1.0, 5.0),
])
def test_measurement_faults_are_inconclusive_never_silent(tmp_path, path, start, end):
    """An unreadable measurement must never look like proof that nothing was said."""
    _write_wav(tmp_path / "tail.wav", [(2.0, 0.3)])
    result = IX.measure_window_speech_capacity(tmp_path / path, start, end)
    assert result["status"] == "inconclusive"
    assert "audible_seconds" not in result


# --- the impossibility bound -----------------------------------------------------------------

def test_phrase_minimum_stays_below_any_human_speaking_rate():
    """The bound exists to prove impossibility, so it must undercut real delivery by a lot."""
    words = len(QUOTE.split())
    required = R._phrase_minimum_speech_seconds(QUOTE)
    # Normal delivery of this line runs ~2.5s; the bound must be far under that.
    assert required < 1.5
    assert required >= words / R.QUOTE_CAPACITY_MAX_WORDS_PER_SEC
    assert R.QUOTE_CAPACITY_MAX_WORDS_PER_SEC >= 8.0
    assert R._phrase_minimum_speech_seconds("") == R.QUOTE_CAPACITY_MIN_PHRASE_SEC
    assert R._phrase_minimum_speech_seconds("Hi") >= R.QUOTE_CAPACITY_MIN_PHRASE_SEC
    # Longer phrases need strictly more room.
    assert R._phrase_minimum_speech_seconds(QUOTE + " " + QUOTE) > required


# --- the confirmation decision ---------------------------------------------------------------

def test_prompted_hit_over_silence_is_conclusively_rejected(tmp_path, monkeypatch):
    """The exact job-0ca9dc4c2f failure: rejected on measured bytes, with no decode needed."""
    media = _write_wav(tmp_path / "s1.wav", [(20.0, 0.0)])
    proj, _seg, _sel = _project(tmp_path, media)
    decodes = []
    monkeypatch.setattr(IX, "transcribe_unprompted_window",
                        lambda *a, **k: decodes.append(a) or {
                            "status": "inconclusive",
                            "reason": "segment_confidence_below_floor"})

    result = R._confirm_prompted_quote_span_unprompted(
        proj, proj.source("s1"), QUOTE, [5.0, 15.0, 0.857], load_clip_config(),
        exact_contiguous_required=False)

    assert result["status"] == "rejected"
    assert result["reason"] == "unprompted_window_below_phrase_speech_capacity"
    assert result["confirmed_span"] is None
    assert result["speech_capacity"]["audible_seconds"] < result["required_speech_seconds"]
    # Proving absence from the waveform costs no ASR at all.
    assert decodes == []


def test_audible_window_still_fails_closed_on_an_uncertain_decode(tmp_path, monkeypatch):
    """Where speech could exist, an uncertain decoder keeps blocking exactly as before."""
    media = _write_wav(tmp_path / "s1.wav", [(20.0, 0.3)])
    proj, _seg, _sel = _project(tmp_path, media)
    monkeypatch.setattr(IX, "transcribe_unprompted_window", lambda *a, **k: {
        "status": "inconclusive", "reason": "segment_confidence_below_floor"})

    result = R._confirm_prompted_quote_span_unprompted(
        proj, proj.source("s1"), QUOTE, [5.0, 15.0, 0.857], load_clip_config(),
        exact_contiguous_required=False)

    assert result["status"] == "inconclusive"
    assert result["reason"] == "segment_confidence_below_floor"


def test_audible_window_can_still_confirm_a_real_line(tmp_path, monkeypatch):
    """The gate must not stand between an audible window and a genuine confirmation."""
    media = _write_wav(tmp_path / "s1.wav", [(20.0, 0.3)])
    proj, _seg, _sel = _project(tmp_path, media)
    spoken = [[5.0 + i * 0.2, 5.15 + i * 0.2, word] for i, word in enumerate(QUOTE.split())]

    def decode(_path, _cfg, start, end, **_kwargs):
        return {
            "status": "ok", "reason": "",
            "decode_window": [round(float(start), 3), round(float(end), 3)],
            "timed_words": spoken,
            "segment_confidence": [
                {"no_speech_prob": 0.02, "avg_logprob": -0.2, "accepted": True}],
        }

    monkeypatch.setattr(IX, "transcribe_unprompted_window", decode)
    result = R._confirm_prompted_quote_span_unprompted(
        proj, proj.source("s1"), QUOTE, [5.0, 6.6, 0.9], load_clip_config(),
        exact_contiguous_required=False)

    assert result["status"] == "confirmed"
    assert result["confirmed_span"] is not None


def test_measurement_fault_falls_through_to_the_decode(tmp_path, monkeypatch):
    """A broken probe must not become a licence to reject; the decoder still rules."""
    media = _write_wav(tmp_path / "s1.wav", [(20.0, 0.0)])
    proj, _seg, _sel = _project(tmp_path, media)
    monkeypatch.setattr(IX, "measure_window_speech_capacity", lambda *a, **k: {
        "status": "inconclusive", "reason": "capacity_extract_error:OSError"})
    decodes = []
    monkeypatch.setattr(IX, "transcribe_unprompted_window",
                        lambda *a, **k: decodes.append(a) or {
                            "status": "inconclusive",
                            "reason": "segment_confidence_below_floor"})

    result = R._confirm_prompted_quote_span_unprompted(
        proj, proj.source("s1"), QUOTE, [5.0, 15.0, 0.857], load_clip_config(),
        exact_contiguous_required=False)

    assert result["status"] == "inconclusive"
    assert decodes, "a failed measurement must not skip the decoder"


# --- what the render actually needed ----------------------------------------------------------

def test_hallucinated_pool_hit_types_the_quote_as_paraphrase_and_unblocks_the_beat(tmp_path):
    """End to end: the beat that killed the render now publishes, without softening a gate.

    The retrieval sidecar holds the authored line (a prompted decode wrote it there), the source
    audio is silent, and no other source carries the phrase. The only honest reading is that the
    essayist wrote the line, so the beat must not be held to a spoken-dialogue floor.
    """
    media = _write_wav(tmp_path / "s1.wav", [(20.0, 0.0)])
    proj, seg, _sel = _project(tmp_path, media)
    hallucinated = [[5.0 + i * 1.2, 5.9 + i * 1.2, word] for i, word in enumerate(QUOTE.split())]
    (proj.index_dir / "s1.words.json").write_text(json.dumps([]))
    assert IX._save_quote_retrieval_words(
        proj, proj.source("s1"), load_clip_config(), hallucinated)

    evidence = R._quote_pool_branches(proj, [seg])[0]
    assert evidence["branch"] == "paraphrase"
    assert evidence["branch_reason"] == "prompted_asr_hits_rejected_by_unprompted_confirmation"
    assert evidence["unprompted_confirmation_rejected_count"] >= 1
    assert evidence["unprompted_confirmation_inconclusive_count"] == 0
    assert evidence["pool_match"] is None

    # The quote lane is what killed the render; it must now be silent. (The synthetic selection
    # still owes ordinary verifier evidence — that lane is untouched by this fix.)
    audit = R.evaluate_selection_relevance(proj, [seg])
    quote_reasons = [reason for entry in audit["blockers"]
                     for reason in entry["reasons"] if reason.startswith("exact_quote")]
    assert quote_reasons == []
