"""Transcribing the same audio twice is waste, not diligence.

`transcribe_breakout_words` is the expensive half of breakout selection: every candidate window is
decoded into 3-5 overlapping chunks and each chunk goes through Whisper. A breakout pass over ~25
quotes across ~41 sources reaches the same source audio again and again — a different quote lands
on the same window, the next build pass after self-heal repeats the whole stage, and every resume
starts over. The stage was measured taking hours, which is why breakouts were switched off for one
emergency render. None of that repetition is a judgement; it is the same bytes producing the same
words.

So it is memoised, and the memo is keyed on the AUDIO'S OWN BYTES. Not a path, not an mtime: a wav
rewritten in place with different content gets a different key, and an entry whose stored digest no
longer matches the file is discarded rather than trusted. The key also carries the ASR model
identity, the window/overlap parameters and a schema version, because each of those changes the
words that come back.

The cache cannot approve anything. It returns exactly the list a fresh transcription returns, so
every gate downstream — relevance, caption coverage, the admission judge — sees identical input.
"""
from __future__ import annotations

import json

import pytest

from vidlore.clipstudio import breakout_asr as A


class FakeModel:
    """Counts transcriptions so a cache hit is observable without running Whisper."""
    model_size_or_path = "base"

    def __init__(self):
        self.calls = 0

    def transcribe(self, path, **kw):
        self.calls += 1
        w = type("W", (), {"word": "hello", "start": 0.0, "end": 0.4, "probability": 0.9})()
        seg = type("S", (), {"words": [w]})()
        return [seg], None


def _wav(tmp_path, name="a.wav", payload=b"RIFFfake-audio-bytes"):
    p = tmp_path / name
    p.write_bytes(payload + b"\x00" * 64)
    return p


# ------------------------------------------------------------------ the key
def test_the_key_is_the_audio_content_not_the_path(tmp_path):
    a, b = _wav(tmp_path, "a.wav"), _wav(tmp_path, "b.wav")
    ka = A._asr_cache_key(A._wav_digest(a), FakeModel(), 3.8, 0.6, 5.0)
    kb = A._asr_cache_key(A._wav_digest(b), FakeModel(), 3.8, 0.6, 5.0)
    assert ka == kb, "identical bytes at different paths are the same transcription"


def test_different_audio_is_a_different_key(tmp_path):
    a = _wav(tmp_path, "a.wav", b"RIFFone")
    b = _wav(tmp_path, "b.wav", b"RIFFtwo")
    assert A._asr_cache_key(A._wav_digest(a), FakeModel(), 3.8, 0.6, 5.0) \
        != A._asr_cache_key(A._wav_digest(b), FakeModel(), 3.8, 0.6, 5.0)


@pytest.mark.parametrize("window,overlap", [(2.0, 0.6), (3.8, 0.3)])
def test_changing_the_windowing_changes_the_key(tmp_path, window, overlap):
    d = A._wav_digest(_wav(tmp_path))
    base = A._asr_cache_key(d, FakeModel(), 3.8, 0.6, 5.0)
    assert A._asr_cache_key(d, FakeModel(), window, overlap, 5.0) != base


def test_changing_the_model_changes_the_key(tmp_path):
    d = A._wav_digest(_wav(tmp_path))
    other = FakeModel()
    other.model_size_or_path = "large-v3"
    assert A._asr_cache_key(d, other, 3.8, 0.6, 5.0) != A._asr_cache_key(d, FakeModel(), 3.8, 0.6, 5.0)


def test_the_schema_version_is_part_of_the_key(tmp_path):
    assert A._asr_cache_key(A._wav_digest(_wav(tmp_path)), FakeModel(), 3.8, 0.6, 5.0)["schema"] \
        == A._ASR_CACHE_SCHEMA


# ------------------------------------------------------------------ read / write
def test_a_written_entry_reads_back_identically(tmp_path):
    p = tmp_path / "c.asr.json"
    key = {"schema": 3, "sha256": "x", "model": "base", "window": 3.8, "overlap": 0.6}
    words = [("hi", 0.0, 0.4, 0.9), ("there", 0.4, 0.8, 0.8)]
    A._asr_cache_write(p, key, words)
    assert A._asr_cache_read(p, key) == words


def test_a_mismatched_key_is_not_trusted(tmp_path):
    p = tmp_path / "c.asr.json"
    A._asr_cache_write(p, {"schema": 3, "sha256": "OLD"}, [("hi", 0.0, 0.4, 0.9)])
    assert A._asr_cache_read(p, {"schema": 3, "sha256": "NEW"}) is None


def test_a_corrupt_entry_is_discarded_not_returned(tmp_path):
    p = tmp_path / "c.asr.json"
    p.write_text("{ this is not json")
    assert A._asr_cache_read(p, {"schema": 3}) is None


def test_a_truncated_payload_is_discarded(tmp_path):
    p = tmp_path / "c.asr.json"
    key = {"schema": 3}
    p.write_text(json.dumps({"key": key, "words": [["hi", 0.0]]}))   # missing fields
    assert A._asr_cache_read(p, key) is None


def test_a_missing_entry_is_simply_a_miss(tmp_path):
    assert A._asr_cache_read(tmp_path / "nope.json", {"schema": 3}) is None


def test_the_write_is_atomic(tmp_path):
    """An interrupted run must leave the old entry or the new one, never a half transcript."""
    import inspect
    src = inspect.getsource(A._asr_cache_write)
    assert "os.replace(" in src
    assert ".tmp" in src


def test_a_cache_write_failure_never_costs_the_transcript(tmp_path):
    """Caching is an optimisation; it must not be able to fail the work it memoises."""
    import inspect
    assert "except Exception" in inspect.getsource(A._asr_cache_write)
    A._asr_cache_write(tmp_path / "no" / "such" / "dir" / "c.json", {"schema": 3}, [])


# ------------------------------------------------------------------ end to end
def test_the_second_call_does_not_transcribe_again(tmp_path, monkeypatch):
    wav = _wav(tmp_path)
    model = FakeModel()
    monkeypatch.setattr(A, "probe", lambda p: {"duration": 2.0})
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 1})())
    first = A.transcribe_breakout_words(wav, model=model, duration=2.0)
    calls_after_first = model.calls
    second = A.transcribe_breakout_words(wav, model=model, duration=2.0)
    assert second == first, "the cached result must be identical, not merely similar"
    assert model.calls == calls_after_first, "a cache hit must not re-transcribe"
    assert (wav.with_suffix(wav.suffix + ".asr.json")).exists()


def test_cache_false_forces_a_fresh_transcription(tmp_path, monkeypatch):
    wav = _wav(tmp_path)
    model = FakeModel()
    monkeypatch.setattr(A, "probe", lambda p: {"duration": 2.0})
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 1})())
    A.transcribe_breakout_words(wav, model=model, duration=2.0)
    n = model.calls
    A.transcribe_breakout_words(wav, model=model, duration=2.0, cache=False)
    assert model.calls >= n, "cache=False must not read the memo"


def test_rewriting_the_audio_in_place_invalidates_the_entry(tmp_path, monkeypatch):
    """The failure a path/mtime key would allow: same filename, different audio, stale words."""
    wav = _wav(tmp_path, payload=b"RIFFfirst")
    model = FakeModel()
    monkeypatch.setattr(A, "probe", lambda p: {"duration": 2.0})
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 1})())
    A.transcribe_breakout_words(wav, model=model, duration=2.0)
    key_before = A._asr_cache_key(A._wav_digest(wav), model, 3.8, 0.6, 5.0)
    wav.write_bytes(b"RIFFsecond" + b"\x00" * 64)
    key_after = A._asr_cache_key(A._wav_digest(wav), model, 3.8, 0.6, 5.0)
    assert key_before != key_after
    assert A._asr_cache_read(wav.with_suffix(wav.suffix + ".asr.json"), key_after) is None


def test_hits_and_misses_are_counted_for_the_audit_trail():
    assert set(A._ASR_CACHE_STATS) == {"hit", "miss", "unidentified_model"}


# ------------------------------------------------------------------ what the first key could not see
#
# The key shipped in b05fa21 resolved the model with
#     getattr(model, "model_size_or_path", None) or getattr(model, "model_path", None)
#         or type(model).__name__
# and on the installed faster-whisper (1.2.1) BOTH attributes are absent — verified by probing a
# real WhisperModel. So a real render's key carried the string "WhisperModel": a constant, identical
# for `base` and for `large-v3`. Swapping the model would have served the old model's transcript.
# Duration was missing for a different reason: it is not descriptive, it decides the chunk walk.
class UnidentifiableModel:
    """What faster-whisper 1.2.1 actually hands back: no size, no path."""

    def transcribe(self, path, **kw):
        return [], None


class TaggedModel(UnidentifiableModel):
    def __init__(self, spec):
        setattr(self, A._SPEC_ATTR, spec)


def test_a_model_that_cannot_be_identified_disables_the_cache(tmp_path):
    """No key, so no hit and no write — the caller transcribes. Slower, never another model's words."""
    assert A._model_identity(UnidentifiableModel()) is None
    assert A._asr_cache_key(A._wav_digest(_wav(tmp_path)), UnidentifiableModel(),
                            3.8, 0.6, 5.0) is None


def test_two_different_models_never_share_a_key(tmp_path):
    d = A._wav_digest(_wav(tmp_path))
    a = A._asr_cache_key(d, TaggedModel("base"), 3.8, 0.6, 5.0)
    b = A._asr_cache_key(d, TaggedModel("large-v3"), 3.8, 0.6, 5.0)
    assert a and b and a != b


def test_the_declared_duration_is_part_of_the_key(tmp_path):
    """duration picks the window starts, decides whether a tail window is anchored, and clamps the
    word end times — same bytes, different duration, legitimately different words."""
    d = A._wav_digest(_wav(tmp_path))
    assert A._asr_cache_key(d, FakeModel(), 3.8, 0.6, 5.0) \
        != A._asr_cache_key(d, FakeModel(), 3.8, 0.6, 9.0)


def test_the_decode_code_itself_is_part_of_the_key(tmp_path):
    """A schema integer only invalidates when a human remembers to raise it. The three functions
    that decide every returned word are hashed into the key instead."""
    k = A._asr_cache_key(A._wav_digest(_wav(tmp_path)), FakeModel(), 3.8, 0.6, 5.0)
    assert k["decode"] == A._DECODE_FP
    assert k["decode"] and k["decode"] != "unavailable"


def test_the_decode_fingerprint_covers_flags_merge_and_normalisation():
    import inspect
    src = inspect.getsource(A._decode_fingerprint)
    for fn in ("_materialize_words", "_dedupe_overlaps", "_norm_token"):
        assert fn in src, fn


def test_quantisation_and_device_reach_the_identity():
    """int8 and float16 produce different numbers from the same weights."""
    src = __import__("inspect").getsource(A._model_identity)
    for attr in ("compute_type", "device", "n_mels", "num_languages"):
        assert attr in src, attr


def test_an_unidentified_model_is_counted_not_hidden():
    assert "unidentified_model" in A._ASR_CACHE_STATS
