from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vidlore.clipstudio import index as IX


def _seg(*words, no_speech_prob=0.1, avg_logprob=-0.2):
    return NS(words=[NS(start=a, end=b, word=text) for a, b, text in words],
              no_speech_prob=no_speech_prob, avg_logprob=avg_logprob)


class _TailDroppingModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, _path, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("vad_filter"):
            return iter([_seg((280.0, 280.5, "poison"),
                              (280.5, 281.0, "before"))]), NS()
        return iter([
            _seg((280.02, 280.52, "poison"), (280.52, 281.02, "before")),
            _seg((287.84, 288.20, "Tell"), (288.20, 288.70, "Cersei"),
                 (289.10, 289.25, "I"), (289.25, 289.60, "want"),
                 (289.60, 289.75, "her"), (289.75, 289.95, "to"),
                 (289.95, 290.20, "know"), (290.20, 290.35, "it"),
                 (290.35, 290.55, "was"), (290.55, 290.85, "me")),
            # Typical no-VAD EOF hallucination: plausible text, no real interval.
            _seg((309.336, 309.336, "Thanks")),
        ]), NS()


def test_transcribe_words_recovers_vad_dropped_eof_dialogue(monkeypatch, tmp_path):
    model = _TailDroppingModel()
    monkeypatch.setattr(IX, "_whisper", lambda _cfg: model)

    words = IX.transcribe_words(tmp_path / "source.mp4", NS(), duration=309.336)

    assert [word for _, _, word in words].count("poison") == 1
    assert [word for _, _, word in words][-10:] == [
        "Tell", "Cersei", "I", "want", "her", "to", "know", "it", "was", "me"
    ]
    assert all(end > start for start, end, _ in words)
    assert len(model.calls) == 2
    assert model.calls[1]["vad_filter"] is False
    assert model.calls[1]["condition_on_previous_text"] is False
    assert model.calls[1]["clip_timestamps"] == [279.336, 309.336]


def test_eof_rescue_rejects_single_word_outro_hallucination():
    class Model:
        def transcribe(self, _path, **_kwargs):
            return iter([_seg((99.0, 99.4, "Subscribe"))]), NS()

    primary = [(80.0, 80.5, "done")]
    assert IX._rescue_eof_words("source.mp4", Model(), primary, 100.0) == primary


def test_eof_rescue_rejects_timed_low_confidence_multiword_hallucination():
    class Model:
        def transcribe(self, _path, **_kwargs):
            return iter([_seg((98.0, 98.4, "Subscribe"), (98.4, 98.8, "now"),
                              no_speech_prob=0.91, avg_logprob=-1.4)]), NS()

    primary = [(80.0, 80.5, "done")]
    assert IX._rescue_eof_words("source.mp4", Model(), primary, 100.0) == primary


@pytest.mark.parametrize(("no_speech", "avg_logprob"), [
    (0.601, -0.2),
    (0.25, -1.001),
    (float("nan"), -0.2),
    (0.25, float("inf")),
])
def test_eof_rescue_fails_closed_on_each_confidence_boundary(no_speech, avg_logprob):
    class Model:
        def transcribe(self, _path, **_kwargs):
            return iter([_seg((98.0, 98.4, "Subscribe"), (98.4, 98.8, "now"),
                              no_speech_prob=no_speech, avg_logprob=avg_logprob)]), NS()

    primary = [(80.0, 80.5, "done")]
    assert IX._rescue_eof_words("source.mp4", Model(), primary, 100.0) == primary


def test_eof_rescue_fails_closed_when_segment_confidence_is_missing():
    class Model:
        def transcribe(self, _path, **_kwargs):
            return iter([NS(words=[NS(start=98.0, end=98.4, word="Subscribe"),
                                   NS(start=98.4, end=98.8, word="now")])]), NS()

    primary = [(80.0, 80.5, "done")]
    assert IX._rescue_eof_words("source.mp4", Model(), primary, 100.0) == primary


def test_eof_rescue_accepts_measured_real_beat72_confidence():
    class Model:
        def transcribe(self, _path, **_kwargs):
            return iter([_seg((98.0, 98.4, "Tell"), (98.4, 98.8, "Susie"),
                              no_speech_prob=0.253, avg_logprob=-0.450)]), NS()

    primary = [(80.0, 80.5, "done")]
    assert IX._rescue_eof_words("source.mp4", Model(), primary, 100.0)[-2:] == [
        (98.0, 98.4, "Tell"), (98.4, 98.8, "Susie")]


def test_eof_rescue_is_skipped_when_primary_reaches_tail():
    class MustNotRun:
        def transcribe(self, *_args, **_kwargs):
            raise AssertionError("tail pass should not run")

    primary = [(97.0, 98.0, "done")]
    assert IX._rescue_eof_words("source.mp4", MustNotRun(), primary, 100.0) == primary


def test_probe_failure_preserves_successful_primary_transcript(monkeypatch, tmp_path):
    class Model:
        def transcribe(self, _path, **_kwargs):
            return iter([_seg((1.0, 1.4, "still"), (1.4, 1.8, "valid"))]), NS()

    monkeypatch.setattr(IX, "_whisper", lambda _cfg: Model())
    monkeypatch.setattr(IX, "probe", lambda _path: (_ for _ in ()).throw(OSError("no probe")))
    assert IX.transcribe_words(tmp_path / "source.mp4", NS()) == [
        (1.0, 1.4, "still"), (1.4, 1.8, "valid")]


def test_timed_words_rejects_nonoverlapping_boundary_spill():
    rows = IX._timed_words(iter([_seg(
        (9.8, 9.9, "before"), (9.9, 10.1, "overlap-left"),
        (19.9, 20.1, "overlap-right"), (20.1, 20.2, "after"),
    )]), lo=10.0, hi=20.0)
    assert rows == [(10.0, 10.1, "overlap-left"), (19.9, 20.0, "overlap-right")]


def test_refresh_pair_rolls_back_words_if_shots_replace_fails(monkeypatch, tmp_path):
    words_file = tmp_path / "s.words.json"
    shots_file = tmp_path / "s.shots.json"
    words_file.write_text('[[0,1,"old"]]')
    shots_file.write_text('[{"transcript":"old"}]')
    old_words = words_file.read_bytes()
    old_shots = shots_file.read_bytes()

    class Proj:
        index_dir = tmp_path

        @staticmethod
        def shots_path(_sid):
            return shots_file

    source = NS(id="s", status=IX.SOURCE_OK, local_path=str(tmp_path / "source.mp4"),
                duration=10.0)
    shot = NS(start=0.0, end=10.0, transcript="old",
              to_dict=lambda: {"start": 0.0, "end": 10.0, "transcript": shot.transcript})
    monkeypatch.setattr(IX, "load_shots", lambda _proj, _sid: [shot])
    monkeypatch.setattr(IX, "transcribe_words", lambda *_args, **_kwargs: [(1.0, 2.0, "new")])
    real_replace = Path.replace

    def fail_shots_replace(self, target):
        if self.name.endswith(".shots.json.refresh.tmp"):
            raise OSError("injected second replace failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_shots_replace)
    try:
        IX.refresh_source_words(Proj(), source, NS())
    except OSError as exc:
        assert "injected" in str(exc)
    else:
        raise AssertionError("refresh must fail loudly when the pair cannot commit")
    assert words_file.read_bytes() == old_words
    assert shots_file.read_bytes() == old_shots


def test_refresh_invalidates_only_committed_sources_quote_cache(monkeypatch, tmp_path):
    from vidlore.clipstudio import match as M

    words_file = tmp_path / "s.words.json"
    shots_file = tmp_path / "s.shots.json"
    words_file.write_text('[[0,1,"old"]]')
    shots_file.write_text('[{"transcript":"old"}]')

    class Proj:
        index_dir = tmp_path

        @staticmethod
        def shots_path(_sid):
            return shots_file

    source = NS(id="s", status=IX.SOURCE_OK, local_path=str(tmp_path / "source.mp4"),
                duration=10.0)
    shot = NS(start=0.0, end=10.0, transcript="old",
              to_dict=lambda: {"start": 0.0, "end": 10.0, "transcript": shot.transcript})
    monkeypatch.setattr(IX, "load_shots", lambda _proj, _sid: [shot])
    monkeypatch.setattr(IX, "transcribe_words", lambda *_args, **_kwargs: [(1.0, 2.0, "new")])
    M._QSPAN_CACHE.clear()
    M._QSPAN_CACHE[("s", "line")] = None
    M._QSPAN_CACHE[("other", "line")] = (1.0, 2.0, 1.0)

    assert IX.refresh_source_words(Proj(), source, NS()) == [shot]
    assert ("s", "line") not in M._QSPAN_CACHE
    assert M._QSPAN_CACHE[("other", "line")] == (1.0, 2.0, 1.0)
