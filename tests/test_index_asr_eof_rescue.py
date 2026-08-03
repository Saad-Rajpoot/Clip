from types import SimpleNamespace as NS

from vidlore.clipstudio import index as IX


def _seg(*words):
    return NS(words=[NS(start=a, end=b, word=text) for a, b, text in words])


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


def test_eof_rescue_is_skipped_when_primary_reaches_tail():
    class MustNotRun:
        def transcribe(self, *_args, **_kwargs):
            raise AssertionError("tail pass should not run")

    primary = [(97.0, 98.0, "done")]
    assert IX._rescue_eof_words("source.mp4", MustNotRun(), primary, 100.0) == primary
