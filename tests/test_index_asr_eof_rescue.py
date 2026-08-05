import json
import sys
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


def test_transcribe_threads_name_hotwords_without_initial_prompt_to_primary_and_rescue(
        monkeypatch, tmp_path):
    model = _TailDroppingModel()
    monkeypatch.setattr(IX, "_whisper", lambda _cfg: model)

    IX.transcribe_words(
        tmp_path / "source.mp4", NS(), duration=309.336,
        hotwords="Cersei Lannister Olenna Tyrell")

    assert model.calls[0]["hotwords"] == "Cersei Lannister Olenna Tyrell"
    assert model.calls[1]["hotwords"] == "Cersei Lannister Olenna Tyrell"
    assert "initial_prompt" not in model.calls[0]
    assert "initial_prompt" not in model.calls[1]


def test_old_faster_whisper_signature_uses_initial_prompt_without_hotwords(monkeypatch, tmp_path):
    class OldModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, _path, word_timestamps=False, vad_filter=False,
                       condition_on_previous_text=True, clip_timestamps="0",
                       initial_prompt=None):
            self.calls.append({"initial_prompt": initial_prompt, "vad_filter": vad_filter})
            return iter([_seg((97.0, 97.4, "Cersei"), (97.4, 98.0, "speaks"))]), NS()

    model = OldModel()
    monkeypatch.setattr(IX, "_whisper", lambda _cfg: model)
    words = IX.transcribe_words(
        tmp_path / "source.mp4", NS(), duration=100.0,
        hotwords="Cersei Lannister Olenna Tyrell")

    assert [word for _, _, word in words] == ["Cersei", "speaks"]
    assert model.calls == [{
        "initial_prompt": (
            "Cast, character names, and quoted dialogue: "
            "Cersei Lannister Olenna Tyrell"),
        "vad_filter": True,
    }]


def test_old_faster_whisper_large_roster_keeps_priority_prefix_inside_half_context(tmp_path):
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return Encoding(list(range(len(text.split()))))

    class OldModel:
        max_length = 48
        hf_tokenizer = Tokenizer()

        def __init__(self):
            self.prompt = ""

        def transcribe(self, _path, word_timestamps=False, vad_filter=False,
                       condition_on_previous_text=True, clip_timestamps="0",
                       initial_prompt=None):
            self.prompt = initial_prompt
            return iter([]), NS()

    model = OldModel()
    names = ", ".join(f"Priority Character{i}" for i in range(40))
    IX._transcribe_with_vocabulary(model, tmp_path / "s.mp4", hotwords=names)

    assert model.prompt.startswith(
        "Cast, character names, and quoted dialogue: "
        "Priority Character0, Priority Character1")
    assert "Character39" not in model.prompt
    prompt_tokens = model.hf_tokenizer.encode(
        " " + model.prompt, add_special_tokens=False).ids
    assert len(prompt_tokens) <= model.max_length // 2 - 1


def test_old_faster_whisper_retrieval_keeps_dialogue_ahead_of_large_actor_roster(tmp_path):
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return Encoding(list(range(len(text.split()))))

    class OldModel:
        max_length = 64
        hf_tokenizer = Tokenizer()

        def __init__(self):
            self.prompt = ""

        def transcribe(self, _path, word_timestamps=False, vad_filter=False,
                       condition_on_previous_text=True, clip_timestamps="0",
                       initial_prompt=None):
            self.prompt = initial_prompt
            return iter([]), NS()

    actors = [f"Actor Name{i}" for i in range(20)]
    proj = NS(
        meta={"analysis": {
            "characters": [{"name": "Priority Character"}],
            "actors": actors,
        }},
        segments=[NS(index=0, quote="You shot me.")],
    )
    model = OldModel()
    IX._transcribe_with_vocabulary(
        model, tmp_path / "s.mp4",
        hotwords=IX._project_asr_hotwords(proj),
        initial_prompt=IX._project_authored_retrieval_prompt(proj),
        legacy_initial_prompt=IX._project_quote_retrieval_legacy_prompt(proj),
    )

    assert model.prompt.startswith(
        "Cast, character names, and quoted dialogue: "
        "Priority Character, You shot me.")
    assert "You shot me." in model.prompt
    assert "Actor Name19" not in model.prompt


def test_asr_cache_identity_changes_with_model_options_and_vocabulary():
    base = NS(whisper_model="base", whisper_compute="int8")
    other_model = NS(whisper_model="small", whisper_compute="int8")
    fp = IX._asr_prompt_fingerprint(base, "Cersei Olenna")
    assert fp != IX._asr_prompt_fingerprint(base, "Olenna Cersei")
    assert fp != IX._asr_prompt_fingerprint(other_model, "Cersei Olenna")


def test_project_general_asr_excludes_dialogue_and_retrieval_prompt_keeps_it_separate():
    proj = NS(
        meta={"analysis": {
            "characters": [{"name": "Tywin Lannister"}],
            "actors": ["Charles Dance", "charles dance"],
            "anchor_scenes": [{"dialogue": [
                "You shot me",  # punctuation-insensitive duplicate of the beat quote
                "I am your son. I have always been your son.",
                "A longer line, whose comma must not split the decoder entry.",
            ]}],
        }},
        segments=[
            NS(quote=" You're no son of mine. "),
            NS(quote="You shot me."),
            NS(quote=""),
        ],
    )

    retrieval_vocabulary = IX._project_authored_retrieval_prompt(proj)

    assert IX._project_asr_initial_prompt(proj) == ""
    assert retrieval_vocabulary.split(", ") == [
        "You shot me.",
        "You're no son of mine.",
        "I am your son. I have always been your son.",
        "A longer line whose comma must not split the decoder entry.",
    ]
    assert IX._project_asr_hotwords(proj) == "Tywin Lannister, Charles Dance"


def test_authored_quote_changes_retrieval_fingerprint_not_general_asr_fingerprint():
    cfg = NS(whisper_model="base", whisper_compute="int8")
    proj = NS(
        meta={"analysis": {
            "characters": [{"name": "Tywin Lannister"}],
            "actors": ["Charles Dance"],
        }},
        segments=[NS(quote="You shot me.")],
    )
    general_baseline = IX.asr_semantic_fingerprint(proj, cfg)
    retrieval_baseline = IX._quote_retrieval_fingerprint(proj, cfg)

    proj.segments[0].quote = "I am your son. I have always been your son."

    assert IX.asr_semantic_fingerprint(proj, cfg) == general_baseline
    assert IX._quote_retrieval_fingerprint(proj, cfg) != retrieval_baseline


def test_audited_softening_keeps_original_retrieval_prompt_identity_stable():
    cfg = NS(whisper_model="base", whisper_compute="int8")
    quote = "Kill his men."
    proj = NS(
        meta={"analysis": {"characters": [], "actors": []}},
        segments=[NS(quote=quote)],
    )
    baseline = IX._quote_retrieval_fingerprint(proj, cfg)

    proj.segments[0].quote = ""
    proj.meta["selection_relevance_gap_softening"] = {"beats": [{
        "original": {"quote": quote},
    }]}

    assert IX._quote_retrieval_fingerprint(proj, cfg) == baseline


def test_equal_length_quotes_keep_order_when_first_moves_to_softening_provenance():
    first = "Kill his men."
    second = "Kill the man."
    proj = NS(
        meta={"analysis": {"characters": [], "actors": []}},
        segments=[NS(index=0, quote=first), NS(index=1, quote=second)],
    )
    baseline = IX._project_authored_retrieval_prompt(proj)

    proj.segments[0].quote = ""
    proj.meta["selection_relevance_gap_softening"] = {"beats": [{
        "segment_index": 0,
        "original": {"quote": first},
    }]}

    assert IX._project_authored_retrieval_prompt(proj) == baseline
    assert baseline.split(", ") == [first, second]


def test_bounded_project_prompt_keeps_compact_quotes_deterministically():
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return Encoding(list(range(len(text.split()))))

    model = NS(max_length=64, hf_tokenizer=Tokenizer())
    proj = NS(
        meta={"analysis": {
            "characters": [
                {"name": "Priority Character One"},
                {"name": "Priority Character Two"},
            ],
            "actors": [],
        }},
        segments=[NS(quote=f"Line {i}") for i in range(40)],
    )
    requested = IX._project_authored_retrieval_prompt(proj)

    first = IX._bound_asr_vocabulary(model, requested, duplicated=True)
    second = IX._bound_asr_vocabulary(model, requested, duplicated=True)

    assert first == second
    assert first.startswith("Line 0, Line 1")
    assert "Line 39" not in first
    assert IX._project_asr_hotwords(proj).startswith(
        "Priority Character One, Priority Character Two")


def test_authored_dialogue_is_retrieval_prompt_only_never_hotword_bias(tmp_path):
    class Model:
        max_length = 448
        hf_tokenizer = None

        def __init__(self):
            self.kwargs = None

        def transcribe(self, _path, **kwargs):
            self.kwargs = kwargs
            return iter([]), NS()

    model = Model()
    names = "Tywin Lannister, Charles Dance"
    context = names + ", You're no son of mine., I am your son."

    IX._transcribe_with_vocabulary(
        model, tmp_path / "s.mp4", hotwords=names, initial_prompt=context)

    assert model.kwargs["hotwords"] == names
    assert "You're no son of mine." in model.kwargs["initial_prompt"]
    assert "I am your son." in model.kwargs["initial_prompt"]
    assert "son of mine" not in model.kwargs["hotwords"]


def test_general_decode_never_receives_authored_dialogue_but_retrieval_decode_does(
        tmp_path):
    class Model:
        max_length = 448
        hf_tokenizer = None

        def __init__(self):
            self.kwargs = None

        def transcribe(self, _path, **kwargs):
            self.kwargs = kwargs
            return iter([]), NS()

    quote = "You're no son of mine."
    proj = NS(
        meta={"analysis": {
            "characters": [{"name": "Tywin Lannister"}],
            "actors": ["Charles Dance"],
        }},
        segments=[NS(index=0, quote=quote)],
    )
    names = IX._project_asr_hotwords(proj)

    general = Model()
    IX._transcribe_with_vocabulary(
        general, tmp_path / "general.mp4", hotwords=names,
        initial_prompt=IX._project_asr_initial_prompt(proj),
        legacy_initial_prompt=IX._project_asr_legacy_initial_prompt(proj))

    assert general.kwargs["hotwords"] == names
    assert "initial_prompt" not in general.kwargs
    assert quote not in " ".join(str(value) for value in general.kwargs.values())

    retrieval = Model()
    IX._transcribe_with_vocabulary(
        retrieval, tmp_path / "retrieval.mp4", hotwords=names,
        initial_prompt=IX._project_authored_retrieval_prompt(proj),
        legacy_initial_prompt=IX._project_quote_retrieval_legacy_prompt(proj))

    assert retrieval.kwargs["hotwords"] == names
    assert quote in retrieval.kwargs["initial_prompt"]
    assert quote not in retrieval.kwargs["hotwords"]


def test_saturated_retrieval_prompt_chunks_cover_every_authored_line_and_incomplete_fails(
        monkeypatch, tmp_path):
    from vidlore.clipstudio.models import ClipProject, SourceVideo

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return Encoding(list(range(len(text.split()))))

    class Model:
        max_length = 48
        hf_tokenizer = Tokenizer()

        def transcribe(self, _path, **_kwargs):
            return iter([]), NS()

    proj = ClipProject(name="chunked", root=str(tmp_path))
    proj.ensure_dirs()
    media = tmp_path / "source.mp4"
    media.write_bytes(b"source-bytes")
    source = SourceVideo(
        id="s", url="u", title="Game of Thrones scene", permission="owner",
        status=IX.SOURCE_OK, local_path=str(media), duration=30.0)
    proj.sources = [source]
    proj.meta["analysis"] = {
        "characters": [{"name": "Tywin Lannister"}], "actors": ["Charles Dance"]}
    proj.segments = [NS(index=i, quote=f"Authored line number {i}.") for i in range(30)]
    cfg = NS(whisper_model="base", whisper_compute="int8")
    model = Model()
    monkeypatch.setattr(IX, "_whisper", lambda _cfg: model)
    delivered = []

    def decode(_path, _cfg, *, initial_prompt="", **_kwargs):
        delivered.append(initial_prompt)
        i = len(delivered)
        return True, [(float(i), float(i) + .1, f"chunk{i}")]

    monkeypatch.setattr(IX, "_transcribe_words_result", decode)

    assert IX.refresh_source_quote_retrieval(proj, source, cfg)
    expected = IX._project_authored_quotes(proj, proj.meta["analysis"])
    actual = [entry for prompt in delivered for entry in prompt.split(", ")]
    assert len(delivered) > 1
    assert actual == expected

    valid, streams, reason, complete = IX._load_quote_retrieval_streams_result(
        proj, source, cfg, require_complete=True)
    assert valid and complete and reason == ""
    assert [entry for stream in streams for entry in stream["covered_authored_quotes"]] \
        == expected

    sidecar = IX._quote_retrieval_path(proj, source.id)
    payload = json.loads(sidecar.read_text())
    removed = payload["streams"].pop()
    sidecar.write_text(json.dumps(payload))
    valid, _streams, reason, complete = IX._load_quote_retrieval_streams_result(
        proj, source, cfg, require_complete=True)
    assert not valid and not complete
    assert reason == "quote_retrieval_coverage_incomplete"

    prior_decode_count = len(delivered)
    assert IX.refresh_source_quote_retrieval(proj, source, cfg)
    assert len(delivered) == prior_decode_count + 1
    assert delivered[-1] == removed["prompt"]


def test_whisper_model_cache_is_bound_to_every_constructor_input(monkeypatch):
    built = []

    class Model:
        def __init__(self, model, **kwargs):
            built.append((model, kwargs))

    monkeypatch.setattr(IX, "_WHISPER", {})
    monkeypatch.setitem(sys.modules, "faster_whisper", NS(WhisperModel=Model))
    a = NS(whisper_model="base", whisper_compute="int8", whisper_cpu_threads=2)
    b = NS(whisper_model="base", whisper_compute="float32", whisper_cpu_threads=2)
    c = NS(whisper_model="base", whisper_compute="float32", whisper_cpu_threads=4)

    first = IX._whisper(a)
    assert IX._whisper(a) is first
    assert IX._whisper(b) is not first
    assert IX._whisper(c) is not IX._whisper(b)
    assert built == [
        ("base", {"device": "cpu", "compute_type": "int8", "cpu_threads": 2}),
        ("base", {"device": "cpu", "compute_type": "float32", "cpu_threads": 2}),
        ("base", {"device": "cpu", "compute_type": "float32", "cpu_threads": 4}),
    ]


def test_index_cache_prompt_mismatch_refreshes_only_asr(monkeypatch, tmp_path):
    import json
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.models import ClipProject, Shot, SourceVideo

    proj = ClipProject(name="t", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {
        "actors": ["Lena Headey"],
        "characters": [{"name": "Cersei Lannister", "actor": "Lena Headey"}],
    }
    src = SourceVideo(id="s", url="u", title="scene", permission="owner", status=IX.SOURCE_OK,
                      local_path=str(tmp_path / "source.mp4"), duration=100.0)
    shot = Shot(source_id="s", index=0, start=0.0, end=2.0)
    proj.shots_path("s").write_text(json.dumps([shot.to_dict()]))
    (proj.index_dir / "s.words.json").write_text('[[0,1,"old"]]')
    (proj.index_dir / "s.index.meta.json").write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA, "words": True, "ocr": False, "faceid": False,
        "asr_prompt_fingerprint": "stale",
    }))
    cfg = ClipConfig()
    cfg.detect_ocr = False
    seen = {}

    def refresh(_proj, _src, _cfg, **kwargs):
        seen.update(kwargs)
        return [shot]

    monkeypatch.setattr(IX, "refresh_source_words", refresh)
    monkeypatch.setattr(IX, "detect_shots", lambda *_args, **_kwargs:
                        (_ for _ in ()).throw(AssertionError("visual reindex must not run")))
    out = IX.index_source(proj, src, cfg, roster=["Lena Headey"])

    assert out == [shot]
    assert seen["hotwords"] == "Cersei Lannister, Lena Headey"
    assert seen["initial_prompt"] == ""
    assert seen["legacy_initial_prompt"] == "Cersei Lannister, Lena Headey"


def test_vocabulary_is_bounded_by_real_token_count_and_keeps_complete_priority_names(tmp_path):
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return Encoding(list(range(len(text.split()))))

    class Model:
        max_length = 48
        hf_tokenizer = Tokenizer()

        def __init__(self):
            self.kwargs = None

        def transcribe(self, _path, **kwargs):
            self.kwargs = kwargs
            return iter([]), NS()

    model = Model()
    names = ", ".join(f"Character Name{i}" for i in range(30))
    IX._transcribe_with_vocabulary(model, tmp_path / "s.mp4", hotwords=names)

    effective = model.kwargs["hotwords"]
    assert "initial_prompt" not in model.kwargs
    assert effective.startswith("Character Name0, Character Name1")
    assert "Character Name29" not in effective
    assert not effective.endswith("Character")
    hotword_n = len(model.hf_tokenizer.encode(
        " " + effective, add_special_tokens=False).ids)
    assert hotword_n + IX._ASR_PROMPT_RESERVE_TOKENS <= model.max_length


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
    succeeded, words = IX._rescue_eof_words_result("source.mp4", Model(), primary, 100.0)
    assert succeeded is False and words == primary


def test_eof_rescue_nonfinite_confidence_cannot_certify_tail_absence():
    class Model:
        def transcribe(self, _path, **_kwargs):
            return iter([_seg((98.0, 98.4, "possibly"), (98.4, 98.8, "speech"),
                              no_speech_prob=float("nan"), avg_logprob=-0.2)]), NS()

    primary = [(80.0, 80.5, "done")]
    succeeded, words = IX._rescue_eof_words_result("source.mp4", Model(), primary, 100.0)
    assert succeeded is False and words == primary


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
    succeeded, words = IX._transcribe_words_result(tmp_path / "source.mp4", NS())
    assert succeeded is False
    assert words == [(1.0, 1.4, "still"), (1.4, 1.8, "valid")]


def test_tokenizer_failure_cannot_decode_unprompted_then_stamp_roster_identity(
        monkeypatch, tmp_path):
    class BrokenTokenizer:
        def encode(self, *_args, **_kwargs):
            raise RuntimeError("tokenizer unavailable")

    class Model:
        max_length = 448
        hf_tokenizer = BrokenTokenizer()

        def transcribe(self, *_args, **_kwargs):
            raise AssertionError("unprompted decode must not start")

    monkeypatch.setattr(IX, "_whisper", lambda _cfg: Model())
    succeeded, words = IX._transcribe_words_result(
        tmp_path / "source.mp4", NS(), duration=10.0,
        hotwords="Cersei Lannister")
    assert succeeded is False and words == []


def test_triggered_eof_rescue_technical_failure_does_not_certify_primary(monkeypatch, tmp_path):
    class Model:
        def __init__(self):
            self.calls = 0

        def transcribe(self, _path, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return iter([_seg((1.0, 1.4, "primary"))]), NS()

            def broken_tail():
                raise RuntimeError("decoder failed during tail iteration")
                yield

            return broken_tail(), NS()

    monkeypatch.setattr(IX, "_whisper", lambda _cfg: Model())
    succeeded, words = IX._transcribe_words_result(
        tmp_path / "source.mp4", NS(), duration=20.0)

    assert succeeded is False
    assert words == [(1.0, 1.4, "primary")]


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
    monkeypatch.setattr(
        IX, "_transcribe_words_result",
        lambda *_args, **_kwargs: (True, [(1.0, 2.0, "new")]))
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
    monkeypatch.setattr(
        IX, "_transcribe_words_result",
        lambda *_args, **_kwargs: (True, [(1.0, 2.0, "new")]))
    M._QSPAN_CACHE.clear()
    M._QSPAN_CACHE[("s", "line")] = None
    M._QSPAN_CACHE[("other", "line")] = (1.0, 2.0, 1.0)

    assert IX.refresh_source_words(Proj(), source, NS()) == [shot]
    assert ("s", "line") not in M._QSPAN_CACHE
    assert M._QSPAN_CACHE[("other", "line")] == (1.0, 2.0, 1.0)


def test_refresh_technical_failure_never_stamps_current_or_erases_cache(monkeypatch, tmp_path):
    words_file = tmp_path / "s.words.json"
    shots_file = tmp_path / "s.shots.json"
    meta_file = tmp_path / "s.index.meta.json"
    words_file.write_text('[[0,1,"old"]]')
    shots_file.write_text('[{"transcript":"old"}]')
    meta_file.write_text('{"asr_prompt_fingerprint":"stale","words":true}')
    before = (words_file.read_bytes(), shots_file.read_bytes(), meta_file.read_bytes())

    class Proj:
        index_dir = tmp_path
        meta = {}

        @staticmethod
        def shots_path(_sid):
            return shots_file

    source = NS(id="s", status=IX.SOURCE_OK, local_path=str(tmp_path / "source.mp4"),
                duration=10.0)
    monkeypatch.setattr(IX, "load_shots", lambda *_args: [NS(transcript="old")])
    monkeypatch.setattr(IX, "_transcribe_words_result", lambda *_args, **_kwargs: (False, []))

    assert IX.refresh_source_words(Proj(), source, NS(), allow_empty=True) == []
    assert (words_file.read_bytes(), shots_file.read_bytes(), meta_file.read_bytes()) == before


def test_interrupted_pair_commit_leaves_invalid_marker_for_resume(monkeypatch, tmp_path):
    words_file = tmp_path / "s.words.json"
    shots_file = tmp_path / "s.shots.json"
    meta_file = tmp_path / "s.index.meta.json"
    words_file.write_text('[[0,1,"old"]]')
    shots_file.write_text('[{"transcript":"old"}]')
    meta_file.write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA, "words": True,
        "asr_prompt_fingerprint": "previously-current",
    }))

    class Proj:
        index_dir = tmp_path
        meta = {}
        sources = []

        @staticmethod
        def shots_path(_sid):
            return shots_file

    source = NS(id="s", status=IX.SOURCE_OK, local_path=str(tmp_path / "source.mp4"),
                duration=10.0)
    Proj.sources = [source]
    shot = NS(start=0.0, end=10.0, transcript="old",
              to_dict=lambda: {"start": 0.0, "end": 10.0, "transcript": shot.transcript})
    monkeypatch.setattr(IX, "load_shots", lambda *_args: [shot])
    monkeypatch.setattr(
        IX, "_transcribe_words_result",
        lambda *_args, **_kwargs: (True, [(1.0, 2.0, "new")]))
    real_replace = Path.replace

    def interrupt_second_replace(self, target):
        if self.name.endswith(".shots.json.refresh.tmp"):
            raise KeyboardInterrupt("simulated process interruption")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", interrupt_second_replace)
    with pytest.raises(KeyboardInterrupt):
        IX.refresh_source_words(Proj(), source, NS())

    marker = json.loads(meta_file.read_text())
    assert marker["asr_refresh_in_progress"] is True
    assert marker["words"] is False
    assert "asr_prompt_fingerprint" not in marker
    assert IX.asr_pool_current(Proj(), NS(), [source]) is False


def test_refresh_distinguishes_corrupt_cache_from_proven_silence(monkeypatch, tmp_path):
    words_file = tmp_path / "s.words.json"
    shots_file = tmp_path / "s.shots.json"
    meta_file = tmp_path / "s.index.meta.json"
    shots_file.write_text('[{"transcript":""}]')
    meta_file.write_text('{"asr_prompt_fingerprint":"stale","words":true}')

    class Proj:
        index_dir = tmp_path
        meta = {}

        @staticmethod
        def shots_path(_sid):
            return shots_file

    source = NS(id="s", status=IX.SOURCE_OK, local_path=str(tmp_path / "source.mp4"),
                duration=10.0)
    shot = NS(start=0.0, end=10.0, transcript="", to_dict=lambda: {"transcript": ""})
    monkeypatch.setattr(IX, "load_shots", lambda *_args: [shot])
    monkeypatch.setattr(IX, "_transcribe_words_result", lambda *_args, **_kwargs: (True, []))

    words_file.write_text('{broken')
    assert IX.refresh_source_words(Proj(), source, NS(), allow_empty=True) == []
    assert words_file.read_text() == '{broken'
    assert json.loads(meta_file.read_text())["asr_prompt_fingerprint"] == "stale"

    words_file.write_text('[]')
    assert IX.refresh_source_words(Proj(), source, NS()) == [shot]
    assert json.loads(words_file.read_text()) == []
    assert json.loads(meta_file.read_text())["asr_prompt_fingerprint"] != "stale"


def test_index_all_refuses_checkpointable_success_until_whole_asr_pool_is_current(
        monkeypatch, tmp_path):
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.models import ClipProject, Shot, SourceVideo

    proj = ClipProject(name="pool", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {"actors": [], "characters": []}
    source = SourceVideo(
        id="s", url="u", title="scene", permission="owner", status=IX.SOURCE_OK,
        local_path=str(tmp_path / "s.mp4"))
    proj.sources = [source]
    proj.shots_path("s").write_text(json.dumps([
        Shot(source_id="s", index=0, start=0.0, end=1.0).to_dict()
    ]))
    (proj.index_dir / "s.words.json").write_text("[]")
    (proj.index_dir / "s.index.meta.json").write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA, "words": False,
    }))
    monkeypatch.setattr(IX, "index_source", lambda *_args, **_kwargs: [])
    cfg = ClipConfig()

    with pytest.raises(RuntimeError, match="ASR evidence incomplete for 1/1"):
        IX.index_all(proj, cfg)

    (proj.index_dir / "s.index.meta.json").write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA, "words": True,
        "asr_prompt_fingerprint": IX.asr_semantic_fingerprint(proj, cfg),
    }))
    assert IX.index_all(proj, cfg) == {"s": []}


@pytest.mark.parametrize(("shots_payload", "selection_index", "reason"), [
    (None, 0, "shots_cache_invalid_or_missing"),
    ("{corrupt", 0, "shots_cache_invalid_or_missing"),
    ("[]", 0, "shots_cache_invalid_or_missing"),
    (None, 7, "selected_shot_missing_from_index"),
])
def test_asr_resume_artifact_check_includes_shots_and_selected_index(
        tmp_path, shots_payload, selection_index, reason):
    from vidlore.clipstudio.config import ClipConfig
    from vidlore.clipstudio.models import (
        ClipProject, ClipSelection, Shot, SourceVideo,
    )

    proj = ClipProject(name="resume-artifacts", root=str(tmp_path))
    proj.ensure_dirs()
    proj.meta["analysis"] = {"actors": [], "characters": []}
    source = SourceVideo(
        id="s", url="u", title="scene", permission="owner", status=IX.SOURCE_OK,
        local_path=str(tmp_path / "s.mp4"))
    proj.sources = [source]
    proj.selections = [ClipSelection(
        segment_index=0, source_id="s", shot_index=selection_index,
        in_point=0.0, out_point=1.0, confidence=0.8)]
    cfg = ClipConfig()
    (proj.index_dir / "s.words.json").write_text("[]")
    (proj.index_dir / "s.index.meta.json").write_text(json.dumps({
        "schema": IX.INDEX_SCHEMA, "words": True,
        "asr_prompt_fingerprint": IX.asr_semantic_fingerprint(proj, cfg),
    }))
    if shots_payload is None:
        payload = [Shot(source_id="s", index=0, start=0.0, end=1.0).to_dict()]
        if reason == "shots_cache_invalid_or_missing":
            pass  # deliberately leave the file missing
        else:
            proj.shots_path("s").write_text(json.dumps(payload))
    else:
        proj.shots_path("s").write_text(shots_payload)

    audit = IX.asr_pool_cache_audit(proj, cfg)
    assert IX.asr_pool_current(proj, cfg) is False
    assert audit["invalid"] == [{"source_id": "s", "reason": reason}]
