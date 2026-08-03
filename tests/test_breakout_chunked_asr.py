from pathlib import Path
from types import SimpleNamespace as NS

from vidlore.clipstudio.breakout_asr import caption_coverage, transcribe_breakout_words


class TruncatingModel:
    """Whole clip starts at the first line; later chunks expose the omitted tail."""
    def transcribe(self, path, **_kwargs):
        # The test's fake ffmpeg writes the requested -ss offset into the file.
        offset = float(Path(path).read_text().splitlines()[0])
        if offset < 1.0:
            words = [("You", 0.1, 0.4), ("murdered", 0.45, 0.9), ("Lysa", 1.0, 1.4)]
        else:
            words = [("pushed", 0.2, 0.6), ("her", 0.65, 0.9),
                     ("through", 1.0, 1.4), ("door", 1.5, 1.9)]
        return iter([NS(words=[NS(word=t, start=a, end=b, probability=.9)
                              for t, a, b in words])]), None


def test_overlapping_windows_recover_a_truncated_tail(monkeypatch, tmp_path):
    wav = tmp_path / "breakout.wav"
    wav.write_bytes(b"fake")

    def fake_run(args, **_kwargs):
        off = float(args[args.index("-ss") + 1])
        Path(args[-1]).write_text(str(off) + "\n" + ("x" * 80))
        return NS(returncode=0)

    monkeypatch.setattr("vidlore.clipstudio.breakout_asr.subprocess.run", fake_run)
    words = transcribe_breakout_words(wav, model=TruncatingModel(), duration=7.4,
                                      window=3.5, overlap=.5)
    text = " ".join(w[0] for w in words)
    assert "murdered" in text
    assert "door" in text
    assert max(w[2] for w in words) > 5.0


def test_caption_coverage_rejects_missing_dialogue_tail():
    spoken = [("a", 0.0, .3, 1), ("b", .4, .8, 1), ("c", 3.5, 4.0, 1)]
    captioned = spoken[:2]
    audit = caption_coverage(spoken, captioned)
    assert not audit["passed"]
    assert audit["coverage"] < .90
    assert audit["uncaptioned_tail_s"] > .5


def test_caption_coverage_accepts_complete_words():
    spoken = [(str(i), i * .3, i * .3 + .2, 1) for i in range(10)]
    audit = caption_coverage(spoken, spoken)
    assert audit["passed"]
    assert audit["coverage"] == 1.0


def test_caption_coverage_rejects_one_missing_middle_word_even_with_tail_covered():
    spoken = [(str(i), i * .3, i * .3 + .2, 1) for i in range(10)]
    captioned = spoken[:4] + spoken[5:]
    audit = caption_coverage(spoken, captioned)
    assert audit["coverage"] == .9
    assert audit["uncaptioned_tail_s"] == 0.0
    assert audit["passed"] is False
