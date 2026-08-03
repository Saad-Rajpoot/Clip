"""Long-script segmentation must never rewrite the narration viewers hear/read."""

import re

from vidlore.clipstudio.config import ClipConfig
from vidlore.clipstudio.segment import _split_long, segment_script


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def test_long_clause_split_preserves_commas_and_conjunctions_verbatim():
    text = (
        "She has ridden south in secret, hidden her face, and told almost no one she is there, "
        "but someone followed her, so she puts the knife on the table and asks one question."
    )
    pieces = _split_long(text, lo=4, hi=9)

    assert len(pieces) > 1
    assert _normalized(" ".join(pieces)) == _normalized(text)
    assert any("," in piece for piece in pieces)
    assert any(re.search(r"\b(?:and|but|so)\b", piece, re.I) for piece in pieces)


def test_end_to_end_segments_preserve_authored_narration_text():
    text = (
        "Watch the chalice carefully, because it moves from hand to hand all afternoon. "
        "Olenna fusses with Sansa's hair, removes one stone, and walks away, but nobody at the "
        "table reacts because nobody is meant to. Who does the cup belong to?"
    )
    cfg = ClipConfig(max_scene_sec=3.0, min_scene_words=3)
    segments = segment_script(text, cfg)

    assert len(segments) > 2
    assert _normalized(" ".join(seg.text for seg in segments)) == _normalized(text)
    assert segments[-1].text.endswith("?")
