"""The caption line blinked out 246 times in a 12-minute render, and once told a lie.

Both were found by a frame-level audit of job fc41397ea5 and both are burned into the pixels, so
neither could be fixed after the fact.

A. THE BLINK. `write_ass` emits one Dialogue event per WORD, each carrying the whole line with that
   word highlighted. Ending an event at its own word's end therefore removes the ENTIRE caption
   until the next word begins. Measured on the delivered ASS: 303 gaps totalling 138.53s, of which
   57 (29.85s) are the sentence pauses BETWEEN cues, where the band is supposed to be empty. The
   other 246 — 108.7s, 15% of runtime — are mid-sentence blackouts, a hard caption pop every ~2.9s.
   Bridging each event to the next word's start leaves 57 gaps / 29.85s: exactly the pauses.

   The old `ws + 0.06` minimum-duration floor also pushed 4 events past the next event's start, and
   libass stacks the newer of two live events ABOVE the older — on this render that displaced the
   caption a full line height for 0.87s and printed one sentence twice.

B. THE LIE. A caption read "killed Margaery Tyrell, the head of" over a clear shot of MACE Tyrell,
   three cues after another card had said Margaery was killed. `_canonicalize_caption_names` exists
   to repair ASR spellings of cast names, but the owner had uploaded a WRITTEN script and the
   captions were word-synced from it — 2173 caption tokens for 2173 script tokens, not one from ASR.
   Margaery is the roster's only Tyrell, so the bigram rule turned any capitalised word before
   "Tyrell" into her name ('mace' → 'margaery' scores exactly 0.500 against a 0.40 bar). It shipped
   in the .srt too, i.e. as the YouTube subtitle track.

    python3 -m pytest tests/test_caption_continuity.py -q

No network, no LLM.
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidlore.captions import write_ass, WordTiming                 # noqa: E402
from vidlore.clipstudio.build import _canonicalize_caption_names   # noqa: E402
from vidlore.themes import theme as get_theme                      # noqa: E402


def events(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"Dialogue:\s*\d+,(\d+:\d\d:\d\d\.\d\d),(\d+:\d\d:\d\d\.\d\d),", line)
        if m:
            def sec(t):
                h, mi, s = t.split(":")
                return int(h) * 3600 + int(mi) * 60 + float(s)
            out.append((sec(m.group(1)), sec(m.group(2))))
    return sorted(out)


def render(words):
    th = get_theme("history")
    p = Path(tempfile.mkdtemp(prefix="capcont_")) / "c.ass"
    write_ass(words, p, style=th["caption"],
              accent=th.get("caption_accent", th.get("accent", (255, 210, 90))))
    return events(p)


def gaps(ev):
    return [(ev[i][1], ev[i + 1][0]) for i in range(len(ev) - 1)
            if ev[i + 1][0] > ev[i][1] + 0.001]


def overlaps(ev):
    return [i for i in range(len(ev) - 1) if ev[i + 1][0] < ev[i][1] - 0.001]


class TestTheLineStaysUp(unittest.TestCase):
    def test_a_pause_inside_a_sentence_does_not_blank_the_caption(self):
        """The defect, at its smallest: two words 0.5s apart used to leave 0.4s of empty band."""
        w = [WordTiming(word="the", start=0.0, end=0.10),
             WordTiming(word="high", start=0.50, end=0.60),
             WordTiming(word="sparrow", start=0.70, end=1.10)]
        ev = render(w)
        self.assertEqual(gaps(ev), [], "the line must not disappear between words")

    def test_events_never_overlap(self):
        """libass stacks two live events, displacing the caption a full line height."""
        w = [WordTiming(word=f"w{i}", start=i * 0.06, end=i * 0.06 + 0.05) for i in range(12)]
        self.assertEqual(overlaps(render(w)), [])

    def test_overlapping_input_timings_are_clamped_not_propagated(self):
        """The aligner really does hand back overlapping words — word 797 of a delivered render
        started at 251.260 while word 796 still ended at 251.280."""
        w = [WordTiming(word="a", start=251.220, end=251.280),
             WordTiming(word="b", start=251.260, end=251.380),
             WordTiming(word="c", start=251.400, end=251.600)]
        ev = render(w)
        self.assertEqual(overlaps(ev), [])
        self.assertEqual(gaps(ev), [])

    def test_a_zero_length_event_is_dropped_not_flashed(self):
        """Two tokens given the SAME start (seen on an em-dash followed by a word) would otherwise
        emit a zero-length Dialogue line, which renders as a flash."""
        w = [WordTiming(word="x", start=1.00, end=1.06),
             WordTiming(word="y", start=1.00, end=1.90),
             WordTiming(word="z", start=2.00, end=2.30)]
        ev = render(w)
        self.assertTrue(all(b > a for a, b in ev))
        self.assertEqual(overlaps(ev), [])

    def test_the_pause_between_sentences_survives(self):
        """Bridging must not paper over the gaps that are SUPPOSED to be empty — 57 of them,
        29.85s, on the render this came from."""
        w = [WordTiming(word="one", start=0.0, end=0.30),
             WordTiming(word="two", start=0.35, end=0.60),
             WordTiming(word="three", start=6.00, end=6.40)]   # far apart → separate cues
        g = gaps(render(w))
        self.assertTrue(any(b - a > 1.0 for a, b in g),
                        "a multi-second pause between cues must stay empty")

    def test_a_single_word_cue_still_renders(self):
        ev = render([WordTiming(word="alone", start=0.0, end=0.40)])
        self.assertEqual(len(ev), 1)
        self.assertGreater(ev[0][1], ev[0][0])


class TestTheCaptionDoesNotInventNames(unittest.TestCase):
    CAST = ["Margaery Tyrell", "Olenna Tyrell", "Cersei Lannister"]

    def run_pass(self, words, script):
        class W:
            def __init__(self, w):
                self.word = w
        sc = NS(words=[W(x) for x in words])
        proj = NS(meta={"analysis": {"characters": [{"name": c} for c in self.CAST]}})
        n = _canonicalize_caption_names(NS(scenes=[sc]), proj, lambda m: None, script_text=script)
        return n, [w.word for w in sc.words]

    def test_the_shipped_lie(self):
        """'Mace' is in the script the author wrote, so it is not an ASR error and must stand."""
        n, out = self.run_pass(["killed", "Mace", "Tyrell,", "the", "head"],
                               "killed Mace Tyrell the head")
        self.assertEqual(n, 0)
        self.assertIn("Mace", out)
        self.assertNotIn("Margaery", " ".join(out))

    def test_a_plural_surname_is_not_a_misspelling_of_the_singular(self):
        """'Tyrells' scores 0.923 against 'Tyrell' — every plural read as a typo without this."""
        n, out = self.run_pass(["the", "Tyrells", "fell"], "the Tyrells fell")
        self.assertEqual(n, 0)
        self.assertIn("Tyrells", out)

    def test_a_genuine_asr_misspelling_is_STILL_fixed(self):
        """The capability this pass exists for must survive: whisper's 'Alina' for 'Olenna',
        anchored by the following surname, is not in the script."""
        n, out = self.run_pass(["then", "Alina", "Tyrell", "spoke"], "then Olenna Tyrell spoke")
        self.assertEqual(n, 1)
        self.assertIn("Olenna", out)

    def test_the_floor_alone_blocks_it_when_no_script_is_available(self):
        """Belt and braces for the TTS path: 'mace'->'margaery' is 0.500, the bar is 0.52."""
        n, out = self.run_pass(["killed", "Mace", "Tyrell"], "")
        self.assertEqual(n, 0)
        self.assertIn("Mace", out)

    def test_a_canonical_name_is_never_touched(self):
        n, out = self.run_pass(["Olenna", "Tyrell", "smiled"], "Olenna Tyrell smiled")
        self.assertEqual(n, 0)
        self.assertEqual(out, ["Olenna", "Tyrell", "smiled"])

    def test_word_count_and_order_are_preserved(self):
        """Timings are keyed 1:1 to words — a rewrite that changed the count would desync karaoke."""
        words = ["then", "Alina", "Tyrell", "spoke", "again"]
        _n, out = self.run_pass(list(words), "then Olenna Tyrell spoke again")
        self.assertEqual(len(out), len(words))


if __name__ == "__main__":
    unittest.main(verbosity=2)
