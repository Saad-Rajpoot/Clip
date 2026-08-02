"""A recap show is a recap show whatever it calls itself.

Job 6a26707939's DELIVERED cut aired, at scene 44, a presenter with long straight hair talking to
camera over a CGI Meereen backdrop — a third-party recap channel's own host, inside a Game of
Thrones video essay, with our own burned caption over her face. Source title: "Game of Thrones
Season 5 Rewind - Episode 9: The Dance of Dragons".

The gate was not missing; its vocabulary was. In the SAME render, "Game of Thrones Season 5 Episode
9 Review / Recap" was banned on its title. 'review' and 'recap' were both in the list. 'rewind' was
not, and a recap show that calls itself a rewind walked straight through with 108 shots.

This is the enumeration tax: every synonym has to be paid for. The test pins the names already
paid for so the next one is a one-line addition rather than another audit.
"""
from __future__ import annotations

import pytest

from vidlore.clipstudio.discover import _REJECT_TITLE


# ------------------------------------------------------------------ the title that actually aired
def test_the_rewind_show_that_reached_the_delivered_cut_is_rejected():
    assert _REJECT_TITLE.search("Game of Thrones Season 5 Rewind - Episode 9: The Dance of Dragons")


def test_its_twin_was_always_rejected():
    """Proof the gate worked and only the word was missing."""
    assert _REJECT_TITLE.search("Game of Thrones Season 5 Episode 9 Review / Recap")


# ------------------------------------------------------------------ the rest of the family
@pytest.mark.parametrize("title", [
    "Game of Thrones Season 5 Rewind",
    "GoT Season 8 Rewinds",
    "Game of Thrones Season 6 Wrap Up",
    "Game of Thrones Season 6 Wrapup",
    "Thrones After Show Season 5 Episode 9",
    "Game of Thrones Aftershow",
    "Season 5 Rundown - Game of Thrones",
    "Game of Thrones: The Season in Review",
])
def test_every_recap_show_synonym_is_rejected(title):
    assert _REJECT_TITLE.search(title), title


# ------------------------------------------------------------------ what must still get through
@pytest.mark.parametrize("title", [
    "Melisandre and Stannis Scene | Game of Thrones S03E01 [HD]",
    "Game of Thrones S5EP9: Shireen Baratheon Death Scene",
    "Jon Snow vs Ramsay Bolton Full Scene - Game of Thrones 6x09",
    "Davos finds the stag sculpture - Game of Thrones S06E09",
    "Game of Thrones - Jon Snow takes back Winterfell",
    "Stannis And Davos Prepare To Leave Castle Black",
])
def test_real_scene_uploads_are_untouched(title):
    """Every one of these is a source this job actually used. A widened junk vocabulary that starts
    eating scene uploads costs far more than the leak it closes."""
    assert not _REJECT_TITLE.search(title), title


def test_the_new_markers_are_plural_safe():
    """The whole pattern is wrapped \\b(...)\\b, so an alternative ending in a letter must carry its
    own plural or the trailing boundary kills it — this is exactly how 'reaction' once missed
    'Reactions'."""
    for stem in ("Rewind", "Wrap Up", "After Show", "Rundown"):
        assert _REJECT_TITLE.search(f"Game of Thrones {stem}"), stem
        assert _REJECT_TITLE.search(f"Game of Thrones {stem}s"), stem + "s"
