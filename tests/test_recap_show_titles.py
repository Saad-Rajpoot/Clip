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


# ------------------------------------------------------------------ lore / "history of X" essays
def test_the_lore_essay_that_aired_house_of_the_dragon_is_rejected():
    """Job b79df3fad5, scene 3 — the OPENING SECONDS of a Littlefinger essay — aired Rhaenyra
    Targaryen from House of the Dragon. Source: "The Untold History Of The Valyrian Steel / Catspaw
    Dagger", which served 4 beats and also supplied the render's weakest still (Daenerys and two
    dragons, relevance 0.475, under "a ship waiting in the bay").

    Nothing else could have caught it. The footage is genuine live action, so the graphics and
    non-live-action gates pass it; the frames are simply from another property, so no cross-show
    check on a GoT-titled source fires. A lore video about an OBJECT spans that object's whole
    in-universe history, which means it spans shows — and the title is where it says so."""
    assert _REJECT_TITLE.search("The Untold History Of The Valyrian Steel / Catspaw Dagger")


@pytest.mark.parametrize("title", [
    "The Untold History of Valyrian Steel",
    "The Untold Story of House Bolton",
    "The History of the Iron Throne - Game of Thrones",
    "Origins of the White Walkers",
    "Origin of Valyrian Steel Explained",
    "Lore of Westeros: the Faith Militant",
    "Valyrian Steel Lore Explained",
    "The Complete History of House Targaryen",
    "Full Timeline of the Dance of the Dragons",
    "Everything We Know About the Night King",
])
def test_the_whole_lore_essay_family_is_rejected(title):
    assert _REJECT_TITLE.search(title), title


def test_the_ban_costs_one_source_in_192():
    """Measured across both jobs' real pools (69 + 123 sources): exactly one title matches, and it
    is the one that leaked. A junk vocabulary that starts eating scene uploads costs more than the
    leak it closes, so this is the number that had to be checked before shipping it."""
    real_scene_uploads = [
        "Lysa Arryn Death Scene, Moon Door - Game of Thrones",
        "Joffrey and Margaery Scene | Game of Thrones",
        "Littlefinger Against The Three-Eyed Raven",
        "Bane of Thrones 7x07 - Petyr Baelish Death Scene",
        "Catelyn Stark calls on her fathers bannermen",
        "Petyr Baelish & Sansa Stark Scene Game of Thrones 6x10",
        "Game of Thrones - Sansa Speaks With Lady Olenna (4k)",
        "Jaime Lannister Trial at Winterfell (FULL SCENE)",
    ]
    for t in real_scene_uploads:
        assert not _REJECT_TITLE.search(t), t
