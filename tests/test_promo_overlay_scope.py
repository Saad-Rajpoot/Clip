"""Source promo gating must distinguish dedicated boundary cards from body overlays."""

from vidlore.clipstudio.match import _ocr_is_junk, _source_has_promo_overlay
from vidlore.clipstudio.models import Shot


def _shot(index, start, end, text=""):
    return Shot(source_id="scene", index=index, start=start, end=end, ocr_text=text)


def test_dedicated_intro_card_does_not_ban_the_later_clean_scene():
    shots = [
        _shot(0, 0.0, 8.22),
        _shot(1, 8.22, 15.06, "LIKE COMMENT SUBSCRIBE"),
        _shot(2, 15.06, 170.876),
    ]

    assert _source_has_promo_overlay(shots) is False
    assert _ocr_is_junk(shots[1]) is True, "the exempt intro pixels must still be ineligible"


def test_same_promo_card_in_the_body_still_bans_the_source():
    shots = [
        _shot(0, 0.0, 40.0),
        _shot(1, 40.0, 46.0, "LIKE COMMENT SUBSCRIBE"),
        _shot(2, 46.0, 170.0),
    ]

    assert _source_has_promo_overlay(shots) is True


def test_intro_started_overlay_that_bleeds_into_programme_is_a_body_hit():
    shots = [
        _shot(0, 0.0, 10.0),
        _shot(1, 10.0, 30.0, "SUBSCRIBE"),
        _shot(2, 30.0, 170.0),
    ]

    assert _source_has_promo_overlay(shots) is True


def test_dedicated_outro_card_remains_source_safe_but_pixel_ineligible():
    shots = [
        _shot(0, 0.0, 160.0),
        _shot(1, 160.0, 170.0, "THANKS FOR WATCHING SUBSCRIBE"),
    ]

    assert _source_has_promo_overlay(shots) is False
    assert _ocr_is_junk(shots[1]) is True
