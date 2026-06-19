# RC4 — TEXT-CARD MACHINE-PAYLOAD SANITIZER regression.
#
# Proves factual_guard.is_machine_payload / sanitize_card_text catch leaked
# internal/serialized data in a HUMAN text field (the bug: a route card's packed
# waypoint DSL "Tehran@35.69,51.39|Baghdad@33.34,44.40|..." surfaced as an
# act_chapter_card TITLE) while never flagging a legitimate title that merely
# contains a digit, a year, a single stray '@', or one '|'.
#
# Run:  PYTHONPATH=. python tests/test_rc4_textcard_sanitizer.py
import vidlore.factual_guard as fg

_passed = 0


def check(name, cond):
    global _passed
    assert cond, "FAIL: " + name
    print("  ok ", name)
    _passed += 1


# The EXACT payload that leaked onto the act_chapter_card title.
GEO = ("Tehran@35.69,51.39|Baghdad@33.34,44.40|Riyadh@24.69,46.72|"
       "Kuwait City@29.37,47.98")

# ---- is_machine_payload: TRUE (must be flagged) ---------------------------- #
check("exact geo payload flagged", fg.is_machine_payload(GEO) is True)
check("raw json object flagged", fg.is_machine_payload('{"a":1}') is True)
check("raw json array flagged", fg.is_machine_payload('[{"x":1}]') is True)
check("stops= DSL flagged",
      fg.is_machine_payload("stops=Tehran@35.6,51.4|Baghdad@33.3,44.4") is True)
check("bars= DSL flagged",
      fg.is_machine_payload("bars=1870:4|1880:30") is True)
check("region= DSL flagged", fg.is_machine_payload("region=mideast") is True)
check("lone coord-only payload flagged",
      fg.is_machine_payload("Tehran@35.69,51.39") is True)
check("unresolved placeholder flagged",
      fg.is_machine_payload("{{title}}") is True)
check("None@ artifact flagged", fg.is_machine_payload("None@0,0") is True)

# ---- is_machine_payload: FALSE (legit human titles, must NOT be flagged) --- #
check("year+number title kept",
      fg.is_machine_payload("Iraq's 1980 Invasion") is False)
check("single stray @ kept",
      fg.is_machine_payload("email@domain mention") is False)
check("single '|' divider kept",
      fg.is_machine_payload("Chapter 3 | Rise of the Empire") is False)
check("plain title kept", fg.is_machine_payload("The Gulf War") is False)
check("title with a colon+year kept",
      fg.is_machine_payload("Operation Desert Storm: 1991") is False)
check("empty string not flagged", fg.is_machine_payload("") is False)
check("None not flagged", fg.is_machine_payload(None) is False)
check("non-string not flagged", fg.is_machine_payload(12345) is False)

# ---- sanitize_card_text ---------------------------------------------------- #
check("sanitize geo payload -> None", fg.sanitize_card_text(GEO) is None)
check("sanitize stops= -> None",
      fg.sanitize_card_text("stops=A@1,2|B@3,4") is None)
check("sanitize json -> None", fg.sanitize_card_text('{"a":1}') is None)
check("sanitize clean title unchanged",
      fg.sanitize_card_text("The Gulf War") == "The Gulf War")
check("sanitize year title unchanged",
      fg.sanitize_card_text("Iraq's 1980 Invasion") == "Iraq's 1980 Invasion")
check("sanitize empty -> None", fg.sanitize_card_text("") is None)
check("sanitize None -> None", fg.sanitize_card_text(None) is None)

print("\nALL %d CHECKS PASSED" % _passed)
