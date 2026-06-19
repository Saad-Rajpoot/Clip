# RC4 — NICHE ROUTING regression.
#
# Proves look_dna.classify_brief_niche routes a war/geopolitics documentary to
# history/geopolitics (NOT the old silent "business" default), keeps a decisive
# company-revenue brief on "business", and yields a NEUTRAL default (never
# "business") for an empty/vague brief. Also exercises the pipeline-level
# empty/low-confidence -> "history" mapping that consumes this classifier.
#
# Run:  PYTHONPATH=. python tests/test_rc4_niche_routing.py
from vidlore.look_dna import classify_brief_niche

_passed = 0


def check(name, cond):
    global _passed
    assert cond, "FAIL: " + name
    print("  ok ", name)
    _passed += 1


class _StubBrief:
    """Minimal stand-in carrying only what classify_brief_niche reads
    (`.title` / `.prompt`)."""
    def __init__(self, title="", prompt=""):
        self.title = title
        self.prompt = prompt


def _pipeline_niche(brief):
    """Mirror of pipeline.py's RC4 empty/low-confidence niche resolution:
    a '_default'/empty classifier result maps to the neutral documentary
    default 'history', NEVER the old silent 'business'."""
    cls = (classify_brief_niche(brief) or "").strip().lower()
    return cls if cls and cls != "_default" else "history"


# ---- War / geopolitics doc → NOT business ---------------------------------- #
iran_iraq = _StubBrief(
    "Iran-Iraq War: The Gulf in Flames",
    "A documentary on the 1980 invasion, the brutal frontline, the offensive "
    "across the border, and the ceasefire that finally ended the war.")
ii = classify_brief_niche(iran_iraq)
check("iran-iraq war -> history/geopolitics (not business)",
      ii in ("history", "geopolitics"))
check("iran-iraq war is NOT business", ii != "business")

gulf = _StubBrief("The Gulf War",
                  "Operation Desert Storm: the offensive across the border.")
gw = classify_brief_niche(gulf)
check("gulf war -> history/geopolitics (not business)",
      gw in ("history", "geopolitics"))

cambodia = _StubBrief("The Cambodian Genocide",
                      "A history of invasion and military conquest in an "
                      "ancient civilization.")
check("genocide/conquest doc -> history",
      classify_brief_niche(cambodia) == "history")

# ---- Decisive company-revenue brief → business (must still win) ------------ #
biz = _StubBrief(
    "How Standard Oil Built a Monopoly",
    "The corporation's revenue soared after the IPO; the CEO and the company "
    "became a tech-giant-scale business empire on Wall Street.")
check("company-revenue brief -> business",
      classify_brief_niche(biz) == "business")

# A war brief that ALSO mentions a single business word must not flip to
# business on a thin margin.
mixed = _StubBrief("The War Economy",
                   "Sanctions, the frontline, an invasion, and one mention of "
                   "a company.")
check("war-dominant mixed brief stays documentary",
      classify_brief_niche(mixed) in ("history", "geopolitics"))

# ---- Empty / vague brief → NEUTRAL default, never business ----------------- #
vague = _StubBrief("", "A vague brief with no topical signals at all here.")
check("vague brief classifier -> not business",
      classify_brief_niche(vague) != "business")
empty = _StubBrief("", "")
check("empty brief classifier -> _default",
      classify_brief_niche(empty) == "_default")

# Pipeline-level resolution of empty/vague → neutral documentary default.
check("pipeline empty -> history (not business)",
      _pipeline_niche(empty) == "history")
check("pipeline vague -> not business",
      _pipeline_niche(vague) != "business")
check("pipeline iran-iraq -> history/geopolitics",
      _pipeline_niche(iran_iraq) in ("history", "geopolitics"))
check("pipeline company-revenue -> business",
      _pipeline_niche(biz) == "business")

print("\nALL %d CHECKS PASSED" % _passed)
