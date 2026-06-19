#!/usr/bin/env python3
"""Regression guard: RETIRED cheap/template-y motion-graphic cards must NEVER
render again. Locks every gate so they can't silently come back.

Retired (flagged by the user as template-y / repetitive, hurting quality):
  - process_diagram   — numbered four-box step card ("THE FOUR MOVES")
  - cause_effect      — two-box CAUSE -> EFFECT arrow card
  - cause_effect_chain— the causal CARD->ARROW->CARD motion-graphics primitive

For each footage.py card: emission removed, dispatch skips it, renderer inert.
For the registry primitive: kept registered (count stays 70) but dispatch()
skips it via _RETIRED_PRIMITIVES.
"""
import inspect
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

fails = []

import vidlore.script_gen as S
import vidlore.footage as F
import vidlore.footage_strength as FS

menu_src = inspect.getsource(S)
disp = inspect.getsource(F.build_graphic_images)


def check_footage_card(kind, render_fn, render_args):
    # emission: not offered + no "->kind" prompt hint
    if kind in S._EXPLAINER_MENU:
        fails.append(f"{kind}: still in _EXPLAINER_MENU")
    if f"->{kind}" in menu_src:
        fails.append(f"{kind}: prompt still maps ->{kind}")
    # dispatch: immediate skip, no renderer call
    if not re.search(rf'if kind == "{kind}":\s*\n\s*continue', disp):
        fails.append(f"{kind}: build_graphic_images no longer skips it")
    if f"_render_{render_fn.__name__.lstrip('_')}(".replace("_render_render_", "_render_") \
            and f"{render_fn.__name__}(" in disp:
        fails.append(f"{kind}: dispatch still calls {render_fn.__name__}")
    # renderer: inert (returns False, writes nothing)
    out = render_args[0]
    if out.exists():
        out.unlink()
    ok = render_fn(*render_args[1:])
    if ok is not False:
        fails.append(f"{kind}: {render_fn.__name__} returned {ok!r}, expected False")
    if out.exists():
        fails.append(f"{kind}: {render_fn.__name__} wrote a file (should be inert)")


P = Path("/tmp")
check_footage_card(
    "process_diagram", F._render_process_diagram_card,
    (P / "_pd.png", None, {}, P / "_pd.png", "H", "A|x;B|y;C|z;D|w"))
check_footage_card(
    "cause_effect", F._render_cause_effect_card,
    (P / "_ce.png", None, {}, P / "_ce.png", "H", "A|x;;B|y"))

# footage_strength must no longer suggest cause_effect for cause/process beats
if "return \"cause_effect\"" in inspect.getsource(FS) or \
        "return 'cause_effect'" in inspect.getsource(FS):
    fails.append("footage_strength: still suggests cause_effect")

# cause_effect_chain registry primitive: registered (count 70) but dispatch-skipped
from vidlore.motion_graphics.render_dispatch import _RETIRED_PRIMITIVES
from vidlore.motion_graphics.registry import REGISTRY
if "cause_effect_chain" not in _RETIRED_PRIMITIVES:
    fails.append("cause_effect_chain: not in _RETIRED_PRIMITIVES (would still render)")
if len(REGISTRY) != 70:
    fails.append(f"registry count is {len(REGISTRY)}, expected 70")

if fails:
    print("FAIL — a retired template is not fully gated:")
    for f in fails:
        print("  - " + f)
    sys.exit(1)
print("PASS — all retired templates fully gated:")
print("  process_diagram   : emission off · dispatch skip · renderer inert")
print("  cause_effect      : emission off · dispatch skip · renderer inert · no strength-suggest")
print("  cause_effect_chain: dispatch-skipped (registered, registry stays 70)")
sys.exit(0)
