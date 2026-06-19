#!/usr/bin/env python3
"""Micro-render proof of the text_on_black card — no LLM/footage/TTS.
Renders 2 variants (different recipe accents) side by side for visual QA."""
import sys, types
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image
from vidlore.footage import _render_text_on_black_card

CIPHER = ("ALPHA TEAM RENDEZVOUS CONFIRMED|"
          "GRID 34.169N 073.244E AT 0200Z|"
          "PRIMARY ASSET COMPROMISED — DO NOT|"
          "ACKNOWLEDGE ON OPEN CHANNELS|"
          "BURN PROTOCOL AUTHORIZED ON CONTACT")

variants = [
    ("spy steel-blue", {"accent": (110, 150, 178)}, "DECRYPTED TRANSMISSION"),
    ("mystery green",  {"accent": (110, 175, 120)}, "INTERCEPTED // FILE 3301"),
]
imgs = []
for name, theme, title in variants:
    dest = Path(f"/tmp/tob_{name.split()[0]}.png")
    ok = _render_text_on_black_card(types.SimpleNamespace(index=3), theme,
                                    dest, title=title, body=CIPHER)
    print(f"{name}: {'OK' if ok else 'FAIL'} -> {dest}")
    if ok:
        imgs.append(Image.open(dest))

if len(imgs) == 2:
    w = max(i.width for i in imgs) // 2
    scaled = [i.resize((w, i.height * w // i.width)) for i in imgs]
    H = max(i.height for i in scaled)
    canvas = Image.new("RGB", (w * 2 + 12, H), (30, 30, 30))
    canvas.paste(scaled[0], (0, 0)); canvas.paste(scaled[1], (w + 12, 0))
    out = ROOT / "output" / "tob_proof.png"
    out.parent.mkdir(exist_ok=True)
    canvas.save(out)
    print(f"proof: {out}")
