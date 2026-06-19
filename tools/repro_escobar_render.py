"""Cost-locked reproduction driver for the deleted-scene black-span bug.

Mirrors the web editor's /export path: EM.apply_overrides(run_dir) then
render_from_script(...). Forces a $0 render — every paid provider is
zeroed ON THE CONFIG OBJECT (clearing env vars alone does NOT work:
_load_dotenv() reloads any blank key straight from .env). So footage/TTS
come ONLY from the existing cache or free fallbacks, and apply_overrides'
SHA-match skips the LLM. keep_work=True so the work dir survives.

Run from the repo root:
    .venv/bin/python tools/repro_escobar_render.py
"""
import os
import sys
from pathlib import Path

os.environ["VIDLORE_AIMG"] = "0"
os.environ.setdefault("VIDLORE_AI_VIDEO", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vidlore import editor_manifest as EM          # noqa: E402
from vidlore.config import load_config             # noqa: E402
from vidlore.pipeline import load_brief, render_from_script  # noqa: E402

OUT = (ROOT / "output").resolve()
RUN_DIR = OUT / "pablo-escobar--the-rise-and-fall-of-the-king-of-co"


def main():
    print("=" * 70, flush=True)
    print("REPRO: apply_overrides + re-render (cost-locked, $0)", flush=True)
    print("=" * 70, flush=True)

    cfg = load_config(ROOT)
    # ── HARD $0 LOCK: zero every paid provider on the resolved config ──
    cfg.fal_key = ""
    cfg.fal_video_model = ""
    cfg.pexels_api_key = ""
    cfg.shutterstock_api_key = ""
    cfg.pollinations_api_key = ""
    cfg.elevenlabs_api_key = ""
    cfg.anthropic_api_key = ""        # apply_overrides skips LLM anyway
    cfg.aimg_enabled = False
    print(f"providers -> fal={cfg.has_fal} aimg={cfg.has_aimg} "
          f"stock={cfg.has_stock} eleven={cfg.has_elevenlabs} "
          f"llm={cfg.has_llm}  (all must be False)", flush=True)

    ap = EM.apply_overrides(RUN_DIR)
    print("apply_overrides ->", ap, flush=True)

    brief = load_brief(RUN_DIR)
    print(f"brief: {brief.title} | theme={brief.theme} "
          f"| captions={brief.captions}", flush=True)

    res = render_from_script(brief, cfg, OUT, keep_work=True)
    print("RENDER DONE", flush=True)
    print(f"  video_seconds={res.video_seconds}  "
          f"render_wall_s={round(res.seconds, 1)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
