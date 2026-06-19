import os
os.environ["VIDLORE_MOTION_GRAPHICS"]="1"
os.environ["VIDLORE_AIMG"]="0"; os.environ["FAL_KEY"]=""
os.environ["VIDLORE_SHUTTERSTOCK"]="0"; os.environ["WEB_FOOTAGE_ENGINE"]="0"
os.environ["WEB_IMAGE_ENGINE"]="0"; os.environ["VIDLORE_TTS_BACKEND"]="legacy"
os.environ["VIDLORE_MUSIC_VOLUME"]="0.5"
os.environ.setdefault("VIDLORE_ENCODE_WORKERS","6")
from pathlib import Path
from vidlore.config import load_config
from vidlore.pipeline import load_brief, render_from_script
import vidlore.editor_manifest as EM
SLUG="pablo-escobar--the-rise-and-fall-of-the-king-of-co"
OUT=Path("output"); d=OUT/SLUG
brief=load_brief(d); cfg=load_config()
g=EM.load_overrides(d).get("global",{})
if g.get("captions_enabled") is not None: brief.captions=bool(g["captions_enabled"])
print("=== MG SAMPLE RENDER (flag ON, $0) ===", flush=True)
res=render_from_script(brief, cfg, OUT)
print("RENDER_DONE", getattr(res,"video_seconds",None), flush=True)
