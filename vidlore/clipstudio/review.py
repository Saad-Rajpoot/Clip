"""Stage 7 (human) — the clip-review surface.

Generates a self-contained `review.html` in the project root: the compliance summary, the
flagged-for-review queue up top, then the full timeline — each pick shown with its keyframe
thumbnail, confidence, signal breakdown, source/permission, in/out, flags, and the alternate
candidates for quick comparison. Read-only and dependency-free (no server, no engine edits), so
it can't affect the main tool; the interactive approve/swap web surface is the advanced version.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

from .models import ClipProject, ScriptSegment
from . import index as _index
from . import ledger as _ledger


def _rel(path: str, root: Path) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def _conf_color(c: float) -> str:
    return "#3fb968" if c >= 0.7 else ("#d9a13b" if c >= 0.42 else "#d9534f")


def _card(seg: ScriptSegment, sel, shot, root: Path) -> str:
    thumb = _rel(shot.keyframe_path, root) if shot else ""
    clip = _rel(sel.clip_path, root)
    color = _conf_color(sel.confidence)
    sig = sel.signals or {}
    flags = "".join(f'<span class="flag">{html.escape(f)}</span>' for f in sel.flag_reasons)
    alts = ""
    for a in (sel.alternates or [])[:4]:
        alts += (f'<span class="alt" title="{html.escape(a.source_id)} shot#{a.shot_index} '
                 f'score={a.score}">#{a.shot_index}·{a.score}</span>')
    badge = "REVIEW" if (sel.flagged and not sel.approved) else "ok"
    badge_cls = "rev" if (sel.flagged and not sel.approved) else "okb"
    thumb_html = (f'<img loading="lazy" src="{html.escape(thumb)}">' if thumb
                  else '<div class="noimg">no keyframe</div>')
    # identity + required entity
    req = (f"needs {html.escape(seg.required_entity)} ({html.escape(seg.required_kind)})"
           if seg.required_entity else "")
    id_html = (f'<span class="idok">🆔 {html.escape(sel.identity)} {sel.identity_score:.2f}</span>'
               if sel.identity else (f'<span class="idno">no Face-ID</span>' if seg.required_entity else ''))
    # AI verifier verdict
    v = sel.verifier or {}
    vd = v.get("verdict", "")
    if v.get("status") == "unavailable":
        ver_html = '<span class="vna">verifier: off</span>'
    elif vd == "keep":
        try:
            _vc = float(v.get("confidence") or 0.0)
        except (TypeError, ValueError):      # LLM may emit null/"high" — never kill the QC report
            _vc = 0.0
        ver_html = f'<span class="vok">✓ verified {_vc:.2f}</span>'
    elif vd == "replace":
        ver_html = '<span class="vbad">✗ verifier replaced/failed</span>'
    else:
        ver_html = ''
    return f"""
    <div class="card {badge_cls}">
      <div class="thumb">{thumb_html}<span class="badge {badge_cls}">{badge}</span></div>
      <div class="body">
        <div class="seg">[{seg.index}] {html.escape(seg.text)}</div>
        <div class="meta">
          <span class="conf" style="--c:{color}">{sel.confidence:.2f}</span>
          <span>clip {sig.get('clip','-')}</span>
          <span>faceID {sig.get('faceid','-')}</span>
          <span>trans {sig.get('transcript','-')}</span>
          <span>q {sig.get('quality','-')}</span>
          <span>reuse {sel.reuse_count}</span>
        </div>
        <div class="meta2">
          {id_html} {ver_html}
          {f'<span class="req">{req}</span>' if req else ''}
        </div>
        <div class="meta2">
          <span>⏱ {sel.in_point:.1f}–{sel.out_point:.1f}s</span>
          <span>src {html.escape(sel.source_id or '—')}</span>
          {f'<a href="{html.escape(clip)}">▶ clip</a>' if clip else ''}
          {f'<a href="{html.escape(sel.source_url)}" target="_blank">source</a>' if sel.source_url else ''}
        </div>
        <div class="flags">{flags}{(' · ' + html.escape(v.get('reason',''))) if v.get('reason') else ''}</div>
        <div class="alts">{('alts: ' + alts) if alts else ''}</div>
      </div>
    </div>"""


_CSS = """
*{box-sizing:border-box} body{margin:0;background:#0e1014;color:#e6e8ee;
font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px}
h1{font-size:22px;margin:0 0 4px} .sub{color:#9aa0ac;margin:0 0 18px}
.disc{background:#2a1d12;border:1px solid #6b4a23;color:#e9c896;padding:10px 14px;
border-radius:10px;margin:0 0 18px;font-size:13px}
.stats{display:flex;gap:18px;flex-wrap:wrap;background:#161922;border:1px solid #232838;
border-radius:12px;padding:14px 18px;margin:0 0 22px}
.stats b{display:block;font-size:20px} .stats span{color:#9aa0ac;font-size:12px}
h2{font-size:15px;color:#cfd3dc;margin:22px 0 10px;border-bottom:1px solid #232838;padding-bottom:6px}
.card{display:flex;gap:14px;background:#161922;border:1px solid #232838;border-radius:12px;
padding:12px;margin:0 0 12px} .card.rev{border-color:#7a2f2f;background:#1c1416}
.thumb{position:relative;flex:0 0 200px} .thumb img{width:200px;height:113px;object-fit:cover;
border-radius:8px;background:#000} .noimg{width:200px;height:113px;border-radius:8px;
background:#0a0c10;display:flex;align-items:center;justify-content:center;color:#555;font-size:12px}
.badge{position:absolute;top:6px;left:6px;font-size:10px;font-weight:700;padding:2px 7px;
border-radius:6px} .badge.rev{background:#d9534f;color:#fff} .badge.okb{background:#23502f;color:#9fe0b3}
.body{flex:1;min-width:0} .seg{font-weight:600;margin-bottom:6px}
.meta,.meta2{display:flex;gap:14px;flex-wrap:wrap;color:#9aa0ac;font-size:12px;margin-bottom:4px}
.meta2 a{color:#6fa8ff;text-decoration:none} .conf{color:var(--c);font-weight:700}
.flag{display:inline-block;background:#3a1f1f;color:#e79a9a;border:1px solid #6b2f2f;
padding:1px 8px;border-radius:20px;font-size:11px;margin-right:6px}
.alts{color:#7f8696;font-size:11px;margin-top:4px}
.alt{display:inline-block;background:#1d2230;border:1px solid #2c3344;border-radius:6px;
padding:1px 6px;margin-right:5px}
.idok{color:#7fd6a0;font-weight:600} .idno{color:#d9a13b} .req{color:#7f8696}
.vok{color:#7fd6a0} .vbad{color:#e79a9a;font-weight:600} .vna{color:#7f8696}
"""


def write_review(proj: ClipProject, segments: list[ScriptSegment]) -> Path:
    root = proj.root_path
    summ = _ledger.summary(proj)
    by_idx = {s.index: s for s in segments}
    # shots cache per source
    shots_cache: dict[str, dict] = {}

    def shot_for(sel):
        if not sel.source_id:
            return None
        if sel.source_id not in shots_cache:
            shots_cache[sel.source_id] = {s.index: s for s in _index.load_shots(proj, sel.source_id)}
        return shots_cache[sel.source_id].get(sel.shot_index)

    sels = sorted(proj.selections, key=lambda s: s.segment_index)
    flagged = [s for s in sels if s.flagged and not s.approved]

    def cards(items):
        out = []
        for sel in items:
            seg = by_idx.get(sel.segment_index)
            if seg is None:
                continue
            out.append(_card(seg, sel, shot_for(sel), root))
        return "".join(out)

    stats = (
        f'<div><b>{summ["segments"]}</b><span>segments</span></div>'
        f'<div><b>{summ["flagged_for_review"]}</b><span>need review</span></div>'
        f'<div><b>{summ["mean_confidence"]:.2f}</b><span>mean confidence</span></div>'
        f'<div><b>{summ["sources_used"]}</b><span>sources used</span></div>'
        f'<div><b>{summ["total_clip_seconds"]:.0f}s</b><span>total footage</span></div>'
        f'<div><b>{int(summ["dominant_source"]["share"]*100)}%</b><span>from top source</span></div>'
    )
    flagged_html = (f'<h2>⚠ Needs review ({len(flagged)})</h2>{cards(flagged)}'
                    if flagged else '<h2>✓ Nothing flagged</h2>')
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ClipStudio review — {html.escape(proj.name)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>ClipStudio review — {html.escape(proj.name)}</h1>
  <p class="sub">{summ['segments']} clips · generated for human verification</p>
  <div class="disc">⚖ {html.escape(_ledger.DISCLAIMER)}</div>
  <div class="stats">{stats}</div>
  {flagged_html}
  <h2>Full timeline ({len(sels)})</h2>
  {cards(sels)}
</div></body></html>"""
    out = root / "review.html"
    out.write_text(doc, encoding="utf-8")
    return out
