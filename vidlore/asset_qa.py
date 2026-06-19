"""Reusable asset-QA layer (engine-level).

A single place that flags doubtful assets in ANY future render so they are never
silently accepted. It composes the four guards (portrait_intel, period_guard,
niche_palette, card_style_guard) into checks that return structured WARNINGS for
the manifest:

  • suspected mismatched portrait (weak name match / wrong source)
  • a pre-photographic person rendered with a modern-looking photo
  • possible modern footage in a historical scene
  • suspicious contemporary-city footage in a period documentary
  • palette that doesn't fit the niche
  • a bright/high-key card inside a dark niche
  • uncertain asset provenance (no trustworthy source/licence)

Each guard already chooses a safer FALLBACK when confidence is low; this layer is
the reporting surface (it does not itself swap assets). Every check is pure and
unit-testable; callers append `warnings` to the manifest.
"""
from __future__ import annotations

from . import card_style_guard, niche_palette, period_guard, portrait_intel


def _w(check: str, severity: str, message: str, suggestion: str = "",
       **extra) -> dict:
    d = {"check": check, "severity": severity, "message": message}
    if suggestion:
        d["suggestion"] = suggestion
    d.update(extra)
    return d


def check_portrait(name: str, *, prov: dict | None = None, title: str = "",
                   name_match_score: float | None = None,
                   ai_generated: bool = False) -> list[dict]:
    """Flag mismatched / pre-photographic-modern-face / uncertain-provenance."""
    out = []
    prov = prov or {}
    cls, reason = portrait_intel.era_class(name)
    ok, why = portrait_intel.acceptable_source(
        name, title=title, prov=prov, name_match_score=name_match_score)
    if not ok:
        out.append(_w("portrait_mismatch", "warn",
                      f"portrait for '{name}' rejected: {why}",
                      "use a verified painting/engraving or stricter name match",
                      era_class=cls))
    if cls == "pre_photographic":
        is_art = portrait_intel.source_is_artwork(title, prov)
        if ai_generated:
            out.append(_w("prephoto_ai_face", "warn",
                          f"'{name}' is pre-photographic ({reason}) but the portrait "
                          f"is AI-generated — likeness is not authentic",
                          "prefer a verified period painting/engraving"))
        elif not is_art:
            out.append(_w("prephoto_modern_face", "warn",
                          f"'{name}' is pre-photographic ({reason}) but the source "
                          f"isn't artwork", "prefer a contemporary painting/engraving",
                          era_class=cls))
    nm = name_match_score if name_match_score is not None else (
        portrait_intel.name_match(name, title) if title else None)
    if nm is not None and nm < 0.5:
        out.append(_w("portrait_weak_namematch", "warn",
                      f"weak name match {nm:.2f} for '{name}'",
                      "tighten name match or use a different source"))
    if not prov.get("source") and not prov.get("domain"):
        out.append(_w("portrait_uncertain_provenance", "info",
                      f"no provenance recorded for '{name}' portrait",
                      "record source/url/licence for audit"))
    return out


def check_footage_period(scene_text: str = "", *, keywords=None, title: str = "",
                         candidate_title: str = "", candidate_snippet: str = "",
                         era: dict | None = None) -> list[dict]:
    """Flag possible modern / contemporary-city footage in a historical scene."""
    out = []
    era = era or period_guard.detect_era(scene_text, keywords=keywords, title=title)
    if not era.get("period_sensitive"):
        return out
    risk = period_guard.period_risk(
        (candidate_title or "") + " " + (candidate_snippet or ""), era=era)
    if risk["risk"] >= 30:
        city = any(m in ("skyline", "skyscraper", "downtown", "aerial city",
                         "modern city", "glass building", "high-rise")
                   for m in risk["markers"])
        out.append(_w("modern_footage_in_period", "warn",
                      f"period scene (~{era.get('year')}, {era.get('label')}) but "
                      f"candidate looks modern: {risk['reason']}",
                      "reject + use period-neutral / archival / AI-historical fallback",
                      period_risk=risk["risk"], markers=risk["markers"]))
        if city:
            out.append(_w("contemporary_city_in_period", "warn",
                          f"contemporary city footage in a {era.get('label')} doc",
                          "use period-neutral landscape or archival imagery"))
    return out


def check_palette_niche(niche: str, palette: str) -> list[dict]:
    """Flag a palette that isn't among the niche's preferred set."""
    norm = niche_palette.normalize_niche(niche)
    weights = niche_palette._NICHE_WEIGHTS.get(norm, {})
    if weights and palette not in weights:
        pref = max(weights, key=weights.get)
        return [_w("palette_niche_mismatch", "warn",
                   f"palette '{palette}' not preferred for niche '{norm}'",
                   f"prefer one of {sorted(weights)} (lead: {pref})")]
    return []


def check_card_brightness(niche: str, kind: str, *, allow_light: bool,
                          contrast_beat: bool = False,
                          palette: str | None = None) -> list[dict]:
    """Flag a bright/high-key card inside a dark niche (unless intentional beat)."""
    if (card_style_guard.is_light_card_kind(kind)
            and card_style_guard.is_dark_niche(niche, palette=palette)
            and allow_light and not contrast_beat):
        return [_w("bright_card_in_dark_niche", "warn",
                   f"high-key '{kind}' card in dark niche '{niche}'",
                   "force a dark variant unless it is an authorised contrast beat")]
    return []


def check_provenance(prov: dict | None) -> list[dict]:
    """Flag uncertain provenance for any sourced asset."""
    prov = prov or {}
    if not prov.get("source") and not prov.get("source_url") and not prov.get("domain"):
        return [_w("uncertain_provenance", "info",
                   "asset has no recorded source/url/domain",
                   "record provenance; prefer a trusted source when confidence low")]
    return []


def summarize(warnings: list[dict]) -> dict:
    """Compact roll-up for a manifest header."""
    by = {}
    for w in warnings or []:
        by[w["check"]] = by.get(w["check"], 0) + 1
    sev = {s: sum(1 for w in warnings or [] if w.get("severity") == s)
           for s in ("warn", "info")}
    return {"total": len(warnings or []), "by_check": by, "by_severity": sev}
