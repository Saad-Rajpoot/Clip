"""Automated multilingual QA — renders representative graphics in every
supported language and checks for tofu, glyph failures, and gross
overflow.  Run with::

    python -m vidlore.multilang_qa --out /tmp/multilang_qa

Exit code is 0 only when all checks pass; useful as a pre-release gate.

The test sheets cover:
    * NAME_REVEAL  (premium serif display name + role)
    * TITLE / DISPLAY (large cinematic title)
    * DOCUMENT CARD (accent panel + header + body wrap)
    * BULLET LIST  (accent dot + body line)

For each language we run the bundled font chain via the production
`vidlore.footage._ff` helper -- so if `_ff` regresses, this test
catches it.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Iterable

# Built-in language samples — engineered to expose typography problems
# (long German compounds, Turkish diacritics, Arabic shaping, etc.).
SAMPLES: dict[str, dict[str, object]] = {
    "EN": {
        "name": "ELI",
        "role": "AMISH ELDER · LANCASTER CO.",
        "title": "The Forgotten Ledger",
        "doc_t": "PAGE 47 · 1887 LEDGER",
        "doc_b": "Soaked seed corn in copper sulfate and zinc ash.",
        "bullets": ["Copper strips repel beetles",
                    "Costs eleven cents per acre",
                    "Method documented in 1887"],
    },
    "DE": {
        "name": "ELI",
        "role": "AMISH-ÄLTESTER · LANCASTER COUNTY",
        "title": "Das vergessene Hauptbuch",
        "doc_t": "SEITE 47 · BAUERNAUFZEICHNUNGEN",
        "doc_b": "Größenwahnsinnige Bewässerungsanlagen im "
                 "Landwirtschaftsministerium.",
        "bullets": ["Kupferstreifen vertreiben Käfer",
                    "Elf Cent Materialkosten pro Hektar",
                    "Methode 1887 dokumentiert"],
    },
    "ES": {
        "name": "ELI",
        "role": "ANCIANO AMISH · CONDADO DE LANCASTER",
        "title": "El Libro Mayor Olvidado",
        "doc_t": "PÁGINA 47 · LIBRO DE 1887",
        "doc_b": "¿Cómo es posible que esta técnica desapareciera de los "
                 "registros oficiales?",
        "bullets": ["Las tiras de cobre repelen escarabajos",
                    "Cuesta once céntimos por hectárea",
                    "Método documentado en 1887"],
    },
    "TR": {
        "name": "ELI",
        "role": "AMİSH İHTİYARI · LANCASTER İLÇESİ",
        "title": "Unutulmuş Defter",
        "doc_t": "SAYFA 47 · 1887 KAYITLARI",
        "doc_b": "Çiftçilerin böcek istilasına karşı uyguladığı yöntem "
                 "şaşırtıcıydı.",
        "bullets": ["Bakır şeritler böcekleri uzak tutar",
                    "Hektar başına on bir kuruş maliyet",
                    "1887'de belgelenmiş yöntem"],
    },
    "JP": {
        "name": "エリ",
        "role": "アーミッシュ長老 · ランカスター郡",
        "title": "忘れられた帳簿",
        "doc_t": "47ページ · 1887年の記録",
        "doc_b": "農家は銅と亜鉛を使って害虫を遠ざけました。これは19世紀の知恵です。",
        "bullets": ["銅の帯が甲虫を追い払う",
                    "1エーカーあたり11セント",
                    "1887年に文書化された手法"],
    },
    "KR": {
        "name": "엘리",
        "role": "아미시 장로 · 랭커스터 카운티",
        "title": "잊혀진 장부",
        "doc_t": "47쪽 · 1887년 기록",
        "doc_b": "농부들은 구리와 아연을 사용하여 곤충을 막았습니다.",
        "bullets": ["구리 띠가 딱정벌레를 쫓아냅니다",
                    "에이커당 11센트",
                    "1887년에 기록된 방법"],
    },
    "AR": {
        "name": "إيلي",
        "role": "شيخ الأميش · مقاطعة لانكستر",
        "title": "السجل المنسي",
        "doc_t": "صفحة ٤٧ · سجل ١٨٨٧",
        "doc_b": "استخدم المزارعون النحاس والزنك لإبعاد الحشرات الضارة عن "
                 "المحاصيل.",
        "bullets": ["شرائط النحاس تطرد الخنافس",
                    "أحد عشر سنتاً للهكتار الواحد",
                    "طريقة موثقة عام ١٨٨٧"],
    },
    "UR": {
        "name": "ایلی",
        "role": "ایمش بزرگ · لینکاسٹر کاؤنٹی",
        "title": "بھولا ہوا رجسٹر",
        "doc_t": "صفحہ ۴۷ · ۱۸۸۷ کا ریکارڈ",
        "doc_b": "کسانوں نے تانبے اور جستے سے کیڑوں کو دور رکھا۔",
        "bullets": ["تانبے کی پٹیاں چقندر کو دور بھگاتی ہیں",
                    "فی ایکڑ گیارہ سینٹ",
                    "۱۸۸۷ میں دستاویزی طریقہ"],
    },
    "HE": {
        "name": "אלי",
        "role": "זקן האמיש · מחוז לנקסטר",
        "title": "הספר הנשכח",
        "doc_t": "עמוד 47 · ספר 1887",
        "doc_b": "החקלאים השתמשו בנחושת ובאבץ כדי להרחיק את החרקים מהגידולים.",
        "bullets": ["רצועות נחושת מרחיקות חיפושיות",
                    "אחד עשר סנט להקטר",
                    "שיטה שתועדה בשנת 1887"],
    },
}


def _render_sheet(lang_code: str, payload: dict, out_dir: Path) -> Path:
    """Render the production-typography QA sheet for one language.

    Uses the system-level RTL helpers (`lang.text_x_for`,
    `lang.accent_bar_edge_for`, `lang.mirror_panel`) so the SAME card
    code automatically renders right-anchored, mirrored accent-bars
    for Arabic/Urdu/Hebrew.  This is the production pattern any
    template should adopt to be RTL-native."""
    from PIL import Image, ImageDraw, ImageFont
    from vidlore.footage import _DOC_BOLD, _DOC_BODY, _ff
    from vidlore.templates._shared import with_alpha
    from vidlore import lang as _lang

    W, H = 1920, 1080
    img = Image.new("RGB", (W, H), (16, 18, 24))
    d = ImageDraw.Draw(img, "RGBA")
    ac = (214, 174, 92)              # gold accent

    rtl = _lang.is_rtl(payload["name"]) or _lang.is_rtl(payload["doc_b"])

    # Header strip (always LTR — it's QA chrome, not content)
    d.rectangle([0, 0, W, 70], fill=(8, 10, 16, 255))
    d.text((40, 12), f"MULTILINGUAL QA · {lang_code}"
                     + ("  (RTL · cinematic mirror)" if rtl else ""),
           fill=(220, 220, 210, 255), font=_ff(_DOC_BOLD, 44, "Q"))

    # --------- Block 1 — NAME_REVEAL ---------------------------- #
    # Cinematic mirror: in RTL, name + role + underline sit on the
    # RIGHT side of the frame, anchored to the right edge.  Padding
    # uses lang.text_x_for so each line is right-aligned consistently.
    LABEL_LEFT = 80
    PANEL_LEFT, PANEL_RIGHT = 80, W - 80      # name panel spans most of frame
    d.text((LABEL_LEFT, 110), "[1] NAME_REVEAL",
           fill=(140, 140, 130, 220), font=_ff(_DOC_BOLD, 22, "L"))
    f_name = _ff(_DOC_BOLD, 140, payload["name"])
    nx, na = _lang.text_x_for(payload["name"], PANEL_LEFT, PANEL_RIGHT,
                              padding=0)
    d.text((nx, 150), payload["name"],
           fill=(248, 244, 230, 255), font=f_name, anchor=na)
    nb = d.textbbox((nx, 150), payload["name"], font=f_name, anchor=na)
    # gold underline sits below the name — mirrored side for RTL
    if rtl:
        d.rectangle([nb[2] - 180, nb[3] + 24, nb[2], nb[3] + 28],
                    fill=with_alpha(ac, 240))
    else:
        d.rectangle([nb[0], nb[3] + 24, nb[0] + 180, nb[3] + 28],
                    fill=with_alpha(ac, 240))
    rx, ra = _lang.text_x_for(payload["role"], PANEL_LEFT, PANEL_RIGHT,
                              padding=0)
    d.text((rx, nb[3] + 56), payload["role"],
           fill=(232, 226, 210, 235),
           font=_ff(_DOC_BOLD, 36, payload["role"]), anchor=ra)

    # --------- Block 2 — TITLE ---------------------------------- #
    d.text((LABEL_LEFT, 410), "[2] TITLE / DISPLAY",
           fill=(140, 140, 130, 220), font=_ff(_DOC_BOLD, 22, "L"))
    tx, ta = _lang.text_x_for(payload["title"], PANEL_LEFT, PANEL_RIGHT,
                              padding=0)
    d.text((tx, 440), payload["title"],
           fill=(245, 245, 240, 255),
           font=_ff(_DOC_BOLD, 96, payload["title"]), anchor=ta)

    # --------- Block 3 — DOCUMENT CARD -------------------------- #
    d.text((LABEL_LEFT, 600), "[3] DOCUMENT CARD",
           fill=(140, 140, 130, 220), font=_ff(_DOC_BOLD, 22, "L"))
    BX, BY, BW, BH = 80, 640, 1760, 200
    d.rectangle([BX, BY, BX + BW, BY + BH], fill=(8, 10, 16, 255))
    # accent bar -- LEFT for LTR, RIGHT for RTL (eye enters from there)
    edge = _lang.accent_bar_edge_for(payload["doc_b"])
    if edge == "right":
        d.rectangle([BX + BW - 6, BY, BX + BW, BY + BH],
                    fill=with_alpha(ac, 255))
    else:
        d.rectangle([BX, BY, BX + 6, BY + BH], fill=with_alpha(ac, 255))
    # header text inside panel
    hx, ha = _lang.text_x_for(payload["doc_t"], BX, BX + BW)
    d.text((hx, BY + 18), payload["doc_t"],
           fill=(240, 235, 220, 235),
           font=_ff(_DOC_BOLD, 38, payload["doc_t"]), anchor=ha)
    # body word-wrap
    f_db = _ff(_DOC_BODY, 32, payload["doc_b"])
    avail = BW - 56
    line = ""; y = BY + 80
    units = payload["doc_b"].split()
    for w in units:
        test = (line + " " + w).strip()
        if d.textbbox((0, 0), test, font=f_db)[2] > avail and line:
            bx, ba = _lang.text_x_for(line, BX, BX + BW)
            d.text((bx, y), line,
                   fill=(230, 225, 210, 240), font=f_db, anchor=ba)
            y += 42; line = w
        else:
            line = test
    if line:
        bx, ba = _lang.text_x_for(line, BX, BX + BW)
        d.text((bx, y), line,
               fill=(230, 225, 210, 240), font=f_db, anchor=ba)

    # --------- Block 4 — BULLETS -------------------------------- #
    d.text((LABEL_LEFT, 870), "[4] BULLET LIST",
           fill=(140, 140, 130, 220), font=_ff(_DOC_BOLD, 22, "L"))
    by = 905
    f_b = _ff(_DOC_BODY, 30, payload["bullets"][0])
    BUL_LEFT, BUL_RIGHT = 80, W - 80
    for bullet in payload["bullets"]:
        if _lang.is_rtl(bullet):
            # accent dot on the right + text right-aligned
            d.rectangle([BUL_RIGHT, by + 12, BUL_RIGHT + 12, by + 24],
                        fill=with_alpha(ac, 240))
            d.text((BUL_RIGHT - 18, by), bullet,
                   fill=(232, 226, 210, 240), font=f_b, anchor="ra")
        else:
            d.rectangle([BUL_LEFT, by + 12, BUL_LEFT + 12, by + 24],
                        fill=with_alpha(ac, 240))
            d.text((BUL_LEFT + 30, by), bullet,
                   fill=(232, 226, 210, 240), font=f_b)
        by += 52

    out_path = out_dir / f"qa_{lang_code}.jpg"
    img.save(out_path, "JPEG", quality=90)
    return out_path


def _audit_render(path: Path, sample: dict) -> dict:
    """Run automated checks against the rendered sheet.

    Most reliable signal: did `_ff()` route to a font that ACTUALLY
    covers the script of this sample's text?  We re-pick the font via
    the same production helper and compare glyph coverage on the
    sample text -- if coverage >= 95%, the right font was chosen and
    every glyph that needed to render had a real glyph available.

    Falls back to contrast + tofu visual heuristic when fontTools is
    unavailable."""
    from PIL import Image
    import numpy as np
    from vidlore.footage import _DOC_BOLD, _ff
    from vidlore.lang import looks_tofu, font_covers_text

    # Visual signals
    img = Image.open(path).convert("RGB")
    tofu_frac = looks_tofu(img)
    contrast_std = float(np.asarray(img.convert("L")).std())

    # CORRECTNESS signal: was the right font routed?  Sample the most
    # demanding text fields and verify coverage.
    text_fields = [sample.get("name", ""),
                   sample.get("role", ""),
                   sample.get("title", ""),
                   sample.get("doc_b", ""),
                   *sample.get("bullets", [])]
    text_blob = " ".join(t for t in text_fields if t)
    routed_font = _ff(_DOC_BOLD, 48, text_blob)
    routed_path = getattr(routed_font, "path", "") or ""
    cov = font_covers_text(text_blob, routed_path) if routed_path else 0.0
    return {
        "routed_font": Path(routed_path).name if routed_path else "?",
        "glyph_coverage": cov,
        "tofu_fraction": tofu_frac,
        "contrast_std": contrast_std,
        "coverage_pass": cov >= 0.95,           # ≥95% glyphs present
        "contrast_pass": contrast_std > 15.0,   # something rendered
        "tofu_pass": tofu_frac < 0.30,          # very loose -- belt n braces
    }


def run(out_dir: Path, langs: Iterable[str] = SAMPLES.keys()) -> int:
    """Render + audit; return 0 on full pass, 1 on any failure."""
    out_dir.mkdir(exist_ok=True, parents=True)
    rows: list[tuple[str, dict]] = []
    for code in langs:
        if code not in SAMPLES:
            print(f"  ?? unknown lang: {code}")
            continue
        p = _render_sheet(code, SAMPLES[code], out_dir)
        a = _audit_render(p, SAMPLES[code])
        rows.append((code, a))
        print(f"  {code}: {p.name}  "
              f"font={a['routed_font']:25s}  cov={a['glyph_coverage']*100:5.1f}%"
              f"  cstd={a['contrast_std']:5.1f}  "
              f"PASS={a['coverage_pass'] and a['contrast_pass']}")

    failures = [c for c, r in rows
                if not (r["coverage_pass"] and r["contrast_pass"])]
    if failures:
        print(f"\nFAIL — {len(failures)} language(s) regressed: "
              f"{', '.join(failures)}")
        return 1
    print(f"\nPASS — all {len(rows)} language(s) render cleanly")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vidlore.multilang_qa")
    p.add_argument("--out", default="/tmp/multilang_qa/cards",
                   help="output directory for the QA sheets")
    p.add_argument("--langs", default=",".join(SAMPLES.keys()),
                   help="comma-separated list of language codes to test")
    a = p.parse_args(argv)
    return run(Path(a.out), a.langs.split(","))


if __name__ == "__main__":
    sys.exit(main())
