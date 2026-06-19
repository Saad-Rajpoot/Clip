"""DOCUMENTARY MOTION-GRAPHIC TEMPLATE REGISTRY (Path A foundation).

This package is the single source of truth for every "kind" of cinematic
overlay the AI editor can choose for a scene. Each template is a
self-describing module that owns:

  • when to fire   — `llm_rule()` returns the instruction added to the
                     editor-brain prompt teaching the LLM WHEN to pick
                     this kind and what `t`/`b` should contain
  • cap            — `max_per_video` so rare/special panels stay rare
                     (the wholesale dedup + downgrade is applied in
                     script_gen.py's `_apply_graphic_caps`)
  • body required? — `uses_body` (carries the extra graphic_body string)
  • rendering loc  — `render_module` + `assemble_branch` document where
                     the actual PIL pre-render + ffmpeg overlay code
                     currently live (so future migration into this
                     package is mechanical, not archaeological)

The registry is read by:
  • `script_gen.py` to auto-generate the editor-brain prompt's per-kind
    rules and the JSON schema's `k` enum
  • `script_gen.py` to apply the per-kind caps consistently
  • `assemble.py` to dispatch overlay rendering to the right branch
  • `web.py` (future) to expose the template catalogue to the editor UI

To add a NEW template:
  1. Drop a new module here that subclasses `Template`
  2. Register it in `_REGISTRY` below
  3. Implement the rendering in footage.py + assemble.py and point
     `render_module`/`assemble_branch` at it
  4. The LLM prompt updates automatically next render — no code touch
     in script_gen.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Template:
    """Self-describing documentary template.

    The fields are deliberately DATA — not abstract methods — so the
    registry can be inspected, serialised (for the future timeline
    editor) and iterated without instantiation. Rendering code stays
    in footage.py / assemble.py and is pointed at by string locators
    (`render_module`, `assemble_branch`) so the migration into this
    package can be done one template at a time without breaking the
    live pipeline.
    """
    name: str                          # graphic_kind value emitted by LLM
    title: str                         # human label (for UI / docs)
    description: str                   # what the template does visually
    llm_rule: str                      # added to editor-brain prompt
    uses_body: bool = False            # carries scene.graphic_body?
    max_per_video: Optional[int] = None
    # ---- variation / pacing hints ---------------------------------- #
    min_gap_scenes: int = 0            # don't repeat within N scenes
    # ---- rendering locator (until migrated into this package) ------ #
    render_module: str = ""            # e.g. "footage.build_graphic_images"
    assemble_branch: str = ""          # e.g. "assemble._graphic_card_filters:ranking"


# =========================================================================
# CATALOGUE — every kind currently in the pipeline + their LLM rules.
# Keeping the rule text HERE means the editor-brain prompt is generated
# from this registry, so adding a kind here automatically teaches the
# LLM about it without editing script_gen.py.
# =========================================================================

_T = Template       # alias for brevity

_TEMPLATES: list[Template] = [
    # 1) RANKING — v5 multi-card carousel (perspective + zoom progression)
    _T(
        name="ranking",
        title="Ranking Carousel",
        description=(
            "Top-N countdown carousel: focused card on dark grid bg + "
            "smaller dimmed prev/next cards in perspective. Card size "
            "grows toward #1 (camera push-in baked in)."
        ),
        llm_rule=(
            "ranking -> narration introduces ONE entry of a ranked list / "
            "countdown / Top-N reveal ('coming in at number four', "
            "'our #2 pick'). t = rank label (just the NUMBER or '#N', "
            "e.g. '4', '#2'); b = the item's SHORT title in CAPS "
            "(<=40 chars, e.g. 'MOUNT RAINIER WILDFLOWER MEADOW'). "
            "Use ONLY for genuine ranked countdown content. Max 5/video."
        ),
        uses_body=True, max_per_video=5, min_gap_scenes=1,
        render_module="footage._render_carousel_card",
        assemble_branch="assemble._graphic_card_filters:ranking",
    ),

    # 2) LOCATION — title-bar "PLACE · ERA" reveal (lower-third style)
    _T(
        name="location",
        title="Location / Era Reveal",
        description=(
            "Lower-third style title bar revealing 'PLACE · ERA' as "
            "the narration anchors a real time and place."
        ),
        llm_rule=(
            "location -> a real place/era anchor ('LANCASTER, PA · "
            "1923', 'AKROTIRI, SANTORINI · NEOLITHIC'). t = the "
            "place+era line in CAPS; b = ''. Use once per genuine "
            "location anchor — vary placement scene-to-scene."
        ),
        uses_body=False, max_per_video=None, min_gap_scenes=2,
        assemble_branch="assemble._graphic_card_filters:location",
    ),

    # 3) NUMBER — single hero figure with count-up + accent glow
    _T(
        name="number",
        title="Number / Year Reveal",
        description=(
            "Single striking figure (year, price, multiplier) the "
            "scene LANDS on. Rapid count-up animation + accent glow."
        ),
        llm_rule=(
            "number -> a single standalone number/year/price/multiplier "
            "the line lands on (t = '1970', '$20', '3X'; b = ''). Use "
            "when ONE figure carries the line's punch."
        ),
        uses_body=False, max_per_video=1, min_gap_scenes=12,
        assemble_branch="assemble._graphic_card_filters:number",
    ),

    # 4) STAT — same as number but different LLM intent (kept for compat)
    _T(
        name="stat",
        title="Statistic Reveal",
        description="Alias of number — single stat with count-up.",
        llm_rule=(
            "stat -> a standalone statistic (synonym for number). "
            "Same format as number."
        ),
        uses_body=False, max_per_video=1, min_gap_scenes=12,
        assemble_branch="assemble._graphic_card_filters:number",
    ),

    # 5) CHART — animated bar + growing figure (proportion / out-of-X)
    _T(
        name="chart",
        title="Animated Bar / Proportion",
        description=(
            "Animated vertical bar that GROWS to its value + the "
            "figure overlaid (47%, 9-out-of-10, half). For "
            "proportional claims, not single figures."
        ),
        llm_rule=(
            "chart -> the line states a proportion/percentage/'half'/"
            "'most'/'X out of Y' (t = the figure '47%'; b = SHORT "
            "uppercase context, e.g. 'OF U.S. CROPS')."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=3,
        assemble_branch="assemble._graphic_card_filters:chart",
    ),

    # 6) DOCUMENT — realistic article / report page with highlight trace
    _T(
        name="document",
        title="Evidence / Article / Report Page",
        description=(
            "Realistic documentary ARTICLE / report page (plain light "
            "paper, dense left-aligned article text, a Source line) with a "
            "yellow highlighter that TRACES across the key phrase."
        ),
        llm_rule=(
            "document -> narration cites a report/article/letter/study/"
            "record/field survey/'the entry reads'. ALWAYS choose this when "
            "the line uses an evidence-citing phrase — 'this document shows', "
            "'according to', 'the report states/found/concluded', 'records "
            "show', 'the file reads', 'the memo said', 'investigators found' "
            "— because the renderer animates a yellow highlighter that TRACES "
            "the key phrase on the exact word the narrator says it (IMP_009), "
            "which only pays off on a real citation. t = SHORT source line "
            "(e.g. 'Penn State Extension field reports, 1923'); b = a "
            "200-420 char excerpt written as REAL ARTICLE PROSE — 2 short "
            "paragraphs, and INCLUDE a concrete figure/percentage (e.g. "
            "'a 90% drop', 'yields 15-25% higher') so the highlighter has a "
            "strong phrase to mark."
        ),
        uses_body=True, max_per_video=None, min_gap_scenes=2,
        render_module="footage.build_graphic_images:document",
        assemble_branch="assemble._graphic_card_filters:document",
    ),

    # 7) EXPLAINER — clean concept card (image + title + caption)
    _T(
        name="explainer",
        title="Concept Explainer Card",
        description=(
            "Premium 'what is this' card: linen background + photo "
            "panel slides in + title drops + caption rises."
        ),
        llm_rule=(
            "explainer -> narration INTRODUCES an important object / "
            "tool / structure / invention / concept the viewer needs "
            "clarified. t = the THING'S NAME (e.g. 'BADGIRS'); b = "
            "ONE short plain caption that clarifies it."
        ),
        uses_body=True, max_per_video=None, min_gap_scenes=2,
        render_module="footage.build_graphic_images:explainer",
        assemble_branch="assemble._graphic_card_filters:explainer",
    ),

    # 8) EVIDENCE — split-screen panel with sequential bullets
    _T(
        name="evidence",
        title="Split-Screen Evidence Panel",
        description=(
            "Left half footage / right half white panel with headline "
            "+ 3-5 bullets that fade in one by one."
        ),
        llm_rule=(
            "evidence -> narration delivers MULTIPLE proof points / a "
            "trust-building list / 3-5 key reasons. t = headline "
            "question/label CAPS (<=30 chars, 'WHY TRUST THIS LIST?'); "
            "b = 3-5 short bullets separated by '|' (each <=50 chars). "
            "Use SPARINGLY: max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        assemble_branch="assemble._graphic_card_filters:evidence",
    ),

    # 9) CALLOUT — annotated text chip pointing at the subject
    _T(
        name="callout",
        title="Callout Label",
        description="Small text chip floating near the subject.",
        llm_rule=(
            "callout -> identify a specific object/term the camera is "
            "looking at. t = the label, SHORT (<=24 chars). Vary "
            "placement (4 corners rotate)."
        ),
        uses_body=False, max_per_video=None, min_gap_scenes=2,
        assemble_branch="assemble._graphic_card_filters:callout",
    ),

    # 9g) PROGRESS BAR — vertical % thermometer (fill-rise + count-up)
    _T(
        name="progress_bar",
        title="Vertical % Bar",
        description=(
            "A single-metric vertical GOLD bar that fills bottom->top with "
            "the % counting up at the fill level + a small header — the "
            "faceless-doc 'this worked N% of the time' stat beat."
        ),
        llm_rule=(
            "progress_bar -> ONE headline percentage the narration states "
            "(a success/reduction/share rate). t = a SHORT header label "
            "(<=40 chars, e.g. 'COPPER BARRIERS STOP CRAWLERS'); b = the "
            "percent 0-100 (e.g. '96'). Use for a single punchy %, not a "
            "comparison (use vertical_bar for 2+). Max 3/video, min_gap 4."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_progress_bar_card",
        assemble_branch="assemble._graphic_card_filters:progress_bar",
    ),

    # 9f) LINE CHART — measurement curve with a labelled x-axis
    _T(
        name="line_chart",
        title="Line / Area Chart",
        description=(
            "Premium navy data-card line/area chart with a labelled "
            "x-axis, value dots + tags and a soft accent area fill — the "
            "'this measurement rose then fell' science beat."
        ),
        llm_rule=(
            "line_chart -> the narration describes a TREND / curve over an "
            "ordered axis (depth, time, distance, dose). t = the chart "
            "title (<=44 chars, e.g. 'CONDUCTIVITY VS DEPTH'); b = 2-8 "
            "points as 'XLABEL|VALUE;...' (e.g. '0 CM|2;5 CM|9;10 CM|15;"
            "15 CM|7'). Max 2/video, min_gap 6."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_line_chart_card",
        assemble_branch="assemble._graphic_card_filters:line_chart",
    ),

    # 9e) SECTION TITLE — clean act/section header over footage
    _T(
        name="section_title",
        title="Section Title (over footage)",
        description=(
            "A clean centred section/act title on a soft centre-band scrim "
            "over the live footage — thin accent rules + optional small "
            "kicker, NO 'PRESENTS / ACT II' chrome. The dignified "
            "'we are entering the next part' beat."
        ),
        llm_rule=(
            "section_title -> the narration crosses into a NEW major part "
            "of the story (the mechanism, the cover-up, the proof, the "
            "legacy). t = the section TITLE in a few words (<=46 chars, "
            "e.g. 'COMPLETING THE CIRCUIT WITH ZINC'); b = optional small "
            "kicker (<=24 chars, e.g. 'PART THREE'). Rare: max 3/video, "
            "min_gap 8."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=15,
        render_module="footage._render_section_title_card",
        assemble_branch="assemble._graphic_card_filters:section_title",
    ),

    # 9d) DIAGRAM LABELS — multi leader-line part labels on the subject
    _T(
        name="diagram_labels",
        title="Labeled Diagram",
        description=(
            "The scene's footage under a soft vignette with 2-4 technical "
            "leader-line labels, each pointing (via the vision model) at a "
            "real part of the subject — the 'here are the components' "
            "science-explainer beat. Each label = chip + elbow leader + "
            "target dot, revealed one-by-one."
        ),
        llm_rule=(
            "diagram_labels -> the footage shows ONE object with several "
            "nameable PARTS the narration walks through (a setup, an "
            "apparatus, an anatomy). t = the 2-4 part names the camera can "
            "see, separated by '|' (each <=22 chars, e.g. 'COPPER PENNY|"
            "GALV NAIL|RAINWATER'). Only when the parts are literally "
            "visible in the shot. Max 3/video, min_gap 4."
        ),
        uses_body=False, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_diagram_labels_card",
        assemble_branch="assemble._graphic_card_filters:diagram_labels",
    ),

    # 9c) STATEMENT — light airy big-text narration card
    _T(
        name="statement",
        title="Big Statement (light)",
        description=(
            "A near full-screen bold UPPERCASE narration line set as a "
            "paragraph on a bright, airy radial-grey field, one keyword "
            "underlined in the accent, revealed line-by-line. The calm "
            "'let this sentence land' beat that breaks up the dark cards "
            "(faceless-doc light-card look)."
        ),
        llm_rule=(
            "statement -> a PIVOTAL narration sentence you want to land as "
            "a calm full-screen text beat (a thesis, a reveal, a stakes "
            "line) — NOT a quote from a person (use quote_highlight for "
            "that). t = the sentence itself (<=160 chars) — OR, for a hard "
            "word-STAB on the single biggest reveal, just 1-2 words / a name "
            "/ a number (e.g. 'BETRAYAL', '$400 BILLION', '1971'); b = "
            "optional ONE keyword in t to underline. VERY rare + deliberate: "
            "max 2/video, strong cooldown — reserve it for only the 1-2 most "
            "important reveal beats. Never a running caption, never "
            "repetitive, never a TikTok text-slam."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=8,
        render_module="footage._render_statement_card",
        assemble_branch="assemble._graphic_card_filters:statement",
    ),

    # 9b) FRAMED INSERT — tilted gold-framed footage card + caption chip
    _T(
        name="framed_insert",
        title="Annotated Framed Insert",
        description=(
            "A piece of the scene's footage shown in a slightly-tilted "
            "rounded card with a thin GOLD border + corner brackets, "
            "floating on a dark vignette field, with a black caption chip "
            "(gold top-accent) and angular gold bracket leader lines. The "
            "card scales in, leaders draw on, chip drops — the faceless-doc "
            "'here is the specific clip, and here's what it shows' beat."
        ),
        llm_rule=(
            "framed_insert -> a MULTI-POINT documentary explainer built "
            "around the scene's footage as the visual anchor. t = 2-4 SHORT "
            "insight labels separated by ' | ' (a pipe). Each label is a "
            "MEANINGFUL insight about a different aspect of what is shown "
            "(<=34 chars each, sentence case), NOT a generic caption. They "
            "are distributed around the image (left / right / bottom) and "
            "joined to it by connector lines. Example t: 'Copper strip "
            "pressed into soil | Creates a weak earth-battery | Slugs avoid "
            "the charged zone'. Use for a 'let me explain this image' beat. "
            "Max 4/video, min_gap 4."
        ),
        uses_body=False, max_per_video=4, min_gap_scenes=4,
        render_module="footage._render_framed_insert_card",
        assemble_branch="assemble._graphic_card_filters:framed_insert",
    ),

    # 10) PORTRAIT — cinematic full-screen photo + giant name
    _T(
        name="portrait",
        title="Portrait Card",
        description=(
            "Full-screen cinematic photo of a person + giant name "
            "lower-third for biographical introductions."
        ),
        llm_rule=(
            "portrait -> narration names a specific historical figure "
            "for the first time. t = the person's name. Requires "
            "fal.ai key for the AI photo."
        ),
        uses_body=False, max_per_video=None, min_gap_scenes=3,
        render_module="footage.build_graphic_images:portrait",
        assemble_branch="assemble._graphic_card_filters:portrait",
    ),

    # 11) LABEL — minimal text label for a defined term
    _T(
        name="label",
        title="Term Label",
        description="One-word/phrase label for a key term being defined.",
        llm_rule=(
            "label -> a key term FIRST introduced. t = the term in "
            "CAPS (b = '')."
        ),
        uses_body=False, max_per_video=None, min_gap_scenes=2,
        assemble_branch="assemble._graphic_card_filters:label",
    ),

    # ---- NEW templates added by the v5+ system (one-by-one) -------- #

    # 12) NEWSPAPER — print-edition front-page headline
    _T(
        name="newspaper",
        title="Newspaper Headline Reveal",
        description=(
            "Print-newspaper front page: cream paper backdrop with "
            "subtle texture, black masthead band, giant serif "
            "headline, subhead, dateline. Print-record cousin of "
            "news_broadcast (which is TV). For 'the headlines "
            "said…' / historical front-page beats."
        ),
        llm_rule=(
            "newspaper -> narration cites a print newspaper / "
            "front-page story / historical headline / 'the morning "
            "papers read…'. t = 'MASTHEAD :: HEADLINE' (e.g. 'THE "
            "TIMES :: NIXON RESIGNS', 'THE GUARDIAN :: BERLIN WALL "
            "FALLS'); b = 'SUBHEAD || DATELINE' (both optional, "
            "either can be empty). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_newspaper_card",
        assemble_branch="assemble._graphic_card_filters:newspaper",
    ),

    # 13) TERMINAL — hacker/intel green-on-black CRT
    _T(
        name="terminal",
        title="Hacker / Terminal",
        description=(
            "Green-on-black retro terminal window: title bar in "
            "theme accent + 3-dot controls + monospace prompt lines "
            "+ scanlines + phosphor glow. For hacker/intel/data-"
            "mining narrative beats."
        ),
        llm_rule=(
            "terminal -> narration invokes HACKING / data leak / "
            "intelligence feed / breach / surveillance log / "
            "computer-generated report. t = window title CAPS "
            "(<=40 chars, e.g. 'NSA INTEL FEED', 'BREACH LOG'); "
            "b = the body text shown as prompt lines (each line "
            "<=60 chars; up to 12 lines). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_terminal_card",
        assemble_branch="assemble._graphic_card_filters:terminal",
    ),

    # 14) MILITARY HUD — tactical viewfinder + readouts
    _T(
        name="military_hud",
        title="Military HUD",
        description=(
            "Tactical heads-up display overlay: corner brackets + "
            "sparse grid + centre crosshair reticle + side data "
            "readouts (op name, coords, status). Transparent middle "
            "so footage shows through under the reticle."
        ),
        llm_rule=(
            "military_hud -> narration invokes a MILITARY / "
            "tactical / operation moment (raid, mission, target "
            "acquisition). t = 'OP_NAME :: COORDS' (e.g. "
            "'NEPTUNE SPEAR :: 34.169°N 73.244°E'); b = mission "
            "status text (<=80 chars, e.g. 'TARGET CONFIRMED · "
            "EXTRACTION INBOUND'). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_military_hud_card",
        assemble_branch="assemble._graphic_card_filters:military_hud",
    ),

    # 15) QUOTE HIGHLIGHT — cinematic claim emphasis
    _T(
        name="quote_highlight",
        title="Quote Highlight",
        description=(
            "Cinematic typography quote: giant decorative quote "
            "glyph, accent vertical bar, large serif text, "
            "attribution under em-dash. Soft horizontal-band "
            "vignette so it reads over any footage."
        ),
        llm_rule=(
            "quote_highlight -> narration directly quotes someone "
            "or lands a SHOCKING / EMOTIONAL claim that deserves "
            "type emphasis. t = the quote text (<=200 chars, NO "
            "outer quotation marks — they're drawn for you); "
            "b = attribution (e.g. 'GEORGE ORWELL', 'CIA REPORT "
            "1972'). Use SPARINGLY — max 3/video — and only "
            "when the claim genuinely lands as a moment."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_quote_highlight_card",
        assemble_branch="assemble._graphic_card_filters:quote_highlight",
    ),

    # 13) BULLET LIST — standalone full-width bullet column
    _T(
        name="bullet_list",
        title="Animated Bullet List",
        description=(
            "Standalone bullet column (NOT the evidence split-"
            "screen): 3-5 short bullets centred over soft vignette. "
            "Two auto-rotating sub-layouts (left-aligned vs centred) "
            "so adjacent bullet-list scenes don't look identical."
        ),
        llm_rule=(
            "bullet_list -> narration LANDS 3-5 short key facts / "
            "findings / steps that the viewer should see together "
            "but DON'T need split-screen footage beside them (use "
            "evidence for trust-building bulleted panels with "
            "footage). t = optional headline CAPS (<=30 chars); "
            "b = 3-5 short bullets separated by '|', each <=50 "
            "chars. Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_bullet_list_card",
        assemble_branch="assemble._graphic_card_filters:bullet_list",
    ),

    # 14) TYPING DATE — archive-style timestamp overlay
    _T(
        name="typing_date",
        title="Typing Date Overlay",
        description=(
            "Bottom-left archive-style timestamp with monospace "
            "date + blinking cursor + optional context line. "
            "Archive-terminal vibe for historical date anchors."
        ),
        llm_rule=(
            "typing_date -> narration anchors a SPECIFIC date / "
            "archive timestamp that the viewer benefits from seeing "
            "on screen (different from `location` which combines "
            "place+era). t = date string (e.g. '14 MARCH 1987', "
            "'Q3 2008', 'AUTUMN 1547'); b = optional context "
            "label (e.g. 'EMBASSY CABLE No. 0042'). Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=3,
        render_module="footage._render_typing_date_card",
        assemble_branch="assemble._graphic_card_filters:typing_date",
    ),

    # 15) LOWER THIRD — dynamic system with 4 auto-rotating variants
    _T(
        name="lower_third",
        title="Dynamic Lower Third System",
        description=(
            "Documentary lower-third with FOUR layout variants that "
            "auto-rotate across scenes (bottom-left classic, "
            "bottom-centre banner, bottom-right mirror, top-left "
            "tag). All share the same fade/drift DNA from "
            "_shared.py — same fonts, same accent, same pacing — "
            "so they read as ONE system at different positions. "
            "The 4-variant rotation prevents the every-other-scene "
            "repetition that makes amateur docs feel templated."
        ),
        llm_rule=(
            "lower_third -> label a SPECIFIC place + era, an "
            "already-known person's role/credential tag, an "
            "expert/interviewee caption, or a chapter section. "
            "t = 'TITLE :: SUBTITLE' (e.g. 'TENOCHTITLAN :: "
            "MEXICO CITY · 1325', 'DR. ANNA REIMERS :: "
            "ARCHAEOLOGIST, OXFORD'); b = optional extra context "
            "(used when t has no '::'). DO NOT use lower_third for "
            "the FIRST on-screen introduction of a real individual "
            "whose face / footage we are seeing — that beat MUST "
            "use name_reveal (the big cinematic name). lower_third "
            "is the small caption for places, sections, and "
            "secondary credential tags. Placement auto-rotates so "
            "the video never stacks two identical positions. "
            "min_gap 2."
        ),
        uses_body=True, max_per_video=None, min_gap_scenes=2,
        render_module="footage._render_lower_third_card",
        assemble_branch="assemble._graphic_card_filters:lower_third",
    ),

    # 12b) NAME REVEAL — BIG character-name introduction (MagnatesMedia)
    _T(
        name="name_reveal",
        title="Character Name Reveal",
        description=(
            "A BIG editorial NAME that slides in beside a person while "
            "their live footage plays (not a small lower-third) — the "
            "MagnatesMedia 'meet this character' intro. Huge bold name "
            "on a soft left scrim, theme-accent bar, small role/place "
            "line beneath. Assembler gives it an eased slide-in + fade."
        ),
        llm_rule=(
            "name_reveal -> ALWAYS use this (NOT lower_third) the FIRST "
            "time a KEY NAMED PERSON appears on screen / in footage — the "
            "'meet this character' beat. If a scene shows a person and the "
            "narration names them, this is the correct kind. t = THE NAME "
            "in CAPS (<=20 chars, 1-3 words, e.g. 'ELI YODER', "
            "'EDWARD BERNAYS'); b = a SHORT role / place line (<=26 chars "
            "CAPS, e.g. 'AMISH ELDER', 'PROPAGANDA PIONEER', "
            "'MILLERSBURG, OHIO'). Big, cinematic. Max 3/video, min_gap 3."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=3,
        render_module="footage._render_name_reveal_card",
        assemble_branch="assemble._graphic_card_filters:name_reveal",
    ),

    # 12b) FIGURE LOCATOR — portrait tethered to its place on a map (IMP_021)
    _T(
        name="figure_locator",
        title="Figure → Place Tether",
        description=(
            "A 'who-controls-where' beat: a small framed portrait badge "
            "(with a name caption) in the corner, a thin leader line to a "
            "marker on the REAL map of the place the person is tied to. "
            "Subtle + cinematic — the map breathes, one connector, never a "
            "stacked clutter. Assembler fades the composite in/out."
        ),
        llm_rule=(
            "figure_locator -> use SPARINGLY (max 1/video) ONLY when a scene "
            "introduces a KEY PERSON whose link to a SPECIFIC PLACE / "
            "territory IS the point (a leader and the country/city they "
            "ruled or operated from). t = PERSON NAME in CAPS (<=20 chars, "
            "e.g. 'SLOBODAN MILOSEVIC'); b = the geocodable PLACE NAME "
            "(e.g. 'Belgrade, Serbia', 'Baghdad, Iraq'). For a plain person "
            "intro use name_reveal instead; for plain geography use a map. "
            "Only when person<->place is the story. min_gap 8."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=8,
        render_module="footage.build_graphic_images:figure_locator",
        assemble_branch="assemble._graphic_card_filters:figure_locator",
    ),

    # 13) SURVEILLANCE — CCTV viewfinder overlay
    _T(
        name="surveillance",
        title="CCTV / Surveillance",
        description=(
            "Transparent CCTV-aesthetic frame: REC indicator + "
            "camera ID + location + timestamp + centre crosshair "
            "+ corner L-marks + faint scanlines + vignette. Turns "
            "footage into a surveillance feed for 'they were "
            "watching' / 'caught on camera' / observation moments."
        ),
        llm_rule=(
            "surveillance -> narration invokes SURVEILLANCE / "
            "CCTV footage / observation / 'caught on camera' / "
            "'they were watched'. t = 'CAM_ID :: LOCATION' "
            "(e.g. '04 :: WAREHOUSE — EAST GATE', '12 :: "
            "EMBASSY LOBBY'); b = timestamp string ('23:47:12 "
            "· 14-MAR-1987'). Heavy stylisation — use rarely "
            "for genuine surveillance beats. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_surveillance_card",
        assemble_branch="assemble._graphic_card_filters:surveillance",
    ),

    # 13) NEWS BROADCAST — TV news lower-third + LIVE indicator
    _T(
        name="news_broadcast",
        title="News Broadcast Layout",
        description=(
            "TV news overlay: LIVE indicator + channel strip top-"
            "left, lower-third HEADLINE bar with theme accent "
            "stripe, optional BREAKING ticker below, optional "
            "REPORTER name above. Transparent middle — footage "
            "shows through, only the bars are opaque."
        ),
        llm_rule=(
            "news_broadcast -> narration explicitly cites NEWS "
            "REPORTING / a press story / 'as reported by...' / "
            "'breaking news' / 'the headline read'. t = 'CHANNEL "
            ":: HEADLINE' (e.g. 'CNN :: 47 DEAD IN MOUNTAIN "
            "AVALANCHE', 'BBC :: WHISTLEBLOWER GOES PUBLIC'); b "
            "= 'TICKER_TEXT || REPORTER_NAME' (both optional, "
            "either can be empty). Use when narration genuinely "
            "invokes broadcast journalism. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=4,
        render_module="footage._render_news_broadcast_card",
        assemble_branch="assemble._graphic_card_filters:news_broadcast",
    ),

    # 13) CLASSIFIED — manila-folder dossier reveal
    _T(
        name="classified",
        title="Classified Dossier Reveal",
        description=(
            "Manila folder backdrop with classified-document body: "
            "case-no tab, dark agency header, red CLASSIFIED stamp "
            "at angle, typewritten body. Use for intelligence/"
            "government/secret-investigation moments (visually "
            "distinct from `document` which is archival paper)."
        ),
        llm_rule=(
            "classified -> narration cites a SECRET intelligence "
            "file / leaked dossier / restricted government document "
            "/ confidential brief (NOT an academic record — use "
            "document for those). t = 'FILE_NO :: SHORT TITLE' "
            "(e.g. '7595-DA :: PROJECT BLUEBOOK', '0042-K :: "
            "OPERATION GLADIO'); b = a <=300-char excerpt as if "
            "quoting the file. Use for conspiracy/crime/spy "
            "narratives. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=4,
        render_module="footage._render_classified_dossier_card",
        assemble_branch="assemble._graphic_card_filters:classified",
    ),

    # 13) PHOTO COLLAGE — pinned-photos evidence wall
    _T(
        name="photo_collage",
        title="Photo Collage / Evidence Wall",
        description=(
            "Investigation-board layout: 3-5 footage frames pinned "
            "to a dark wall as Polaroid-style cards with slight "
            "rotation, red pin dots, drop shadows, optional red "
            "string connectors between them. Investigative "
            "documentary aesthetic."
        ),
        llm_rule=(
            "photo_collage -> narration references a COLLECTION of "
            "related people/objects/locations the viewer should see "
            "TOGETHER (the suspects, the victims, the discoveries, "
            "the evidence, the photo lineup). t = headline label "
            "CAPS (<=30 chars, e.g. 'THE SUSPECTS', 'WHAT WE "
            "FOUND'); b = OPTIONAL pipe-separated captions for "
            "each photo (e.g. 'CARTER 1922|BINGHAM 1911|EVANS "
            "1900|SCHLIEMANN 1873'). NEVER for a single subject "
            "(use portrait/explainer instead). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_photo_collage_card",
        assemble_branch="assemble._graphic_card_filters:photo_collage",
    ),

    # 13) STAT DASHBOARD — multi-stat infographic (2-4 numbers)
    _T(
        name="stat_dashboard",
        title="Statistic Dashboard",
        description=(
            "Documentary infographic: 2-4 stat cards laid out in a "
            "grid (row for 2-3, 2x2 for 4), each with a BIG number "
            "+ small label. Same dark navy + grid design family as "
            "timeline/map/comparison."
        ),
        llm_rule=(
            "stat_dashboard -> narration delivers MULTIPLE related "
            "numbers in the same beat that the viewer should see "
            "TOGETHER (a stats panel, not separate scenes). t = "
            "headline CAPS (<=30 chars, e.g. 'BY THE NUMBERS', "
            "'CASE FACTS'); b = 2-4 entries separated by ';' with "
            "LABEL|VALUE pairs (label <=20 chars CAPS, value <=14 "
            "chars), e.g. 'YEARS BURNING|52;DEPTH|70 m;TEMPERATURE"
            "|1000°C;TOURISTS/YR|10K'. Use when a SET of numbers "
            "lands together — single figures still use number. "
            "Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=4,
        render_module="footage._render_stat_dashboard_card",
        assemble_branch="assemble._graphic_card_filters:stat_dashboard",
    ),

    # 13) COMPARISON — two-column versus layout with VS badge
    _T(
        name="comparison",
        title="Comparison Layout",
        description=(
            "Side-by-side comparison panel: left column (BEFORE/A/"
            "OLD), right column (AFTER/B/NEW), central VS badge in "
            "theme accent. Each side carries a SMALL label on top "
            "and a LARGE value below. Same dark navy + grid design "
            "family as timeline/map_reveal."
        ),
        llm_rule=(
            "comparison -> narration explicitly contrasts TWO "
            "things on the same dimension: before/after, then/now, "
            "method A vs method B, old/new, with/without. t = "
            "headline label CAPS (<=32 chars, e.g. 'BEFORE VS "
            "AFTER', '1980 VS 2024'); b = "
            "'LEFT_LABEL|LEFT_VALUE;;RIGHT_LABEL|RIGHT_VALUE' (the "
            "';;' separates the two sides; '|' separates the small "
            "label from the big value within each side). Examples: "
            "'1980|92% FOREST COVER;;2024|34% FOREST COVER' / "
            "'BEFORE|EXTINCT IN WILD;;AFTER|2000 RESTORED'. Use "
            "ONLY when the contrast is the line's whole point — "
            "single-side facts use number/document instead. "
            "Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=4,
        render_module="footage._render_comparison_card",
        assemble_branch="assemble._graphic_card_filters:comparison",
    ),

    # 13) TIMELINE SEQUENCE — horizontal axis + dated markers
    _T(
        name="timeline",
        title="Timeline Sequence",
        description=(
            "Documentary timeline graphic: horizontal axis across "
            "dark navy bg, 3-5 evenly-spaced theme-accent markers, "
            "dates BELOW each marker, event labels ABOVE (alternating "
            "sides). Same design language as map_reveal — viewer "
            "groups them as the doc's stylised data views."
        ),
        llm_rule=(
            "timeline -> narration walks the viewer through DATED "
            "events / historical progression / step-by-step "
            "chronology with 3-5 specific dates that should appear "
            "TOGETHER on screen so the arc is visible at a glance. "
            "t = headline in CAPS (<=24 chars, e.g. 'KEY EVENTS', "
            "'THE PROGRESSION'); b = entries separated by ';' with "
            "DATE|LABEL pairs (date <=8 chars, label <=40 chars), "
            "e.g. '1922|HOWARD CARTER OPENS TOMB;1939|CHAMBER 5 "
            "BREACHED;1978|RESTORATION BEGINS'. Use ONCE per "
            "video for a genuine dated arc — don't fire for single "
            "dates (use number for those). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_timeline_card",
        assemble_branch="assemble._graphic_card_filters:timeline",
    ),

    # 13) MAP REVEAL — animated map zoom with location pin/route
    _T(
        name="map_reveal",
        title="Map Reveal",
        description=(
            "REAL satellite/political map of the place + cinematic "
            "zoom-in + pin drop with bounce + pulse ring + sliding "
            "label (with coordinates). Real geographic imagery."
        ),
        llm_rule=(
            "map_reveal -> narration names a SPECIFIC geographic place "
            "(city, country, region, coordinates) the viewer benefits "
            "from seeing on a REAL map — also fire for journeys, "
            "migrations, trade routes, military movements, supply "
            "chains, or 'where in the world' moments. t = the place "
            "name exactly as it should label the map (e.g. 'Berlin, "
            "Germany' or 'Lancaster County, PA'). b = the place's "
            "latitude,longitude when you know it (e.g. '52.52,13.40') "
            "— STRONGLY PREFERRED for accuracy; leave '' only if "
            "unknown (it will be geocoded). Use ONCE per genuinely new "
            "geography, max 3/video — not for every passing name."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=3,
        render_module="footage._render_map_reveal_card",
        assemble_branch="assemble._graphic_card_filters:map_reveal",
    ),

    # 15) COUNTDOWN / URGENCY — corner digital clock w/ red pulse
    _T(
        name="countdown",
        title="Countdown / Urgency",
        description=(
            "Bottom-right digital countdown readout: small URGENT "
            "label, big LCD-style time/days remaining figure with "
            "red pulse glow behind it, optional caption strip below "
            "(what the deadline is). Transparent frame overlay — "
            "composites OVER footage so the urgency lands WITH the "
            "visual, not blocking it. For 'time is running out' / "
            "'they had N hours' / 'T-minus' beats."
        ),
        llm_rule=(
            "countdown -> narration delivers a CONCRETE TIME-LIMITED "
            "stakes line: 'they had 12 hours to evacuate', 'three "
            "days until the avalanche', 'T-minus 47 seconds', "
            "'48 hours later'. t = label CAPS (<=18 chars, e.g. "
            "'TIME TO IMPACT', 'EVACUATION', 'T-MINUS'); b = the "
            "time figure (<=14 chars, e.g. '12:00:00', '03 DAYS', "
            "'00:47:23', 'T-3 DAYS') optionally followed by '|' "
            "and a short context phrase (<=28 chars, e.g. "
            "'12 HOURS|TO EVACUATE THE TOWN'). Use ONLY when "
            "narration is explicitly about a deadline/time-limit; "
            "never as decoration. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_countdown_card",
        assemble_branch="assemble._graphic_card_filters:countdown",
    ),

    # 84) SMS / TEXT MESSAGE — phone chat overlay
    _T(
        name="sms_text",
        title="SMS / Text Message",
        description=(
            "Phone screen with chat bubbles (accent for "
            "received, grey for sent) showing 2-4 SMS "
            "messages. For 'the text exchange' / 'leaked DMs' "
            "/ 'what she texted that night' beats."
        ),
        llm_rule=(
            "sms_text -> narration explicitly references a TEXT "
            "MESSAGE EXCHANGE ('the texts read', 'their last "
            "message', 'leaked DMs'). t = 'CONTACT_NAME|TIME' "
            "(contact <=24 chars CAPS; time <=10 chars e.g. "
            "'11:47 PM'); b = pipe-separated messages where each "
            "is 'SIDE|TEXT' — SIDE='IN' (incoming) or 'OUT' "
            "(outgoing); text in normal case. Example: 'OUT|Are "
            "you ok?|IN|I can't talk now.|OUT|Call me as soon as "
            "you can.|IN|Tomorrow.'. Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_sms_text_card",
        assemble_branch="assemble._graphic_card_filters:sms_text",
    ),

    # 85) POSTMARK / POSTAL STAMP — vintage mail
    _T(
        name="postmark",
        title="Postmark / Postal Stamp",
        description=(
            "Vintage postal piece: circular postmark with city + "
            "date over a rectangular postage stamp (with portrait "
            "or scene). For 'postmarked from X' / mail-trail / "
            "'a letter arrived from' beats."
        ),
        llm_rule=(
            "postmark -> narration explicitly references a "
            "POSTMARKED PIECE OF MAIL ('postmarked from', 'a "
            "letter arrived from', 'the envelope was stamped'). "
            "t = 'CITY|DATE' (city <=22 chars CAPS; date <=14 "
            "chars CAPS e.g. 'MAR 12 1971'); b = optional "
            "'STAMP_VALUE[|COUNTRY]' (value <=10 chars e.g. "
            "'1¢'; country <=24 chars CAPS). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_postmark_card",
        assemble_branch="assemble._graphic_card_filters:postmark",
    ),

    # 86) QUOTE STREAM — stacked social-reaction cards
    _T(
        name="quote_stream",
        title="Quote Stream / Reaction Wall",
        description=(
            "3-5 short stacked reaction cards (mini social "
            "posts) — each with handle + quote text. For 'the "
            "reactions came in' / public-outrage compilation / "
            "multiple-voices beats."
        ),
        llm_rule=(
            "quote_stream -> narration explicitly references "
            "MULTIPLE PEOPLE REACTING ('the reactions came in', "
            "'social media erupted', 'commenters said'). t = "
            "headline CAPS (<=30 chars e.g. 'THE RESPONSE'); "
            "b = entries separated by ';' as 'HANDLE|TEXT' "
            "(handle <=18 chars; text normal case <=80 chars). "
            "3-5 entries. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_quote_stream_card",
        assemble_branch="assemble._graphic_card_filters:quote_stream",
    ),

    # 87) MINI TIMELINE — compact horizontal strip
    _T(
        name="mini_timeline",
        title="Mini Timeline (Compact)",
        description=(
            "Compact bottom-screen horizontal timeline strip "
            "with 4-6 small dated markers + accent line. For "
            "inline timeline context — smaller / lighter than "
            "the full timeline template."
        ),
        llm_rule=(
            "mini_timeline -> narration runs through a SHORT "
            "SEQUENCE of dated events in passing ('over the "
            "next few years', 'across the decade'). t = "
            "headline CAPS (<=24 chars e.g. 'THE FIRST DECADE'); "
            "b = 4-6 entries separated by ';' as 'DATE|EVENT' "
            "(date <=6 chars; event <=20 chars CAPS). Example: "
            "'1971|DRILL;1972|FIRE;1980|BURNED;1990|VISITED;"
            "2024|STILL'. Max 2/video. Distinct from `timeline` "
            "(full-frame opaque)."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_mini_timeline_card",
        assemble_branch="assemble._graphic_card_filters:mini_timeline",
    ),

    # 80) CLOCK FACE — analog clock
    _T(
        name="clock_face",
        title="Analog Clock Face",
        description=(
            "Circular analog clock with hour/minute hands "
            "pointing to a specific time + small AM/PM tag + "
            "label below. For 'at exactly 3:47 PM' / "
            "time-stamped event beats."
        ),
        llm_rule=(
            "clock_face -> narration delivers an EXACT TIME OF "
            "DAY for an event ('at exactly 3:47 PM', 'shortly "
            "after midnight', 'six in the morning'). t = "
            "'HH:MM|AM_PM' (HH 00-23, MM 00-59; am_pm 'AM' or "
            "'PM' or empty); b = optional context label "
            "<=44 chars CAPS e.g. 'MOMENT OF IGNITION'. Max "
            "2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_clock_face_card",
        assemble_branch="assemble._graphic_card_filters:clock_face",
    ),

    # 81) NETWORK GRAPH — multi-node web
    _T(
        name="network_graph",
        title="Network Graph",
        description=(
            "Web-like network of 5-7 nodes connected by accent "
            "lines. Each node can connect to multiple others. "
            "Distinct from relationship_tree (hierarchical) — "
            "this is the 'they all knew each other' web."
        ),
        llm_rule=(
            "network_graph -> narration explicitly describes a "
            "MULTI-WAY NETWORK where members connect to multiple "
            "others ('they were all linked', 'the network had "
            "five key nodes', 'mutual connections'). t = "
            "headline CAPS (<=30 chars e.g. 'THE NETWORK'); b = "
            "5-7 nodes separated by ';' as 'NAME[>EDGE,EDGE,...]' "
            "where edges name other nodes (case-insensitive). "
            "Example: 'EMMA>LIAM,NOAH;LIAM>NOAH,AVA;NOAH>AVA;"
            "AVA>EMMA'. Max 1/video."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=10,
        render_module="footage._render_network_graph_card",
        assemble_branch="assemble._graphic_card_filters:network_graph",
    ),

    # 82) STATUS INDICATOR — red/yellow/green stoplight
    _T(
        name="status_indicator",
        title="Status Indicator / Stoplight",
        description=(
            "Three big dot indicators (red / yellow / green) "
            "with one lit + status label below. For "
            "'OPERATIONAL' / 'COMPROMISED' / 'DESTROYED' / "
            "'ACTIVE' / 'INACTIVE' beats."
        ),
        llm_rule=(
            "status_indicator -> narration delivers a BINARY/"
            "TRINARY STATUS state for a named subject "
            "('operational', 'compromised', 'destroyed', "
            "'active/inactive', 'go/no-go'). t = 'STATUS|COLOR' "
            "(status word <=18 chars CAPS e.g. 'OPERATIONAL'; "
            "color 'GREEN' / 'YELLOW' / 'RED'); b = optional "
            "context (<=44 chars CAPS e.g. 'SYSTEM AT 100% "
            "CAPACITY'). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=4,
        render_module="footage._render_status_indicator_card",
        assemble_branch="assemble._graphic_card_filters:status_indicator",
    ),

    # 83) SPEEDOMETER — semi-circle gauge
    _T(
        name="speedometer",
        title="Speedometer / Gauge",
        description=(
            "Semi-circle gauge with needle pointing to a value, "
            "scale arc with MIN / MAX labels + numeric readout + "
            "unit. For 'exceeded 200 km/h' / 'temperature hit "
            "1000 C' / measurement-with-scale beats."
        ),
        llm_rule=(
            "speedometer -> narration delivers a MEASURED VALUE "
            "with a known SCALE/MAX ('exceeded 200 km/h', 'hit "
            "1000 degrees', 'pressure of 95 PSI'). t = "
            "'VALUE|UNIT' (value <=8 chars; unit <=10 chars "
            "e.g. 'KM/H', '°C', 'PSI'); b = 'MIN|MAX[|LABEL]' "
            "(min and max numeric — value should fit between; "
            "optional label <=30 chars CAPS). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_speedometer_card",
        assemble_branch="assemble._graphic_card_filters:speedometer",
    ),

    # 76) FILM SLATE — clapper board
    _T(
        name="film_slate",
        title="Film Slate / Clapper Board",
        description=(
            "Classic film slate (clapper board): striped top "
            "clapper arm + black-board area with PRODUCTION / "
            "SCENE / TAKE / DIRECTOR / DATE fields in chalky "
            "white type. For 'behind the scenes' / 'on set' / "
            "'they reshot the take' beats."
        ),
        llm_rule=(
            "film_slate -> narration is about FILMMAKING or "
            "PRODUCTION ('behind the scenes', 'on set', "
            "'reshooting the take', 'the director called cut'). "
            "t = 'PRODUCTION|DIRECTOR' (production name <=24 "
            "chars CAPS; director <=18 chars CAPS); b = "
            "'SCENE|TAKE|DATE' (scene <=4 CAPS e.g. '04A'; "
            "take <=2 chars CAPS; date <=12 chars). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_film_slate_card",
        assemble_branch="assemble._graphic_card_filters:film_slate",
    ),

    # 77) SPEECH BUBBLE — comic-style speech
    _T(
        name="speech_bubble",
        title="Speech Bubble",
        description=(
            "Comic-book style speech bubble with tail pointing "
            "to a side, centred over footage. For lighter "
            "narrative recreations of dialogue or character "
            "asides — distinct from quote_highlight which is "
            "documentary cinematic."
        ),
        llm_rule=(
            "speech_bubble -> narration playfully recreates "
            "DIALOGUE in a lighter / illustrative tone (NOT a "
            "serious historical quote). t = optional speaker "
            "name (<=24 chars CAPS) or empty; b = the speech "
            "text in normal case (<=140 chars). Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_speech_bubble_card",
        assemble_branch="assemble._graphic_card_filters:speech_bubble",
    ),

    # 78) COMPASS BEARING — navigation overlay
    _T(
        name="compass_bearing",
        title="Compass Bearing",
        description=(
            "Centred compass rose overlay with cardinal "
            "directions + needle pointing to a specific bearing "
            "+ bearing degrees readout. For 'heading north-west' "
            "/ 'bearing 045°' navigation moments."
        ),
        llm_rule=(
            "compass_bearing -> narration explicitly describes a "
            "NAVIGATIONAL HEADING or DIRECTION ('heading north', "
            "'bearing 045 degrees', 'turn south-east'). t = the "
            "bearing label CAPS (<=12 chars e.g. 'NW' / 'N 045°'); "
            "b = numeric bearing 0-360 + optional context "
            "'BEARING|CONTEXT' (bearing 0-360; context <=44 chars "
            "CAPS e.g. 'TOWARD THE CRATER'). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_compass_bearing_card",
        assemble_branch="assemble._graphic_card_filters:compass_bearing",
    ),

    # 79) STAT INSIGHT — big stat + interpretation
    _T(
        name="stat_insight",
        title="Stat Insight Highlight",
        description=(
            "Big stat number centred + accent-highlighted "
            "insight line below ('That's enough to fill 200 "
            "swimming pools'). Single-stat with context "
            "interpretation. Distinct from `number` (which is "
            "just the number)."
        ),
        llm_rule=(
            "stat_insight -> narration delivers a SINGLE STAT "
            "with an INTERPRETIVE comparison line ('which is "
            "enough to', 'the equivalent of', 'for "
            "comparison'). t = the stat figure CAPS (<=14 chars "
            "e.g. '12 MILLION GAL.'); b = the interpretation "
            "line normal case (<=120 chars e.g. 'enough to fill "
            "200 olympic swimming pools'). Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_stat_insight_card",
        assemble_branch="assemble._graphic_card_filters:stat_insight",
    ),

    # 72) ID CARD — passport/ID badge
    _T(
        name="id_card",
        title="ID Card / Badge",
        description=(
            "Passport/ID-style card overlay: photo placeholder + "
            "name + ID number + issuing org + expiry date. "
            "Government-document feel. For 'the ID card showed' / "
            "passport / staff badge reveals."
        ),
        llm_rule=(
            "id_card -> narration explicitly references a NAMED "
            "PERSON'S ID DOCUMENT (passport, drivers license, "
            "staff badge, military ID). t = 'NAME|ID_NUMBER' "
            "(name <=28 chars CAPS; id_number <=16 chars CAPS "
            "e.g. 'PASS-04-7129'); b = 'ISSUER|EXPIRY[|ROLE]' "
            "(issuer <=24 chars CAPS e.g. 'STATE DEPT'; expiry "
            "<=12 chars e.g. '12/2025'; optional role <=20 chars "
            "CAPS). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_id_card_card",
        assemble_branch="assemble._graphic_card_filters:id_card",
    ),

    # 73) EMAIL SCREENSHOT — email-client UI overlay
    _T(
        name="email_screenshot",
        title="Email Screenshot",
        description=(
            "Email-client UI overlay: header bar with FROM / TO / "
            "SUBJECT, timestamp, then 3-4 line body preview. For "
            "'the email read' / 'leaked email' / 'subject line "
            "alone said it all' beats."
        ),
        llm_rule=(
            "email_screenshot -> narration explicitly quotes or "
            "references an EMAIL ('the email read', 'leaked "
            "internal email', 'the subject line said'). t = "
            "'FROM_NAME|SUBJECT' (from <=24 chars; subject <=44 "
            "chars); b = 'TO_NAME|EMAIL_BODY' (to <=24 chars; "
            "body normal case <=240 chars). Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=5,
        render_module="footage._render_email_screenshot_card",
        assemble_branch="assemble._graphic_card_filters:email_screenshot",
    ),

    # 74) PHONE CALL LOG — timestamped call entries
    _T(
        name="call_log",
        title="Phone Call Log",
        description=(
            "Phone call log overlay: 3-5 timestamped call entries "
            "showing caller / time / duration with small phone "
            "icons. For 'the call records show' / phone-bill "
            "reveal beats."
        ),
        llm_rule=(
            "call_log -> narration explicitly references PHONE "
            "RECORDS or CALL LOGS ('the call records show', "
            "'phone bill reveals', 'three calls in two hours'). "
            "t = headline CAPS (<=30 chars e.g. 'CALL LOG | MAR "
            "12'); b = pipe-separated entries (3-5) as "
            "'CALLER@TIME@DURATION' (caller <=20 CAPS; time "
            "<=8 chars e.g. '14:23'; duration <=8 chars e.g. "
            "'00:47'). Example: 'M. REED@09:12@02:14|"
            "UNKNOWN@10:45@00:18'. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_call_log_card",
        assemble_branch="assemble._graphic_card_filters:call_log",
    ),

    # 75) RECEIPT — itemized bill
    _T(
        name="receipt",
        title="Receipt / Itemized Bill",
        description=(
            "Receipt-style narrow paper overlay: store/issuer "
            "name at top, dated, 3-5 line items with prices, "
            "subtotal line + big TOTAL. For 'the receipt found "
            "in his pocket' / 'the bill came to X' beats."
        ),
        llm_rule=(
            "receipt -> narration explicitly references a "
            "RECEIPT / BILL / ITEMIZED PURCHASE LIST ('the "
            "receipt found', 'his last meal cost', 'items "
            "purchased at'). t = 'STORE_NAME|DATE' (store <=24 "
            "chars CAPS; date <=14 chars); b = pipe-separated "
            "items (3-5) as 'ITEM@PRICE' (item <=24 chars CAPS;"
            " price <=12 chars e.g. '$14.99'). Example: 'COFFEE"
            "@$3.20|PASTRY@$4.50|JOURNAL@$12.00'. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_receipt_card",
        assemble_branch="assemble._graphic_card_filters:receipt",
    ),

    # 68) STICKY NOTE — yellow Post-It overlay
    _T(
        name="sticky_note",
        title="Sticky Note Overlay",
        description=(
            "Yellow Post-It note with handwritten-style text and "
            "subtle curl on one corner. For 'scribbled in the "
            "margin' / 'hand-written note found' beats. Smaller, "
            "intimate, personal."
        ),
        llm_rule=(
            "sticky_note -> narration explicitly mentions a SHORT "
            "INFORMAL NOTE found / written / left by someone "
            "('scribbled', 'a note read', 'left a yellow "
            "post-it'). t = the note text in normal case (<=80 "
            "chars); body is unused. Max 3/video."
        ),
        uses_body=False, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_sticky_note_card",
        assemble_branch="assemble._graphic_card_filters:sticky_note",
    ),

    # 69) PRESS RELEASE — official letterhead
    _T(
        name="press_release",
        title="Press Release / Letterhead",
        description=(
            "Formal letterhead document with organisation "
            "masthead + title + 2-3 paragraph block + signature "
            "line. Distinct from letter (handwritten / personal) "
            "and document (archival page). Press_release is the "
            "OFFICIAL announcement format."
        ),
        llm_rule=(
            "press_release -> narration quotes an OFFICIAL "
            "STATEMENT from a named organisation/agency/company "
            "('the agency issued', 'the official statement read', "
            "'the press release said'). t = 'ORG_NAME|HEADLINE' "
            "(org_name <=30 chars CAPS; headline <=44 chars "
            "CAPS); b = the statement body in normal case (<=240 "
            "chars). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_press_release_card",
        assemble_branch="assemble._graphic_card_filters:press_release",
    ),

    # 70) RADIO DIAL — frequency display
    _T(
        name="radio_dial",
        title="Radio Frequency Dial",
        description=(
            "Analog-radio frequency dial: big station-frequency "
            "readout (MHz/kHz) + call sign + signal-strength "
            "bars + tuning needle on a horizontal scale. For "
            "'broadcasting on' / 'shortwave intercept' beats."
        ),
        llm_rule=(
            "radio_dial -> narration explicitly references a "
            "RADIO BROADCAST or RADIO FREQUENCY ('broadcasting "
            "on 1480 kHz', 'shortwave intercept', 'pirate "
            "station on'). t = 'FREQUENCY|CALL_SIGN' (frequency "
            "<=14 chars e.g. '107.5 MHZ'; call_sign <=14 chars "
            "e.g. 'KGBR-FM'); b = 'BAND|SIGNAL' (band <=10 chars "
            "e.g. 'FM/AM/SW'; signal 0-100 strength). Max "
            "2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_radio_dial_card",
        assemble_branch="assemble._graphic_card_filters:radio_dial",
    ),

    # 71) VERTICAL BAR CHART — column chart
    _T(
        name="vertical_bar_chart",
        title="Vertical Bar Chart",
        description=(
            "3-6 vertical bars with labels below + value tags on "
            "top. Distinct from stats_bar (horizontal). For "
            "time-series / category comparison."
        ),
        llm_rule=(
            "vertical_bar_chart -> narration delivers 3-6 PARALLEL "
            "values across CATEGORIES OR TIME PERIODS ('year by "
            "year', 'across the regions', 'in each decade'). t = "
            "headline CAPS (<=30 chars); b = entries separated by "
            "';' as 'LABEL|VALUE_TEXT|RAW_NUMBER[|*]' where "
            "optional * marks one bar to highlight. label <=10 "
            "CAPS, value <=8 chars. Example: '1970|120|120;"
            "1980|180|180;1990|240|240|*;2000|310|310'. Max "
            "2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_vertical_bar_chart_card",
        assemble_branch="assemble._graphic_card_filters:vertical_bar_chart",
    ),

    # 64) HEATMAP GRID — color-intensity grid
    _T(
        name="heatmap_grid",
        title="Heatmap Grid / Intensity Map",
        description=(
            "Color-graded grid (rows × cols) where each cell's "
            "accent saturation maps to a numeric value 0-100. "
            "For 'highest density' / 'concentration map' / "
            "'risk hotspots' beats."
        ),
        llm_rule=(
            "heatmap_grid -> narration describes a SPATIAL/"
            "CATEGORICAL distribution of intensity ('the densest "
            "concentration', 'risk hotspots', 'where the cases "
            "cluster'). t = headline CAPS (<=30 chars e.g. "
            "'CASE DENSITY MAP'); b = rows separated by ';', "
            "each row a comma-separated list of values 0-100. "
            "Example: '10,30,60,80,95;20,40,70,90,85;5,25,45,"
            "55,70'. 3-5 rows, 3-7 cols. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_heatmap_grid_card",
        assemble_branch="assemble._graphic_card_filters:heatmap_grid",
    ),

    # 65) MAP PIN CLUSTER — multi-pin map
    _T(
        name="map_pin_cluster",
        title="Map Pin Cluster",
        description=(
            "Stylised nautical-chart map with 3-5 labeled pins "
            "clustered in a region — extension of map_reveal. "
            "For 'five sites across the desert' / 'all the "
            "incidents happened in this area' beats."
        ),
        llm_rule=(
            "map_pin_cluster -> narration explicitly references "
            "MULTIPLE LOCATIONS in the same region ('five sites', "
            "'three camps across', 'cases across the city'). t = "
            "headline CAPS (<=24 chars e.g. 'INCIDENT SITES'); "
            "b = pipe-separated pins as 'LABEL@X,Y' (label <=14 "
            "CAPS; X,Y are 0-100 grid coords), e.g. 'CAMP A@20,40|"
            "CAMP B@45,55|CAMP C@70,35'. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_map_pin_cluster_card",
        assemble_branch="assemble._graphic_card_filters:map_pin_cluster",
    ),

    # 66) NUMERICAL RATIO — X : Y comparison
    _T(
        name="numerical_ratio",
        title="Numerical Ratio (X : Y)",
        description=(
            "Big 'X : Y' ratio readout with explanatory labels "
            "for both sides. For 'twice as many' / 'three to "
            "one' / 'eight survivors for every casualty' beats. "
            "Distinct from comparison (which is sides A vs B "
            "with separate values)."
        ),
        llm_rule=(
            "numerical_ratio -> narration explicitly delivers a "
            "RATIO between two counts ('twice as many', 'three "
            "to one', 'eight survivors for every casualty'). t = "
            "'LEFT_NUM:RIGHT_NUM' (e.g. '8:1', '3:2'); b = "
            "'LEFT_LABEL|RIGHT_LABEL[|HEADLINE]' (each label "
            "<=24 chars CAPS; optional headline <=30 chars "
            "CAPS). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_numerical_ratio_card",
        assemble_branch="assemble._graphic_card_filters:numerical_ratio",
    ),

    # 67) DOCUMENT STACK — multiple stacked pages
    _T(
        name="document_stack",
        title="Document Stack",
        description=(
            "Multiple aged document pages stacked at varying "
            "angles in centre, with a top page label tag + page "
            "count. For 'stacks of declassified files' / "
            "'thousands of documents' beats."
        ),
        llm_rule=(
            "document_stack -> narration explicitly references a "
            "COLLECTION of documents/files/pages ('thousands of "
            "files', 'stacks of declassified records', 'the "
            "investigation produced hundreds of pages'). t = "
            "headline CAPS (<=30 chars e.g. 'DECLASSIFIED FILES');"
            " b = 'COUNT|CONTEXT' (count <=12 chars e.g. "
            "'2,847 PAGES'; context <=42 chars CAPS). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_document_stack_card",
        assemble_branch="assemble._graphic_card_filters:document_stack",
    ),

    # 60) CURRENCY / FINANCIAL STAT
    _T(
        name="currency_stat",
        title="Currency / Financial Stat",
        description=(
            "Big monetary figure ($X.XB / £M / etc) with optional "
            "up/down arrow + percentage change indicator + label. "
            "For 'cost $2.4 billion' / 'down 47%' / 'wiped out "
            "$300M' beats."
        ),
        llm_rule=(
            "currency_stat -> narration delivers a SPECIFIC MONEY "
            "FIGURE with optional direction/change ('cost 2.4 "
            "billion dollars', 'down forty-seven percent', '$300M "
            "in damages'). t = the money figure CAPS (<=14 chars "
            "e.g. '$2.4B', '£18M', '¥9.4T'); b = "
            "'DIRECTION|CHANGE_PERCENT|LABEL' (direction "
            "'UP'/'DOWN'/'FLAT', change e.g. '47%', label <=32 "
            "chars CAPS). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_currency_stat_card",
        assemble_branch="assemble._graphic_card_filters:currency_stat",
    ),

    # 61) ERA BANNER — full-screen era label
    _T(
        name="era_banner",
        title="Era / Time Period Banner",
        description=(
            "Full-screen time-period marker: huge ERA LABEL "
            "(POST-WAR EUROPE, COLD WAR PERIOD, INDUSTRIAL AGE) "
            "+ year span + accent rules. Similar to chapter_marker "
            "but specifically for time-period context shifts."
        ),
        llm_rule=(
            "era_banner -> narration shifts to a NAMED HISTORICAL "
            "OR THEMATIC ERA ('the post-war years', 'during the "
            "industrial revolution', 'in the cold war period'). "
            "t = the era label CAPS (<=30 chars e.g. 'POST-WAR "
            "EUROPE'); b = 'YEAR_SPAN[|CONTEXT]' (year_span <=14 "
            "chars e.g. '1945 — 1989'; optional context <=44 "
            "chars CAPS). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=10,
        render_module="footage._render_era_banner_card",
        assemble_branch="assemble._graphic_card_filters:era_banner",
    ),

    # 62) HEADLINE CRAWL — bottom news ticker
    _T(
        name="headline_crawl",
        title="Headline Crawl / News Ticker",
        description=(
            "Bottom-screen scrolling news ticker — single line of "
            "news-style headlines crawling left, with red NEWS tag "
            "on the left. CNN/news-channel style. Static "
            "pre-rendered (illusion of motion via assemble fade)."
        ),
        llm_rule=(
            "headline_crawl -> narration explicitly references a "
            "NEWS-TICKER STYLE feed of multiple short headlines "
            "('the news scrolled across', 'rolling reports said', "
            "'as breaking news ran'). t = the tag CAPS (<=12 "
            "chars e.g. 'BREAKING', 'NEWS'); b = pipe-separated "
            "headlines (3-6) that join into a single scroll line, "
            "each <=80 chars CAPS. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_headline_crawl_card",
        assemble_branch="assemble._graphic_card_filters:headline_crawl",
    ),

    # 63) STUDIO TWO-SHOT — split-screen interview frame
    _T(
        name="studio_two_shot",
        title="Studio Two-Shot Frame",
        description=(
            "Split-screen interview frame: two circular portrait "
            "placeholders side-by-side with name + role tags "
            "below. For 'interviewed by' / 'in conversation "
            "with' / 'sat across from' beats."
        ),
        llm_rule=(
            "studio_two_shot -> narration explicitly stages an "
            "INTERVIEW or CONVERSATION between TWO named people "
            "('interviewed by Marcus Reed', 'in conversation "
            "with', 'sat down with the journalist'). t = "
            "'NAME_LEFT|NAME_RIGHT' (each <=24 chars CAPS); b = "
            "'ROLE_LEFT|ROLE_RIGHT' (each <=22 chars CAPS). "
            "Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_studio_two_shot_card",
        assemble_branch="assemble._graphic_card_filters:studio_two_shot",
    ),

    # 56) SCORE DISPLAY — X/N rating card
    _T(
        name="score_display",
        title="Score / Rating Card",
        description=(
            "Review-style numerical score readout: huge X / N "
            "centred + small label + 5-segment progress bar in "
            "accent. For ratings, scores, ranks (8.5/10 review)."
        ),
        llm_rule=(
            "score_display -> narration delivers a NUMERIC SCORE "
            "or RATING out of a max ('rated 8.5 out of 10', "
            "'scored 92 out of 100', '4 stars'). t = "
            "'SCORE|MAX' (both numeric e.g. '8.5|10'); b = "
            "'LABEL[|FOOTER]' (label <=30 chars CAPS e.g. "
            "'CRITICS SCORE'; optional footer <=24 chars). "
            "Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_score_display_card",
        assemble_branch="assemble._graphic_card_filters:score_display",
    ),

    # 57) POLAROID STACK — pinned-photos stacked centre
    _T(
        name="polaroid_stack",
        title="Polaroid Stack",
        description=(
            "3-5 cream Polaroid frames stacked at varying angles "
            "in the centre of the frame — distinct from "
            "photo_collage (which is a spread evidence wall). "
            "Polaroid stack feels intimate / personal-archive."
        ),
        llm_rule=(
            "polaroid_stack -> narration explicitly references "
            "PERSONAL PHOTOGRAPHS stacked / collected / found "
            "TOGETHER ('a stack of photographs', 'her family "
            "album', 'photographs from his desk'). t = headline "
            "CAPS (<=30 chars e.g. 'PHOTOS FROM THE FILE'); b = "
            "pipe-separated captions for each Polaroid (3-5), "
            "e.g. 'KIEV 1971|FIRST CAMP|LAST PHOTO|FAMILY'. "
            "Max 2/video — distinct from photo_collage."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_polaroid_stack_card",
        assemble_branch="assemble._graphic_card_filters:polaroid_stack",
    ),

    # 58) CROSSHAIR TARGET LOCK — tactical target overlay
    _T(
        name="crosshair_lock",
        title="Crosshair / Target Lock",
        description=(
            "Tactical reticle overlay: large crosshair circle + "
            "centre target + corner brackets + LOCK status badge "
            "+ coords readout. For 'they had the target locked' "
            "/ 'precision strike' beats. Transparent overlay."
        ),
        llm_rule=(
            "crosshair_lock -> narration explicitly describes a "
            "TARGETING / SURVEILLANCE LOCK or precision-aim moment "
            "('they had the target', 'lock acquired', 'precision "
            "strike'). t = 'TARGET_NAME|STATUS' (target name "
            "<=24 chars CAPS, status <=12 chars e.g. 'LOCKED', "
            "'TRACKING'); b = 'COORDS_LINE[|RANGE]' (coords <=24 "
            "chars CAPS, optional range <=10 chars e.g. '2.4 KM'). "
            "Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_crosshair_lock_card",
        assemble_branch="assemble._graphic_card_filters:crosshair_lock",
    ),

    # 59) HASHTAG — trending social tag
    _T(
        name="hashtag_trend",
        title="Hashtag / Trending Tag",
        description=(
            "Social-media trending hashtag overlay — accent "
            "badge with # + tag name, small trend sparkline + "
            "engagement stat (12.4M posts, etc). For 'went "
            "viral' / 'top hashtag' beats."
        ),
        llm_rule=(
            "hashtag_trend -> narration explicitly references a "
            "VIRAL HASHTAG OR TRENDING TOPIC ('the hashtag went "
            "viral', '#DarvazaFire was trending', 'top of the "
            "trending list'). t = the hashtag without # CAPS "
            "(<=24 chars e.g. 'DARVAZAFIRE'); b = "
            "'ENGAGEMENT_STAT[|TIMESPAN]' (engagement <=18 chars "
            "e.g. '12.4M POSTS'; timespan <=14 chars e.g. "
            "'IN 24 HOURS'). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=4,
        render_module="footage._render_hashtag_trend_card",
        assemble_branch="assemble._graphic_card_filters:hashtag_trend",
    ),

    # 52) GLOBE — wide world map with region highlight
    _T(
        name="globe_map",
        title="Globe / World Map Highlight",
        description=(
            "Wide stylized world-map overlay with one region "
            "highlighted by an accent shape + label tag and pulse "
            "rings. For 'halfway around the world' / 'across "
            "three continents' / global-scale beats."
        ),
        llm_rule=(
            "globe_map -> narration explicitly invokes a GLOBAL "
            "OR INTERCONTINENTAL scale ('halfway around the world',"
            " 'across three continents', 'spanning the globe'). "
            "t = the region label CAPS (<=30 chars e.g. 'CENTRAL "
            "ASIA'); b = 'X|Y' grid coordinates where the "
            "highlight should appear (X 0-100 left→right, Y 0-100 "
            "top→bottom — e.g. '64|38' for central Asia). Max "
            "2/video — distinct from map_reveal (single-place)."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_globe_map_card",
        assemble_branch="assemble._graphic_card_filters:globe_map",
    ),

    # 53) FAMILY TREE 3-GEN — three-level hierarchy
    _T(
        name="family_tree_3gen",
        title="Family Tree (3-Generation)",
        description=(
            "Three-level family / org tree: 1 root grandparent → "
            "2-3 children → 2-3 grandchildren below. Each node "
            "is a small badge with a name. Connector accent "
            "lines form a proper genealogical tree."
        ),
        llm_rule=(
            "family_tree_3gen -> narration spans THREE "
            "GENERATIONS or three hierarchy levels (the "
            "founder's son's daughter, the great-grandchild, "
            "the third generation). t = the root node name CAPS "
            "(<=22 chars); b = pipe-blocks of 'GEN2_NAME>GEN3a,"
            "GEN3b,..' separated by ';', e.g. 'ANNA>EMMA,LIAM;"
            "NIKOLAI>NOAH,AVA'. 2-3 gen-2 children, each with "
            "1-2 gen-3 children. Max 1/video."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=10,
        render_module="footage._render_family_tree_3gen_card",
        assemble_branch="assemble._graphic_card_filters:family_tree_3gen",
    ),

    # 54) VERDICT STAMP — court/legal verdict overlay
    _T(
        name="verdict_stamp",
        title="Verdict / Court Stamp",
        description=(
            "Big rotated red rectangular stamp carrying the "
            "VERDICT word (APPROVED, DENIED, GUILTY, DISMISSED, "
            "ACQUITTED) over a small case-info banner. For "
            "court/legal outcome reveals."
        ),
        llm_rule=(
            "verdict_stamp -> narration explicitly delivers a "
            "LEGAL VERDICT or BINARY OUTCOME ('the court ruled', "
            "'the petition was denied', 'verdict: guilty', "
            "'appeal dismissed'). t = the verdict word CAPS "
            "(<=14 chars e.g. 'GUILTY', 'DENIED', 'ACQUITTED'); "
            "b = 'CASE_INFO|DATE' (case_info <=40 chars CAPS, "
            "date <=14 chars). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=8,
        render_module="footage._render_verdict_stamp_card",
        assemble_branch="assemble._graphic_card_filters:verdict_stamp",
    ),

    # 55) MINI BIO — compact bottom-right bio card
    _T(
        name="mini_bio",
        title="Mini Bio Card",
        description=(
            "Compact bottom-right bio card with a circular "
            "portrait avatar + name + 2-3 fact rows (BORN, "
            "ROLE, KNOWN FOR). Lighter than case_file — used "
            "for minor or supporting characters."
        ),
        llm_rule=(
            "mini_bio -> narration introduces a SECONDARY person "
            "by name + minimal context (the witness, the deputy, "
            "the engineer's wife). t = 'INITIALS|NAME' (initials "
            "1-3 CAPS chars; name <=28 chars CAPS); b = entries "
            "separated by ';' as 'LABEL: VALUE' (max 3 rows), "
            "e.g. 'BORN: 1939;ROLE: WITNESS;KNOWN FOR: TESTIMONY'. "
            "Max 4/video."
        ),
        uses_body=True, max_per_video=4, min_gap_scenes=3,
        render_module="footage._render_mini_bio_card",
        assemble_branch="assemble._graphic_card_filters:mini_bio",
    ),

    # 48) PULL QUOTE WITH PORTRAIT — quote + face
    _T(
        name="pull_quote_portrait",
        title="Pull Quote with Portrait",
        description=(
            "Centred quote with a CIRCULAR portrait thumbnail of "
            "the speaker on the LEFT, big italic-leaning serif "
            "quote on the right, attribution name + role below. "
            "Distinct from quote_highlight (no portrait) — use "
            "when the speaker's face matters."
        ),
        llm_rule=(
            "pull_quote_portrait -> narration quotes a NAMED "
            "person whose IDENTITY is part of the impact ('the "
            "President said', 'the survivor recalled', 'Carter "
            "would later write'). t = 'INITIALS|NAME, ROLE' "
            "(initials 1-3 CAPS chars for the portrait placeholder, "
            "name+role <=48 chars CAPS); b = the quote body in "
            "normal case (<=160 chars). Max 3/video — only "
            "when person matters; otherwise use quote_highlight."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=5,
        render_module="footage._render_pull_quote_portrait_card",
        assemble_branch="assemble._graphic_card_filters:pull_quote_portrait",
    ),

    # 49) SPOTLIGHT — dim-everything-except-this
    _T(
        name="spotlight",
        title="Spotlight / Eye-Focus Highlight",
        description=(
            "Dim-everything-except-a-bright-circle overlay. The "
            "underlying footage is darkened EXCEPT inside a "
            "centred (or labeled-region) ellipse — for 'the eye "
            "should be here' moments. Optional caption arrow + "
            "label pointing to the spotlight zone."
        ),
        llm_rule=(
            "spotlight -> narration draws attention to a SPECIFIC "
            "PART of the on-screen image ('notice the figure on "
            "the left', 'pay attention to the corner', 'in the "
            "background you can see'). t = the caption label CAPS "
            "(<=24 chars e.g. 'NOTICE', 'OBSERVE', 'HERE'); b = "
            "one-line context (<=80 chars CAPS) describing what "
            "the spotlight reveals. Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_spotlight_card",
        assemble_branch="assemble._graphic_card_filters:spotlight",
    ),

    # 50) CALENDAR GRID — full-month with highlighted dates
    _T(
        name="calendar_grid",
        title="Calendar Grid",
        description=(
            "Full-month calendar layout: MONTH YEAR header + "
            "7-column grid of dates with one or more dates "
            "highlighted in accent. For 'the events of March "
            "1971' / 'across three weeks in April' beats."
        ),
        llm_rule=(
            "calendar_grid -> narration references EVENTS "
            "OCCURRING ACROSS SPECIFIC DAYS of a single month "
            "('the events of March 1971', 'across three weeks "
            "in April', 'each day from the 4th to the 12th'). "
            "t = 'MONTH YEAR' (e.g. 'MARCH 1971'); b = "
            "comma-separated day numbers to highlight (e.g. "
            "'4,5,6,11,12,13'). Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_calendar_grid_card",
        assemble_branch="assemble._graphic_card_filters:calendar_grid",
    ),

    # 51) DISCLAIMER — "based on..." / "names changed"
    _T(
        name="disclaimer",
        title="Disclaimer / Based-on Card",
        description=(
            "Small documentary disclaimer card: italic body, "
            "subtle border, bottom-centre placement. For 'based "
            "on a true story', 'names have been changed', "
            "'viewer discretion advised', 'reenactment'."
        ),
        llm_rule=(
            "disclaimer -> narration or context calls for a "
            "STANDARD DISCLAIMER (based on real events, names "
            "have been changed, dramatized, viewer discretion). "
            "t = the disclaimer text in normal case (<=120 "
            "chars); body is unused. Max 2/video — typically "
            "scene 1 or scene 2 only."
        ),
        uses_body=False, max_per_video=2, min_gap_scenes=10,
        render_module="footage._render_disclaimer_card",
        assemble_branch="assemble._graphic_card_filters:disclaimer",
    ),

    # 44) CAUTION TAPE — yellow-black hazard band
    _T(
        name="caution_tape",
        title="Caution / Hazard Tape Overlay",
        description=(
            "Diagonal yellow-black caution-tape band crossing the "
            "frame with a centred CAUTION / WARNING / RESTRICTED "
            "label. Transparent overlay so footage stays visible. "
            "For disaster, hazard, restricted-area, or "
            "do-not-enter beats."
        ),
        llm_rule=(
            "caution_tape -> narration explicitly invokes a "
            "HAZARD, WARNING, RESTRICTED AREA, OR DANGER ZONE "
            "('the area was sealed off', 'restricted-access "
            "site', 'do not enter'). t = the warning label CAPS "
            "(<=24 chars e.g. 'CAUTION', 'RESTRICTED', 'KEEP "
            "OUT'); b = optional sub-line CAPS (<=44 chars "
            "e.g. 'HAZARDOUS MATERIAL ZONE') or empty. Max "
            "2/video — use sparingly."
        ),
        uses_body=False, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_caution_tape_card",
        assemble_branch="assemble._graphic_card_filters:caution_tape",
    ),

    # 45) AWARD / MEDAL RIBBON — laurel wreath ribbon
    _T(
        name="award_ribbon",
        title="Award / Medal Ribbon",
        description=(
            "Centred laurel-wreath medal overlay with award "
            "title, recipient name, and year. For 'winner of' "
            "/ 'recipient of the Nobel' / 'awarded the Medal of "
            "Honour' beats. Same dark navy data bg."
        ),
        llm_rule=(
            "award_ribbon -> narration explicitly notes someone "
            "WAS AWARDED, WON, or RECEIVED a recognised honour "
            "(Nobel Prize, Pulitzer, Order of Lenin, Medal of "
            "Valor). t = the award name CAPS (<=30 chars e.g. "
            "'NOBEL PRIZE IN PHYSICS'); b = 'RECIPIENT_NAME|"
            "YEAR' (name <=30 chars CAPS, year <=8 chars). "
            "Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_award_ribbon_card",
        assemble_branch="assemble._graphic_card_filters:award_ribbon",
    ),

    # 46) SFX CUE — bracketed sound caption
    _T(
        name="sfx_cue",
        title="Sound Effect Cue / SFX Label",
        description=(
            "Small italic bracketed sound-caption label like "
            "[ EXPLOSION SOUNDS ] or [ ALARM SIRENS ]. Bottom-"
            "centre overlay simulating closed-caption SFX hints. "
            "For audio-event tagging without playing actual SFX."
        ),
        llm_rule=(
            "sfx_cue -> narration is describing an AUDIO EVENT "
            "the viewer should imagine ('then came the sound "
            "of', 'in the recording you hear', '[siren], "
            "[explosion]'). t = the SFX cue inside brackets, "
            "CAPS (<=44 chars e.g. 'EXPLOSION SOUNDS', 'ALARM "
            "SIRENS', 'INDISTINCT SHOUTING'); body is unused. "
            "Max 4/video — only for genuine audio cues."
        ),
        uses_body=False, max_per_video=4, min_gap_scenes=3,
        render_module="footage._render_sfx_cue_card",
        assemble_branch="assemble._graphic_card_filters:sfx_cue",
    ),

    # 47) FOOTNOTE / CITATION — bottom-right source tag
    _T(
        name="footnote",
        title="Footnote / Source Citation",
        description=(
            "Small bottom-right citation tag — 'SOURCE: NYT "
            "1971', 'DECLASSIFIED MEMO 04-712', 'CDC REPORT "
            "2023', etc. Tiny attribution overlay for academic-"
            "feel sourcing. Pairs with most other templates."
        ),
        llm_rule=(
            "footnote -> narration cites or implies a "
            "SOURCE/REFERENCE for a fact just said ('according "
            "to the CDC', 'declassified in 1989', 'NYT "
            "investigation'). t = the source citation CAPS "
            "(<=44 chars); body is unused. Max 4/video — only "
            "when the line really needs attribution on screen."
        ),
        uses_body=False, max_per_video=4, min_gap_scenes=2,
        render_module="footage._render_footnote_card",
        assemble_branch="assemble._graphic_card_filters:footnote",
    ),

    # 40) TALLY COUNTER — running count display
    _T(
        name="tally_counter",
        title="Live Tally Counter",
        description=(
            "Large running tally readout — huge bold number "
            "in TYPE_DISPLAY + label below + tiny 'AND COUNTING' "
            "tag. For 'the death toll climbed to' / 'days under "
            "siege' / 'casualties this week' beats. Same data "
            "family bg."
        ),
        llm_rule=(
            "tally_counter -> narration gives a RUNNING TOTAL "
            "that is escalating or still climbing (the death toll "
            "reached X and counting, N days under siege, X "
            "confirmed dead). t = the figure CAPS (<=8 chars, "
            "e.g. '2,347'); b = 'LABEL[|FOOTER]' (label <=32 "
            "chars CAPS e.g. 'CONFIRMED DEAD', optional footer "
            "<=20 chars e.g. 'AND COUNTING'). Distinct from "
            "number (which is a single static stat); tally "
            "implies escalation. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_tally_counter_card",
        assemble_branch="assemble._graphic_card_filters:tally_counter",
    ),

    # 41) GPS COORDINATE STAMP — small location pill
    _T(
        name="gps_stamp",
        title="GPS Coordinate Stamp",
        description=(
            "Small bottom-left coordinate pill: LIVE GPS lock "
            "icon + lat/long + altitude + location name. "
            "Transparent overlay for 'we are at exactly' / "
            "'37.7749° N, 122.4194° W' beats. Lightweight."
        ),
        llm_rule=(
            "gps_stamp -> narration gives or implies EXACT "
            "COORDINATES of where something happened (the "
            "expedition landed at, the wreck lies at exactly). "
            "t = 'LATITUDE|LONGITUDE' (each <=16 chars e.g. "
            "'40.2526° N|58.4397° E'); b = "
            "'LOCATION_NAME[|ALTITUDE]' (name <=32 chars CAPS, "
            "optional altitude <=14 chars e.g. '180 M BSL'). "
            "Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=3,
        render_module="footage._render_gps_stamp_card",
        assemble_branch="assemble._graphic_card_filters:gps_stamp",
    ),

    # 42) MICROSCOPE INSET — circular zoom detail
    _T(
        name="microscope_inset",
        title="Microscope Detail Inset",
        description=(
            "Circular magnified detail inset at top-right of "
            "footage with a leader line connecting back to the "
            "source point on the main frame. Carries a small "
            "DETAIL label and a one-line caption. For 'look "
            "closer' / 'a closer view shows' / 'enhanced "
            "imagery' beats."
        ),
        llm_rule=(
            "microscope_inset -> narration explicitly draws "
            "ATTENTION to a CLOSE-UP DETAIL within the current "
            "footage ('look closer', 'a closer view shows', "
            "'enhanced imagery reveals', 'in the corner of the "
            "frame'). t = label CAPS (<=16 chars e.g. 'CLOSE-UP', "
            "'ENHANCED', 'DETAIL'); b = one-line caption "
            "describing what the close-up reveals (<=70 chars "
            "CAPS). Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_microscope_inset_card",
        assemble_branch="assemble._graphic_card_filters:microscope_inset",
    ),

    # 43) RELATIONSHIP TREE — 2-level org/family chart
    _T(
        name="relationship_tree",
        title="Relationship / Org Tree",
        description=(
            "Two-level organisational/family tree: one parent "
            "node at top + 2-4 child nodes connected by clean "
            "accent lines. Each node is a small badge with a "
            "name + optional sub-label. For 'his three sons' / "
            "'the four directors' / 'the chain of command' beats."
        ),
        llm_rule=(
            "relationship_tree -> narration explicitly maps a "
            "HIERARCHICAL relationship from one person/entity "
            "to 2-4 others (his three children, the four "
            "deputies, the chain of command). t = the parent "
            "node name CAPS (<=22 chars); b = entries separated "
            "by ';' as 'NAME[|SUBLABEL]' (name <=18 chars CAPS, "
            "optional sublabel <=14 chars CAPS), 2-4 children "
            "total. Example: 'EMMA|SCIENTIST;LIAM|JOURNALIST;"
            "NOAH|ATTORNEY'. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_relationship_tree_card",
        assemble_branch="assemble._graphic_card_filters:relationship_tree",
    ),

    # 36) STEP INDICATOR — linear progress dots
    _T(
        name="step_indicator",
        title="Step Indicator / Progress Dots",
        description=(
            "Compact top-screen STEP N OF M strip — small connected "
            "dots/chevrons left-to-right showing completed (filled), "
            "current (accent halo), and upcoming (dim) steps + the "
            "current step's name. For multi-stage walk-throughs."
        ),
        llm_rule=(
            "step_indicator -> narration explicitly says 'step N of "
            "M', 'phase N of M', 'stage N of M' (the second of "
            "four phases, step three in the process). t = "
            "'CURRENT|TOTAL' (both integers e.g. '3|5'); b = "
            "'STEP_NAME[|HEADLINE]' (step_name <=30 chars CAPS; "
            "optional headline <=24 chars CAPS). Use sparingly — "
            "max 3/video, only when the step count matters."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_step_indicator_card",
        assemble_branch="assemble._graphic_card_filters:step_indicator",
    ),

    # 37) DID YOU KNOW — trivia interjection
    _T(
        name="did_you_know",
        title="Did You Know? Trivia Card",
        description=(
            "Documentary interjection card carrying a surprising "
            "tangential fact. Large '?' icon + DID YOU KNOW? tag "
            "+ bold fact body. Centred card with subtle accent "
            "border. For aside-of-narrative trivia beats."
        ),
        llm_rule=(
            "did_you_know -> narration explicitly stops to deliver "
            "a TANGENTIAL surprising fact ('did you know...', "
            "'fun fact:', 'incidentally...', 'few people realise'). "
            "t = optional headline CAPS (<=24 chars; defaults to "
            "'DID YOU KNOW?'); b = the fact body (<=180 chars). "
            "Max 2/video — use only when the line is genuinely a "
            "tangent, not the main beat."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_did_you_know_card",
        assemble_branch="assemble._graphic_card_filters:did_you_know",
    ),

    # 38) VOICE MEMO — analog cassette/tape reel overlay
    _T(
        name="voice_memo",
        title="Voice Memo / Tape Reel",
        description=(
            "Vintage cassette / reel-to-reel audio overlay — drawn "
            "tape reels on either side, central timecode display, "
            "REC indicator + transcript line below. The analog "
            "complement to audio_waveform (which is digital). For "
            "'taken from the field tape' / 'cassette dated 1971'."
        ),
        llm_rule=(
            "voice_memo -> narration references an ANALOG audio "
            "recording (a cassette, reel-to-reel tape, dictaphone, "
            "field recording — pre-digital). t = 'TAPE_LABEL|"
            "TIMECODE' (tape_label <=24 chars CAPS, timecode "
            "<=10 chars e.g. '00:14:23'); b = the transcript "
            "line (<=140 chars). Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=5,
        render_module="footage._render_voice_memo_card",
        assemble_branch="assemble._graphic_card_filters:voice_memo",
    ),

    # 39) INSCRIPTION — engraved monument plaque
    _T(
        name="inscription",
        title="Inscription / Monument Plaque",
        description=(
            "Engraved-text plaque overlay — stone or bronze "
            "backdrop with deep-relief letterforms (drawn via "
            "text + shadow technique). For 'the inscription "
            "read' / 'the monument says' / 'carved on the "
            "headstone' beats."
        ),
        llm_rule=(
            "inscription -> narration is quoting words "
            "PHYSICALLY ENGRAVED somewhere (inscription on a "
            "monument, headstone, plaque, building lintel, "
            "memorial wall). t = the engraved text in CAPS "
            "(<=44 chars); b = optional 'LOCATION[|YEAR]' "
            "(location <=36 chars CAPS; year <=8 chars). Max "
            "2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_inscription_card",
        assemble_branch="assemble._graphic_card_filters:inscription",
    ),

    # 32) SUBTITLE / TRANSLATION — foreign-language quote
    _T(
        name="subtitle_translation",
        title="Subtitle / Translation Strip",
        description=(
            "Bottom-screen translation strip — original-language "
            "quote on top (in italic/smaller), English translation "
            "below (in larger bold), small language tag in the "
            "corner. For '[in Russian]' / '[translated from German]' "
            "moments when a non-English source is being relayed."
        ),
        llm_rule=(
            "subtitle_translation -> narration is RELAYING a quote "
            "from a non-English source (a Russian newspaper, a "
            "German letter, a Spanish witness) and the line "
            "WORKS BETTER as a translated subtitle than as a "
            "standalone quote_highlight. t = 'LANG_TAG|ORIGINAL' "
            "(lang_tag <=12 chars CAPS e.g. 'RUSSIAN', "
            "'TRANSLATED FROM'; original <=80 chars); b = the "
            "English TRANSLATION (<=140 chars). Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_subtitle_translation_card",
        assemble_branch="assemble._graphic_card_filters:subtitle_translation",
    ),

    # 33) REACTION INSERT — picture-in-picture frame
    _T(
        name="reaction_insert",
        title="Reaction Insert / PIP Frame",
        description=(
            "Small picture-in-picture frame top-right of footage. "
            "Carries a small REACTION/WATCHING/AT THE TIME tag, a "
            "frame placeholder (or footage frame), and a one-line "
            "caption. For 'meanwhile, the world watched' beats."
        ),
        llm_rule=(
            "reaction_insert -> narration explicitly references "
            "a SECONDARY VIEWPOINT happening AT THE SAME TIME "
            "as the main scene (the family watched on TV, "
            "officials reacted to the news, the crowd gasped). "
            "t = label CAPS (<=18 chars e.g. 'REACTION', "
            "'AT THE TIME', 'WATCHING'); b = the caption "
            "describing what's happening in the inset (<=70 "
            "chars CAPS). Max 3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_reaction_insert_card",
        assemble_branch="assemble._graphic_card_filters:reaction_insert",
    ),

    # 34) FORENSIC EVIDENCE TAG — individual evidence card
    _T(
        name="evidence_tag",
        title="Forensic Evidence Tag",
        description=(
            "Numbered forensic evidence card: yellow EVIDENCE #N "
            "tag, short description of the item, recovered-from "
            "location + date stamp. For 'the murder weapon' / "
            "'the missing letter' / 'item recovered from the "
            "scene' beats — individual evidence items, not the "
            "whole collection."
        ),
        llm_rule=(
            "evidence_tag -> narration names a SPECIFIC PHYSICAL "
            "ITEM that was found, recovered, seized, or entered "
            "as evidence (the bloodied knife, the missing letter, "
            "the burned passport, item recovered from the scene). "
            "t = 'NUMBER|ITEM' (number e.g. 'EVIDENCE 03', item "
            "<=36 chars CAPS); b = 'FROM|DATE' (recovered-from "
            "<=36 chars CAPS, date <=14 chars). Max 4/video — "
            "use for THE item, not 'they found stuff'."
        ),
        uses_body=True, max_per_video=4, min_gap_scenes=3,
        render_module="footage._render_evidence_tag_card",
        assemble_branch="assemble._graphic_card_filters:evidence_tag",
    ),

    # 35) DIARY ENTRY — multi-dated journal page
    _T(
        name="diary_entry",
        title="Diary Entry / Journal Page",
        description=(
            "Aged journal page with 1-3 dated entries stacked "
            "vertically. Each entry: small date stub on left, "
            "italic body text on right. Different visual from "
            "`letter` (which is one single message); diary is "
            "the multi-entry temporal collection."
        ),
        llm_rule=(
            "diary_entry -> narration explicitly quotes from a "
            "PERSONAL JOURNAL OR DIARY across MULTIPLE DATES "
            "('in her diary across the years', 'on three "
            "separate days she wrote', 'his journal entries "
            "from 1941-1943'). t = headline CAPS (<=30 chars, "
            "e.g. 'DIARY OF M. PETROVA'); b = entries separated "
            "by ';' as 'DATE|ENTRY_TEXT' (date <=16 chars CAPS, "
            "entry <=120 chars). Max 2 entries shown. Max "
            "2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_diary_entry_card",
        assemble_branch="assemble._graphic_card_filters:diary_entry",
    ),

    # 29) CASE FILE / MUGSHOT — investigator profile card
    _T(
        name="case_file",
        title="Police Case File / Mugshot Card",
        description=(
            "True-crime investigator card: dark backdrop with "
            "height-line ruler, mugshot frame on the left (or "
            "footage cell as fallback), and rows of CASE / NAME / "
            "DOB / CHARGE / STATUS on the right. For true-crime "
            "subject-introduction beats — the 'meet the case' "
            "visual."
        ),
        llm_rule=(
            "case_file -> narration introduces a CRIMINAL CASE, "
            "SUSPECT, or PERSON-OF-INTEREST by name with at least "
            "one identifier (DOB, charge, case number, status). "
            "t = 'NAME|CASE_NO' (name <=30 chars CAPS; case_no "
            "<=14 chars e.g. 'CASE-0471'); b = pipe-separated "
            "rows of 'LABEL: VALUE' (max 4 rows), e.g. "
            "'DOB: 1937-04-12|CHARGE: MURDER × 3|STATUS: "
            "FUGITIVE|LAST SEEN: WARSAW, 1972'. Max 2/video — "
            "only for genuine named case subjects."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_case_file_card",
        assemble_branch="assemble._graphic_card_filters:case_file",
    ),

    # 30) DEMOGRAPHIC SPLIT — proportional segmented bar
    _T(
        name="demographic_split",
        title="Demographic / Population Split",
        description=(
            "Horizontal proportional split bar showing a population "
            "or category breakdown (e.g. AGES 0-17 | 18-44 | 45-64 "
            "| 65+) with each segment width proportional to its "
            "share. Each segment in a different accent shade with "
            "label + percentage above it."
        ),
        llm_rule=(
            "demographic_split -> narration explicitly breaks a "
            "WHOLE POPULATION or category into NAMED SHARES "
            "(of the survivors, 40% were children, 35% adults, "
            "25% elderly). t = headline CAPS (<=30 chars, e.g. "
            "'POPULATION BREAKDOWN'); b = entries separated by "
            "';' as 'LABEL|PERCENT' (label <=18 CAPS, percent "
            "float 0-100 — must sum to ~100). Example: "
            "'CHILDREN 0-17|40;ADULTS 18-64|45;ELDERLY 65+|15'. "
            "Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_demographic_split_card",
        assemble_branch="assemble._graphic_card_filters:demographic_split",
    ),

    # 31) GLOSSARY DEFINITION — DEF tag + term + definition
    _T(
        name="glossary",
        title="Glossary / Definition Card",
        description=(
            "Documentary 'what is X?' definition card. Layout: "
            "small DEF tag in accent, BIG TERM in TYPE_H1, short "
            "italic definition body below, optional source line. "
            "Used to define a key technical or unfamiliar term "
            "the narrator just introduced."
        ),
        llm_rule=(
            "glossary -> narration introduces a TECHNICAL TERM, "
            "JARGON WORD, or unfamiliar concept the viewer would "
            "benefit from seeing defined ('what is methanogenesis?',"
            " 'the term tactical-grade refers to...'). t = the "
            "TERM in CAPS (<=24 chars, e.g. 'METHANOGENESIS'); b "
            "= 'DEFINITION[|SOURCE]' (definition <=180 chars; "
            "optional source <=24 chars e.g. 'OXFORD'). Max "
            "3/video — only for genuinely worth-defining terms."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_glossary_card",
        assemble_branch="assemble._graphic_card_filters:glossary",
    ),
    _T(
        name="define_the_term",
        title="Define-the-Term Thesis Card",
        description=(
            "The Johnny-Harris 'define the central term' beat: the SAME "
            "clean bottom-third definition card as glossary, but used "
            "ONCE, EARLY, as the video's THESIS — it frames the whole "
            "story around one concept (a coup, an empire, a doctrine). "
            "Not a full-screen infographic — a restrained lower-third."
        ),
        llm_rule=(
            "define_the_term -> use ONCE per video, EARLY (in the first "
            "third), to define the ONE central concept the whole story "
            "turns on (e.g. 'COUP', 'MONOPOLY', 'PROVINCE'). This is the "
            "THESIS card, distinct from glossary (which defines incidental "
            "jargon mid-video). t = the central TERM in CAPS (<=24 chars); "
            "b = 'DEFINITION[|SOURCE]' (definition <=160 chars, plain). "
            "Best for geopolitics / science / history / business. Max 1."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=20,
        render_module="footage._render_glossary_card",
        assemble_branch="assemble._graphic_card_filters:glossary",
    ),
    _T(
        name="long_quote",
        title="Sustained Primary-Source Quote",
        description=(
            "A serious-history hallmark: a longer PRIMARY-SOURCE quotation "
            "held on screen long enough to READ (10-40s), in the same "
            "restrained serif quote treatment as quote_highlight (auto-fit "
            "so a long quote stays readable, never giant). For a "
            "contemplative beat where the period voice itself carries the "
            "moment — history, biography, investigation."
        ),
        llm_rule=(
            "long_quote -> a SUSTAINED primary-source quotation (a longer "
            "passage from a letter / chronicle / testimony, ~15-45 words) "
            "the viewer is meant to READ on a slow contemplative beat — "
            "distinct from quote_highlight (a short punchy line). Place on "
            "a LONG, calm scene. t = the quotation (no surrounding quote "
            "marks); b = the attribution (e.g. 'TACITUS, AGRICOLA, c.98 "
            "AD'). Best for history / biography / investigation. Max 2."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=8,
        render_module="footage._render_quote_highlight_card",
        assemble_branch="assemble._graphic_card_filters:quote_highlight",
    ),

    # 27) HANDWRITTEN LETTER — aged-paper note
    _T(
        name="letter",
        title="Handwritten Letter / Correspondence",
        description=(
            "Aged note-paper card with a handwritten-style serif "
            "body, optional date in the upper-right, salutation "
            "line ('Dear...'), and a signature flourish at the "
            "bottom. Subtle yellowed paper, soft tear-edge. For "
            "'the letter read' / 'in her final note' / 'his diary "
            "entry' beats."
        ),
        llm_rule=(
            "letter -> narration explicitly references a LETTER, "
            "DIARY ENTRY, PERSONAL NOTE, OR HANDWRITTEN MESSAGE "
            "from a named person (the letter said, in his diary, "
            "her final note). t = 'DATE|SIGNED-AS' (date <=20 "
            "chars CAPS e.g. 'MARCH 12, 1971'; signed name <=24 "
            "chars CAPS); b = the letter body text in NORMAL "
            "case (<=240 chars). Use ONLY when the line is "
            "genuinely about a written personal communication. "
            "Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_letter_card",
        assemble_branch="assemble._graphic_card_filters:letter",
    ),

    # 28) HORIZONTAL STATS BAR — bar chart with highlight
    _T(
        name="stats_bar",
        title="Horizontal Stats Bar",
        description=(
            "Horizontal bar chart with 3-5 labeled bars where ONE "
            "is accent-filled (the highlighted stat) and the rest "
            "are dim — for 'highest of any region' / 'compared to "
            "every other case' beats. Each bar: small LABEL on "
            "the left, accent-or-dim fill bar to the right scaled "
            "to the value, then the VALUE printed at the bar end."
        ),
        llm_rule=(
            "stats_bar -> narration explicitly RANKS or COMPARES "
            "a single value against 2-4 peers (X has the highest, "
            "Y is more than Z, A vs B vs C vs D where one stands "
            "out). t = headline CAPS (<=30 chars, e.g. 'WORST "
            "AFFECTED REGIONS'); b = entries separated by ';' as "
            "'LABEL|VALUE_TEXT|RAW_NUMBER[|*]' where * marks the "
            "highlighted bar. label <=14 CAPS, value_text <=10 "
            "chars (e.g. '12,400'), raw_number for bar scale. "
            "Example: 'KARAKUM|12,400|12400|*;GOBI|7,800|7800;"
            "SAHARA|6,200|6200;SONORAN|3,400|3400'. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_stats_bar_card",
        assemble_branch="assemble._graphic_card_filters:stats_bar",
    ),

    # 25) SOCIAL POST — Twitter/X-style overlay
    _T(
        name="social_post",
        title="Social Media Post",
        description=(
            "Twitter/X-style social-media post card: avatar circle "
            "in theme accent + handle + verified tick + post body "
            "in serif + timestamp. Cream/dark card with soft "
            "shadow, slight rotation so it reads as 'pinned to "
            "screen'. For 'this was posted online' / 'the tweet "
            "went viral' beats — modern documentary signal."
        ),
        llm_rule=(
            "social_post -> narration is quoting / showing / "
            "referencing a SOCIAL MEDIA POST (a tweet, an X post, "
            "a Facebook post, a viral message). t = "
            "'@handle|Display Name' (handle <=18 chars, display "
            "name <=24 chars); b = the post body text (<=180 "
            "chars). Use ONLY when the line is genuinely about "
            "what was POSTED ONLINE — not as decoration. Max "
            "3/video."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_social_post_card",
        assemble_branch="assemble._graphic_card_filters:social_post",
    ),

    # 26) NEWS ARTICLE EXCERPT — column snippet w/ highlight
    _T(
        name="news_article",
        title="News Article Excerpt",
        description=(
            "Snippet of a printed newspaper column: small SOURCE / "
            "DATE tag at top, serif headline, 3-line article body "
            "excerpt in newspaper-column type, optional yellow "
            "highlighter sweep on a key phrase. For 'the article "
            "reported' / 'as published in' beats."
        ),
        llm_rule=(
            "news_article -> narration explicitly quotes or cites "
            "what a NEWSPAPER, MAGAZINE, or WIRE SERVICE published "
            "about the subject (the Times wrote, the Post reported, "
            "as covered in...). t = 'SOURCE|DATE' (source <=24 "
            "chars CAPS e.g. 'THE NEW YORK TIMES', date <=14 chars "
            "e.g. 'APRIL 14, 1971'); b = 'HEADLINE|EXCERPT' "
            "(headline <=64 chars; excerpt <=170 chars). Max "
            "3/video — different from newspaper (full front-page "
            "treatment)."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=5,
        render_module="footage._render_news_article_card",
        assemble_branch="assemble._graphic_card_filters:news_article",
    ),

    # 23) CAUSE / EFFECT — two-box arrow flow
    _T(
        name="cause_effect",
        title="Cause / Effect Flow",
        description=(
            "Two-box horizontal flow: left CAUSE box → big accent "
            "arrow → right EFFECT box. Each box has a small label "
            "tag (CAUSE/EFFECT) and a big bold value. Used for 'X "
            "led to Y' / 'one decision created decades of...' "
            "beats. Simpler than process_diagram (which is 3-4 "
            "steps); cause_effect is the SINGLE-link causal beat."
        ),
        llm_rule=(
            "cause_effect -> narration explicitly says ONE thing "
            "DIRECTLY CAUSED another (X led to Y, this triggered "
            "that, the decision produced the consequence). t = "
            "headline CAPS (<=30 chars, e.g. 'ONE DECISION'); b = "
            "'LEFT_LABEL|LEFT_VALUE;;RIGHT_LABEL|RIGHT_VALUE' "
            "(left = the cause, right = the effect; labels short "
            "CAPS <=14 chars, values <=32 chars), e.g. 'DECISION|"
            "IGNITE THE METHANE;;OUTCOME|50+ YEARS OF FIRE'. Use "
            "when the LINK is the line's whole point. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_cause_effect_card",
        assemble_branch="assemble._graphic_card_filters:cause_effect",
    ),

    # 24) MAP ROUTE — REAL map, route line DRAWS between stops
    _T(
        name="map_route",
        title="Map Route Animation",
        description=(
            "REAL satellite/political map fitted to the journey; the "
            "route line DRAWS progressively with a glowing dot, "
            "start/end pins land, direction arrows + distance label. "
            "For journeys, military movements, trade routes, "
            "migrations, escapes, supply chains."
        ),
        llm_rule=(
            "map_route -> narration describes MOVEMENT between places "
            "(fled X to Y, marched A to B, the trade route, migration, "
            "supply line, attack path). t = 2-5 stops separated by "
            "'|', each 'Name@lat,lon' with coordinates STRONGLY "
            "preferred (e.g. 'Berlin@52.52,13.40|Moscow@55.75,37.62'); "
            "plain 'Name' also works (geocoded). b = optional short "
            "caption (<=44 chars, e.g. 'The Eastern Advance'). Use "
            "ONCE per genuine route, max 2/video — not for vague "
            "'moved around' lines."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_real_map_route_card",
        assemble_branch="assemble._graphic_card_filters:map_route",
    ),

    # 24b) MAP REGION — REAL territory/country highlight
    _T(
        name="map_region",
        title="Map Region Highlight",
        description=(
            "REAL map zoomed to a country/region; surroundings dim, "
            "the territory glows inside a boundary ring sized to its "
            "real extent, label + optional context tag (population / "
            "area / note), with a pulse. For countries, territories, "
            "conflict zones, ancient regions, disaster/military areas."
        ),
        llm_rule=(
            "map_region -> narration focuses on a COUNTRY/REGION/"
            "TERRITORY as an area (not a single point) — a nation, a "
            "conflict zone, an ancient region, a disaster/military "
            "area, a trade region. t = the region name (optionally "
            "'Name@lat,lon'); b = optional short context tag (<=46 "
            "chars, e.g. '82M people · 357,000 km²' or 'Contested "
            "since 1947'). Use ONCE per genuine region, max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_real_map_region_card",
        assemble_branch="assemble._graphic_card_filters:map_region",
    ),

    # 22) DONUT CHART — single-percentage center stat
    _T(
        name="donut_chart",
        title="Donut Chart / Percentage Split",
        description=(
            "Documentary donut chart for a SINGLE-percentage stat: "
            "ring filled to N% in theme accent, dim remainder, big "
            "centre stat (e.g. '67%'), small label below ring "
            "(e.g. 'OF SURVIVORS NEVER SPOKE AGAIN'). Same dark "
            "navy + grid bg as the data templates. For 'X% of...' "
            "or 'two-thirds of...' beats where a single proportion "
            "is the line."
        ),
        llm_rule=(
            "donut_chart -> narration delivers a SINGLE percentage "
            "or fraction that defines a relationship (X% of "
            "survivors, two-thirds of cases, 1 in 4 victims). t = "
            "the percentage value CAPS (<=8 chars, e.g. '67%', "
            "'2 IN 3', '1/4'); b = "
            "'PERCENT|LABEL_LINE[|HEADLINE]' (percent as float "
            "0-100, label <=60 chars CAPS, optional headline <=30 "
            "chars CAPS), e.g. '67|OF SURVIVORS NEVER SPOKE AGAIN|"
            "THE SILENT MAJORITY'. Use sparingly — single-percent "
            "is the WHOLE point; for multiple numbers use "
            "stat_dashboard. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=5,
        render_module="footage._render_donut_chart_card",
        assemble_branch="assemble._graphic_card_filters:donut_chart",
    ),

    # 21) TITLE CARD — cinematic series/episode open
    _T(
        name="title_card",
        title="Title Card / Cold Open",
        description=(
            "Cinematic Netflix-style title card: HUGE series title "
            "(TYPE_DISPLAY) centered, subtitle in TYPE_H1, small "
            "EPISODE N tag at the top, accent brackets framing the "
            "title block. Used as a hero opener — typically the "
            "first or second scene of a video. Same dark navy + "
            "grid family so it groups with the data templates."
        ),
        llm_rule=(
            "title_card -> the cold-open title beat. Place it on the "
            "SECOND or THIRD scene — AFTER a brief footage HOOK, never "
            "the very first scene (opening straight on the title looks "
            "cheap; a premium doc hooks with footage first, THEN drops "
            "the title). The line establishes the documentary's subject "
            "as a CAPITAL-T TITLE. t = the series/episode title CAPS "
            "(<=36 chars, e.g. 'THE DOOR TO HELL', 'LOST CITIES', "
            "'CHERNOBYL'); b = 'SUBTITLE[|EPISODE_TAG]' (subtitle <=44 "
            "chars CAPS, optional episode tag <=20 chars e.g. "
            "'EPISODE 01' or 'A DOCUMENTARY'). Use ONCE per video — "
            "max 1, scene 2-3 only, never scene 1."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=99,
        render_module="footage._render_title_card",
        assemble_branch="assemble._graphic_card_filters:title_card",
    ),

    # 20) REDACTED STAMP — censored-document overlay
    _T(
        name="redacted",
        title="Redacted / Censored Stamp",
        description=(
            "Censored-document overlay: cream paper field with "
            "typewriter-text lines (real and BLACK-BAR redacted), "
            "huge red 'REDACTED' angled stamp across the document, "
            "and a CASE No. tag in the corner. For 'this part has "
            "been blacked out' / 'the report was heavily censored' "
            "beats. Distinct from classified (which is the dossier "
            "wrapper) — redacted is the leaked-page payload."
        ),
        llm_rule=(
            "redacted -> narration explicitly mentions content "
            "that was CENSORED, REDACTED, BLACKED OUT, classified-"
            "out, or removed from a public document. t = the "
            "report title CAPS (<=36 chars, e.g. 'INTERNAL "
            "MEMORANDUM', 'REPORT 1971-04-12'); b = pipe-separated "
            "lines (3-6); use '#' to mark which words should be "
            "redacted (black bars), e.g. 'THE WITNESS REPORTED "
            "SEEING #SUBJECT-7# AT #LOCATION-9#|FOLLOW-UP CONTACT "
            "IS #PENDING#|CASE STATUS: #CLASSIFIED#'. Max 2/video."
        ),
        uses_body=True, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_redacted_card",
        assemble_branch="assemble._graphic_card_filters:redacted",
    ),
    _T(
        name="text_on_black",
        title="Text-on-Black / Cipher Card",
        description=(
            "Full-screen monospace WHITE text on a near-black field, "
            "where the raw text ITSELF is the subject: an intercepted "
            "transmission, a decrypted message, a coded cipher, a "
            "manifesto, a top-secret brief. One accent colours the "
            "header label + rule + cursor; faint scanlines + grain give "
            "a terminal / intercept texture. The investigative "
            "(LEMMiNO-style) 'read this' beat — for mystery, spy and "
            "true-crime decryption / message reveals."
        ),
        llm_rule=(
            "text_on_black -> the narration presents RAW TEXT that is "
            "itself the dramatic SUBJECT: an intercepted/decrypted "
            "transmission, a coded cipher or message, a manifesto, a "
            "top-secret brief — NOT a label over footage. Use VERY "
            "RARELY (max 1/video), only at the cryptic-text reveal beat "
            "of a mystery / spy / true-crime story. t = the header label "
            "CAPS (<=40 chars, e.g. 'DECRYPTED TRANSMISSION', 'THE "
            "CIPHER'); b = pipe-separated monospace lines (3-6 lines, "
            "each <=60 chars) — the actual message text."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=10,
        render_module="footage._render_text_on_black_card",
        assemble_branch="assemble._graphic_card_filters:text_on_black",
    ),

    # 19) AUDIO WAVEFORM — bottom-screen intercepted-audio strip
    _T(
        name="audio_waveform",
        title="Audio Waveform / Transcript",
        description=(
            "Bottom-screen 'we are now listening to a recording' "
            "overlay: stylized waveform bars in theme accent + "
            "small SPEAKER tag + the transcript line beneath. For "
            "witness statements, intercepted communications, voice "
            "memos, 911 calls — the 'this is what was said' beat. "
            "Transparent so footage shows above the strip."
        ),
        llm_rule=(
            "audio_waveform -> narration is quoting / playing / "
            "referencing an AUDIO RECORDING (witness statement, "
            "intercepted call, voicemail, 911 call, leaked tape, "
            "voice memo). t = optional 'SPEAKER, ROLE' tag (<=36 "
            "chars CAPS, e.g. 'WITNESS 03, AGE 51' or '911 CALL "
            "TRANSCRIPT'); b = the transcript text in caps "
            "(<=120 chars, e.g. 'I SAW THE GLOW FROM THE RIDGE "
            "AT ABOUT THREE IN THE MORNING'). If t is empty the "
            "tag becomes 'AUDIO TRANSCRIPT'. Max 3/video — only "
            "when an actual recorded statement is being relayed."
        ),
        uses_body=True, max_per_video=3, min_gap_scenes=4,
        render_module="footage._render_audio_waveform_card",
        assemble_branch="assemble._graphic_card_filters:audio_waveform",
    ),

    # 18) CHAPTER MARKER — cinematic act break full-screen
    _T(
        name="chapter_marker",
        title="Chapter / Act Marker",
        description=(
            "Full-screen act-break: huge centred chapter number "
            "(ACT II, PART 3, CHAPTER FIVE) over a dark navy/grid "
            "field, with chapter title beneath in TYPE_H1 and "
            "optional dateline at the bottom. Two accent rules "
            "bracket the title. Used between major narrative "
            "sections — the cinematic 'we're entering Act 2' beat."
        ),
        llm_rule=(
            "chapter_marker -> narration crosses a clear act/"
            "chapter boundary in the documentary (the investigation "
            "begins, years later, the second wave, the aftermath). "
            "t = the act number label CAPS (<=14 chars, e.g. "
            "'ACT II', 'PART 3', 'CHAPTER FIVE'); b = "
            "'TITLE[|SUBTITLE]' (title <=32 chars CAPS, optional "
            "subtitle <=40 chars CAPS), e.g. 'THE INVESTIGATION|"
            "1971-1980'. Use sparingly — max 2/video, only at "
            "GENUINE narrative breaks, not paragraph transitions."
        ),
        # DISABLED by request — the full-screen "PRESENTS / ACT I" act
        # break read as cheap. max_per_video=0 makes _apply_graphic_caps
        # drop every occurrence (0 >= 0 on the first hit). Re-enable by
        # restoring max_per_video=2 if a long doc needs act structure.
        uses_body=True, max_per_video=0, min_gap_scenes=12,
        render_module="footage._render_chapter_marker_card",
        assemble_branch="assemble._graphic_card_filters:chapter_marker",
    ),

    # 17) BREAKING NEWS — bottom-screen red banner + ticker
    _T(
        name="breaking_news",
        title="Breaking News Banner",
        description=(
            "TV-news bottom-screen breaking-news banner: large red "
            "BREAKING badge on the left, headline text in white "
            "block beside it, smaller dateline/source line above, "
            "and a thin red ticker strip below carrying a short "
            "rolling subhead. Transparent frame overlay so the "
            "footage stays visible above the banner. For 'BREAKING:' "
            "and 'JUST IN:' beats — sharper than news_broadcast "
            "(which is the full studio frame)."
        ),
        llm_rule=(
            "breaking_news -> narration delivers a HIGH-URGENCY "
            "news beat that should land like a network alert: "
            "'BREAKING:', 'JUST IN:', 'developing story', 'sources "
            "confirm', 'authorities have just announced'. t = the "
            "headline IN CAPS (<=58 chars, e.g. 'AUTHORITIES "
            "CONFIRM SECOND ATTACK', 'EVACUATION ORDER ISSUED'); "
            "b = optional 'BADGE|TICKER' (badge <=10 chars default "
            "'BREAKING'; ticker <=80 chars rolling subhead) — if "
            "no '|' the body becomes the ticker. Use SPARINGLY — "
            "max 2/video — only when the beat is genuinely a "
            "news-style escalation."
        ),
        uses_body=False, max_per_video=2, min_gap_scenes=6,
        render_module="footage._render_breaking_news_card",
        assemble_branch="assemble._graphic_card_filters:breaking_news",
    ),

    # 16) PROCESS DIAGRAM — labeled steps with arrows
    _T(
        name="process_diagram",
        title="Process / Mechanism Diagram",
        description=(
            "Documentary 'how it works' diagram: 3-4 labeled boxes "
            "arranged horizontally, connected by accent arrows "
            "(Step 1 → Step 2 → Step 3 → Step 4). Each box has a "
            "step number in accent, a short STEP TITLE, and a "
            "one-line description below. Same dark navy + grid bg "
            "family as the other data templates."
        ),
        llm_rule=(
            "process_diagram -> narration explicitly walks through "
            "a MECHANISM, PROCESS, or SEQUENCE of cause-and-effect "
            "steps (how a thing works, how it happened, the 4 "
            "stages of, the lifecycle of). t = headline CAPS "
            "(<=30 chars, e.g. 'HOW IT WORKS', 'THE PROCESS'); b "
            "= 3-4 entries separated by ';', each entry as "
            "'TITLE|DESCRIPTION' (title <=18 chars, description "
            "<=42 chars), e.g. 'DRILL|RIG HITS GAS POCKET;"
            "IGNITION|METHANE CATCHES FIRE;BURN|CRATER COLLAPSES;"
            "CONTINUE|STILL ON FIRE TODAY'. Use ONCE per video "
            "for the central explanation. Max 1/video."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=8,
        render_module="footage._render_process_diagram_card",
        assemble_branch="assemble._graphic_card_filters:process_diagram",
    ),

    # 14) CONSPIRACY INVESTIGATION BOARD — pinned cards + red string
    _T(
        name="conspiracy_board",
        title="Conspiracy Investigation Board",
        description=(
            "Detective/investigator cork-board overlay: 3-6 footage "
            "frames or document cards pinned at slight rotations to "
            "a brown corkboard texture, each with a red push-pin "
            "and short caption, linked by hand-drawn RED STRING "
            "connectors that map relationships between them. The "
            "'who-knew-whom' / 'how-it-all-connects' beat. Distinct "
            "from photo_collage (which is a clean evidence wall) — "
            "this one is messy, hand-pinned, with explicit links."
        ),
        llm_rule=(
            "conspiracy_board -> narration explicitly maps a "
            "RELATIONSHIP or NETWORK between 3-6 people/places/"
            "events that connect (the suspects who knew each "
            "other, the companies that shared a director, the "
            "victims linked by one location). t = headline in CAPS "
            "(<=30 chars, e.g. 'THE CONNECTIONS', 'WHO KNEW WHOM'); "
            "b = pipe-separated card captions (e.g. 'SUSPECT A|"
            "BANKER|SHELL CO|VICTIM 1') — order matters because "
            "consecutive cards get red-string links. Use ONCE per "
            "video for the 'it all connects' beat. Max 1/video."
        ),
        uses_body=True, max_per_video=1, min_gap_scenes=8,
        render_module="footage._render_conspiracy_board_card",
        assemble_branch="assemble._graphic_card_filters:conspiracy_board",
    ),
]

# Lookup by name + ordered list (preserves catalogue order for UI/prompt).
_REGISTRY: dict[str, Template] = {t.name: t for t in _TEMPLATES}


def get(kind: str) -> Template | None:
    """Return the Template spec for a graphic_kind name (or None)."""
    return _REGISTRY.get((kind or "").strip().lower())


def all_names() -> list[str]:
    """All registered template names in catalogue order."""
    return [t.name for t in _TEMPLATES]


def all_templates() -> list[Template]:
    """Every registered template, catalogue order."""
    return list(_TEMPLATES)


def body_kinds() -> set[str]:
    """Names of templates whose scene.graphic_body is meaningful."""
    return {t.name for t in _TEMPLATES if t.uses_body}


def caps() -> dict[str, int]:
    """{name: max_per_video} for templates that have a hard cap."""
    return {t.name: t.max_per_video for t in _TEMPLATES
            if t.max_per_video is not None}


def min_gaps() -> dict[str, int]:
    """{name: min_gap_scenes} for templates that need spacing."""
    return {t.name: t.min_gap_scenes for t in _TEMPLATES
            if t.min_gap_scenes > 0}


def llm_prompt_rules() -> str:
    """Concatenate every template's `llm_rule` as the per-kind block
    the editor-brain prompt feeds the model. Adding a new template
    here updates the prompt automatically — no edits to script_gen.py.
    """
    parts = ["* " + t.llm_rule for t in _TEMPLATES]
    return " ".join(parts)


def llm_kind_enum() -> str:
    """Pipe-separated list of valid graphic_kind values for the JSON
    schema in the LLM prompt (e.g. 'number|document|...|none')."""
    return "|".join(t.name for t in _TEMPLATES) + "|none"
