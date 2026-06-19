export const meta = {
  name: 'mg-batch11-build',
  description: 'Build + micro-render 3 Batch-11 MG primitives in parallel (spotlight_object_hold, flowchart_decision, world_map_arc)',
  phases: [{ title: 'Build', detail: 'one agent per primitive: write file, compile, micro-render, extract frame' }],
}

const FF = '/Users/hussnain/Library/Python/3.9/lib/python/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'

const COMMON = [
  'You are building ONE premium motion-graphics primitive for the Vidlore engine (repo cwd = /Users/hussnain/Desktop/vidrush-clone).',
  '',
  'FIRST read these for the EXACT conventions (do not skip):',
  '- vidlore/motion_graphics/look.py  — the shared look API. You MUST use it.',
  '- vidlore/motion_graphics/charts/statistic_bar_reveal.py — the canonical primitive structure to mirror EXACTLY (module docstring; SPEC dict; a _coerce helper; the render() signature; the frame loop saving td/f{i:05d}.png; the ffmpeg encode block via vidlore.ffmpeg_tool.ffmpeg_exe(); look.cleanup_frames(td); the returned dict).',
  'Plus the reference noted in your specific task for layout ideas.',
  '',
  'Key look.py API (verify by reading look.py): palette(name)->dict keys bg_b/text/muted/accent/accent_hi; font(role,size) roles numeral/title->serif, label/caption->condensed; graded_background(w,h,pal,seed=,drift=); grade_media(img,pal,*,strength=); vignette(img,*,strength); film_grain(img,*,seed,amount,t); text_with_glow(text,font,fill=,glow=,glow_radius=,glow_alpha=,pad=)->RGBA; gold_fill(text,font,pal,**kw)->RGBA; paste_center(base,layer,*,cx,cy,scale=1,opacity=1) where layer MUST be RGBA; hairline(d,cx,y,half_w,pal); fade_alpha(t,dur,fps); ease_out_cubic / ease_in_out_cubic / ease_out_expo; cleanup_frames(td).',
  '',
  'HARD RULES:',
  '- Pure-local only (PIL + numpy -> ffmpeg). NO paid API, NO network. Deterministic given seed.',
  '- render() signature MUST be: render(out_path, *, <your inputs>, dur=6.0, fps=30, w=1920, h=1080, palette_name="amber_gold", layout="", seed=0, crf=18) -> dict',
  '- Frame loop: graded background each frame, draw on RGBA, save td/f{i:05d}.png, ffmpeg encode (libx264 crf, yuv420p, bt709, +faststart), look.cleanup_frames(td). Return {"ok","path","frames","dur_s","render_s","w","h","err"} exactly like statistic_bar_reveal.',
  '- Premium aesthetic: graded look, serif numerals/titles, restrained gold, clean reveals, vignette+grain. NO cheap/templated/generic/clip-art look. Editorial documentary feel.',
  '- DO NOT edit registry.py, director.py, render_dispatch.py, pipeline.py, or any __init__.py. Create ONLY your one primitive module file. Integration is handled separately.',
  '- Self-clean the PNG temp dir every run (look.cleanup_frames). Be defensive parsing inputs (None/str/list/dict). Clamp positions to keep content on-frame.',
  '',
  'AFTER writing the file:',
  '1) python3 -m py_compile <your file>   (must succeed)',
  '2) mkdir -p research/motion_graphics_qa/batch11',
  '3) micro-render to research/motion_graphics_qa/batch11/<NAME>.mp4 via:  python3 -c "import sys; sys.path.insert(0,\'.\'); from vidlore.motion_graphics.<FAMILY> import <NAME> as M; r=M.render(\'research/motion_graphics_qa/batch11/<NAME>.mp4\', <EXAMPLE_KWARGS>, dur=6.0, seed=4); print(r[\'ok\'], r[\'render_s\'], r[\'err\'][:120])"',
  '4) extract a late frame to research/motion_graphics_qa/batch11/<NAME>.jpg:  ' + FF + ' -nostdin -loglevel error -ss 5.2 -i research/motion_graphics_qa/batch11/<NAME>.mp4 -frames:v 1 -q:v 2 research/motion_graphics_qa/batch11/<NAME>.jpg -y',
  '5) confirm 0 stray temp dirs: find /tmp -maxdepth 2 -name \'f00000.png\' | wc -l   (should be 0)',
  '',
  'Return the StructuredOutput with the REAL results (run the commands, report actual numbers).',
  '',
].join('\n')

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    file_path: { type: 'string' },
    compiles: { type: 'boolean' },
    render_ok: { type: 'boolean' },
    render_s: { type: 'number' },
    frames: { type: 'integer' },
    frame_path: { type: 'string' },
    temp_leak: { type: 'integer' },
    design_notes: { type: 'string', description: 'one paragraph: layout, animation, palette, what makes it premium and distinct' },
  },
  required: ['file_path', 'compiles', 'render_ok', 'render_s', 'frames', 'frame_path', 'temp_leak', 'design_notes'],
}

const SPOTLIGHT = [
  'PRIMITIVE: spotlight_object_hold  (FAMILY = reveals)',
  'FILE: vidlore/motion_graphics/reveals/spotlight_object_hold.py',
  'Also read vidlore/motion_graphics/reveals/before_after_slider.py for family conventions.',
  '',
  'DESIGN — a dramatic "behold" reveal, TEXT-LED (NO photo frame — this MUST be clearly distinct from framed_evidence_spotlight, which is a static gold-framed photo card). A near-black stage. A soft circular SPOTLIGHT pool (a radial warm-gold gradient built with numpy, bright centre falling to black) SWEEPS in from an off-centre/edge start position and EASES (easeOutExpo) to settle at the frame centre over the first ~45% of the clip; as the light arrives it lifts the SUBJECT out of the darkness — a bold serif word/short phrase (gold_fill), fading up only as the light reaches it. A small condensed-caps KICKER label sits just above the subject (e.g. THE CULPRIT), an optional sub line just below, and an optional title on a hairline near the bottom. Everything outside the moving light pool is crushed to near-black with a heavy vignette; add a slow 3% push-in and faint film grain for a cinematic stage feel. The SIGNATURE is the MOVING spotlight that sweeps then holds — restraint and drama, not a busy card.',
  'SPEC: id "spotlight_object_hold", family "reveals", roles ["reveal","spotlight","subject","unveil","focus"], niches_ok ["crime","history","biography","geopolitics","business","tech"], intensity_range [3,5], duration_range [4.0,6.5], easing "easeOutExpo", audio_cue "soft_spot_swell", repeat_cooldown_s 55, per_video_cap 2, cost "low", layout_variants ["moving_spotlight"], review_override ["subject","kicker","sub","title","palette"], fallback "framed_evidence_spotlight if there is a real artifact image to show".',
  'render inputs: subject: str = "" (the revealed word/phrase, REQUIRED), kicker: str = "", sub: str = "", title: str = "". palette_name default "amber_gold".',
  'EXAMPLE_KWARGS: subject=\'THE BUTLER\', kicker=\'THE ONE WHO LIED\', sub=\'Seen in the library at midnight\', title=\'CASE CLOSED\'',
  'Use <FAMILY>=reveals, <NAME>=spotlight_object_hold.',
].join('\n')

const FLOWCHART = [
  'PRIMITIVE: flowchart_decision  (FAMILY = diagrams)',
  'FILE: vidlore/motion_graphics/diagrams/flowchart_decision.py',
  'Also read vidlore/motion_graphics/diagrams/process_flow_steps.py AND vidlore/motion_graphics/diagrams/cause_effect_chain.py — make this VISUALLY DISTINCT from both: process_flow is a linear numbered row; cause_effect is a linear domino row; THIS one BRANCHES (a fork).',
  '',
  'DESIGN — a yes/no decision fork: a QUESTION node at top-centre (a rounded rect or a diamond, gold-outlined, holding the question wrapped to <=2 condensed-caps lines). From its bottom, TWO gold connector lines DIVERGE down-left and down-right to two OUTCOME cards (rounded rects). Each branch line carries a small chip near its midpoint labelled with the branch label (default left YES, right NO). Reveal sequence: question node fades/scales in first, then the two branch lines DRAW downward to their corners with small arrowheads, then the YES/NO chips pop, then the two outcome cards rise+fade in. If a chosen ("yes" or "no") is supplied, brighten that whole path (line + chip + outcome card) to accent_hi/gold and dim the other; otherwise show both neutral-gold. Title rides a hairline at the very top. Graded bg, vignette, grain. Keep labels readable (wrap, shrink font for long text).',
  'SPEC: id "flowchart_decision", family "diagrams", roles ["decision","branch","flowchart","choice","fork"], niches_ok ["business","tech","crime","geopolitics","history","biography"], intensity_range [2,4], duration_range [4.5,7.0], easing "easeOutCubic", audio_cue "soft_branch_tick", repeat_cooldown_s 55, per_video_cap 2, cost "low", layout_variants ["yes_no_fork"], review_override ["question","yes","no","chosen","palette"], fallback "process_flow_steps if the logic is linear rather than a branch".',
  'render inputs: question: str = "" (REQUIRED), yes: str = "" (YES outcome label), no: str = "" (NO outcome label), yes_label: str = "YES", no_label: str = "NO", chosen: str = "" (one of yes / no / empty), title: str = "".',
  'EXAMPLE_KWARGS: question=\'Did the alibi hold up?\', yes=\'Released without charge\', no=\'Held for questioning\', chosen=\'no\', title=\'THE DECISION\'',
  'Use <FAMILY>=diagrams, <NAME>=flowchart_decision.',
].join('\n')

const WORLDARC = [
  'PRIMITIVE: world_map_arc  (FAMILY = maps)',
  'FILE: vidlore/motion_graphics/maps/world_map_arc.py',
  'Also read vidlore/motion_graphics/maps/map_route_spread.py AND vidlore/motion_graphics/maps/location_establish_card.py — REUSE the antique-map bed helper (from .location_establish_card import _antique_map) if present for a graded period world-map bed; make this DISTINCT from map_route_spread (which is a flat multi-stop polyline) — this is a single graceful great-circle ARC between TWO distant points.',
  '',
  'DESIGN — "across the world": a graded dark/period world-map bed (antique map helper, dimmed + vignetted). A glowing GOLD great-circle ARC sweeps from an ORIGIN point to a DESTINATION point: build it as a quadratic/cubic bezier that BOWS upward (control point lifted above the chord midpoint by ~18-25% of the chord length) so it reads as a curved global path, NOT a straight line. The arc DRAWS IN progressively (easeInOutCubic) with a bright comet/dot head riding its leading edge and a soft glow trail; pulsing pin dots sit at both endpoints with city NAME labels (serif/condensed, gold text_with_glow) placed clear of the arc; an optional title on a centred hairline near the bottom. Restraint: one arc, two pins, two labels — premium, atmospheric.',
  'SPEC: id "world_map_arc", family "maps", roles ["connection","arc","route","link","global"], niches_ok ["history","geopolitics","business","crime","biography","tech"], intensity_range [2,4], duration_range [4.5,7.0], easing "easeInOutCubic", audio_cue "soft_arc_swell", repeat_cooldown_s 55, per_video_cap 2, cost "low", layout_variants ["great_circle"], review_override ["from_place","to_place","from_pos","to_pos","map_image","palette"], fallback "map_route_spread if there are more than two waypoints".',
  'render inputs: from_place: str = "" (origin name, REQUIRED), to_place: str = "" (destination name, REQUIRED), from_pos=None (a "fx,fy" fraction string or [fx,fy]; default left), to_pos=None ("fx,fy"; default right), map_image=None, title: str = "". palette_name default "parchment_sepia". Parse pos defensively, clamp 0.06-0.94.',
  'EXAMPLE_KWARGS: from_place=\'London\', from_pos=\'0.49,0.33\', to_place=\'New York\', to_pos=\'0.27,0.42\', title=\'THE TRANSATLANTIC CABLE\'',
  'Use <FAMILY>=maps, <NAME>=world_map_arc.',
].join('\n')

const SPECS = [
  { label: 'spotlight_object_hold', prompt: COMMON + '\n' + SPOTLIGHT },
  { label: 'flowchart_decision', prompt: COMMON + '\n' + FLOWCHART },
  { label: 'world_map_arc', prompt: COMMON + '\n' + WORLDARC },
]

phase('Build')
const results = await parallel(
  SPECS.map(s => () => agent(s.prompt, { label: s.label, phase: 'Build', schema: SCHEMA }))
)
log('Batch-11 build complete: ' + results.filter(Boolean).length + '/3 primitives')
return results
