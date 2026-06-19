export const meta = {
  name: 'ytal-music-curation',
  description: 'Curate the ingested YTAL CC-BY/official documentary music: refine category + niche tags, score documentary usefulness, reject weak filler, adversarially verify license completeness',
  phases: [
    { title: 'Curate', detail: 'parallel batches: refine category/niche, quality-score, flag filler' },
    { title: 'Verify', detail: 'adversarial license-completeness + restrictive-term check' },
  ],
}

// args.tracks = [{id,title,artist,category,niches17,quality_score,license,attribution_required,attribution,duration,tension,darkness,energy_arc}]
const tracks = (args && args.tracks) || []
if (!tracks.length) { log('no tracks passed — nothing to curate'); return { curated: [] } }

const CATS = ['suspense','mystery','dark_investigation','emotional_piano','ambient','historical_epic','military_tension','tech_cyber','financial','survival_urgency','slow_reveal','climax_build','aftermath','neutral','archive_texture']
const NICHES17 = ['spy','true_crime','dark_investigation','suspense','business','wealth','history','geopolitics','war_tension','reflective','emotional','atmospheric','intro_energy','body_bed','reveal','climax','outro']

const CURATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    decisions: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: {
        id: { type: 'string' },
        keep: { type: 'boolean' },
        weak_filler: { type: 'boolean' },
        refined_category: { type: 'string' },
        refined_niches: { type: 'array', items: { type: 'string' } },
        usefulness: { type: 'integer' },
        reason: { type: 'string' },
      },
      required: ['id', 'keep', 'weak_filler', 'refined_category', 'refined_niches', 'usefulness', 'reason'],
    } },
  },
  required: ['decisions'],
}

const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    flagged: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      properties: { id: { type: 'string' }, issue: { type: 'string' } },
      required: ['id', 'issue'],
    } },
    ok_count: { type: 'integer' },
  },
  required: ['flagged', 'ok_count'],
}

// batch the tracks (~12 per agent)
function chunk(a, n) { const o = []; for (let i = 0; i < a.length; i += n) o.push(a.slice(i, i + n)); return o }
const batches = chunk(tracks, 12)

phase('Curate')
const FACTS = [
  'You are a documentary music supervisor curating royalty-free / CC-BY tracks for a faceless-documentary generator.',
  'Valid musiclib categories: ' + CATS.join(', ') + '.',
  'Valid documentary niche/role tags (pick the applicable subset): ' + NICHES17.join(', ') + '.',
  'For each track decide: keep (documentary-useful, not generic filler) or drop (weak_filler=true for bland/loopy/amateur/wrong-genre).',
  'Refine the musiclib category + niche tags from the title + audio features. usefulness = 1 (drop) .. 5 (premium documentary bed).',
  'Be STRICT: the goal is a clean documentary catalog, not a big number. Drop EDM/pop/vlog-y/over-bright tracks; keep cinematic/tense/atmospheric/orchestral/piano/ambient beds.',
].join('\n')

const curated = await pipeline(
  batches,
  (batch, _orig, i) => agent([
    FACTS, '',
    'Curate this batch of ' + batch.length + ' tracks. Return a decision per id.',
    JSON.stringify(batch),
  ].join('\n'), { label: 'curate:batch' + i, phase: 'Curate', schema: CURATE_SCHEMA }),
)

const allDecisions = curated.filter(Boolean).flatMap(r => r.decisions || [])
const kept = allDecisions.filter(d => d.keep && !d.weak_filler)
log('curated: ' + kept.length + ' kept / ' + allDecisions.length + ' reviewed')

phase('Verify')
// adversarial license check on the kept set
const keptTracks = tracks.filter(t => kept.some(k => k.id === t.id))
const verify = await agent([
  'You are an adversarial licensing reviewer. The following tracks were ingested as CC BY 4.0 or official YTAL (royalty-free) for COMMERCIAL documentary use.',
  'Flag any track whose license/attribution looks INCOMPLETE or RESTRICTIVE (missing artist credit on a CC-BY track, non-commercial wording, or no clear free-use basis).',
  'Default to flagging when uncertain. Return flagged ids + the issue, and ok_count.',
  JSON.stringify(keptTracks.map(t => ({ id: t.id, title: t.title, artist: t.artist, license: t.license, attribution_required: t.attribution_required, attribution: t.attribution }))),
].join('\n'), { label: 'verify:license', phase: 'Verify', schema: VERIFY_SCHEMA })

const flaggedIds = new Set((verify.flagged || []).map(f => f.id))
const final = kept.filter(k => !flaggedIds.has(k.id))
log('license verify: ' + final.length + ' clean / ' + (verify.flagged || []).length + ' flagged')

return {
  curated: final,
  dropped_filler: allDecisions.filter(d => d.weak_filler || !d.keep).map(d => d.id),
  license_flagged: verify.flagged || [],
}
