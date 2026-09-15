// Must match agent/tools.py's FACILITY_TYPES (itself sourced from
// generate_risk_assessments.py) — the frontend has no way to import that
// Python constant directly, so this is a deliberate, documented duplication.
export const FACILITY_TYPES = ['Museum', 'Mall', 'Office Tower', 'Utility Substation']

export const NUMERIC_FIELDS = [
  { name: 'entry_points', label: 'Entry points', min: 0, step: 1 },
  { name: 'surveillance_coverage_pct', label: 'Surveillance coverage (%)', min: 0, max: 100, step: 0.1 },
  { name: 'staffing_level', label: 'Staffing level', min: 0, step: 1 },
  { name: 'past_incident_count', label: 'Past incident count (12 mo.)', min: 0, step: 1 },
  { name: 'structural_vulnerabilities', label: 'Structural vulnerabilities', min: 0, step: 1 },
]

// factor key -> human label, reused by the full-report view's factor cards
// so a raw JSON key like "surveillance_coverage_pct" isn't shown verbatim —
// this is a presentational label for a field NAME, not a reformatting of
// any grounding-guaranteed prose/quote content.
export const FACTOR_LABELS = Object.fromEntries(NUMERIC_FIELDS.map((f) => [f.name, f.label]))
