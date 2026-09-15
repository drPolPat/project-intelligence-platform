// Renders one query_standards result. Shared by the full-report view (per
// contributing factor) and the standalone standards search box, so citation
// styling never drifts between the two call sites. The three branches
// mirror the three shapes query_standards can return (see agent/tools.py) —
// do not collapse them, each needs visually distinct treatment.
export default function StandardsLookup({ lookup }) {
  if (lookup.status === 'answered') {
    return (
      <div className="standards-lookup standards-answered">
        <p className="standards-answer">{lookup.answer}</p>
        {lookup.citations?.map((citation, i) => (
          <div className="citation" key={i}>
            <span className="citation-ref">
              {citation.document_id} § {citation.section_id}
            </span>
            <blockquote className="citation-quote">&ldquo;{citation.supporting_quote}&rdquo;</blockquote>
          </div>
        ))}
      </div>
    )
  }

  if (lookup.status === 'refused') {
    return (
      <div className="standards-lookup standards-refused">
        <p className="standards-refused-label">No standards guidance found for this question</p>
        <p className="standards-refused-reason">Reason: {lookup.refusal_reason}</p>
        {lookup.model_reasoning && <p className="standards-refused-reasoning">{lookup.model_reasoning}</p>}
        {lookup.top_retrieved?.length > 0 && (
          <p className="standards-top-retrieved">Considered but not cited: {lookup.top_retrieved.join('; ')}</p>
        )}
      </div>
    )
  }

  // generation_skipped: no LLM API key configured for this run
  return (
    <div className="standards-lookup standards-skipped">
      <p className="standards-refused-label">Standards lookup unavailable</p>
      <p className="standards-refused-reason">{lookup.reason}</p>
      {lookup.top_retrieved?.length > 0 && (
        <p className="standards-top-retrieved">Top retrieved (not synthesized): {lookup.top_retrieved.join('; ')}</p>
      )}
    </div>
  )
}
