import { useState } from 'react'
import { predictRisk, draftReport } from '../apiClient'
import { FACILITY_TYPES, NUMERIC_FIELDS } from '../facilityFields'
import { RISK_LEVEL_CLASS } from '../riskLevel'
import RiskReportView from './RiskReportView'

const EMPTY_FORM = {
  entry_points: '',
  surveillance_coverage_pct: '',
  staffing_level: '',
  past_incident_count: '',
  structural_vulnerabilities: '',
  facility_type: FACILITY_TYPES[0],
}

export default function FacilityRiskForm() {
  const [form, setForm] = useState(EMPTY_FORM)
  const [result, setResult] = useState(null)
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  function handleChange(event) {
    const { name, value } = event.target
    setForm((prev) => ({ ...prev, [name]: value }))
  }

  async function handleSubmit(event) {
    event.preventDefault()
    // Two submit buttons share this one form; the submitter's `value`
    // tells us which action the user asked for.
    const action = event.nativeEvent.submitter?.value ?? 'score'

    setLoading(true)
    setError(null)
    setResult(null)
    setReport(null)
    try {
      const payload = {
        entry_points: Number(form.entry_points),
        surveillance_coverage_pct: Number(form.surveillance_coverage_pct),
        staffing_level: Number(form.staffing_level),
        past_incident_count: Number(form.past_incident_count),
        structural_vulnerabilities: Number(form.structural_vulnerabilities),
        facility_type: form.facility_type,
      }
      if (action === 'report') {
        setReport(await draftReport(payload))
      } else {
        setResult(await predictRisk(payload))
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>Facility Risk Score</h2>
      <form onSubmit={handleSubmit} className="risk-form">
        {NUMERIC_FIELDS.map(({ name, label, min, max, step }) => (
          <label key={name}>
            {label}
            <input
              type="number"
              name={name}
              min={min}
              max={max}
              step={step}
              value={form[name]}
              onChange={handleChange}
              required
            />
          </label>
        ))}
        <label>
          Facility type
          <select name="facility_type" value={form.facility_type} onChange={handleChange}>
            {FACILITY_TYPES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>
        <div className="button-row">
          <button type="submit" name="action" value="score" disabled={loading}>
            {loading ? 'Working…' : 'Get risk score'}
          </button>
          <button type="submit" name="action" value="report" disabled={loading}>
            {loading ? 'Working…' : 'Get full report'}
          </button>
        </div>
      </form>

      {error && <div className="error-box">Error: {error}</div>}

      {result && (
        <div className={`result-box ${RISK_LEVEL_CLASS[result.predicted_risk_level] ?? ''}`}>
          <div className="result-score">{result.predicted_risk_score}</div>
          <div className="result-level">{result.predicted_risk_level}</div>
          <div className="result-model">{result.model}</div>
          <p className="result-caveat">{result.caveat}</p>
        </div>
      )}

      {report && <RiskReportView report={report} />}
    </div>
  )
}
