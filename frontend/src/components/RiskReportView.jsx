import { FACTOR_LABELS } from '../facilityFields'
import { RISK_LEVEL_CLASS } from '../riskLevel'
import StandardsLookup from './StandardsLookup'

function FactorCard({ factor }) {
  const label = FACTOR_LABELS[factor.factor] ?? factor.factor
  return (
    <div className="factor-card">
      <div className="factor-header">
        <span className="factor-name">{label}</span>
        <span className="factor-value">value: {factor.value}</span>
        <span className="factor-score">contributing_score: {factor.contributing_score}</span>
      </div>
      <StandardsLookup lookup={factor.standards_lookup} />
    </div>
  )
}

export default function RiskReportView({ report }) {
  const { ml_result, top_contributing_factors } = report
  return (
    <div className="report">
      <div className={`result-box ${RISK_LEVEL_CLASS[ml_result.predicted_risk_level] ?? ''}`}>
        <div className="result-score">{ml_result.predicted_risk_score}</div>
        <div className="result-level">{ml_result.predicted_risk_level}</div>
        <div className="result-model">{ml_result.model}</div>
      </div>

      <div className="factor-list">
        {top_contributing_factors.map((factor) => (
          <FactorCard factor={factor} key={factor.factor} />
        ))}
      </div>

      <p className="result-caveat">{ml_result.caveat}</p>
    </div>
  )
}
