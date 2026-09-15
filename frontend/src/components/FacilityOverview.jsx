import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { getFacilities } from '../apiClient'
import { FACILITY_TYPES } from '../facilityFields'

const INSPECTION_RESULTS = ['Pass', 'Pass with Corrective Actions', 'Fail']
const INSPECTION_RESULT_COLORS = {
  Pass: '#2f9e44',
  'Pass with Corrective Actions': '#e8a400',
  Fail: '#e5484d',
}

// Ordered low-to-high severity so the stack reads as a Heinrich's-triangle
// pyramid: wide Near-Miss base at the bottom, narrow Critical tip on top.
const INCIDENT_SEVERITIES = ['Near-Miss', 'Minor', 'Moderate', 'Major', 'Critical']
const INCIDENT_SEVERITY_COLORS = {
  'Near-Miss': '#2f9e44',
  Minor: '#94d82d',
  Moderate: '#e8a400',
  Major: '#e8590c',
  Critical: '#e5484d',
}

// Explicit round-number ticks, same fix as the inspection chart's Y axis —
// letting recharts auto-derive ticks from a non-round max can produce
// floating-point artifacts like "100.10000000000001%".
function niceTicks(maxValue, step) {
  const ceiling = Math.ceil(maxValue / step) * step
  const ticks = []
  for (let v = 0; v <= ceiling; v += step) ticks.push(v)
  return ticks
}

// Recharts' default Legend does not reliably preserve declaration order for
// stacked bars (observed reordering alphabetically) — render it ourselves so
// the legend always reads in the same order as the stack.
function OrderedLegend({ order, colors }) {
  return (
    <ul className="chart-legend">
      {order.map((key) => (
        <li key={key}>
          <span className="chart-legend-swatch" style={{ background: colors[key] }} />
          {key}
        </li>
      ))}
    </ul>
  )
}

function aggregateRiskByFacilityType(riskAssessments) {
  const byType = new Map(FACILITY_TYPES.map((t) => [t, []]))
  for (const row of riskAssessments) {
    if (!byType.has(row.facility_type)) byType.set(row.facility_type, [])
    byType.get(row.facility_type).push(row.computed_risk_score)
  }

  return [...byType.entries()]
    .filter(([, scores]) => scores.length > 0)
    .map(([facility_type, scores]) => {
      const mean = scores.reduce((a, b) => a + b, 0) / scores.length
      const min = Math.min(...scores)
      const max = Math.max(...scores)
      return {
        facility_type,
        count: scores.length,
        mean: Number(mean.toFixed(1)),
        min,
        max,
        // ErrorBar takes distances from the bar's value, not absolute bounds.
        spread: [Number((mean - min).toFixed(1)), Number((max - mean).toFixed(1))],
      }
    })
}

function RiskByTypeTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-title">{d.facility_type}</div>
      <div>mean: {d.mean}</div>
      <div>range: {d.min}–{d.max}</div>
      <div>n = {d.count}</div>
    </div>
  )
}

function aggregateInspectionsByFacilityType(inspections) {
  const byType = new Map(FACILITY_TYPES.map((t) => [t, { Pass: 0, 'Pass with Corrective Actions': 0, Fail: 0 }]))
  for (const row of inspections) {
    if (!byType.has(row.facility_type)) {
      byType.set(row.facility_type, { Pass: 0, 'Pass with Corrective Actions': 0, Fail: 0 })
    }
    const bucket = byType.get(row.facility_type)
    if (row.overall_result in bucket) bucket[row.overall_result] += 1
  }

  return [...byType.entries()]
    .map(([facility_type, counts]) => {
      const total = INSPECTION_RESULTS.reduce((sum, k) => sum + counts[k], 0)
      if (total === 0) return null
      const pct = Object.fromEntries(
        INSPECTION_RESULTS.map((k) => [k, Number(((counts[k] / total) * 100).toFixed(1))]),
      )
      return { facility_type, total, counts, ...pct }
    })
    .filter(Boolean)
}

function InspectionResultTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-title">{d.facility_type}</div>
      {INSPECTION_RESULTS.map((k) => (
        <div key={k}>
          {k}: {d.counts[k]} ({d[k]}%)
        </div>
      ))}
      <div>n = {d.total}</div>
    </div>
  )
}

function aggregateIncidentsByCategory(incidents) {
  const byCategory = new Map()
  for (const row of incidents) {
    if (!byCategory.has(row.category)) {
      byCategory.set(row.category, Object.fromEntries(INCIDENT_SEVERITIES.map((s) => [s, 0])))
    }
    const bucket = byCategory.get(row.category)
    if (row.severity in bucket) bucket[row.severity] += 1
  }

  return [...byCategory.entries()]
    .map(([category, counts]) => {
      const total = INCIDENT_SEVERITIES.reduce((sum, s) => sum + counts[s], 0)
      return { category, total, ...counts }
    })
    .sort((a, b) => b.total - a.total)
}

function aggregateIncidentsByFacilityType(incidents) {
  const byType = new Map(FACILITY_TYPES.map((t) => [t, 0]))
  for (const row of incidents) {
    byType.set(row.facility_type, (byType.get(row.facility_type) ?? 0) + 1)
  }
  return [...byType.entries()].sort((a, b) => b[1] - a[1])
}

function IncidentSeverityTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-title">{d.category}</div>
      {INCIDENT_SEVERITIES.map((s) => (
        <div key={s}>
          {s}: {d[s]}
        </div>
      ))}
      <div>total = {d.total}</div>
    </div>
  )
}

export default function FacilityOverview() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    getFacilities()
      .then((result) => {
        if (!cancelled) setData(result)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const riskByType = useMemo(() => {
    if (!data) return null
    return aggregateRiskByFacilityType(data.facility_risk_assessments)
  }, [data])

  const inspectionsByType = useMemo(() => {
    if (!data) return null
    return aggregateInspectionsByFacilityType(data.safety_inspection_reports)
  }, [data])

  const incidentsByCategory = useMemo(() => {
    if (!data) return null
    return aggregateIncidentsByCategory(data.incident_reports)
  }, [data])

  const topIncidentFacilityType = useMemo(() => {
    if (!data) return null
    return aggregateIncidentsByFacilityType(data.incident_reports)[0]
  }, [data])

  const incidentYTicks = useMemo(() => {
    if (!incidentsByCategory) return null
    const maxTotal = Math.max(...incidentsByCategory.map((d) => d.total))
    return niceTicks(maxTotal, 25)
  }, [incidentsByCategory])

  return (
    <div className="card card-wide">
      <h2>Facility Overview</h2>

      {error && <div className="error-box">Error: {error}</div>}

      {!error && !data && <p className="chart-loading">Loading facility data…</p>}

      {riskByType && (
        <div className="chart-block">
          <h3>Risk score by facility type</h3>
          <p className="chart-subtitle">Mean computed_risk_score, error bars show min–max range</p>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={riskByType} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="facility_type" tick={{ fill: 'var(--text)', fontSize: 13 }} />
              <YAxis tick={{ fill: 'var(--text)', fontSize: 13 }} label={{ value: 'Risk score', angle: -90, position: 'insideLeft', fill: 'var(--text)' }} />
              <Tooltip content={<RiskByTypeTooltip />} />
              <Bar dataKey="mean" fill="var(--accent)" radius={[4, 4, 0, 0]}>
                <ErrorBar dataKey="spread" width={4} strokeWidth={1.5} stroke="var(--text-h)" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {inspectionsByType && (
        <div className="chart-block">
          <h3>Inspection results by facility type</h3>
          <p className="chart-subtitle">Share of inspections by overall_result</p>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={inspectionsByType} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="facility_type" tick={{ fill: 'var(--text)', fontSize: 13 }} />
              <YAxis
                domain={[0, 100]}
                ticks={[0, 25, 50, 75, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fill: 'var(--text)', fontSize: 13 }}
                width={64}
                label={{ value: '% of inspections', angle: -90, position: 'insideLeft', dx: -14, fill: 'var(--text)' }}
              />
              <Tooltip content={<InspectionResultTooltip />} />
              <Legend content={<OrderedLegend order={INSPECTION_RESULTS} colors={INSPECTION_RESULT_COLORS} />} />
              {INSPECTION_RESULTS.map((result, i) => (
                <Bar
                  key={result}
                  dataKey={result}
                  stackId="result"
                  fill={INSPECTION_RESULT_COLORS[result]}
                  radius={i === INSPECTION_RESULTS.length - 1 ? [4, 4, 0, 0] : undefined}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {incidentsByCategory && (
        <div className="chart-block">
          <h3>Incident volume by category and severity</h3>
          <p className="chart-subtitle">Incident count per category, stacked by severity</p>
          {topIncidentFacilityType && (
            <p className="chart-callout">
              {topIncidentFacilityType[0]} has the highest incident volume overall ({topIncidentFacilityType[1]}) —
              driven by its foot-traffic base rate, not a security gap (see Phase 1 data design).
            </p>
          )}
          <ResponsiveContainer width="100%" height={380}>
            <BarChart data={incidentsByCategory} margin={{ top: 8, right: 16, left: 46, bottom: 70 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                dataKey="category"
                angle={-30}
                textAnchor="end"
                interval={0}
                height={80}
                tick={{ fill: 'var(--text)', fontSize: 12 }}
              />
              <YAxis
                ticks={incidentYTicks}
                domain={[0, incidentYTicks[incidentYTicks.length - 1]]}
                tick={{ fill: 'var(--text)', fontSize: 13 }}
                width={40}
                label={{ value: 'Incident count', angle: -90, position: 'insideLeft', dx: -14, fill: 'var(--text)' }}
              />
              <Tooltip content={<IncidentSeverityTooltip />} />
              <Legend content={<OrderedLegend order={INCIDENT_SEVERITIES} colors={INCIDENT_SEVERITY_COLORS} />} />
              {INCIDENT_SEVERITIES.map((severity, i) => (
                <Bar
                  key={severity}
                  dataKey={severity}
                  stackId="severity"
                  fill={INCIDENT_SEVERITY_COLORS[severity]}
                  radius={i === INCIDENT_SEVERITIES.length - 1 ? [4, 4, 0, 0] : undefined}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
