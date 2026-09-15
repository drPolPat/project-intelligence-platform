import { useState } from 'react'
import { queryStandards } from '../apiClient'
import StandardsLookup from './StandardsLookup'

export default function StandardsSearch() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await queryStandards(question))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>Standards Search</h2>
      <form onSubmit={handleSubmit} className="risk-form">
        <label>
          Question about the GFPA standards corpus
          <input
            type="text"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="e.g. What surveillance coverage is required for a museum?"
            required
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? 'Searching…' : 'Search standards'}
        </button>
      </form>

      {error && <div className="error-box">Error: {error}</div>}

      {result && (
        <div className="report">
          <StandardsLookup lookup={result} />
        </div>
      )}
    </div>
  )
}
