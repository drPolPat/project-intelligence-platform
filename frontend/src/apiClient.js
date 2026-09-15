const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function getJson(path) {
  const response = await fetch(`${API_BASE_URL}${path}`)
  const data = await response.json()
  if (!response.ok) {
    const detail = data.detail
    const message = Array.isArray(detail)
      ? detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join('; ')
      : detail || response.statusText
    throw new Error(message)
  }
  return data
}

async function postJson(path, body) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data = await response.json()
  if (!response.ok) {
    // FastAPI validation errors (422) return detail as an array of
    // {loc, msg, ...}; other errors (503/500) return detail as a plain
    // string. Normalize both into one readable message instead of
    // rendering "[object Object]".
    const detail = data.detail
    const message = Array.isArray(detail)
      ? detail.map((d) => `${d.loc?.at(-1)}: ${d.msg}`).join('; ')
      : detail || response.statusText
    throw new Error(message)
  }
  return data
}

export function predictRisk(features) {
  return postJson('/api/predict-risk', features)
}

export function draftReport(features) {
  return postJson('/api/draft-report', features)
}

export function queryStandards(question) {
  return postJson('/api/query-standards', { question })
}

export function getFacilities() {
  return getJson('/api/facilities')
}

export function sendChatMessage(message, sessionId) {
  return postJson('/api/chat', { message, session_id: sessionId ?? null })
}
