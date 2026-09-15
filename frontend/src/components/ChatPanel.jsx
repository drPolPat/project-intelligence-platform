import { useEffect, useRef, useState } from 'react'
import { sendChatMessage } from '../apiClient'

// Backend tool function name -> short display label. The three tools
// mirror agent/agent_loop.py's TOOLS list.
const TOOL_LABELS = {
  predict_risk_score: 'Risk prediction',
  query_standards: 'Standards search',
  draft_risk_report: 'Full risk report',
}

export default function ChatPanel() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, loading])

  async function handleSend(event) {
    event.preventDefault()
    const text = input.trim()
    if (!text || loading) return

    setMessages((prev) => [...prev, { role: 'user', text }])
    setInput('')
    setLoading(true)
    setError(null)
    try {
      const data = await sendChatMessage(text, sessionId)
      setSessionId(data.session_id)
      setMessages((prev) => [...prev, { role: 'assistant', text: data.response, toolCalls: data.tool_calls }])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card card-wide">
      <h2>Chat</h2>

      <div className="chat-history">
        {messages.length === 0 && (
          <p className="chat-empty">Ask about facility risk, GFPA standards, or request a full report.</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-message chat-message-${m.role}`}>
            <div className="chat-bubble">{m.text}</div>
            {m.toolCalls?.length > 0 && (
              <div className="chat-tool-badges">
                {m.toolCalls.map((tc, j) => (
                  <span className="chat-tool-badge" key={j}>
                    Used: {TOOL_LABELS[tc.name] ?? tc.name}
                  </span>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-message chat-message-assistant">
            <div className="chat-bubble chat-bubble-loading">Thinking…</div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <div className="error-box">Error: {error}</div>}

      <form onSubmit={handleSend} className="chat-input-row">
        <input
          type="text"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Type a message…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? 'Sending…' : 'Send'}
        </button>
      </form>
    </div>
  )
}
