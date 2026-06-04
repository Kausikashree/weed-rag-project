const API_BASE = '/api'

export async function sendMessage(message, sessionId) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  })
  if (!res.ok) {
    throw new Error(`Chat API error: ${res.status}`)
  }
  return res.json()
}

export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`)
  return res.json()
}
