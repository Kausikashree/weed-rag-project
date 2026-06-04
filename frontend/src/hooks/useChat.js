import { useState, useCallback, useEffect } from 'react'
import { sendMessage } from '../api/chat'

function getSessionId() {
  let id = localStorage.getItem('session_id')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('session_id', id)
  }
  return id
}

export function useChat() {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [sessionId] = useState(getSessionId)

  const appendMessage = useCallback((role, content, sources) => {
    setMessages(prev => [...prev, { role, content, sources: sources || [] }])
  }, [])

  const send = useCallback(async (text) => {
    if (!text.trim() || loading) return

    appendMessage('user', text)
    setLoading(true)

    try {
      const data = await sendMessage(text, sessionId)
      appendMessage('assistant', data.reply, data.sources)
    } catch {
      appendMessage('assistant', 'Sorry, something went wrong. Please try again.')
    } finally {
      setLoading(false)
    }
  }, [loading, sessionId, appendMessage])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return { messages, loading, send, clearMessages }
}
