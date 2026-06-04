import { useState } from 'react'
import { useChat } from './hooks/useChat'
import LandingPage from './components/LandingPage'
import ChatInterface from './components/ChatInterface'

export default function App() {
  const [page, setPage] = useState('home')
  const { messages, loading, send, clearMessages } = useChat()

  if (page === 'home') {
    return <LandingPage onStart={() => setPage('chat')} />
  }

  return (
    <ChatInterface
      messages={messages}
      loading={loading}
      onSend={send}
      onBack={() => {
        clearMessages()
        setPage('home')
      }}
    />
  )
}
