export default function ChatMessage({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] md:max-w-[70%] rounded-2xl px-5 py-3 shadow-sm ${
          isUser
            ? 'bg-emerald-500 text-white rounded-br-md'
            : 'bg-white border border-gray-200 rounded-bl-md'
        }`}
      >
        <p className="text-sm md:text-base whitespace-pre-wrap">{message.content}</p>

        {message.sources && message.sources.length > 0 && (
          <details className="mt-2 text-xs text-gray-500">
            <summary className="cursor-pointer hover:text-gray-700">
              Sources ({message.sources.length})
            </summary>
            <ul className="mt-1 space-y-1">
              {message.sources.map((src, i) => (
                <li key={i} className="border-t border-gray-100 pt-1">
                  {src.page && (
                    <span className="font-medium">Page {src.page}: </span>
                  )}
                  <span>{src.content}</span>
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  )
}
