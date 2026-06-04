import landingImage from '/landing_page.png'

export default function LandingPage({ onStart }) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-b from-green-50 to-emerald-100 px-4">
      <h1 className="text-4xl md:text-5xl font-bold text-green-800 mb-8 text-center">
        Weed Knowledge Assistant
      </h1>
      <img
        src={landingImage}
        alt="Weed management"
        className="max-w-lg w-full rounded-2xl shadow-lg mb-10"
      />
      <button
        onClick={onStart}
        className="px-10 py-4 bg-emerald-600 hover:bg-emerald-700 text-white text-lg font-semibold rounded-xl shadow-md transition-colors"
      >
        Start Chat
      </button>
    </div>
  )
}
