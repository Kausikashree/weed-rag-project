from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_engine import RAGEngine
from schemas import ChatRequest, ChatResponse, Source

app = FastAPI(title="Weed RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    engine = RAGEngine()
    print("[INFO] RAGEngine initialized successfully")
except Exception as e:
    engine = None
    import traceback

    print(f"[ERROR] RAGEngine init failed: {e}")
    traceback.print_exc()


@app.get("/api/health")
def health():
    return {"status": "ok", "rag_ready": engine is not None}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if engine is None:
        return ChatResponse(reply="RAG engine not available.", sources=[])

    answer, raw_sources = engine.ask(req.message)
    sources = [Source(**s) for s in raw_sources]
    return ChatResponse(reply=answer, sources=sources)
