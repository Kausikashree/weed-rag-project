from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class Source(BaseModel):
    content: str
    page: int | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source]
