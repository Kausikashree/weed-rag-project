import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_MODEL,
    PDF_PATH,
    UNKNOWN_ANSWER,
)

load_dotenv()


def _clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace("colour", "color")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_keywords(question: str) -> list[str]:
    words = _clean_text(question).split()
    return [w for w in words if len(w) > 2 and w not in ENGLISH_STOP_WORDS]


def _keyword_search(documents, question: str, limit: int = 6):
    keywords = _get_keywords(question)
    if not keywords:
        return []
    scored = []
    for doc in documents:
        text = _clean_text(doc.page_content)
        score = sum(text.count(kw) for kw in keywords)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


def _merge_documents(*document_lists, max_docs: int = 8):
    merged = []
    seen = set()
    for doc_list in document_lists:
        for doc in doc_list:
            page = doc.metadata.get("page", "unknown")
            key = (page, doc.page_content[:300])
            if key not in seen:
                seen.add(key)
                merged.append(doc)
            if len(merged) >= max_docs:
                return merged
    return merged


def _format_context(documents) -> str:
    parts = []
    for i, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else "unknown"
        parts.append(f"[Source {i} | Page {page_number}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RAGEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        self._chunks = splitter.split_documents(pages)

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

        self._vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

        existing_ids = self._vector_store.get().get("ids", [])
        if len(existing_ids) == 0:
            ids = [f"chunk_{i}" for i in range(len(self._chunks))]
            self._vector_store.add_documents(self._chunks, ids=ids)

        self._retriever = self._vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 30, "lambda_mult": 0.3},
        )

        llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

        prompt = PromptTemplate.from_template(
            """You are a weed knowledge assistant for farmers.
Answer ONLY using the provided guide context.
Do NOT use outside knowledge.
Do NOT guess.
Do NOT add plant facts that are not present in the context.
If the answer is available in the context, answer in simple language.
If the answer is not available in the context, say exactly:
"I don't know based on the provided weed dataset."

Context:
{context}

Question:
{question}

Answer:"""
        )

        self._chain = prompt | llm

    def ask(self, question: str) -> tuple[str, list[dict]]:
        vector_docs = self._retriever.invoke(question)
        keyword_docs = _keyword_search(
            documents=self._chunks, question=question, limit=6
        )

        retrieved_docs = _merge_documents(keyword_docs, vector_docs, max_docs=8)
        context = _format_context(retrieved_docs)

        if not context.strip():
            return UNKNOWN_ANSWER, []

        response = self._chain.invoke({"context": context, "question": question})
        answer = response.content.strip()

        sources = []
        for doc in retrieved_docs:
            page = doc.metadata.get("page")
            sources.append(
                {
                    "content": doc.page_content[:300],
                    "page": page + 1 if isinstance(page, int) else None,
                }
            )

        return answer, sources
