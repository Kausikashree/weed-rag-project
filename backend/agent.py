import re
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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


MAX_ITERATIONS = 2


class AgentState(TypedDict):
    original_question: str
    current_query: str
    answer: str
    gathered_chunks: list[Document]
    evaluation: str | None
    missing_info: str | None
    sources: list[dict]
    iteration_count: int


class Evaluation(BaseModel):
    status: Literal["answerable", "needs_more"] = Field(
        description="'answerable' — facts are sufficient, 'needs_more' — need additional retrieval"
    )
    reasoning: str = Field(
        description="Brief explanation of what we know and what's missing"
    )
    missing_piece: str = Field(
        default="", description="If needs_more: describe the specific information gap"
    )


class Reformulation(BaseModel):
    new_query: str = Field(
        description="A specific search query to find the missing information."
    )
    rationale: str = Field(description="Why this query should fill the gap")


class AgenticRAG:
    def __init__(self, vector_retriever, chunks, llm, prompt_template):
        self._retriever = vector_retriever
        self._chunks = chunks
        self._llm = llm
        self._prompt = prompt_template
        self._graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("retrieve", self._retrieve_facts)
        workflow.add_node("evaluate", self._evaluate)
        workflow.add_node("reformulate", self._reformulate)
        workflow.add_node("respond", self._respond)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "evaluate")
        workflow.add_conditional_edges(
            "evaluate",
            self._evaluate_router,
            {"answerable": "respond", "needs_more": "reformulate"},
        )
        workflow.add_edge("reformulate", "retrieve")
        workflow.add_edge("respond", END)

        return workflow.compile()

    def invoke(self, question: str) -> tuple[str, list[dict]]:
        initial_state = {
            "original_question": question,
            "current_query": question,
            "answer": "",
            "gathered_chunks": [],
            "evaluation": None,
            "missing_info": None,
            "sources": [],
            "iteration_count": 0,
        }
        final_state = self._graph.invoke(initial_state)
        return final_state["answer"], final_state["sources"]

    def _retrieve_facts(self, state: AgentState):
        query = state["current_query"]
        gathered = state.get("gathered_chunks", []).copy()
        iteration = state.get("iteration_count", 0) + 1

        vector_docs = self._retriever.invoke(query)
        keyword_docs = _keyword_search(documents=self._chunks, question=query, limit=6)

        new_docs = _merge_documents(keyword_docs, vector_docs, max_docs=8)

        seen = set()
        for doc in gathered:
            page = doc.metadata.get("page", "unknown")
            key = (page, doc.page_content[:300])
            seen.add(key)

        for doc in new_docs:
            page = doc.metadata.get("page", "unknown")
            key = (page, doc.page_content[:300])
            if key not in seen:
                seen.add(key)
                gathered.append(doc)

        return {"gathered_chunks": gathered, "iteration_count": iteration}

    def _evaluate(self, state: AgentState):
        question = state["original_question"]
        gathered = state.get("gathered_chunks", [])

        eval_llm = self._llm.with_structured_output(Evaluation)

        context_summary = (
            "\n".join(f"- {doc.page_content[:200]}" for doc in gathered)
            if gathered
            else "(no relevant context found)"
        )

        result = eval_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an evaluator for a weed knowledge assistant. "
                        "Given a question and gathered context from a weed management guide, "
                        "determine if the context is sufficient.\n\n"
                        "Rules:\n"
                        "- If the question is a greeting (hi, hello, hey, thanks, good morning), mark 'answerable'.\n"
                        "- If the question mentions a weed name but isn't asking anything specific, mark 'answerable'.\n"
                        "- If the context contains relevant info to answer the question, mark 'answerable'.\n"
                        "- If some relevant context exists but a gap remains, mark 'needs_more' and describe what's missing.\n"
                        "- If no relevant context exists at all and it's a real question, mark 'answerable' (responder will say 'don't know')."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {question}\n\nGathered context:\n{context_summary}"
                    )
                ),
            ]
        )

        return {"evaluation": result.status, "missing_info": result.missing_piece}

    def _evaluate_router(
        self, state: AgentState
    ) -> Literal["answerable", "needs_more"]:
        if state.get("iteration_count", 0) >= MAX_ITERATIONS:
            return "answerable"
        return state["evaluation"] or "answerable"

    def _respond(self, state: AgentState):
        question = state["original_question"]
        gathered = state.get("gathered_chunks", [])
        context = _format_context(gathered)

        response = self._llm.invoke(
            self._prompt.format(context=context, question=question)
        )
        answer = response.content.strip()

        sources = []
        for doc in gathered:
            page = doc.metadata.get("page")
            sources.append(
                {
                    "content": doc.page_content[:300],
                    "page": page + 1 if isinstance(page, int) else None,
                }
            )

        return {"answer": answer, "sources": sources}

    def _reformulate(self, state: AgentState):
        question = state["original_question"]
        gathered = state.get("gathered_chunks", [])
        context_summary = (
            "\n".join(f"- {doc.page_content[:200]}" for doc in gathered)
            if gathered
            else "(none)"
        )
        missing = state.get("missing_info", "")

        reform_llm = self._llm.with_structured_output(Reformulation)
        result = reform_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a query reformulation specialist for a weed knowledge assistant. "
                        "Given the original question, gathered context, and identified gap, "
                        "create a NEW search query targeting the missing piece.\n\n"
                        "Rules:\n"
                        "- Make it different from what produced existing results.\n"
                        "- Use specific weed names or terms.\n"
                        "- Keep it under 10 words."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original question: {question}\n\n"
                        f"Gathered context:\n{context_summary}\n\n"
                        f"Information gap: {missing}"
                    )
                ),
            ]
        )

        return {"current_query": result.new_query}
