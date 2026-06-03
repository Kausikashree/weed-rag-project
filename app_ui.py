import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# -------------------------------------------------
# Basic setup
# -------------------------------------------------

load_dotenv()

PDF_PATH = "Nox_Weed_Management_Guide.pdf"
CHROMA_DIR = "chroma_db_ui"
COLLECTION_NAME = "weed_guide_clean_v2"

UNKNOWN_ANSWER = "I don't know based on the provided weed dataset."

st.set_page_config(page_title="Weed RAG Chat", layout="wide")

# -------------------------------------------------
# Generic helper functions
# -------------------------------------------------



def clean_text(text):
    text = text.lower()
    text = text.replace("colour", "color")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_keywords(question):
    words = clean_text(question).split()
    return [
        word for word in words
        if len(word) > 2 and word not in ENGLISH_STOP_WORDS
    ]

def keyword_search(documents, question, limit=6):
    keywords = get_keywords(question)
    if not keywords:
        return []
    scored_documents = []
    for doc in documents:
        text = clean_text(doc.page_content)
        score = sum(text.count(keyword) for keyword in keywords)
        if score > 0:
            scored_documents.append((score, doc))
    scored_documents.sort(key=lambda item: item[0], reverse=True)
    return [doc for score, doc in scored_documents[:limit]]

def merge_documents(*document_lists, max_docs=8):
    merged = []
    seen = set()
    for document_list in document_lists:
        for doc in document_list:
            page = doc.metadata.get("page", "unknown")
            content_key = doc.page_content[:300]
            key = (page, content_key)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
            if len(merged) >= max_docs:
                return merged
    return merged

def format_context(documents):
    context_parts = []
    for index, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else "unknown"
        context_parts.append(
            f"[Source {index} | Page {page_number}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)

# -------------------------------------------------
# Build RAG system
# -------------------------------------------------

@st.cache_resource(show_spinner="Loading weed guide...")
def setup_rag():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}. Keep it in the same folder as app_ui.py."
        )
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    existing_ids = vector_store.get().get("ids", [])
    if len(existing_ids) == 0:
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        vector_store.add_documents(chunks, ids=ids)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30,
            "lambda_mult": 0.3
        }
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
You are a weed knowledge assistant for farmers.
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

Answer:
""")

    chain = prompt | llm

    return chunks, retriever, chain

try:
    chunks, retriever, chain = setup_rag()
except Exception as error:
    st.error(str(error))
    st.stop()

# -------------------------------------------------
# Session state
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "page" not in st.session_state:
    st.session_state.page = "home"

# -------------------------------------------------
# Home Page
# -------------------------------------------------

if st.session_state.get("page", "home") == "home":
    st.markdown(
        """
        <h1 style='text-align:center;'>
            🌱 Weed Knowledge Assistant
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.image(
        "landing_page.png",
        use_container_width=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    center1, center2, center3 = st.columns([1, 2, 1])
    with center2:
        if st.button(
            "Start Chat",
            use_container_width=True
        ):
            st.session_state.page = "chat"
            st.rerun()
    st.stop()

# -------------------------------------------------
# Display previous chat
# -------------------------------------------------

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):

          st.success(message["content"])

# -------------------------------------------------
# Chat input
# -------------------------------------------------

user_question = st.chat_input("Ask about weeds...")

if user_question:
    with st.spinner("Searching the guide..."):

        vector_docs = retriever.invoke(user_question)
        keyword_docs = keyword_search(
            documents=chunks,
            question=user_question,
            limit=6
        )

        retrieved_docs = merge_documents(
            keyword_docs,
            vector_docs,
            max_docs=8
        )

        context = format_context(retrieved_docs)

        if not context.strip():
            answer = UNKNOWN_ANSWER
        else:
            response = chain.invoke({
                "context": context,
                "question": user_question
            })

            answer = response.content.strip()

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })

    st.rerun()