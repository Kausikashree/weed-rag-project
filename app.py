from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# -----------------------------
# 1. Load PDF
# -----------------------------
loader = PyPDFLoader("Nox_Weed_Management_Guide.pdf")
documents = loader.load()

# -----------------------------
# 2. Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(documents)

# -----------------------------
# 3. Create embeddings
# -----------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# -----------------------------
# 4. Vector DB (Chroma)
# -----------------------------
vector_store = Chroma(
    collection_name="weed_knowledge_base",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

# Store documents
vector_store.add_documents(chunks)

# -----------------------------
# 5. Retriever (MMR)
# -----------------------------
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10
    }
)

# -----------------------------
# 6. LLM
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -----------------------------
# 7. Prompt Template
# -----------------------------
prompt_template = """
Answer the question ONLY using the context below.

If the answer is not present in the context, say:
"I don't know based on the provided weed dataset."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)

# Create chain once (IMPORTANT optimization)
chain = prompt | llm

# -----------------------------
# 8. Chat Loop
# -----------------------------
print("\n🌱 Weed RAG Chatbot (type 'exit' to quit)\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit", "bye"]:
        print("Bot: Bye 👋 Stay weed-aware 🌿")
        break

    # -------------------------
    # Retrieve relevant chunks
    # -------------------------
    documents = retriever.invoke(question)

    # Remove duplicate chunks
    unique_docs = []
    seen = set()

    for doc in documents:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    documents = unique_docs

    # -------------------------
    # Build context
    # -------------------------
    context = "\n\n".join(doc.page_content for doc in documents)

    # -------------------------
    # Run LLM
    # -------------------------
    response = chain.invoke({
        "context": context,
        "question": question
    })

    # -------------------------
    # Output
    # -------------------------
    print("\nFINAL ANSWER:\n")
    print(response.content)
    print("\n" + "-" * 50 + "\n")