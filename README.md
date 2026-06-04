# Weed Knowledge Assistant 🌱

An AI-powered Weed Knowledge Assistant built using Retrieval-Augmented Generation (RAG).

## Features

* PDF-based knowledge retrieval
* OpenAI GPT-4o-mini integration
* LangChain framework
* Chroma Vector Database
* Hybrid Search (Keyword + Vector Search)
* Streamlit Chat Interface
* Source Page Citation

## Tech Stack

* Python
* React JS
* Fastapi
* OpenAI
* LangChain
* ChromaDB

## How It Works

1. Load Weed Management Guide PDF
2. Split content into chunks
3. Generate embeddings
4. Store embeddings in ChromaDB
5. Retrieve relevant chunks using Hybrid Search
6. Generate answers using GPT-4o-mini

## Project Architecture

User Question
→ Hybrid Retrieval
→ Chroma Vector Database
→ Relevant Context
→ GPT-4o-mini
→ Final Answer

**COMMANDS**
streamlit run app_ui.py

## Author

Kausikashree
