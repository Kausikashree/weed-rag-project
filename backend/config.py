import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = str(PROJECT_ROOT / "Nox_Weed_Management_Guide.pdf")
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db_ui")
COLLECTION_NAME = "weed_guide_clean_v2"
OPENAI_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
UNKNOWN_ANSWER = "I don't know based on the provided weed dataset."
