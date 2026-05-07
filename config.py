"""
Project-wide configuration.

Reads from environment variables (with sensible defaults) so the same code
runs in development, in CI, and on a teammate's machine without edits. Any
deployment-specific value - API keys, database paths, model names - can be
overridden via the .env file at the project root.
"""
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


# --- paths ---

PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"

SQL_DB_PATH    = os.getenv("SQL_DB_PATH",    str(DATA_DIR / "customer_support.db"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "chroma_db"))
PDF_UPLOAD_PATH = os.getenv("PDF_UPLOAD_PATH", str(DATA_DIR / "uploaded_pdfs"))

# Create data directories at import time so first-run setup doesn't fail
# on a missing folder. Cheap; idempotent.
DATA_DIR.mkdir(exist_ok=True)
Path(VECTOR_DB_PATH).mkdir(exist_ok=True, parents=True)
Path(PDF_UPLOAD_PATH).mkdir(exist_ok=True, parents=True)


# --- LLM ---

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# --- SQL ---

# SQLAlchemy URI form, in case any caller wants to use SQLAlchemy directly.
# The agent's tools talk to sqlite3 instead, but exposing both keeps the
# config portable.
SQL_DB_URL = f"sqlite:///{SQL_DB_PATH}"


# --- RAG / chunking ---

# See utils/vector_store.py for the rationale behind these values.
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
TOP_K_RESULTS  = 4
VECTOR_COLLECTION_NAME = "policy_documents"