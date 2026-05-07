"""
Vector store backing the RAG agent.

PDFs are loaded with PyPDFLoader, split with RecursiveCharacterTextSplitter,
embedded with OpenAI's text-embedding-3-small, and persisted to a local
ChromaDB collection on disk. Everything is local-first - no managed vector
service, nothing to provision before running.

A note on chunking: chunk_size=1000 with overlap=200 is the sweet spot for
policy text in our testing. Smaller chunks (~500) lose the surrounding context
needed to answer eligibility-style questions. Larger chunks (~2000) dilute
relevance scores and waste context window on the synthesis call. The 200-token
overlap is enough to keep paragraph-spanning ideas intact.
"""
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import (
    VECTOR_DB_PATH, VECTOR_COLLECTION_NAME,
    EMBEDDING_MODEL, OPENAI_API_KEY,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
)


class VectorStoreManager:
    """Wraps ChromaDB for PDF ingestion and similarity search."""

    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )
        # Separators are tried in order. Paragraph breaks first, then line
        # breaks, then sentence ends - keeps chunks readable when possible.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vector_store = Chroma(
            collection_name=VECTOR_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DB_PATH,
        )

    def ingest_pdf(self, pdf_path: str) -> int:
        """Load, chunk, embed, and index a PDF. Returns the chunk count."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        loader = PyPDFLoader(str(path))
        pages = loader.load()
        chunks = self.text_splitter.split_documents(pages)

        # Tag each chunk with the source filename so the RAG agent can cite
        # it back to the user. PyPDFLoader already adds 'page' metadata.
        for chunk in chunks:
            chunk.metadata.update({
                "source_file": path.name,
                "doc_type": "policy",
            })

        if chunks:
            self.vector_store.add_documents(chunks)

        return len(chunks)

    def search(self, query: str, k: int = TOP_K_RESULTS) -> List[Document]:
        """Top-k similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = TOP_K_RESULTS):
        """Top-k similarity search with relevance scores attached."""
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)

    def list_documents(self) -> List[str]:
        """Return the unique source filenames currently indexed."""
        try:
            data = self.vector_store.get()
            metadatas = data.get("metadatas", []) or []
            sources = {m.get("source_file", "unknown") for m in metadatas if m}
            return sorted(sources)
        except Exception:
            return []

    def remove_document(self, source_filename: str) -> int:
        """Delete every chunk belonging to one source PDF. Returns the
        number of chunks removed (0 if the file wasn't indexed)."""
        try:
            data = self.vector_store.get(where={"source_file": source_filename})
            ids = data.get("ids", [])
            if ids:
                self.vector_store.delete(ids=ids)
            return len(ids)
        except Exception as e:
            print(f"Warning: could not remove '{source_filename}' cleanly: {e}")
            return 0

    def is_indexed(self, source_filename: str) -> bool:
        """True if a PDF with this filename is already in the collection.
        Used by the UI to skip redundant re-embedding on duplicate uploads."""
        return source_filename in self.list_documents()

    def clear_all(self) -> None:
        """Drop every chunk in the collection. Useful for resetting state
        during development - not exposed in the UI."""
        try:
            data = self.vector_store.get()
            ids = data.get("ids", [])
            if ids:
                self.vector_store.delete(ids=ids)
        except Exception as e:
            print(f"Warning: could not clear vector store cleanly: {e}")


# Singleton accessor. Reused across requests so we don't re-instantiate
# the OpenAI client or re-open the Chroma collection on every call.
_instance: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    global _instance
    if _instance is None:
        _instance = VectorStoreManager()
    return _instance