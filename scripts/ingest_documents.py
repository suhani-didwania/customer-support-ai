"""
Bulk-ingest every PDF in the uploads directory into the vector store.

Run this after dropping new policy PDFs into data/uploaded_pdfs/, or to
rebuild the index from scratch (delete data/chroma_db/ first).

Usage:
    python scripts/ingest_documents.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PDF_UPLOAD_PATH
from utils.vector_store import get_vector_store


def main() -> None:
    folder = Path(PDF_UPLOAD_PATH)
    pdfs = sorted(folder.glob("*.pdf"))

    if not pdfs:
        print(f"\nNo PDFs found in {folder}.")
        print("Drop your policy PDFs in that folder and re-run.\n")
        return

    print(f"\nFound {len(pdfs)} PDF(s) in {folder}\n")
    vs = get_vector_store()

    total_chunks = 0
    for pdf in pdfs:
        print(f"  {pdf.name}")
        try:
            n = vs.ingest_pdf(str(pdf))
            print(f"    indexed {n} chunks")
            total_chunks += n
        except Exception as e:
            print(f"    error: {e}")

    print(f"\nDone. {total_chunks} chunks indexed across {len(pdfs)} document(s).\n")
    print("Currently in the index:")
    for d in vs.list_documents():
        print(f"  - {d}")
    print()


if __name__ == "__main__":
    main()