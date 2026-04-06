"""Run from project root: python scripts/injest_data.py"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.injest import load_documents_from_directory
from src.rag_system import RAGSystem


def setup_rag_db() -> None:
    raw_dir = _PROJECT_ROOT / "data" / "raw"
    data_dir = _PROJECT_ROOT / "data"

    try:
        documents = load_documents_from_directory(raw_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if not documents:
        print("No documents loaded from data/raw (add non-empty .pdf or .txt files).")
        sys.exit(1)

    rag = RAGSystem(data_dir)
    rag.setup_db_documents(documents)
    n_chunks = len(rag.documents)
    print(
        f"Done: {len(documents)} file(s), {n_chunks} chunk(s) -> {data_dir / 'vector_store'}"
    )


def main() -> None:
    setup_rag_db()


if __name__ == "__main__":
    main()
