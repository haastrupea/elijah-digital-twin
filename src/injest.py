from pathlib import Path
from typing import Dict
from pypdf import PdfReader


def _doc_id_for_path(path: Path) -> str:
    """Stable id from filename: e.g. about.txt -> about_txt (unique per file in a flat folder)."""
    return path.name.replace(".", "_")


def load_pdf(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def load_text_file(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents_from_directory(raw_dir: Path | str) -> Dict[str, str]:
    """Load all .pdf and .txt files from a directory (non-recursive). Skips files that fail to load."""
    base = Path(raw_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist or is not a directory: {base.resolve()}")

    documents: Dict[str, str] = {}

    for path in sorted(base.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in (".pdf", ".txt"):
            continue

        base_id = _doc_id_for_path(path)
        doc_id = base_id
        n = 2
        while doc_id in documents:
            doc_id = f"{base_id}_{n}"
            n += 1

        try:
            if suffix == ".pdf":
                text = load_pdf(path)
            else:
                text = load_text_file(path)
            if not text.strip():
                print(f"[SKIP] Empty content: {path.name}")
                continue
            documents[doc_id] = text
            print(f"[OK] Loaded {path.name} -> {doc_id}: {len(text)} chars")
        except Exception as e:
            print(f"[ERROR] {path.name}: {e}")

    return documents

