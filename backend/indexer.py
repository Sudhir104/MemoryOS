"""
indexer.py
----------
Walks a directory, pulls text out of supported files, splits it into
overlapping chunks, embeds the chunks, and writes them into the VectorStore.

Supported for the MVP: .txt, .md, .pdf
Add more extractors (docx, code files, screenshots via OCR) once this
core loop is proven end-to-end.
"""

import os
import time
from datetime import datetime

from embeddings import embed_texts
from store import VectorStore

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150


def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(path)
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        else:  # .txt, .md
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print(f"  [skip] could not read {path}: {e}")
        return ""


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def index_directory(directory: str) -> dict:
    """Scan `directory`, embed all supported files, add to the store."""
    store = VectorStore()
    all_vectors = []
    all_meta = []
    files_indexed = 0
    chunks_indexed = 0
    t0 = time.time()

    for root, _, files in os.walk(directory):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTS:
                continue
            fpath = os.path.join(root, fname)
            text = extract_text(fpath)
            chunks = chunk_text(text)
            if not chunks:
                continue

            mtime = os.path.getmtime(fpath)
            vectors = embed_texts(chunks)
            for i, chunk in enumerate(chunks):
                all_meta.append({
                    "file": fpath,
                    "file_name": fname,
                    "chunk_index": i,
                    "snippet": chunk[:300],
                    "modified": datetime.fromtimestamp(mtime).isoformat(),
                })
            all_vectors.append(vectors)
            files_indexed += 1
            chunks_indexed += len(chunks)
            print(f"  indexed {fname} ({len(chunks)} chunks)")

    if all_vectors:
        import numpy as np
        stacked = np.vstack(all_vectors)
        store.add(stacked, all_meta)

    return {
        "files_indexed": files_indexed,
        "chunks_indexed": chunks_indexed,
        "seconds": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    result = index_directory(target)
    print(result)
