# MemoryOS

**Your laptop remembers your digital life — privately, on-device.**

Built for the Qualcomm Snapdragon® AI Lab Build & Present Challenge.

## The problem

Personal knowledge tools ("where did I read this?", "what did I work on
last week?") usually mean uploading your notes, PDFs and documents to a
cloud AI. That's a privacy trade-off nobody should have to make just to
get a good search experience.

## What it does

MemoryOS indexes your local files (PDFs, notes, markdown) using an
on-device embedding model, and lets you ask natural-language questions
about your own past work — entirely locally. Nothing leaves the device.

- "Where did I see this concept before?"
- "Show me everything related to TCP congestion control."
- "What did I work on last week?" (via the timeline view)

## Architecture

```
frontend/index.html  →  FastAPI backend (backend/main.py)
                              │
                    embeddings.py (all-MiniLM-L6-v2)
                              │
                    store.py (local vector store, cosine similarity)
                              │
                    indexer.py (walks a folder, extracts + chunks text)
```

## Snapdragon / on-device AI story

- The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is small
  enough to export to ONNX and run via QNN on the Snapdragon NPU.
- Development was done on a non-Snapdragon machine; on-device performance
  was validated using **Qualcomm AI Hub's remote Snapdragon device
  access**, since we didn't have Snapdragon hardware in hand.
- No file content or query is ever sent to a cloud API — indexing,
  embedding, and search all happen on-device.

## Running it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser (or serve it with any
static file server). Point the "Index a folder" box at a folder of your
own PDFs/notes and search away.

## Roadmap / what's next

- OCR support for indexing screenshots
- Code file indexing (functions/comments as separate chunks)
- Swap the flat vector store for FAISS once the index grows large
- On-device local LLM (via Qualcomm AI Hub) to synthesize a natural-
  language answer from the top search results, not just show snippets
