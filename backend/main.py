"""
main.py
-------
MemoryOS backend API.

Run:
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000

Endpoints:
    POST /index    {"directory": "/path/to/folder"}   -> indexes files
    POST /search   {"query": "...", "top_k": 5}        -> semantic search
    GET  /timeline                                     -> recent activity view
    GET  /health                                       -> sanity check
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import defaultdict

from embeddings import embed_query
from store import VectorStore
from indexer import index_directory

app = FastAPI(title="MemoryOS API")

# Allow the frontend (served separately or opened as a local file) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    directory: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def index(req: IndexRequest):
    result = index_directory(req.directory)
    return result


@app.post("/search")
def search(req: SearchRequest):
    store = VectorStore()
    qvec = embed_query(req.query)
    results = store.search(qvec, top_k=req.top_k)
    return {"query": req.query, "results": results}


@app.get("/timeline")
def timeline():
    """Group indexed chunks by file, ordered by most-recently-modified file first.
    This powers the 'what did I work on last week?' demo moment."""
    store = VectorStore()
    by_file = defaultdict(lambda: {"file_name": "", "modified": "", "chunks": 0})
    for item in store.meta:
        f = item["file"]
        by_file[f]["file_name"] = item["file_name"]
        by_file[f]["modified"] = item["modified"]
        by_file[f]["chunks"] += 1

    files = sorted(by_file.values(), key=lambda x: x["modified"], reverse=True)
    return {"files": files}
