"""
store.py
--------
Minimal local vector store. No external DB needed - keeps everything in a
single .npz file plus a JSON metadata sidecar. Good enough for a personal
"memory index" of a few thousand chunks; swap for FAISS/sqlite-vec later
if you need to scale.
"""

import json
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORS_PATH = os.path.join(DATA_DIR, "vectors.npz")
META_PATH = os.path.join(DATA_DIR, "meta.json")


class VectorStore:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.vectors: np.ndarray | None = None
        self.meta: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(VECTORS_PATH) and os.path.exists(META_PATH):
            self.vectors = np.load(VECTORS_PATH)["vectors"]
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.vectors = np.zeros((0, 384), dtype="float32")  # 384 = MiniLM dim
            self.meta = []

    def save(self):
        np.savez_compressed(VECTORS_PATH, vectors=self.vectors)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def add(self, new_vectors: np.ndarray, new_meta: list[dict]):
        if self.vectors.shape[0] == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
        self.meta.extend(new_meta)
        self.save()

    def clear(self):
        self.vectors = np.zeros((0, 384), dtype="float32")
        self.meta = []
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.vectors.shape[0] == 0:
            return []
        scores = self.vectors @ query_vector  # cosine sim (vectors are normalized)
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for i in top_idx:
            item = dict(self.meta[i])
            item["score"] = float(scores[i])
            results.append(item)
        return results
