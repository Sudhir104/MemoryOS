"""
embeddings.py
-------------
Wraps a small local sentence-embedding model.

Model: sentence-transformers/all-MiniLM-L6-v2
- Only ~90MB, fast on CPU, and has a ready ONNX/QNN path via Qualcomm AI Hub
  for NPU deployment on Snapdragon devices later.
- On a non-Snapdragon dev machine (e.g. Dell G15) this runs on CPU/GPU via
  PyTorch. When you move to a real Snapdragon HP PC (or Qualcomm AI Hub's
  remote device access), swap load_model() to load the exported ONNX/QNN
  version instead - the rest of the app doesn't need to change.
"""

from functools import lru_cache
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of strings -> (N, D) float32 numpy array, L2-normalized."""
    model = load_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    return (vectors / norms).astype("float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
