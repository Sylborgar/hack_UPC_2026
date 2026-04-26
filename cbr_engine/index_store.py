from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


class SklearnIndexStore:
    backend_name = "sklearn"

    def __init__(self):
        self.index: NearestNeighbors | None = None
        self.matrix: np.ndarray | None = None

    def build(self, matrix: np.ndarray) -> "SklearnIndexStore":
        self.matrix = matrix.astype(np.float32)
        self.index = NearestNeighbors(metric="cosine", algorithm="brute")
        self.index.fit(self.matrix)
        return self

    def search(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        distances, idx = self.index.kneighbors(vectors.astype(np.float32), n_neighbors=min(k, len(self.matrix)))
        return 1.0 - distances.astype(np.float32), idx.astype(np.int64)

    def save(self, path: str | Path) -> None:
        joblib.dump(self.index, Path(path) / "index_sklearn.joblib")

    def load(self, path: str | Path, matrix: np.ndarray) -> "SklearnIndexStore":
        self.matrix = matrix.astype(np.float32)
        self.index = joblib.load(Path(path) / "index_sklearn.joblib")
        return self


class FaissIndexStore:
    backend_name = "faiss"

    def __init__(self):
        self.index = None
        self.matrix: np.ndarray | None = None

    def build(self, matrix: np.ndarray) -> "FaissIndexStore":
        import faiss

        self.matrix = matrix.astype(np.float32)
        self.index = faiss.IndexFlatIP(self.matrix.shape[1])
        self.index.add(self.matrix)
        return self

    def search(self, vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, idx = self.index.search(vectors.astype(np.float32), min(k, self.index.ntotal))
        return scores.astype(np.float32), idx.astype(np.int64)

    def save(self, path: str | Path) -> None:
        import faiss

        faiss.write_index(self.index, str(Path(path) / "index.faiss"))

    def load(self, path: str | Path, matrix: np.ndarray) -> "FaissIndexStore":
        import faiss

        self.matrix = matrix.astype(np.float32)
        self.index = faiss.read_index(str(Path(path) / "index.faiss"))
        return self


def make_index_store(backend: str = "faiss"):
    if backend == "faiss":
        try:
            import faiss  # noqa: F401

            return FaissIndexStore()
        except Exception:
            return SklearnIndexStore()
    return SklearnIndexStore()

