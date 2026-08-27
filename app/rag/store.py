"""Vector stores: Chroma (when installed) with a zero-dependency NumPy fallback."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.config import settings

logger = logging.getLogger("nyaya.store")


@dataclass
class Chunk:
    id: str
    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    def set_meta(self, meta: dict) -> None: ...
    def get_meta(self) -> dict: ...
    def count(self) -> int: ...
    def reset(self) -> None: ...
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def query(self, embedding: list[float], k: int = 10) -> list[tuple[Chunk, float]]: ...
    def all_chunks(self) -> list[Chunk]: ...


class NumpyStore(VectorStore):
    """Cosine-similarity store persisted as .npy + .json. Fine for small corpora."""

    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._meta_path = os.path.join(directory, "meta.json")
        self._chunks_path = os.path.join(directory, "chunks.json")
        self._vectors_path = os.path.join(directory, "vectors.npy")
        self._lock = threading.Lock()
        self._matrix: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._meta: dict = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._chunks_path) and os.path.exists(self._vectors_path):
            try:
                self._matrix = np.load(self._vectors_path)
                with open(self._chunks_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                self._chunks = [Chunk(**item) for item in raw]
            except Exception as exc:
                logger.warning("NumpyStore load failed, starting empty: %s", exc)
                self._matrix, self._chunks = None, []
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "r", encoding="utf-8") as fh:
                    self._meta = json.load(fh)
            except Exception:
                self._meta = {}

    def _persist(self) -> None:
        with open(self._chunks_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(c) for c in self._chunks], fh, ensure_ascii=False)
        if self._matrix is not None:
            np.save(self._vectors_path, self._matrix)
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, ensure_ascii=False)

    def set_meta(self, meta: dict) -> None:
        self._meta = dict(meta)
        self._persist()

    def get_meta(self) -> dict:
        return dict(self._meta)

    def count(self) -> int:
        return len(self._chunks)

    def reset(self) -> None:
        with self._lock:
            self._matrix, self._chunks = None, []
            self._persist()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        with self._lock:
            vectors = np.array(embeddings, dtype=np.float32)
            if self._matrix is None:
                self._matrix = vectors
                self._chunks = list(chunks)
            else:
                if vectors.shape[1] != self._matrix.shape[1]:
                    raise ValueError(
                        f"Embedding dimension mismatch: store={self._matrix.shape[1]} "
                        f"new={vectors.shape[1]}. Re-index the knowledge base."
                    )
                self._matrix = np.vstack([self._matrix, vectors])
                self._chunks.extend(chunks)
            self._persist()

    def query(self, embedding: list[float], k: int = 10) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            return []
        query_vec = np.array(embedding, dtype=np.float32)
        if query_vec.shape[0] != self._matrix.shape[1]:
            raise ValueError("Embedding dimension mismatch — re-index the knowledge base.")
        mat = self._matrix
        norm_q = np.linalg.norm(query_vec) or 1.0
        norm_m = np.linalg.norm(mat, axis=1)
        norm_m[norm_m == 0] = 1.0
        sims = (mat @ query_vec) / (norm_m * norm_q)
        k = min(k, len(self._chunks))
        top_idx = np.argsort(-sims)[:k]
        return [(self._chunks[i], float(sims[i])) for i in top_idx]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)


class ChromaStore(VectorStore):
    """Chroma persistent store (spec default for development)."""

    def __init__(self, directory: str) -> None:
        import chromadb  # imported lazily; optional dependency

        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._meta_path = os.path.join(directory, "nyaya_meta.json")
        self.client = chromadb.PersistentClient(path=directory)
        self.collection = self.client.get_or_create_collection(
            name="nyaya_legal", metadata={"hnsw:space": "cosine"}
        )

    def _read_meta(self) -> dict:
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                return {}
        return {}

    def _write_meta(self, meta: dict) -> None:
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)

    def set_meta(self, meta: dict) -> None:
        self._write_meta(meta)

    def get_meta(self) -> dict:
        return self._read_meta()

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection("nyaya_legal")
        self.collection = self.client.get_or_create_collection(
            name="nyaya_legal", metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[self._clean_meta(c.metadata) for c in chunks],
        )

    def query(self, embedding: list[float], k: int = 10) -> list[tuple[Chunk, float]]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[Chunk, float]] = []
        if not result.get("ids") or not result["ids"][0]:
            return out
        for cid, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            source_id = str(meta.get("source_id", ""))
            out.append((Chunk(id=cid, source_id=source_id, text=doc, metadata=dict(meta)),
                        1.0 - float(dist)))
        return out

    def all_chunks(self) -> list[Chunk]:
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        out: list[Chunk] = []
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            out.append(
                Chunk(id=cid, source_id=str(meta.get("source_id", "")), text=doc,
                      metadata=dict(meta))
            )
        return out

    @staticmethod
    def _clean_meta(meta: dict) -> dict:
        """Chroma metadata must be scalars."""
        clean = {}
        for key, value in meta.items():
            if value is None:
                clean[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean


_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Singleton store. Chroma when available, NumPy fallback otherwise."""
    global _store
    if _store is not None:
        return _store
    choice = settings.vector_backend.lower()
    store: VectorStore | None = None
    if choice in ("auto", "chroma"):
        try:
            store = ChromaStore(settings.chroma_dir)
            logger.info("Vector store: Chroma (%s)", settings.chroma_dir)
        except Exception as exc:
            if choice == "chroma":
                raise
            logger.warning("Chroma unavailable (%s); using built-in NumPy store", exc)
    if store is None:
        store = NumpyStore(settings.vector_dir)
        logger.info("Vector store: NumPy (%s)", settings.vector_dir)
    _store = store
    return store


async def store_count() -> int:
    return await asyncio.to_thread(get_store().count)
