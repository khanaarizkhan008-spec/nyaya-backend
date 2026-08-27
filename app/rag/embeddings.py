"""Embedding providers.

Resolution order in "auto" mode:
  HF_TOKEN (BGE-M3 via Hugging Face) -> local Ollama (bge-m3) -> hash embeddings

Hash embeddings are deterministic and dependency-free — they guarantee the
RAG pipeline works offline. Combined with keyword-rank fusion in the
retriever, retrieval quality on the small curated corpus stays high.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re

import httpx

from app.config import settings

logger = logging.getLogger("nyaya.embeddings")


class EmbeddingError(Exception):
    pass


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9₹]+", text.lower())


class BaseEmbedder:
    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbedder(BaseEmbedder):
    """Feature hashing over token unigrams + bigrams. Deterministic, offline."""

    def __init__(self, dim: int = 384) -> None:
        self.name = "hash-384"
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = _tokenize(text)
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for gram in grams:
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 30) & 1 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class OllamaEmbedder(BaseEmbedder):
    def __init__(self, model: str = "bge-m3") -> None:
        self.name = f"ollama:{model}"
        self.model = model
        self.dim = 0  # discovered on first call

    async def embed(self, texts: list[str]) -> list[list[float]]:
        url = settings.ollama_base_url.rstrip("/") + "/api/embed"
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json={"model": self.model, "input": texts})
        if resp.status_code != 200:
            raise EmbeddingError(f"Ollama embed HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            raise EmbeddingError("Ollama returned no embeddings")
        self.dim = len(embeddings[0])
        return embeddings


class HFEmbedder(BaseEmbedder):
    def __init__(self, model: str) -> None:
        self.name = f"hf:{model.split('/')[-1].lower()}"
        self.model = model
        self.dim = 0

    async def _embed_one(self, client: httpx.AsyncClient, text: str) -> list[float]:
        url = f"https://router.huggingface.co/hf-inference/models/{self.model}"
        resp = await client.post(url, json={"inputs": text})
        if resp.status_code != 200:
            raise EmbeddingError(f"HF embed HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        vec = self._flatten(data)
        if not vec:
            raise EmbeddingError("HF returned empty embedding")
        return vec

    @staticmethod
    def _flatten(data) -> list[float]:
        if isinstance(data, dict):
            for key in ("embeddings", "sentence_embedding", "data"):
                if key in data:
                    return HFEmbedder._flatten(data[key])
            return []
        if isinstance(data, list):
            if data and isinstance(data[0], (int, float)):
                return [float(x) for x in data]
            if data and isinstance(data[0], list):
                # token-level embeddings -> mean pool
                inner = data
                dims = len(inner[0])
                pooled = []
                for i in range(dims):
                    vals = [row[i] for row in inner if len(row) == dims]
                    pooled.append(sum(vals) / len(vals) if vals else 0.0)
                return pooled
        return []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        async with httpx.AsyncClient(
            timeout=60, headers={"Authorization": f"Bearer {settings.hf_token.strip()}"}
        ) as client:
            for text in texts:
                vec = await self._embed_one(client, text)
                out.append(vec)
        if out:
            self.dim = len(out[0])
        return out


class EmbeddingService:
    """Resolved singleton embedder with startup probing."""

    def __init__(self) -> None:
        self._embedder: BaseEmbedder | None = None
        self._lock = asyncio.Lock()

    async def _ollama_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(
                    settings.ollama_base_url.rstrip("/") + "/api/tags"
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def _resolve(self) -> BaseEmbedder:
        if self._embedder is not None:
            return self._embedder
        async with self._lock:
            if self._embedder is not None:
                return self._embedder
            choice = settings.embeddings_provider.lower()
            candidate: BaseEmbedder | None = None
            if choice == "auto":
                if settings.hf_token.strip():
                    candidate = HFEmbedder(settings.embeddings_model)
                elif await self._ollama_alive():
                    candidate = OllamaEmbedder("bge-m3")
            elif choice == "huggingface" and settings.hf_token.strip():
                candidate = HFEmbedder(settings.embeddings_model)
            elif choice == "ollama":
                candidate = OllamaEmbedder("bge-m3")

            if candidate is not None:
                try:
                    await candidate.embed(["probe"])
                    self._embedder = candidate
                    logger.info("Embeddings: %s", candidate.name)
                    return candidate
                except Exception as exc:
                    logger.warning("Embedder %s failed probe, falling back to hash: %s",
                                   candidate.name, exc)
            self._embedder = HashEmbedder()
            logger.info("Embeddings: hash-384 (deterministic fallback)")
            return self._embedder

    async def name(self) -> str:
        embedder = await self._resolve()
        return embedder.name

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embedder = await self._resolve()
        return await embedder.embed(texts)

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]


embedding_service = EmbeddingService()
