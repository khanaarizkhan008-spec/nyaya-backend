"""Ingestion pipeline: load sources -> chunk -> embed -> store vectors."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.rag import kb
from app.rag.embeddings import embedding_service
from app.rag.store import Chunk, get_store

logger = logging.getLogger("nyaya.ingest")


async def reindex() -> dict:
    """Rebuild the vector index from the corpus on disk."""
    from app.config import settings  # local import to avoid cycles at module load

    sources = kb.scan_corpus()
    triples = kb.build_chunks(sources)
    if not triples:
        raise RuntimeError("Knowledge base is empty — nothing to index")

    texts = [piece for _src, piece, _cid in triples]
    embeddings = await embedding_service.embed(texts)

    store = get_store()
    chunks = [
        Chunk(
            id=cid,
            source_id=src.id,
            text=piece,
            metadata=kb.source_metadata(src),
        )
        for src, piece, cid in triples
    ]
    await asyncio.to_thread(store.reset)
    await asyncio.to_thread(store.add, chunks, embeddings)
    meta = {
        "embedding": await embedding_service.name(),
        "sources": len(sources),
        "chunks": len(chunks),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": settings.data_dir,
    }
    await asyncio.to_thread(store.set_meta, meta)
    logger.info("Indexed %d sources / %d chunks (%s)", len(sources), len(chunks), meta["embedding"])
    return meta


async def ensure_indexed() -> None:
    """Index on first boot; auto re-index if the embedding provider changed."""
    store = get_store()
    count = await asyncio.to_thread(store.count)
    current_embedder = await embedding_service.name()
    if count == 0:
        logger.info("Knowledge base empty — running initial ingestion")
        await reindex()
        return
    meta = await asyncio.to_thread(store.get_meta)
    if meta.get("embedding") and meta.get("embedding") != current_embedder:
        logger.info(
            "Embedding provider changed (%s -> %s) — re-indexing",
            meta.get("embedding"),
            current_embedder,
        )
        await reindex()
