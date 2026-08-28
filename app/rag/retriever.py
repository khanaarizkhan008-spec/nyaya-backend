"""Hybrid retriever: vector search + keyword scoring fused with RRF.

Every returned source carries full metadata so downstream agents can cite
it (title, act, section, court, date, source type, URL, demo flag).
"""
from __future__ import annotations

import asyncio
import re

from app.rag.embeddings import embedding_service
from app.rag.store import Chunk, get_store
from app.schemas.agents import SourceRef

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "under", "into",
    "not", "are", "was", "were", "has", "have", "had", "you", "your", "my",
    "his", "her", "their", "them", "they", "what", "which", "who", "whom",
    "how", "why", "when", "where", "can", "could", "should", "would", "may",
    "might", "shall", "will", "must", "does", "did", "been", "being", "any",
    "all", "some", "such", "than", "then", "also", "but", "its", "it's",
    "about", "after", "before", "between", "during", "over", "upon", "per",
}


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9₹]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


def _keyword_score(query_tokens: set[str], chunk_text: str) -> float:
    chunk_tokens = _keywords(chunk_text)
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / (len(query_tokens) ** 0.5)


async def retrieve(
    queries: list[str],
    k_final: int = 8,
    k_per_query: int = 6,
) -> list[SourceRef]:
    """Retrieve ranked sources for the given queries."""
    store = get_store()
    all_chunks: list[Chunk] = await asyncio.to_thread(store.all_chunks)
    if not all_chunks:
        return []

    # ---- vector rankings per query
    vector_rankings: list[list[Chunk]] = []
    for query in queries:
        try:
            embedding = await embedding_service.embed_one(query)
            hits = await asyncio.to_thread(store.query, embedding, k_per_query)
            vector_rankings.append([chunk for chunk, _score in hits])
        except Exception:
            continue  # keyword path still works

    # ---- keyword ranking over the whole corpus
    query_tokens: set[str] = set()
    for query in queries:
        query_tokens |= _keywords(query)
    keyword_ranked = sorted(
        all_chunks,
        key=lambda chunk: _keyword_score(query_tokens, chunk.text),
        reverse=True,
    )
    keyword_ranked = [c for c in keyword_ranked if _keyword_score(query_tokens, c.text) > 0]

    # ---- reciprocal-rank fusion
    scores: dict[str, float] = {}
    best_chunk: dict[str, Chunk] = {}
    for ranking in vector_rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (60 + rank)
            best_chunk.setdefault(chunk.id, chunk)
    for rank, chunk in enumerate(keyword_ranked[:30]):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 0.7 / (60 + rank)
        best_chunk.setdefault(chunk.id, chunk)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:k_final]
    if not ranked_ids:
        return []

    max_score = scores[ranked_ids[0]] or 1.0
    refs: list[SourceRef] = []
    for cid in ranked_ids:
        chunk = best_chunk[cid]
        meta = chunk.metadata
        snippet = chunk.text.strip()
        if len(snippet) > 2000:
            snippet = snippet[:1997].rstrip() + "..."
        refs.append(
            SourceRef(
                source_id=meta.get("source_id", chunk.source_id),
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                instrument=meta.get("instrument") or None,
                section=meta.get("section") or None,
                court=meta.get("court") or None,
                authority=meta.get("authority") or None,
                date=meta.get("date") or None,
                jurisdiction=meta.get("jurisdiction") or None,
                source_type=meta.get("source_type", "primary"),
                source_url=meta.get("source_url") or None,
                demo_data=bool(meta.get("demo_data", True)),
                snippet=snippet,
                relevance=round(scores[cid] / max_score, 3),
            )
        )
    return refs


async def retrieve_by_ids(source_ids: list[str]) -> list[SourceRef]:
    """Fetch full SourceRefs for known ids (used when reasoning needs specifics)."""
    wanted = set(source_ids)
    if not wanted:
        return []
    store = get_store()
    all_chunks = await asyncio.to_thread(store.all_chunks)
    seen: dict[str, Chunk] = {}
    for chunk in all_chunks:
        if chunk.source_id in wanted and chunk.source_id not in seen:
            seen[chunk.source_id] = chunk
    refs: list[SourceRef] = []
    for sid in source_ids:
        if sid not in seen:
            continue
        chunk = seen[sid]
        meta = chunk.metadata
        snippet = chunk.text.strip()
        if len(snippet) > 2000:
            snippet = snippet[:1997].rstrip() + "..."
        refs.append(
            SourceRef(
                source_id=meta.get("source_id", sid),
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                instrument=meta.get("instrument") or None,
                section=meta.get("section") or None,
                court=meta.get("court") or None,
                authority=meta.get("authority") or None,
                date=meta.get("date") or None,
                jurisdiction=meta.get("jurisdiction") or None,
                source_type=meta.get("source_type", "primary"),
                source_url=meta.get("source_url") or None,
                demo_data=bool(meta.get("demo_data", True)),
                snippet=snippet,
                relevance=1.0,
            )
        )
    return refs
