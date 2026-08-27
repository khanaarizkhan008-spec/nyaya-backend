"""AGENT 2 — Legal Research / RAG Agent.

Pure retrieval (no LLM): converts the structured case into search queries,
retrieves ranked legal sources from the vector store, and returns them with
full citation metadata. If reliable information cannot be found, the result
is explicitly marked insufficient — the pipeline never fabricates law.
"""
from __future__ import annotations

import re

from app.agents.taxonomy import get_domain
from app.rag.retriever import retrieve
from app.schemas.agents import ResearchResult, SourceRef, StructuredCase


def _case_keywords(structured: StructuredCase) -> list[str]:
    """Pull distinctive words out of the problem summary for retrieval."""
    text = f"{structured.problem_summary} {structured.desired_outcome}"
    text += " " + " ".join(structured.key_facts)
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "under", "not",
        "has", "have", "was", "were", "are", "his", "her", "their", "my",
        "after", "been", "being", "would", "could", "should", "about",
    }
    words = re.findall(r"[a-z]{4,}", text.lower())
    seen: list[str] = []
    for word in words:
        if word not in stop and word not in seen:
            seen.append(word)
    return seen[:12]


async def run_research(structured: StructuredCase) -> ResearchResult:
    domain = get_domain(structured.domain_code)
    queries: list[str] = list(domain["queries"])
    keywords = _case_keywords(structured)
    if keywords:
        queries.append(" ".join(keywords[:10]))
    if structured.amount_in_dispute:
        queries.append(f"recovery claim {structured.amount_in_dispute} india")

    sources: list[SourceRef] = await retrieve(queries, k_final=8)

    # Prioritise sources that the taxonomy associates with the domain
    preferred = set(domain["source_ids"])
    sources.sort(key=lambda s: (0 if s.source_id in preferred else 1, -s.relevance))

    insufficient = len(sources) < 2
    note = None
    if insufficient:
        note = (
            "Insufficient verified legal information was found for this fact "
            "pattern. No legal conclusions are offered; consider re-describing "
            "the problem or consulting a lawyer."
        )
    return ResearchResult(queries=queries, sources=sources, insufficient=insufficient, note=note)
