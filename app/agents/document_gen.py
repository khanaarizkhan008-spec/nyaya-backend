"""AGENT 5 — Document Generation Agent.

Assembles a document from a template using only verified case data,
collects the verified facts / legal sources / needs-review lists, and runs
the anti-hallucination validator before the document is ever returned.
Deterministic by design: the generator is template+rules (the LLM never
writes document prose directly), so every line is traceable.
"""
from __future__ import annotations

import logging

from app.agents.templates import TEMPLATES, DocContext, get_template
from app.agents.validator import validate_document
from app.rag.retriever import retrieve_by_ids
from app.schemas.agents import (
    Confidence,
    GeneratedDocument,
)

logger = logging.getLogger("nyaya.document")


def _trace_texts(ctx: DocContext) -> list[str]:
    """Everything a 'verified fact' may legitimately be traced back to."""
    texts = [ctx.source_description, ctx.structured.problem_summary]
    texts.extend(ctx.structured.key_facts)
    texts.extend(v for v in ctx.answers.values() if v)
    for _filename, analysis in ctx.evidence:
        texts.extend(analysis.key_excerpts)
        texts.append(analysis.classification_reason or "")
    return [t for t in texts if t]


async def run_document(doc_type: str, ctx: DocContext) -> tuple[GeneratedDocument, str]:
    """Returns (document, generator_label). Raises ValueError for unknown types."""
    template = get_template(doc_type)
    if template is None:
        raise ValueError(
            f"Unknown document type “{doc_type}”. Available types: "
            + ", ".join(TEMPLATES.keys())
        )

    title, sections = template.build(ctx)

    # ---- legal sources actually cited by the template
    source_map = {s.source_id: s for s in ctx.research.sources}
    cited_ids = [sid for sid in dict.fromkeys(ctx.cited)]
    missing = [sid for sid in cited_ids if sid not in source_map]
    if missing:
        for source in await retrieve_by_ids(missing):
            source_map[source.source_id] = source
    legal_sources = [source_map[sid] for sid in cited_ids if sid in source_map]

    # ---- verified facts (what the user said + verbatim evidence excerpts)
    verified_facts = list(ctx.structured.key_facts)[:12]
    for filename, analysis in ctx.evidence:
        for excerpt in analysis.key_excerpts[:2]:
            verified_facts.append(f"[{filename}] “{excerpt}”")

    # ---- needs-review list
    needs_review: list[str] = []
    unfilled = [
        f.label
        for f in template.fields
        if f.required and not (ctx.extra_fields.get(f.key) or "").strip()
    ]
    for label in unfilled:
        needs_review.append(
            f"“{label}” was not provided — the document contains a placeholder."
        )
    if ctx.reasoning and ctx.reasoning.overall_confidence in (
        Confidence.low,
        Confidence.professional_review,
    ):
        needs_review.append(
            "The case analysis itself is marked "
            f"“{ctx.reasoning.overall_confidence.value}” — have the "
            "document checked before relying on it."
        )
    if ctx.reasoning and ctx.reasoning.note:
        needs_review.append(ctx.reasoning.note)

    document = GeneratedDocument(
        doc_type=doc_type,
        title=title,
        sections=sections,
        verified_facts=verified_facts,
        legal_sources=legal_sources,
        needs_review=needs_review,
        generator="template+rules",
    )
    # Validator mutates the document in place (strips unverified sources,
    # traces facts, fills the validation report).
    validate_document(document, ctx.research, _trace_texts(ctx))
    logger.info(
        "Generated %s: %d sections, %d/%d citations verified",
        doc_type,
        len(sections),
        document.validation_report.citations_verified,
        document.validation_report.citations_checked,
    )
    return document, "template+rules"
