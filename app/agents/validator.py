"""Anti-hallucination validation layer.

Final gate before any document reaches the user.  Every ``[source-id]``
citation marker in the document must exist in the retrieved source set,
every "verified fact" must be traceable to something the user actually
said (description, key facts, answers) or to a verbatim evidence excerpt,
and unfilled placeholders are surfaced instead of being silently filled
with invented details.
"""
from __future__ import annotations

import re
import unicodedata

from app.schemas.agents import GeneratedDocument, ResearchResult, ValidationReport

# [source-id] markers — lowercase letters, digits, hyphens.
CITATION_RE = re.compile(r"\[([a-z0-9][a-z0-9-]{2,40})\]")

# Documents where an unverified line can cause real-world harm.
HIGH_STAKES_TYPES = {"writ_petition", "police_complaint"}

PLACEHOLDER_MARK = "TO BE PROVIDED"

# Evidence wrappers look like:  [filename] "excerpt"
_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")


def _normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(text.split())


def _traceable(fact: str, corpus: list[str]) -> bool:
    """A fact is traceable when it appears in (or contains) a traced text."""
    variants = (fact, _PREFIX_RE.sub("", fact).strip("\u201c\u201d\"' "))
    for variant in variants:
        needle = _normalize(variant)
        if not needle:
            continue
        for hay in corpus:
            if not hay:
                continue
            if needle in hay:
                return True
            # reverse containment only for substantial texts (avoid noise)
            if len(hay) >= 24 and hay in needle:
                return True
    return False


def validate_document(
    doc: GeneratedDocument,
    research: ResearchResult,
    trace_texts: list[str],
) -> GeneratedDocument:
    """Run anti-hallucination checks and update the validation report."""
    available = {s.source_id for s in research.sources}
    corpus = [_normalize(t) for t in trace_texts if t and t.strip()]
    warnings: list[str] = []
    checked = 0
    verified = 0

    # 1. Every [source-id] marker must point to a retrieved source.
    for section in doc.sections:
        haystack = f"{section.body}\n{section.notes or ''}"
        for marker in CITATION_RE.findall(haystack):
            checked += 1
            if marker in available:
                verified += 1
            else:
                warnings.append(
                    f"Unverified citation [{marker}] in section "
                    f"\u201c{section.heading}\u201d \u2014 not among the "
                    "retrieved sources."
                )

    # 2. The legal-sources list must be a subset of the retrieved sources.
    kept_sources = []
    for src in doc.legal_sources:
        checked += 1
        if src.source_id in available:
            verified += 1
            kept_sources.append(src)
        else:
            warnings.append(
                f"Source \u201c{src.title}\u201d ({src.source_id}) is not among "
                "the retrieved sources \u2014 removed from the document."
            )
    doc.legal_sources = kept_sources

    # 3. Sections presenting legal content must carry at least one citation.
    for section in doc.sections:
        if section.tag == "legal_source" and not CITATION_RE.search(section.body):
            warnings.append(
                f"Section \u201c{section.heading}\u201d presents legal content "
                "without a verifiable [source-id] citation."
            )

    # 4. Verified facts must be traceable to user input or evidence excerpts.
    untraceable = [
        fact for fact in doc.verified_facts if not _traceable(fact, corpus)
    ]
    if untraceable:
        doc.verified_facts = [
            f for f in doc.verified_facts if f not in untraceable
        ]
        for fact in untraceable:
            doc.needs_review.append(
                "Not traceable to case data \u2014 verify before relying on "
                f"it: {fact[:120]}"
            )
        warnings.append(
            f"{len(untraceable)} verified fact(s) could not be traced to user "
            "input or evidence \u2014 moved to needs-review."
        )

    # 5. Count unfilled placeholders so the user knows what is missing.
    placeholders = sum(
        1 for section in doc.sections if PLACEHOLDER_MARK in section.body
    )
    if placeholders:
        doc.needs_review.append(
            f"{placeholders} section(s) still contain \u201c[LABEL \u2014 TO BE "
            "PROVIDED]\u201d placeholders \u2014 fill these before using the "
            "document."
        )

    # 6. High-stakes documents always require professional review.
    if doc.doc_type in HIGH_STAKES_TYPES:
        doc.needs_review.append(
            "This document type can have serious legal consequences \u2014 "
            "review by an advocate is strongly recommended before filing."
        )

    if warnings:
        doc.needs_review.insert(
            0,
            f"{len(warnings)} anti-hallucination warning(s) \u2014 see the "
            "validation report.",
        )

    doc.validation_report = ValidationReport(
        citations_checked=checked,
        citations_verified=verified,
        warnings=warnings,
    )
    return doc
