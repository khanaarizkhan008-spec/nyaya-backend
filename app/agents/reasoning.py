"""AGENT 3 — Legal Reasoning / Remedy Agent.

Turns the structured case + retrieved sources into a practical pathway:
legal issue, applicable laws, remedies, steps, evidence, risks, confidence.

Anti-hallucination: the retrieved sources are the ONLY permitted citation
universe. LLM output is post-processed to drop any citation not present in
the retrieved set; the deterministic fallback is built from the taxonomy
filtered against what was actually retrieved. When retrieval is
insufficient, no remedies or deadlines are asserted at all.
"""
from __future__ import annotations

import json
import logging

from app.agents.taxonomy import get_domain, match_domain
from app.llm.client import LLMUnavailable, llm_client
from app.llm.prompts import REASONING_SYSTEM
from app.schemas.agents import (
    Confidence,
    EvidenceRequirement,
    LegalReasoning,
    PathwayStep,
    Remedy,
    ResearchResult,
    StructuredCase,
)

logger = logging.getLogger("nyaya.reasoning")

INSUFFICIENT_NOTE = (
    "Insufficient verified legal information was retrieved for this fact "
    "pattern. No statutes, remedies or deadlines are asserted. Consider "
    "re-describing the problem with more detail, or consult a lawyer / "
    "National Legal Services Authority helpline 15100."
)


def _available_ids(research: ResearchResult) -> set[str]:
    return {s.source_id for s in research.sources}


def insufficient_reasoning(structured: StructuredCase) -> LegalReasoning:
    """Reasoning used when retrieval could not find verified information."""
    domain = get_domain(structured.domain_code)
    return LegalReasoning(
        legal_issue=(
            f"{structured.legal_domain or domain['label']}: "
            f"{structured.problem_summary}"
        ),
        classification=structured.classification or domain["classification"],
        possible_laws=[],
        possible_remedies=[],
        recommended_pathway=[],
        alternative_pathways=[],
        required_evidence=[
            EvidenceRequirement(item=item, why=why)
            for item, why in domain["evidence_required"][:3]
        ],
        risks=["Without verified sources, no legal assessment can be offered."],
        uncertainties=[
            "Whether the knowledge base covers this situation.",
            "Whether important facts are still missing.",
        ],
        overall_confidence=Confidence.low,
        suggested_doc_types=[],
        note=INSUFFICIENT_NOTE,
    )


# ------------------------------------------------------------ deterministic
def deterministic_reasoning(
    structured: StructuredCase, research: ResearchResult
) -> LegalReasoning:
    """Build reasoning from the taxonomy, filtered to retrieved sources."""
    domain = get_domain(structured.domain_code)
    available = _available_ids(research)

    remedies: list[Remedy] = []
    for spec in domain["remedies"]:
        cited = [sid for sid in spec["source_ids"] if sid in available]
        if not cited:
            continue  # cannot verify this remedy against retrieved sources
        remedies.append(
            Remedy(
                title=spec["title"],
                description=spec["description"],
                authority=spec["authority"],
                confidence=Confidence(spec["confidence"]),
                source_ids=cited,
                priority=len(remedies) + 1,
            )
        )

    if not remedies:
        # Nothing in the taxonomy could be verified against retrieval —
        # refuse to assert anything rather than cite unretrieved law.
        result = insufficient_reasoning(structured)
        result.possible_laws = [
            s for s in research.sources if s.source_type == "primary"
        ][:8]
        return result

    pathway: list[PathwayStep] = []
    for title, desc, ids in domain["pathway"]:
        cited = [sid for sid in ids if sid in available]
        if not cited:
            continue
        pathway.append(
            PathwayStep(
                step=len(pathway) + 1,
                title=title,
                description=desc,
                confidence=Confidence.medium,
                source_ids=cited,
            )
        )

    laws = [s for s in research.sources if s.source_type == "primary"]
    laws = laws or list(research.sources)
    possible_laws = laws[:8]

    # ---- overall confidence
    if structured.emergency_flag or any(
        r.confidence == Confidence.professional_review for r in remedies
    ):
        overall = Confidence.professional_review
    else:
        overall = Confidence.medium
        _, score = match_domain(
            f"{structured.problem_summary} {' '.join(structured.key_facts)}"
        )
        if structured.domain_code == "general_contract" and score == 0:
            overall = Confidence.low
        elif len(research.sources) < 3:
            overall = Confidence.low

    note = None
    if overall == Confidence.low:
        note = (
            "The analysis is based on limited verified information. Verify "
            "with a legal professional before acting."
        )

    return LegalReasoning(
        legal_issue=(
            f"{domain['label']}: {structured.desired_outcome or domain['desired_outcome']}"
        ),
        classification=domain["classification"],
        possible_laws=possible_laws,
        possible_remedies=remedies,
        recommended_pathway=pathway,
        alternative_pathways=[],
        required_evidence=[
            EvidenceRequirement(item=item, why=why)
            for item, why in domain["evidence_required"]
        ],
        risks=list(domain["risks"]),
        uncertainties=list(domain["uncertainties"]),
        overall_confidence=overall,
        suggested_doc_types=list(domain["doc_types"]),
        note=note,
    )


# ------------------------------------------------------------ LLM guardrails
def _clamp_reasoning(
    reasoning: LegalReasoning, research: ResearchResult
) -> LegalReasoning:
    """Strip any citation not present in the retrieved set; rebuild law
    metadata from the authoritative retrieved versions."""
    available = _available_ids(research)
    by_id = {s.source_id: s for s in research.sources}

    # Rebuild possible_laws from retrieved versions (never trust LLM metadata)
    reasoning.possible_laws = [
        by_id[s.source_id] for s in reasoning.possible_laws if s.source_id in by_id
    ][:8]

    kept_remedies: list[Remedy] = []
    for remedy in reasoning.possible_remedies:
        remedy.source_ids = [sid for sid in remedy.source_ids if sid in available]
        kept_remedies.append(remedy)
    reasoning.possible_remedies = kept_remedies

    kept_steps: list[PathwayStep] = []
    for step in reasoning.recommended_pathway:
        step.source_ids = [sid for sid in step.source_ids if sid in available]
        step.step = len(kept_steps) + 1
        kept_steps.append(step)
    reasoning.recommended_pathway = kept_steps

    kept_alts: list[PathwayStep] = []
    for step in reasoning.alternative_pathways:
        step.source_ids = [sid for sid in step.source_ids if sid in available]
        step.step = len(kept_alts) + 1
        kept_alts.append(step)
    reasoning.alternative_pathways = kept_alts

    return reasoning


async def run_reasoning(
    structured: StructuredCase, research: ResearchResult
) -> tuple[LegalReasoning, str]:
    """Returns (reasoning, provider_label)."""
    if research.insufficient or not research.sources:
        return insufficient_reasoning(structured), "deterministic"

    sources_block = "\n".join(
        f"[{i}] source_id: {s.source_id} | {s.title}"
        + (f" | {s.instrument}" if s.instrument else "")
        + (f", {s.section}" if s.section else "")
        + (f" | {s.court}" if s.court else "")
        + (f" | {s.authority}" if s.authority else "")
        + (f" | snippet: {s.snippet}" if s.snippet else "")
        for i, s in enumerate(research.sources, start=1)
    )
    case_block = json.dumps(structured.model_dump(mode="json"), indent=1)
    user = (
        f"STRUCTURED CASE:\n{case_block}\n\n"
        f"PROVIDED SOURCES (the ONLY laws you may cite — use these source_id "
        f"values exactly):\n{sources_block}\n\n"
        f"Produce the LegalReasoning JSON."
    )

    try:
        raw = await llm_client.chat_json(REASONING_SYSTEM, user, LegalReasoning)
        reasoning = _clamp_reasoning(raw, research)
        if structured.emergency_flag:
            reasoning.overall_confidence = Confidence.professional_review
        if not reasoning.possible_remedies:
            # Every LLM remedy failed citation checks — refuse to assert them
            return deterministic_reasoning(structured, research), "deterministic-fallback"
        provider = await llm_client.resolve()
        return reasoning, f"{provider.provider}:{provider.model}"
    except LLMUnavailable:
        return deterministic_reasoning(structured, research), "deterministic"
    except Exception:
        logger.exception("Reasoning agent fell back to deterministic mode")
        return deterministic_reasoning(structured, research), "deterministic-fallback"
