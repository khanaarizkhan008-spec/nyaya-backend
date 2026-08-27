"""Structured schemas exchanged between the five agents (Pydantic-validated JSON)."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- Confidence
class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"
    professional_review = "professional_review"


CONFIDENCE_LABELS = {
    Confidence.high: "High confidence",
    Confidence.medium: "Medium confidence",
    Confidence.low: "Low confidence",
    Confidence.professional_review: "Professional review recommended",
}


# ---------------------------------------------------------------- Agent 1: Intake
class IntakeQuestion(BaseModel):
    id: str
    question: str
    why_it_matters: str = ""
    input_type: Literal["text", "textarea", "select"] = "text"
    options: list[str] | None = None


class StructuredCase(BaseModel):
    problem_summary: str
    legal_domain: str
    domain_code: str = "general_contract"
    classification: str = ""
    parties: list[str] = []
    key_facts: list[str] = []
    dates: list[str] = []
    locations: list[str] = []
    amount_in_dispute: str | None = None
    duration: str | None = None
    desired_outcome: str = ""
    urgency: Literal["normal", "high"] = "normal"
    emergency_flag: bool = False
    emergency_reason: str | None = None
    missing_information: list[str] = []
    questions: list[IntakeQuestion] = []


# ---------------------------------------------------------------- Agent 2: Research / RAG
class SourceRef(BaseModel):
    source_id: str
    title: str
    category: str = ""
    instrument: str | None = None
    section: str | None = None
    court: str | None = None
    authority: str | None = None
    date: str | None = None
    jurisdiction: str | None = None
    source_type: Literal["primary", "secondary"] = "primary"
    source_url: str | None = None
    demo_data: bool = True
    snippet: str | None = None
    relevance: float = 0.0


class ResearchResult(BaseModel):
    queries: list[str] = []
    sources: list[SourceRef] = []
    insufficient: bool = False
    note: str | None = None


# ---------------------------------------------------------------- Agent 3: Reasoning
class Remedy(BaseModel):
    title: str
    description: str
    authority: str | None = None
    confidence: Confidence = Confidence.medium
    source_ids: list[str] = []
    priority: int = 1


class PathwayStep(BaseModel):
    step: int
    title: str
    description: str
    confidence: Confidence = Confidence.medium
    source_ids: list[str] = []


class EvidenceRequirement(BaseModel):
    item: str
    why: str = ""
    have_it: bool | None = None


class LegalReasoning(BaseModel):
    legal_issue: str
    classification: str = ""
    possible_laws: list[SourceRef] = []
    possible_remedies: list[Remedy] = []
    recommended_pathway: list[PathwayStep] = []
    alternative_pathways: list[PathwayStep] = []
    required_evidence: list[EvidenceRequirement] = []
    risks: list[str] = []
    uncertainties: list[str] = []
    overall_confidence: Confidence = Confidence.medium
    suggested_doc_types: list[str] = []
    note: str | None = None


# ---------------------------------------------------------------- Agent 4: Evidence
class EvidenceEntities(BaseModel):
    names: list[str] = []
    dates: list[str] = []
    amounts: list[str] = []
    addresses: list[str] = []
    emails: list[str] = []
    phones: list[str] = []


class TimelineEvent(BaseModel):
    date: str
    description: str
    origin: Literal["document", "user"] = "document"


class EvidenceAnalysis(BaseModel):
    document_type_guess: str | None = None
    entities: EvidenceEntities = EvidenceEntities()
    key_excerpts: list[str] = []
    inferences: list[str] = []
    timeline_events: list[TimelineEvent] = []
    classification: Literal[
        "strong_support", "possible_support", "contradictory", "irrelevant"
    ] = "possible_support"
    classification_reason: str = ""
    related_facts: list[str] = []


# ---------------------------------------------------------------- Agent 5: Document
class DocSection(BaseModel):
    heading: str
    body: str
    tag: Literal[
        "verified_fact", "legal_source", "ai_generated", "needs_review", "placeholder"
    ] = "ai_generated"
    notes: str | None = None


class ValidationReport(BaseModel):
    citations_checked: int = 0
    citations_verified: int = 0
    warnings: list[str] = []


class GeneratedDocument(BaseModel):
    doc_type: str
    title: str
    sections: list[DocSection]
    verified_facts: list[str] = []
    legal_sources: list[SourceRef] = []
    needs_review: list[str] = []
    validation_report: ValidationReport = ValidationReport()
    generator: str = "template+rules"
