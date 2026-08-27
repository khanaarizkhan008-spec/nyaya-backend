"""AGENT 4 — Evidence Analysis Agent.

Analyzes ONE uploaded document and relates it to the case: entity
extraction (names/dates/amounts/contacts), verbatim key excerpts, an
explicit fact-vs-inference split, a dated timeline and a relevance
classification.

Anti-hallucination: excerpts are VERBATIM quotes from the document text —
LLM-proposed excerpts that cannot be located in the document are dropped,
and every inference is prefixed "AI inference:".
"""
from __future__ import annotations

import logging
import re

from app.agents.intake import AMOUNT_PATTERN, DATE_PATTERNS
from app.agents.taxonomy import get_domain
from app.llm.client import LLMUnavailable, llm_client
from app.llm.prompts import EVIDENCE_SYSTEM
from app.schemas.agents import (
    EvidenceAnalysis,
    EvidenceEntities,
    StructuredCase,
    TimelineEvent,
)

logger = logging.getLogger("nyaya.evidence")

PHONE_PATTERN = r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.-]+"

DOC_TYPE_HINTS: list[tuple[str, list[str]]] = [
    ("rental agreement", ["rental agreement", "tenancy agreement", "lessee", "lessor", "landlord and"]),
    ("salary slip", ["salary slip", "payslip", "pay slip", "gross salary", "net pay", "employee code"]),
    ("bank statement", ["bank statement", "account statement", "ifsc", "closing balance", "opening balance", "utr"]),
    ("invoice / bill", ["tax invoice", "invoice no", "bill no", "gst", "amount payable"]),
    ("legal notice", ["legal notice", "under instructions", "advocate", "whereas the"]),
    ("police complaint / FIR", ["first information report", "f i r", "fir", "police station", "sho"]),
    ("medical report", ["discharge summary", "diagnosis", "patient name", "prescription"]),
    ("employment letter", ["offer letter", "appointment letter", "designation", "ctc", "employee id"]),
    ("written agreement / contract", ["witnesseth", "hereinafter", "terms and conditions", "shall be binding"]),
    ("email / message", ["from:", "to:", "subject:", "regards"]),
]

CONTRADICTION_MARKERS = [
    "already refunded", "no dues", "paid in full", "no amount is due",
    "denies any liability", "no payment is outstanding", "stands withdrawn",
]

_NAME_STOPWORDS = {
    "dear", "sir", "madam", "thank", "you", "regards", "the", "this",
    "that", "your", "our", "their", "and", "for", "with", "from", "to",
    "subject", "re", "fyi", "hi", "hello", "whom", "it", "may",
    "concern", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "india", "bank", "state", "court", "commission", "act", "section",
    "annexure", "witness", "signed", "yours", "faithfully", "sincerely",
}


# ------------------------------------------------------------ extractors
def _guess_doc_type(text: str) -> str | None:
    lowered = text.lower()
    best: tuple[int, str] = (0, "")
    for label, hints in DOC_TYPE_HINTS:
        hits = sum(1 for kw in hints if kw in lowered)
        if hits > best[0]:
            best = (hits, label)
    return best[1] or None


def _extract_names(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", text)
    counts: dict[str, int] = {}
    for name in candidates:
        words = [w for w in name.split() if w.lower() not in _NAME_STOPWORDS]
        if not words:
            continue
        cleaned = " ".join(words)
        if len(cleaned) < 6:
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1
    ranked = sorted(counts, key=lambda n: -counts[n])
    return ranked[:6]


def _extract_dates(text: str) -> list[str]:
    found: list[str] = []
    for pattern in DATE_PATTERNS:
        found.extend(m.group(0) for m in re.finditer(pattern, text))
    seen: list[str] = []
    for d in found:
        if d not in seen:
            seen.append(d)
    return seen[:10]


def _extract_amounts(text: str) -> list[str]:
    seen: list[str] = []
    for m in re.finditer(AMOUNT_PATTERN, text):
        val = m.group(0).strip()
        if val not in seen:
            seen.append(val)
    return seen[:10]


def _extract_contacts(text: str) -> tuple[list[str], list[str]]:
    emails = []
    for m in re.finditer(EMAIL_PATTERN, text):
        if m.group(0) not in emails:
            emails.append(m.group(0))
    phones = []
    for m in re.finditer(PHONE_PATTERN, text):
        if m.group(0) not in phones:
            phones.append(m.group(0))
    return emails[:5], phones[:5]


def _extract_addresses(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits = []
    for ln in lines:
        lowered = ln.lower()
        if any(
            marker in lowered
            for marker in ("flat no", "plot no", "door no", "street", "road", "nagar", "colony", "village", "sector", "pin")
        ) and 10 < len(ln) < 140:
            if ln not in hits:
                hits.append(ln)
    return hits[:4]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [" ".join(p.split()) for p in parts if p.strip()]


def _overlap_terms(text: str, structured: StructuredCase) -> list[str]:
    """Domain keywords + distinctive case words that appear in the document."""
    domain = get_domain(structured.domain_code)
    lowered = text.lower()
    terms = [kw for kw in domain["keywords"] if kw in lowered]
    case_words = set(re.findall(r"[a-z]{5,}", structured.problem_summary.lower()))
    case_words |= {w.lower() for f in structured.key_facts for w in re.findall(r"[a-z]{5,}", f.lower())}
    terms += [w for w in case_words if w in lowered and w not in terms]
    return terms


def _extract_excerpts(text: str, terms: list[str]) -> list[str]:
    scored: list[tuple[int, str]] = []
    for sentence in _sentences(text):
        if not 20 < len(sentence) < 300:
            continue
        score = sum(2 for t in terms if t in sentence.lower())
        if re.search(AMOUNT_PATTERN, sentence):
            score += 2
        if any(re.search(p, sentence) for p in DATE_PATTERNS):
            score += 2
        if score > 0:
            scored.append((score, sentence))
    scored.sort(key=lambda pair: -pair[0])
    return [s for _, s in scored[:5]]


def _extract_timeline(text: str) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    seen_dates: set[str] = set()
    for pattern in DATE_PATTERNS:
        for m in re.finditer(pattern, text):
            date_str = m.group(0)
            if date_str in seen_dates:
                continue
            sentence = next(
                (s for s in _sentences(text) if date_str in s), ""
            )
            if not sentence:
                start = max(0, m.start() - 120)
                sentence = " ".join(text[start : m.end() + 60].split())
            seen_dates.add(date_str)
            events.append(
                TimelineEvent(date=date_str, description=sentence[:220], origin="document")
            )
    return events[:6]


def _classify(
    text: str, structured: StructuredCase
) -> tuple[str, str, list[str]]:
    lowered = text.lower()
    terms = _overlap_terms(text, structured)
    contra = [mk for mk in CONTRADICTION_MARKERS if mk in lowered]
    related = [
        f for f in structured.key_facts
        if any(w in f.lower() for w in terms[:8])
    ][:3]

    if contra:
        return (
            "contradictory",
            f'The document contains language that may contradict the claim ("{contra[0]}").',
            related,
        )
    if len(terms) >= 3:
        return (
            "strong_support",
            f"The document matches key aspects of the case ({', '.join(terms[:3])}).",
            related,
        )
    if terms:
        return (
            "possible_support",
            f"Some case-related terms appear ({', '.join(terms[:3])}); verify against the facts.",
            related,
        )
    return (
        "possible_support",
        "No direct link was detected — the AI could not confidently relate "
        "this document to the case; please review it manually.",
        [],
    )


# ------------------------------------------------------------ deterministic
def deterministic_evidence(
    text: str, structured: StructuredCase
) -> EvidenceAnalysis:
    terms = _overlap_terms(text, structured)
    emails, phones = _extract_contacts(text)
    guess = _guess_doc_type(text)
    classification, reason, related = _classify(text, structured)

    inferences: list[str] = []
    if guess:
        inferences.append(f"AI inference: the document appears to be a {guess}.")
    if _extract_amounts(text):
        inferences.append(
            "AI inference: monetary amounts were detected, which may help quantify the claim."
        )
    if _extract_dates(text):
        inferences.append(
            "AI inference: dated entries were detected, which may help build a chronology."
        )

    return EvidenceAnalysis(
        document_type_guess=guess,
        entities=EvidenceEntities(
            names=_extract_names(text),
            dates=_extract_dates(text),
            amounts=_extract_amounts(text),
            addresses=_extract_addresses(text),
            emails=emails,
            phones=phones,
        ),
        key_excerpts=_extract_excerpts(text, terms),
        inferences=inferences,
        timeline_events=_extract_timeline(text),
        classification=classification,
        classification_reason=reason,
        related_facts=related,
    )


# ------------------------------------------------------------ LLM guardrails
def _clamp_analysis(analysis: EvidenceAnalysis, text: str) -> EvidenceAnalysis:
    """Keep only verbatim excerpts and document-backed timeline dates."""
    norm = " ".join(text.split())
    kept: list[str] = []
    for excerpt in analysis.key_excerpts:
        ex = " ".join(excerpt.split())
        if not ex:
            continue
        if ex in norm or ex.rstrip(".…") in norm or ex.strip("“”\"'") in norm:
            kept.append(ex[:300])
    analysis.key_excerpts = kept[:6]

    analysis.inferences = [
        inf if inf.lower().startswith("ai inference") else f"AI inference: {inf}"
        for inf in analysis.inferences
    ][:8]

    analysis.timeline_events = [
        ev for ev in analysis.timeline_events if ev.date and ev.date in text
    ][:6]
    return analysis


async def run_evidence(
    filename: str, text: str, structured: StructuredCase
) -> tuple[EvidenceAnalysis, str]:
    """Returns (analysis, provider_label)."""
    if not text or not text.strip():
        return (
            EvidenceAnalysis(
                document_type_guess=None,
                classification="possible_support",
                classification_reason=(
                    "No text could be extracted from this file (it may be a "
                    "scanned image requiring OCR, or an unsupported format). "
                    "Please review it manually."
                ),
            ),
            "deterministic",
        )

    terms = _overlap_terms(text, structured)
    facts_block = "\n".join(f"- {f}" for f in structured.key_facts[:8])
    user = (
        f"CASE SUMMARY:\n{structured.problem_summary}\n"
        f"LEGAL DOMAIN: {structured.legal_domain}\n"
        f"KEY FACTS:\n{facts_block}\n\n"
        f"DOCUMENT FILENAME: {filename}\n"
        f"DOCUMENT TEXT (extracted, may be partial):\n{text[:6000]}\n\n"
        f"Produce the EvidenceAnalysis JSON for THIS document."
    )

    try:
        raw = await llm_client.chat_json(EVIDENCE_SYSTEM, user, EvidenceAnalysis)
        analysis = _clamp_analysis(raw, text)
        if not analysis.key_excerpts:
            # The LLM produced no verbatim quotes — use deterministic ones
            analysis.key_excerpts = _extract_excerpts(text, terms)
        if not analysis.classification_reason:
            analysis.classification_reason = _classify(text, structured)[1]
        provider = await llm_client.resolve()
        return analysis, f"{provider.provider}:{provider.model}"
    except LLMUnavailable:
        return deterministic_evidence(text, structured), "deterministic"
    except Exception:
        logger.exception("Evidence agent fell back to deterministic mode")
        return deterministic_evidence(text, structured), "deterministic-fallback"
