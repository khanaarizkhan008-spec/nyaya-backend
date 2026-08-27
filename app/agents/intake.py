"""AGENT 1 — Legal Intake Agent.

Converts the user's natural-language description into a structured case
(StructuredCase). Tries the LLM first; falls back to the deterministic
taxonomy engine so intake always works — including in demo mode.
"""
from __future__ import annotations

import re
from datetime import date

from app.agents.taxonomy import EMERGENCY_KEYWORDS, domain_choices, get_domain, match_domain
from app.llm.client import LLMUnavailable, llm_client
from app.llm.prompts import INTAKE_SYSTEM
from app.schemas.agents import IntakeQuestion, StructuredCase

DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\.?\s+\d{4}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
]
AMOUNT_PATTERN = r"(?:₹|Rs\.?|INR|inr)\s*[\d,]+(?:\.\d{1,2})?|[\d,]{3,}\s*(?:rupees|RUPEES|Rupees)"
DURATION_PATTERN = r"\b\d+\s*(?:day|week|month|year)s?\b"


def _extract_dates(text: str) -> list[str]:
    found: list[str] = []
    for pattern in DATE_PATTERNS:
        found.extend(m.group(0) for m in re.finditer(pattern, text))
    return found[:10]


def _extract_amounts(text: str) -> list[str]:
    return [m.group(0).strip() for m in re.finditer(AMOUNT_PATTERN, text)][:10]


def _extract_duration(text: str) -> str | None:
    match = re.search(DURATION_PATTERN, text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _extract_facts(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    facts = [s.strip() for s in sentences if len(s.strip()) > 12]
    return facts[:12]


def _detect_parties(text: str, domain_code: str) -> list[str]:
    lowered = text.lower()
    parties = ["You (complainant)"]
    if domain_code == "tenancy_deposit":
        parties.append("Landlord")
    elif domain_code == "unpaid_wages":
        parties.append("Employer")
    elif domain_code == "consumer_defective":
        parties.append("Seller / service provider")
    elif domain_code == "police_fir_refusal":
        parties.append("Police (station concerned)")
    elif domain_code == "cyber_fraud":
        parties.append("Unknown fraudster(s)")
        if "bank" in lowered:
            parties.append("Bank")
    elif domain_code == "workplace_harassment":
        parties.append("Employer / Internal Committee")
    elif domain_code == "domestic_violence":
        parties.append("Respondent (family member)")
    elif domain_code == "motor_accident":
        parties.append("Vehicle owner / driver / insurer")
    elif domain_code == "cheque_bounce":
        parties.append("Drawer of the cheque")
    else:
        parties.append("Opposite party (details to be added)")
    return parties


def _detect_emergency(text: str) -> tuple[bool, str | None]:
    lowered = text.lower()
    hits = [kw for kw in EMERGENCY_KEYWORDS if kw in lowered]
    if hits:
        reason = (
            "The description indicates a possible emergency situation "
            f"(indicators: {', '.join(hits[:4])}). Immediate safety and "
            "emergency-authority contact take priority over documentation."
        )
        return True, reason
    return False, None


def deterministic_intake(
    description: str, answers: dict[str, str] | None = None
) -> StructuredCase:
    """Rule-based intake over the domain taxonomy."""
    answers = answers or {}
    domain_code, score = match_domain(description)
    domain = get_domain(domain_code)

    emergency, emergency_reason = _detect_emergency(description)
    if domain.get("emergency_default") and not emergency:
        emergency, emergency_reason = True, (
            "This category may involve safety concerns. Please confirm safety and "
            "contact emergency services (112 / 1091) if there is any immediate risk."
        )

    facts = _extract_facts(description)
    dates = _extract_dates(description)
    amounts = _extract_amounts(description)
    duration = _extract_duration(description)

    answered_notes = [
        f"{key.replace('_', ' ').capitalize()}: {value.strip()}"
        for key, value in answers.items()
        if value.strip()
    ]
    if answered_notes:
        facts = facts + answered_notes

    questions = [
        IntakeQuestion(id=qid, question=question, why_it_matters=why)
        for qid, question, why in domain["questions"]
    ]
    unanswered = [q for q in questions if not answers.get(q.id, "").strip()]

    missing = [q.why_it_matters for q in unanswered][:6]
    if score == 0:
        missing.append("The situation could not be confidently classified — more detail would help")

    return StructuredCase(
        problem_summary=_summarize(description, domain),
        legal_domain=domain["label"],
        domain_code=domain_code,
        classification=domain["classification"],
        parties=_detect_parties(description, domain_code),
        key_facts=facts,
        dates=dates,
        locations=[],
        amount_in_dispute=amounts[0] if amounts else None,
        duration=duration,
        desired_outcome=domain["desired_outcome"],
        urgency="high" if emergency else "normal",
        emergency_flag=emergency,
        emergency_reason=emergency_reason,
        missing_information=missing,
        questions=unanswered,
    )


def _summarize(description: str, domain: dict) -> str:
    first = description.strip().split(".")[0]
    summary = first if 15 < len(first) < 200 else description.strip()[:180]
    return f"{summary} — likely a {domain['label'].lower()} matter."


async def run_intake(
    description: str, answers: dict[str, str] | None = None
) -> tuple[StructuredCase, str]:
    """Returns (structured_case, provider_label)."""
    answers = answers or {}
    try:
        system = INTAKE_SYSTEM.format(today=date.today().isoformat())
        choices = ", ".join(domain_choices())
        answers_block = ""
        if answers:
            answers_block = (
                "\n\nThe user has already answered these follow-up questions:\n"
                + "\n".join(
                    f"- {key}: {value}"
                    for key, value in answers.items()
                    if str(value).strip()
                )
                + "\n\nThis is the SECOND interview pass: fold these answers "
                "into the structured case and return an EMPTY questions list "
                "unless something critical is still missing."
            )
        user = (
            f"ALLOWED DOMAIN CODES: {choices}\n\nUSER'S PROBLEM DESCRIPTION:\n{description}"
            f"{answers_block}\n\nProduce the structured intake JSON."
        )
        structured = await llm_client.chat_json(system, user, StructuredCase)
        # Guardrails: clamp domain code, enforce emergency detection deterministically
        if structured.domain_code not in domain_choices():
            structured.domain_code = match_domain(description)[0]
        if structured.legal_domain == "":
            structured.legal_domain = get_domain(structured.domain_code)["label"]
        det_emergency, det_reason = _detect_emergency(description)
        if det_emergency and not structured.emergency_flag:
            structured.emergency_flag = True
            structured.emergency_reason = det_reason
            structured.urgency = "high"
        provider = (await llm_client.resolve())
        return structured, f"{provider.provider}:{provider.model}"
    except LLMUnavailable:
        return deterministic_intake(description, answers), "deterministic"
    except Exception:
        return deterministic_intake(description, answers), "deterministic-fallback"
