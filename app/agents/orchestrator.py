"""Pipeline orchestration — runs the five agents in sequence.

Flow:
  1. ``start_case``        — Agent 1 (intake) runs synchronously so the user
                             immediately receives follow-up questions.
  2. ``submit_answers``    — intake re-runs with the answers merged in, then
                             analysis is scheduled in the background.
  3. ``analyze_case``      — Agents 2 (research) and 3 (reasoning) run in a
                             background task, updating AgentRun rows so the
                             UI can show live agent status.
  4. ``analyze_evidence``  — Agent 4 runs per uploaded document.
  5. ``generate_document`` — Agent 5 + the anti-hallucination validator.

Every stage records an AgentRun row (agent / status / provider / detail),
so the frontend can show the pipeline state at any moment by polling.
"""
from __future__ import annotations

import asyncio
import logging
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.document_gen import run_document
from app.agents.evidence import run_evidence
from app.agents.intake import run_intake
from app.agents.reasoning import run_reasoning
from app.agents.research import run_research
from app.agents.templates import DocContext
from app.database import SessionLocal
from app.models import AgentRun, Case, Evidence, LegalDocument, utcnow
from app.schemas.agents import (
    EvidenceAnalysis,
    LegalReasoning,
    ResearchResult,
    StructuredCase,
)
from app.services.extract import extract_text

logger = logging.getLogger("nyaya.pipeline")

_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """Run a background task with a strong reference so it is never GC'd."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def schedule_analysis(case_id: str) -> None:
    """Public helper: run Agents 2-3 in the background for a case."""
    _spawn(analyze_case(case_id))


def schedule_evidence_analysis(evidence_id: str) -> None:
    """Public helper: run Agent 4 in the background for one upload."""
    _spawn(analyze_evidence(evidence_id))


# ---------------------------------------------------------------- stage 1
async def start_case(db: AsyncSession, case: Case) -> None:
    """Run Agent 1 (intake) synchronously and persist the structured case."""
    run = AgentRun(case_id=case.id, agent="intake", status="running")
    db.add(run)
    await db.commit()

    try:
        structured, provider = await run_intake(case.description)
        case.structured_case = structured.model_dump(mode="json")
        case.domain_code = structured.domain_code
        case.emergency_flag = structured.emergency_flag
        case.emergency_reason = structured.emergency_reason or ""
        if not case.title.strip():
            case.title = structured.problem_summary[:90] or "New legal problem"

        run.status = "done"
        run.provider = provider
        run.detail = {
            "legal_domain": structured.legal_domain,
            "urgency": structured.urgency,
            "questions_asked": len(structured.questions),
        }
        run.finished_at = utcnow()

        if structured.questions:
            case.status = "awaiting_answers"
        else:
            case.status = "analyzing"
        await db.commit()
        if not structured.questions:
            schedule_analysis(case.id)
    except Exception as exc:
        logger.exception("Intake failed for case %s", case.id)
        run.status = "error"
        run.error = str(exc)[:500]
        run.finished_at = utcnow()
        case.status = "error"
        await db.commit()


# ---------------------------------------------------------------- stage 2
async def submit_answers(db: AsyncSession, case: Case, answers: dict[str, str]) -> None:
    """Merge answers, re-run intake, then schedule the analysis pipeline."""
    run = AgentRun(
        case_id=case.id,
        agent="intake",
        status="running",
        detail={"pass": "answers received"},
    )
    db.add(run)
    await db.commit()

    try:
        merged = {**(case.answers or {}), **{k: v for k, v in answers.items()}}
        structured, provider = await run_intake(case.description, merged)
        case.structured_case = structured.model_dump(mode="json")
        case.answers = merged
        case.domain_code = structured.domain_code
        case.emergency_flag = structured.emergency_flag
        case.emergency_reason = structured.emergency_reason or ""

        run.status = "done"
        run.provider = provider
        run.detail = {
            "legal_domain": structured.legal_domain,
            "remaining_questions": len(structured.questions),
        }
        run.finished_at = utcnow()
        case.status = "analyzing"
        await db.commit()
        schedule_analysis(case.id)
    except Exception as exc:
        logger.exception("Answer submission failed for case %s", case.id)
        run.status = "error"
        run.error = str(exc)[:500]
        run.finished_at = utcnow()
        case.status = "error"
        await db.commit()


# ---------------------------------------------------------------- stage 3
def _mark_have_it(evidence_rows: list[Evidence], reasoning: LegalReasoning) -> None:
    """Flag evidence requirements that match an uploaded document."""
    if not reasoning.required_evidence or not evidence_rows:
        return
    texts: list[str] = []
    for ev in evidence_rows:
        analysis = ev.analysis or {}
        guess = analysis.get("document_type_guess") or ""
        texts.append(f"{ev.filename} {guess}".lower())
    for req in reasoning.required_evidence:
        tokens = [t for t in re.findall(r"[a-z]{4,}", req.item.lower())]
        if tokens and any(any(tok in text for tok in tokens) for text in texts):
            req.have_it = True


async def analyze_case(case_id: str) -> None:
    """Background: Agent 2 (research) then Agent 3 (reasoning)."""
    async with SessionLocal() as db:
        case = await db.get(Case, case_id)
        if case is None or not case.structured_case:
            return
        structured = StructuredCase.model_validate(case.structured_case)

        try:
            # ---- Agent 2: research / retrieval
            r_run = AgentRun(case_id=case.id, agent="research", status="running")
            db.add(r_run)
            await db.commit()

            research = await run_research(structured)
            case.research = research.model_dump(mode="json")
            r_run.status = "done"
            r_run.provider = "hybrid-retrieval"
            r_run.detail = {
                "sources": len(research.sources),
                "insufficient": research.insufficient,
            }
            r_run.finished_at = utcnow()
            await db.commit()

            # ---- Agent 3: reasoning / remedy
            g_run = AgentRun(case_id=case.id, agent="reasoning", status="running")
            db.add(g_run)
            await db.commit()

            reasoning, provider = await run_reasoning(structured, research)
            evidence_rows = (
                (await db.execute(select(Evidence).where(Evidence.case_id == case.id)))
                .scalars()
                .all()
            )
            _mark_have_it(evidence_rows, reasoning)
            case.reasoning = reasoning.model_dump(mode="json")
            g_run.status = "done"
            g_run.provider = provider
            g_run.detail = {
                "remedies": len(reasoning.possible_remedies),
                "confidence": reasoning.overall_confidence.value,
            }
            g_run.finished_at = utcnow()
            case.status = "ready"
            await db.commit()
        except Exception as exc:
            logger.exception("Analysis failed for case %s", case_id)
            await db.rollback()
            case = await db.get(Case, case_id)
            if case is not None:
                case.status = "error"
                await db.commit()


# ---------------------------------------------------------------- stage 4
async def analyze_evidence(evidence_id: str) -> None:
    """Background: Agent 4 on one uploaded document."""
    async with SessionLocal() as db:
        ev = await db.get(Evidence, evidence_id)
        if ev is None:
            return
        run = AgentRun(
            case_id=ev.case_id,
            agent="evidence",
            status="running",
            detail={"filename": ev.filename},
        )
        db.add(run)
        await db.commit()

        try:
            case = await db.get(Case, ev.case_id)
            if case is None or not case.structured_case:
                raise ValueError("Case intake has not completed for this evidence")
            structured = StructuredCase.model_validate(case.structured_case)

            text = await extract_text(ev.stored_path, ev.mime)
            analysis, provider = await run_evidence(ev.filename, text, structured)

            ev.analysis = analysis.model_dump(mode="json")
            ev.status = "analyzed"
            run.status = "done"
            run.provider = provider
            run.detail = {
                "filename": ev.filename,
                "classification": analysis.classification,
                "excerpts": len(analysis.key_excerpts),
            }
        except Exception as exc:
            logger.exception("Evidence analysis failed for %s", evidence_id)
            await db.rollback()
            ev = await db.get(Evidence, evidence_id)
            if ev is not None:
                ev.status = "error"
                ev.error = str(exc)[:500]
                latest_run = (
                    await db.execute(
                        select(AgentRun)
                        .where(AgentRun.case_id == ev.case_id, AgentRun.agent == "evidence")
                        .order_by(AgentRun.started_at.desc())
                    )
                ).scalars().first()
                if latest_run is not None:
                    latest_run.status = "error"
                    latest_run.error = str(exc)[:500]
                    latest_run.finished_at = utcnow()
        finally:
            try:
                if run.finished_at is None:
                    run.finished_at = utcnow()
                await db.commit()
            except Exception:
                logger.exception("Failed to finalise evidence run for %s", evidence_id)


# ---------------------------------------------------------------- stage 5
async def generate_document(
    db: AsyncSession, case: Case, user: User, doc_type: str, extra_fields: dict[str, str]
) -> LegalDocument:
    """Run Agent 5 + validator and persist the resulting document."""
    from datetime import date
    structured = StructuredCase.model_validate(case.structured_case)
    research = ResearchResult.model_validate(case.research or {"sources": []})
    reasoning = (
        LegalReasoning.model_validate(case.reasoning) if case.reasoning else None
    )
    evidence_rows = (
        await db.execute(select(Evidence).where(Evidence.case_id == case.id))
    ).scalars().all()
    evidence_pairs = [
        (row.filename, EvidenceAnalysis.model_validate(row.analysis))
        for row in evidence_rows
        if row.analysis
    ]

    # Smart prefill automation
    if not extra_fields:
        extra_fields = {}

    # Identify sender and recipient names from case details
    sender_val = user.full_name or ""
    if not sender_val and getattr(structured, "parties", None):
        sender_val = structured.parties[0]
    
    recipient_val = ""
    if getattr(structured, "parties", None) and len(structured.parties) > 1:
        recipient_val = structured.parties[1]
    
    # Map to document-specific fields
    doc_field_mappings = {
        "legal_notice": {
            "sender_name": sender_val,
            "recipient_name": recipient_val,
        },
        "consumer_complaint": {
            "complainant_name": sender_val,
            "opposite_party_name": recipient_val,
        },
        "police_complaint": {
            "complainant_name": sender_val,
        },
        "rti_application": {
            "applicant_name": sender_val,
        },
        "labour_complaint": {
            "worker_name": sender_val,
            "employer_name": recipient_val,
        },
        "cyber_complaint": {
            "victim_name": sender_val,
        },
        "writ_petition": {
            "petitioner_name": sender_val,
            "respondent_authority": recipient_val,
        },
        "representation": {
            "applicant_name": sender_val,
            "authority_name": recipient_val,
        }
    }

    # Apply mappings if not already provided by user
    mappings = doc_field_mappings.get(doc_type, {})
    for key, val in mappings.items():
        if key not in extra_fields and val:
            extra_fields[key] = val

    # General fallback prefills
    if "client_name" not in extra_fields and user.full_name:
        extra_fields["client_name"] = user.full_name
    if "contact" not in extra_fields and user.email:
        extra_fields["contact"] = user.email
    if "date" not in extra_fields:
        extra_fields["date"] = date.today().strftime("%d %B %Y")

    ctx = DocContext(
        structured=structured,
        research=research,
        reasoning=reasoning,
        evidence=evidence_pairs,
        extra_fields=extra_fields,
        source_description=case.description,
        answers=case.answers or {},
    )

    run = AgentRun(
        case_id=case.id, agent="document", status="running", detail={"doc_type": doc_type}
    )
    db.add(run)
    await db.commit()

    document, generator = await run_document(doc_type, ctx)

    row = LegalDocument(
        case_id=case.id,
        doc_type=doc_type,
        title=document.title,
        content=document.model_dump(mode="json"),
    )
    db.add(row)
    run.status = "done"
    run.provider = generator
    run.detail = {
        "doc_type": doc_type,
        "sections": len(document.sections),
        "citations_verified": document.validation_report.citations_verified,
        "citations_checked": document.validation_report.citations_checked,
        "warnings": len(document.validation_report.warnings),
    }
    run.finished_at = utcnow()
    await db.commit()
    await db.refresh(row)
    return row
