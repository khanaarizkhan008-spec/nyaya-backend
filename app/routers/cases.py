"""Cases router — create, list, get (with agent status), submit answers."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import orchestrator
from app.deps import get_current_user, get_db
from app.models import AgentRun, Case, Feedback, User
from app.schemas.api import AnswersIn, CaseCreateIn, CaseListItem, FeedbackIn, FeedbackOut

router = APIRouter(prefix="/api/cases", tags=["cases"])


def _case_list_item(c: Case) -> dict:
    return CaseListItem(
        id=c.id,
        title=c.title,
        status=c.status,
        domain_code=c.domain_code,
        emergency_flag=c.emergency_flag,
        created_at=c.created_at.isoformat(),
        updated_at=c.updated_at.isoformat(),
    ).model_dump()


async def _get_user_case(
    case_id: str, user: User, db: AsyncSession
) -> Case:
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == user.id)
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


async def _agent_runs(case_id: str, db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(AgentRun)
        .where(AgentRun.case_id == case_id)
        .order_by(AgentRun.started_at.asc())
    )
    return [
        {
            "id": r.id,
            "agent": r.agent,
            "status": r.status,
            "provider": r.provider,
            "detail": r.detail,
            "error": r.error,
            "started_at": r.started_at.isoformat(),
            "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        }
        for r in result.scalars()
    ]


@router.post("", status_code=201)
async def create_case(
    body: CaseCreateIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Describe a legal problem: Agent 1 (intake) runs synchronously so the
    user immediately gets the follow-up questions (or the analysis starts)."""
    case = Case(
        user_id=user.id,
        description=body.description,
        title=body.title or "",
    )
    db.add(case)
    await db.commit()
    await db.refresh(case)

    await orchestrator.start_case(db, case)
    await db.refresh(case)
    return {
        "case": _case_list_item(case),
        "structured_case": case.structured_case,
        "status": case.status,
        "agent_runs": await _agent_runs(case.id, db),
    }


@router.get("")
async def list_cases(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Case)
        .where(Case.user_id == user.id)
        .order_by(Case.updated_at.desc())
    )
    cases = result.scalars().all()
    return [_case_list_item(c) for c in cases]


@router.get("/{case_id}")
async def get_case(
    case_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Full case state — polled by the frontend to track agent progress."""
    case = await _get_user_case(case_id, user, db)
    return {
        "case": _case_list_item(case),
        "structured_case": case.structured_case,
        "research": case.research,
        "reasoning": case.reasoning,
        "answers": case.answers,
        "emergency_reason": case.emergency_reason,
    }


@router.delete("/{case_id}", status_code=204)
async def delete_case(
    case_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    import os
    from app.models import Evidence
    
    case = await _get_user_case(case_id, user, db)
    
    # Clean up evidence files on disk
    result = await db.execute(
        select(Evidence).where(Evidence.case_id == case.id)
    )
    for ev in result.scalars():
        try:
            if ev.stored_path and os.path.exists(ev.stored_path):
                os.remove(ev.stored_path)
        except Exception:
            pass
            
    await db.delete(case)
    await db.commit()


@router.post("/{case_id}/answers")
async def submit_answers(
    case_id: str,
    body: AnswersIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit follow-up answers; research + reasoning then run in the
    background (poll GET /{case_id} for agent status)."""
    case = await _get_user_case(case_id, user, db)
    if case.status not in ("awaiting_answers", "analyzing", "ready", "error"):
        raise HTTPException(status_code=400, detail=f"Case status is {case.status}")
    if not case.structured_case:
        raise HTTPException(status_code=400, detail="Intake not completed")

    await orchestrator.submit_answers(db, case, body.answers)
    await db.refresh(case)
    return {
        "case": _case_list_item(case),
        "structured_case": case.structured_case,
        "status": case.status,
        "agent_runs": await _agent_runs(case.id, db),
    }


@router.post("/{case_id}/reanalyze")
async def reanalyze(
    case_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-run research + reasoning (e.g. after the knowledge base changed)."""
    case = await _get_user_case(case_id, user, db)
    if not case.structured_case:
        raise HTTPException(status_code=400, detail="Intake not completed")

    case.status = "analyzing"
    await db.commit()
    orchestrator.schedule_analysis(case.id)
    return {"case": _case_list_item(case), "status": case.status}


@router.post("/{case_id}/change-domain")
async def change_case_domain(
    case_id: str,
    domain_code: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Manually override the case's matched domain taxonomy code and trigger re-analysis."""
    case = await _get_user_case(case_id, user, db)
    if not case.structured_case:
        raise HTTPException(status_code=400, detail="Intake not completed")
    
    from app.agents.taxonomy import get_domain
    domain = get_domain(domain_code)
    
    # Update structured case domain
    structured = dict(case.structured_case)
    structured["domain_code"] = domain_code
    structured["legal_domain"] = domain["label"]
    case.structured_case = structured
    case.domain_code = domain_code
    case.status = "analyzing"
    await db.commit()
    
    orchestrator.schedule_analysis(case.id)
    return {"case": _case_list_item(case), "status": case.status}


@router.post("/{case_id}/feedback", status_code=201)
async def submit_case_feedback(
    case_id: str,
    body: FeedbackIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Report inaccurate advice or submit domain expert feedback for fine-tuning."""
    case = await _get_user_case(case_id, user, db)
    
    feedback = Feedback(
        case_id=case.id,
        user_id=user.id,
        feedback_type=body.feedback_type,
        suggested_domain=body.suggested_domain,
        comments=body.comments,
    )
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return FeedbackOut(
        id=feedback.id,
        case_id=feedback.case_id,
        feedback_type=feedback.feedback_type,
        suggested_domain=feedback.suggested_domain,
        comments=feedback.comments,
        created_at=feedback.created_at.isoformat(),
    )


@router.get("/{case_id}/feedback")
async def list_case_feedback(
    case_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    case = await _get_user_case(case_id, user, db)
    result = await db.execute(
        select(Feedback).where(Feedback.case_id == case.id).order_by(Feedback.created_at.desc())
    )
    feedbacks = result.scalars().all()
    return [
        FeedbackOut(
            id=f.id,
            case_id=f.case_id,
            feedback_type=f.feedback_type,
            suggested_domain=f.suggested_domain,
            comments=f.comments,
            created_at=f.created_at.isoformat(),
        )
        for f in feedbacks
    ]
