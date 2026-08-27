"""Documents router — generate, list, get, update, export."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import orchestrator
from app.agents.templates import template_summaries
from app.deps import get_current_user, get_db
from app.export import export_document
from app.models import Case, LegalDocument, User
from app.schemas.agents import GeneratedDocument
from app.schemas.api import DocumentCreateIn, DocumentUpdateIn

router = APIRouter(prefix="/api/cases/{case_id}/documents", tags=["documents"])


async def _get_user_case(case_id: str, user: User, db: AsyncSession) -> Case:
    result = await db.execute(
        select(Case).where(Case.id == case_id, Case.user_id == user.id)
    )
    case = result.scalar_one_or_none()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


def _doc_out(d: LegalDocument) -> dict:
    return {
        "id": d.id,
        "doc_type": d.doc_type,
        "title": d.title,
        "document": d.content,
        "created_at": d.created_at.isoformat(),
        "updated_at": d.updated_at.isoformat(),
    }


@router.get("/types")
async def document_types(user: User = Depends(get_current_user)):
    """Template metadata (labels, descriptions, information-check fields)."""
    return {"types": template_summaries()}


@router.post("", status_code=201)
async def generate_document(
    case_id: str,
    body: DocumentCreateIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run Agent 5 (template + rules) and the anti-hallucination validator."""
    case = await _get_user_case(case_id, user, db)
    if not case.structured_case:
        raise HTTPException(status_code=400, detail="Intake not completed")
    if not case.research:
        raise HTTPException(status_code=400, detail="Analysis not completed yet")

    try:
        legal_doc = await orchestrator.generate_document(
            db, case, user, body.doc_type, body.extra_fields
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _doc_out(legal_doc)


@router.get("")
async def list_documents(
    case_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    case = await _get_user_case(case_id, user, db)
    result = await db.execute(
        select(LegalDocument)
        .where(LegalDocument.case_id == case.id)
        .order_by(LegalDocument.created_at.desc())
    )
    return [
        {
            "id": d.id,
            "doc_type": d.doc_type,
            "title": d.title,
            "created_at": d.created_at.isoformat(),
            "updated_at": d.updated_at.isoformat(),
        }
        for d in result.scalars()
    ]


@router.get("/{doc_id}")
async def get_document(
    case_id: str,
    doc_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    case = await _get_user_case(case_id, user, db)
    result = await db.execute(
        select(LegalDocument).where(
            LegalDocument.id == doc_id, LegalDocument.case_id == case.id
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_out(doc)


@router.put("/{doc_id}")
async def update_document(
    case_id: str,
    doc_id: str,
    body: DocumentUpdateIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Edit section bodies in the review UI."""
    case = await _get_user_case(case_id, user, db)
    result = await db.execute(
        select(LegalDocument).where(
            LegalDocument.id == doc_id, LegalDocument.case_id == case.id
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    content = dict(doc.content or {})
    existing = {s["heading"]: s for s in content.get("sections", [])}
    # Apply edits by heading, keeping tag/notes of the original sections
    updated: list[dict] = []
    for section in body.sections:
        base = existing.get(section.heading, {})
        merged = {
            **base,
            "heading": section.heading,
            "body": section.body,
        }
        updated.append(merged)
    content["sections"] = updated
    doc.content = content
    await db.commit()
    await db.refresh(doc)
    return _doc_out(doc)


@router.get("/{doc_id}/export/{fmt}")
async def export_doc(
    case_id: str,
    doc_id: str,
    fmt: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if fmt not in ("pdf", "docx", "txt"):
        raise HTTPException(status_code=400, detail="Format must be pdf, docx, or txt")

    case = await _get_user_case(case_id, user, db)
    result = await db.execute(
        select(LegalDocument).where(
            LegalDocument.id == doc_id, LegalDocument.case_id == case.id
        )
    )
    doc_row = result.scalar_one_or_none()
    if not doc_row:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc_row.content:
        raise HTTPException(status_code=400, detail="Document has no content")

    gen_doc = GeneratedDocument.model_validate(doc_row.content)
    content_bytes, content_type, ext = export_document(gen_doc, fmt)

    filename = f"nyaya_{doc_row.doc_type}_{doc_row.id[:8]}{ext}"
    return Response(
        content=content_bytes,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
