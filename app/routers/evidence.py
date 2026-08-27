"""Evidence router — upload documents, list, get analysis."""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents import orchestrator
from app.config import settings
from app.deps import get_current_user, get_db
from app.models import Case, Evidence, User

router = APIRouter(prefix="/api/cases/{case_id}/evidence", tags=["evidence"])

ALLOWED_MIME = {
    "application/pdf",
    "image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "application/octet-stream",
}

ALLOWED_EXT = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".docx", ".doc", ".txt", ".bin"}


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


def _evidence_out(e: Evidence) -> dict:
    return {
        "id": e.id,
        "filename": e.filename,
        "mime": e.mime,
        "size_bytes": e.size_bytes,
        "status": e.status,
        "analysis": e.analysis,
        "error": e.error,
        "created_at": e.created_at.isoformat(),
    }


@router.post("", status_code=201)
async def upload_evidence(
    case_id: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document: Agent 4 analyzes it in the background — poll the
    list/get endpoints (or the case view) for the analysis result."""
    case = await _get_user_case(case_id, user, db)
    if not case.structured_case:
        raise HTTPException(status_code=400, detail="Complete intake first")

    original_name = file.filename or "document"
    content_type = file.content_type or "application/octet-stream"
    ext = os.path.splitext(original_name)[1].lower()

    if content_type == "application/octet-stream" and ext:
        import mimetypes
        guessed, _ = mimetypes.guess_type(original_name)
        if guessed:
            content_type = guessed

    if content_type not in ALLOWED_MIME and not content_type.startswith("text/") and ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}")

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_upload_mb}MB limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    stored_name = f"{uuid.uuid4().hex}{ext or '.bin'}"
    os.makedirs(settings.uploads_dir, exist_ok=True)
    stored_path = os.path.join(settings.uploads_dir, stored_name)
    with open(stored_path, "wb") as f:
        f.write(data)

    evidence = Evidence(
        case_id=case.id,
        filename=original_name,
        stored_path=stored_path,
        mime=content_type,
        size_bytes=len(data),
        status="processing",
    )
    db.add(evidence)
    await db.commit()
    await db.refresh(evidence)

    orchestrator.schedule_evidence_analysis(evidence.id)
    return _evidence_out(evidence)


@router.get("")
async def list_evidence(
    case_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    case = await _get_user_case(case_id, user, db)
    result = await db.execute(
        select(Evidence)
        .where(Evidence.case_id == case.id)
        .order_by(Evidence.created_at.asc())
    )
    return [_evidence_out(e) for e in result.scalars()]


@router.get("/{evidence_id}")
async def get_evidence(
    case_id: str,
    evidence_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    case = await _get_user_case(case_id, user, db)
    result = await db.execute(
        select(Evidence).where(
            Evidence.id == evidence_id, Evidence.case_id == case.id
        )
    )
    evidence = result.scalar_one_or_none()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return _evidence_out(evidence)


@router.delete("/{evidence_id}")
async def delete_evidence(
    case_id: str,
    evidence_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    case = await _get_user_case(case_id, user, db)
    result = await db.execute(
        select(Evidence).where(
            Evidence.id == evidence_id, Evidence.case_id == case.id
        )
    )
    evidence = result.scalar_one_or_none()
    if not evidence:
        raise HTTPException(status_code=404, detail="Evidence not found")
    try:
        if evidence.stored_path and os.path.exists(evidence.stored_path):
            os.remove(evidence.stored_path)
    except OSError:
        pass  # file removal is best-effort
    await db.delete(evidence)
    await db.commit()
    return {"message": "Evidence removed"}
