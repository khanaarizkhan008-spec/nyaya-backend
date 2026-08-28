"""Admin router — knowledge base management + system info."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_db, require_admin
from app.llm.client import llm_client
from app.models import User
from app.rag.ingest import reindex
from app.rag.kb import (
    SourceDoc,
    delete_admin_source,
    scan_corpus,
    write_admin_source,
)
from app.rag.store import get_store
from app.schemas.api import AdminSourceIn, AdminSourceOut

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/status")
async def system_status(user: User = Depends(require_admin)):
    store = get_store()
    count = await asyncio.to_thread(store.count)
    meta = await asyncio.to_thread(store.get_meta)
    llm_info = await llm_client.provider_info()
    return {
        "vector_store": {
            "chunks": count,
            "embedding": meta.get("embedding", "unknown"),
            "sources": meta.get("sources", 0),
            "updated_at": meta.get("updated_at"),
        },
        "llm": llm_info,
    }


@router.get("/sources")
async def list_sources(user: User = Depends(require_admin)):
    sources = scan_corpus()
    return [
        AdminSourceOut(
            id=s.id,
            title=s.title,
            category=s.category,
            source_type=s.source_type,
            instrument=s.instrument,
            section=s.section,
            court=s.court,
            authority=s.authority,
            date=s.date,
            jurisdiction=s.jurisdiction,
            source_url=s.source_url,
            demo_data=s.demo_data,
            admin_added=s.admin_added,
            summary=s.summary[:200],
        ).model_dump()
        for s in sources
    ]


@router.post("/sources", status_code=201)
async def add_source(
    body: AdminSourceIn,
    user: User = Depends(require_admin),
):
    from app.rag.kb import _slugify
    source_id = body.id or _slugify(body.title)
    source = SourceDoc(
        id=source_id,
        title=body.title,
        category=body.category,
        source_type=body.source_type,
        instrument=body.instrument,
        section=body.section,
        court=body.court,
        authority=body.authority,
        date=body.date,
        jurisdiction=body.jurisdiction,
        source_url=body.source_url,
        verified=body.verified,
        demo_data=body.demo_data,
        summary=body.summary,
        text=body.text,
        admin_added=True,
    )
    path = write_admin_source(source)
    return {"id": source.id, "path": path, "message": "Source saved. Re-index to update vector store."}


@router.delete("/sources/{source_id}")
async def remove_source(
    source_id: str,
    user: User = Depends(require_admin),
):
    removed = delete_admin_source(source_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Admin source not found")
    return {"message": "Source removed. Re-index to update vector store."}


@router.post("/reindex")
async def trigger_reindex(user: User = Depends(require_admin)):
    meta = await reindex()
    return {"message": "Re-indexed successfully", "meta": meta}


@router.get("/feedbacks")
async def list_all_feedbacks(
    user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    from app.models import Feedback, Case
    result = await db.execute(
        select(Feedback, Case.title, Case.domain_code)
        .join(Case, Feedback.case_id == Case.id, isouter=True)
        .order_by(Feedback.created_at.desc())
    )
    rows = result.all()
    return [
        {
            "id": f.id,
            "case_id": f.case_id,
            "case_title": title or "Untitled",
            "domain_code": domain_code or "",
            "feedback_type": f.feedback_type,
            "suggested_domain": f.suggested_domain,
            "comments": f.comments,
            "created_at": f.created_at.isoformat(),
        }
        for f, title, domain_code in rows
    ]

