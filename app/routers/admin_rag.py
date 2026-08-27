"""Admin RAG Management Router"""
import os
import uuid
import re
import html
import httpx
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Form
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.deps import get_current_user, get_db
from app.models import User
from app.config import settings
from app.services.extract import extract_text
from app.rag.kb import SourceDoc, write_admin_source, _slugify
from app.rag.ingest import reindex

router = APIRouter(prefix="/api/admin/rag", tags=["admin_rag"])

def _require_admin(user: User):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

@router.post("/upload")
async def upload_knowledge_source(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    _require_admin(user)
    
    original_name = file.filename or "uploaded_knowledge"
    ext = os.path.splitext(original_name)[1].lower()
    stored_name = f"admin_{uuid.uuid4().hex}{ext or '.bin'}"
    os.makedirs(settings.uploads_dir, exist_ok=True)
    stored_path = os.path.join(settings.uploads_dir, stored_name)
    
    try:
        content = await file.read()
        with open(stored_path, "wb") as f:
            f.write(content)
        
        # Extract text
        text = await extract_text(stored_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text could be extracted from this file.")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        # Delete temp file
        if os.path.exists(stored_path):
            os.remove(stored_path)
            
    source_id = _slugify(os.path.splitext(original_name)[0])
    source = SourceDoc(
        id=source_id,
        title=original_name,
        category=category or "acts",
        source_type="primary",
        text=text,
        summary=description or f"Uploaded admin source: {original_name}",
        admin_added=True,
        demo_data=False,
    )
    write_admin_source(source)
    
    background_tasks.add_task(reindex)
    return {"status": "Ready", "filename": file.filename, "message": "Source added. Re-indexing vector store in background."}

@router.post("/url")
async def add_knowledge_url(
    background_tasks: BackgroundTasks,
    url: str,
    description: Optional[str] = None,
    category: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    _require_admin(user)
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")
    
    text = html.unescape(response.text)
    # strip script and style tags
    text = re.sub(r"<(script|style).*?>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # strip all other HTML tags
    text = re.sub(r"<[^>]*>", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Extract title from HTML
    title_match = re.search(r"<title>(.*?)</title>", response.text, flags=re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else url
    
    source_id = _slugify(title[:30] or "url-source")
    source = SourceDoc(
        id=source_id,
        title=title,
        category=category or "acts",
        source_type="primary",
        source_url=url,
        text=text,
        summary=description or f"URL: {url}",
        admin_added=True,
        demo_data=False,
    )
    write_admin_source(source)
    
    background_tasks.add_task(reindex)
    return {"status": "Ready", "url": url, "message": "URL added. Re-indexing vector store in background."}
