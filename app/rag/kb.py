"""Knowledge-base corpus scanning + chunking.

Corpus layout (spec §15):
    data/legal/constitution/*.json
    data/legal/acts/*.json
    data/legal/rules/*.json
    data/legal/judgments/*.json
    data/legal/government/*.json
    data/legal/_admin/*.json   <- sources added via the admin UI

Each JSON file contains a source object or a list of source objects.
Source metadata is preserved on every chunk (never lost during chunking).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger("nyaya.kb")

VALID_CATEGORIES = {"constitution", "acts", "rules", "judgments", "government"}


class SourceDoc(BaseModel):
    id: str
    title: str
    category: str = "acts"
    source_type: str = "primary"  # primary | secondary
    instrument: str | None = None
    section: str | None = None
    court: str | None = None
    authority: str | None = None
    date: str | None = None
    jurisdiction: str | None = "India"
    source_url: str | None = None
    verified: bool = False
    demo_data: bool = True
    summary: str = ""
    text: str = ""
    admin_added: bool = False


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60] or "source"


def _validate_source(raw: dict, category: str, admin_added: bool) -> SourceDoc:
    data = dict(raw)
    data.setdefault("category", category)
    if data.get("category") not in VALID_CATEGORIES:
        data["category"] = "acts"
    if data.get("source_type") not in ("primary", "secondary"):
        data["source_type"] = "primary"
    if not data.get("id"):
        data["id"] = _slugify(str(data.get("title", "source")))
    data["admin_added"] = admin_added
    data.setdefault("demo_data", True)
    data.setdefault("verified", False)
    return SourceDoc(**data)


def admin_dir() -> str:
    return os.path.join(settings.data_dir, "_admin")


def scan_corpus() -> list[SourceDoc]:
    """Load every source from the corpus directory tree."""
    sources: list[SourceDoc] = []
    seen_ids: set[str] = set()
    root = settings.data_dir
    if not os.path.isdir(root):
        logger.warning("Knowledge base directory not found: %s", root)
        return sources

    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root).replace("\\", "/")
        parts = [p for p in rel.split("/") if p and p != "."]
        category = parts[0] if parts and parts[0] in VALID_CATEGORIES else "acts"
        is_admin = "_admin" in parts
        for filename in sorted(filenames):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping unreadable corpus file %s: %s", path, exc)
                continue
            entries = payload if isinstance(payload, list) else [payload]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                try:
                    source = _validate_source(entry, category, is_admin)
                except Exception as exc:
                    logger.warning("Skipping invalid source in %s: %s", path, exc)
                    continue
                if source.id in seen_ids:
                    logger.warning("Duplicate source id skipped: %s", source.id)
                    continue
                seen_ids.add(source.id)
                sources.append(source)

    sources.sort(key=lambda s: (s.category, s.title))
    return sources


def write_admin_source(source: SourceDoc) -> str:
    """Persist an admin-added source as its own JSON file."""
    directory = admin_dir()
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{source.id}.json")
    payload = source.model_dump(exclude={"admin_added"})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def delete_admin_source(source_id: str) -> bool:
    path = os.path.join(admin_dir(), f"{source_id}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking that keeps citations self-contained."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n{para}".strip()
        if len(candidate) <= size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(para) > size:
            chunks.append(para[:size])
            para = para[max(0, size - overlap):]
        current = para
    if current:
        chunks.append(current)
    return chunks


def build_chunks(sources: list[SourceDoc]) -> list[tuple[SourceDoc, str, str]]:
    """Return (source, chunk_text, chunk_id) triples."""
    out: list[tuple[SourceDoc, str, str]] = []
    for source in sources:
        body = source.text.strip() or source.summary.strip()
        if not body:
            continue
        pieces = chunk_text(body)
        for index, piece in enumerate(pieces):
            chunk_id = f"{source.id}::{index}" if len(pieces) > 1 else source.id
            out.append((source, piece, chunk_id))
    return out


def source_metadata(source: SourceDoc) -> dict[str, Any]:
    return {
        "source_id": source.id,
        "title": source.title,
        "category": source.category,
        "instrument": source.instrument or "",
        "section": source.section or "",
        "court": source.court or "",
        "authority": source.authority or "",
        "date": source.date or "",
        "jurisdiction": source.jurisdiction or "",
        "source_type": source.source_type,
        "source_url": source.source_url or "",
        "demo_data": bool(source.demo_data),
        "verified": bool(source.verified),
    }
