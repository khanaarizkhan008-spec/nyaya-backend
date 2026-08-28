"""Nyaya — FastAPI application factory + lifespan + admin seed."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.config import settings
from app.database import Base, SessionLocal, engine
from app.models import User
from app.rag.ingest import ensure_indexed
from app.security import hash_password

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nyaya")


async def _seed_admin():
    """Create the default admin user if it doesn't exist."""
    async with SessionLocal() as db:
        result = await db.execute(select(User).where(User.email == settings.admin_email))
        if result.scalar_one_or_none():
            return
        admin = User(
            email=settings.admin_email,
            hashed_password=hash_password(settings.admin_password),
            full_name="Nyaya Admin",
            role="admin",
        )
        db.add(admin)
        await db.commit()
        logger.info("Seeded admin user: %s", settings.admin_email)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")

    await _seed_admin()
    logger.info("Admin seed check done")

    await ensure_indexed()
    logger.info("Vector index ready")

    yield

    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-assisted Indian legal navigation platform. "
        "Provides legal information and document preparation — "
        "does NOT replace a qualified legal professional."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nyaya-frontend-git-main-khanaarizkhan008-specs-projects.vercel.app","https://nyaya-frontend-gules.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
from app.routers import admin, admin_rag, auth, cases, documents, evidence  # noqa: E402

app.include_router(auth.router)
app.include_router(cases.router)
app.include_router(evidence.router)
app.include_router(documents.router)
app.include_router(admin.router)
app.include_router(admin_rag.router)


@app.get("/api/health")
async def health():
    from app.llm.client import llm_client
    llm_info = await llm_client.provider_info()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "llm": llm_info,
    }
