"""API request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


# ---------------------------------------------------------------- Auth
class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(default="", max_length=120)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    role: str

    model_config = {"from_attributes": True}


class TokenOut(BaseModel):
    token: str
    user: UserOut


# ---------------------------------------------------------------- Cases
class CaseCreateIn(BaseModel):
    description: str = Field(min_length=15, max_length=8000)
    title: str = Field(default="", max_length=200)


class AnswersIn(BaseModel):
    answers: dict[str, str] = Field(default_factory=dict)
    skip: bool = False


class CaseListItem(BaseModel):
    id: str
    title: str
    status: str
    domain_code: str
    emergency_flag: bool
    created_at: str
    updated_at: str


# ---------------------------------------------------------------- Documents
class DocumentCreateIn(BaseModel):
    doc_type: str
    extra_fields: dict[str, str] = Field(default_factory=dict)


class SectionUpdate(BaseModel):
    heading: str
    body: str


class DocumentUpdateIn(BaseModel):
    sections: list[SectionUpdate]


# ---------------------------------------------------------------- Admin
class AdminSourceIn(BaseModel):
    id: str | None = None
    title: str
    category: str = "acts"
    source_type: str = "primary"
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
    text: str = Field(min_length=20, max_length=50000)


class AdminSourceOut(BaseModel):
    id: str
    title: str
    category: str
    source_type: str
    instrument: str | None = None
    section: str | None = None
    court: str | None = None
    authority: str | None = None
    date: str | None = None
    jurisdiction: str | None = None
    source_url: str | None = None
    demo_data: bool = True
    admin_added: bool = False
    summary: str = ""


# ---------------------------------------------------------------- Feedback
class FeedbackIn(BaseModel):
    feedback_type: str = "misclassification"
    suggested_domain: str = ""
    comments: str = Field(min_length=5, max_length=4000)


class FeedbackOut(BaseModel):
    id: str
    case_id: str
    feedback_type: str
    suggested_domain: str
    comments: str
    created_at: str

    model_config = {"from_attributes": True}

