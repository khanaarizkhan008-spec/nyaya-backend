"""Document templates for the Document Generation Agent.

Eight structured templates, each built from the case's verified data:

  * ``verified_fact``   — copied from what the user said / evidence excerpts
  * ``legal_source``    — quotes the retrieved legal sources (with source ids)
  * ``ai_generated``    — wording generated from the recommended pathway
  * ``needs_review``    — content that must be human-checked before use
  * ``placeholder``     — personal details the user must fill in

Nothing outside these categories is ever inserted into a document, and no
statute is ever cited unless it appears in the retrieved research for the
case (enforced by validator.py on the assembled document).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from app.schemas.agents import (
    DocSection,
    EvidenceAnalysis,
    LegalReasoning,
    ResearchResult,
    SourceRef,
    StructuredCase,
)


@dataclass
class DocContext:
    """Everything a template may draw on while assembling a document."""

    structured: StructuredCase
    research: ResearchResult
    reasoning: LegalReasoning | None = None
    evidence: list[tuple[str, EvidenceAnalysis]] = field(default_factory=list)
    extra_fields: dict[str, str] = field(default_factory=dict)
    today: str = field(default_factory=lambda: date.today().strftime("%d %B %Y"))
    # the user's raw inputs — used by the validator for fact tracing
    source_description: str = ""
    answers: dict[str, str] = field(default_factory=dict)
    # source ids actually cited while building (consumed by document_gen)
    cited: list[str] = field(default_factory=list)


@dataclass
class FieldSpec:
    """An information item the generator asks the user for."""

    key: str
    label: str
    example: str = ""
    required: bool = True
    input_type: str = "text"  # text | textarea


@dataclass
class Template:
    doc_type: str
    label: str
    description: str
    fields: list[FieldSpec]
    build: Callable[[DocContext], tuple[str, list[DocSection]]]


# ---------------------------------------------------------------- helpers
def _fv(ctx: DocContext, key: str, label: str) -> str:
    """Field value, or an explicit placeholder when not provided."""
    value = (ctx.extra_fields.get(key) or "").strip()
    return value if value else f"[{label.upper()} — TO BE PROVIDED]"


def _filled(ctx: DocContext, key: str) -> bool:
    return bool((ctx.extra_fields.get(key) or "").strip())


def _header_tag(ctx: DocContext, keys: list[str]) -> str:
    """A header section is a verified fact once every detail is filled in."""
    return "verified_fact" if all(_filled(ctx, k) for k in keys) else "placeholder"


def _cite(ctx: DocContext, source: SourceRef) -> str:
    ctx.cited.append(source.source_id)
    parts: list[str] = []
    if source.section:
        parts.append(source.section)
    if source.instrument:
        parts.append(source.instrument)
    label = ", ".join(parts) if parts else source.title
    if source.court:
        label = f"{source.court} in {label}" if label else source.court
    return f"{label} [{source.source_id}]"


def _law_sources(ctx: DocContext, limit: int = 4) -> list[SourceRef]:
    if ctx.reasoning and ctx.reasoning.possible_laws:
        return ctx.reasoning.possible_laws[:limit]
    primary = [s for s in ctx.research.sources if s.source_type == "primary"]
    return (primary or ctx.research.sources)[:limit]


def _legal_position_section(ctx: DocContext, heading: str = "Legal position") -> DocSection:
    lines: list[str] = []
    for source in _law_sources(ctx):
        snippet = (source.snippet or "").strip()
        if len(snippet) > 220:
            snippet = snippet[:217].rstrip() + "..."
        line = f"- {_cite(ctx, source)}"
        if snippet:
            line += f" — “{snippet}”"
        lines.append(line)
    body = (
        "\n".join(lines)
        if lines
        else "[NO VERIFIED LEGAL SOURCE COULD BE RETRIEVED FOR THIS MATTER — "
        "LEGAL POSITION TO BE COMPLETED BY A QUALIFIED PERSON]"
    )
    return DocSection(
        heading=heading,
        body=body,
        tag="legal_source" if lines else "needs_review",
        notes="Quoted from the sources retrieved for this case."
        if lines
        else "Retrieval found no verified source; nothing is asserted.",
    )


def _facts_section(ctx: DocContext, heading: str = "Statement of facts") -> DocSection:
    facts = ctx.structured.key_facts or [ctx.structured.problem_summary]
    body = "\n".join(f"{i}. {fact}" for i, fact in enumerate(facts[:10], start=1))
    return DocSection(
        heading=heading,
        body=body,
        tag="verified_fact",
        notes="Each point is taken from your own description and answers — "
        "edit anything that is inaccurate before sending.",
    )


def _evidence_section(ctx: DocContext) -> DocSection | None:
    if not ctx.evidence:
        return None
    lines = []
    for filename, analysis in ctx.evidence:
        guess = analysis.document_type_guess or "document"
        line = f"- {filename} — appears to be a {guess}"
        if analysis.entities.dates:
            line += f", dated entries found ({', '.join(analysis.entities.dates[:3])})"
        if analysis.entities.amounts:
            line += f", amounts found ({', '.join(analysis.entities.amounts[:3])})"
        lines.append(line)
    return DocSection(
        heading="Annexure — documents relied upon",
        body="\n".join(lines)
        + "\n(Copies of the above are attached to this document.)",
        tag="needs_review",
        notes="Generated from the evidence-analysis agent. Verify each "
        "attachment before filing.",
    )


def _closing_section(ctx: DocContext, role: str, name_key: str) -> DocSection:
    body = (
        f"Place: {_fv(ctx, 'place', 'place')}\n"
        f"Date: {ctx.today}\n\n"
        f"{_fv(ctx, name_key, 'your full name')}\n"
        f"({role})"
    )
    return DocSection(
        heading="Signature",
        body=body,
        tag="placeholder",
        notes="Fill in your details and sign.",
    )


def _demand_section(ctx: DocContext) -> DocSection:
    outcome = ctx.structured.desired_outcome or "the relief described above"
    days = _fv(ctx, "compliance_days", "number of days for compliance")
    remedy_desc = ""
    if ctx.reasoning and ctx.reasoning.possible_remedies:
        remedy_desc = ctx.reasoning.possible_remedies[0].description
    body = (
        f"In the circumstances stated above, you are called upon to "
        f"{outcome.lower()} within {days} days of receipt of this notice. "
        + (f"For context: {remedy_desc} " if remedy_desc else "")
        + "This notice is issued without prejudice to all other rights and "
        "remedies available in law."
    )
    return DocSection(
        heading="Demand",
        body=" ".join(body.split()),
        tag="ai_generated",
        notes="Generated from the recommended pathway — review the wording "
        "before sending.",
    )


def _parse_amount(amount: str | None) -> float | None:
    if not amount:
        return None
    digits = re.sub(r"[^\d.]", "", amount)
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


# ---------------------------------------------------------------- 1. legal notice
def _build_legal_notice(ctx: DocContext) -> tuple[str, list[DocSection]]:
    header = (
        f"LEGAL NOTICE\n\n"
        f"Date: {ctx.today}\n\n"
        f"By:\n{_fv(ctx, 'sender_name', 'your full name')}\n"
        f"{_fv(ctx, 'sender_address', 'your address')}\n\n"
        f"To:\n{_fv(ctx, 'recipient_name', 'recipient name')}\n"
        f"{_fv(ctx, 'recipient_address', 'recipient address')}\n\n"
        f"Subject: {_fv(ctx, 'subject', 'subject of the notice')}"
    )
    sections = [
        DocSection(
            heading="Notice particulars",
            body=header,
            tag=_header_tag(
                ctx,
                ["sender_name", "sender_address", "recipient_name", "recipient_address"],
            ),
            notes="Sender and recipient details are required before sending.",
        ),
        _facts_section(ctx),
        _legal_position_section(ctx),
        _demand_section(ctx),
        DocSection(
            heading="Consequences of non-compliance",
            body=(
                "If the demand made above is not complied with within the "
                "stated period, the undersigned will be constrained to seek "
                "the remedies available under law, including approaching the "
                "appropriate forum, at your risk as to costs and consequences."
            ),
            tag="ai_generated",
            notes="Standard boilerplate generated for this notice type.",
        ),
    ]
    annex = _evidence_section(ctx)
    if annex:
        sections.append(annex)
    sections.append(_closing_section(ctx, "Sender", "sender_name"))
    return f"Legal Notice — {ctx.structured.legal_domain}", sections


# ---------------------------------------------------------------- 2. consumer complaint
def _commission_for(amount: str | None) -> str:
    value = _parse_amount(amount)
    if value is None:
        return "District Consumer Commission (confirm the value of your claim)"
    if value <= 5_000_000:
        return "District Consumer Commission (claims up to ₹50 lakh)"
    if value <= 20_000_000:
        return "State Consumer Commission (₹50 lakh – ₹2 crore)"
    return "National Consumer Commission (claims above ₹2 crore)"


def _build_consumer_complaint(ctx: DocContext) -> tuple[str, list[DocSection]]:
    commission = _commission_for(ctx.structured.amount_in_dispute)
    caption = (
        f"BEFORE THE {_commission_for(ctx.structured.amount_in_dispute).split(' (')[0].upper()}, "
        f"{_fv(ctx, 'city', 'your city')}\n\n"
        f"{_fv(ctx, 'complainant_name', 'your full name')}\n"
        f"{_fv(ctx, 'complainant_address', 'your address')}\n"
        "                                    ... Complainant\n"
        f"VERSUS\n"
        f"{_fv(ctx, 'opposite_party_name', 'opposite party name')}\n"
        f"{_fv(ctx, 'opposite_party_address', 'opposite party address')}\n"
        "                                    ... Opposite Party"
    )
    sections = [
        DocSection(
            heading="Caption",
            body=caption,
            tag=_header_tag(
                ctx,
                ["complainant_name", "complainant_address", "opposite_party_name", "city"],
            ),
            notes=f"Based on your claim value, the suggested forum is: {commission}.",
        ),
        DocSection(
            heading="Facts of the complaint",
            body=(
                f"1. The Complainant purchased / availed the goods or service "
                f"on {_fv(ctx, 'purchase_date', 'date of purchase / service')} "
                f"for {_fv(ctx, 'amount_paid', 'amount paid')}.\n"
                + "\n".join(
                    f"{i}. {fact}"
                    for i, fact in enumerate(ctx.structured.key_facts[:8], start=2)
                )
            ),
            tag="verified_fact",
            notes="Point 1 uses the details you provided; the remaining points "
            "come from your description — verify each.",
        ),
        _legal_position_section(ctx, heading="Law and jurisdiction"),
        DocSection(
            heading="Deficiency / complaint",
            body=(
                f"The Opposite Party is guilty of deficiency in service / "
                f"unfair trade practice in that {ctx.structured.problem_summary} "
                "The particulars are as stated above."
            ),
            tag="ai_generated",
            notes="Generated from your problem summary — review the wording.",
        ),
        DocSection(
            heading="Reliefs sought",
            body=(
                f"It is respectfully prayed that this Hon'ble Commission be "
                f"pleased to: (a) direct refund / replacement / repair of the "
                f"goods or services; (b) award compensation for the loss and "
                f"inconvenience suffered; (c) award the cost of this "
                f"complaint; and (d) pass such other orders as deemed fit."
            ),
            tag="ai_generated",
            notes="Adjust the reliefs to what you actually want.",
        ),
    ]
    annex = _evidence_section(ctx)
    if annex:
        sections.append(annex)
    sections.append(
        DocSection(
            heading="Verification",
            body=(
                "I, the Complainant above named, verify that the contents of "
                "this complaint are true to my knowledge and belief and "
                "nothing material has been concealed.\n\n"
                f"Place: {_fv(ctx, 'place', 'place')}\nDate: {ctx.today}"
            ),
            tag="placeholder",
        )
    )
    return "Consumer Complaint — under the Consumer Protection Act, 2019", sections


# ---------------------------------------------------------------- 3. police complaint
def _build_police_complaint(ctx: DocContext) -> tuple[str, list[DocSection]]:
    sections = [
        DocSection(
            heading="Addressee",
            body=(
                f"To\nThe Station House Officer\n"
                f"{_fv(ctx, 'police_station', 'police station name')}\n"
                f"{_fv(ctx, 'city', 'city / district')}"
            ),
            tag=_header_tag(ctx, ["police_station", "city"]),
        ),
        DocSection(
            heading="Complaint",
            body=(
                f"I, {_fv(ctx, 'complainant_name', 'your full name')}, "
                f"resident of {_fv(ctx, 'complainant_address', 'your address')}, "
                f"submit as under:\n\n"
                + "\n".join(
                    f"{i}. {fact}"
                    for i, fact in enumerate(ctx.structured.key_facts[:10], start=1)
                )
            ),
            tag="verified_fact",
            notes="Facts are taken verbatim from your description — verify "
            "before submitting.",
        ),
        _legal_position_section(ctx, heading="Offence indicated"),
        DocSection(
            heading="Request",
            body=(
                "In view of the above, I request that the information be "
                "registered as a complaint / First Information Report as the "
                "facts may disclose a cognizable offence, and that a copy of "
                "the registered complaint with the diary number be provided "
                "to me. I am enclosing the documents listed in the annexure."
            ),
            tag="ai_generated",
            notes="If the police refuse to register the FIR, the law provides "
            "further remedies (see your case pathway).",
        ),
    ]
    annex = _evidence_section(ctx)
    if annex:
        sections.append(annex)
    sections.append(_closing_section(ctx, "Complainant", "complainant_name"))
    return "Police Complaint", sections


# ---------------------------------------------------------------- 4. RTI application
RTI_HINTS: dict[str, list[str]] = {
    "tenancy_deposit": [
        "copies of any registered tenancy agreement and related records for the property",
    ],
    "property_dispute": [
        "mutation register extracts, survey records and municipal file copies for the property",
    ],
    "police_fir_refusal": [
        "a certified copy of the Daily Diary register entry for the date of my visit",
    ],
    "unpaid_wages": [
        "records of the establishment's registration and wage-related filings",
    ],
    "consumer_defective": [
        "records of any complaints or escalations registered against the seller/service provider",
    ],
}


def _build_rti_application(ctx: DocContext) -> tuple[str, list[DocSection]]:
    details = (ctx.extra_fields.get("info_details") or "").strip()
    if details:
        info_body = details
        info_tag = "verified_fact"
        info_notes = "Recorded as provided by you."
    else:
        hints = RTI_HINTS.get(ctx.structured.domain_code, [])
        items = hints or ["records relating to the matter described in the facts below"]
        info_body = (
            "I request the following information:\n"
            + "\n".join(f"- {item}" for item in items)
        )
        info_tag = "ai_generated"
        info_notes = "Suggested by the system based on your case — edit to "
        "describe exactly what you need."
    sections = [
        DocSection(
            heading="Addressee",
            body=(
                f"To\nThe Public Information Officer\n"
                f"{_fv(ctx, 'public_authority', 'name of the public authority')}\n"
                f"{_fv(ctx, 'pio_office', 'office / address')}"
            ),
            tag=_header_tag(ctx, ["public_authority", "pio_office"]),
        ),
        DocSection(
            heading="Application under the Right to Information Act, 2005",
            body=(
                f"I, {_fv(ctx, 'applicant_name', 'your full name')}, resident "
                f"of {_fv(ctx, 'applicant_address', 'your address')}, seek the "
                "information specified below, in my capacity as a citizen of "
                "India. The relevant facts are: "
                f"{ctx.structured.problem_summary}"
            ),
            tag="ai_generated",
            notes="Generated from your case summary — verify before sending.",
        ),
        DocSection(heading="Information requested", body=info_body, tag=info_tag, notes=info_notes),
        _legal_position_section(ctx, heading="Applicable provisions"),
        DocSection(
            heading="Fee and declaration",
            body=(
                "The prescribed application fee is being paid "
                f"{_fv(ctx, 'fee_mode', 'mode of payment (e.g. ₹10 postal order / online)')}. "
                "I declare that the information sought is not exempt under the "
                "Act. Kindly provide the information within the period "
                "prescribed by the Act, failing which an appeal lies as of right."
            ),
            tag="placeholder",
        ),
    ]
    sections.append(_closing_section(ctx, "Applicant", "applicant_name"))
    return "RTI Application", sections


# ---------------------------------------------------------------- 5. labour complaint
def _build_labour_complaint(ctx: DocContext) -> tuple[str, list[DocSection]]:
    sections = [
        DocSection(
            heading="Addressee",
            body=(
                f"To\nThe Labour Commissioner / Assistant Labour Commissioner\n"
                f"{_fv(ctx, 'office_city', 'labour office city / jurisdiction')}"
            ),
            tag=_header_tag(ctx, ["office_city"]),
        ),
        DocSection(
            heading="Complaint",
            body=(
                f"I, {_fv(ctx, 'worker_name', 'your full name')}, residing at "
                f"{_fv(ctx, 'worker_address', 'your address')}, was employed by "
                f"{_fv(ctx, 'employer_name', 'employer name')}, "
                f"{_fv(ctx, 'employer_address', 'employer address')}. "
                f"The following wages/dues remain unpaid: "
                f"{_fv(ctx, 'months_due', 'months / amounts due')}.\n\n"
                + "\n".join(
                    f"- {fact}" for fact in ctx.structured.key_facts[:8]
                )
            ),
            tag="verified_fact",
            notes="Employment details come from your answers — verify each item.",
        ),
        _legal_position_section(ctx, heading="Legal position"),
        DocSection(
            heading="Relief sought",
            body=(
                "I request that the employer be called upon to pay the "
                "unpaid wages/dues with interest as applicable, and that "
                "conciliation proceedings be initiated. I undertake to "
                "produce my employment records as listed in the annexure."
            ),
            tag="ai_generated",
        ),
    ]
    annex = _evidence_section(ctx)
    if annex:
        sections.append(annex)
    sections.append(_closing_section(ctx, "Complainant / Workman", "worker_name"))
    return "Labour Complaint — unpaid wages / dues", sections


# ---------------------------------------------------------------- 6. cyber complaint
def _build_cyber_complaint(ctx: DocContext) -> tuple[str, list[DocSection]]:
    sections = [
        DocSection(
            heading="Addressee",
            body=(
                "To\nThe Station House Officer / Nodal Officer — Cyber Crime\n"
                f"{_fv(ctx, 'police_station', 'police / cyber police station')}\n"
                f"{_fv(ctx, 'city', 'city / district')}\n\n"
                "(This complaint is also being filed on the National Cyber "
                "Crime Reporting Portal — cybercrime.gov.in)"
            ),
            tag=_header_tag(ctx, ["police_station", "city"]),
        ),
        DocSection(
            heading="Complaint",
            body=(
                f"I, {_fv(ctx, 'victim_name', 'your full name')}, resident of "
                f"{_fv(ctx, 'victim_address', 'your address')}, submit that "
                f"on {_fv(ctx, 'incident_date', 'date of the incident')} I was "
                f"defrauded through {_fv(ctx, 'platform', 'platform / app / website / phone number used')}. "
                "The relevant facts are:\n\n"
                + "\n".join(f"- {fact}" for fact in ctx.structured.key_facts[:8])
            ),
            tag="verified_fact",
        ),
        DocSection(
            heading="Transaction details",
            body=(
                "The following transactions are connected to this matter: "
                f"{_fv(ctx, 'transaction_refs', 'transaction references / UTRs / account numbers, with amounts and times')}. "
                "Bank statements evidencing these transactions are enclosed."
            ),
            tag="verified_fact",
            notes="Replace the placeholder with your actual transaction records.",
        ),
        _legal_position_section(ctx, heading="Offence indicated"),
        DocSection(
            heading="Request",
            body=(
                "I request that (a) the complaint be registered and "
                "investigated; (b) the concerned bank / platform be directed "
                "to preserve the transaction and account records, and to "
                "place a hold on the beneficiary account where possible; and "
                "(c) a copy of the registered complaint be provided to me. "
                "I am available to provide any further particulars."
            ),
            tag="ai_generated",
        ),
    ]
    annex = _evidence_section(ctx)
    if annex:
        sections.append(annex)
    sections.append(_closing_section(ctx, "Complainant", "victim_name"))
    return "Cyber Crime Complaint", sections


# ---------------------------------------------------------------- 7. writ petition
def _build_writ_petition(ctx: DocContext) -> tuple[str, list[DocSection]]:
    sections = [
        DocSection(
            heading="Court",
            body=(
                f"IN THE HIGH COURT OF {_fv(ctx, 'high_court', 'high court (state)')}\n"
                "AT [BENCH — TO BE PROVIDED]\n\n"
                "(SPECIAL CIVIL JURISDICTION — WRIT PETITION UNDER ARTICLE 226 "
                "OF THE CONSTITUTION OF INDIA)"
            ),
            tag="placeholder",
            notes="A writ petition must be filed through an advocate on record "
            "of the High Court concerned.",
        ),
        DocSection(
            heading="Parties",
            body=(
                f"{_fv(ctx, 'petitioner_name', 'petitioner name')}, "
                f"{_fv(ctx, 'petitioner_address', 'petitioner address')}\n"
                "                              ... Petitioner\n"
                "VERSUS\n"
                f"{_fv(ctx, 'respondent_authority', 'respondent authority / State')}\n"
                "                              ... Respondents"
            ),
            tag="placeholder",
        ),
        _facts_section(ctx, heading="Statement of facts"),
        _legal_position_section(ctx, heading="Grounds"),
        DocSection(
            heading="Prayer",
            body=(
                "It is respectfully prayed that this Hon'ble Court may be "
                "pleased to issue a writ, order or direction in the nature of "
                "[RELIEF SOUGHT — TO BE SPECIFIED BY COUNSEL], with such other "
                "orders as this Hon'ble Court deems fit in the circumstances."
            ),
            tag="needs_review",
            notes="The prayer must be settled by a qualified advocate.",
        ),
        DocSection(
            heading="IMPORTANT — professional review required",
            body=(
                "This draft is a starting point only. Writ petitions are "
                "high-stakes constitutional proceedings: the grounds, "
                "limitation and forum must be settled by a qualified "
                "advocate, and free legal aid may be available through the "
                "State Legal Services Authority (helpline 15100)."
            ),
            tag="needs_review",
        ),
    ]
    return "Writ Petition (Draft) — Article 226", sections


# ---------------------------------------------------------------- 8. representation
def _build_representation(ctx: DocContext) -> tuple[str, list[DocSection]]:
    sections = [
        DocSection(
            heading="Addressee",
            body=(
                f"To\n{_fv(ctx, 'authority_name', 'name of the authority / officer')}\n"
                f"{_fv(ctx, 'authority_address', 'address of the authority')}"
            ),
            tag=_header_tag(ctx, ["authority_name", "authority_address"]),
        ),
        DocSection(
            heading="Subject",
            body=_fv(ctx, "subject", "subject of the representation"),
            tag="placeholder",
        ),
        DocSection(
            heading="Representation",
            body=(
                f"I, {_fv(ctx, 'applicant_name', 'your full name')}, submit as under:\n\n"
                + "\n".join(
                    f"{i}. {fact}"
                    for i, fact in enumerate(ctx.structured.key_facts[:10], start=1)
                )
            ),
            tag="verified_fact",
        ),
        _legal_position_section(ctx),
        DocSection(
            heading="Request",
            body=(
                f"In view of the above, I respectfully request that "
                f"{(ctx.structured.desired_outcome or 'the matter be examined and appropriate action taken').lower()}. "
                "An acknowledgment of this representation is requested."
            ),
            tag="ai_generated",
        ),
    ]
    annex = _evidence_section(ctx)
    if annex:
        sections.append(annex)
    sections.append(_closing_section(ctx, "Applicant", "applicant_name"))
    return "Representation to Authority", sections


# ---------------------------------------------------------------- registry
TEMPLATES: dict[str, Template] = {
    t.doc_type: t
    for t in [
        Template(
            doc_type="legal_notice",
            label="Legal Notice",
            description="A formal written demand sent to the opposite party before escalation — the standard first step in most civil and consumer matters.",
            fields=[
                FieldSpec("sender_name", "Your full name", "Priya Sharma"),
                FieldSpec("sender_address", "Your postal address", "12 MG Road, Bengaluru 560001"),
                FieldSpec("recipient_name", "Recipient's name", "Ramesh Kumar / Kumar Enterprises"),
                FieldSpec("recipient_address", "Recipient's address", "34 Station Road, Pune 411001"),
                FieldSpec("subject", "Subject of the notice", "Refund of security deposit of ₹75,000"),
                FieldSpec("compliance_days", "Days allowed for compliance", "15"),
                FieldSpec("place", "Place of signing", "Bengaluru"),
            ],
            build=_build_legal_notice,
        ),
        Template(
            doc_type="consumer_complaint",
            label="Consumer Commission Complaint",
            description="A complaint draft for the District/State Consumer Commission under the Consumer Protection Act, 2019 (filed via e-Daakhil).",
            fields=[
                FieldSpec("complainant_name", "Your full name", "Priya Sharma"),
                FieldSpec("complainant_address", "Your address", "12 MG Road, Bengaluru 560001"),
                FieldSpec("opposite_party_name", "Opposite party (seller/service provider)", "ShopNest Retail Pvt Ltd"),
                FieldSpec("opposite_party_address", "Opposite party's address", "5th Floor, Tech Park, Hyderabad"),
                FieldSpec("city", "Your city (for the Commission's jurisdiction)", "Bengaluru"),
                FieldSpec("purchase_date", "Date of purchase / service", "2 January 2026"),
                FieldSpec("amount_paid", "Amount paid", "₹48,999"),
                FieldSpec("place", "Place of signing", "Bengaluru"),
            ],
            build=_build_consumer_complaint,
        ),
        Template(
            doc_type="police_complaint",
            label="Police Complaint",
            description="A written complaint to the Station House Officer seeking registration of the incident (and FIR where a cognizable offence is disclosed).",
            fields=[
                FieldSpec("complainant_name", "Your full name", "Amit Verma"),
                FieldSpec("complainant_address", "Your address", "22 Nehru Colony, Delhi 110018"),
                FieldSpec("police_station", "Police station", "Sector 21 Police Station"),
                FieldSpec("city", "City / district", "Noida"),
                FieldSpec("place", "Place of signing", "Noida"),
            ],
            build=_build_police_complaint,
        ),
        Template(
            doc_type="rti_application",
            label="RTI Application",
            description="A Right to Information application to the Public Information Officer of a public authority.",
            fields=[
                FieldSpec("applicant_name", "Your full name", "Sana Khan"),
                FieldSpec("applicant_address", "Your address", "8 Lake View, Bhopal 462001"),
                FieldSpec("public_authority", "Public authority", "Municipal Corporation of Delhi"),
                FieldSpec("pio_office", "Office / address of the PIO", "RTI Cell, MCD HQ"),
                FieldSpec("info_details", "Information you want (be specific)", "Copies of the sanctioned building plan and occupation certificate for plot 44...", required=False, input_type="textarea"),
                FieldSpec("fee_mode", "Mode of paying the ₹10 fee", "₹10 Indian Postal Order / online payment"),
                FieldSpec("place", "Place of signing", "Bhopal"),
            ],
            build=_build_rti_application,
        ),
        Template(
            doc_type="labour_complaint",
            label="Labour Complaint",
            description="A complaint to the Labour Commissioner / Assistant Labour Commissioner for unpaid wages and dues.",
            fields=[
                FieldSpec("worker_name", "Your full name", "Rahul Yadav"),
                FieldSpec("worker_address", "Your address", "15 Mill Lane, Kanpur 208001"),
                FieldSpec("employer_name", "Employer's name", "Zenith Textiles Ltd"),
                FieldSpec("employer_address", "Employer's address", "Industrial Area, Kanpur"),
                FieldSpec("months_due", "Months / amounts due", "March–May 2026, ₹45,000 per month"),
                FieldSpec("office_city", "Labour office city", "Kanpur"),
                FieldSpec("place", "Place of signing", "Kanpur"),
            ],
            build=_build_labour_complaint,
        ),
        Template(
            doc_type="cyber_complaint",
            label="Cyber Crime Complaint",
            description="A complaint for online fraud / cyber crime to the cyber police and the National Cyber Crime Reporting Portal.",
            fields=[
                FieldSpec("victim_name", "Your full name", "Neha Gupta"),
                FieldSpec("victim_address", "Your address", "9 Rose Villa, Jaipur 302001"),
                FieldSpec("police_station", "Police / cyber police station", "Cyber Police Station, Jaipur"),
                FieldSpec("city", "City / district", "Jaipur"),
                FieldSpec("platform", "Platform / app / number used by the fraudster", "Telegram @coin_pro; phone 98xxxxxx21"),
                FieldSpec("incident_date", "Date of the incident", "14 July 2026"),
                FieldSpec("transaction_refs", "Transaction references / amounts", "UPI ref 4123xxxx88 — ₹20,000; IMPS ref 9981xx — ₹35,000"),
                FieldSpec("place", "Place of signing", "Jaipur"),
            ],
            build=_build_cyber_complaint,
        ),
        Template(
            doc_type="writ_petition",
            label="Writ Petition (Draft)",
            description="A starting draft for a writ petition under Article 226 of the Constitution — MUST be settled by a qualified advocate before filing.",
            fields=[
                FieldSpec("petitioner_name", "Petitioner's name", "Mohammed Ali"),
                FieldSpec("petitioner_address", "Petitioner's address", "31 Gandhi Nagar, Srinagar 190001"),
                FieldSpec("respondent_authority", "Respondent authority", "State of J&K through its Chief Secretary"),
                FieldSpec("high_court", "High Court", "High Court of Jammu & Kashmir and Ladakh"),
            ],
            build=_build_writ_petition,
        ),
        Template(
            doc_type="representation",
            label="Representation to Authority",
            description="A general written representation to a government office, society, or any authority seeking a specific corrective action.",
            fields=[
                FieldSpec("applicant_name", "Your full name", "Devika Nair"),
                FieldSpec("applicant_address", "Your address", "17 Palm Grove, Kochi 682001"),
                FieldSpec("authority_name", "Authority / officer", "The Secretary, Housing Society"),
                FieldSpec("authority_address", "Authority's address", "Society Office, Kochi"),
                FieldSpec("subject", "Subject of the representation", "Leakage repair — flat 3B"),
                FieldSpec("place", "Place of signing", "Kochi"),
            ],
            build=_build_representation,
        ),
    ]
}

TEMPLATE_KEYS = list(TEMPLATES.keys())


def get_template(doc_type: str) -> Template | None:
    return TEMPLATES.get(doc_type)


def template_summaries() -> list[dict]:
    """Template metadata for the generator UI."""
    return [
        {
            "doc_type": t.doc_type,
            "label": t.label,
            "description": t.description,
            "fields": [
                {
                    "key": f.key,
                    "label": f.label,
                    "example": f.example,
                    "required": f.required,
                    "input_type": f.input_type,
                }
                for f in t.fields
            ],
        }
        for t in TEMPLATES.values()
    ]
