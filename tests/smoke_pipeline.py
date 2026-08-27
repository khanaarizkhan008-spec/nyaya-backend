"""Smoke test: run the full 5-agent pipeline in demo mode (no LLM keys).

Usage: python -m tests.smoke_pipeline
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force demo mode for determinism
os.environ.setdefault("LLM_PROVIDER", "demo")
os.environ.setdefault("EMBEDDINGS_PROVIDER", "hash")
os.environ.setdefault("VECTOR_BACKEND", "numpy")


async def main() -> None:
    from app.agents.evidence import run_evidence
    from app.agents.document_gen import run_document
    from app.agents.intake import run_intake
    from app.agents.reasoning import run_reasoning
    from app.agents.research import run_research
    from app.agents.templates import DocContext, template_summaries
    from app.rag.ingest import ensure_indexed

    print("== indexing demo knowledge base ==")
    await ensure_indexed()
    print("   index OK")

    description = (
        "My landlord is refusing to return my security deposit of Rs 75,000 "
        "after I vacated the flat on 15 March 2026. I paid the deposit on "
        "1 June 2023 and there is a written rental agreement. He is claiming "
        "repairs but has shown no bills."
    )

    print("== AGENT 1: intake ==")
    structured, provider = await run_intake(description)
    print(f"   provider={provider} domain={structured.domain_code}")
    print(f"   summary={structured.problem_summary[:100]}")
    print(f"   questions={len(structured.questions)} amount={structured.amount_in_dispute}")
    assert structured.domain_code == "tenancy_deposit", structured.domain_code
    assert structured.amount_in_dispute is not None

    print("== AGENT 2: research ==")
    research = await run_research(structured)
    print(f"   sources={len(research.sources)} insufficient={research.insufficient}")
    for s in research.sources[:5]:
        print(f"   - {s.source_id}: {s.title[:70]}")
    assert len(research.sources) >= 3, "expected several retrieved sources"
    assert research.insufficient is False

    print("== AGENT 3: reasoning ==")
    reasoning, provider = await run_reasoning(structured, research)
    print(f"   provider={provider} confidence={reasoning.overall_confidence.value}")
    print(f"   remedies={len(reasoning.possible_remedies)} steps={len(reasoning.recommended_pathway)}")
    for r in reasoning.possible_remedies:
        print(f"   - {r.title[:70]} [{','.join(r.source_ids)}]")
    assert reasoning.possible_remedies, "expected remedies in demo mode"
    # every citation must be in the retrieved set
    retrieved_ids = {s.source_id for s in research.sources}
    for r in reasoning.possible_remedies:
        assert set(r.source_ids) <= retrieved_ids, f"unverified citation: {r.source_ids}"
    for step in reasoning.recommended_pathway:
        assert set(step.source_ids) <= retrieved_ids, f"unverified citation: {step.source_ids}"

    print("== AGENT 4: evidence ==")
    doc_text = (
        "RENTAL AGREEMENT\nThis rental agreement is made on 1 June 2023 between "
        "Ramesh Kumar (Landlord) and Priya Sharma (Tenant) for Flat 402, Green "
        "Residency, MG Road, Bengaluru 560001.\nThe tenant has paid a security "
        "deposit of Rs. 75,000 by bank transfer on 1 June 2023.\n"
        "Monthly rent is Rs 18,000 payable by the 5th of each month.\n"
        "Handover of the premises took place on 15 March 2026.\n"
        "Contact: landlord@example.com, 9876543210"
    )
    analysis, provider = await run_evidence("rental_agreement.txt", doc_text, structured)
    print(f"   provider={provider} guess={analysis.document_type_guess} class={analysis.classification}")
    print(f"   excerpts={len(analysis.key_excerpts)} names={analysis.entities.names}")
    print(f"   amounts={analysis.entities.amounts} timeline={len(analysis.timeline_events)}")
    assert analysis.document_type_guess == "rental agreement"
    assert analysis.classification == "strong_support"
    for ex in analysis.key_excerpts:
        assert ex in doc_text, f"non-verbatim excerpt: {ex!r}"

    print("== AGENT 5: document ==")
    summaries = template_summaries()
    print(f"   templates={len(summaries)}: {[t['doc_type'] for t in summaries]}")
    ctx = DocContext(
        structured=structured,
        research=research,
        reasoning=reasoning,
        evidence=[("rental_agreement.txt", analysis)],
        extra_fields={
            "sender_name": "Priya Sharma",
            "sender_address": "Flat 402, Green Residency, Bengaluru 560001",
            "recipient_name": "Ramesh Kumar",
            "recipient_address": "12 Park Street, Bengaluru 560034",
            "subject": "Refund of security deposit of Rs 75,000",
            "compliance_days": "15",
            "place": "Bengaluru",
        },
        source_description=description,
        answers={"state": "Karnataka", "agreement": "Yes, written, I have a copy"},
    )
    document, generator = await run_document("legal_notice", ctx)
    print(f"   generator={generator} sections={len(document.sections)}")
    print(f"   title={document.title}")
    for s in document.sections:
        print(f"   [{s.tag:<14}] {s.heading}")
    print(f"   verified_facts={len(document.verified_facts)} legal_sources={len(document.legal_sources)}")
    print(f"   validation: {document.validation_report.citations_verified}/{document.validation_report.citations_checked} citations verified")
    for w in document.validation_report.warnings:
        print(f"   warning: {w[:110]}")
    assert document.validation_report.citations_checked > 0
    assert document.validation_report.citations_verified == document.validation_report.citations_checked, \
        "all citations must verify against retrieved sources"

    # ---- anti-hallucination probe: unrelated case must yield insufficient reasoning
    print("== ANTI-HALLUCINATION PROBE ==")
    weird = "My neighbour's parrot repeats my private conversations and it is very distressing to me."
    structured2, _ = await run_intake(weird)
    research2 = await run_research(structured2)
    reasoning2, _ = await run_reasoning(structured2, research2)
    print(f"   domain={structured2.domain_code} sources={len(research2.sources)} remedies={len(reasoning2.possible_remedies)}")
    print(f"   note={reasoning2.note}")
    if research2.insufficient or len(research2.sources) < 2:
        assert not reasoning2.possible_remedies, "must not assert remedies without sources"
        assert reasoning2.overall_confidence.value == "low"

    print()
    print("ALL PIPELINE SMOKE TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
