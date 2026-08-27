"""Deterministic domain taxonomy.

This module powers the offline "demo mode" and the fallback path for every
agent when no LLM is reachable. It maps everyday problem descriptions to:
legal domains, retrieval queries, remedies, pathways, evidence lists and
suggested document types — every recommendation references source ids that
exist in the demo knowledge base, so even the deterministic mode cites the
knowledge base (never the LLM's memory).
"""
from __future__ import annotations

EMERGENCY_KEYWORDS = [
    "threat to my life", "threatening to kill", "kill me", "kill", "murder",
    "suicide", "self harm", "self-harm", "violence", "violent", "beating",
    "beats me", "beat me", "hit me", "hits me", "assault", "detained",
    "custody", "kidnap", "kidnapped", "abduct", "abduction", "rape",
    "molest", "molestation", "stalker", "stalking", "unsafe", "danger",
    "domestic violence", "blackmail", "sextortion", "abuse", "abusing me",
]

DOMAINS: dict[str, dict] = {
    "tenancy_deposit": {
        "label": "Tenancy — security deposit / rent dispute",
        "classification": "Civil dispute (contract / tenancy)",
        "keywords": [
            "landlord", "landlady", "deposit", "security deposit", "rent",
            "tenant", "rental", "flat", "apartment", "lease", "evict",
            "eviction", "moved out", "move out", "advance", "vacated",
            "rental agreement", "maintenance charges deducted",
        ],
        "queries": [
            "security deposit refund landlord tenant agreement",
            "rental agreement breach deposit recovery remedy",
            "legal notice landlord deposit refund",
            "civil suit recovery money tenancy",
        ],
        "source_ids": ["ica-1872", "sra-1963", "mta-2021", "civil-courts-note", "nalsa-15100"],
        "desired_outcome": "Recovery of the security deposit wrongfully withheld by the landlord",
        "remedies": [
            {
                "title": "Written demand / legal notice to the landlord",
                "description": "Send a dated written demand asking for refund of the deposit within a reasonable period, enclosing proof of payment and handover. Most deposit disputes are resolved at this stage.",
                "authority": "Sent by you (optionally through a lawyer)",
                "source_ids": ["ica-1872", "civil-courts-note"],
                "confidence": "medium",
            },
            {
                "title": "Direct negotiation or mediation",
                "description": "Propose a written settlement — part-refund, or a deduction schedule supported by bills — possibly through a neutral third party or pre-litigation mediation via the District Legal Services Authority.",
                "authority": "Party-led / DLSA mediation",
                "source_ids": ["lsa-1987"],
                "confidence": "medium",
            },
            {
                "title": "Civil suit for recovery of the deposit",
                "description": "If the notice fails, a suit for recovery of money based on the tenancy agreement lies before the civil court of competent pecuniary jurisdiction, with interest as compensation where proved.",
                "authority": "Civil court / court of small causes",
                "source_ids": ["sra-1963", "ica-1872", "civil-courts-note"],
                "confidence": "medium",
            },
            {
                "title": "Rent Authority under the applicable state rent law",
                "description": "Where your state has adopted a modern tenancy law or continues a rent-control Act, the Rent Authority/Controller may adjudicate deposit disputes — this depends on your state.",
                "authority": "State Rent Authority / Rent Controller",
                "source_ids": ["mta-2021"],
                "confidence": "low",
            },
        ],
        "pathway": [
            ["Gather your evidence", "Collect the rental agreement, payment proof (bank statement/receipts), and all correspondence with the landlord about the deposit.", ["ica-1872"]],
            ["Send a written demand / legal notice", "State the amount paid, the handover date and a clear demand for refund within a reasonable period (commonly 15–30 days).", ["ica-1872", "civil-courts-note"]],
            ["Record the response", "Keep the landlord's reply (or proof of no reply). Any refusal in writing strengthens your case.", ["civil-courts-note"]],
            ["Evaluate the forum", "Depending on your state: Rent Authority under the state rent law, or a civil suit for recovery; check legal-aid eligibility with DLSA (15100).", ["mta-2021", "sra-1963", "nalsa-15100"]],
            ["File if unresolved", "Proceed with the chosen forum, exhibiting your agreement, payment proof and correspondence.", ["sra-1963", "civil-courts-note"]],
        ],
        "evidence_required": [
            ["Rental / tenancy agreement copy", "Proves the deposit terms and refund conditions"],
            ["Payment proof of the deposit (receipts / bank statement)", "Proves the amount actually paid"],
            ["Communication with the landlord (emails, messages)", "Shows demands made and the landlord's refusal"],
            ["Move-out photos / videos and handover record", "Proves the condition of the premises at handover"],
            ["Any written deductions claimed by the landlord", "To test whether deductions are justified"],
        ],
        "risks": [
            "Verbal (undocumented) agreements make proof of deposit terms difficult",
            "Claims are subject to limitation periods — delay can defeat recovery",
            "State rent laws differ; the landlord may claim damages/repairs that need rebuttal",
        ],
        "uncertainties": [
            "Which rent law applies in your state (Model Tenancy Act adoption varies)",
            "Whether any lawful deductions are being claimed by the landlord",
        ],
        "doc_types": ["legal_notice", "representation"],
        "questions": [
            ["state", "Which state/city is the rented property in?", "Rent laws and forums differ by state"],
            ["agreement", "Was there a written rental agreement? Do you have a copy?", "The agreement proves the deposit terms"],
            ["deposit_amount", "How much was the deposit, and when was it paid?", "Establishes the claim amount"],
            ["handover", "Did you hand over possession (keys/photos/handover note)?", "Refund obligations usually start at handover"],
        ],
    },
    "unpaid_wages": {
        "label": "Employment — unpaid salary / dues",
        "classification": "Labour & employment dispute",
        "keywords": [
            "salary", "salaries", "wages", "unpaid", "not paid my salary",
            "employer", "hasn't paid", "has not paid", "salary delay",
            "full and final", "fnf", "termination", "fired", "laid off",
            "notice pay", "gratuity", "overtime", "my company", "manager",
            "increment", "reimbursement",
        ],
        "queries": [
            "unpaid salary recovery labour commissioner",
            "payment of wages act timely payment",
            "industrial disputes act section 33c recovery workman",
            "legal notice employer unpaid salary",
        ],
        "source_ids": ["pwa-1936", "ida-1947", "cow-2019", "labour-offices", "nalsa-15100"],
        "desired_outcome": "Recovery of unpaid salary and dues from the employer",
        "remedies": [
            {
                "title": "Written demand / legal notice to the employer",
                "description": "A dated demand listing the months and amounts due, enclosing salary slips and bank statements, is the standard first step and often triggers settlement.",
                "authority": "Sent by you (optionally through a lawyer)",
                "source_ids": ["ida-1947"],
                "confidence": "medium",
            },
            {
                "title": "Complaint to the Labour Commissioner / ALC",
                "description": "Approach the area's labour office for conciliation between you and the employer; officials can summon the employer and facilitate settlement of wage disputes.",
                "authority": "Labour Commissionerate / Assistant Labour Commissioner",
                "source_ids": ["labour-offices", "pwa-1936", "ida-1947"],
                "confidence": "medium",
            },
            {
                "title": "Recovery application (for workmen)",
                "description": "Where the employee is a 'workman', money due under the contract or a settlement can be recovered through the Industrial Disputes Act machinery, including Section 33C recovery applications.",
                "authority": "Appropriate government / labour court",
                "source_ids": ["ida-1947"],
                "confidence": "medium",
            },
            {
                "title": "Civil suit (for non-workman employees)",
                "description": "Managerial/supervisory staff outside the 'workman' definition may recover salary as contractual dues through a civil suit.",
                "authority": "Civil court",
                "source_ids": ["civil-courts-note", "ica-1872"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Assemble employment proof", "Collect the offer/appointment letter, salary slips, bank statements showing credits and gaps, and any correspondence on the delay.", ["pwa-1936", "labour-offices"]],
            ["Send a written demand", "Set out the months and amounts due and request payment within a reasonable period.", ["ida-1947"]],
            ["File a complaint with the labour office", "The Labour Commissioner/ALC can convene conciliation; bring your documents.", ["labour-offices"]],
            ["Escalate to recovery machinery", "Workmen may pursue recovery through the industrial-disputes machinery; others may file a civil suit. Check free legal aid eligibility (15100).", ["ida-1947", "civil-courts-note", "nalsa-15100"]],
        ],
        "evidence_required": [
            ["Offer / appointment letter", "Proves the employment relationship and salary terms"],
            ["Salary slips", "Proves the agreed monthly salary"],
            ["Bank statements showing non-payment", "Objective proof of the unpaid months"],
            ["Employment contract (if any)", "Shows notice, deduction and FnF terms"],
            ["Emails/messages about the delay", "Shows demands made and employer's responses"],
        ],
        "risks": [
            "Whether you fall within the 'workman' definition affects the forum",
            "Long delays can hit limitation periods for recovery",
            "Resignation vs termination nuances affect notice pay and FnF dues",
        ],
        "uncertainties": [
            "Your employment category (workman vs managerial) — determines the remedy",
            "Enforcement status of the Code on Wages, 2019 in your state",
        ],
        "doc_types": ["labour_complaint", "legal_notice"],
        "questions": [
            ["state", "Which state/city do you work in?", "Labour offices and rules differ by state"],
            ["employment_type", "What is your role — workman/staff, or managerial/supervisory?", "Determines the applicable remedy"],
            ["salary_amount", "What is your monthly salary and how many months are due?", "Establishes the claim"],
            ["contract", "Do you have a written contract/appointment letter?", "Proves the terms of employment"],
        ],
    },
    "consumer_defective": {
        "label": "Consumer — defective product / deficient service",
        "classification": "Consumer dispute",
        "keywords": [
            "defective", "product", "refund", "replacement", "warranty",
            "guarantee", "deficiency", "service", "online order", "e-commerce",
            "delivered damaged", "damaged product", "repair", "brand",
            "company refuses", "shop", "bill", "invoice", "seller",
            "faulty", "not working", "broken", "adulterated",
        ],
        "queries": [
            "consumer protection act 2019 deficiency service defective goods",
            "district consumer commission complaint jurisdiction",
            "consumer complaint e-daakhil filing procedure",
            "national consumer helpline grievance",
        ],
        "source_ids": ["cpa-2019", "cpa-jurisdiction-rules-2021", "cpa-commissions-rules-2020", "nch-1915", "lda-v-mkgupta-1994"],
        "desired_outcome": "Refund / replacement of the defective product or compensation for the deficient service",
        "remedies": [
            {
                "title": "Written complaint to the seller / brand",
                "description": "A dated written complaint with the invoice, photos/videos of the defect and a clear demand (refund/replacement/repair) — companies respond better to written trails.",
                "authority": "Seller / manufacturer / service provider",
                "source_ids": ["cpa-2019"],
                "confidence": "medium",
            },
            {
                "title": "National Consumer Helpline docket (1915)",
                "description": "Register the grievance on 1915 / consumerhelpline.gov.in — the docket creates a dated record and routes the complaint to the company.",
                "authority": "Department of Consumer Affairs",
                "source_ids": ["nch-1915"],
                "confidence": "medium",
            },
            {
                "title": "Consumer commission complaint",
                "description": "If unresolved, a complaint under the Consumer Protection Act, 2019 lies before the District Commission (claims up to ₹50 lakh), State Commission (₹50 lakh–₹2 crore) or National Commission (above ₹2 crore) — filable online via e-Daakhil. Complaints should ordinarily be filed within two years of the cause of action.",
                "authority": "District / State / National Consumer Commission",
                "source_ids": ["cpa-2019", "cpa-jurisdiction-rules-2021", "cpa-commissions-rules-2020"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Preserve the evidence", "Keep the invoice/bill, warranty card, packaging, and take photos/videos of the defect. Do not discard the product.", ["cpa-2019"]],
            ["Complain in writing to the seller/brand", "State the purchase details, the defect and your demand, with a reasonable deadline.", ["cpa-2019"]],
            ["Register a helpline docket", "Call 1915 or use consumerhelpline.gov.in; note the docket number.", ["nch-1915"]],
            ["File before the consumer commission", "If unresolved, prepare the complaint (facts, deficiency, relief) with annexures and file on e-Daakhil — within two years of the cause of action.", ["cpa-2019", "cpa-commissions-rules-2020", "cpa-jurisdiction-rules-2021"]],
            ["Attend hearings / seek settlement", "Commissions often facilitate settlement; compensation for harassment is also awardable.", ["lda-v-mkgupta-1994"]],
        ],
        "evidence_required": [
            ["Invoice / bill of purchase", "Proves the transaction and value"],
            ["Warranty / guarantee card", "Proves the applicable coverage"],
            ["Photos / video of the defect", "Documents the defect"],
            ["Complaint correspondence with the seller", "Shows the deficiency was brought to notice"],
            ["Delivery / order records (for online purchases)", "Links the seller to the transaction"],
        ],
        "risks": [
            "Two-year limitation for consumer complaints from the cause of action",
            "Forum depends on claim value; fee slabs apply",
            "Expert evidence may be needed to prove technical defects",
        ],
        "uncertainties": [
            "Whether the seller will contest or settle",
            "Exact valuation of claim (including compensation) for jurisdiction",
        ],
        "doc_types": ["legal_notice", "consumer_complaint"],
        "questions": [
            ["purchase_details", "When and for how much was the product/service purchased? Do you have the invoice?", "Proves the transaction and claim value"],
            ["warranty", "Is the product under warranty or the service under a contract?", "Affects the deficiency claim"],
            ["seller", "Who is the seller (online platform / shop / brand)?", "Identifies the opposite party"],
            ["prior_complaint", "Have you complained to the seller already? What was the response?", "Builds the deficiency trail"],
        ],
    },
    "police_fir_refusal": {
        "label": "Criminal procedure — police complaint / FIR refusal",
        "classification": "Criminal law / administrative action",
        "keywords": [
            "police", "fir", "refused to register", "didn't register",
            "did not register", "not registering", "complaint refused",
            "thana", "police station", "sho", "station house officer",
            "cognizable", "theft", "stolen", "assault", "cheating by",
            "no action taken", "police not helping",
        ],
        "queries": [
            "fir registration refusal remedy",
            "bnss section 173 fir cognizable offence",
            "superintendent of police escalation fir",
            "magistrate direction investigation",
            "lalita kumari fir mandatory registration",
        ],
        "source_ids": ["bnss-2023", "lalita-kumari-2014", "constitution-art-226", "nalsa-15100"],
        "desired_outcome": "Registration of the complaint/FIR and investigation of the offence",
        "remedies": [
            {
                "title": "Written complaint to the SHO (keep proof)",
                "description": "Submit a written, dated complaint narrating the facts chronologically; ask for a receipt/stamp copy. If registered, the FIR copy must be given free of cost.",
                "authority": "Station House Officer, local police station",
                "source_ids": ["bnss-2023"],
                "confidence": "medium",
            },
            {
                "title": "Escalation to the Superintendent of Police",
                "description": "If the SHO refuses, send the same complaint to the SP by registered post/official email; the SP may investigate or direct investigation.",
                "authority": "Superintendent of Police",
                "source_ids": ["bnss-2023"],
                "confidence": "medium",
            },
            {
                "title": "Application to the Magistrate",
                "description": "The concerned Magistrate may be moved to order investigation into a cognizable offence; a private complaint is also possible.",
                "authority": "Judicial Magistrate",
                "source_ids": ["bnss-2023", "lalita-kumari-2014"],
                "confidence": "medium",
            },
            {
                "title": "Writ petition in the High Court",
                "description": "Where registration is wrongfully refused despite escalation, a writ of mandamus under Article 226 can seek a direction to register and investigate.",
                "authority": "High Court",
                "source_ids": ["constitution-art-226", "lalita-kumari-2014"],
                "confidence": "professional_review",
            },
        ],
        "pathway": [
            ["Document the refusal", "Note the date, time, police station and officers. Keep a copy of your written complaint and any dispatch proof.", ["bnss-2023"]],
            ["Send a written complaint to the SHO", "Facts chronologically; request registration and investigation; ask for the stamped copy.", ["bnss-2023"]],
            ["Escalate to the SP", "If refused or ignored, address the same complaint to the Superintendent of Police with proof.", ["bnss-2023"]],
            ["Approach the Magistrate", "An application for direction to investigate, or a private complaint, can be filed — free legal aid is available (15100).", ["bnss-2023", "nalsa-15100"]],
            ["High Court in obstinate cases", "A writ under Article 226 seeking registration/investigation, supported by the refusal record.", ["constitution-art-226"]],
        ],
        "evidence_required": [
            ["Copy of the written complaint", "Proves what was reported and when"],
            ["Dispatch/receipt proof", "Proves submission to the police"],
            ["List of witnesses", "Supports investigation"],
            ["Medical report (if injured)", "Corroborates injuries"],
            ["Call records / diary of events", "Timeline corroboration"],
        ],
        "risks": [
            "Criminal proceedings are high-stakes — professional review is strongly advised",
            "Escalation steps and their sequencing matter; keep dated proof at each stage",
            "False or exaggerated allegations carry their own legal consequences",
        ],
        "uncertainties": [
            "Whether the facts disclose a cognizable offence",
            "Applicable provisions under the new criminal codes (BNS/BNSS) on the specific facts",
        ],
        "doc_types": ["police_complaint", "representation"],
        "questions": [
            ["station", "Which police station/city was the complaint made to?", "Identifies the escalation chain"],
            ["incident", "What happened, and on what date(s)?", "Drives the offence narrative"],
            ["written", "Did you submit a written complaint? Do you have a stamped copy?", "Proves the report"],
            ["witnesses", "Are there witnesses or supporting documents (medical/FIR drafts)?", "Strengthens the complaint"],
        ],
    },
    "cyber_fraud": {
        "label": "Cybercrime — online fraud / financial scam",
        "classification": "Cybercrime (criminal + banking remedies)",
        "keywords": [
            "online fraud", "fraud", "scam", "scammed", "upi", "phishing",
            "otp", "hacked", "account hacked", "unauthorised transaction",
            "unauthorized transaction", "card fraud", "loan app", "fake",
            "impersonation", "cyber", "instagram", "whatsapp scam",
            "investment app", "task scam", "binary", "crypto scam",
        ],
        "queries": [
            "online financial fraud remedy",
            "information technology act 66c 66d cheating personation",
            "national cyber crime reporting portal complaint",
            "bank unauthorised transaction chargeback",
        ],
        "source_ids": ["ita-2000", "bns-2023", "cybercrime-portal"],
        "desired_outcome": "Recovery of the defrauded amount and investigation of the offenders",
        "remedies": [
            {
                "title": "Immediate bank call + 1930 helpline",
                "description": "Call your bank's fraud line and the national cybercrime helpline 1930 immediately — quick reporting can trigger freezing of the remaining funds in the beneficiary account.",
                "authority": "Bank / national cybercrime helpline 1930",
                "source_ids": ["cybercrime-portal"],
                "confidence": "medium",
            },
            {
                "title": "Complaint on the National Cyber Crime Reporting Portal",
                "description": "File the complaint on cybercrime.gov.in with transaction references (UTR/RRN), account details, screenshots and device details; it is routed to the state cyber cell.",
                "authority": "cybercrime.gov.in / state cyber police",
                "source_ids": ["cybercrime-portal"],
                "confidence": "medium",
            },
            {
                "title": "Police / cyber cell complaint",
                "description": "A written complaint to the local police station or cyber cell citing the facts; the IT Act and penal provisions (cheating by personation, identity theft) are applied by the investigating agency.",
                "authority": "Local police / cyber cell",
                "source_ids": ["ita-2000", "bns-2023"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Act within minutes", "Call the bank and 1930 to report and seek a transaction hold/freeze.", ["cybercrime-portal"]],
            ["File on cybercrime.gov.in", "Upload transaction IDs, screenshots, chats and account details; save the complaint number.", ["cybercrime-portal"]],
            ["Lodge the written police complaint", "Submit a printed complaint with annexures at the local police station/cyber cell.", ["ita-2000", "bns-2023"]],
            ["Follow up with the bank", "Pursue the bank's grievance/chargeback process and escalate to the banking ombudsman / consumer commission if the bank's service was deficient.", ["ita-2000", "cpa-2019"]],
        ],
        "evidence_required": [
            ["Transaction details (UTR/RRN, amount, date-time)", "Traces the money trail"],
            ["Screenshots of chats / profiles / ads", "Identifies the fraudster's trail"],
            ["Bank statement showing the debit", "Proves the loss"],
            ["Complaint reference numbers", "Shows timely reporting"],
            ["Device details used", "Supports investigation"],
        ],
        "risks": [
            "Recovery chances fall sharply with delay — act immediately",
            "Money mule accounts complicate recovery",
            "Do not engage 'refund recovery agents' — secondary frauds are common",
        ],
        "uncertainties": [
            "Whether the defrauded funds remain traceable",
            "The investigating agency's ability to identify the account holders",
        ],
        "doc_types": ["cyber_complaint", "police_complaint"],
        "questions": [
            ["transaction", "What amount was lost, on what date, through which app/bank?", "Drives the freeze and complaint"],
            ["platform", "Which platform/app/site was used (UPI, wallet, trading app)?", "Identifies the modus operandi"],
            ["bank_informed", "Has the bank been informed? Any ticket/reference number?", "Shows prompt reporting"],
            ["evidence", "Do you have screenshots of chats and the transaction?", "Core evidence for investigation"],
        ],
    },
    "cheque_bounce": {
        "label": "Cheque dishonour (Section 138 NI Act)",
        "classification": "Quasi-criminal financial dispute",
        "keywords": [
            "cheque", "bounced", "dishonoured", "dishonored", "dishonor",
            "ni act", "return memo", "stop payment", "insufficient funds",
        ],
        "queries": [
            "cheque dishonour section 138 procedure",
            "demand notice dishonoured cheque",
            "complaint before magistrate cheque bounce",
        ],
        "source_ids": ["ni-1881", "civil-courts-note", "nalsa-15100"],
        "desired_outcome": "Payment of the cheque amount through the statutory Section 138 process",
        "remedies": [
            {
                "title": "Statutory demand notice",
                "description": "On receiving the bank's return memo, send a written demand notice to the drawer (ordinarily within 30 days of learning of dishonour) calling for payment within 15 days of receipt.",
                "authority": "Sent by you / through a lawyer",
                "source_ids": ["ni-1881"],
                "confidence": "medium",
            },
            {
                "title": "Criminal complaint before the Magistrate",
                "description": "If unpaid after the notice period, a complaint under Section 138 lies before the Magistrate (ordinarily within one month of the notice period expiring) — timelines are strict.",
                "authority": "Metropolitan / Judicial Magistrate",
                "source_ids": ["ni-1881"],
                "confidence": "medium",
            },
            {
                "title": "Civil recovery alternative",
                "description": "A parallel or alternative civil suit for recovery of the underlying debt remains available, independent of the cheque proceedings.",
                "authority": "Civil court",
                "source_ids": ["civil-courts-note"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Secure the instruments", "Keep the original dishonoured cheque and the bank's return memo safe — both are primary evidence.", ["ni-1881"]],
            ["Send the demand notice", "State the cheque details, dishonour reason and the 15-day demand; dispatch by registered post/courier and preserve proof.", ["ni-1881"]],
            ["Await the period", "If payment arrives, close the matter with a written receipt; if not, proceed promptly.", ["ni-1881"]],
            ["File the complaint", "File before the competent Magistrate within the limitation window, annexing the cheque, memo, notice and delivery proof.", ["ni-1881"]],
        ],
        "evidence_required": [
            ["Original dishonoured cheque", "Primary instrument of the claim"],
            ["Bank return memo", "Proves dishonour and the reason"],
            ["Proof of notice dispatch and delivery", "Proves the statutory demand"],
            ["Proof of the underlying debt/liability", "Shows the cheque was for a legally enforceable debt"],
        ],
        "risks": [
            "Strict statutory timelines — missing a step can defeat the complaint",
            "Address and service of notice must be accurate",
        ],
        "uncertainties": [
            "Current provisos and limitation refinements — verify before filing",
        ],
        "doc_types": ["legal_notice"],
        "questions": [
            ["cheque_details", "Cheque amount, number and date?", "Drives the notice contents"],
            ["memo_date", "When did the bank return the cheque (return memo date)?", "Fixes the notice timeline"],
            ["notice_sent", "Has any demand notice already been sent?", "Determines the current stage"],
            ["drawer_address", "What is the drawer's current address?", "Service of notice is critical"],
        ],
    },
    "rti_info": {
        "label": "Right to Information — access to records",
        "classification": "Administrative law",
        "keywords": [
            "rti", "right to information", "pio", "public information officer",
            "information officer", "government record", "file notings",
            "refused information", "no reply to rti",
        ],
        "queries": [
            "rti application section 6 format",
            "rti first appeal section 19 timeline",
            "rti fee rules 2012",
        ],
        "source_ids": ["rti-2005", "rti-fee-rules-2012"],
        "desired_outcome": "Obtaining the requested information from the public authority",
        "remedies": [
            {
                "title": "File/redo the RTI application",
                "description": "A precise application to the PIO with the ₹10 fee — no reasons needed. Responses are due in 30 days (48 hours where life or liberty is involved).",
                "authority": "Public Information Officer of the public authority",
                "source_ids": ["rti-2005", "rti-fee-rules-2012"],
                "confidence": "medium",
            },
            {
                "title": "First appeal",
                "description": "If the reply is deficient or absent, a first appeal lies to the first appellate authority of the same public authority.",
                "authority": "First Appellate Authority",
                "source_ids": ["rti-2005"],
                "confidence": "medium",
            },
            {
                "title": "Second appeal / complaint to the Information Commission",
                "description": "A second appeal to the CIC/SIC (or a complaint about refusal to accept the application) completes the ladder; the Commission can impose penalties on erring officers.",
                "authority": "Central / State Information Commission",
                "source_ids": ["rti-2005"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Draft the application precisely", "Identify the public authority and the exact records/information sought; multiple specific points work better than broad requests.", ["rti-2005"]],
            ["Submit with the fee", "₹10 by IPO/DD/online (rtionline.gov.in for central authorities); keep the receipt.", ["rti-fee-rules-2012"]],
            ["Track the 30-day clock", "No reply or an unsatisfactory reply triggers the first appeal (commonly within 30 days).", ["rti-2005"]],
            ["Appeal to the Commission", "File the second appeal to the CIC/SIC with the paper trail if still unresolved.", ["rti-2005"]],
        ],
        "evidence_required": [
            ["Copy of the RTI application", "Proves what was sought"],
            ["Proof of fee payment", "Proves a valid application"],
            ["Postal/online submission receipts", "Proves filing dates"],
            ["Any response received", "Frames the appeal"],
        ],
        "risks": [
            "Appeal windows are time-bound",
            "Vague requests invite rejections",
        ],
        "uncertainties": [
            "Whether the sought record is exempt under the Act",
        ],
        "doc_types": ["rti_application", "representation"],
        "questions": [
            ["authority", "Which public authority/department holds the information?", "Identifies the PIO"],
            ["information", "What specific information/records do you need?", "Frames the request points"],
            ["status", "Have you already applied? When? Any response?", "Determines the current stage"],
        ],
    },
    "domestic_violence": {
        "label": "Domestic violence / family safety",
        "classification": "Family law / protection remedies",
        "keywords": [
            "husband", "in-laws", "in laws", "dowry", "beats", "beaten",
            "violence", "abuse", "threatened", "domestic", "protection",
            "threw me out", "mental harassment", "marriage", "wife",
            "divorce", "maintenance",
        ],
        "queries": [
            "domestic violence act protection order magistrate",
            "protection officer complaint domestic violence",
            "women helpline domestic abuse remedy",
        ],
        "source_ids": ["pwda-2005", "constitution-art-21", "emergency-helplines", "nalsa-15100"],
        "desired_outcome": "Safety, protection orders and appropriate legal remedies",
        "remedies": [
            {
                "title": "Immediate police help (112 / 1091)",
                "description": "Where there is violence or a threat to safety, contact police on 112 and the women's helpline 1091 first; One Stop Centres provide integrated medical, police and legal assistance.",
                "authority": "Police / women helpline",
                "source_ids": ["emergency-helplines"],
                "confidence": "medium",
            },
            {
                "title": "Application to the Magistrate (DV Act)",
                "description": "A woman facing domestic abuse may apply for protection, residence, monetary and compensation orders; interim and ex parte orders are possible in urgent cases.",
                "authority": "Judicial Magistrate (through Protection Officer or directly)",
                "source_ids": ["pwda-2005"],
                "confidence": "professional_review",
            },
            {
                "title": "Complaint through the Protection Officer",
                "description": "The district Protection Officer can record the incident report and assist in placing the matter before the Magistrate with support services.",
                "authority": "District Protection Officer",
                "source_ids": ["pwda-2005"],
                "confidence": "medium",
            },
            {
                "title": "Free legal aid (DLSA / 15100)",
                "description": "Women are categorically eligible for free legal services — panel lawyers assist with protection applications and allied family matters.",
                "authority": "DLSA / NALSA",
                "source_ids": ["nalsa-15100", "lsa-1987"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Safety first", "If in danger, call 112 / 1091 immediately and move to a safe place; seek medical examination for injuries — the record becomes evidence.", ["emergency-helplines"]],
            ["Document the abuse", "Preserve messages, medical records, photographs and witness details without confronting the aggressor.", ["pwda-2005"]],
            ["Approach the Protection Officer / police", "An incident report or police complaint starts the formal trail.", ["pwda-2005"]],
            ["Apply to the Magistrate", "Through free legal aid (15100), seek protection/residence/monetary orders as needed.", ["pwda-2005", "nalsa-15100"]],
        ],
        "evidence_required": [
            ["Medical reports of injuries", "Corroborates physical abuse"],
            ["Photographs of injuries/damage", "Visual evidence"],
            ["Messages/call records showing threats", "Documented abuse trail"],
            ["Witness statements (neighbours/relatives)", "Corroboration"],
            ["Copies of any earlier complaints", "Pattern evidence"],
        ],
        "risks": [
            "Personal safety must take priority over documentation — professional review is strongly advised",
            "Family-law matters are sensitive; sequential legal steps matter",
        ],
        "uncertainties": [
            "Appropriate mix of civil protection and criminal remedies on your facts",
            "Long-term family-law outcomes (maintenance, custody) need case-specific advice",
        ],
        "doc_types": ["police_complaint", "representation"],
        "questions": [
            ["safety", "Are you safe right now? (If not, please call 112/1091 immediately)", "Safety comes before paperwork"],
            ["state", "Which state/city are you in?", "Protection Officers and courts are local"],
            ["living", "Are you living together or separated?", "Affects residence orders and safety planning"],
            ["prior_complaints", "Any earlier police complaints or medical visits?", "Pattern and evidence trail"],
        ],
        "emergency_default": True,
    },
    "workplace_harassment": {
        "label": "Workplace sexual harassment (POSH)",
        "classification": "Labour / workplace redressal",
        "keywords": [
            "sexual harassment", "workplace harassment", "posh",
            "harassed at work", "internal committee", "colleague",
            "boss", "manager made", "touched inappropriately", "ic complaint",
            "comments on my", "inappropriate messages at work",
        ],
        "queries": [
            "posh act internal committee complaint",
            "workplace sexual harassment remedy",
            "shebox complaint workplace",
        ],
        "source_ids": ["posh-2013", "emergency-helplines", "nalsa-15100"],
        "desired_outcome": "Redressal of workplace sexual harassment and a safe work environment",
        "remedies": [
            {
                "title": "Written complaint to the Internal Committee",
                "description": "A written complaint to the workplace Internal Committee — ordinarily within three months of the incident (extendable for good reason) — triggers a time-bound inquiry with interim reliefs possible.",
                "authority": "Internal Committee (IC) of the employer",
                "source_ids": ["posh-2013"],
                "confidence": "medium",
            },
            {
                "title": "Local Committee / SHe-Box escalation",
                "description": "Where the employer has no IC or where the complaint is against the employer, the Local Committee under the Act (and the SHe-Box portal) provide the alternative forum.",
                "authority": "Local Committee / SHe-Box",
                "source_ids": ["posh-2013"],
                "confidence": "medium",
            },
            {
                "title": "Police complaint for criminal conduct",
                "description": "Conduct amounting to criminal offences can be reported to the police (112 / 1091) independent of the IC process.",
                "authority": "Police",
                "source_ids": ["emergency-helplines"],
                "confidence": "medium",
            },
            {
                "title": "Free legal aid",
                "description": "Women are eligible for free legal services — DLSA panel lawyers can assist with complaints and representation.",
                "authority": "DLSA / NALSA",
                "source_ids": ["nalsa-15100"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Preserve evidence", "Save messages, emails, notes of incidents with dates, and identify witnesses.", ["posh-2013"]],
            ["File the written IC complaint", "Within the limitation window; request interim measures (transfer/leave) if needed.", ["posh-2013"]],
            ["Cooperate with the inquiry", "The IC records evidence and submits findings with recommendations the employer must act on.", ["posh-2013"]],
            ["Escalate as needed", "SHe-Box/Local Committee where the IC route fails, police for criminal conduct, and DLSA for legal aid.", ["posh-2013", "emergency-helplines", "nalsa-15100"]],
        ],
        "evidence_required": [
            ["Messages/emails showing the conduct", "Direct evidence"],
            ["Notes/diary of incidents with dates", "Timeline of events"],
            ["Witness accounts", "Corroboration"],
            ["Any earlier complaints to HR", "Shows notice to the employer"],
        ],
        "risks": [
            "Three-month limitation for IC complaints (extendable)",
            "Retaliation risks — interim reliefs are available",
        ],
        "uncertainties": [
            "Applicability details for your establishment type",
        ],
        "doc_types": ["representation", "police_complaint"],
        "questions": [
            ["organisation", "What is your organisation and does it have an Internal Committee?", "Determines the complaint route"],
            ["incidents", "When did the incident(s) occur?", "Fixes the limitation window"],
            ["witnesses", "Any witnesses or saved messages?", "Evidence strength"],
            ["prior_steps", "Have you reported to HR/management?", "Builds the trail"],
        ],
    },
    "motor_accident": {
        "label": "Motor accident / insurance claim",
        "classification": "Compensation claim (tort/statutory)",
        "keywords": [
            "accident", "car", "bike", "scooter", "hit by", "injured",
            "injury", "mact", "insurance claim", "claim rejected",
            "third party", "driver", "road accident", "vehicle",
        ],
        "queries": [
            "motor accident claim tribunal application",
            "motor vehicles act section 166 compensation",
            "insurance claim rejection grievance ombudsman",
        ],
        "source_ids": ["mv-1988", "sarla-verma-2009", "nalsa-15100"],
        "desired_outcome": "Compensation for the accident / settlement of the insurance claim",
        "remedies": [
            {
                "title": "FIR + medical records first",
                "description": "Ensure an FIR is registered and injuries are medically documented — both anchor the claim.",
                "authority": "Police / treating hospital",
                "source_ids": ["mv-1988"],
                "confidence": "medium",
            },
            {
                "title": "Claim petition before the MACT",
                "description": "Compensation applications for death or injury lie before the Motor Accident Claims Tribunal; no-fault compensation is available in death/permanent disablement cases irrespective of fault.",
                "authority": "Motor Accident Claims Tribunal",
                "source_ids": ["mv-1988"],
                "confidence": "professional_review",
            },
            {
                "title": "Insurer grievance / ombudsman route",
                "description": "For rejection or delay of insurance claims, the insurer's grievance channel followed by the insurance ombudsman (where eligible) provides a quicker remedy than litigation.",
                "authority": "Insurer grievance cell / insurance ombudsman",
                "source_ids": ["mv-1988"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Secure the basics", "FIR copy, medical records and bills, photographs, witness details, vehicle documents.", ["mv-1988"]],
            ["Notify the insurer", "Report the claim to the insurer promptly in writing.", ["mv-1988"]],
            ["Assess the claim", "Compensation methodology follows Supreme Court guidance (income, dependency, multipliers).", ["sarla-verma-2009"]],
            ["File before the MACT", "Through a lawyer (free legal aid eligibility via 15100), file the claim petition with annexures.", ["mv-1988", "nalsa-15100"]],
        ],
        "evidence_required": [
            ["FIR copy", "Records the accident"],
            ["Medical records and bills", "Proves injuries and expenses"],
            ["Vehicle documents and licence", "Compliance and ownership"],
            ["Photographs of the scene/damage", "Corroboration"],
            ["Witness details", "Supports fault findings"],
        ],
        "risks": [
            "Delay can raise limitation objections — file promptly",
            "Claim computation needs income proof",
        ],
        "uncertainties": [
            "Fault apportionment on the facts",
            "Current compensation benchmarks — verify latest position",
        ],
        "doc_types": ["legal_notice", "representation"],
        "questions": [
            ["accident_details", "When and where did the accident occur? Any FIR number?", "Anchors the claim"],
            ["injuries", "What injuries and treatment costs were involved?", "Quantum inputs"],
            ["insurance", "Which insurance is involved (your own / third party)? Claim status?", "Determines the route"],
        ],
    },
    "property_dispute": {
        "label": "Property / land dispute",
        "classification": "Civil / property dispute",
        "keywords": [
            "property", "land", "plot", "encroach", "encroachment",
            "boundary", "title", "sale deed", "mutation", "builder",
            "possession delayed", "flat not delivered", "registry",
            "partition", "ancestral",
        ],
        "queries": [
            "property title dispute civil court remedy",
            "specific performance agreement sale",
            "transfer of property act lease sale",
        ],
        "source_ids": ["tpa-1882", "sra-1963", "constitution-art-300a", "civil-courts-note"],
        "desired_outcome": "Protection or enforcement of property rights",
        "remedies": [
            {
                "title": "Legal notice asserting rights",
                "description": "A written notice setting out your title/possession and the specific violation (encroachment, non-delivery, refusal to execute documents), demanding rectification.",
                "authority": "Sent by you / through a lawyer",
                "source_ids": ["ica-1872", "civil-courts-note"],
                "confidence": "medium",
            },
            {
                "title": "Civil suit (declaration / injunction / specific performance)",
                "description": "Depending on the facts: a suit for declaration of title, injunction against dispossession/encroachment, or specific performance of an agreement to sell.",
                "authority": "Civil court",
                "source_ids": ["sra-1963", "tpa-1882"],
                "confidence": "professional_review",
            },
            {
                "title": "RTI / records route",
                "description": "Obtain mutation registers, survey records and municipal files through RTI to build the documentary base.",
                "authority": "Public Information Officer",
                "source_ids": ["rti-2005"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Assemble title documents", "Sale deed, tax receipts, mutation records, survey maps; use RTI where records are withheld.", ["tpa-1882", "rti-2005"]],
            ["Send a legal notice", "Assert your right and demand the specific corrective action.", ["civil-courts-note"]],
            ["Evaluate the suit", "Declaration/injunction/specific performance as advised — property matters are document-heavy and professional review is essential.", ["sra-1963"]],
            ["Consider interim protection", "Urgent injunctions can prevent dispossession or further encroachment.", ["sra-1963"]],
        ],
        "evidence_required": [
            ["Title documents (sale deed, will, partition deed)", "Foundation of the claim"],
            ["Tax receipts and mutation records", "Continuity of title"],
            ["Survey map / boundary records", "Locates the encroachment/extent"],
            ["Correspondence with the other party", "Shows notice and refusal"],
        ],
        "risks": [
            "Property litigation is document-intensive and long-drawn",
            "Limitation periods vary by relief sought",
            "Stamp-duty/registration defects can complicate title",
        ],
        "uncertainties": [
            "Exact relief mix appropriate to your facts",
            "Local land laws and revenue records practice",
        ],
        "doc_types": ["legal_notice", "representation"],
        "questions": [
            ["state", "Which state is the property in?", "Land laws are state-specific"],
            ["documents", "Which documents do you hold (sale deed, tax receipts, mutation)?", "Determines proof of title"],
            ["dispute", "What exactly is the dispute — encroachment, title, possession, documents?", "Shapes the relief"],
        ],
    },
    "general_contract": {
        "label": "General civil / contract dispute",
        "classification": "Contract dispute",
        "keywords": [
            "agreement", "contract", "breach", "service not provided",
            "paid but", "advance", "mou", "vendor", "vendor did not",
            "not delivered", "refuse to return money",
        ],
        "queries": [
            "contract breach remedy india",
            "legal notice breach of contract",
            "specific relief act remedies",
        ],
        "source_ids": ["ica-1872", "sra-1963", "civil-courts-note", "nalsa-15100"],
        "desired_outcome": "Enforcement of the agreement or recovery of losses",
        "remedies": [
            {
                "title": "Demand / legal notice",
                "description": "A written notice invoking the agreement's terms and demanding performance or refund within a reasonable period.",
                "authority": "Sent by you / through a lawyer",
                "source_ids": ["ica-1872"],
                "confidence": "medium",
            },
            {
                "title": "Negotiation / mediation",
                "description": "Structured settlement talks, including pre-litigation mediation through DLSA, often resolve commercial disagreements faster and cheaper.",
                "authority": "Party-led / DLSA mediation",
                "source_ids": ["lsa-1987"],
                "confidence": "medium",
            },
            {
                "title": "Civil suit for recovery / performance",
                "description": "Damages for breach, or where money is inadequate, specific performance; recovery suits follow the civil court's pecuniary jurisdiction.",
                "authority": "Civil court",
                "source_ids": ["ica-1872", "sra-1963", "civil-courts-note"],
                "confidence": "medium",
            },
        ],
        "pathway": [
            ["Collect the contract and payment proof", "The agreement, invoices, bank statements and correspondence form the core record.", ["ica-1872"]],
            ["Send a demand notice", "Set out the breach and the relief sought with a deadline.", ["ica-1872"]],
            ["Attempt settlement", "Mediation via DLSA where both sides are open to it.", ["lsa-1987"]],
            ["File if unresolved", "Recovery/performance suit in the competent civil court; check legal-aid eligibility.", ["civil-courts-note", "nalsa-15100"]],
        ],
        "evidence_required": [
            ["Written agreement / order confirmation", "Proves the terms"],
            ["Payment proof", "Proves performance on your side"],
            ["Correspondence", "Shows demands and refusals"],
            ["Proof of loss", "Quantifies damages"],
        ],
        "risks": [
            "Verbal agreements complicate proof",
            "Limitation applies to contract claims",
        ],
        "uncertainties": [
            "Whether the agreement is enforceable as drafted",
        ],
        "doc_types": ["legal_notice", "representation"],
        "questions": [
            ["agreement", "What was agreed — written or verbal? Do you hold any documents?", "Proves the contract"],
            ["amounts", "What amounts have you paid or lost?", "Quantifies the claim"],
            ["responses", "What has the other party said so far?", "Shows the dispute trail"],
        ],
    },
}


def match_domain(text: str) -> tuple[str, float]:
    """Score every domain by keyword hits; return (best_code, score)."""
    lowered = f" {text.lower()} "
    best_code, best_score = "general_contract", 0.0
    for code, domain in DOMAINS.items():
        score = 0.0
        for kw in domain["keywords"]:
            if kw in lowered:
                score += 1.0 if " " in kw else 0.6
        if score > best_score:
            best_code, best_score = code, score
    return best_code, best_score


def get_domain(code: str) -> dict:
    return DOMAINS.get(code, DOMAINS["general_contract"])


def domain_choices() -> list[str]:
    """Allowed domain codes for the LLM intake prompt."""
    return sorted(DOMAINS.keys())
