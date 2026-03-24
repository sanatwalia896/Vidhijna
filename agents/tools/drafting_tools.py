"""
tools/drafting_tools.py — Drafting helper utilities

Provides:
  - Standard clause libraries per draft type
  - Jurisdiction-specific requirements
  - Draft validation helpers
"""

# ── Standard clauses per draft type ───────────────────────────────────────────

STANDARD_CLAUSES = {
    "nda": [
        "Definition of Confidential Information",
        "Obligations of Receiving Party",
        "Exclusions from Confidential Information",
        "Term and Termination",
        "Return or Destruction of Information",
        "No License Granted",
        "Remedies",
        "Governing Law and Jurisdiction",
        "Entire Agreement",
    ],
    "service_agreement": [
        "Scope of Services",
        "Payment Terms",
        "Term and Termination",
        "Intellectual Property",
        "Confidentiality",
        "Limitation of Liability",
        "Indemnification",
        "Governing Law",
        "Dispute Resolution",
        "Force Majeure",
    ],
    "employment": [
        "Position and Duties",
        "Compensation and Benefits",
        "Working Hours",
        "Leave Policy",
        "Confidentiality",
        "Non-Compete (if applicable)",
        "Termination",
        "Governing Law",
    ],
    "legal_notice": [
        "Statement of Facts",
        "Legal Basis",
        "Demand/Relief Sought",
        "Deadline for Response",
        "Consequences of Non-Compliance",
        "Reservation of Rights",
    ],
    "arbitration_notice": [
        "Parties to Dispute",
        "Nature of Dispute",
        "Invocation of Arbitration Clause",
        "Seat and Venue",
        "Number of Arbitrators",
        "Relief Sought",
    ],
    "nclt_petition": [
        "Particulars of Petitioner",
        "Particulars of Respondent Company",
        "Statement of Facts",
        "Grounds for Petition",
        "Relief Sought",
        "List of Documents",
        "Verification",
    ],
    "consumer_complaint": [
        "Particulars of Complainant",
        "Particulars of Opposite Party",
        "Facts of the Case",
        "Deficiency in Service / Defective Goods",
        "Loss Suffered",
        "Relief Claimed",
        "Declaration",
    ],
}


def get_standard_clauses(draft_type: str) -> list[str]:
    """Return standard clauses for a given draft type."""
    return STANDARD_CLAUSES.get(draft_type, [])


def get_relevant_acts(draft_type: str) -> list[str]:
    """Return acts most relevant to each draft type."""
    ACT_MAP = {
        "nda":                ["Indian Contract Act, 1872"],
        "service_agreement":  ["Indian Contract Act, 1872", "Specific Relief Act, 1963"],
        "employment":         ["Indian Contract Act, 1872"],
        "sale_deed":          ["Transfer of Property Act, 1882", "Sale of Goods Act, 1930"],
        "lease":              ["Transfer of Property Act, 1882", "Indian Contract Act, 1872"],
        "legal_notice":       ["Indian Contract Act, 1872", "Limitation Act, 1963"],
        "cease_desist":       ["Trade Marks Act, 1999", "Copyright Act, 1957"],
        "reply_notice":       ["Indian Contract Act, 1872"],
        "nclt_petition":      ["Companies Act, 2013", "Insolvency and Bankruptcy Code, 2016"],
        "consumer_complaint": ["Consumer Protection Act, 2019"],
        "arbitration_notice": ["Arbitration and Conciliation Act, 1996"],
    }
    return ACT_MAP.get(draft_type, ["Indian Contract Act, 1872"])


def validate_draft_completeness(draft_text: str, draft_type: str) -> list[str]:
    """
    Check if a draft contains required standard clauses.
    Returns list of missing clause names.
    """
    required = get_standard_clauses(draft_type)
    draft_lower = draft_text.lower()
    missing = []
    for clause in required:
        # Simple keyword check — if clause keyword not found in draft
        keyword = clause.split()[0].lower()
        if keyword not in draft_lower:
            missing.append(clause)
    return missing