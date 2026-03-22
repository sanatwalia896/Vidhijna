"""
mark_done.py — Mark already-ingested files as done so they get skipped.
Run ONCE: python vector_store_creation/mark_done.py

Edit ALREADY_INGESTED to match exactly what's already in your Pinecone index.
"""

from pathlib import Path

CHECKPOINT_DIR = Path("logs/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# These are already in Pinecone — mark them as done
ALREADY_INGESTED = [
    "indian_contract_act_1872",
    "companies_act_2013",
    "companies_act_2013_v2",
    "arbitration_act_1996",
    "insolvency_bankruptcy_code",
    "competition_act_2002",
    "constitution_of_india",
    "fema_1999",
    "indian_partnership_act_1932",
    "negotiable_instruments_act",
    "sale_of_goods_act_1930",
    "specific_relief_act_1963",
    "transfer_of_property_act",
    "sebi_lodr_2015",
    "sebi_lodr_regulations_2015",
    "sebi_icdr_regulations_2018",
    "business_laws_study_notes",
    "laws_ethics_inter",
    "foundation_laws",
    "ibc_study_notes",
    "ibc_sc_judgment",
    "contract_law_explained",
    "common_law_explained",
]

for stem in ALREADY_INGESTED:
    cp = CHECKPOINT_DIR / f"{stem}.done"
    cp.touch()
    print(f"  marked done: {stem}")

print(f"\nDone. {len(ALREADY_INGESTED)} files marked as already ingested.")
print("Now run: python vector_store_creation/legal_ingest_v2.py")
print("It will skip these and only process the new files.")