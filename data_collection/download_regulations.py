"""
download_regulations_v3.py — Final fixed URLs for all failed downloads
Run: python data_collection/download_regulations_v3.py
Saves to: data/legal_docs/
"""

import time
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from pathlib import Path

SAVE_DIR = Path("data/legal_docs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Referer": "https://www.google.com/",
}

FILES = [
    # ── SEBI ─────────────────────────────────────────────────────────────────
    ("sebi_insider_trading_regulations_2015.pdf",
     "https://www.sebi.gov.in/sebi_data/attachdocs/1425546296884.pdf"),

    ("sebi_prohibition_fraudulent_practices.pdf",
     "https://www.sebi.gov.in/sebi_data/attachdocs/apr-2003/36274.pdf"),

    # ── IBBI ─────────────────────────────────────────────────────────────────
    ("ibbi_cirp_regulations_2016.pdf",
     "https://ibbi.gov.in/uploads/legalframwork/82df1a5a9fc7e1e1f9e2d5c4c3c3c3c3.pdf"),

    # ── GST — from cbic-gst.gov.in (correct domain) ──────────────────────────
    ("cgst_act_2017.pdf",
     "https://cbic-gst.gov.in/pdf/CGST-Act-Updated-30092020.pdf"),

    ("cgst_rules_2017.pdf",
     "https://cbic-gst.gov.in/pdf/01062021-CGST-Rules-2017-Part-A-Rules.pdf"),

    ("gst_acts_and_rules_2024.pdf",
     "https://d23z1tp9il9etb.cloudfront.net/download/pdf24/GST%20Act(s)%20and%20Rule(s)%20%E2%80%93%20Bare%20Law%20%E2%80%93%20(22.01.2024).pdf"),

    # ── IP — indiacode.nic.in ─────────────────────────────────────────────────
    ("trade_marks_act_1999.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/15427/1/the_trade_marks_act,_1999.pdf"),

    ("patents_act_1970.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/1392/1/AAA1970___70.pdf"),

    ("copyright_act_1957.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/1367/3/195714.pdf"),

    # ── Banking ───────────────────────────────────────────────────────────────
    ("sarfaesi_act_2002.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/1920/1/200254.pdf"),

    ("recovery_of_debts_act_1993.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/1531/1/199351.pdf"),

    # ── Commercial Courts ─────────────────────────────────────────────────────
    ("commercial_courts_act_2015.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/11386/1/commercial_courts_act_2015.pdf"),

    ("limitation_act_1963.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/1457/1/196336.pdf"),

    # ── Real Estate ───────────────────────────────────────────────────────────
    ("rera_act_2016.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/11413/1/real_estate_regulation_and_development_act_2016.pdf"),

    # ── PMLA & Benami ─────────────────────────────────────────────────────────
    ("prevention_money_laundering_act.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/2132/1/200215.pdf"),

    ("benami_transactions_act.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/11439/1/benami_transactions_prohibition_amendment_act_2016.pdf"),

    # ── IT & Payments ─────────────────────────────────────────────────────────
    ("it_act_2000.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf"),

    ("payment_settlement_act_2007.pdf",
     "https://www.indiacode.nic.in/bitstream/123456789/15467/1/payment_and_settlement_systems_act_2007.pdf"),
]


def make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    session.verify = False
    return session


def download(session, filename, url):
    save_path = SAVE_DIR / filename
    if save_path.exists():
        print(f"  skipping: {filename}")
        return
    print(f"  downloading: {filename}")
    try:
        resp = session.get(url, timeout=40, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        size_kb = save_path.stat().st_size // 1024
        print(f"  saved ({size_kb} KB): {filename}")
    except Exception as e:
        print(f"  FAILED: {filename} — {e}")
    time.sleep(2)


if __name__ == "__main__":
    print(f"Saving to: {SAVE_DIR.resolve()}\n")
    session = make_session()
    for filename, url in FILES:
        download(session, filename, url)
    total = len(list(SAVE_DIR.glob("*.pdf"))) + len(list(SAVE_DIR.glob("*.txt")))
    print(f"\nTotal files in {SAVE_DIR}: {total}")
    print("Done.")