"""
download_legal.py  —  Indian commercial law PDFs from working sources
Run: python download_legal.py
Saves to: data/legal_docs/
"""

import time
import requests
from pathlib import Path

SAVE_DIR = Path("data/legal_docs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Referer": "https://www.google.com/",
}

FILES = [
    # Acts
    ("companies_act_2013.pdf",           "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf"),
    ("insolvency_bankruptcy_code.pdf",   "https://prsindia.org/files/bills_acts/acts_parliament/2016/the-insolvency-and-bankruptcy-code-act,-2016.pdf"),
    ("ibc_2016_mca.pdf",                 "https://www.mca.gov.in/Ministry/pdf/TheInsolvencyandBankruptcyofIndia.pdf"),
    ("sebi_lodr_2015.pdf",               "https://www.sebi.gov.in/sebi_data/attachdocs/1441284401427.pdf"),
    ("constitution_of_india.pdf",        "https://www.mca.gov.in/Ministry/pdf/Constitution_of_India.pdf"),
    # ICMAI study PDFs (bare acts + explanations combined)
    ("business_laws_study_notes.pdf",    "https://icmai.in/upload/Students/Syllabus2022/Inter_Stdy_Mtrl/P5_160824.pdf"),
    ("laws_ethics_inter.pdf",            "https://icmai.in/upload/Students/Syllabus2016/Inter/Paper-6-Sep-2021.pdf"),
    ("foundation_laws.pdf",              "https://icmai.in/upload/Students/Syllabus2016/Foundation/Paper-3-16092021.pdf"),
    ("ibc_study_notes.pdf",              "https://icmai.in/upload/Students/Supplementary/IBC-2016.pdf"),
    # IBBI judgment
    ("ibc_sc_judgment.pdf",              "https://ibbi.gov.in/webadmin/pdf/order/2018/Aug/11958_2018_Judgement_14-Aug-2018_2018-08-14%2022:04:34.pdf"),
    # Public domain
    ("contract_law_explained.txt",       "https://www.gutenberg.org/cache/epub/46738/pg46738.txt"),
    ("common_law_explained.txt",         "https://www.gutenberg.org/cache/epub/2527/pg2527.txt"),
    # Replace the failing constitution and ibc_2016_mca entries with these:
    ("constitution_of_india.pdf",
    "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf"),

    ("ibc_2016_mca.pdf","https://ibbi.gov.in/uploads/whatsnew/ccedf8ad5c8d6aaed49af5d3ae71c9da.pdf"),
    
]


def download(filename, url):
    save_path = SAVE_DIR / filename
    if save_path.exists():
        print(f"  skipping (exists): {filename}")
        return
    print(f"  downloading: {filename}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=40, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  saved ({save_path.stat().st_size // 1024} KB): {filename}")
    except Exception as e:
        print(f"  FAILED: {filename} — {e}")
    time.sleep(2)


if __name__ == "__main__":
    print(f"Saving to: {SAVE_DIR.resolve()}\n")
    for f, u in FILES:
        download(f, u)
    print(f"\nDone. Files in: {SAVE_DIR.resolve()}")