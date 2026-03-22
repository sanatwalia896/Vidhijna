# """
# download_books.py  —  Commercial law books & explanatory materials
# Run: python download_books.py
# Saves to: data/legal_books/
# """

# import time
# import requests
# from pathlib import Path

# SAVE_DIR = Path("data/legal_books")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "Accept": "application/pdf,application/octet-stream,*/*",
#     "Referer": "https://www.google.com/",
# }

# FILES = [
#     # ICMAI advanced study materials
#     ("commercial_law_advanced.pdf",
#         "https://icmai.in/upload/Students/Syllabus2016/Archive/Inter/Paper-6.pdf"),

#     ("mcq_business_law.pdf",
#         "https://www.icmai.in/upload/Students/mcq/Foundation/PAPER-3.pdf"),

#     # Law journal — IBC commentary
#     ("ibc_commentary_journal.pdf",
#         "https://www.lawjournals.org/assets/archives/2017/vol3issue6/2-6-68-137.pdf"),

#     # Public domain — classic legal reasoning books (Project Gutenberg)
#     ("elements_of_law.txt",
#         "https://www.gutenberg.org/cache/epub/22canons/pg22canons.txt"),

#     ("principles_of_contract.txt",
#         "https://www.gutenberg.org/cache/epub/46738/pg46738.txt"),

#     ("law_of_evidence.txt",
#         "https://www.gutenberg.org/cache/epub/22canons/pg22canons.txt"),

#     # Free Law Project — legal reasoning texts
#     ("introduction_to_law.txt",
#         "https://www.gutenberg.org/files/2527/2527-0.txt"),
# ]


# def download(filename, url):
#     save_path = SAVE_DIR / filename
#     if save_path.exists():
#         print(f"  skipping (exists): {filename}")
#         return
#     print(f"  downloading: {filename}")
#     try:
#         resp = requests.get(url, headers=HEADERS, timeout=40, stream=True)
#         resp.raise_for_status()
#         with open(save_path, "wb") as f:
#             for chunk in resp.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print(f"  saved ({save_path.stat().st_size // 1024} KB): {filename}")
#     except Exception as e:
#         print(f"  FAILED: {filename} — {e}")
#     time.sleep(2)


# if __name__ == "__main__":
#     print(f"Saving to: {SAVE_DIR.resolve()}\n")
#     for f, u in FILES:
#         download(f, u)
#     print(f"\nDone. Files in: {SAVE_DIR.resolve()}")

"""
download_reasoning_books.py — Downloads legal reasoning books
Run: python data_collection/download_reasoning_books.py
Saves to: data/legal_books/
"""

import time
import requests
from pathlib import Path

SAVE_DIR = Path("data/legal_books")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/octet-stream,*/*",
    "Referer": "https://www.google.com/",
}

FILES = [
    # Legal reasoning classics (Archive.org — public domain)
    ("introduction_to_legal_reasoning_levi.pdf",
     "https://archive.org/download/introductiontole01levi/introductiontole01levi.pdf"),

    ("handbook_of_indian_law.pdf",
     "https://archive.org/download/dli.ministry.02483/02483.pdf"),

    # Gutenberg public domain legal texts
    ("principles_of_contract_anson.txt",
     "https://www.gutenberg.org/files/46738/46738-0.txt"),

    ("elements_of_jurisprudence.txt",
     "https://www.gutenberg.org/files/22canons/22canons.txt"),
]


def download(filename: str, url: str):
    save_path = SAVE_DIR / filename
    if save_path.exists():
        print(f"  skipping (exists): {filename}")
        return
    print(f"  downloading: {filename}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
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
    for filename, url in FILES:
        download(filename, url)
    print(f"\nDone.")