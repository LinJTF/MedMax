from pathlib import Path

# Base configuration
BASE_DATA_DIR = Path("data")
REQUEST_DELAY = 2.0
MAX_RETRIES = 3

# WHO configuration
WHO_BASE_URL = "https://www.who.int"
WHO_FACT_SHEETS_URL = f"{WHO_BASE_URL}/news-room/fact-sheets"
WHO_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Cochrane configuration
COCHRANE_PUBMED_SEARCH_URL = (
    "https://pubmed.ncbi.nlm.nih.gov/?term=%22Cochrane+Database+syst+rev%22%5BJournal%5D&filter=years.2018-2025"
)

COCHRANE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://www.google.com/"
}

# PubMed configuration (using HuggingFace dataset)