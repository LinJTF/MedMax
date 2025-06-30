import json
import random
import time
from pathlib import Path
from typing import Dict, Optional, List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

class BaseScraper:
    def __init__(self, source_name: str, base_data_dir: Path = Path("data")):
        self.session = requests.Session()
        self.source_name = source_name
        self.data_dir = base_data_dir / source_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_urls = set()
        self.failed_urls = set()
        self.load_progress()

    def get_page(self, url: str, retries: int = 3, delay: float = 2.0) -> str:
        """Fetch a page with retries and delay."""
        for attempt in range(retries):
            try:
                time.sleep(delay + random.uniform(0, 1))
                response = self.session.get(url)
                response.raise_for_status()
                return response.text
            except requests.HTTPError as e:
                print(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == retries - 1:
                    raise
                time.sleep(5 * (attempt + 1))

    def save_document(self, data: Dict, filename: str):
        """Save document data to JSON file"""
        safe_name = "".join(c if c.isalnum() else "_" for c in filename)
        output_file = self.data_dir / f"{safe_name}.json"

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save {filename}: {str(e)}")

    def is_already_processed(self, title: str) -> bool:
        safe_name = "".join(c if c.isalnum() else "_" for c in title)
        output_file = self.data_dir / f"{safe_name}.json"
        return output_file.exists()

    def load_progress(self):
        processed_file = self.data_dir / "processed_urls.txt"
        failed_file = self.data_dir / "failed_urls.txt"

        if processed_file.exists():
            with open(processed_file, "r") as f:
                self.processed_urls = set(line.strip() for line in f if line.strip())

        if failed_file.exists():
            with open(failed_file, "r") as f:
                self.failed_urls = set(line.strip() for line in f if line.strip())

    def save_progress(self):
        with open(self.data_dir / "processed_urls.txt", "w") as f:
            f.write("\n".join(self.processed_urls))
        with open(self.data_dir / "failed_urls.txt", "w") as f:
            f.write("\n".join(self.failed_urls))

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        return ' '.join(text.split())

    def _get_text(self, element: Optional[Tag]) -> str:
        """Safely get text from a BeautifulSoup element"""
        return element.get_text(strip=True) if element else ""
