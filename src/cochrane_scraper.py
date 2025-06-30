import json
import time
import random
from pathlib import Path
from typing import Dict, Optional, List
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from .base_scraper import BaseScraper
from .config import COCHRANE_HEADERS, COCHRANE_PUBMED_SEARCH_URL

class CochraneScraper(BaseScraper):
    def __init__(self):
        super().__init__('Cochrane')
        self.session.headers.update(COCHRANE_HEADERS)

    def get_review_links_from_pubmed(self, page: int = 1) -> List[str]:
        url = f"{COCHRANE_PUBMED_SEARCH_URL}&page={page}"
        html = self.get_page(url)
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for link in soup.select("a.docsum-title[href]"):
            href = link.get("href")
            if href:
                links.append(urljoin("https://pubmed.ncbi.nlm.nih.gov", href))

        return links

    def split_abstract_sections(self, abstract_html: Tag) -> Dict[str, str]:
        sections = {}
        current_section = None
        content_lines = []

        for p in abstract_html.find_all("p"):
            strong = p.find("strong", class_="sub-title")
            if strong:
                if current_section and content_lines:
                    sections[current_section] = "\n".join(content_lines).strip()
                    content_lines = []
                current_section = self._get_text(strong).rstrip(":")
                remaining = p.get_text().replace(strong.get_text(), "").strip()
                if remaining:
                    content_lines.append(remaining)
            else:
                text = self._get_text(p)
                if text:
                    content_lines.append(text)

        if current_section and content_lines:
            sections[current_section] = "\n".join(content_lines).strip()

        if not sections:
            all_text = self._clean_text(abstract_html.get_text("\n", strip=True))
            sections = {"RawText": all_text}

        return sections

    def parse_review(self, url: str) -> Optional[Dict]:
        print(f"Processing: {url}")
        try:
            html = self.get_page(url)
            soup = BeautifulSoup(html, "html.parser")

            title = self._get_text(soup.find("h1"))
            date_element = soup.find("span", class_="cit")
            date = self._get_text(date_element)

            abstract_div = soup.find("div", class_="abstract")
            sections = self.split_abstract_sections(abstract_div) if abstract_div else {}

            return {
                "source": "Cochrane",
                "metadata": {
                    "title": title,
                    "url": url,
                    "last_updated": date
                },
                "content": {
                    "sections": {
                        "Abstract": sections
                    }
                }
            }
        except Exception as e:
            print(f"Error parsing {url}: {str(e)}")
            return None

    def scrape_all(self, max_pages: int | None = None):
        total_downloaded = 0
        page = 1

        while True:
            if max_pages is not None and page > max_pages:
                break
            try:
                review_links = self.get_review_links_from_pubmed(page)
                if not review_links:
                    break
            except Exception as e:
                print(f"Failed to get review links from page {page}: {e}")
                break

            for review_url in review_links:
                if review_url in self.processed_urls:
                    continue
                try:
                    data = self.parse_review(review_url)
                    if data:
                        self.save_document(data, data["metadata"]["title"])
                        self.processed_urls.add(review_url)
                        total_downloaded += 1
                except Exception as e:
                    print(f"Failed to process review {review_url}: {str(e)}")
                    self.failed_urls.add(review_url)

            page += 1

        self.save_progress()
