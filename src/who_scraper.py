from urllib.parse import urljoin
from typing import Dict, Optional, List
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from .base_scraper import BaseScraper
from .config import WHO_BASE_URL, WHO_FACT_SHEETS_URL, WHO_HEADERS


class WHOScraper(BaseScraper):
    def __init__(self):
        super().__init__("WHO")
        self.session.headers.update(WHO_HEADERS)

    def get_fact_sheet_links(self) -> List[str]:
        """Get all fact sheet links from the main WHO fact sheets page"""
        print("Fetching WHO fact sheet list...")
        html = self.get_page(WHO_FACT_SHEETS_URL)
        soup = BeautifulSoup(html, 'html.parser')

        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '/news-room/fact-sheets/detail/' in href:
                full_url = urljoin(WHO_BASE_URL, href)
                links.append(full_url)

        seen = set()
        return [x for x in links if not (x in seen or seen.add(x))]

    def parse_fact_sheet(self, url: str) -> Optional[Dict]:
        """Parse a single WHO fact sheet page"""
        print(f"Processing: {url}")
        try:
            html = self.get_page(url)
            soup = BeautifulSoup(html, 'html.parser')

            title = self._get_text(soup.find('h1'))
            date_element = soup.find('span', class_='date')
            date = self._get_text(date_element)

            content = (soup.find('article', class_='sf-detail-body-wrapper') or
                       soup.find('div', class_='sf-detail-body-wrapper'))

            if not content:
                print(f"Warning: Could not find main content for {url}")
                return None

            document = {
                "source": "WHO",
                "metadata": {
                    "title": title,
                    "url": url,
                    "last_updated": date
                },
                "content": {
                    "key_facts": "",
                    "sections": {}
                }
            }

            key_facts_div = content.find('div', class_='key-facts')
            if key_facts_div:
                document["content"]["key_facts"] = self._clean_text(key_facts_div.get_text('\n', strip=True))

            sections = {}
            current_section = None
            current_content = []

            for element in content.find_all(['h2', 'h3', 'div', 'p']):
                if element.name in ['h2', 'h3']:
                    if current_section:
                        sections[current_section] = self._clean_text('\n'.join(current_content))
                        current_content = []
                    current_section = self._clean_text(element.get_text(strip=True))
                elif current_section:
                    text = self._clean_text(element.get_text('\n', strip=True))
                    if text:
                        current_content.append(text)

            if current_section and current_content:
                sections[current_section] = self._clean_text('\n'.join(current_content))

            document["content"]["sections"] = sections
            return document

        except Exception as e:
            print(f"Error parsing {url}: {str(e)}")
            return None

    def scrape_all(self, limit: Optional[int] = None):
        """Run the complete scraping workflow"""
        fact_sheet_urls = self.get_fact_sheet_links()
        if limit:
            fact_sheet_urls = fact_sheet_urls[:limit]

        for url in tqdm(fact_sheet_urls, desc="Scraping WHO fact sheets"):
            if url in self.processed_urls:
                continue

            try:
                data = self.parse_fact_sheet(url)
                if data:
                    self.save_document(data, data["metadata"]["title"])
                    self.processed_urls.add(url)
            except Exception as e:
                print(f"Failed to scrape {url}: {str(e)}")
                self.failed_urls.add(url)
