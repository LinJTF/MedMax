from pathlib import Path

from src.who_scraper import WHOScraper
from src.pubmed_scraper import PubMedScraper
from src.cochrane_scraper import CochraneScraper

# LIMIT = 5

def run_all_scrapers():
    print("==> Running WHO Scraper")
    who_scraper = WHOScraper()
    who_scraper.scrape_all()

    print("\n==> Running PubMed Scraper")
    pubmed_scraper = PubMedScraper()
    pubmed_scraper.scrape_all()

    print("\n==> Running Cochrane Scraper")
    cochrane_scraper = CochraneScraper()
    cochrane_scraper.scrape_all()

if __name__ == "__main__":
    run_all_scrapers()