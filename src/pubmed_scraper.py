from pathlib import Path
from typing import Dict, Optional
import json

from tqdm import tqdm
from datasets import load_dataset
from .base_scraper import BaseScraper


class PubMedScraper(BaseScraper):
    def __init__(self):
        super().__init__("PubMed")

    def load_pubmedqa_dataset(self):
        """Load the PubMedQA dataset from HuggingFace"""
        print("Loading PubMedQA dataset from HuggingFace...")
        dataset = load_dataset("qiaojin/pubmedqa", "pqa_artificial")
        return dataset["train"]

    def transform_pubmed_entry(self, entry: Dict) -> Dict:
        """Transform a PubMedQA entry into our document format"""
        return {
            "source": "PubMed",
            "metadata": {
                "pubid": entry["pubid"],
                "question": entry["question"],
                "final_decision": entry["final_decision"]
            },
            "content": {
                "contexts": entry.get("context", {}).get("contexts", []),
                "labels": entry.get("context", {}).get("labels", []),
                "meshes": entry.get("context", {}).get("meshes", []),
                "long_answer": entry.get("long_answer", "")
            }
        }

    def scrape_all(self, limit: Optional[int] = None):
        """Run the PubMed scraping and saving workflow"""
        dataset = self.load_pubmedqa_dataset()
        if limit:
            dataset = dataset.select(range(limit))

        for entry in tqdm(dataset, desc="Processing PubMed entries"):
            try:
                transformed = self.transform_pubmed_entry(entry)
                self.save_document(transformed, f"pubmed_{entry['pubid']}")
            except Exception as e:
                print(f"Failed to process entry {entry['pubid']}: {str(e)}")
