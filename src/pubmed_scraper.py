from pathlib import Path
from typing import Dict, Optional
import json

from tqdm import tqdm
from datasets import load_dataset
from .base_scraper import BaseScraper


class PubMedScraper(BaseScraper):
    def __init__(self):
        super().__init__("PubMed-compact")

    def load_pubmedqa_dataset(self):
        print("Loading PubMedQA dataset from HuggingFace...")
        dataset = load_dataset("qiaojin/pubmedqa", "pqa_artificial")
        return dataset["train"]

    def transform_pubmed_entry(self, entry: dict) -> dict:
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

    def scrape_all(self, limit: int | None = None):
        dataset = self.load_pubmedqa_dataset()
        if limit:
            dataset = dataset.select(range(limit))

        output_file = self.data_dir / "pubmedqa.jsonl"
        with open(output_file, "a", encoding="utf-8") as f:
            for entry in tqdm(dataset, desc="Processing PubMed entries"):
                pubid = entry.get("pubid")
                if not pubid:
                    continue
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pubid}/"
                if url in self.processed_urls:
                    continue
                try:
                    transformed = self.transform_pubmed_entry(entry)
                    f.write(json.dumps(transformed, ensure_ascii=False) + "\n")
                    self.add_processed_url(url)
                except Exception as e:
                    print(f"Failed to process entry {pubid}: {str(e)}")
                    self.add_failed_url(url)
