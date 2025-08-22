"""Wikipedia data fetching and formatting to match PubMed record schema.

This module fetches Wikipedia pages (health related or provided titles) and
maps them into the record structure expected by the existing embedding and
ingestion pipeline:

    {
        'question': <str>,            # Pseudo-question derived from title
        'contexts': [<str>, ...],     # List of paragraph chunks
        'final_decision': <str>,      # Left empty / 'N/A' for Wikipedia
        'long_answer': <str>,         # Summary / first paragraph
    }

We purposefully keep the same keys so we can reuse format_pubmed_for_embedding
without changes.
"""
from __future__ import annotations

import json
import time
from typing import List, Dict, Any, Iterable
import requests

WIKIPEDIA_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"

def _fetch_page_raw(title: str, timeout: int = 15) -> dict | None:
    """Fetch a single Wikipedia page extract in plain text.

    Returns the JSON page dict or None if not found / error.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": title,
    }
    try:
        resp = requests.get(WIKIPEDIA_API_ENDPOINT, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():  # there should be only one
            if "missing" in page:
                return None
            return page
    except Exception as e:
        print(f"Failed to fetch Wikipedia title '{title}': {e}")
        return None

def _split_into_contexts(text: str, max_paragraphs: int = 6, max_len: int = 1200) -> List[str]:
    """Split raw page text into paragraph contexts with length capping."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    selected = []
    for p in paragraphs[:max_paragraphs]:
        if len(p) > max_len:
            p = p[:max_len] + "..."
        selected.append(p)
    return selected if selected else [text[:max_len] + ("..." if len(text) > max_len else "")]

def fetch_wikipedia_records(
    titles: Iterable[str],
    delay: float = 0.5,
    question_prefix: str = "Information about",
) -> List[Dict[str, Any]]:
    """Fetch multiple Wikipedia titles and convert to record schema.

    Args:
        titles: Iterable of page titles.
        delay: Optional delay between requests to be polite to API.
        question_prefix: Prefix to turn title into pseudo-question.
    """
    records: List[Dict[str, Any]] = []
    for title in titles:
        title_clean = title.strip()
        if not title_clean:
            continue
        page = _fetch_page_raw(title_clean)
        if not page:
            print(f"Skipping missing page: {title_clean}")
            continue
        extract = page.get("extract", "").strip()
        if not extract:
            print(f"No extract for page: {title_clean}")
            continue
        contexts = _split_into_contexts(extract)
        long_answer = contexts[0] if contexts else extract[:1200]
        record = {
            "question": f"{question_prefix} {title_clean}?",
            "contexts": contexts,
            "final_decision": "",  # Not applicable for Wikipedia
            "long_answer": long_answer,
            "source": "wikipedia",
        }
        records.append(record)
        print(f"Fetched Wikipedia page: {title_clean} -> {len(contexts)} contexts")
        time.sleep(delay)
    print(f"Collected {len(records)} Wikipedia records")
    return records

def load_wikipedia_titles_from_file(path: str) -> List[str]:
    """Load a list of Wikipedia titles from a text or JSON file.

    - .txt: one title per line
    - .json / .jsonl: attempt to read list or lines
    """
    try:
        if path.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            raise ValueError("JSON file must contain a list of titles")
        if path.lower().endswith(".jsonl"):
            titles: List[str] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, str):
                                titles.append(obj)
                            elif isinstance(obj, dict) and "title" in obj:
                                titles.append(str(obj["title"]))
                        except json.JSONDecodeError:
                            continue
            return titles
        raise ValueError("Unsupported file format for titles list")
    except Exception as e:
        print(f"Failed to load titles from {path}: {e}")
        return []
