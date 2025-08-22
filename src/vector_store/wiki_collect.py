"""Collect health-related Wikipedia page titles for controlled ingestion.

This avoids indexing the full Wikipedia (too large / noisy) by:
 1. Seeding from curated medical categories.
 2. Traversing category members with bounded depth and page count.
 3. Filtering: namespace=0 (content pages), min length, exclude disambiguation.
 4. Writing unique titles to data/wiki_health_titles.txt (one per line).

Usage examples:
  python -m src.vector_store.wiki_collect                          # defaults
  python -m src.vector_store.wiki_collect --max-pages 500 --depth 1
  python -m src.vector_store.wiki_collect --output data/custom.txt --include "Category:Oncology" --include "Category:Cardiology"

Then ingest:
  python -m src.vector_store.main populate_wiki --collection-name medmax_wiki_health --wikipedia-titles-file data/wiki_health_titles.txt

Or build combined collection:
  python -m src.vector_store.main populate_combined --wikipedia-titles-file data/wiki_health_titles.txt --combined-collection-name medmax_pubmed_wiki
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Set, List
import requests

API = "https://en.wikipedia.org/w/api.php"

DEFAULT_SEED_CATEGORIES = [
    "Category:Medicine",
    "Category:Health sciences",
    "Category:Human diseases",
    "Category:Medical treatments",
    "Category:Pharmacology",
    "Category:Anatomy",
    "Category:Physiology",
    "Category:Epidemiology",
    "Category:Public health",
    "Category:Symptoms and signs",
]

DISAMBIG_HINTS = ["may refer to", "disambiguation page"]

def api_get(params: dict, delay: float) -> dict:
    """Thin wrapper around requests.get with basic retry."""
    tries = 3
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(API, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == tries:
                print(f"API request failed after {tries} attempts: {e} ({params})")
                return {}
            time.sleep(1.0 * attempt)
    return {}

def fetch_category_members(category: str, limit: int, delay: float) -> List[str]:
    """Fetch content page titles (namespace 0) in a category (non-recursive)."""
    titles: List[str] = []
    cmcontinue = None
    while True and len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": 0,  # only articles
            "cmlimit": min(500, limit - len(titles)),
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_get(params, delay)
        cms = data.get("query", {}).get("categorymembers", [])
        for cm in cms:
            titles.append(cm["title"])
            if len(titles) >= limit:
                break
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue or len(titles) >= limit:
            break
        time.sleep(delay)
    return titles

def fetch_page_extract(title: str, delay: float) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exintro": 1,
        "titles": title,
        "format": "json",
    }
    data = api_get(params, delay)
    pages = data.get("query", {}).get("pages", {})
    for p in pages.values():
        return p.get("extract", "") or ""
    return ""

def looks_disambiguation(text: str) -> bool:
    low = text.lower()
    return any(hint in low for hint in DISAMBIG_HINTS)

def collect_titles(seeds: List[str], max_pages: int, depth: int, delay: float, min_chars: int) -> List[str]:
    collected: Set[str] = set()
    queue = [(s, 0) for s in seeds]
    seen_categories: Set[str] = set()
    while queue and len(collected) < max_pages:
        category, lvl = queue.pop(0)
        if category in seen_categories:
            continue
        seen_categories.add(category)
        print(f"Traversing {category} (level {lvl})")
        titles = fetch_category_members(category, limit=max_pages - len(collected), delay=delay)
        for t in titles:
            if len(collected) >= max_pages:
                break
            if t in collected:
                continue
            extract = fetch_page_extract(t, delay)
            if not extract or len(extract) < min_chars or looks_disambiguation(extract):
                continue
            collected.add(t)
            if len(collected) % 50 == 0:
                print(f"Collected {len(collected)} titles so far...")
        if lvl < depth:
            sub_params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": category,
                "cmnamespace": 14,
                "cmlimit": 200,
                "format": "json",
            }
            sub_data = api_get(sub_params, delay)
            subs = sub_data.get("query", {}).get("categorymembers", [])
            for sub in subs:
                sub_title = sub.get("title")
                if sub_title and sub_title not in seen_categories:
                    queue.append((sub_title, lvl + 1))
    print(f"Finished collection. Total titles: {len(collected)}")
    return sorted(collected)

def write_titles(titles: List[str], path: Path, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with open(path, mode, encoding="utf-8") as f:
        for t in titles:
            f.write(t + "\n")
    print(f"Wrote {len(titles)} titles to {path} (append={append})")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Collect health-related Wikipedia titles")
    p.add_argument("--max-pages", type=int, default=800, help="Maximum titles to collect")
    p.add_argument("--depth", type=int, default=1, help="Category traversal depth (0 = seeds only)")
    p.add_argument("--delay", type=float, default=0.4, help="Delay between API calls (seconds)")
    p.add_argument("--min-chars", type=int, default=400, help="Minimum intro extract chars to keep a page")
    p.add_argument("--output", type=str, default="data/wiki_health_titles.txt", help="Output file path")
    p.add_argument("--append", action="store_true", help="Append instead of overwrite output file")
    p.add_argument("--include", action="append", default=[], help="Additional seed category (can repeat)")
    p.add_argument("--exclude", action="append", default=[], help="Category seeds to remove")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    seeds = [c for c in DEFAULT_SEED_CATEGORIES if c not in set(args.exclude)] + args.include
    print("Seed categories:")
    for c in seeds:
        print(f"  - {c}")
    titles = collect_titles(seeds, max_pages=args.max_pages, depth=args.depth, delay=args.delay, min_chars=args.min_chars)
    write_titles(titles, Path(args.output), append=args.append)
    print("Next step: ingest with 'populate_wiki' or 'populate_combined' using --wikipedia-titles-file")
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
