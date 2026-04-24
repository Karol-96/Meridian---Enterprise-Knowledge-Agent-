"""
src/ingestion/scraper.py
Pulls real pages from Apache Software Foundation's public Confluence.
No auth needed — these are fully public pages.
Run: python3 -m src.ingestion.scraper
"""

import asyncio
import httpx
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
BASE_URL   = "https://cwiki.apache.org/confluence/rest/api/content"
DATA_DIR   = Path("data/raw")
RATE_LIMIT = 0.5  # seconds between requests — be respectful

# These spaces give us rich engineering content:
# KAFKA  → KIPs (design proposals), architecture docs
# FLINK  → FLIPs (improvement proposals), dev guides  
# AIRFLOW → release notes, architecture, how-tos
SPACES = ["KAFKA", "FLINK", "AIRFLOW"]

# ── Scraper ──────────────────────────────────────────────────────────────────
async def fetch_pages(space_key: str, limit: int = 200) -> list[dict]:
    """Fetch all pages from one Apache Confluence space."""
    pages = []
    start = 0

    print(f"\n  Fetching {space_key} space...")

    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            try:
                response = await client.get(
                    BASE_URL,
                    params={
                        "spaceKey": space_key,
                        "expand": "body.storage,ancestors,metadata.labels,version,space",
                        "limit": 50,
                        "start": start,
                        "type": "page",
                        "status": "current",
                    }
                )
                response.raise_for_status()
                data = response.json()

            except httpx.HTTPError as e:
                print(f"    HTTP error at start={start}: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            pages.extend(results)
            print(f"    Fetched {len(pages)} pages so far...")

            # Stop if we have enough or no more pages
            if len(pages) >= limit or len(results) < 50:
                break

            start += len(results)
            await asyncio.sleep(RATE_LIMIT)

    return pages


def clean_page(page: dict) -> dict:
    """Extract only the fields we need — discard Confluence noise."""
    
    # Build ancestor path e.g. "Kafka Improvement Proposals > KIP-500"
    ancestors = page.get("ancestors", [])
    ancestor_path = " > ".join(a.get("title", "") for a in ancestors)

    # Get labels as a simple list
    labels_data = page.get("metadata", {}).get("labels", {}).get("results", [])
    labels = [l.get("name", "") for l in labels_data]

    # Get the raw HTML body (we strip HTML later in chunker)
    body = page.get("body", {}).get("storage", {}).get("value", "")

    # Get version info
    version = page.get("version", {})
    author  = version.get("by", {}).get("displayName", "unknown")
    modified = version.get("when", "")

    return {
        "id":            page.get("id"),
        "title":         page.get("title", ""),
        "space_key":     page.get("space", {}).get("key", ""),
        "space_name":    page.get("space", {}).get("name", ""),
        "ancestor_path": ancestor_path,
        "labels":        labels,
        "author":        author,
        "last_modified": modified,
        "url":           f"https://cwiki.apache.org/confluence/display/{page.get('space', {}).get('key', '')}/{page.get('title', '').replace(' ', '+')}",
        "body_html":     body,
        "scraped_at":    datetime.utcnow().isoformat(),
    }


async def scrape_all():
    """Scrape all configured spaces and save to data/raw/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = 0

    for space_key in SPACES:
        raw_pages = await fetch_pages(space_key, limit=150)
        cleaned   = [clean_page(p) for p in raw_pages]

        # Save each space as its own JSON file
        out_path = DATA_DIR / f"{space_key.lower()}_pages.json"
        with open(out_path, "w") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)

        total += len(cleaned)
        print(f"  Saved {len(cleaned)} pages → {out_path}")

    print(f"\n  Done! {total} pages total across {len(SPACES)} spaces.")
    print(f"  Files saved to: {DATA_DIR.resolve()}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Meridian Scraper — Apache Confluence")
    print("=" * 40)
    asyncio.run(scrape_all())