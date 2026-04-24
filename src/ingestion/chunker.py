"""
src/ingestion/chunker.py
Converts raw HTML Confluence pages into clean, sized text chunks.
Each chunk keeps full metadata so we know exactly where it came from.
Run: python3 -m src.ingestion.chunker
"""

import json
import re
import tiktoken
from pathlib import Path
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR    = PROJECT_ROOT / "data" / "raw"
CHUNK_DIR  = PROJECT_ROOT / "data" / "chunks"
MAX_TOKENS = 512   # Titan Embed v2 sweet spot
MIN_TOKENS = 50    # skip tiny chunks — usually nav/header noise

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))


# ── HTML → clean text ─────────────────────────────────────────────────────────
def html_to_text(html: str) -> str:
    """Strip Confluence HTML storage format to plain text."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements
    for tag in soup(["script", "style", "nav", "footer",
                     "ac:structured-macro", "ac:parameter"]):
        tag.decompose()

    # Preserve heading structure with markdown markers
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        level = int(tag.name[1])
        tag.insert_before(f"\n{'#' * level} ")
        tag.append("\n")

    # Preserve list items
    for li in soup.find_all("li"):
        li.insert_before("\n- ")

    # Preserve code blocks
    for code in soup.find_all(["code", "pre"]):
        code.insert_before("`")
        code.append("`")

    text = soup.get_text(separator=" ")

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    return text


# ── Split by headers ──────────────────────────────────────────────────────────
def split_by_headers(text: str) -> list[str]:
    """Split text at h1/h2/h3 markdown headers into sections."""
    pattern = r"(?=\n#{1,3} )"
    sections = re.split(pattern, text)
    return [s.strip() for s in sections if s.strip()]


# ── Chunk one section ─────────────────────────────────────────────────────────
def chunk_section(section: str) -> list[str]:
    """
    If a section fits in MAX_TOKENS return it as-is.
    If it's too big, split by sentences until each piece fits.
    """
    if count_tokens(section) <= MAX_TOKENS:
        return [section]

    # Split by sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", section)
    chunks = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip()
        if count_tokens(candidate) <= MAX_TOKENS:
            current = candidate
        else:
            if current and count_tokens(current) >= MIN_TOKENS:
                chunks.append(current)
            current = sentence

    if current and count_tokens(current) >= MIN_TOKENS:
        chunks.append(current)

    return chunks


# ── Process one page ──────────────────────────────────────────────────────────
def chunk_page(page: dict) -> list[dict]:
    """Turn one raw page dict into a list of chunk dicts."""
    text = html_to_text(page.get("body_html", ""))

    if not text:
        return []

    sections = split_by_headers(text)
    chunks   = []

    for section in sections:
        for i, chunk_text in enumerate(chunk_section(section)):
            if count_tokens(chunk_text) < MIN_TOKENS:
                continue

            chunks.append({
                # Content
                "text":          chunk_text,
                "token_count":   count_tokens(chunk_text),

                # Source metadata — critical for citations
                "page_id":       page["id"],
                "page_title":    page["title"],
                "space_key":     page["space_key"],
                "space_name":    page["space_name"],
                "ancestor_path": page["ancestor_path"],
                "labels":        page["labels"],
                "author":        page["author"],
                "last_modified": page["last_modified"],
                "url":           page["url"],

                # Chunk position
                "chunk_index":   i,
                "chunk_id":      f"{page['id']}_{i}",
            })

    return chunks


# ── Process all spaces ────────────────────────────────────────────────────────
def chunk_all():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_pages  = 0

    for json_file in sorted(RAW_DIR.glob("*_pages.json")):
        space = json_file.stem.replace("_pages", "").upper()
        print(f"\n  Processing {space}...")

        with open(json_file) as f:
            pages = json.load(f)

        all_chunks = []
        for page in pages:
            page_chunks = chunk_page(page)
            all_chunks.extend(page_chunks)

        # Save
        out_path = CHUNK_DIR / f"{space.lower()}_chunks.json"
        with open(out_path, "w") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        total_pages  += len(pages)
        total_chunks += len(all_chunks)

        avg = len(all_chunks) // len(pages) if pages else 0
        print(f"  {len(pages)} pages → {len(all_chunks)} chunks (avg {avg}/page)")
        print(f"  Saved → {out_path}")

    print(f"\n  Done! {total_pages} pages → {total_chunks} total chunks")
    print(f"  Ready for embedding.\n")


if __name__ == "__main__":
    print("Meridian Chunker")
    print("=" * 40)
    chunk_all()