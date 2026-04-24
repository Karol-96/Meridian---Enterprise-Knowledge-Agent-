"""
src/ingestion/embedder.py

WHAT THIS FILE DOES:
Takes our 3,240 text chunks from JSON files and:
1. Sends each chunk's text to AWS Bedrock Titan Embed
2. Gets back 1024 numbers (the "embedding") that represent the meaning
3. Saves the text + metadata + 1024 numbers into PostgreSQL

After this runs, our chunks table has 3,240 rows each with a vector.
That vector is what makes search possible later.

Run: python3 -m src.ingestion.embedder
"""

import asyncio
import boto3
import json
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

# load_dotenv() reads our .env file and sets environment variables
# so os.getenv("AWS_ACCESS_KEY_ID") works below
load_dotenv()


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Where our chunk JSON files live (created by chunker.py)
CHUNK_DIR = Path("data/chunks")

# The Bedrock model that converts text → numbers
# Titan Embed v2 produces 1024-dimensional vectors
EMBED_MODEL = "amazon.titan-embed-text-v2:0"

# How many chunks we process in one loop iteration
# We don't process all 3,240 at once — that would use too much memory
# Instead we do 20 at a time, repeatedly until all done
BATCH_SIZE = 20

# Maximum number of Bedrock API calls running at the same time
# This is the "nightclub capacity" for our Semaphore
# Too high → AWS throttles us (rate limit error)
# Too low  → slow (not using available concurrency)
# 10 is a safe sweet spot for Bedrock's free tier
MAX_CONCURRENT = 10


# ── SEMAPHORE ─────────────────────────────────────────────────────────────────
# Semaphore = bouncer that allows max MAX_CONCURRENT coroutines at once
# Created ONCE at module level so all embed_text() calls share the same bouncer
# If 10 calls are running and an 11th arrives, it WAITS here until one finishes
semaphore = asyncio.Semaphore(MAX_CONCURRENT)


# ── AWS BEDROCK CLIENT ────────────────────────────────────────────────────────
# boto3.client() creates a connection to AWS Bedrock
# "bedrock-runtime" = the service that runs model inference (not bedrock management)
# We pass credentials from .env so AWS knows who we are
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


# ── POSTGRESQL CONNECTION ─────────────────────────────────────────────────────
def get_conn():
    """
    Creates and returns a new PostgreSQL connection.
    
    Why a function instead of one global connection?
    Because psycopg2 connections are NOT thread-safe.
    asyncio.to_thread() runs code in threads, so we need
    fresh connections rather than sharing one.
    """
    return psycopg2.connect(
        host="127.0.0.1",   # explicit IPv4 — avoids Mac local Postgres conflict
        port=5432,
        dbname="meridian",
        user="postgres",
        password="password"
    )


# ── RESUME SUPPORT ────────────────────────────────────────────────────────────
def get_existing_ids() -> set:
    """
    Returns a set of chunk_ids already in the database.
    
    WHY THIS EXISTS:
    Embedding 3,240 chunks takes ~5-10 minutes.
    If the script crashes at chunk 2,000, we don't want to
    re-embed the first 2,000 all over again — that wastes
    time and costs money on Bedrock.
    
    By checking what's already in the DB, we skip those
    and only embed the remaining ones.
    
    set() = like a list but checking "is X in it?" is O(1) instant
    vs a list where checking takes O(n) — matters at 3,240 items
    """
    conn = get_conn()
    cur  = conn.cursor()
    
    # Fetch all existing chunk_ids from the table
    cur.execute("SELECT chunk_id FROM chunks;")
    
    # Build a set from the results
    # cur.fetchall() returns: [("27821303_0",), ("27821303_1",), ...]
    # {row[0] for row in ...} = set comprehension → {"27821303_0", "27821303_1", ...}
    ids = {row[0] for row in cur.fetchall()}
    
    cur.close()
    conn.close()
    return ids


# ── EMBED ONE CHUNK ───────────────────────────────────────────────────────────
async def embed_text(text: str) -> list[float]:
    """
    Sends one chunk's text to Bedrock Titan Embed.
    Returns a list of 1024 floats (the embedding vector).
    
    KEY CONCEPTS USED HERE:
    
    1. async def — this is a coroutine, not a regular function
       It can be paused and resumed, allowing other coroutines
       to run while we wait for AWS to respond
    
    2. async with semaphore — acquires one "slot" from the bouncer
       If 10 slots are taken, this line PAUSES until one frees up
       When we exit the `with` block, the slot is automatically released
    
    3. await asyncio.to_thread(...) — THIS IS CRITICAL
       boto3 (the AWS SDK) is SYNCHRONOUS — it BLOCKS while waiting
       for the HTTP response from AWS. If we called it directly in
       async code, it would freeze the entire event loop and nothing
       else could run.
       
       asyncio.to_thread() solves this by:
       - Taking the blocking boto3 call
       - Running it in a separate thread (not the main event loop)
       - Returning control to the event loop while waiting
       - Resuming this coroutine when the thread finishes
       
       Think of it like: "go do this blocking thing over there,
       tell me when you're done, I'll do other work meanwhile"
    """
    async with semaphore:
        # asyncio.to_thread(func, *args) runs func(*args) in a thread
        # bedrock.invoke_model is the blocking boto3 call
        # We pass the model ID and the request body as JSON
        response = await asyncio.to_thread(
            bedrock.invoke_model,           # the blocking function
            modelId=EMBED_MODEL,            # which model to use
            body=json.dumps({
                "inputText": text[:8000],   # Titan max is 8,192 tokens — trim to be safe
                "dimensions": 1024,         # vector size (also supports 256, 512)
                "normalize": True           # normalize to unit length for cosine similarity
            })
        )
        
        # response["body"] is a streaming body object
        # .read() reads all bytes, json.loads() converts to Python dict
        result = json.loads(response["body"].read())
        
        # result["embedding"] is a list of 1024 floats like:
        # [0.023, -0.412, 0.887, 0.134, ...]
        return result["embedding"]


# ── SAVE ONE CHUNK TO POSTGRESQL ──────────────────────────────────────────────
def save_chunk(cur, chunk: dict, embedding: list[float]):
    """
    Inserts one chunk row into the PostgreSQL chunks table.
    
    cur = psycopg2 cursor (the object that executes SQL)
    chunk = the dict from our JSON file (text, metadata etc.)
    embedding = the 1024 floats we got from Bedrock
    
    ABOUT THE SQL:
    
    INSERT INTO chunks (...) VALUES (%s, %s, ...)
    %s = placeholder — psycopg2 replaces these with actual values safely
    Never use f-strings for SQL — that's a SQL injection vulnerability
    
    ON CONFLICT (chunk_id) DO NOTHING
    = "if a row with this chunk_id already exists, skip it silently"
    This is our safety net — if resume support fails for some reason,
    we won't get duplicate rows or errors, just quiet skips
    
    str(embedding)
    pgvector expects the vector as a string like "[0.23, -0.41, 0.88, ...]"
    Python's str() on a list gives exactly that format
    """
    cur.execute("""
        INSERT INTO chunks (
            chunk_id, text, token_count,
            page_id, page_title, space_key, space_name,
            ancestor_path, labels, author,
            last_modified, url, chunk_index, embedding
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s
        )
        ON CONFLICT (chunk_id) DO NOTHING;
    """, (
        chunk["chunk_id"],
        chunk["text"],
        chunk["token_count"],
        chunk["page_id"],
        chunk["page_title"],
        chunk["space_key"],
        chunk["space_name"],
        chunk["ancestor_path"],
        chunk["labels"],       # PostgreSQL TEXT[] array
        chunk["author"],
        chunk["last_modified"],
        chunk["url"],
        chunk["chunk_index"],
        str(embedding)         # pgvector string format
    ))


# ── PROCESS ONE BATCH ─────────────────────────────────────────────────────────
async def process_batch(batch: list[dict], conn) -> int:
    """
    Takes a batch of chunks (default 20), embeds them ALL concurrently,
    then saves them all to the database.
    
    WHY CONCURRENT?
    If we did this sequentially:
      for chunk in batch:
          embedding = await embed_text(chunk["text"])  # wait for AWS
    
    Each AWS call takes ~200ms. For 20 chunks = 4 seconds per batch.
    For 162 batches = 648 seconds = 10+ minutes.
    
    With concurrency (asyncio.gather):
    All 20 AWS calls fire at the same time (limited by Semaphore to 10 actual
    concurrent ones). Total time ≈ 400ms per batch instead of 4 seconds.
    That's 10x faster.
    
    ABOUT return_exceptions=True:
    Normally if one task in gather() fails with an exception,
    gather() cancels everything and raises the error immediately.
    
    With return_exceptions=True:
    If one embed call fails (e.g. AWS hiccups on one request),
    gather() doesn't crash — it returns the Exception object
    as the result for that item instead.
    
    We then check: if isinstance(embedding, Exception): skip it
    This way one bad chunk doesn't kill the entire batch.
    """

    # Create one embed_text() task per chunk in this batch
    # This does NOT start them yet — just creates the coroutine objects
    tasks = [embed_text(c["text"]) for c in batch]

    # asyncio.gather() starts ALL tasks concurrently and waits for all to finish
    # Returns a list of results in the same order as tasks
    # return_exceptions=True means errors come back as Exception objects, not crashes
    embeddings = await asyncio.gather(*tasks, return_exceptions=True)

    # Now save everything to the database
    cur   = conn.cursor()
    saved = 0

    # zip() pairs each chunk with its corresponding embedding result
    # batch[0] → embeddings[0], batch[1] → embeddings[1], etc.
    for chunk, embedding in zip(batch, embeddings):
        
        # Check if this embedding is an error (Exception object)
        # If AWS failed on this specific chunk, skip it and keep going
        if isinstance(embedding, Exception):
            print(f"    Skipping {chunk['chunk_id']}: {embedding}")
            continue

        save_chunk(cur, chunk, embedding)
        saved += 1

    # conn.commit() = "make these changes permanent in the database"
    # Without commit(), changes are in a transaction but not saved yet
    # If the script crashes before commit, nothing is saved — safe rollback
    conn.commit()
    cur.close()
    return saved


# ── MAIN ORCHESTRATOR ─────────────────────────────────────────────────────────
async def embed_all():
    """
    Main function — orchestrates the entire embedding pipeline.
    
    Flow:
    1. Load all chunks from JSON files into memory
    2. Check which ones are already in the DB (resume support)
    3. Split remaining into batches of BATCH_SIZE
    4. For each batch: embed concurrently → save to DB → report progress
    """
    print("Meridian Embedder")
    print("=" * 40)

    # STEP 1: Load all chunks from all three space JSON files
    # Path.glob("*_chunks.json") finds all files matching the pattern
    all_chunks = []
    for json_file in sorted(CHUNK_DIR.glob("*_chunks.json")):
        with open(json_file) as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)  # extend adds all items to the list
        print(f"  Loaded {len(chunks)} chunks from {json_file.name}")

    print(f"\n  Total: {len(all_chunks)} chunks to process")

    # STEP 2: Resume support — skip already embedded chunks
    existing = get_existing_ids()
    if existing:
        print(f"  Already embedded: {len(existing)} — skipping")

    # List comprehension to filter out already-done chunks
    # [c for c in all_chunks if c["chunk_id"] NOT in existing set]
    remaining = [c for c in all_chunks if c["chunk_id"] not in existing]
    print(f"  To embed now: {len(remaining)} chunks")

    if not remaining:
        print("\n  All chunks already embedded! Nothing to do.")
        return

    # STEP 3: Process in batches
    conn = get_conn()
    total_saved   = 0
    
    # Calculate total number of batches for progress display
    # ceiling division: (162 + 19) // 20 = 181 // 20 = 9 batches for 162 items
    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

    # range(0, total, BATCH_SIZE) gives: 0, 20, 40, 60, ...
    for i in range(0, len(remaining), BATCH_SIZE):
        
        # Slice the list to get this batch
        # remaining[0:20] = first 20, remaining[20:40] = next 20, etc.
        batch     = remaining[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        saved       = await process_batch(batch, conn)
        total_saved += saved

        print(f"  Batch {batch_num}/{total_batches} — "
              f"saved {saved}/{len(batch)} — "
              f"total so far: {total_saved}")

    conn.close()

    print(f"\n  Done! {total_saved} chunks embedded and stored.")
    print(f"  PostgreSQL chunks table now has {total_saved} rows.")
    print(f"  Each row has: text + metadata + 1024-dimensional vector")
    print(f"\n  Next step: build the retrieval API\n")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# asyncio.run() starts the event loop and runs embed_all() until complete
# This is always how you kick off an async program from a sync context
if __name__ == "__main__":
    asyncio.run(embed_all())