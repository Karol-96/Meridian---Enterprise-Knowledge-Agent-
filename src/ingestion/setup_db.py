"""
src/ingestion/setup_db.py
Creates the PostgreSQL schema with pgvector extension.
Run once before embedding: python3 -m src.ingestion.setup_db
"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/meridian")

def setup():
    conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    dbname="meridian",
    user="postgres",
    password="password"
)
    cur  = conn.cursor()

    print("Setting up database...")

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Main chunks table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id     TEXT PRIMARY KEY,
            text         TEXT NOT NULL,
            token_count  INTEGER,
            page_id      TEXT,
            page_title   TEXT,
            space_key    TEXT,
            space_name   TEXT,
            ancestor_path TEXT,
            labels       TEXT[],
            author       TEXT,
            last_modified TEXT,
            url          TEXT,
            chunk_index  INTEGER,
            embedding    vector(1024)
        );
    """)

    # Index for fast cosine similarity search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)

    # Index for fast space filtering
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_space_key_idx
        ON chunks (space_key);
    """)

    # Full text search column for BM25
    cur.execute("""
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS fts tsvector
        GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
    """)

    # Index for full text search
    cur.execute("""
        CREATE INDEX IF NOT EXISTS chunks_fts_idx
        ON chunks USING gin(fts);
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("Done! Tables and indexes created.")
    print("  chunks table    ✓")
    print("  vector index    ✓")
    print("  space_key index ✓")
    print("  full text index ✓")
    print("\nReady for embedding.")

if __name__ == "__main__":
    setup()