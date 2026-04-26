# Meridian - Enterprise Knowledge Agent

A production-grade RAG system that answers technical questions from Apache Confluence documentation using hybrid retrieval (dense + sparse search), Reciprocal Rank Fusion, and LLM generation with source citations.

Built for Apache **Kafka**, **Flink**, and **Airflow** docs. Designed to give engineers fast, accurate, cited answers from their actual documentation — not hallucinated guesses.

---

## Architecture

```
                              USER QUESTION
                                   |
                                   v
                    +------------------------------+
                    |        RETRIEVAL LAYER        |
                    |------------------------------|
                    |                              |
                    |   +--------+   +---------+   |
                    |   | DENSE  |   | SPARSE  |   |
                    |   | SEARCH |   | SEARCH  |   |
                    |   +--------+   +---------+   |
                    |       |             |         |
                    |   Embed query   BM25/FTS      |
                    |   via AWS      via PostgreSQL  |
                    |   Bedrock      ts_rank +       |
                    |   Titan v2     plainto_tsquery  |
                    |       |             |         |
                    |   Top 50        Top 50        |
                    |   by cosine     by keyword    |
                    |   similarity    relevance     |
                    |       |             |         |
                    |       +------+------+         |
                    |              |                |
                    |              v                |
                    |    +-------------------+      |
                    |    | RECIPROCAL RANK   |      |
                    |    | FUSION (RRF)      |      |
                    |    |                   |      |
                    |    | score = SUM of    |      |
                    |    | 1/(rank + 60)     |      |
                    |    | across both lists |      |
                    |    +-------------------+      |
                    |              |                |
                    |         Top 5 chunks          |
                    +------------------------------+
                                   |
                                   v
                    +------------------------------+
                    |       GENERATION LAYER        |
                    |------------------------------|
                    |                              |
                    |  System Prompt (Meridian)     |
                    |  + formatted context chunks   |
                    |  + user question              |
                    |          |                    |
                    |          v                    |
                    |     OpenAI GPT-4o             |
                    |     (temp=0.1)                |
                    |          |                    |
                    |          v                    |
                    |  Answer with [Source N]        |
                    |  inline citations             |
                    +------------------------------+
                                   |
                                   v
                           CITED ANSWER + SOURCES
```

### Full Data Pipeline

```
INGESTION                    RETRIEVAL + GENERATION              EVALUATION
=========                    =======================              ==========

Apache Confluence            User Query                          5 ground-truth
(Kafka, Flink, Airflow)         |                               Q&A pairs
     |                          |                                    |
     v                          v                                    v
 scraper.py              retriever.py                          ragas_eval.py
 Fetch pages via            |         |                             |
 REST API                   |         |                        Run live RAG
     |                      v         v                        on each question
     v                  dense     sparse                            |
 chunker.py             search    search                            v
 HTML -> text              |         |                         RAGAS scores:
 Split by headers          +---+-----+                         - Faithfulness
 512 token chunks              |                               - Relevance
     |                         v                               - Ctx Precision
     v                    RRF Fusion                           - Ctx Recall
 embedder.py                   |                                    |
 AWS Bedrock Titan             v                                    v
 1024-dim vectors         generator.py                         ragas_results.csv
     |                    GPT-4o with
     v                    context-only prompt
 PostgreSQL                    |
 (pgvector)                    v
 + IVFFlat index          Cited answer
 + GIN FTS index          with source URLs
```

---

## What Each Module Does

### Ingestion (`src/ingestion/`)

| Module | What it does |
|--------|-------------|
| `scraper.py` | Fetches pages from Apache's public Confluence API. Handles pagination, rate limiting (0.5s between requests), and metadata extraction (author, labels, ancestor paths). Outputs raw JSON to `data/raw/`. |
| `chunker.py` | Strips Confluence HTML, splits by headers, then chunks to 512 tokens max (50 min). Uses tiktoken for accurate token counting. Preserves all source metadata through the pipeline. Outputs to `data/chunks/`. |
| `setup_db.py` | Creates PostgreSQL schema with pgvector extension. Builds 3 indexes: IVFFlat for vector search, GIN for full-text search, B-tree for space_key filtering. |
| `embedder.py` | Embeds all chunks using AWS Bedrock Titan Embed v2 (1024 dimensions). Runs 10 concurrent API calls with asyncio semaphore. Supports resume — skips already-embedded chunks on restart. |

### Retrieval (`src/retrieval/`)

| Module | What it does |
|--------|-------------|
| `retriever.py` | Hybrid search: runs dense (cosine similarity via pgvector) and sparse (PostgreSQL full-text search with ts_rank) in parallel, then fuses results with RRF. Returns top 5 chunks with scores. Supports optional space_key filtering. |
| `generator.py` | Formats retrieved chunks as numbered sources, sends to GPT-4o with a strict context-only system prompt. Supports both full and streaming generation. The `rag()` function orchestrates the full retrieve-then-generate pipeline. |

### Evaluation (`src/evaluation/`)

| Module | What it does |
|--------|-------------|
| `ragas_eval.py` | Benchmarks the live RAG pipeline using RAGAS. Runs 5 eval questions through actual retrieval + generation, then scores with GPT-4o as judge across 4 metrics: Faithfulness, Response Relevancy, Context Precision, Context Recall. Exports results to CSV. |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Embeddings | AWS Bedrock Titan Embed v2 | 1024-dim vectors, fast, managed |
| Vector DB | PostgreSQL + pgvector | Single DB for vectors + metadata + FTS |
| Dense Search | pgvector cosine similarity (IVFFlat) | Sub-second approximate nearest neighbor search |
| Sparse Search | PostgreSQL tsvector + ts_rank | Built-in BM25-style ranking, no extra infra |
| Fusion | Reciprocal Rank Fusion (k=60) | Score-agnostic merging of ranked lists |
| Generation | OpenAI GPT-4o | Strong instruction following, low hallucination |
| Evaluation | RAGAS 0.4.3 | Industry standard RAG benchmarking |
| Infra | Docker Compose | One-command PostgreSQL + pgvector setup |

---

## Why Hybrid Search?

Neither dense nor sparse search is enough on its own:

- **Dense search** understands meaning ("data durability" matches "replication") but struggles with exact terms like error codes or config names
- **Sparse search** nails exact keyword matches but misses semantic similarity ("copying data across brokers" won't match "replication")

Running both and fusing with RRF means chunks that **both methods agree on** rise to the top. RRF is score-agnostic — it only looks at rank positions, so it doesn't matter that cosine similarity (0-1) and BM25 scores (0-20+) are on completely different scales.

---

## Quick Start

### Prerequisites
- Python 3.12+
- Docker (for PostgreSQL + pgvector)
- AWS credentials (for Bedrock embeddings)
- OpenAI API key (for generation)

### Setup

```bash
# 1. Clone and install
git clone <repo-url>
cd "Meridian - Entreprise Knowledge Agent"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Add your AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, OPENAI_API_KEY

# 3. Start PostgreSQL
docker compose up -d

# 4. Run the ingestion pipeline
python3 -m src.ingestion.setup_db       # Create schema + indexes
python3 -m src.ingestion.scraper        # Fetch Confluence pages
python3 -m src.ingestion.chunker        # Split into chunks
python3 -m src.ingestion.embedder       # Embed with Titan v2

# 5. Test it
python3 -m src.retrieval.retriever      # Test hybrid retrieval
python3 -m src.retrieval.generator      # Test full RAG pipeline

# 6. Benchmark
python3 -m src.evaluation.ragas_eval    # Run RAGAS evaluation
```

---

## RAGAS Evaluation — How We Benchmark the Pipeline

RAG systems are hard to evaluate because there are two things that can go wrong independently: the **retriever** can pull bad chunks, or the **generator** can hallucinate even with good chunks. RAGAS (Retrieval Augmented Generation Assessment) scores both sides separately using an LLM-as-judge approach.

We run 5 ground-truth Q&A pairs through the live pipeline, then GPT-4o judges the results across 4 metrics.

### The 4 Metrics

| Metric | Evaluates | How It Works | Score Range |
|--------|-----------|-------------|-------------|
| **Faithfulness** | Generator | Breaks the answer into individual claims, then checks if each claim is supported by the retrieved context. Claims not found in context = hallucination. | 0.0 - 1.0 |
| **Response Relevancy** | Generator | Generates hypothetical questions from the answer, then measures how similar those questions are to the original question. If the answer is off-topic, the generated questions won't match. | 0.0 - 1.0 |
| **Context Precision** | Retriever | Checks if the retrieved chunks that are actually relevant are ranked higher than irrelevant ones. Having the right chunk at position #5 instead of #1 hurts this score. | 0.0 - 1.0 |
| **Context Recall** | Retriever | Compares the ground-truth answer against retrieved chunks to see if the retriever found all the information needed. Missing a key fact = low recall. | 0.0 - 1.0 |

### How the Evaluation Works

```
                         5 Eval Questions
                       (with ground-truth answers)
                                |
                                v
                   Run each through LIVE RAG pipeline
                   (actual retriever + actual generator)
                                |
                                v
                   For each question, we now have:
                   - user_input:         the question
                   - response:           LLM's generated answer
                   - retrieved_contexts: the 5 chunks retriever found
                   - reference:          human-written ground truth
                                |
                                v
                   GPT-4o judges each sample on all 4 metrics
                                |
                                v
                   +-----------------------------------+
                   |         RAGAS SCORES              |
                   |-----------------------------------|
                   | Faithfulness:        0.XXXX       |
                   | Response Relevancy:  0.XXXX       |
                   | Context Precision:   0.XXXX       |
                   | Context Recall:      0.XXXX       |
                   +-----------------------------------+
                                |
                                v
                        ragas_results.csv
                   (per-question breakdown)
```

### What the Scores Tell You

- **All scores high (> 0.8)** — Pipeline is working well. Retriever finds the right docs, generator stays faithful to them.
- **Faithfulness low, others high** — Retriever is doing its job, but the LLM is hallucinating or adding information not in the chunks. Fix: tighten the system prompt or lower temperature.
- **Context Recall low, others high** — The retriever is missing relevant documents. Fix: adjust chunking strategy, add more overlap, or tune RRF weights.
- **Context Precision low** — Retriever is returning too much noise. The relevant chunk is buried under irrelevant ones. Fix: tune TOP_K, add metadata filters, or add a reranker.
- **Response Relevancy low** — The answer doesn't address what was asked. Fix: improve the prompt template or check if retrieved context is leading the LLM off-topic.

### Eval Dataset

We test on questions that cover all three documentation sources:

| # | Question | Tests |
|---|----------|-------|
| 1 | How does Kafka handle replication? | Kafka — core distributed systems concept |
| 2 | What is the checkpoint mechanism in Apache Flink? | Flink — fault tolerance internals |
| 3 | How do you schedule a DAG in Apache Airflow? | Airflow — basic scheduling |
| 4 | How does Kafka guarantee message ordering? | Kafka — ordering guarantees |
| 5 | What is a Flink savepoint vs checkpoint? | Flink — distinguishing similar concepts |

Each question has a human-written ground-truth answer that RAGAS compares against. Add more questions to `EVAL_QUESTIONS` in `ragas_eval.py` to expand coverage.

---

## Project Structure

```
Meridian - Enterprise Knowledge Agent/
|
|-- docker-compose.yml          # PostgreSQL + pgvector
|-- requirements.txt
|-- .env                        # API keys (not committed)
|-- tester.py                   # Quick AWS connectivity check
|
|-- data/
|   |-- raw/                    # Scraped Confluence pages (JSON)
|   |-- chunks/                 # Processed chunks (JSON)
|
|-- src/
    |-- ingestion/
    |   |-- scraper.py          # Confluence API fetcher
    |   |-- chunker.py          # HTML -> text chunks (512 tokens)
    |   |-- embedder.py         # AWS Bedrock Titan embeddings
    |   |-- setup_db.py         # PostgreSQL schema + indexes
    |
    |-- retrieval/
    |   |-- retriever.py        # Dense + Sparse + RRF hybrid search
    |   |-- generator.py        # GPT-4o RAG generation
    |
    |-- evaluation/
        |-- ragas_eval.py       # RAGAS benchmarking (4 metrics)
```

---

## License

Proprietary - Enterprise Use Only
