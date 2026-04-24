# Meridian - Enterprise Knowledge Agent

An AI-powered enterprise knowledge agent that leverages advanced retrieval techniques including sparse and dense retrievers with reciprocal rank fusion, powered by Amazon Titan embeddings.

## Features

- **Sparse and Dense Retrieval**: Hybrid retrieval combining BM25 sparse retrieval with dense embedding-based search
- **Reciprocal Rank Fusion**: Intelligent fusion of multiple retrieval signals for improved relevance
- **Amazon Titan Embeddings**: High-quality semantic embeddings for document understanding
- **Ingestion & Tokenization**: Robust document processing pipeline

## Getting Started

### Requirements

See `requirements.txt` for dependencies.

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your `.env` file (use `.env.example` as reference)
4. Run the application

## Architecture

The system is built with a modular architecture supporting:
- Document ingestion and tokenization
- Semantic embedding generation
- Hybrid retrieval pipelines
- Enterprise knowledge base management

## License

Proprietary - Enterprise Use Only
