# LipidBot

A RAG-based chatbot for lipid biology, combining semantic citation retrieval, Neo4j knowledge graph traversal, and LLM-synthesized streaming responses.

## Architecture Overview

```
User Query
    │
    ├─► Query Classification (LLM)
    │       ├─ Is it lipid-relevant?
    │       └─ Does it need graph traversal?
    │
    ├─► Citation Retrieval (Parallel)
    │       ├─ Semantic search (3 PubMed embedding models + FAISS)
    │       └─ Keyword search (BM25)
    │           └─ Fused via RRF / vote / max score
    │
    ├─► Cypher Query (Parallel, if needed)
    │       ├─ Entity extraction (KEGG autocomplete + LLM)
    │       ├─ Cypher generation (rule-based or LLM)
    │       └─ Neo4j async execution
    │
    └─► LLM Synthesis (Streaming SSE)
            ├─ Prompt with citations + graph results
            └─ Appends references + pathway results
```

## Folder Structure

```
FatPlants_LipidBot/
├── main.py                        # FastAPI app, endpoints, lifespan hooks
├── lipidbot.py                    # Query classification & response formatting
├── llm_factory.py                 # LLM provider factory (Gemini, Ollama, OpenRouter)
├── data_service.py                # Pydantic data models & enums
├── config.py                      # API keys, DB credentials, model config (see Setup)
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Multi-stage Docker build (Python 3.11-slim)
├── docker-compose.dev.yaml        # Services: lipidbot + Neo4j
│
├── citation/                      # Citation retrieval module
│   ├── search.py                  # Hybrid search: multi-model embedding + BM25 fusion
│   ├── built_retriever.py         # Builds hybrid retriever from cached embeddings
│   ├── bm25_cache.py              # BM25 ranking cache
│   ├── build_cache.py             # Cache building utilities
│   ├── cache_helper.py            # Cache I/O helpers
│   ├── chunking.py                # Document chunking
│   ├── embedding.py               # Embedding model selection
│   ├── load_sentence_transformers.py
│   ├── index.py                   # FAISS indexing
│   └── citation_cache/            # Pre-built FAISS indices + BM25 index (not in repo)
│       ├── BAAI_bge-m3__*.{faiss,jsonl,manifest.json}
│       ├── NeuML_pubmedbert-*.{faiss,jsonl,manifest.json}
│       ├── pritamdeka_S-PubMedBert-*.{faiss,jsonl,manifest.json}
│       └── bm25_index.pkl
│
├── cypher/                        # Neo4j knowledge graph module
│   ├── cypher_query.py            # Main cypher query executor
│   ├── db_enginer.py              # Async Neo4j client
│   ├── cypher_generator.py        # Rule-based cypher query builder
│   ├── llm_cypher_generator.py    # LLM-based cypher query generation
│   ├── entity_extractor.py        # Biomedical entity extraction
│   ├── llm_entity_extractor.py    # LLM-based entity extraction
│   ├── ac.py                      # KEGG entity autocomplete cache
│   └── ac_kegg.pkl                # Pre-built KEGG entity cache
│
├── evaluation/                    # Benchmarking & evaluation scripts
│   ├── citation_retrieval_make_question_semantic.py
│   ├── get_citation_pred_result.py
│   ├── get_cypher_result_lipidbot.py
│   ├── get_cypher_result_QA.py
│   └── process_citation_result.py
│
├── file/                          # Data files & evaluation results (CSV)
└── models/                        # HuggingFace model cache (mounted via Docker)
    └── huggingface/
```

## Setup

### 1. Configure `config.py`

Request the config file from the project maintainer. It should define:

```python
GEMINI_API_KEY = "..."
GEMINI_MODEL_NAME = "gemini-2.5-flash"
OLLAMA_HOST = "http://..."
NEO4J_URI = "bolt://neo4j-fatplants:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "..."
OPENROUTER_API_KEY = "..."

DEFAULT_EMBEDDING_MODEL = [
    "NeuML/pubmedbert-base-embeddings",
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "BAAI/bge-m3",
]
```

> **Do not commit `config.py` to version control.** It contains secrets.

### 2. Get the Citation Cache

Request the pre-built citation cache from the project maintainer and place it at:

```
citation/citation_cache/
```

This directory contains pre-built FAISS indices and a BM25 index for the PubMed literature corpus. Without it, citation retrieval will not work.

### 3. Run with Docker Compose

```bash
docker compose -f docker-compose.dev.yaml up -d
```

This starts two services:
- **lipid-bot** — FastAPI app on port `7120`
- **neo4j** — Neo4j graph database on ports `7474` (browser) and `7687` (bolt)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/lipidbot/cypher/` | Run a raw Cypher query against Neo4j |
| `POST` | `/lipidbot/stream` | Main endpoint — streams a synthesized response |

### Stream Request Body

```json
{
  "query": "What enzymes are involved in fatty acid elongation?",
  "llm_type": "gemini",
  "top_k": 5,
  "fusion_strategy": "rrf"
}
```

Response is streamed as Server-Sent Events (SSE).

## LLM Providers

| Provider | Models | Notes |
|----------|--------|-------|
| `gemini` | `gemini-2.5-flash` | Default |
| `ollama` | `llama3.1`, `gpt-oss-20b` | Requires Ollama server |
| `openrouter` | `meta-llama/llama-3.1-8b`, `openai/gpt-oss-20b` | API gateway |

## Embedding Models

Three biomedical embedding models are used in parallel for citation retrieval:

- `NeuML/pubmedbert-base-embeddings`
- `pritamdeka/S-PubMedBert-MS-MARCO`
- `BAAI/bge-m3`

Results are fused using **Reciprocal Rank Fusion (RRF)** by default.

## Knowledge Graph

The Neo4j graph models lipid biology entities:

- **Nodes**: Gene, Compound, Reaction, Pathway, EC/Enzyme, Ortholog
- **Source**: KEGG pathway database
- **Query generation**: Rule-based templates or LLM-written Cypher
