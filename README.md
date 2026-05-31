# DocumentGPT – Enterprise RAG System

A production-grade Retrieval-Augmented Generation (RAG) system capable of ingesting 10,000+ documents and answering queries using **hybrid retrieval** (dense + sparse), **parent-child chunking**, **Redis caching**, and a **scalable async FastAPI** backend.

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │              FastAPI (async)                  │
                         │  POST /ingest  │  POST /query  │  GET /health │
                         └───────┬────────────────┬────────────────────┘
                                 │                │
              ┌──────────────────▼──┐     ┌───────▼──────────────────────┐
              │  Ingestion Pipeline  │     │       RAG Pipeline            │
              │                      │     │                               │
              │  parse → normalize   │     │  query rewrite →              │
              │  parent-child chunk  │     │  embedding →                  │
              │  embed → upsert      │     │  hybrid retrieval →           │
              └──────┬───────────────┘     │  parent expand →              │
                     │                     │  LLM call →                   │
          ┌──────────▼──────────┐          │  structured response           │
          │   OpenAI Embeddings  │          └──────────────┬────────────────┘
          │   (batched + cached) │                         │
          └──────────┬──────────┘          ┌──────────────▼────────────────┐
                     │                     │       Hybrid Retriever         │
          ┌──────────▼──────────┐          │  Dense (Pinecone 0.7α) +       │
          │   Pinecone Vector DB │◄─────────│  Sparse (BM25 0.3α)           │
          │   (child chunks)     │          └────────────────────────────────┘
          └──────────────────────┘
                                             ┌──────────────────────────────┐
                                             │     Redis Cache              │
                                             │  embeddings + responses      │
                                             └──────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI (async) + Uvicorn |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | Pinecone (Serverless) |
| Sparse Search | BM25 (rank-bm25) |
| Caching | Redis 7.2 |
| Containerization | Docker + docker-compose |
| Cloud | AWS ECS (Fargate) ready |

---

## Features

- **Document Ingestion**: PDF, DOCX, TXT, Markdown
- **Parent-Child Chunking**: Parents (1500 tokens) for context, Children (400 tokens) for retrieval
- **Hybrid Search**: `score = 0.7 × dense + 0.3 × sparse`
- **Multi-Vector Retrieval**: Child → Parent context expansion
- **Query Rewriting**: LLM-powered search query optimization
- **Redis Caching**: Embeddings (24h TTL) + Responses (1h TTL)
- **Prometheus Metrics**: `/metrics` endpoint
- **Structured Output**: JSON with `answer`, `sources`, `confidence_score`

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# Docker & docker-compose
docker --version
docker compose version
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys:
#   OPENAI_API_KEY=sk-...
#   PINECONE_API_KEY=pcsk-...
```

### 3. Launch with Docker Compose

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`

### 4. Run with Dev Tools (Redis Commander UI)

```bash
docker compose --profile dev up
# Redis Commander: http://localhost:8081
```

---

## API Reference

### `GET /api/v1/health`

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "services": {
    "redis": "healthy",
    "pinecone": "healthy",
    "bm25_corpus": "loaded (1250 chunks)"
  },
  "uptime_seconds": 3600.5
}
```

---

### `POST /api/v1/ingest`

Upload a document for processing.

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@/path/to/document.pdf" \
  -F "namespace=my-project"
```

```json
{
  "doc_id": "3f8a2b1c-...",
  "status": "success",
  "file_name": "document.pdf",
  "parent_chunks": 12,
  "child_chunks": 48,
  "ingestion_time_ms": 1240.5,
  "namespace": "my-project"
}
```

---

### `POST /api/v1/query`

Query the knowledge base.

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key benefits of ERP systems?",
    "namespace": "my-project",
    "top_k": 5,
    "rewrite_query": true
  }'
```

```json
{
  "query_id": "q-7f2a...",
  "query": "What are the key benefits of ERP systems?",
  "answer": "ERP systems provide several key benefits including...",
  "sources": [
    {
      "doc_id": "3f8a2b1c-...",
      "chunk_id": "c-ab12...",
      "source_file": "erp_overview.pdf",
      "relevance_score": 0.9234,
      "excerpt": "Enterprise Resource Planning systems integrate..."
    }
  ],
  "confidence_score": 0.87,
  "status": "success",
  "cached": false,
  "latency_ms": 823.4,
  "model_used": "gpt-4o",
  "tokens_used": 1542
}
```

---

## Local Development (without Docker)

```bash
# Create venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis locally
docker run -d -p 6379:6379 redis:7.2-alpine

# Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

---

## AWS ECS Deployment

### 1. Build and Push to ECR

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build -t documentgpt .
docker tag documentgpt:latest ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/documentgpt:latest
docker push ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/documentgpt:latest
```

### 2. Register Task Definition

```bash
# Edit docker/ecs-task-definition.json — replace ACCOUNT_ID and endpoints
aws ecs register-task-definition \
  --cli-input-json file://docker/ecs-task-definition.json
```

### 3. Create ECS Service

```bash
aws ecs create-service \
  --cluster documentgpt-cluster \
  --service-name documentgpt-api \
  --task-definition documentgpt-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}"
```

### Infrastructure Notes

- **Stateless backend**: all state in Redis/Pinecone — scale horizontally freely
- **Secrets**: stored in AWS Secrets Manager, injected via ECS task definition
- **Logging**: CloudWatch Logs via `awslogs` driver
- **Cache**: Use AWS ElastiCache (Redis) for production
- **Metrics**: Prometheus `/metrics` endpoint — scrape with CloudWatch Container Insights or Grafana

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `PINECONE_API_KEY` | required | Pinecone API key |
| `PINECONE_INDEX_NAME` | `documentgpt-index` | Pinecone index name |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | LLM for answer generation |
| `HYBRID_DENSE_WEIGHT` | `0.7` | Weight for dense retrieval score |
| `HYBRID_SPARSE_WEIGHT` | `0.3` | Weight for BM25 score |
| `PARENT_CHUNK_SIZE` | `1500` | Parent chunk size in tokens |
| `CHILD_CHUNK_SIZE` | `400` | Child chunk size in tokens |
| `MAX_CONTEXT_TOKENS` | `6000` | Max tokens sent to LLM |
| `REDIS_TTL_SECONDS` | `3600` | Response cache TTL |
| `REDIS_EMBEDDING_TTL` | `86400` | Embedding cache TTL |

---

## Project Structure

```
documentgpt/
├── app/
│   ├── main.py                    # FastAPI app factory
│   ├── api/v1/
│   │   ├── ingest.py              # POST /ingest
│   │   ├── query.py               # POST /query
│   │   └── health.py              # GET /health
│   ├── core/
│   │   ├── config.py              # Pydantic settings
│   │   ├── models.py              # Domain models
│   │   ├── exceptions.py          # Custom exceptions
│   │   └── logging.py             # Structured logging
│   ├── ingestion/
│   │   ├── parsers.py             # PDF/DOCX/TXT/MD parsers
│   │   ├── chunker.py             # Parent-child chunking
│   │   └── pipeline.py            # Ingestion orchestrator
│   ├── vectorstore/
│   │   ├── embedder.py            # OpenAI embeddings + caching
│   │   └── pinecone_store.py      # Pinecone client
│   ├── retrieval/
│   │   ├── sparse_retriever.py    # BM25 in-memory
│   │   └── hybrid_retriever.py    # Dense + sparse fusion
│   ├── cache/
│   │   └── redis_cache.py         # Redis async client
│   └── services/
│       ├── rag_pipeline.py        # Full RAG orchestration
│       └── dependencies.py        # DI container + lifespan
├── tests/
│   └── test_all.py                # Full test suite
├── docker/
│   └── ecs-task-definition.json   # AWS ECS task definition
├── Dockerfile                     # Multi-stage production build
├── docker-compose.yml             # Local + production compose
├── requirements.txt
└── .env.example
```

---

## License

MIT
