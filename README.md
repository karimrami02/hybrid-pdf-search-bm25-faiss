# Hybrid PDF Search Engine  
BM25 + FAISS Hybrid Retrieval System

A high-performance hybrid search engine for large collections of PDF documents.  
This system combines **BM25 keyword search** with **FAISS semantic vector search** to deliver accurate, scalable, and fast document retrieval.

Designed for:
- RAG systems
- Enterprise document search
- Research and knowledge bases
- AI assistants over PDFs

---

## Features

- Keyword-based retrieval using BM25
- Semantic search using dense embeddings + FAISS
- Hybrid ranking (lexical + semantic)
- PDF ingestion and chunking
- REST API
- Web UI
- Dockerized deployment
- Scales to millions of document chunks

---

## Project Structure

```
.
├── api.py           # REST API
├── engine.py        # Hybrid search logic
├── index.py         # PDF indexing pipeline
├── ui.py            # Web interface
├── templates/       # UI templates
├── data/            # Stores PDFs and generated indexes
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Architecture

```
PDFs → index.py
         |
         |---> BM25 Index
         |---> FAISS Vector Index
         |
       Stored on disk
            |
User Query → api.py → engine.py
                    |
            BM25 + FAISS Retrieval
                    |
               Ranked Results
```

---

## Installation

### Option 1 — Local Installation

Requires Python 3.10+

```bash
pip install -r requirements.txt
```

---

### Option 2 — Docker (recommended)

Build image:

```bash
docker build -t hybrid-search .
```

Run:

```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data hybrid-search
```

---

## Indexing PDFs

Place your PDFs inside:

```
data/pdfs/
```

Then build the index:

```bash
python index.py
```

This will:
- Extract text from all PDFs
- Chunk documents
- Create a BM25 index
- Create a FAISS embedding index

The generated files:

```
data/
├── bm25.index
├── faiss.index
└── documents.json
```

These files are loaded by the engine.

---

## Running the Search API

```bash
python api.py
```

API runs at:

```
http://localhost:8000
```

---

## Querying the API

Example:

```bash
curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "reinforcement learning bellman equation"}'
```

Response contains:
- Relevant document chunks
- Scores
- Source files

---

## Web UI

Launch:

```bash
python ui.py
```

Open:

```
http://localhost:8501
```

Allows interactive searching over indexed PDFs.

---

## How Hybrid Search Works

Two independent searches are executed:

- **BM25** → finds exact keyword matches  
- **FAISS** → finds semantic similarity via embeddings  

Results are normalized and merged into a single ranked list.  
This avoids:
- Missing results due to vocabulary mismatch
- Keyword spam dominating ranking

---

## Scaling

This system supports large-scale collections:

- FAISS handles millions of vectors
- BM25 uses efficient inverted indices
- Docker enables GPU and multi-core deployment

For massive scale:
- Use FAISS IVF or HNSW
- Enable GPU FAISS
- Shard BM25 index

---

## Typical Applications

- AI assistants over internal documents
- Legal and compliance search
- Research paper retrieval
- Knowledge management systems
- RAG pipelines for LLMs

---

## Rebuilding the Index

Whenever PDFs change:

```bash
python index.py
```

Then restart the API.

---

## License

MIT
