"""FastAPI-powered UI for BM25 + FAISS hybrid search."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from engine import search as engine_search


BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


RESULT_LIMIT = int(os.getenv("TOP_K", 10))
MAX_RESULT_LIMIT = int(os.getenv("MAX_RESULT_LIMIT", 100))
MIN_RESULT_LIMIT = 1

RANKING_PROFILES = [
    {"value": "fusion", "label": "Hybrid (semantic + BM25)"},
    {"value": "semantic", "label": "Semantic (dense vector only)"},
    {"value": "bm25", "label": "BM25 (lexical only)"},
]
DEFAULT_RANKING_PROFILE = "fusion"

class SearchRequest(BaseModel):
    query: str
    limit: int | None = None
    ranking: str | None = None

app = FastAPI(title="Hybrid Search UI", version="0.1.0")

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

def _resolve_limit(candidate: int | None) -> int:
    limit = candidate if candidate is not None else RESULT_LIMIT
    try:
        limit_value = int(limit)
    except (TypeError, ValueError):
        limit_value = RESULT_LIMIT
    return max(MIN_RESULT_LIMIT, min(MAX_RESULT_LIMIT, limit_value))


def _normalize_ranking(candidate: str | None) -> str:
    if candidate in {p["value"] for p in RANKING_PROFILES}:
        return candidate
    return DEFAULT_RANKING_PROFILE


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    """Render the main search page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_limit": RESULT_LIMIT,
            "max_limit": MAX_RESULT_LIMIT,
            "ranking_profiles": RANKING_PROFILES,
            "default_ranking_profile": DEFAULT_RANKING_PROFILE,
        },
    )


@app.post("/search")
async def search(request: SearchRequest) -> Dict[str, Any]:
    """Execute search and return results."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    
    limit = _resolve_limit(request.limit)
    ranking = _normalize_ranking(request.ranking)
    
    try:
        hits = engine_search(query, mode=ranking, top_k=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {
        "query": query,
        "hits": hits,
        "returned": len(hits),
        "limit": limit,
        "ranking_profile": ranking,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)