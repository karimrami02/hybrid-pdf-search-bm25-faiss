from fastapi import FastAPI
from engine import search

app = FastAPI()

@app.post("/search")
def run_search(query: str, mode: str = "fusion"):
    return {
        "query": query,
        "mode": mode,
        "results": search(query, mode)
    }
