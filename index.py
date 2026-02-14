"""
Hybrid BM25 + FAISS indexer
Supports large datasets (>1M docs) with chunked ingestion
Windows + Docker compatible
"""
import os
import pickle
from tqdm import tqdm
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import warnings

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "en"
SPLIT = "train"

MAX_DOCS = int(os.getenv("MAX_DOCS", 100000))  
CHUNK_SIZE = 5000  
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


print(f"Streaming dataset {DATASET_NAME}/{DATASET_CONFIG} split={SPLIT}...")
dataset_stream = load_dataset(
    DATASET_NAME, 
    DATASET_CONFIG, 
    split=SPLIT, 
    streaming=True
)

documents = []
total = 0

for row in tqdm(dataset_stream, desc="Streaming docs", unit="doc", total=MAX_DOCS):
    text = row.get("text")
    if text and len(text.strip()) > 0:  
        documents.append(text)
        total += 1
    if total >= MAX_DOCS:
        break

print(f"Loaded {len(documents)} documents.")


print("Tokenizing documents for BM25...")
tokenized_docs = [doc.split() for doc in tqdm(documents, desc="Tokenizing")]

with open(os.path.join(DATA_DIR, "bm25_tokenized.pkl"), "wb") as f:
    pickle.dump(tokenized_docs, f)

print("BM25 tokenized docs saved.")


print(f"Encoding documents with {EMBEDDING_MODEL}...")
encoder = SentenceTransformer(EMBEDDING_MODEL)

d = encoder.get_sentence_embedding_dimension()

print(f"Building FAISS Flat index (dimension={d})...")
faiss_index = faiss.IndexHNSWFlat(d, 32) 

print(f"Processing {len(documents)} documents in chunks of {CHUNK_SIZE}...")
all_embeddings = []

for start in tqdm(range(0, len(documents), CHUNK_SIZE), desc="Encoding chunks"):
    end = min(start + CHUNK_SIZE, len(documents))
    chunk_docs = documents[start:end]
    
    embeddings = encoder.encode(
        chunk_docs, 
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32  
    ).astype("float32")
    
    all_embeddings.append(embeddings)

print("Combining embeddings...")
all_embeddings = np.vstack(all_embeddings)

print("Normalizing vectors...")
faiss.normalize_L2(all_embeddings)

print("Adding to FAISS index...")
faiss_index.add(all_embeddings)

faiss.write_index(faiss_index, os.path.join(DATA_DIR, "faiss_hybrid.index"))
print(f"FAISS index saved with {faiss_index.ntotal} vectors.")

with open(os.path.join(DATA_DIR, "original_docs.pkl"), "wb") as f:
    pickle.dump(documents, f)

print("\n" + "="*60)
print(" Indexing complete!")
print("="*60)
print(f"Documents indexed: {len(documents)}")
print(f"BM25 tokenized docs: {DATA_DIR}/bm25_tokenized.pkl")
print(f"FAISS index: {DATA_DIR}/faiss_hybrid.index")
print(f"Original documents: {DATA_DIR}/original_docs.pkl")
print("="*60)
print("\nYou can now run: python engine.py (for testing)")
print("Or run: uvicorn api:app --reload (for API server)")