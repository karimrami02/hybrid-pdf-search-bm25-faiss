FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

# Upgrade pip and install PyTorch CPU + dependencies
RUN pip install --upgrade pip \
    && pip install --timeout 120 --retries 10 \
        filelock typing-extensions sympy networkx jinja2 fsspec \
    && pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install "numpy<2" rank_bm25

# Install the rest of your dependencies (FastAPI, SentenceTransformers, etc.)
RUN pip install -r requirements.txt

# Optional: ensure uvicorn CLI dependencies
RUN pip install uvicorn[standard]

COPY . .

EXPOSE 8000
CMD ["uvicorn", "ui:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
