# Semantic Search over a Document Collection

A working semantic search tool built with ChromaDB, sentence-transformers, and
Streamlit. Documents are loaded from a `docs/` folder, split into chunks with a
configurable strategy, embedded, and stored in a persistent vector database.
The Streamlit interface ranks chunks by meaning rather than keyword overlap.

Built for Module 7 (Applied AI: Embeddings & Retrieval).

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the index
python ingest.py --chunk-strategy fixed --chunk-size 300 --overlap 50

# 4. Run the app
streamlit run app.py
```

Note on the environment: this project was developed against a virtual environment created for the earlier Module 7 exercises, which already contained chromadb, sentence-transformers, and streamlit. The dependency list in requirements.txt reflects that environment. On a machine with limited RAM, install torch from the CPU-only index before the rest — the default PyTorch wheel requires more memory than a 2.7 GB container can spare.
