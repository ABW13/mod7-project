# Semantic Search over a Document Collection

A working semantic search tool built with ChromaDB, sentence-transformers, and
Streamlit. Documents are loaded from a `docs/` folder, split into chunks with a
configurable strategy, embedded, and stored in a persistent vector database.
The Streamlit interface ranks chunks by meaning rather than keyword overlap.

Built for Module 7 (Applied AI: Embeddings & Retrieval).

---

## Setup

    # 1. Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate

    # 2. Install dependencies
    pip install -r requirements.txt

    # 3. Build the index
    python ingest.py --chunk-strategy fixed --chunk-size 300 --overlap 50

    # 4. Run the app
    streamlit run app.py

The first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB). It is
cached locally and only downloaded once.

### Environment note

This project was developed on two machines. The initial work was done in a
ChromeOS Linux container with 2.7 GB RAM and no swap, where PyTorch — a
dependency of `sentence-transformers` — could not be installed at all, so the
exercises ran against an environment created for an earlier module. The
chunking experiment below was run on a Dell Precision 3561 under WSL2 with
32 GB RAM, where the full dependency set installs normally.

---

## Architecture

    mod7-project/
    ├── app.py                   # Streamlit UI (presentation only)
    ├── ingest.py                # Loading, chunking, embedding, storage
    ├── search.py                # Querying and ranking
    ├── evaluate.py              # Precision / recall, plus the chunking experiment
    ├── chunking_experiment.py   # Convenience wrapper for the experiment
    ├── docs/                    # Source documents
    ├── chroma_data/             # Persistent ChromaDB storage (gitignored)
    ├── requirements.txt
    └── README.md

The modules are deliberately independent:

| Module | Responsibility | Does not |
|---|---|---|
| ingest.py | Read files, split into chunks, write to ChromaDB | Rank or display results |
| search.py | Embed a query, rank chunks, apply threshold and filters | Touch the filesystem |
| evaluate.py | Score search quality, run the chunking experiment | Contain UI code |
| app.py | Render widgets, call the modules above | Contain search or ingestion logic |

This split is what makes the chunking experiment possible: evaluate.py can
rebuild the index at several chunk sizes and score each one without involving
the browser.

### Cosine similarity, and why it is configured explicitly

ChromaDB defaults to L2 (squared Euclidean) distance, whose values are not
comparable with cosine similarity. The collection is created with
metadata={"hnsw:space": "cosine"}, so collection.query() returns cosine
distance — lower is better. SearchResult.similarity converts it with
1 - distance so the UI and the evaluation report in the same units used
elsewhere in this project.

### Chunking strategies

- fixed — splits on a character count with overlap. Predictable, works on
  any input, but can cut mid-sentence.
- paragraph — splits on blank lines. Preserves complete ideas in
  well-structured documents, but chunk sizes vary widely.
- sentence — groups three sentences per chunk. A middle ground.

Only fixed is used in the experiment below, because chunk size is the
variable under test and the other two do not have a size parameter.

---

## Chunking experiment

### Method

The same corpus of eight documents was indexed three times, once for each chunk
size, with overlap fixed at 15% of chunk size. The same six evaluation queries
were run against each index at n_results=5 and threshold=0.30.

Relevance is scored at source-file granularity rather than chunk
granularity. A returned chunk counts as correct when its source file is one of
the files expected to answer that query. This is deliberate: chunk IDs change
whenever the chunk size changes, so anchoring relevance to the file is what
makes the three indexes comparable.

> Reproduce with: python evaluate.py --chunking

### Results

| Chunk size | Chunks | Avg chars | Precision | Recall | hit@5 |
|---|---|---|---|---|---|
| 150 | 120 | 146 | 52.8% | 83.3% | 83.3% |
| 300 | 63 | 278 | 40.3% | 83.3% | 83.3% |
| 600 | 32 | 538 | 52.8% | 83.3% | 83.3% |

Per-query recall by chunk size:

| Query | 150 | 300 | 600 |
|---|---|---|---|
| How do I authenticate requests to my API? | 100% | 100% | 100% |
| How do I make sure incoming request data is valid? | 100% | 100% | 100% |
| Why does my data disappear when I click a button? | 0% | 0% | 0% |
| How is text turned into something a computer can compare? | 100% | 100% | 100% |
| How should I split up a long document before embedding it? | 100% | 100% | 100% |
| How do I keep Python objects and database tables in sync? | 100% | 100% | 100% |

### Interpretation

Recall and hit@5 were identical at all three chunk sizes (83.3%). Every
query that retrieved the correct document at one size retrieved it at all
three. Within this range, chunk size did not change whether the right
document was found — only how many irrelevant chunks accompanied it.

Precision was flat at 150 and 600 (52.8%) and dipped at 300 (40.3%). This
is almost certainly noise rather than a real effect. With six queries, a single
query's precision moving shifts the average by roughly 17 points: at 300
characters, Query 1 returned chunks from four different source files, where at
150 characters it returned chunks from one. That one query accounts for the
entire difference between the middle and the extremes.

Query 3 failed at every chunk size — a threshold effect, not a chunking one. "Why does my data disappear when I click a button?" retrieved nothing above the 0.30 threshold at 150, 300, or 600 characters, despite streamlit_guide.txt containing the answer. Re-running the same query at threshold 0.1 returns the correct chunk first, at similarity 0.249:

1. [ low] sim=0.249 (streamlit_guide.txt #1)
   reset on every interaction. A counter incremented in the script body
   returns to its starting value on the next click...

So the failure was calibration, not retrieval: the right passage ranked first but scored below the threshold. The query's phrasing ("data disappear", "click a button") sits far from the document's wording ("variables reset", "re-run", "widget interaction"), so the match is weak but present. This is the recall/precision tradeoff from the threshold experiment, and it is the strongest argument in this report for testing thresholds per corpus rather than adopting a fixed value.

    python search.py "Why does my data disappear when I click a button?" --threshold 0.1 --n-results 5

The top similarity score fell as chunks grew. For most queries the highest
scoring chunk was more similar at 150 characters than at 600 — for example,
Query 2 scored 0.554 at 150 and 0.365 at 600. This is the expected dilution
effect: a larger chunk's embedding has to represent several ideas at once, so
the match to any single concept is weaker. It did not affect recall here,
because the correct file still ranked in the top five, but it would begin to
matter at a tighter n_results.

Which result to lead with. The finding is that chunk size, across a 4x
range, affected the quality of the result set more than the presence of the
correct answer.

### Caveats

- Six queries is a small evaluation. A single query changing rank moves the
  averages by roughly 17 points. The ordering above is a direction, not a
  verdict.
- File-granularity relevance cannot distinguish "found the right file" from
  "found the exact sentence that answers the question."
- The corpus is small (eight documents, six topics). Cross-topic similarity
  is higher than it would be in a larger collection, which compresses the score
  range and makes thresholds less transferable.
- Only the fixed strategy was tested. Paragraph and sentence chunking would
  need a different experimental design, since neither has a size parameter.

---

## Usage

### Ingest

    python ingest.py --chunk-strategy fixed --chunk-size 300 --overlap 50
    python ingest.py --chunk-strategy paragraph
    python ingest.py --stats
    python ingest.py --reset

### Search from the command line

    python search.py "how does authentication work"
    python search.py "how do I deploy" --n-results 5 --threshold 0.4
    python search.py "chunking" --source embeddings.txt
    python search.py --interactive

### Evaluate

    python evaluate.py
    python evaluate.py --sweep
    python evaluate.py --chunking
    python evaluate.py --sweep --json out.json

### Streamlit app

    streamlit run app.py

Sidebar controls: chunking strategy and size, re-index button, results slider,
similarity threshold slider, and a source-file multiselect. Metrics show the
chunk count and source file count.

---

## Edge cases handled

| Case | Behaviour |
|---|---|
| Empty or whitespace-only query | Returns nothing; the UI shows a prompt instead of searching |
| No results above threshold | UI shows a warning suggesting a lower threshold |
| Source filter excludes everything | Count reads 0 and no results are shown |
| Overlap greater than or equal to chunk size | ValueError from ingest.fixed_chunks; the UI reports it in place |
| Empty docs folder | ValueError on ingest; the UI warns before attempting |
| Re-running ingest | upsert with deterministic IDs overwrites rather than duplicating |
| Collection still using L2 space | Re-index after a config change; the space is fixed at creation |

---

## Known limitations

- No reranking. A cross-encoder rerank over the top ~20 results would
  improve precision noticeably; cosine similarity over a single bi-encoder is
  the cheap first pass.
- No hybrid search. BM25 alongside dense retrieval would help on rare
  tokens and exact identifiers, which embeddings handle poorly.
- File-granularity evaluation. Chunk-level relevance labels would give a
  sharper signal, at the cost of relabelling whenever chunk size changes.
- In-memory result display. Results are not paginated; a large n_results
  renders a long page.
- One query failed at every configuration. Query 3 points to a threshold
  calibration issue rather than a chunking one, and is unresolved in the
  current results.
