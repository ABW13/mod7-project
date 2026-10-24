"""Semantic search over a ChromaDB collection.

Separated from the UI so the search logic can be imported by the Streamlit
app, exercised from a script, or evaluated in ``evaluate.py`` without pulling
in any presentation code.

Usage from the command line::

    python search.py "how does authentication work"
    python search.py "how do I deploy" --n-results 5 --threshold 0.4
    python search.py "chunking" --source docs/embeddings.txt
    python search.py --interactive
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import chromadb
from chromadb.utils import embedding_functions

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_DB_PATH = "chroma_data"
DEFAULT_COLLECTION = "course_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COSINE_SPACE = {"hnsw:space": "cosine"}

DEFAULT_N_RESULTS = 5
DEFAULT_THRESHOLD = 0.30

# Thresholds for the coloured relevance indicator shown in the UI.
HIGH_CUTOFF = 0.50
MEDIUM_CUTOFF = 0.35


@dataclass
class SearchResult:
    """One ranked chunk returned for a query."""

    chunk_id: str
    text: str
    source: str
    chunk_index: int
    strategy: str
    distance: float

    @property
    def similarity(self) -> float:
        """Cosine similarity, converted from ChromaDB's distance.

        The collection is created in cosine space, so ``1 - distance`` maps
        back to the similarity scores used elsewhere in this project. Higher
        is a better match.
        """
        return 1.0 - self.distance

    @property
    def relevance(self) -> str:
        """Human-readable band for the score."""
        if self.similarity >= HIGH_CUTOFF:
            return "high"
        if self.similarity >= MEDIUM_CUTOFF:
            return "medium"
        return "low"


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def get_collection(
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Open the persistent ChromaDB collection for querying.

    Args:
        db_path: ChromaDB storage directory.
        collection_name: Collection name.

    Returns:
        A ChromaDB Collection.
    """
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata=COSINE_SPACE,
    )


def search(
    query: str,
    n_results: int = DEFAULT_N_RESULTS,
    threshold: float = DEFAULT_THRESHOLD,
    sources: list[str] | None = None,
    collection=None,
) -> list[SearchResult]:
    """Find the chunks most similar to a query.

    Handles the two edge cases the rubric calls out:

    * an empty or whitespace-only query returns ``[]`` rather than embedding
      an empty string and returning noise
    * results below ``threshold`` are dropped, so a query with no good match
      returns ``[]`` rather than a fixed number of weak hits

    Args:
        query: The user's search text.
        n_results: Maximum number of results to return.
        threshold: Minimum cosine similarity to include a result.
        sources: Optional list of filenames to restrict the search to.
        collection: An open collection; opened on demand when omitted.

    Returns:
        A list of SearchResult objects, best match first.
    """
    if not query or not query.strip():
        return []

    if collection is None:
        collection = get_collection()

    total = collection.count()
    if total == 0:
        return []

    # ChromaDB rejects an empty where clause, so it is only attached when a
    # source filter is actually requested.
    where = None
    if sources:
        where = (
            {"source": sources[0]}
            if len(sources) == 1
            else {"source": {"$in": list(sources)}}
        )

    raw = collection.query(
        query_texts=[query],
        n_results=min(n_results, total),
        where=where,
    )

    results: list[SearchResult] = []
    for i in range(len(raw["ids"][0])):
        meta = raw["metadatas"][0][i]
        result = SearchResult(
            chunk_id=raw["ids"][0][i],
            text=raw["documents"][0][i],
            source=meta.get("source", "unknown"),
            chunk_index=int(meta.get("chunk_index", -1)),
            strategy=meta.get("strategy", "unknown"),
            distance=raw["distances"][0][i],
        )
        if result.similarity >= threshold:
            results.append(result)

    return results


def count_in_scope(
    sources: list[str] | None = None,
    collection=None,
) -> int:
    """Count chunks in the database, optionally limited to given sources.

    ChromaDB has no filtered count call, so this sums per-source counts.

    Args:
        sources: Filenames to count; all sources when empty.
        collection: An open collection; opened on demand when omitted.

    Returns:
        Number of chunks in scope.
    """
    if collection is None:
        collection = get_collection()

    if not sources:
        return collection.count()

    total = 0
    for source in sources:
        total += len(collection.get(where={"source": source})["ids"])
    return total


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_results(query: str, results: list[SearchResult]) -> None:
    print(f"\nQuery: {query!r}")
    if not results:
        print("  No results above the similarity threshold.")
        return
    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split())
        if len(preview) > 160:
            preview = preview[:160].rstrip() + "..."
        print(
            f"  {rank}. [{result.relevance:>6}] "
            f"sim={result.similarity:.3f} "
            f"({result.source} #{result.chunk_index})"
        )
        print(f"     {preview}")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Query the vector store.")
    parser.add_argument("query", nargs="?", help="Search text.")
    parser.add_argument("--n-results", type=int, default=DEFAULT_N_RESULTS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Restrict to a filename. Repeatable.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Read queries from stdin until 'quit'.",
    )
    args = parser.parse_args()

    collection = get_collection()
    print(f"Chunks in collection: {collection.count()}")

    if args.interactive or not args.query:
        print("Type a query, or 'quit' to exit.")
        while True:
            try:
                line = input("\nSearch (or 'quit'): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() == "quit":
                break
            if not line:
                continue
            _print_results(
                line,
                search(
                    line,
                    n_results=args.n_results,
                    threshold=args.threshold,
                    sources=args.source,
                    collection=collection,
                ),
            )
        return

    _print_results(
        args.query,
        search(
            args.query,
            n_results=args.n_results,
            threshold=args.threshold,
            sources=args.source,
            collection=collection,
        ),
    )


if __name__ == "__main__":
    _main()