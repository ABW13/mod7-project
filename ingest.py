"""Document loading, chunking, and ChromaDB storage.

This module owns everything that happens *before* a query: reading files off
disk, splitting them into chunks, and writing those chunks into a persistent
ChromaDB collection.

Two chunking strategies are provided so the chunking experiment in the README
can be reproduced:

* ``fixed_chunks``    - split on a character count with overlap
* ``paragraph_chunks`` - split on blank lines

Usage from the command line::

    python ingest.py --chunk-strategy fixed --chunk-size 300 --overlap 50
    python ingest.py --chunk-strategy paragraph
    python ingest.py --stats
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_DOCS_DIR = "docs"
DEFAULT_DB_PATH = "chroma_data"
DEFAULT_COLLECTION = "course_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Cosine space is set explicitly. ChromaDB defaults to L2 (squared Euclidean),
# whose distances are not comparable with the cosine similarities used
# elsewhere in this project.
COSINE_SPACE = {"hnsw:space": "cosine"}


@dataclass
class Chunk:
    """A single chunk of a source document, ready to be embedded."""

    text: str
    source: str
    chunk_index: int
    strategy: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    @property
    def chunk_id(self) -> str:
        """Stable ID: re-ingesting the same file overwrites rather than duplicates."""
        return f"{self.source}::{self.strategy}::{self.chunk_index}"

    @property
    def metadata(self) -> dict:
        """Metadata stored alongside the embedding in ChromaDB."""
        return {
            "source": self.source,
            "chunk_index": str(self.chunk_index),
            "strategy": self.strategy,
            "char_count": str(self.char_count),
        }


# --------------------------------------------------------------------------
# Chunking strategies
# --------------------------------------------------------------------------


def fixed_chunks(
    text: str,
    source: str,
    size: int = 300,
    overlap: int = 50,
    strategy_name: str = "fixed",
) -> list[Chunk]:
    """Split text into fixed-size character chunks with overlap.

    The overlap exists so that a sentence spanning a boundary is still present,
    whole, in one of the neighbouring chunks. Overlap must be smaller than the
    chunk size or the window would never advance.

    Args:
        text: The full document text.
        source: Filename to record in metadata.
        size: Target characters per chunk.
        overlap: Characters repeated at the start of the next chunk.
        strategy_name: Label stored in metadata, so two strategies can coexist.

    Returns:
        A list of Chunk objects.

    Raises:
        ValueError: If overlap is not smaller than size.
    """
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than size ({size})"
        )

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    chunk_index=index,
                    strategy=strategy_name,
                )
            )
            index += 1

        if end == len(text):
            break
        start += size - overlap

    return chunks


def paragraph_chunks(
    text: str,
    source: str,
    strategy_name: str = "paragraph",
) -> list[Chunk]:
    """Split text on blank lines, keeping each paragraph whole.

    Preserves complete ideas in well-structured documents, but chunk sizes vary
    with paragraph length: a heading becomes a tiny chunk, a long section a
    large one.

    Args:
        text: The full document text.
        source: Filename to record in metadata.
        strategy_name: Label stored in metadata.

    Returns:
        A list of Chunk objects, one per non-empty paragraph.
    """
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    return [
        Chunk(
            text=para,
            source=source,
            chunk_index=i,
            strategy=strategy_name,
        )
        for i, para in enumerate(paragraphs)
    ]


def sentence_chunks(
    text: str,
    source: str,
    sentences_per_chunk: int = 3,
    strategy_name: str = "sentence",
) -> list[Chunk]:
    """Group sentences into chunks of roughly ``sentences_per_chunk`` each.

    A middle ground: more uniform than paragraphs, more semantically coherent
    than a fixed character window.

    Args:
        text: The full document text.
        source: Filename to record in metadata.
        sentences_per_chunk: Sentences per chunk.
        strategy_name: Label stored in metadata.

    Returns:
        A list of Chunk objects.
    """
    import re

    flat = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", flat) if flat else []

    chunks: list[Chunk] = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = " ".join(sentences[i : i + sentences_per_chunk]).strip()
        if group:
            chunks.append(
                Chunk(
                    text=group,
                    source=source,
                    chunk_index=len(chunks),
                    strategy=strategy_name,
                )
            )
    return chunks


STRATEGIES = {
    "fixed": fixed_chunks,
    "paragraph": paragraph_chunks,
    "sentence": sentence_chunks,
}


# --------------------------------------------------------------------------
# Document loading
# --------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".txt", ".md")


def load_documents(docs_dir: str = DEFAULT_DOCS_DIR) -> list[tuple[str, str]]:
    """Read every .txt and .md file in a directory.

    Args:
        docs_dir: Directory to scan.

    Returns:
        A list of ``(filename, content)`` tuples, sorted by filename.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(
            f"Document directory not found: {docs_dir!r}. "
            f"Run this script from the project root."
        )

    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(SUPPORTED_EXTENSIONS):
            continue
        path = os.path.join(docs_dir, filename)
        with open(path, "r", encoding="utf-8") as handle:
            documents.append((filename, handle.read()))
    return documents


def build_chunks(
    documents: Iterable[tuple[str, str]],
    strategy: str = "fixed",
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[Chunk]:
    """Apply a chunking strategy to a set of documents.

    Args:
        documents: ``(filename, content)`` pairs from ``load_documents``.
        strategy: One of the keys in ``STRATEGIES``.
        chunk_size: Character size, used by the ``fixed`` strategy.
        overlap: Character overlap, used by the ``fixed`` strategy.

    Returns:
        A flat list of Chunk objects across all documents.

    Raises:
        ValueError: If the strategy name is unknown.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from {sorted(STRATEGIES)}"
        )

    chunks: list[Chunk] = []
    for filename, content in documents:
        if strategy == "fixed":
            chunks.extend(
                fixed_chunks(
                    content, filename, size=chunk_size, overlap=overlap
                )
            )
        elif strategy == "sentence":
            chunks.extend(sentence_chunks(content, filename))
        else:
            chunks.extend(paragraph_chunks(content, filename))
    return chunks


# --------------------------------------------------------------------------
# ChromaDB storage
# --------------------------------------------------------------------------


def get_collection(
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    reset: bool = False,
):
    """Open (or create) the persistent ChromaDB collection.

    Uses the sentence-transformers embedding function already cached locally
    rather than ChromaDB's default embedder, which fetches an ONNX model on
    first use and can appear to hang while it downloads.

    Args:
        db_path: Directory for persistent storage.
        collection_name: Name of the collection.
        reset: If True, delete the collection first for a clean rebuild.

    Returns:
        A ChromaDB Collection.
    """
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=db_path)

    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            # Collection did not exist; nothing to delete.
            pass

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata=COSINE_SPACE,
    )


def ingest(
    docs_dir: str = DEFAULT_DOCS_DIR,
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    strategy: str = "fixed",
    chunk_size: int = 300,
    overlap: int = 50,
    reset: bool = False,
) -> dict:
    """Load, chunk, embed, and store the document collection.

    Args:
        docs_dir: Folder holding the source documents.
        db_path: ChromaDB storage directory.
        collection_name: Collection to write into.
        strategy: Chunking strategy name.
        chunk_size: Character chunk size for the ``fixed`` strategy.
        overlap: Character overlap for the ``fixed`` strategy.
        reset: Drop the existing collection before writing.

    Returns:
        A summary dict: number of files, chunks, and the chunk size range.
    """
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(
            f"No .txt or .md files found in {docs_dir!r}."
        )

    chunks = build_chunks(
        documents, strategy=strategy, chunk_size=chunk_size, overlap=overlap
    )
    if not chunks:
        raise ValueError("Chunking produced no chunks.")

    collection = get_collection(db_path, collection_name, reset=reset)

    collection.upsert(
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        ids=[c.chunk_id for c in chunks],
    )

    sizes = [c.char_count for c in chunks]
    return {
        "files": len(documents),
        "chunks": len(chunks),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "avg_chars": sum(sizes) // len(sizes),
        "strategy": strategy,
        "stored_total": collection.count(),
    }


def collection_stats(
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> dict:
    """Report what is currently stored, without modifying anything.

    Args:
        db_path: ChromaDB storage directory.
        collection_name: Collection to inspect.

    Returns:
        A dict with the total count, per-source counts, strategies present,
        and the chunk size range.
    """
    collection = get_collection(db_path, collection_name)
    payload = collection.get(include=["metadatas"])

    metadatas = payload["metadatas"] or []
    by_source: dict[str, int] = {}
    by_strategy: dict[str, int] = {}

    for meta in metadatas:
        by_source[meta["source"]] = by_source.get(meta["source"], 0) + 1
        strategy = meta.get("strategy", "unknown")
        by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

    sizes = [int(m["char_count"]) for m in metadatas if "char_count" in m]

    return {
        "total": collection.count(),
        "by_source": dict(sorted(by_source.items())),
        "by_strategy": dict(sorted(by_strategy.items())),
        "min_chars": min(sizes) if sizes else 0,
        "max_chars": max(sizes) if sizes else 0,
        "avg_chars": (sum(sizes) // len(sizes)) if sizes else 0,
    }


def list_sources(
    docs_dir: str = DEFAULT_DOCS_DIR,
) -> list[str]:
    """Return the filenames available in the docs folder.

    Used by the Streamlit sidebar to populate the source filter.

    Args:
        docs_dir: Folder holding the source documents.

    Returns:
        Sorted filenames with supported extensions.
    """
    if not os.path.isdir(docs_dir):
        return []
    return sorted(
        f
        for f in os.listdir(docs_dir)
        if f.endswith(SUPPORTED_EXTENSIONS)
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the ChromaDB collection."
    )
    parser.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--chunk-strategy",
        default="fixed",
        choices=sorted(STRATEGIES),
    )
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before ingesting.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print what is stored and exit without ingesting.",
    )
    args = parser.parse_args()

    if args.stats:
        stats = collection_stats(args.db_path, args.collection)
        print(f"Total chunks stored: {stats['total']}")
        print(f"Strategies present:  {stats['by_strategy']}")
        print(f"Chunk size range:    {stats['min_chars']}-{stats['max_chars']} "
              f"(avg {stats['avg_chars']})")
        print("Chunks per source:")
        for source, count in stats["by_source"].items():
            print(f"  {count:>4}  {source}")
        return

    summary = ingest(
        docs_dir=args.docs_dir,
        db_path=args.db_path,
        collection_name=args.collection,
        strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        reset=args.reset,
    )

    print(f"Strategy:        {summary['strategy']}")
    print(f"Files loaded:    {summary['files']}")
    print(f"Chunks created:  {summary['chunks']}")
    print(f"Chunk sizes:     {summary['min_chars']}-{summary['max_chars']} chars "
          f"(avg {summary['avg_chars']})")
    print(f"Stored total:    {summary['stored_total']}")


if __name__ == "__main__":
    _main()