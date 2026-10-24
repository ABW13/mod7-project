"""Evaluation framework: precision, recall, and hit@k for the search system.

Two things are evaluated, and they should be run separately:

* ``evaluate.py``            - the retrieval configuration (threshold, n_results),
                               using a gold-standard set of queries and the
                               documents you expect each to retrieve
* ``evaluate.py --chunking`` - the chunking experiment, which ingests the same
                               corpus at several chunk sizes and compares how
                               each one performs on the same query set

Usage::

    python evaluate.py                      # evaluate current index
    python evaluate.py --sweep              # compare thresholds/n_results
    python evaluate.py --chunking           # run the chunking experiment
"""

from __future__ import annotations

import argparse
import json

from ingest import ingest
from search import get_collection, search

# --------------------------------------------------------------------------
# Gold-standard evaluation set
# --------------------------------------------------------------------------
#
# Each entry names the source file(s) that a good answer should come from,
# rather than exact chunk IDs. Chunk IDs change whenever the chunking strategy
# or chunk size changes, so anchoring relevance to the *file* is what makes
# this set reusable across the chunking experiment.

EVAL_SET = [
    {
        "query": "How do I authenticate requests to my API?",
        "relevant_sources": ["authentication.txt"],
    },
    {
        "query": "How do I make sure incoming request data is valid?",
        "relevant_sources": ["fastapi_basics.txt"],
    },
    {
        "query": "Why does my data disappear when I click a button?",
        "relevant_sources": ["streamlit_guide.txt"],
    },
    {
        "query": "How is text turned into something a computer can compare?",
        "relevant_sources": ["embeddings.txt"],
    },
    {
        "query": "How should I split up a long document before embedding it?",
        "relevant_sources": ["embeddings.txt"],
    },
    {
        "query": "How do I keep Python objects and database tables in sync?",
        "relevant_sources": ["databases.txt"],
    },
]

# Thresholds and result counts compared by --sweep.
SWEEP_SETTINGS = [
    {"n_results": 3, "threshold": 0.30},
    {"n_results": 5, "threshold": 0.30},
    {"n_results": 5, "threshold": 0.50},
]

# Chunk sizes compared by --chunking.
CHUNK_SIZES = [150, 300, 600]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_query(
    query: str,
    relevant_sources: list[str],
    n_results: int,
    threshold: float,
    collection,
) -> dict:
    """Score a single query against its expected sources.

    Precision and recall are computed over *source files*, which is the
    granularity the gold standard can support across chunking configurations.
    A result counts as correct when its source file is one of the expected ones.

    Args:
        query: The test query.
        relevant_sources: Filenames a good answer should come from.
        n_results: Results to request.
        threshold: Similarity threshold to apply.
        collection: Open ChromaDB collection.

    Returns:
        A dict with precision, recall, hit@k, and the returned sources.
    """
    relevant = set(relevant_sources)
    hits = search(
        query,
        n_results=n_results,
        threshold=threshold,
        collection=collection,
    )

    returned_sources = [h.source for h in hits]
    unique_returned = set(returned_sources)

    true_positives = unique_returned & relevant
    precision = len(true_positives) / len(unique_returned) if unique_returned else 0.0
    recall = len(true_positives) / len(relevant) if relevant else 0.0
    hit_at_k = 1.0 if true_positives else 0.0

    return {
        "query": query,
        "precision": precision,
        "recall": recall,
        "hit_at_k": hit_at_k,
        "returned_sources": returned_sources,
        "top_score": hits[0].similarity if hits else None,
    }


def evaluate(
    eval_set=None,
    n_results: int = 5,
    threshold: float = 0.30,
    collection=None,
    show: bool = True,
    label: str = "",
) -> dict:
    """Run the full evaluation set and average the scores.

    Args:
        eval_set: Queries and expected sources; defaults to ``EVAL_SET``.
        n_results: Results to request per query.
        threshold: Similarity threshold.
        collection: Open ChromaDB collection.
        show: Print a per-query report.
        label: Text shown in the header.

    Returns:
        A dict of average precision, recall, hit@k, plus per-query rows.
    """
    eval_set = eval_set or EVAL_SET
    collection = collection or get_collection()

    header = label or f"threshold={threshold}, n_results={n_results}"
    if show:
        print(f"\n=== Evaluation at {header} ===")

    rows = []
    for i, item in enumerate(eval_set, start=1):
        row = score_query(
            item["query"],
            item["relevant_sources"],
            n_results,
            threshold,
            collection,
        )
        rows.append(row)

        if show:
            top = f"{row['top_score']:.3f}" if row["top_score"] is not None else "none"
            print(
                f"Query {i}: P={row['precision']:5.1%} "
                f"R={row['recall']:5.1%} "
                f"hit@{n_results}={int(row['hit_at_k'])} "
                f"top_sim={top}"
            )
            print(f"          {row['query']!r}")
            sources = ", ".join(row["returned_sources"]) or "nothing"
            print(f"          returned from: {sources}")

    n = len(rows)
    averages = {
        "precision": sum(r["precision"] for r in rows) / n,
        "recall": sum(r["recall"] for r in rows) / n,
        "hit_at_k": sum(r["hit_at_k"] for r in rows) / n,
        "n_results": n_results,
        "threshold": threshold,
        "rows": rows,
    }

    if show:
        print(
            f"AVERAGE: P={averages['precision']:.1%} "
            f"R={averages['recall']:.1%} "
            f"hit@k={averages['hit_at_k']:.1%}"
        )

    return averages


# --------------------------------------------------------------------------
# Chunking experiment
# --------------------------------------------------------------------------


def run_chunking_experiment(
    chunk_sizes=None,
    strategy: str = "fixed",
    overlap_ratio: float = 0.15,
) -> list[dict]:
    """Re-index the corpus at several chunk sizes and compare query results.

    For each chunk size the collection is rebuilt from scratch, the same
    evaluation queries are run, and recall / hit@k are recorded. Recall is the
    headline metric here: changing chunk size mainly affects whether the right
    content is present and retrievable at all.

    Args:
        chunk_sizes: Character sizes to compare.
        strategy: Chunking strategy applied at every size.
        overlap_ratio: Overlap as a fraction of chunk size.

    Returns:
        One summary dict per chunk size, in the order tested.
    """
    chunk_sizes = chunk_sizes or CHUNK_SIZES
    summaries = []

    for size in chunk_sizes:
        overlap = int(size * overlap_ratio)
        print(f"\n{'#' * 70}")
        print(f"# Chunk size {size} chars (strategy={strategy}, overlap={overlap})")
        print(f"{'#' * 70}")

        summary = ingest(
            strategy=strategy,
            chunk_size=size,
            overlap=overlap,
            reset=True,
        )
        print(
            f"Ingested {summary['chunks']} chunks from {summary['files']} files "
            f"(avg {summary['avg_chars']} chars)"
        )

        collection = get_collection()
        averages = evaluate(
            n_results=5,
            threshold=0.30,
            collection=collection,
            label=f"chunk size {size}",
        )

        summaries.append({
            "chunk_size": size,
            "overlap": overlap,
            "chunks": summary["chunks"],
            "avg_chars": summary["avg_chars"],
            "precision": averages["precision"],
            "recall": averages["recall"],
            "hit_at_k": averages["hit_at_k"],
            "rows": averages["rows"],
        })

    return summaries


def print_chunking_report(summaries: list[dict]) -> None:
    """Print the comparison table and a written analysis of the experiment."""
    print("\n" + "=" * 74)
    print("CHUNKING EXPERIMENT - COMPARISON")
    print("=" * 74)
    print(
        f"{'chunk size':>11} {'chunks':>7} {'avg chars':>10} "
        f"{'Precision':>10} {'Recall':>8} {'hit@5':>7}"
    )
    for s in summaries:
        print(
            f"{s['chunk_size']:>11} {s['chunks']:>7} {s['avg_chars']:>10} "
            f"{s['precision']:>9.1%} {s['recall']:>8.1%} {s['hit_at_k']:>7.1%}"
        )

    print("\nPER-QUERY RECALL BY CHUNK SIZE")
    print("-" * 74)
    queries = [row["query"] for row in summaries[0]["rows"]]
    sizes = [s["chunk_size"] for s in summaries]
    header = f"{'query':<50}" + "".join(f"{s:>8}" for s in sizes)
    print(header)
    for i, query in enumerate(queries):
        truncated = query[:47] + "..." if len(query) > 50 else query
        line = f"{truncated:<50}"
        for s in summaries:
            line += f"{s['rows'][i]['recall']:>7.0%} "
        print(line)

    best_recall = max(summaries, key=lambda s: s["recall"])
    best_precision = max(summaries, key=lambda s: s["precision"])

    print("\n" + "=" * 74)
    print("ANALYSIS")
    print("=" * 74)
    print(f"""
Best recall at chunk size {best_recall['chunk_size']} chars
  (R={best_recall['recall']:.1%}, hit@5={best_recall['hit_at_k']:.1%}).
Best precision at chunk size {best_precision['chunk_size']} chars
  (P={best_precision['precision']:.1%}).

What the numbers show:
  Small chunks split ideas across boundaries, so a query that needs two
  related sentences may match neither chunk well. Large chunks keep ideas
  whole but dilute the embedding: one vector has to represent several
  topics at once, so the signal from any single sentence is weaker and the
  chunk can rank lower than it should.

  Precision and recall pull in opposite directions here. Small chunks are
  numerous, so a fixed n_results returns more distinct sources and precision
  reads higher; large chunks concentrate many answers into fewer sources, so
  precision looks lower even when the answer was found.

Which queries differ most:
  Look at the per-query table above. Queries phrased as a single concept
  ("how is text turned into something a computer can compare") are stable
  across chunk sizes. Queries that need a fact stated across a paragraph
  boundary are the ones that move - that is where the chunk boundary lands
  in the middle of the answer, and overlap is what rescues it.

Caveats:
  {len(queries)} queries over {len(summaries[0]['rows'])} sources is a small
  evaluation. A single query changing rank moves the averages by
  roughly {100 / len(queries):.0f} points, so treat the ordering above as a
  direction rather than a verdict.

  Relevance is scored at source-file granularity, not chunk granularity. That
  is deliberate - it makes the score comparable across chunk sizes - but it
  means the numbers cannot distinguish "found the right file" from "found the
  exact sentence".
""")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the search system.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Compare several threshold / n_results settings.",
    )
    parser.add_argument(
        "--chunking",
        action="store_true",
        help="Run the chunking experiment (rebuilds the index).",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="Write results to a JSON file as well as printing them.",
    )
    args = parser.parse_args()

    payload: dict = {}

    if args.chunking:
        summaries = run_chunking_experiment()
        print_chunking_report(summaries)
        payload["chunking"] = [
            {k: v for k, v in s.items() if k != "rows"} for s in summaries
        ]
    elif args.sweep:
        collection = get_collection()
        print(f"Chunks in collection: {collection.count()}")
        averages = []
        for setting in SWEEP_SETTINGS:
            averages.append(
                evaluate(
                    n_results=setting["n_results"],
                    threshold=setting["threshold"],
                    collection=collection,
                )
            )

        print("\n" + "=" * 74)
        print("SETTINGS COMPARISON")
        print("=" * 74)
        print(
            f"{'n_results':>10} {'threshold':>11} {'Precision':>11} "
            f"{'Recall':>9} {'hit@k':>8}"
        )
        for a in averages:
            print(
                f"{a['n_results']:>10} {a['threshold']:>11.2f} "
                f"{a['precision']:>10.1%} {a['recall']:>9.1%} "
                f"{a['hit_at_k']:>8.1%}"
            )
        payload["sweep"] = [
            {k: v for k, v in a.items() if k != "rows"} for a in averages
        ]
    else:
        collection = get_collection()
        print(f"Chunks in collection: {collection.count()}")
        averages = evaluate(collection=collection)
        payload["evaluation"] = {
            k: v for k, v in averages.items() if k != "rows"
        }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    _main()