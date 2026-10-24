"""Run the chunking experiment and record results for the README.

This is a convenience script that mirrors ``evaluate.py --chunking`` but writes
the output to a text file you can paste into your write-up.

Usage::

    python chunking_experiment.py
"""

from evaluate import print_chunking_report, run_chunking_experiment

if __name__ == "__main__":
    summaries = run_chunking_experiment()

    print_chunking_report(summaries)

    with open("chunking_results.txt", "w", encoding="utf-8") as handle:
        handle.write("CHUNKING EXPERIMENT RESULTS\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(
            f"{'chunk size':>11} {'chunks':>7} {'Precision':>10} "
            f"{'Recall':>8} {'hit@5':>7}\n"
        )
        for s in summaries:
            handle.write(
                f"{s['chunk_size']:>11} {s['chunks']:>7} "
                f"{s['precision']:>9.1%} {s['recall']:>8.1%} "
                f"{s['hit_at_k']:>7.1%}\n"
            )
        handle.write("\n\nPER-QUERY RECALL\n")
        for row in summaries[0]["rows"]:
            line = f"{row['query']}\n"
            for s in summaries:
                match = next(
                    r for r in s["rows"] if r["query"] == row["query"]
                )
                line += (
                    f"    {s['chunk_size']:>4} chars: "
                    f"recall={match['recall']:.0%} "
                    f"top_sim="
                    f"{match['top_score']:.3f}\n"
                    if match["top_score"] is not None
                    else f"    {s['chunk_size']:>4} chars: no results\n"
                )
            handle.write(line + "\n")

    print("\nResults also written to chunking_results.txt")