"""Streamlit search interface for the semantic search project.

Thin layer over ``ingest.py`` and ``search.py``: all the loading, chunking,
embedding, and ranking lives in those modules so the logic can be tested and
evaluated without a browser.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from ingest import (
    DEFAULT_DB_PATH,
    STRATEGIES,
    collection_stats,
    ingest,
    list_sources,
)
from search import (
    DEFAULT_N_RESULTS,
    DEFAULT_THRESHOLD,
    count_in_scope,
    get_collection,
    search,
)

DOCS_DIR = "docs"

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------


@st.cache_resource
def load_collection():
    """Open the ChromaDB collection once and reuse it across re-runs.

    Streamlit re-executes the whole script on every interaction, so anything
    expensive is cached. The collection handle must be cached; the data inside
    it is not cached and always reflects the current database.
    """
    return get_collection(DEFAULT_DB_PATH)


collection = load_collection()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


def init_state() -> None:
    """Initialise session keys so widgets have stable values across re-runs."""
    st.session_state.setdefault("query_text", "")


init_state()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("📁 Document Manager")

    available_sources = list_sources(DOCS_DIR)
    stats = collection_stats(DEFAULT_DB_PATH)

    # --- Metrics -----------------------------------------------------
    col_a, col_b = st.columns(2)
    col_a.metric("Chunks in DB", stats["total"])
    col_b.metric("Source files", len(available_sources))

    if stats["total"]:
        st.caption(
            f"Chunk sizes {stats['min_chars']}–{stats['max_chars']} chars "
            f"(avg {stats['avg_chars']})"
        )
        strategies = ", ".join(
            f"{name} ({count})" for name, count in stats["by_strategy"].items()
        )
        st.caption(f"Strategies: {strategies}")

    st.divider()

    # --- Re-indexing -------------------------------------------------
    st.subheader("Re-index")

    strategy_name = st.selectbox(
        "Chunking strategy",
        options=sorted(STRATEGIES),
        index=0,
        help="Fixed splits on a character count; paragraph keeps whole paragraphs.",
    )

    chunk_size = st.number_input(
        "Chunk size (chars)",
        min_value=50,
        max_value=2000,
        value=300,
        step=50,
        disabled=(strategy_name != "fixed"),
        help="Used by the fixed strategy only.",
    )

    overlap = st.number_input(
        "Overlap (chars)",
        min_value=0,
        max_value=1000,
        value=50,
        step=10,
        disabled=(strategy_name != "fixed"),
        help="Must be smaller than the chunk size.",
    )

    if st.button("🔄 Re-index Documents", use_container_width=True):
        if not available_sources:
            st.warning(f"No .txt or .md files found in {DOCS_DIR}/")
        elif strategy_name == "fixed" and overlap >= chunk_size:
            st.error("Overlap must be smaller than the chunk size.")
        else:
            with st.spinner("Loading, chunking, and embedding..."):
                try:
                    summary = ingest(
                        docs_dir=DOCS_DIR,
                        db_path=DEFAULT_DB_PATH,
                        strategy=strategy_name,
                        chunk_size=int(chunk_size),
                        overlap=int(overlap),
                        reset=True,
                    )
                    st.success(
                        f"Indexed {summary['chunks']} chunks from "
                        f"{summary['files']} files "
                        f"({summary['min_chars']}–{summary['max_chars']} chars)"
                    )
                    # The cached collection handle is still valid; the data
                    # behind it changed, so the page re-runs to refresh metrics.
                    st.rerun()
                except (ValueError, FileNotFoundError) as exc:
                    st.error(str(exc))

    st.divider()

    # --- Search controls ---------------------------------------------
    st.subheader("Search settings")

    n_results = st.slider(
        "Results to show",
        min_value=1,
        max_value=15,
        value=DEFAULT_N_RESULTS,
    )

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Results below this cosine similarity are dropped.",
    )

    selected_sources = st.multiselect(
        "Search only in these files",
        options=available_sources,
        default=[],
        help="Leave empty to search every file.",
    )


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

st.title("🔍 Semantic Search")
st.write(
    "Search the course documents by meaning rather than by keyword. "
    "Results are ranked by cosine similarity between your query and each chunk."
)

if collection.count() == 0:
    st.info(
        "👈 Click **Re-index Documents** in the sidebar to load and embed "
        "your documents first."
    )
    st.stop()

query = st.text_input(
    "Enter your search query",
    value=st.session_state.query_text,
    placeholder="How does authentication work?",
)

# Keep the widget value in session state so the "Similar to this" button can
# drive a new search on the next run.
st.session_state.query_text = query

if not query.strip():
    st.caption("Type a question above to search the indexed documents.")
    st.stop()

scope_total = count_in_scope(selected_sources, collection=collection)
results = search(
    query,
    n_results=n_results,
    threshold=threshold,
    sources=selected_sources,
    collection=collection,
)

scope_note = (
    f" (filtered to {len(selected_sources)} file(s))" if selected_sources else ""
)
st.caption(
    f"Showing {len(results)} of {scope_total} indexed chunks{scope_note} · "
    f"threshold {threshold:.2f}"
)

if not results:
    st.warning(
        "No chunks scored above the similarity threshold. "
        "Try lowering the threshold in the sidebar, or rephrasing the query."
    )
    st.stop()

BADGES = {
    "high": ("🟢 High", "#1a7f37"),
    "medium": ("🟡 Medium", "#9a6700"),
    "low": ("🔴 Low", "#a40e26"),
}

for rank, result in enumerate(results, start=1):
    label, colour = BADGES[result.relevance]

    with st.container(border=True):
        meta_col, score_col = st.columns([3, 1])

        with meta_col:
            st.markdown(
                f"**{rank}. {result.source}** · chunk {result.chunk_index} · "
                f"`{result.strategy}`"
            )

        with score_col:
            st.markdown(
                f"<div style='text-align:right'>"
                f"<span style='background:{colour};color:white;"
                f"padding:3px 10px;border-radius:8px;font-size:0.8em;'>"
                f"{label} · {result.similarity:.3f}</span></div>",
                unsafe_allow_html=True,
            )

        preview = " ".join(result.text.split())
        if len(preview) > 150:
            preview = preview[:150].rstrip() + "…"
        st.write(preview)

        with st.expander("Show full chunk"):
            st.write(result.text)
            st.caption(
                f"distance {result.distance:.4f} · "
                f"similarity {result.similarity:.4f} · "
                f"id `{result.chunk_id}`"
            )

        if st.button("🔎 Similar to this", key=f"similar_{rank}_{result.chunk_id}"):
            st.session_state.query_text = result.text
            st.rerun()