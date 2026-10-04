"""Search tests.

Both arms are driven with hand-made vectors and hand-made text, so ranking is
checked against an expected order rather than against whatever the embedder
happens to produce. What matters most here is that fusion genuinely fuses --
that a chunk both arms like beats a chunk only one arm likes -- because a bug
that quietly degenerates into single-arm search produces no error, only worse
results.
"""

from __future__ import annotations

import sqlite3
import sys
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_search import search, store  # noqa: E402
from code_search.chunker import Chunk  # noqa: E402
from code_search.corpus import FileState  # noqa: E402
from code_search.embedder import Embedder  # noqa: E402

DIM = 4


class FakeEmbedder(Embedder):
    """Returns a vector chosen by the test instead of calling Ollama."""

    def __init__(self, query_vector: list[float] | None = None) -> None:
        super().__init__("bge-m3")
        self.spec = self.spec.__class__("bge-m3", DIM)
        self.query_vector = query_vector or [1.0, 0.0, 0.0, 0.0]
        self.queries: list[str] = []

    def embed_query(self, text: str) -> array[float]:
        self.queries.append(text)
        return array("f", self.query_vector)


def add(conn: sqlite3.Connection, path: str, text: str, vector: list[float],
        symbol: str | None = None) -> None:
    chunk = Chunk(path=path, kind="source", start_line=1, end_line=1,
                  text=text, symbol=symbol)
    store.replace_file(conn, path, FileState(1, 2), "h", [chunk],
                       [array("f", vector)])


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = store.open_index(tmp_path / ".code-search" / "index.sqlite")
    store.stamp(connection, "bge-m3", DIM)
    add(connection, "src/router.py", "def compute_route(origin): pass",
        [1.0, 0.0, 0.0, 0.0], symbol="compute_route")
    add(connection, "src/loader.py", "def load_record(payload): pass",
        [0.0, 1.0, 0.0, 0.0], symbol="load_record")
    add(connection, "docs/guide.md", "Маршрутизация объясняется здесь",
        [0.0, 0.0, 1.0, 0.0])
    yield connection
    connection.close()


def paths(results: search.SearchResults) -> list[str]:
    return [hit.chunk.path for hit in results.hits]


# --------------------------------------------------------------------------- #
# query parsing
# --------------------------------------------------------------------------- #


def test_terms_are_or_ed_and_quoted() -> None:
    assert search.fts_expression("load record") == '"load" OR "record"'


def test_single_characters_are_dropped() -> None:
    """They match half the corpus and only add noise to BM25."""
    assert search.fts_expression("a b route") == '"route"'


@pytest.mark.parametrize("query", ["", "   ", "?!.", "x"])
def test_unusable_queries_yield_no_expression(query: str) -> None:
    assert search.fts_expression(query) == ""


def test_punctuation_cannot_reach_the_match_expression(conn: sqlite3.Connection) -> None:
    """FTS5 syntax characters in a query must not become FTS5 syntax."""
    for hostile in ['route" OR "', "route*", "NEAR(a b)", "route AND -x"]:
        search.lexical_rank(conn, hostile, 10)     # must not raise


def test_cyrillic_survives_tokenisation() -> None:
    assert search.fts_expression("разбор строки") == '"разбор" OR "строки"'


# --------------------------------------------------------------------------- #
# lexical arm
# --------------------------------------------------------------------------- #


def test_bm25_finds_the_exact_identifier(conn: sqlite3.Connection) -> None:
    found = store.load_chunks(conn, search.lexical_rank(conn, "load_record", 5))
    assert [c.path for c in found.values()] == ["src/loader.py"]


def test_unmatched_query_returns_nothing(conn: sqlite3.Connection) -> None:
    assert search.lexical_rank(conn, "quantum chromodynamics", 5) == []


def test_lexical_respects_the_path_glob(conn: sqlite3.Connection) -> None:
    assert search.lexical_rank(conn, "route", 5, "docs/*") == []
    assert search.lexical_rank(conn, "route", 5, "src/*") != []


# --------------------------------------------------------------------------- #
# dense arm
# --------------------------------------------------------------------------- #


def test_vectors_rank_by_cosine(conn: sqlite3.Connection) -> None:
    index = search.VectorIndex.load(conn)
    ranked = store.load_chunks(conn, index.rank(array("f", [0.0, 1.0, 0.0, 0.0]), 1))
    assert [c.path for c in ranked.values()] == ["src/loader.py"]


def test_vector_index_reads_its_dimension_from_meta(conn: sqlite3.Connection) -> None:
    index = search.VectorIndex.load(conn)
    assert (index.dimension, len(index)) == (DIM, 3)


def test_a_query_of_the_wrong_dimension_is_an_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError):
        search.VectorIndex.load(conn).rank(array("f", [1.0, 0.0]), 5)


def test_allowed_set_restricts_the_scan(conn: sqlite3.Connection) -> None:
    index = search.VectorIndex.load(conn)
    permitted = search.allowed_ids(conn, "docs/*")
    assert permitted is not None
    ranked = index.rank(array("f", [1.0, 0.0, 0.0, 0.0]), 5, permitted)
    assert list(store.load_chunks(conn, ranked).values())[0].path == "docs/guide.md"


def test_no_glob_means_no_filter(conn: sqlite3.Connection) -> None:
    assert search.allowed_ids(conn, None) is None


def test_empty_index_does_not_crash(tmp_path: Path) -> None:
    empty = store.open_index(tmp_path / ".code-search" / "index.sqlite")
    try:
        results = search.Searcher(empty, FakeEmbedder()).search("что угодно")
        assert results.hits == []
    finally:
        empty.close()


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #


def test_fusion_prefers_agreement_over_one_arms_favourite() -> None:
    """A chunk both arms rank highly must beat a chunk only one arm likes.

    This is the entire reason for combining the arms; if it fails, the merge
    has degenerated into whichever ranking happened to come first.
    """
    fused = search.reciprocal_rank_fusion([[1, 2], [2, 3]])
    assert [chunk_id for chunk_id, _ in fused][0] == 2


def test_fusion_scores_are_the_sum_over_arms() -> None:
    fused = dict(search.reciprocal_rank_fusion([[7], [7]]))
    assert fused[7] == pytest.approx(2 / (search.RRF_K + 1))


def test_fusion_is_deterministic_on_ties() -> None:
    assert search.reciprocal_rank_fusion([[5, 3]]) == search.reciprocal_rank_fusion([[5, 3]])


def test_fusion_of_nothing_is_empty() -> None:
    assert search.reciprocal_rank_fusion([[], []]) == []


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #


def test_hybrid_uses_both_arms(conn: sqlite3.Connection) -> None:
    embedder = FakeEmbedder([0.0, 0.0, 1.0, 0.0])       # points at the docs chunk
    results = search.Searcher(conn, embedder).search("load_record", k=3)
    top = results.hits[0]
    assert top.vector_rank is not None or top.lexical_rank is not None
    assert {"docs/guide.md", "src/loader.py"} <= set(paths(results))


def test_vector_mode_ignores_the_lexical_arm(conn: sqlite3.Connection) -> None:
    embedder = FakeEmbedder([0.0, 0.0, 1.0, 0.0])
    results = search.Searcher(conn, embedder).search("load_record", k=3, mode="vector")
    assert paths(results)[0] == "docs/guide.md"
    assert all(hit.lexical_rank is None for hit in results.hits)


def test_fusion_hears_only_the_lexical_head(conn: sqlite3.Connection) -> None:
    """BM25 below its first few ranks is noise, and RRF would give it a full vote.

    Measured on the dev set: feeding the whole BM25 pool into fusion cost 13
    points of Recall@10 on conceptual queries.
    """
    for i in range(6):
        add(conn, f"src/route_{i}.py", "route " * (10 - i), [0.0, 0.0, 0.0, 1.0])
    results = search.Searcher(conn, FakeEmbedder()).search("route", k=9)
    voted = [hit for hit in results.hits if hit.lexical_rank is not None]
    assert len(voted) <= search.LEXICAL_POOL
    assert all(hit.lexical_rank <= search.LEXICAL_POOL      # type: ignore[operator]
               for hit in voted)


def test_lexical_mode_keeps_its_full_depth(conn: sqlite3.Connection) -> None:
    """Capping the BM25-only mode would destroy it as an evaluation baseline."""
    for i in range(6):
        add(conn, f"src/route_{i}.py", "route " * (10 - i), [0.0, 0.0, 0.0, 1.0])
    results = search.Searcher(conn, FakeEmbedder()).search(
        "route", k=9, mode="lexical")
    assert len(results) > search.LEXICAL_POOL


def test_lexical_mode_needs_no_embedder(conn: sqlite3.Connection) -> None:
    """Ollama being down must not stop BM25 -- and must not be contacted at all."""
    searcher = search.Searcher(conn, embedder=None)
    results = searcher.search("load_record", k=3, mode="lexical")
    assert paths(results) == ["src/loader.py"]
    assert searcher._embedder is None


def test_the_query_reaches_the_embedder_once(conn: sqlite3.Connection) -> None:
    embedder = FakeEmbedder()
    search.Searcher(conn, embedder).search("разбор заголовка")
    assert embedder.queries == ["разбор заголовка"]


def test_path_glob_filters_both_arms(conn: sqlite3.Connection) -> None:
    embedder = FakeEmbedder([1.0, 0.0, 0.0, 0.0])      # points at src/router.py
    results = search.Searcher(conn, embedder).search("route", k=5, path_glob="docs/*")
    assert paths(results) == ["docs/guide.md"]


def test_k_bounds_the_result_count(conn: sqlite3.Connection) -> None:
    results = search.Searcher(conn, FakeEmbedder()).search("parse record", k=2)
    assert len(results) == 2


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_nothing(conn: sqlite3.Connection, k: int) -> None:
    assert search.Searcher(conn, FakeEmbedder()).search("route", k=k).hits == []


# --------------------------------------------------------------------------- #
# output budget
# --------------------------------------------------------------------------- #


def test_budget_turns_the_tail_into_refs_not_nothing(conn: sqlite3.Connection) -> None:
    results = search.Searcher(conn, FakeEmbedder()).search(
        "parse record разбор", k=3, max_tokens=1)
    assert len(results.hits) == 1
    # The tail is still there, ranked and addressable -- just without its text.
    assert len(results.refs) == 2
    assert all(ref.chunk.path for ref in results.refs)
    # `k` bounds the whole ranking, so that is what `len()` counts, and `ranked`
    # is the one list the budget cut in two, in its original order.
    assert len(results) == 3
    assert results.ranked == (*results.hits, *results.refs)


def test_the_first_hit_keeps_its_text_under_any_budget(conn: sqlite3.Connection) -> None:
    """An oversized single result is the caller's problem, not a bare reference."""
    results = search.Searcher(conn, FakeEmbedder()).search("route", k=3, max_tokens=0)
    assert len(results.hits) == 1
    assert len(results.refs) == len(results) - 1


def test_no_budget_keeps_everything(conn: sqlite3.Connection) -> None:
    results = search.Searcher(conn, FakeEmbedder()).search(
        "parse record разбор", k=3, max_tokens=None)
    assert (len(results), results.refs) == (3, ())
    assert results.tokens > 0


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #


def test_vectors_are_loaded_once(conn: sqlite3.Connection) -> None:
    searcher = search.Searcher(conn, FakeEmbedder())
    searcher.search("route")
    first = searcher.vectors
    searcher.search("route")
    assert searcher.vectors is first


def test_invalidate_reloads_after_a_reindex(conn: sqlite3.Connection) -> None:
    searcher = search.Searcher(conn, FakeEmbedder())
    assert len(searcher.vectors) == 3
    add(conn, "src/extra.py", "def extra(): pass", [0.0, 0.0, 0.0, 1.0])
    assert len(searcher.vectors) == 3, "the cache is deliberately stale until told"
    searcher.invalidate()
    assert len(searcher.vectors) == 4


def test_hits_carry_a_clickable_location(conn: sqlite3.Connection) -> None:
    hit = search.Searcher(conn, FakeEmbedder()).search("route", k=1).hits[0]
    assert hit.location == "src/router.py:1-1"
