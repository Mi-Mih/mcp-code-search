"""Hybrid search over the index: dense vectors, BM25, and reciprocal rank fusion.

Two retrievers with different failure modes are combined. The dense arm finds
code that *means* what was asked without sharing any words with the query; the
lexical arm finds the exact identifier the dense arm blurred away. Each is bad
where the other is good, so neither is trusted alone.

They are merged with reciprocal rank fusion -- `1 / (60 + rank)` summed across
arms -- rather than by mixing their scores. Cosine similarity and BM25 live on
incomparable scales (BM25 is unbounded and corpus-dependent, cosine is bounded
and here always positive), and any weighted sum of the two needs a normalisation
constant that has to be retuned whenever the corpus changes. Ranks need nothing.

The dense arm is a full scan in pure Python. Measured on the reference corpus
(3044 chunks x 1024 dims): 20 ms to load, 130 ms per query. Ollama spends more
than that embedding the query itself, so an ANN index would optimise the wrong
half, and numpy -- 40x faster here -- would buy ~125 ms at the cost of a
dependency the system interpreter does not have. Revisit above ~30k chunks.
"""

from __future__ import annotations

import argparse
import operator
import re
import sqlite3
import sys
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from code_search import store
from code_search.chunker import estimate_tokens
from code_search.embedder import DEFAULT_MODEL, Embedder
from code_search.store import StoredChunk

__all__ = [
    "DEFAULT_K",
    "Hit",
    "MAX_RESULT_TOKENS",
    "Mode",
    "SearchResults",
    "Searcher",
    "VectorIndex",
]

Mode = Literal["hybrid", "vector", "lexical"]
MODES: tuple[Mode, ...] = ("hybrid", "vector", "lexical")

DEFAULT_K = 15

# The RRF constant. 60 is the value from the original paper and the one every
# published comparison uses; it damps the top ranks so that a single arm's
# first place cannot by itself outvote agreement between both arms.
RRF_K = 60

# How deep the dense arm goes before fusion. Fusion can only promote what an arm
# retrieved, so the pool has to be several times k.
CANDIDATE_FACTOR = 4
MIN_CANDIDATES = 50

# How deep the lexical arm goes *into a fusion*. Far shallower, and measured:
# on the dev set, BM25 below its first few ranks is noise on exactly the queries
# where the dense arm is strong, and RRF gives that noise a full vote. Feeding
# the whole BM25 pool into fusion cost 13 points of Recall@10 on conceptual
# queries; anything from 1 to 4 recovered them, and the plateau is flat in the
# RRF constant too. 3 sits inside that plateau rather than at its edge.
#
# BM25's head still earns its place: the one dev query the dense arm missed
# outright (rank 106) is BM25's rank 1, and fusion puts it back at rank 2.
#
# This also makes the dense pool depth harmless: extra dense candidates enter at
# scores below 1/(RRF_K + 51), under anything the lexical head contributes, so
# they only ever extend the tail. Measuring at a deeper k cannot move the top.
LEXICAL_POOL = 3

# Budget for *full-text* results, not a ceiling on the result set. Hits beyond
# it are still returned, but as references (path, lines, symbol) without their
# text: the caller keeps all k for discoverability and reads the tail on demand.
# 3000 was the original context-budget target; the measured k=15 median of full
# text was ~5.4k, so this typically returns the top ~8 hits with text.
MAX_RESULT_TOKENS = 3000

# Query terms. Splitting on anything else keeps the expression free of FTS5
# syntax characters, so a query can never be a malformed MATCH expression.
_TERM = re.compile(r"[0-9A-Za-z_Ѐ-ӿ]+")


@dataclass(frozen=True)
class Hit:
    """One result, with the evidence for its position."""

    chunk: StoredChunk
    score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None

    @property
    def location(self) -> str:
        """Return the repository-relative, 1-based inclusive line range."""
        return f"{self.chunk.path}:{self.chunk.start_line}-{self.chunk.end_line}"


@dataclass(frozen=True)
class SearchResults:
    """Everything the ranking produced, split by whether the text came along.

    `hits` and `refs` are one ranked list cut in two by the token budget, not two
    kinds of result: a hit in `refs` was ranked exactly as it would have been in
    `hits`, it simply did not fit. So `len()` counts both -- it is the number
    the caller's `k` bounds -- and `ranked` is the list in its original order.
    """

    hits: list[Hit]
    """Hits returned with their text, best first."""
    mode: Mode
    tokens: int = 0
    """Estimated tokens of the text in `hits`."""
    refs: tuple[Hit, ...] = ()
    """Hits past the token budget: still ranked, returned without their text."""

    @property
    def ranked(self) -> tuple[Hit, ...]:
        """Return full-text and reference-only results in original rank order."""
        return (*self.hits, *self.refs)

    def __len__(self) -> int:
        return len(self.hits) + len(self.refs)


# --------------------------------------------------------------------------- #
# dense arm
# --------------------------------------------------------------------------- #


class VectorIndex:
    """Every vector in the index, held in one contiguous buffer.

    Loading is the reason this is a class: pulling 12 MB of blobs out of SQLite
    per query would cost more than the arithmetic does. The buffer is a single
    `array('f')` so that scoring is a memoryview slice per chunk with no copying
    and no per-vector Python object.
    """

    def __init__(self, ids: list[int], buffer: array[float], dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if len(buffer) != len(ids) * dimension:
            raise ValueError(
                f"{len(ids)} vectors x {dimension} dims != {len(buffer)} floats"
            )
        self.ids = ids
        self.dimension = dimension
        self._buffer = buffer
        self._view = memoryview(buffer)

    @classmethod
    def load(cls, conn: sqlite3.Connection) -> VectorIndex:
        """Load all stored vectors into one contiguous in-memory index."""
        dimension = int(store.read_meta(conn).get("embedding_dimension", "0"))
        ids: list[int] = []
        buffer = array("f")
        for chunk_id, blob in store.iter_vectors(conn):
            ids.append(chunk_id)
            buffer.frombytes(blob)
        if not dimension:
            dimension = len(buffer) // len(ids) if ids else 1
        return cls(ids, buffer, dimension)

    def __len__(self) -> int:
        return len(self.ids)

    def rank(self, query: Sequence[float], limit: int,
             allowed: set[int] | None = None) -> list[int]:
        """Chunk ids by descending cosine similarity to `query`.

        Both sides are L2-normalised by the embedder, so the dot product *is*
        the cosine and no division is needed.
        """
        if not self.ids:
            # An empty index matches nothing, and its dimension is unknowable --
            # checking it here would turn "nothing indexed yet" into an error.
            return []
        if len(query) != self.dimension:
            raise ValueError(
                f"query has {len(query)} dims, index has {self.dimension}"
            )
        dim, view, mul = self.dimension, self._view, operator.mul
        scored: list[tuple[float, int]] = []
        for position, chunk_id in enumerate(self.ids):
            if allowed is not None and chunk_id not in allowed:
                continue
            offset = position * dim
            scored.append((sum(map(mul, view[offset:offset + dim], query)), chunk_id))
        # Ascending id as the tie-break: equal scores must not reorder between
        # runs, or the eval harness measures noise.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [chunk_id for _, chunk_id in scored[:limit]]


# --------------------------------------------------------------------------- #
# lexical arm
# --------------------------------------------------------------------------- #


def fts_expression(query: str) -> str:
    """Turn free text into an FTS5 MATCH expression, or "" if nothing is usable.

    Terms are OR-ed, not AND-ed. A natural-language query carries words that
    appear nowhere in the code ("как", "where", "logic"); requiring all of them
    returns nothing, and BM25 already handles the ranking -- a chunk matching
    four terms outranks one matching a single common term.
    """
    terms = {term for term in _TERM.findall(query) if len(term) > 1}
    return " OR ".join(f'"{term}"' for term in sorted(terms))


_LEXICAL_SQL = """
SELECT c.id AS id
FROM chunks_fts
JOIN chunks c ON c.id = chunks_fts.rowid
JOIN files f ON f.id = c.file_id
WHERE chunks_fts MATCH :expr
  AND (:glob IS NULL OR f.path GLOB :glob)
ORDER BY bm25(chunks_fts)
LIMIT :limit
"""


def lexical_rank(conn: sqlite3.Connection, query: str, limit: int,
                 path_glob: str | None = None) -> list[int]:
    """Chunk ids by BM25 relevance. Lower bm25() is better, hence plain ORDER BY."""
    expression = fts_expression(query)
    if not expression:
        return []
    rows = conn.execute(
        _LEXICAL_SQL, {"expr": expression, "glob": path_glob, "limit": limit}
    ).fetchall()
    return [int(row["id"]) for row in rows]


def allowed_ids(conn: sqlite3.Connection, path_glob: str | None) -> set[int] | None:
    """Chunk ids under `path_glob`, or None when the whole index is in scope.

    The dense arm cannot push the filter into SQL, so the permitted set is
    materialised once per query. It is one integer column over a few thousand
    rows -- cheaper than any of the arithmetic that follows.
    """
    if path_glob is None:
        return None
    rows = conn.execute(
        "SELECT c.id AS id FROM chunks c JOIN files f ON f.id = c.file_id "
        "WHERE f.path GLOB ?",
        (path_glob,),
    ).fetchall()
    return {int(row["id"]) for row in rows}


# --------------------------------------------------------------------------- #
# fusion
# --------------------------------------------------------------------------- #


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]],
                           k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge ranked id lists into one, scored by the sum of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def _positions(ranking: Sequence[int]) -> dict[int, int]:
    return {chunk_id: position for position, chunk_id in enumerate(ranking, start=1)}


# --------------------------------------------------------------------------- #
# the searcher
# --------------------------------------------------------------------------- #


class Searcher:
    """A live index plus the two retrievers over it.

    Held open across queries by the MCP server: the vector buffer and the
    warm Ollama model are both worth keeping, and re-creating either per call
    would dominate the latency.
    """

    def __init__(self, conn: sqlite3.Connection, embedder: Embedder | None = None,
                 model: str = DEFAULT_MODEL) -> None:
        self.conn = conn
        self.model = model
        self._embedder = embedder
        self._vectors: VectorIndex | None = None

    @property
    def embedder(self) -> Embedder:
        """Built on first use, so that lexical-only work needs no Ollama."""
        if self._embedder is None:
            self._embedder = Embedder(self.model)
        return self._embedder

    @property
    def vectors(self) -> VectorIndex:
        """Load and cache the dense index on first vector search."""
        if self._vectors is None:
            self._vectors = VectorIndex.load(self.conn)
        return self._vectors

    def invalidate(self) -> None:
        """Forget the cached vectors after the index has been rewritten."""
        self._vectors = None

    def search(self, query: str, k: int = DEFAULT_K, path_glob: str | None = None,
               mode: Mode = "hybrid",
               max_tokens: int | None = MAX_RESULT_TOKENS) -> SearchResults:
        """Rank chunks for ``query`` and apply the full-text token budget.

        ``mode`` selects lexical, vector, or fused retrieval. ``path_glob`` is
        matched against repository-relative paths. A ``None`` token budget keeps
        full text for every returned hit.
        """
        if k <= 0:
            return SearchResults(hits=[], mode=mode)

        pool = max(MIN_CANDIDATES, k * CANDIDATE_FACTOR)
        dense: list[int] = []
        lexical: list[int] = []

        if mode in ("hybrid", "vector"):
            dense = self.vectors.rank(
                self.embedder.embed_query(query), pool, allowed_ids(self.conn, path_glob)
            )
        if mode in ("hybrid", "lexical"):
            # Full depth when BM25 *is* the answer; only its head when it is a
            # vote inside fusion. Cutting the lexical-only mode down to three
            # results would also destroy it as an evaluation baseline.
            depth = LEXICAL_POOL if mode == "hybrid" else pool
            lexical = lexical_rank(self.conn, query, depth, path_glob)

        if mode == "vector":
            ordered = [(chunk_id, 1.0 / (RRF_K + i))
                       for i, chunk_id in enumerate(dense, start=1)]
        elif mode == "lexical":
            ordered = [(chunk_id, 1.0 / (RRF_K + i))
                       for i, chunk_id in enumerate(lexical, start=1)]
        else:
            ordered = reciprocal_rank_fusion([dense, lexical])

        top = ordered[:k]
        chunks = store.load_chunks(self.conn, [chunk_id for chunk_id, _ in top])
        dense_at, lexical_at = _positions(dense), _positions(lexical)
        hits = [
            Hit(chunk=chunks[chunk_id], score=score,
                vector_rank=dense_at.get(chunk_id), lexical_rank=lexical_at.get(chunk_id))
            for chunk_id, score in top
            # A concurrent reindex can delete a row out from under a ranking
            # computed on the buffer from before it. Dropping it is the whole
            # correction needed, because `chunks.id` is AUTOINCREMENT and an id
            # is never handed to a different chunk -- so a surviving id is still
            # the chunk that was scored.
            if chunk_id in chunks
        ]
        return _split_by_budget(hits, mode, max_tokens)


def _split_by_budget(hits: list[Hit], mode: Mode, max_tokens: int | None) -> SearchResults:
    """Split hits into full-text head and reference-only tail at the token budget.

    The first hit always keeps its text: a single oversized chunk should be
    returned and let the caller decide, not turned into a bare reference. The
    tail is not discarded -- every ranked hit stays visible, just without the
    text that would not fit the budget.
    """
    if max_tokens is None:
        return SearchResults(hits, mode, sum(estimate_tokens(h.chunk.text) for h in hits))

    kept: list[Hit] = []
    used = 0
    for hit in hits:
        cost = estimate_tokens(hit.chunk.text)
        if kept and used + cost > max_tokens:
            break
        kept.append(hit)
        used += cost
    return SearchResults(kept, mode, used, refs=tuple(hits[len(kept):]))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic search CLI and return a process exit code."""
    from code_search.indexer import index_path

    parser = argparse.ArgumentParser(
        prog="python -m code_search.search",
        description="Поиск по индексу (для ручной проверки; рабочий вход — MCP).",
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument("--path-glob", default=None)
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--full", action="store_true", help="печатать текст чанков")
    args = parser.parse_args(argv)

    db_path = index_path(args.root.resolve())
    if not db_path.exists():
        print(f"индекс не найден: {db_path}", file=sys.stderr)
        return 2

    conn = store.open_index(db_path)
    try:
        started = time.perf_counter()
        results = Searcher(conn).search(args.query, args.k, args.path_glob, args.mode)
        elapsed = time.perf_counter() - started
    finally:
        conn.close()

    for position, hit in enumerate(results.hits, start=1):
        symbol = f"  {hit.chunk.symbol}" if hit.chunk.symbol else ""
        print(f"{position:2}. {hit.location}{symbol}   "
              f"rrf={hit.score:.4f} v={hit.vector_rank} b={hit.lexical_rank}")
        if args.full:
            print(hit.chunk.text)
            print("-" * 70)
    if results.refs:
        print("без текста (за бюджетом):")
        for position, hit in enumerate(results.refs, start=len(results.hits) + 1):
            symbol = f"  {hit.chunk.symbol}" if hit.chunk.symbol else ""
            print(f"{position:2}. {hit.location}{symbol}")
    tail = f", {len(results.refs)} без текста" if results.refs else ""
    print(f"\n{len(results)} результат(ов): {len(results.hits)} с текстом "
          f"(~{results.tokens} токенов){tail}, {elapsed * 1000:.0f} мс")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
