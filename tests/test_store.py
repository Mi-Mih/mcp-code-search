"""Storage tests.

The invariant worth most here is that `chunks` and `chunks_fts` never drift
apart. Drift produces no error: BM25 simply ranks rows that no longer exist, or
misses rows that do, and the only symptom is search quality quietly degrading.
Every mutation path is therefore checked against the FTS row count.
"""

from __future__ import annotations

import sqlite3
import sys
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_search import store  # noqa: E402
from code_search.chunker import CHUNKER_VERSION, Chunk  # noqa: E402
from code_search.corpus import FileState  # noqa: E402

DIM = 4


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    connection = store.open_index(tmp_path / ".code-search" / "index.sqlite")
    yield connection
    connection.close()


def make_chunk(path: str, start: int, text: str, symbol: str | None = None) -> Chunk:
    return Chunk(path=path, kind="source", start_line=start, end_line=start,
                 text=text, symbol=symbol)


def vec(seed: float) -> array[float]:
    return array("f", [seed] * DIM)


def fts_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])


def write(conn: sqlite3.Connection, path: str, chunks: list[Chunk],
          file_hash: str = "h0") -> None:
    store.replace_file(conn, path, FileState(1, 2), file_hash, chunks,
                       [vec(float(i)) for i in range(len(chunks))])


# --------------------------------------------------------------------------- #
# FTS consistency
# --------------------------------------------------------------------------- #


def test_insert_populates_fts(conn: sqlite3.Connection) -> None:
    write(conn, "a.py", [make_chunk("a.py", 1, "def alpha(): pass")])
    assert store.stats(conn) == (1, 1)
    assert fts_count(conn) == 1


def test_replacing_a_file_leaves_no_orphan_fts_rows(conn: sqlite3.Connection) -> None:
    write(conn, "a.py", [make_chunk("a.py", i, f"line {i}") for i in range(1, 6)])
    write(conn, "a.py", [make_chunk("a.py", 1, "only one now")])
    assert store.stats(conn) == (1, 1)
    assert fts_count(conn) == 1


def test_dropping_a_file_clears_its_fts_rows(conn: sqlite3.Connection) -> None:
    write(conn, "a.py", [make_chunk("a.py", 1, "alpha")])
    write(conn, "b.py", [make_chunk("b.py", 1, "beta")])
    store.drop_files(conn, ["a.py"])
    assert store.stats(conn) == (1, 1)
    assert fts_count(conn) == 1
    hits = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'alpha'")
    assert hits.fetchall() == []


def test_fts_matches_return_chunk_ids(conn: sqlite3.Connection) -> None:
    write(conn, "a.py", [make_chunk("a.py", 1, "def calculate_deficit(): pass")])
    row = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'deficit' "
        "ORDER BY bm25(chunks_fts)"
    ).fetchone()
    assert row is not None
    loaded = store.load_chunks(conn, [row["rowid"]])
    assert loaded[row["rowid"]].path == "a.py"


def test_compound_identifier_is_reachable_by_its_parts(conn: sqlite3.Connection) -> None:
    write(conn, "a.py", [make_chunk("a.py", 1, "def setDeficitMode(): pass")])
    for term in ("setDeficitMode", "deficit", "mode"):
        found = conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", (f'"{term}"',)
        ).fetchall()
        assert found, f"{term!r} did not match"


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


def test_replace_is_atomic_on_mismatch(conn: sqlite3.Connection) -> None:
    chunks = [make_chunk("a.py", 1, "x")]
    with pytest.raises(ValueError):
        store.replace_file(conn, "a.py", FileState(1, 2), "h", chunks, [])
    assert store.stats(conn) == (0, 0)


def test_indexed_files_round_trip(conn: sqlite3.Connection) -> None:
    write(conn, "a.py", [make_chunk("a.py", 1, "x")], file_hash="abc123")
    assert store.indexed_files(conn) == {"a.py": (FileState(1, 2), "abc123")}


def test_file_with_no_chunks_is_still_recorded(conn: sqlite3.Connection) -> None:
    """An empty file must be remembered, or it is re-read on every run."""
    store.replace_file(conn, "empty.py", FileState(1, 2), "h", [], [])
    assert store.stats(conn) == (1, 0)
    assert "empty.py" in store.indexed_files(conn)


def test_load_chunks_returns_file_hash(conn: sqlite3.Connection) -> None:
    write(conn, "a.py", [make_chunk("a.py", 7, "body", symbol="f")], file_hash="deadbeef")
    (chunk_id, _), = list(store.iter_vectors(conn))
    loaded = store.load_chunks(conn, [chunk_id])[chunk_id]
    assert (loaded.symbol, loaded.start_line, loaded.file_hash) == ("f", 7, "deadbeef")


def test_vectors_round_trip_exactly(conn: sqlite3.Connection) -> None:
    original = array("f", [0.5, -0.25, 0.125, 0.0625])
    store.replace_file(conn, "a.py", FileState(1, 2), "h",
                       [make_chunk("a.py", 1, "x")], [original])
    (_, blob), = list(store.iter_vectors(conn))
    assert array("f", blob) == original


def test_load_chunks_with_no_ids(conn: sqlite3.Connection) -> None:
    assert store.load_chunks(conn, []) == {}


# --------------------------------------------------------------------------- #
# compatibility
# --------------------------------------------------------------------------- #


def test_fresh_index_is_compatible_with_anything(conn: sqlite3.Connection) -> None:
    assert store.compatibility_problems(conn, "bge-m3", 1024) == []


def test_stamped_index_accepts_matching_settings(conn: sqlite3.Connection) -> None:
    store.stamp(conn, "bge-m3", 1024)
    assert store.compatibility_problems(conn, "bge-m3", 1024) == []


@pytest.mark.parametrize(("model", "dimension"), [
    ("nomic-embed-text", 1024),
    ("bge-m3", 768),
])
def test_model_or_dimension_change_is_incompatible(
    conn: sqlite3.Connection, model: str, dimension: int
) -> None:
    store.stamp(conn, "bge-m3", 1024)
    assert store.compatibility_problems(conn, model, dimension)


def test_chunker_version_change_is_incompatible(conn: sqlite3.Connection) -> None:
    store.stamp(conn, "bge-m3", 1024)
    store.write_meta(conn, {"chunker_version": str(CHUNKER_VERSION + 1)})
    problems = store.compatibility_problems(conn, "bge-m3", 1024)
    assert any("chunker_version" in p for p in problems)


def test_meta_is_upserted_not_duplicated(conn: sqlite3.Connection) -> None:
    store.write_meta(conn, {"embedding_model": "a"})
    store.write_meta(conn, {"embedding_model": "b"})
    assert store.read_meta(conn)["embedding_model"] == "b"


# --------------------------------------------------------------------------- #
# chunk ids are never reissued
#
# This is the one storage property a stale `Searcher` depends on. It holds its
# vector buffer across a refresh, and the buffer is nothing but ids; the search
# path drops ids that vanished. That correction is only sound while a surviving
# id still means the same chunk -- which a plain INTEGER PRIMARY KEY does not
# guarantee, because it is the rowid and SQLite picks max(rowid)+1.
# --------------------------------------------------------------------------- #


def test_rewriting_the_highest_ids_does_not_hand_them_to_other_chunks(
        conn: sqlite3.Connection) -> None:
    """The case that actually collides: rewrite the file holding the top ids.

    Without AUTOINCREMENT the replacement chunks take the ids just freed, and a
    ranking computed before the refresh then loads the wrong text under a score
    it never earned -- a wrong answer, silently, with no row missing anywhere.
    """
    store.replace_file(conn, "a.py", FileState(1, 1), "ha",
                       [make_chunk("a.py", i, f"A{i}") for i in range(3)],
                       [vec(0.1)] * 3)
    store.replace_file(conn, "b.py", FileState(1, 1), "hb",
                       [make_chunk("b.py", i, f"B{i}") for i in range(2)],
                       [vec(0.2)] * 2)
    before = {row["id"]: row["text"] for row in conn.execute("SELECT id, text FROM chunks")}

    store.replace_file(conn, "b.py", FileState(2, 2), "hb2",
                       [make_chunk("b.py", i, f"C{i}") for i in range(2)],
                       [vec(0.3)] * 2)
    after = {row["id"]: row["text"] for row in conn.execute("SELECT id, text FROM chunks")}

    reused = {i: (before[i], after[i]) for i in before if i in after and before[i] != after[i]}
    assert not reused, f"id выданы другим чанкам: {reused}"


def test_a_freed_id_is_not_reused_even_when_the_table_empties(
        conn: sqlite3.Connection) -> None:
    """The strongest form: drop everything, insert again, ids still move forward."""
    store.replace_file(conn, "a.py", FileState(1, 1), "ha",
                       [make_chunk("a.py", 1, "A")], [vec(0.1)])
    first = conn.execute("SELECT id FROM chunks").fetchone()["id"]
    store.drop_files(conn, ["a.py"])
    store.replace_file(conn, "a.py", FileState(2, 2), "ha2",
                       [make_chunk("a.py", 1, "B")], [vec(0.2)])
    assert conn.execute("SELECT id FROM chunks").fetchone()["id"] > first
