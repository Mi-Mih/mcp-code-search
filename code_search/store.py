"""SQLite storage for the index: schema, compatibility, reads and writes.

All SQL lives here. Two things are worth knowing before changing anything:

Consistency between `chunks` and `chunks_fts` is enforced by triggers, not by
call sites. A contentless FTS5 table that silently drifts out of sync with its
companion produces wrong search results with no error anywhere, and the only
reliable defence is to make it impossible to insert or delete a chunk without
the FTS row following.

The FTS table uses `contentless_delete=1` (SQLite 3.43+), so the text is stored
exactly once, in `chunks.text`, and deletion still works with a plain DELETE.
"""

from __future__ import annotations

import sqlite3
from array import array
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from code_search.chunker import CHUNKER_VERSION, Chunk, fts_text
from code_search.corpus import FileState

__all__ = [
    "IncompatibleIndex",
    "SCHEMA_VERSION",
    "StoredChunk",
    "compatibility_problems",
    "drop_files",
    "extra_suffixes",
    "indexed_files",
    "open_index",
    "read_meta",
    "replace_file",
    "stats",
    "write_meta",
]

# 2: chunks.id is AUTOINCREMENT, so a chunk id is never reissued. See the note
#    on the column itself.
SCHEMA_VERSION = "2"
MIN_SQLITE = (3, 43)

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id        INTEGER PRIMARY KEY,
    path      TEXT UNIQUE NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    size      INTEGER NOT NULL,
    file_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    -- AUTOINCREMENT, and it is load-bearing rather than decorative. A plain
    -- INTEGER PRIMARY KEY is the rowid, and SQLite picks max(rowid)+1, so ids
    -- freed by a DELETE come back to *different* chunks later. A Searcher
    -- holding the vector buffer from before a refresh would then score chunk A
    -- and load chunk B under the same id -- a wrong result, not a missing one,
    -- and nothing anywhere would say so. AUTOINCREMENT guarantees an id is
    -- never reused, which reduces the whole race to ids that no longer exist,
    -- and those the search path already drops (see search.Searcher.search).
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    start_line INTEGER NOT NULL,
    end_line   INTEGER NOT NULL,
    symbol     TEXT,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    fts_text   TEXT NOT NULL,
    vector     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    fts_text,
    content='',
    contentless_delete=1,
    tokenize="unicode61 tokenchars '_'"
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, fts_text) VALUES (new.id, new.fts_text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.id;
END;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


class IncompatibleIndex(RuntimeError):
    """The stored index cannot be reused and has to be rebuilt from scratch."""


@dataclass(frozen=True)
class StoredChunk:
    """A chunk as it comes back out of the index."""

    id: int
    path: str
    kind: str
    start_line: int
    end_line: int
    symbol: str | None
    text: str
    file_hash: str


def open_index(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) an index database with the schema applied."""
    if tuple(int(p) for p in sqlite3.sqlite_version.split(".")[:2]) < MIN_SQLITE:
        raise IncompatibleIndex(
            f"SQLite {'.'.join(map(str, MIN_SQLITE))}+ required for "
            f"contentless_delete, found {sqlite3.sqlite_version}"
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# --------------------------------------------------------------------------- #
# meta and compatibility
# --------------------------------------------------------------------------- #


def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    """Return all index metadata as key/value strings."""
    return {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}


def write_meta(conn: sqlite3.Connection, values: dict[str, str]) -> None:
    """Upsert metadata values and commit them as one transaction."""
    conn.executemany(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        sorted(values.items()),
    )
    conn.commit()


def extra_suffixes(conn: sqlite3.Connection) -> frozenset[str] | None:
    """Suffixes this index was built to cover beyond the defaults.

    `None` means the index predates the setting or never had extras, which is not
    the same as an empty set: an empty set is a caller who said "drop them".
    """
    stored = read_meta(conn).get("extra_suffixes")
    if stored is None:
        return None
    return frozenset(part for part in stored.split(",") if part)


def compatibility_problems(conn: sqlite3.Connection, model: str, dimension: int) -> list[str]:
    """Reasons the stored index cannot be reused; empty list means reusable.

    A fresh, never-populated index is compatible with anything: there is nothing
    in it to be inconsistent with.

    `extra_suffixes` is deliberately absent from this check. Widening or narrowing
    the corpus is an ordinary incremental change -- files appear or are dropped,
    the rest keeps its vectors -- and treating it as an incompatibility would turn
    "also index .ts" into a forty-minute rebuild of everything.
    """
    meta = read_meta(conn)
    if not meta:
        return []
    expected = {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": model,
        "embedding_dimension": str(dimension),
        "chunker_version": str(CHUNKER_VERSION),
    }
    return [
        f"{key}: index has {meta.get(key, '<missing>')!r}, code expects {want!r}"
        for key, want in expected.items()
        if meta.get(key) != want
    ]


def stamp(conn: sqlite3.Connection, model: str, dimension: int,
          suffixes: frozenset[str] = frozenset()) -> None:
    """Record the compatibility settings used to build an index."""
    write_meta(conn, {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": model,
        "embedding_dimension": str(dimension),
        "chunker_version": str(CHUNKER_VERSION),
        "extra_suffixes": ",".join(sorted(suffixes)),
    })


# --------------------------------------------------------------------------- #
# writes
# --------------------------------------------------------------------------- #


def indexed_files(conn: sqlite3.Connection) -> dict[str, tuple[FileState, str]]:
    """Everything currently in the index, keyed by path."""
    return {
        row["path"]: (FileState(row["mtime_ns"], row["size"]), row["file_hash"])
        for row in conn.execute("SELECT path, mtime_ns, size, file_hash FROM files")
    }


def replace_file(conn: sqlite3.Connection, path: str, state: FileState, file_hash: str,
                 chunks: Sequence[Chunk], vectors: Sequence[array[float]]) -> None:
    """Atomically swap all chunks of one file for a new set.

    Deleting the `files` row cascades to `chunks`, which fires the FTS trigger,
    so no stale lexical rows survive.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors for {path}")

    with conn:
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        cursor = conn.execute(
            "INSERT INTO files(path, mtime_ns, size, file_hash) VALUES (?, ?, ?, ?)",
            (path, state.mtime_ns, state.size, file_hash),
        )
        file_id = cursor.lastrowid
        conn.executemany(
            "INSERT INTO chunks(file_id, start_line, end_line, symbol, kind, "
            "text, fts_text, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (file_id, chunk.start_line, chunk.end_line, chunk.symbol, chunk.kind,
                 chunk.text, fts_text(chunk), vector.tobytes())
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )


def drop_files(conn: sqlite3.Connection, paths: Sequence[str]) -> None:
    """Remove files that no longer exist in the corpus."""
    with conn:
        conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in paths])


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #


def stats(conn: sqlite3.Connection) -> tuple[int, int]:
    """(files, chunks) currently stored."""
    files = conn.execute("SELECT count(*) FROM files").fetchone()[0]
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return int(files), int(chunks)


def iter_vectors(conn: sqlite3.Connection) -> Iterator[tuple[int, bytes]]:
    """Every (chunk id, raw float32 vector) in storage, for a full scan."""
    for row in conn.execute("SELECT id, vector FROM chunks"):
        yield int(row["id"]), bytes(row["vector"])


def load_chunks(conn: sqlite3.Connection, ids: Sequence[int]) -> dict[int, StoredChunk]:
    """Fetch full chunk records by id, preserving nothing about order."""
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT c.id, f.path, c.kind, c.start_line, c.end_line, c.symbol, "
        "c.text, f.file_hash FROM chunks c JOIN files f ON f.id = c.file_id "
        f"WHERE c.id IN ({placeholders})",
        list(ids),
    )
    return {
        int(row["id"]): StoredChunk(
            id=int(row["id"]), path=row["path"], kind=row["kind"],
            start_line=int(row["start_line"]), end_line=int(row["end_line"]),
            symbol=row["symbol"], text=row["text"], file_hash=row["file_hash"],
        )
        for row in rows
    }
