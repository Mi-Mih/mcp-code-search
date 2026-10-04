"""Build and refresh the index.

The unit of work is one file: chunk it, embed its chunks, write them in one
transaction. That makes a run interruptible at any point -- indexing a 61k-LOC
project takes tens of minutes, and losing all of it to a stray Ctrl-C would be
the difference between a tool people use and one they avoid.

Change detection is three-tiered, cheapest first:

  mtime_ns + size unchanged  -> skip, no read at all
  changed, hash unchanged    -> refresh the stat, keep the vectors (a touch,
                                a `git checkout` of the same content, a reformat
                                that produced identical bytes)
  hash changed               -> re-chunk and re-embed

Only the third tier costs money, and on a normal edit session it applies to a
handful of files.

Which suffixes belong to the corpus is stored in the index, not taken from the
command line, and `reindex` reads it back whenever the caller does not say. That
indirection is load-bearing. The MCP server refreshes in the background and knows
nothing about the flags the index was built with; if it fell back to the defaults
instead, every `.ts` file would look deleted, `plan` would report the index
permanently stale, and the first query would quietly drop all of them.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

from code_search import corpus, store
from code_search.chunker import SUPPORTED_SUFFIXES, chunk_text, enrich
from code_search.embedder import DEFAULT_MODEL, Embedder, EmbeddingError

__all__ = ["INDEX_DIR", "IndexStats", "Plan", "index_path", "plan", "refresh_stats",
           "reindex"]

# Kept inside the indexed project so that several projects never share state.
INDEX_DIR = ".code-search"
INDEX_FILE = "index.sqlite"

ProgressFn = Callable[[str, int, int], None]


def index_path(root: Path) -> Path:
    """Return the SQLite index path owned by ``root``."""
    return root / INDEX_DIR / INDEX_FILE


@dataclass
class Plan:
    """What a refresh would do, computed without embedding anything."""

    unchanged: list[str] = field(default_factory=list)
    touched: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        """Whether search results would be computed from an outdated index."""
        return bool(self.changed or self.removed)


@dataclass
class IndexStats:
    """Outcome of a refresh."""

    files_embedded: int = 0
    files_touched: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_without_chunks: int = 0
    chunks_embedded: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        """Render the refresh counters as a concise human-readable report."""
        rate = self.chunks_embedded / self.seconds if self.seconds else 0.0
        text = (
            f"{self.files_embedded} файл(ов) переиндексировано, "
            f"{self.chunks_embedded} чанков, {self.seconds / 60:.1f} мин "
            f"({rate:.1f} чанк/с); без изменений {self.files_unchanged}, "
            f"обновлён stat {self.files_touched}, удалено {self.files_removed}"
        )
        if self.files_without_chunks:
            # Reported because it is otherwise invisible: the file is in the
            # index, contributes nothing, and will never appear in a result.
            text += (f"\n{self.files_without_chunks} файл(ов) не дали ни одного "
                     f"чанка (пустые или сгенерированные)")
        return text


def _resolve_suffixes(conn: sqlite3.Connection,
                      requested: frozenset[str] | None) -> frozenset[str]:
    """Extra suffixes to use: the caller's choice, else whatever the index says."""
    if requested is not None:
        return requested
    stored = store.extra_suffixes(conn)
    return stored if stored is not None else frozenset()


def plan(conn: sqlite3.Connection, root: Path, excludes: tuple[str, ...] = (),
         extra_suffixes: frozenset[str] | None = None) -> Plan:
    """Classify every file into unchanged / touched / changed / removed."""
    known = store.indexed_files(conn)
    present = corpus.list_files(root, excludes, _resolve_suffixes(conn, extra_suffixes))
    result = Plan(removed=sorted(set(known) - set(present)))

    for rel in present:
        state = corpus.state_of(root, rel)
        if state is None:                       # vanished between listing and stat
            if rel in known:
                result.removed.append(rel)
            continue
        if rel not in known:
            result.changed.append(rel)
            continue
        stored_state, stored_hash = known[rel]
        if state == stored_state:
            result.unchanged.append(rel)
        elif corpus.content_hash(corpus.read_text(root, rel)) == stored_hash:
            result.touched.append(rel)
        else:
            result.changed.append(rel)
    return result


def refresh_stats(conn: sqlite3.Connection, root: Path, paths: Sequence[str]) -> None:
    """Record current mtime/size for files whose content did not change.

    Public because the query path needs it too: a `touched` file costs nothing
    but this UPDATE to settle, and until it is settled every `plan` re-reads and
    re-hashes the file. See `Refresher._classify`.

    One transaction for the whole batch, not one per file: after a branch switch
    `touched` is the entire corpus, and 241 separate commits on the query path
    would be worse than the problem.
    """
    updates = [
        (state.mtime_ns, state.size, rel)
        for rel, state in ((rel, corpus.state_of(root, rel)) for rel in paths)
        if state is not None                    # vanished between plan and here
    ]
    if not updates:
        return
    with conn:
        conn.executemany(
            "UPDATE files SET mtime_ns = ?, size = ? WHERE path = ?", updates
        )


def reindex(root: Path, model: str = DEFAULT_MODEL, excludes: tuple[str, ...] = (),
            rebuild: bool = False, progress: ProgressFn | None = None,
            embedder: Embedder | None = None,
            extra_suffixes: frozenset[str] | None = None) -> IndexStats:
    """Bring the index for `root` up to date, embedding only what changed.

    `embedder` is injectable so that tests can exercise the change-detection
    logic without a running Ollama; production callers leave it None.

    `extra_suffixes` of None means "whatever this index was last built with",
    which is what every caller except the CLI wants. Passing a set replaces the
    stored one -- that is how a language is added or dropped, and dropping costs
    nothing but the deletion, since the remaining files keep their vectors.
    """
    root = root.resolve()
    if embedder is None:
        embedder = Embedder(model)
    db_path = index_path(root)

    if rebuild and db_path.parent.exists():
        shutil.rmtree(db_path.parent)

    conn = store.open_index(db_path)
    problems = store.compatibility_problems(conn, model, embedder.dimension)
    if problems:
        conn.close()
        print("Индекс несовместим и будет пересобран:", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        shutil.rmtree(db_path.parent)
        conn = store.open_index(db_path)

    suffixes = _resolve_suffixes(conn, extra_suffixes)
    store.stamp(conn, model, embedder.dimension, suffixes)
    _write_gitignore(db_path.parent)

    todo = plan(conn, root, excludes, suffixes)
    stats = IndexStats(
        files_unchanged=len(todo.unchanged),
        files_touched=len(todo.touched),
        files_removed=len(todo.removed),
    )

    if todo.removed:
        store.drop_files(conn, todo.removed)
    refresh_stats(conn, root, todo.touched)

    started = time.perf_counter()
    try:
        for done, rel in enumerate(todo.changed, start=1):
            text = corpus.read_text(root, rel)
            state = corpus.state_of(root, rel)
            if state is None:                   # deleted mid-run
                continue
            # The suffix was already vetted by `corpus.list_files`; re-checking it
            # here keeps the generic chunker from being reachable for anything the
            # caller did not opt into, whatever a future corpus change allows.
            generic = PurePosixPath(rel).suffix not in SUPPORTED_SUFFIXES
            chunks = chunk_text(rel, text, fallback=generic)
            vectors = embedder.embed_documents([enrich(c) for c in chunks]) if chunks else []
            store.replace_file(conn, rel, state, corpus.content_hash(text), chunks, vectors)
            stats.files_embedded += 1
            stats.chunks_embedded += len(chunks)
            if not chunks:
                stats.files_without_chunks += 1
            if progress is not None:
                progress(rel, done, len(todo.changed))
    finally:
        stats.seconds = time.perf_counter() - started
        # Stamped even on an interrupted run: what it records is when the index
        # was last touched, which is exactly what a partial run also changes.
        store.write_meta(conn, {"last_indexed": datetime.now().isoformat(timespec="seconds")})
        conn.close()
    return stats


def _write_gitignore(index_dir: Path) -> None:
    """Keep the index out of the indexed repository's history."""
    marker = index_dir / ".gitignore"
    if not marker.exists():
        index_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("*\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_suffixes(values: list[str] | None) -> frozenset[str] | None:
    """`--suffix` occurrences into a set, or None when the flag was not used.

    None and the empty set are different answers: "did not ask" keeps whatever
    the index already covers, "asked for nothing" drops the extras.
    """
    if values is None:
        return None
    if any(value.strip().lower() == "none" for value in values):
        if len(values) > 1:
            raise ValueError("--suffix none нельзя совмещать с другими расширениями")
        return frozenset()
    out: set[str] = set()
    for raw in values:
        value = raw.strip()
        if not value.strip(".") or "/" in value or "\\" in value:
            raise ValueError(f"не похоже на расширение: {raw!r}")
        out.add(value if value.startswith(".") else f".{value}")
    return frozenset(out)


def _effective_suffixes(root: Path, requested: frozenset[str] | None,
                        rebuild: bool) -> frozenset[str]:
    """What the coming run will cover, so the report can be printed before it.

    A rebuild discards the stored setting along with the rest of the index, so
    an unasked-for rebuild narrows the corpus back to the defaults.
    """
    if requested is not None or rebuild:
        return requested or frozenset()
    conn = store.open_index(index_path(root))
    try:
        return _resolve_suffixes(conn, None)
    finally:
        conn.close()


def _report_skipped(root: Path, excludes: tuple[str, ...],
                    suffixes: frozenset[str]) -> None:
    """Name the files the suffix filter drops, so the gap is never inferred."""
    skipped = corpus.skipped_by_suffix(root, excludes, suffixes)
    if not skipped:
        return
    shown = list(skipped.items())[:8]
    detail = ", ".join(f"{suffix} {count}" for suffix, count in shown)
    if len(skipped) > len(shown):
        detail += f", ... (ещё {len(skipped) - len(shown)} расширений)"
    print(f"пропущено по расширению: {sum(skipped.values())} файл(ов) — {detail}")
    print("  чтобы добавить язык: --suffix .ts (обобщённая нарезка, см. README)")


def main(argv: list[str] | None = None) -> int:
    """Run the indexer CLI and return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m code_search.indexer",
        description="Собрать или обновить индекс семантического поиска.",
    )
    parser.add_argument("root", type=Path, help="корень индексируемого проекта")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--exclude", action="append", default=[],
                        metavar="GLOB", help="исключить пути по fnmatch-шаблону")
    parser.add_argument("--rebuild", action="store_true",
                        help="снести существующий индекс и собрать заново "
                             "(вместе с настройкой --suffix)")
    parser.add_argument("--suffix", action="append", metavar="EXT",
                        help="индексировать ещё и файлы с этим расширением, например "
                             "--suffix .ts; они режутся обобщённо, по пустым строкам, "
                             "без имён функций. Настройка запоминается в индексе. "
                             "--suffix none — отказаться от всех добавленных ранее")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать, что будет переиндексировано, и выйти")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"нет каталога: {root}", file=sys.stderr)
        return 2

    excludes = tuple(args.exclude)
    try:
        requested = _parse_suffixes(args.suffix)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.dry_run:
        conn = store.open_index(index_path(root))
        suffixes = _resolve_suffixes(conn, requested)
        todo = plan(conn, root, excludes, suffixes)
        conn.close()
        if suffixes:
            print(f"дополнительные расширения: {' '.join(sorted(suffixes))}")
        print(f"без изменений {len(todo.unchanged)}, только stat {len(todo.touched)}, "
              f"переиндексировать {len(todo.changed)}, удалить {len(todo.removed)}")
        for rel in todo.changed[:20]:
            print(f"  + {rel}")
        if len(todo.changed) > 20:
            print(f"  ... ещё {len(todo.changed) - 20}")
        _report_skipped(root, excludes, suffixes)
        return 0

    _report_skipped(root, excludes, _effective_suffixes(root, requested, args.rebuild))

    started = time.perf_counter()

    def show(rel: str, done: int, total: int) -> None:
        elapsed = time.perf_counter() - started
        eta = (total - done) * elapsed / done / 60 if done else 0.0
        print(f"  {done}/{total}  ETA {eta:.0f} мин  {rel}", flush=True)

    try:
        stats = reindex(root, args.model, excludes, args.rebuild, show,
                        extra_suffixes=requested)
    except EmbeddingError as exc:
        print(f"ошибка эмбеддинга: {exc}", file=sys.stderr)
        print("проверьте, что Ollama запущена: ollama serve", file=sys.stderr)
        return 1
    except store.IncompatibleIndex as exc:
        # Reached when SQLite itself is too old: a traceback would send the reader
        # looking for a bug in the indexer instead of at their Python build.
        print(f"индекс не открыть: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nпрервано; уже проиндексированные файлы сохранены", file=sys.stderr)
        return 130

    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
