"""Keep the index current without ever making a query wait for it.

There is no synchronous refresh mode, and adding one would be a mistake. At the
measured embedding rate, twenty changed files is roughly three and a half
minutes of work; any staleness threshold small enough to hide that from a caller
is too small to be worth a second code path. So the rule is one line:

    stale files exist -> answer from the index we have, say `stale: true`,
                         and start a background refresh
    otherwise         -> answer from the index we have

`stale: true` is a normal operating state, not an error. The caller is told the
truth and decides for itself whether to wait; the search layer does not make
that decision on its behalf.

Two things this module exists to prevent, both of which fail silently:

  * A refresh finishing while a `Searcher` still holds the vectors from before
    it. Chunk ids are reissued on rewrite, so the stale buffer scores rows that
    no longer exist and the only symptom is worse results. The searcher is
    therefore invalidated here, structurally, rather than by remembering to.
  * A background thread dying with nothing to show for it. Ollama being down is
    the ordinary case, not an exotic one, and it must surface in `status()`
    instead of leaving the index quietly frozen.

Staleness is recomputed on every call rather than cached behind a TTL. It costs
about 10 ms against a 145 ms query, and a TTL would buy that back by sometimes
reporting a stale index as fresh -- the one thing this module must not do. That
10 ms holds only because classification settles what it cheaply can as it goes;
see `_classify`.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from code_search import store
from code_search.embedder import DEFAULT_MODEL, Embedder
from code_search.indexer import Plan, plan, refresh_stats, reindex
from code_search.search import Searcher

__all__ = ["Refresher", "Status"]


@dataclass(frozen=True)
class Status:
    """What `index_status()` reports. Index only -- Ollama's health is separate."""

    files: int
    chunks: int
    model: str
    chunker_version: str
    stale_files: int
    last_indexed: str | None
    refreshing: bool
    last_error: str | None = None

    @property
    def stale(self) -> bool:
        """Whether at least one indexed file differs from the working tree."""
        return self.stale_files > 0


class Refresher:
    """Owns the decision to reindex, and the thread that does it.

    One refresh at a time. A second request while one is running is dropped
    rather than queued: by the time the running one finishes it will have picked
    up whatever changed in the meantime, because it plans its work at the moment
    it starts, not at the moment it was requested.
    """

    def __init__(self, root: Path, conn: sqlite3.Connection,
                 searcher: Searcher | None = None, model: str = DEFAULT_MODEL,
                 excludes: tuple[str, ...] = (),
                 embedder: Embedder | None = None) -> None:
        self.root = root.resolve()
        self.conn = conn
        self.searcher = searcher
        self.model = model
        self.excludes = excludes
        self._embedder = embedder          # injectable for tests
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self._generation = 0

    # ----------------------------------------------------------------- state #

    @property
    def refreshing(self) -> bool:
        """Whether this process currently owns a live refresh thread."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def generation(self) -> int:
        """Increments once per completed refresh. Lets tests observe progress."""
        with self._lock:
            return self._generation

    def _classify(self) -> Plan:
        """Classify the corpus, and settle the cheap tier while we are here.

        A `touched` file -- stat moved, content identical -- costs one UPDATE to
        settle and nothing else. Settling it has to happen here, because nothing
        else will: `reindex` also heals touched stats, but `reindex` only runs
        when some file genuinely changed. So a `git checkout` to another branch
        and back, a `stash pop`, or a reformat that produced the same bytes left
        the whole corpus permanently `touched`, and *every* query re-read and
        re-hashed all of it -- on the reference project a 2 MB read and 241
        SHA-256s per search, for as long as nobody ran the indexer by hand.

        Skipped while a refresh is running: WAL admits a single writer, and that
        refresh is going to heal the same files itself. Nothing is lost -- the
        next call classifies again.
        """
        todo = plan(self.conn, self.root, self.excludes)
        if todo.touched and not self.refreshing:
            try:
                refresh_stats(self.conn, self.root, todo.touched)
            except sqlite3.OperationalError:
                # Lost the write lock to a refresh that started meanwhile. This
                # is an optimisation, not a correctness step: drop it and retry
                # on the next call rather than failing the search.
                pass
        return todo

    def stale_files(self) -> int:
        """How many files the index is behind by. Zero means fully current.

        Neither this nor `_run` passes a suffix set: both let `plan` and `reindex`
        read it out of the index, which is the only copy the server can trust (see
        the note at the top of `indexer`). It is re-read per call, so adding a
        language from the shell takes effect on the next query rather than on the
        next restart.
        """
        todo = self._classify()
        return len(todo.changed) + len(todo.removed)

    def status(self) -> Status:
        """Return a point-in-time status suitable for the MCP status tool."""
        files, chunks = store.stats(self.conn)
        meta = store.read_meta(self.conn)
        with self._lock:
            error = self._last_error
        return Status(
            files=files,
            chunks=chunks,
            model=meta.get("embedding_model", self.model),
            chunker_version=meta.get("chunker_version", "?"),
            stale_files=self.stale_files(),
            last_indexed=meta.get("last_indexed"),
            refreshing=self.refreshing,
            last_error=error,
        )

    # --------------------------------------------------------------- refresh #

    def ensure_fresh(self) -> bool:
        """Start a background refresh if one is warranted. Never blocks.

        Returns whether the index is stale -- which is what the caller reports
        as `stale: true`, and is deliberately not the same as "a refresh was
        started". An index can be stale while a refresh is already running.
        """
        stale = self.stale_files() > 0
        if stale:
            self._start()
        return stale

    def _start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # daemon: a half-finished refresh must never hold up process exit.
            # Losing it costs nothing, because indexing commits file by file.
            self._thread = threading.Thread(target=self._run, name="code-search-refresh",
                                            daemon=True)
            self._thread.start()

    def _run(self) -> None:
        """The background body. Opens its own connection -- SQLite handles are
        not shareable across threads, and WAL admits one writer beside readers.
        """
        error: str | None = None
        try:
            reindex(self.root, self.model, self.excludes, embedder=self._embedder)
        except Exception as exc:                      # noqa: BLE001 -- see below
            # Deliberately broad. This thread has no caller to propagate to, and
            # anything escaping it would vanish into the interpreter's default
            # handler, leaving the index frozen with no indication why.
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._last_error = error
            self._generation += 1
        if error is None and self.searcher is not None:
            self.searcher.invalidate()

    def wait(self, timeout: float = 60.0) -> bool:
        """Block until any running refresh finishes. For tests and the CLI only."""
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()
