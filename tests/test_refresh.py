"""Background refresh tests.

The failure modes here are all silent ones: an index that stops updating and
says nothing, a searcher holding vectors from before a rewrite, two refreshes
racing each other. None of them raise, so each gets an explicit test.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_search import store  # noqa: E402
from code_search.embedder import Embedder, EmbeddingError  # noqa: E402
from code_search.indexer import index_path, plan, reindex  # noqa: E402
from code_search.refresh import Refresher  # noqa: E402
from code_search.search import Searcher  # noqa: E402

MODULE = '''\
"""Docstring long enough to survive."""


def compute_route(origin, target):
    """Считает маршрут между точками."""
    return origin, target
'''


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__("bge-m3")
        self.documents = 0

    def embed_documents(self, texts: list[str]) -> list[array[float]]:
        self.documents += len(texts)
        return [array("f", [0.1] * self.dimension) for _ in texts]

    def embed_query(self, text: str) -> array[float]:
        return array("f", [0.1] * self.dimension)


class BrokenEmbedder(FakeEmbedder):
    def embed_documents(self, texts: list[str]) -> list[array[float]]:
        raise EmbeddingError("Ollama недоступна")


class BlockingEmbedder(FakeEmbedder):
    """Holds the refresh open until the test lets it go."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def embed_documents(self, texts: list[str]) -> list[array[float]]:
        self.entered.set()
        self.release.wait(timeout=10)
        return super().embed_documents(texts)


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "routes.py").write_text(MODULE, encoding="utf-8")
    (root / "README.md").write_text("# Title\n\nBody long enough.\n", encoding="utf-8")
    reindex(root, embedder=FakeEmbedder())
    return root


@pytest.fixture()
def conn(project: Path) -> sqlite3.Connection:
    connection = store.open_index(index_path(project))
    yield connection
    connection.close()


def refresher(project: Path, conn: sqlite3.Connection, embedder: Embedder | None = None,
              searcher: Searcher | None = None) -> Refresher:
    return Refresher(project, conn, searcher=searcher,
                     embedder=embedder or FakeEmbedder())


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #


def test_status_reports_the_index_not_ollama(project: Path,
                                             conn: sqlite3.Connection) -> None:
    status = refresher(project, conn).status()
    assert status.files == 2
    assert status.chunks > 0
    assert status.model == "bge-m3"
    assert status.last_indexed is not None
    assert (status.stale_files, status.stale, status.refreshing) == (0, False, False)


def test_an_edit_makes_the_index_stale(project: Path, conn: sqlite3.Connection) -> None:
    (project / "src" / "routes.py").write_text(MODULE + "\ndef extra():\n    return 1\n",
                                               encoding="utf-8")
    assert refresher(project, conn).status().stale_files == 1


def test_a_deleted_file_makes_the_index_stale(project: Path,
                                              conn: sqlite3.Connection) -> None:
    (project / "src" / "routes.py").unlink()
    assert refresher(project, conn).stale_files() == 1


def test_a_touch_without_an_edit_is_not_stale(project: Path,
                                              conn: sqlite3.Connection) -> None:
    """Same bytes, new mtime -- re-embedding that would be pure waste."""
    (project / "src" / "routes.py").write_text(MODULE, encoding="utf-8")
    assert refresher(project, conn).stale_files() == 0


def test_a_touch_is_settled_and_not_rehashed_forever(
        project: Path, conn: sqlite3.Connection) -> None:
    """The stat has to be written back, or every later query pays for the hash.

    Nothing else settles it: `reindex` does, but only runs when a file genuinely
    changed, and a touch alone is not stale. So before this, a `git checkout` to
    another branch and back left the whole corpus permanently `touched` and every
    single search re-read and re-hashed all of it.
    """
    for path in project.rglob("*.py"):
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    assert plan(conn, project).touched, "фикстура не воспроизвела touch"

    worker = refresher(project, conn)
    assert worker.stale_files() == 0
    assert plan(conn, project).touched == [], "stat не записан обратно"
    assert worker.refreshing is False, "переиндексация тут не нужна и не должна стартовать"


def test_settling_a_touch_does_not_disturb_the_vectors(
        project: Path, conn: sqlite3.Connection) -> None:
    """It is a stat write and nothing else: no chunk may be re-embedded."""
    embedder = FakeEmbedder()
    (project / "src" / "routes.py").write_text(MODULE, encoding="utf-8")
    refresher(project, conn, embedder=embedder).stale_files()
    assert embedder.documents == 0


def test_a_real_edit_is_still_stale_after_a_touch_is_settled(
        project: Path, conn: sqlite3.Connection) -> None:
    """Settling the cheap tier must not swallow the expensive one."""
    worker = refresher(project, conn)
    (project / "README.md").write_text("# Title\n\nBody long enough.\n", encoding="utf-8")
    (project / "src" / "routes.py").write_text(MODULE + "\ndef extra():\n    return 1\n",
                                               encoding="utf-8")
    assert worker.stale_files() == 1


def test_a_new_file_makes_the_index_stale(project: Path,
                                          conn: sqlite3.Connection) -> None:
    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    assert refresher(project, conn).stale_files() == 1


# --------------------------------------------------------------------------- #
# the refresh itself
# --------------------------------------------------------------------------- #


def test_a_current_index_starts_no_thread(project: Path,
                                          conn: sqlite3.Connection) -> None:
    worker = refresher(project, conn)
    assert worker.ensure_fresh() is False
    assert (worker.refreshing, worker.generation) == (False, 0)


def test_ensure_fresh_reports_staleness_and_returns_at_once(
        project: Path, conn: sqlite3.Connection) -> None:
    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    worker = refresher(project, conn)
    assert worker.ensure_fresh() is True
    assert worker.wait(timeout=30)
    assert worker.stale_files() == 0


def test_the_refresh_actually_indexes_the_change(project: Path,
                                                 conn: sqlite3.Connection) -> None:
    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    worker = refresher(project, conn)
    worker.ensure_fresh()
    assert worker.wait(timeout=30)
    assert store.stats(conn)[0] == 3


def test_only_one_refresh_runs_at_a_time(project: Path,
                                         conn: sqlite3.Connection) -> None:
    blocking = BlockingEmbedder()
    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    worker = refresher(project, conn, embedder=blocking)

    worker.ensure_fresh()
    assert blocking.entered.wait(timeout=10)
    worker.ensure_fresh()                      # must be dropped, not queued
    blocking.release.set()
    assert worker.wait(timeout=30)
    assert worker.generation == 1


# --------------------------------------------------------------------------- #
# invalidation
# --------------------------------------------------------------------------- #


def test_the_searcher_is_invalidated_after_a_refresh(
        project: Path, conn: sqlite3.Connection) -> None:
    """Chunk ids are reissued on rewrite; stale vectors would score dead rows."""
    searcher = Searcher(conn, FakeEmbedder())
    before = len(searcher.vectors)
    worker = refresher(project, conn, searcher=searcher)

    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    worker.ensure_fresh()
    assert worker.wait(timeout=30)
    assert len(searcher.vectors) > before


def test_search_still_works_against_a_stale_vector_cache(
        project: Path, conn: sqlite3.Connection) -> None:
    """Whatever else happens, a rewrite underneath a live searcher must not crash."""
    searcher = Searcher(conn, FakeEmbedder())
    searcher.search("маршрут")                             # populate the cache
    worker = refresher(project, conn)                      # no searcher attached
    (project / "src" / "routes.py").write_text(MODULE + "\nx = 1\n", encoding="utf-8")
    worker.ensure_fresh()
    assert worker.wait(timeout=30)
    searcher.search("маршрут")                             # ids now dangle


# --------------------------------------------------------------------------- #
# failure
# --------------------------------------------------------------------------- #


def test_a_failed_refresh_is_reported_not_swallowed(project: Path,
                                                    conn: sqlite3.Connection) -> None:
    """Ollama being down is ordinary. A frozen index with no explanation is not."""
    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    worker = refresher(project, conn, embedder=BrokenEmbedder())
    worker.ensure_fresh()
    assert worker.wait(timeout=30)

    status = worker.status()
    assert status.last_error is not None
    assert "Ollama" in status.last_error
    assert status.stale_files == 1, "и индекс честно остаётся устаревшим"


def test_a_failure_does_not_block_the_next_attempt(project: Path,
                                                   conn: sqlite3.Connection) -> None:
    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    worker = refresher(project, conn, embedder=BrokenEmbedder())
    worker.ensure_fresh()
    worker.wait(timeout=30)

    worker._embedder = FakeEmbedder()
    worker.ensure_fresh()
    assert worker.wait(timeout=30)
    assert worker.status().last_error is None
    assert worker.stale_files() == 0


def test_a_failed_refresh_does_not_invalidate_the_searcher(
        project: Path, conn: sqlite3.Connection) -> None:
    """Nothing was written, so throwing away a good cache would be pure cost."""
    searcher = Searcher(conn, FakeEmbedder())
    loaded = searcher.vectors
    worker = refresher(project, conn, embedder=BrokenEmbedder(), searcher=searcher)
    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    worker.ensure_fresh()
    assert worker.wait(timeout=30)
    assert searcher.vectors is loaded


# --------------------------------------------------------------------------- #
# other languages
# --------------------------------------------------------------------------- #


def test_the_server_never_drops_a_language_it_was_not_told_about(
        project: Path, conn: sqlite3.Connection) -> None:
    """A `Refresher` is constructed with no suffix set, ever.

    It reads the corpus setting out of the index, so `.ts` files stay indexed and
    the status stays fresh. Were it to fall back to the defaults instead, every
    query would report `stale: true` and the first refresh would delete them --
    silently, since deletion is not an error.
    """
    (project / "web").mkdir()
    (project / "web" / "planner.ts").write_text(
        "export class Planner {\n  plan(): void {}\n}\n", encoding="utf-8")
    reindex(project, embedder=FakeEmbedder(), extra_suffixes=frozenset({".ts"}))

    worker = refresher(project, conn)
    assert worker.stale_files() == 0
    assert worker.status().files == 3

    worker.ensure_fresh()
    assert worker.wait(timeout=30)
    assert "web/planner.ts" in store.indexed_files(conn)


def test_a_language_added_from_the_shell_shows_up_without_a_restart(
        project: Path, conn: sqlite3.Connection) -> None:
    """The setting is re-read per call, so a live server picks it up."""
    worker = refresher(project, conn)
    (project / "web").mkdir()
    (project / "web" / "planner.ts").write_text(
        "export class Planner {\n  plan(): void {}\n}\n", encoding="utf-8")
    assert worker.stale_files() == 0, "not opted in yet"

    reindex(project, embedder=FakeEmbedder(), extra_suffixes=frozenset({".ts"}))
    assert worker.status().files == 3
