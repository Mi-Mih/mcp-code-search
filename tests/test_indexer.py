"""Indexer and corpus tests.

These run against a real temporary git repository and a fake embedder, so the
expensive part (Ollama) is absent while the part that decides *what* to send it
is exercised for real. The counter on the fake embedder is the assertion that
matters: re-embedding an unchanged file is the whole cost of this system.
"""

from __future__ import annotations

import os
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_search import corpus, store  # noqa: E402
from code_search.embedder import Embedder  # noqa: E402
from code_search.indexer import (  # noqa: E402
    _parse_suffixes,
    index_path,
    main,
    plan,
    reindex,
)

MODULE = '''\
"""Docstring long enough to survive."""

import os


def compute_route(origin, target):
    """Считает маршрут между точками."""
    return origin, target
'''


class FakeEmbedder(Embedder):
    """Counts calls instead of talking to Ollama."""

    def __init__(self) -> None:
        super().__init__("bge-m3")
        self.documents = 0

    def embed_documents(self, texts: list[str]) -> list[array[float]]:
        self.documents += len(texts)
        return [array("f", [0.1] * self.dimension) for _ in texts]


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "routes.py").write_text(MODULE, encoding="utf-8")
    (root / "src" / "config.py").write_text("LIMIT = 5\nMODE = 'fast'\n", encoding="utf-8")
    (root / "README.md").write_text("# Title\n\nBody long enough to matter.\n",
                                    encoding="utf-8")
    return root


def git_init(root: Path) -> None:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, env=env)


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


def test_lists_supported_files_without_git(project: Path) -> None:
    assert corpus.list_files(project) == ["README.md", "src/config.py", "src/routes.py"]


def test_lists_tracked_and_untracked_files_with_git(project: Path) -> None:
    git_init(project)
    (project / "src" / "draft.py").write_text("x = 1\n", encoding="utf-8")
    assert "src/draft.py" in corpus.list_files(project), "work in progress must be indexed"


def test_gitignored_files_are_skipped(project: Path) -> None:
    (project / ".gitignore").write_text("secrets.py\n", encoding="utf-8")
    git_init(project)
    (project / "secrets.py").write_text("TOKEN = 'x'\n", encoding="utf-8")
    assert "secrets.py" not in corpus.list_files(project)


def test_unsupported_suffixes_and_junk_dirs_are_skipped(project: Path) -> None:
    (project / "src" / "notes.txt").write_text("hi\n", encoding="utf-8")
    (project / "src" / "__pycache__").mkdir()
    (project / "src" / "__pycache__" / "cached.py").write_text("x = 1\n", encoding="utf-8")
    listed = corpus.list_files(project)
    assert "src/notes.txt" not in listed
    assert not any("__pycache__" in p for p in listed)


def test_exclude_globs(project: Path) -> None:
    assert "src/config.py" not in corpus.list_files(project, ("src/config.py",))
    assert corpus.list_files(project, ("src/*",)) == ["README.md"]


def test_state_of_missing_file_is_none(project: Path) -> None:
    assert corpus.state_of(project, "nope.py") is None


# --------------------------------------------------------------------------- #
# first index
# --------------------------------------------------------------------------- #


def test_first_run_indexes_everything(project: Path) -> None:
    fake = FakeEmbedder()
    stats = reindex(project, embedder=fake)
    assert stats.files_embedded == 3
    assert stats.chunks_embedded == fake.documents > 0

    conn = store.open_index(index_path(project))
    files, chunks = store.stats(conn)
    conn.close()
    assert (files, chunks) == (3, stats.chunks_embedded)


def test_index_lives_inside_the_project_and_is_gitignored(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    assert index_path(project).exists()
    assert (project / ".code-search" / ".gitignore").read_text() == "*\n"


def test_meta_is_stamped(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    conn = store.open_index(index_path(project))
    meta = store.read_meta(conn)
    conn.close()
    assert meta["embedding_model"] == "bge-m3"
    assert meta["embedding_dimension"] == "1024"


# --------------------------------------------------------------------------- #
# change detection
# --------------------------------------------------------------------------- #


def test_second_run_embeds_nothing(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    again = FakeEmbedder()
    stats = reindex(project, embedder=again)
    assert again.documents == 0
    assert (stats.files_embedded, stats.files_unchanged) == (0, 3)


def test_touch_without_edit_does_not_re_embed(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    target = project / "src" / "routes.py"
    target.write_text(MODULE, encoding="utf-8")      # same bytes, new mtime
    os.utime(target, (0, 0))

    again = FakeEmbedder()
    stats = reindex(project, embedder=again)
    assert again.documents == 0
    assert stats.files_touched == 1


def test_touch_updates_the_stored_stat(project: Path) -> None:
    """Otherwise the same file is re-hashed on every single run."""
    reindex(project, embedder=FakeEmbedder())
    target = project / "src" / "routes.py"
    os.utime(target, (0, 0))
    reindex(project, embedder=FakeEmbedder())

    conn = store.open_index(index_path(project))
    todo = plan(conn, project)
    conn.close()
    assert todo.touched == []
    assert len(todo.unchanged) == 3


def test_edit_re_embeds_only_that_file(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    (project / "src" / "routes.py").write_text(
        MODULE + "\n\ndef extra():\n    return 1\n", encoding="utf-8")

    again = FakeEmbedder()
    stats = reindex(project, embedder=again)
    assert stats.files_embedded == 1
    assert stats.files_unchanged == 2
    assert again.documents > 0


def test_deleted_file_is_removed_from_the_index(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    (project / "src" / "config.py").unlink()
    stats = reindex(project, embedder=FakeEmbedder())
    assert stats.files_removed == 1

    conn = store.open_index(index_path(project))
    assert "src/config.py" not in store.indexed_files(conn)
    orphans = conn.execute(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'LIMIT'").fetchone()[0]
    conn.close()
    assert orphans == 0


def test_new_file_is_picked_up(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
    stats = reindex(project, embedder=FakeEmbedder())
    assert stats.files_embedded == 1


def test_plan_reports_staleness_without_embedding(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    conn = store.open_index(index_path(project))
    assert plan(conn, project).is_stale is False
    (project / "src" / "routes.py").write_text("x = 2\ny = 3\n", encoding="utf-8")
    assert plan(conn, project).is_stale is True
    conn.close()


# --------------------------------------------------------------------------- #
# rebuilds
# --------------------------------------------------------------------------- #


def test_model_change_forces_a_full_rebuild(project: Path, capsys: pytest.CaptureFixture[str]
                                            ) -> None:
    reindex(project, embedder=FakeEmbedder())

    class OtherModel(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__()
            self.spec = self.spec.__class__("nomic-embed-text", 768)

    stats = reindex(project, model="nomic-embed-text", embedder=OtherModel())
    assert stats.files_embedded == 3, "every vector has to be recomputed"
    assert "несовместим" in capsys.readouterr().err


def test_rebuild_flag_discards_the_existing_index(project: Path) -> None:
    reindex(project, embedder=FakeEmbedder())
    stats = reindex(project, rebuild=True, embedder=FakeEmbedder())
    assert stats.files_embedded == 3


def test_empty_project_is_not_an_error(tmp_path: Path) -> None:
    root = tmp_path / "blank"
    root.mkdir()
    stats = reindex(root, embedder=FakeEmbedder())
    assert (stats.files_embedded, stats.chunks_embedded) == (0, 0)


# --------------------------------------------------------------------------- #
# other languages
# --------------------------------------------------------------------------- #

TYPESCRIPT = """\
export class RoutePlanner {
  addLeg(leg: RouteLeg): void {
    this.legs.push(leg);
  }
}
"""

TS = frozenset({".ts"})


@pytest.fixture()
def polyglot(project: Path) -> Path:
    (project / "web").mkdir()
    (project / "web" / "planner.ts").write_text(TYPESCRIPT, encoding="utf-8")
    (project / "web" / "styles.css").write_text("body { margin: 0 }\n", encoding="utf-8")
    return project


def test_other_languages_are_ignored_by_default(polyglot: Path) -> None:
    assert "web/planner.ts" not in corpus.list_files(polyglot)


def test_extra_suffix_widens_the_corpus_additively(polyglot: Path) -> None:
    listed = corpus.list_files(polyglot, (), TS)
    assert "web/planner.ts" in listed
    assert "src/routes.py" in listed, "opting into .ts must not opt out of .py"
    assert "web/styles.css" not in listed


def test_skipped_report_counts_what_the_suffix_filter_drops(polyglot: Path) -> None:
    assert corpus.skipped_by_suffix(polyglot) == {".css": 1, ".ts": 1}
    assert corpus.skipped_by_suffix(polyglot, (), TS) == {".css": 1}


def test_skipped_report_does_not_count_what_the_caller_excluded(polyglot: Path) -> None:
    """Otherwise `--exclude web/*` would keep nagging about the files it removed."""
    assert corpus.skipped_by_suffix(polyglot, ("web/*",)) == {}


def test_extensionless_files_are_reported_under_their_own_label(polyglot: Path) -> None:
    (polyglot / "Makefile").write_text("all:\n\techo hi\n", encoding="utf-8")
    assert corpus.skipped_by_suffix(polyglot)["(без расширения)"] == 1


def test_extra_suffix_is_indexed_and_remembered(polyglot: Path) -> None:
    stats = reindex(polyglot, embedder=FakeEmbedder(), extra_suffixes=TS)
    assert stats.files_embedded == 4

    conn = store.open_index(index_path(polyglot))
    assert store.extra_suffixes(conn) == TS
    assert "web/planner.ts" in store.indexed_files(conn)
    conn.close()


def test_a_refresh_that_was_not_told_keeps_the_extra_language(polyglot: Path) -> None:
    """The failure this whole mechanism exists to prevent.

    The MCP server refreshes with no flags. If the suffix set did not travel in
    the index, `plan` would call every `.ts` file removed, report the index
    permanently stale, and the first background refresh would delete them all.
    """
    reindex(polyglot, embedder=FakeEmbedder(), extra_suffixes=TS)

    conn = store.open_index(index_path(polyglot))
    assert plan(conn, polyglot).is_stale is False, "nothing changed on disk"
    conn.close()

    stats = reindex(polyglot, embedder=FakeEmbedder())      # as the server calls it
    assert (stats.files_embedded, stats.files_removed) == (0, 0)

    conn = store.open_index(index_path(polyglot))
    assert "web/planner.ts" in store.indexed_files(conn)
    conn.close()


def test_a_changed_file_of_an_extra_language_is_re_embedded(polyglot: Path) -> None:
    reindex(polyglot, embedder=FakeEmbedder(), extra_suffixes=TS)
    (polyglot / "web" / "planner.ts").write_text(
        TYPESCRIPT.replace("addLeg", "appendLeg"), encoding="utf-8")
    stats = reindex(polyglot, embedder=FakeEmbedder())
    assert (stats.files_embedded, stats.chunks_embedded) == (1, 1)


def test_an_empty_suffix_set_drops_the_extra_language(polyglot: Path) -> None:
    reindex(polyglot, embedder=FakeEmbedder(), extra_suffixes=TS)
    stats = reindex(polyglot, embedder=FakeEmbedder(), extra_suffixes=frozenset())
    assert stats.files_removed == 1

    conn = store.open_index(index_path(polyglot))
    assert "web/planner.ts" not in store.indexed_files(conn)
    assert store.extra_suffixes(conn) == frozenset()
    conn.close()


def test_changing_the_suffix_set_is_not_an_incompatibility(polyglot: Path) -> None:
    """Adding a language must cost one file, not a full re-embed of the project."""
    reindex(polyglot, embedder=FakeEmbedder())
    stats = reindex(polyglot, embedder=FakeEmbedder(), extra_suffixes=TS)
    assert stats.files_embedded == 1
    assert stats.files_unchanged == 3


@pytest.mark.parametrize(("argv", "expected"), [
    (None, None),                                  # flag absent: keep the index's
    (["ts"], frozenset({".ts"})),                  # dot is optional
    ([".ts", ".tsx"], frozenset({".ts", ".tsx"})),
    (["none"], frozenset()),                       # explicit "drop the extras"
    (["NONE"], frozenset()),
])
def test_suffix_parsing(argv: list[str] | None, expected: frozenset[str] | None) -> None:
    assert _parse_suffixes(argv) == expected


@pytest.mark.parametrize("argv", [["none", ".ts"], [""], ["."], ["a/b"]])
def test_suffix_parsing_rejects_nonsense(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        _parse_suffixes(argv)


def test_cli_dry_run_reports_the_skipped_suffixes(
        polyglot: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(polyglot), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "пропущено по расширению" in out
    assert ".ts 1" in out and ".css 1" in out
    assert "--suffix" in out, "the report has to say what to do about it"


def test_cli_dry_run_names_the_extra_suffixes_in_effect(
        polyglot: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(polyglot), "--dry-run", "--suffix", ".ts"]) == 0
    out = capsys.readouterr().out
    assert "дополнительные расширения: .ts" in out
    assert "пропущено по расширению: 1 файл(ов) — .css 1" in out, "the .ts is in now"


def test_cli_rejects_a_bad_suffix(polyglot: Path,
                                 capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(polyglot), "--suffix", "none", "--suffix", ".ts"]) == 2
    assert "нельзя совмещать" in capsys.readouterr().err


def test_a_generated_bundle_is_indexed_as_zero_chunks(polyglot: Path) -> None:
    (polyglot / "web" / "app.min.js").write_text(
        "!function(){" + "a=a+1;" * 2000 + "}();\n", encoding="utf-8")
    fake = FakeEmbedder()
    stats = reindex(polyglot, embedder=fake, extra_suffixes=frozenset({".js"}))
    assert stats.files_without_chunks == 1
    assert "не дали ни одного" in stats.summary()
    assert fake.documents == stats.chunks_embedded, "nothing was sent to Ollama for it"
