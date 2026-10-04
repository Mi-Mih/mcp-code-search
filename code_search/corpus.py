"""Decide which files belong in the index, and detect when they change.

File enumeration goes through git, so `.gitignore` is honoured for free and a
`venv/` full of third-party code never reaches the embedder. Untracked-but-not-
ignored files are included too: work in progress is exactly what you search for.
A non-git directory falls back to a manual walk.

Staleness is `mtime_ns` + `size` first, content hash only when those differ --
the hash costs a full read, and on a 61k-LOC project most files are unchanged.

Which suffixes count as source is not decided here. Only `.py`, `.rst` and `.md`
have a chunker of their own; anything else has to be opted into, and the set of
extra suffixes travels with the index rather than with the command line, so that
a background refresh does not disagree with the build about what the corpus is.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from code_search.chunker import SUPPORTED_SUFFIXES

__all__ = ["FileState", "list_files", "read_text", "skipped_by_suffix", "state_of"]

# Directories git cannot know to skip (it happily tracks a vendored copy) and
# that are never worth embedding. Matched against any path segment.
EXCLUDED_SEGMENTS = frozenset({
    ".code-search", "__pycache__", "node_modules", "site-packages",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})

_GIT_TIMEOUT = 60


@dataclass(frozen=True)
class FileState:
    """Cheap identity of a file: what we compare to decide "did this change"."""

    mtime_ns: int
    size: int


def _git(root: Path, *args: str) -> list[str] | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            timeout=_GIT_TIMEOUT, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.splitlines()


def _wanted(rel: str, excludes: tuple[str, ...]) -> bool:
    """Everything except the suffix test: segments and caller's exclude globs."""
    if any(part in EXCLUDED_SEGMENTS for part in PurePosixPath(rel).parts):
        return False
    return not any(fnmatch(rel, pattern) for pattern in excludes)


def _enumerate(root: Path) -> list[str]:
    tracked = _git(root, "ls-files", "-z")
    if tracked is None:
        found = (p for p in root.rglob("*") if p.is_file())
        return [p.relative_to(root).as_posix() for p in found]
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z") or []
    # -z means one NUL-separated record; splitlines() gave us that record back.
    return [name for line in [*tracked, *untracked] for name in line.split("\0") if name]


def list_files(root: Path, excludes: tuple[str, ...] = (),
               extra_suffixes: frozenset[str] = frozenset()) -> list[str]:
    """Repository-relative paths to index, sorted, deduplicated.

    `excludes` are fnmatch patterns applied to the relative path. `extra_suffixes`
    widens the corpus beyond the suffixes that have a dedicated chunker; it is
    additive, so `.py` and the documentation formats are always in.
    """
    accept = SUPPORTED_SUFFIXES | extra_suffixes
    return sorted({
        rel for rel in _enumerate(root)
        if PurePosixPath(rel).suffix in accept and _wanted(rel, excludes)
    })


def skipped_by_suffix(root: Path, excludes: tuple[str, ...] = (),
                      extra_suffixes: frozenset[str] = frozenset()) -> dict[str, int]:
    """How many otherwise-eligible files each unindexed suffix accounts for.

    Exists so that a polyglot repository cannot be half-indexed in silence: the
    suffix filter is the one exclusion the caller never asked for and would
    otherwise have to infer from a file count. Sorted by count, descending.

    Not called from the query path -- it re-enumerates the tree, and `plan` runs
    on every search.
    """
    accept = SUPPORTED_SUFFIXES | extra_suffixes
    counts: dict[str, int] = {}
    for rel in _enumerate(root):
        suffix = PurePosixPath(rel).suffix
        if suffix in accept or not _wanted(rel, excludes):
            continue
        key = suffix or "(без расширения)"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def state_of(root: Path, rel: str) -> FileState | None:
    """Current mtime/size, or None if the file has disappeared."""
    try:
        st = (root / rel).stat()
    except OSError:
        return None
    return FileState(mtime_ns=st.st_mtime_ns, size=st.st_size)


def read_text(root: Path, rel: str) -> str:
    """Read a source file, tolerating stray bytes rather than failing the run."""
    return (root / rel).read_text(encoding="utf-8", errors="replace")


def content_hash(text: str) -> str:
    """Short content digest, stored per file and returned to the caller.

    The caller uses it to tell whether the line numbers it was given still
    describe the file on disk (see ``docs/ARCHITECTURE.md``,
    "Актуальность и конкурентность").
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
