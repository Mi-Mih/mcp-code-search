"""Split source files into retrieval chunks.

The chunker is the only language-dependent part of the index. Everything else
(storage, invalidation, ranking) works on `Chunk` objects and never looks at the
source language, so adding a language means registering one more function in
`_CHUNKERS` -- see `chunk_text`.

Boundaries, in order of preference:

  .py    AST nodes: one chunk per function, one per method, one class header,
         plus the module-level remainder (imports, constants, top-level code).
  .rst   Sections, delimited by underlined titles.
  .md    Sections, delimited by ATX headings.
  other  Blank-line blocks, packed to size -- `_chunk_generic`, reached only for
         suffixes the caller opted into (`fallback=True`).
  any    Blank-line blocks, then a hard line window, when a unit is oversized.

`_chunk_generic` is what makes a non-Python language searchable at all, and it
is deliberately worse than a parser: no `symbol`, no signature, no docstring, so
the enrichment that `enrich` and `fts_text` build has less to work with. It is
offered because the alternative for those languages is nothing -- stdlib has an
AST for Python only, and a real parser for the rest means `tree-sitter`, which
this package cannot depend on (see README, "Известные ограничения").

`Chunk.text` is always a verbatim slice of the file. Enrichment (path, class,
symbol, signature, docstring) is produced separately by `enrich`, so the text
that is stored and shown to the caller is never modified.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Iterable, Iterator, Sequence, Union

__all__ = [
    "Chunk",
    "CHUNKER_VERSION",
    "GENERATED_LINE_CHARS",
    "SUPPORTED_SUFFIXES",
    "chunk_text",
    "classify",
    "enrich",
    "fts_text",
    "estimate_tokens",
]

# Bumped whenever chunk boundaries change; a stored index built by an older
# chunker must be discarded rather than silently mixed with new chunks.
#
# 2: `#` inside a fenced code block no longer reads as a markdown heading. That
#    moves `.md` boundaries, so version 1 indexes cannot be reused. The cost is
#    a full rebuild, which is why version 1 held on through changes that did not
#    strictly need it -- see ``docs/ARCHITECTURE.md``.
CHUNKER_VERSION = 2

TARGET_TOKENS = 512
MAX_TOKENS = 800

# A line this long means a generated or minified file, and `_chunk_generic`
# declines to index one. The reason is mechanical, not aesthetic: `_line_windows`
# cannot split inside a line, so a one-line 2 MB bundle would be embedded as a
# single chunk -- minutes of Ollama time for a row nobody will ever want to read.
# The dedicated chunkers deliberately do not apply this guard: changing what they
# emit would invalidate every index already built with CHUNKER_VERSION 1.
GENERATED_LINE_CHARS = 5000

# Directory names that mean "these are tests" across ecosystems. Python's own
# convention (`tests/`, `test_*.py`) was all `classify` knew, which mislabelled
# every `foo.spec.ts` as source once other languages became indexable.
_TEST_DIRS = frozenset({"tests", "test", "__tests__", "spec", "specs"})
_TEST_STEM_SUFFIXES = ("_test", ".test", ".spec", "_spec")

# There is deliberately no minimum chunk size. A size floor looks harmless and
# is not: a two-line `__repr__` becomes unfindable, and a settings module that
# is nothing but one-line constants disappears from the index entirely. The only
# filter is "contains something other than whitespace". Runs of module-level
# code are already maximal (see `_uncovered_runs`), so this costs few chunks.

_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_RST_UNDERLINE = re.compile(r"^[=\-~^\"'`#*+_:.]{3,}\s*$")
# A fenced code block, opening or closing. Up to three leading spaces, per
# CommonMark; the fence character matters and its length does not, which is a
# simplification -- a ``` inside a ~~~ block must not close it, but a longer
# closing fence than its opener is not something worth tracking here.
_MD_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_COMPOUND = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def estimate_tokens(text: str) -> int:
    """Approximate token count without loading a tokenizer.

    Cyrillic runs roughly 2.5 characters per token, Latin code roughly 3.7.
    Used only to pick chunk boundaries, so a few percent of error is harmless.
    """
    cyrillic = len(_CYRILLIC.findall(text))
    return int(cyrillic / 2.5 + (len(text) - cyrillic) / 3.7)


@dataclass(frozen=True)
class Chunk:
    """One retrievable fragment of one file.

    `start_line` and `end_line` are 1-based and inclusive, and describe the file
    as it was when chunked; callers compare the stored file hash before trusting
    them (see ``docs/ARCHITECTURE.md``, "Актуальность и конкурентность").
    """

    path: str
    kind: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None = None
    parent: str | None = None
    signature: str | None = None
    doc: str | None = None


Chunker = Callable[[str, str, str], "list[Chunk]"]


def classify(rel_path: str) -> str:
    """Map a repository-relative path to the `kind` stored on every chunk.

    `kind` is metadata: it reaches the caller and can be read, but it is not part
    of what gets embedded or indexed lexically. Widening the test conventions here
    therefore does not invalidate a stored index.
    """
    path = PurePosixPath(rel_path)
    stem = path.stem                        # "foo.test" for foo.test.ts
    if (_TEST_DIRS.intersection(path.parts)
            or stem.startswith("test_")
            or stem.endswith(_TEST_STEM_SUFFIXES)):
        return "test"
    if rel_path.endswith((".rst", ".md")):
        return "docs"
    return "source"


# --------------------------------------------------------------------------- #
# splitting helpers
# --------------------------------------------------------------------------- #


def _blank_line_blocks(lines: Sequence[str]) -> list[tuple[int, int]]:
    """Group lines into blank-line-separated blocks as (offset, length) pairs."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for i, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = i
        elif start is not None:
            blocks.append((start, i - start))
            start = None
    if start is not None:
        blocks.append((start, len(lines) - start))
    return blocks


def _pack(units: Iterable[tuple[int, int]], lines: Sequence[str],
          first_line: int) -> Iterator[tuple[int, int, str]]:
    """Greedily merge adjacent units up to TARGET_TOKENS.

    Yields (start_line, end_line, text). A single unit larger than MAX_TOKENS is
    emitted alone; the caller splits it further.
    """
    buf: list[tuple[int, int]] = []

    def flush() -> Iterator[tuple[int, int, str]]:
        if not buf:
            return
        lo = buf[0][0]
        hi = buf[-1][0] + buf[-1][1]
        yield first_line + lo, first_line + hi - 1, "\n".join(lines[lo:hi])
        buf.clear()

    for off, length in units:
        unit_text = "\n".join(lines[off:off + length])
        if buf:
            merged = "\n".join(lines[buf[0][0]:off + length])
            if estimate_tokens(merged) > TARGET_TOKENS:
                yield from flush()
        buf.append((off, length))
        if estimate_tokens(unit_text) >= TARGET_TOKENS:
            yield from flush()
    yield from flush()


def _line_windows(lines: Sequence[str], first_line: int) -> Iterator[tuple[int, int, str]]:
    """Last-resort split: fixed windows of lines, never exceeding TARGET_TOKENS."""
    start = 0
    for i in range(len(lines)):
        window = "\n".join(lines[start:i + 1])
        if estimate_tokens(window) > TARGET_TOKENS and i > start:
            yield first_line + start, first_line + i - 1, "\n".join(lines[start:i])
            start = i
    if start < len(lines):
        yield first_line + start, first_line + len(lines) - 1, "\n".join(lines[start:])


def _split_oversized(lines: Sequence[str], first_line: int) -> Iterator[tuple[int, int, str]]:
    """Recursive split: blank-line blocks first, then hard line windows."""
    blocks = _blank_line_blocks(lines)
    if len(blocks) > 1:
        for lo, hi, text in _pack(blocks, lines, first_line):
            if estimate_tokens(text) <= MAX_TOKENS:
                yield lo, hi, text
            else:
                yield from _line_windows(text.split("\n"), lo)
        return
    yield from _line_windows(lines, first_line)


def _emit(out: list[Chunk], path: str, kind: str, start: int, end: int, text: str,
          symbol: str | None = None, parent: str | None = None,
          signature: str | None = None, doc: str | None = None) -> None:
    """Append a chunk, splitting it first if it exceeds MAX_TOKENS."""
    if not text.strip():
        return
    if estimate_tokens(text) <= MAX_TOKENS:
        out.append(Chunk(path, kind, start, end, text, symbol, parent, signature, doc))
        return
    parts = list(_split_oversized(text.split("\n"), start))
    for index, (lo, hi, part) in enumerate(parts):
        if not part.strip():
            continue
        out.append(Chunk(
            path, kind, lo, hi, part,
            symbol=f"{symbol}#{index}" if symbol else None,
            parent=parent,
            signature=signature if index == 0 else None,
            doc=doc if index == 0 else None,
        ))


# --------------------------------------------------------------------------- #
# Python
# --------------------------------------------------------------------------- #

Definition = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]
_DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _leading_comments(lines: Sequence[str], start: int) -> int:
    """First line (1-based) of the comment block directly above line `start`."""
    i = start
    while i > 1 and lines[i - 2].lstrip().startswith("#"):
        i -= 1
    return i


def _span(node: ast.AST, lines: Sequence[str]) -> tuple[int, int]:
    """Line span of a definition, including decorators and leading comments."""
    starts = [getattr(node, "lineno", 1)]
    starts += [d.lineno for d in getattr(node, "decorator_list", [])]
    start = _leading_comments(lines, min(starts))
    return start, getattr(node, "end_lineno", start)


def _signature(node: ast.AST, lines: Sequence[str]) -> str:
    """The `def`/`class` header, without the body."""
    head = getattr(node, "lineno", 1)
    body = getattr(node, "body", None)
    stop = body[0].lineno - 1 if body else head
    text = " ".join(line.strip() for line in lines[head - 1:max(stop, head)])
    return text[:300]


def _chunk_python(path: str, text: str, kind: str) -> list[Chunk]:
    lines = text.split("\n")
    out: list[Chunk] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Unparseable file still deserves to be searchable.
        for lo, hi, part in _split_oversized(lines, 1):
            _emit(out, path, kind, lo, hi, part)
        return out

    covered: set[int] = set()

    def take(node: Definition, parent: str | None) -> None:
        start, end = _span(node, lines)
        covered.update(range(start, end + 1))
        _emit(out, path, kind, start, end, "\n".join(lines[start - 1:end]),
              symbol=node.name, parent=parent,
              signature=_signature(node, lines), doc=ast.get_docstring(node))

    def visit(body: Sequence[ast.stmt], parent: str | None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Nested functions stay inside their parent chunk on purpose.
                take(node, parent)
            elif isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, _DEFINITION_TYPES)]
                if not methods:
                    take(node, parent)
                    continue
                start, _ = _span(node, lines)
                header_end = min(_span(m, lines)[0] for m in methods) - 1
                if header_end >= start:
                    covered.update(range(start, header_end + 1))
                    _emit(out, path, kind, start, header_end,
                          "\n".join(lines[start - 1:header_end]),
                          symbol=node.name, parent=parent,
                          signature=_signature(node, lines),
                          doc=ast.get_docstring(node))
                visit(node.body, node.name)

    visit(tree.body, None)

    for lo, hi in _uncovered_runs(covered, len(lines)):
        block = "\n".join(lines[lo - 1:hi])
        if block.strip():
            _emit(out, path, kind, lo, hi, block)

    out.sort(key=lambda c: (c.start_line, c.end_line))
    return out


def _uncovered_runs(covered: set[int], total: int) -> Iterator[tuple[int, int]]:
    """Contiguous 1-based line ranges that no definition claimed."""
    start: int | None = None
    for i in range(1, total + 1):
        if i in covered:
            if start is not None:
                yield start, i - 1
                start = None
        elif start is None:
            start = i
    if start is not None:
        yield start, total


# --------------------------------------------------------------------------- #
# reStructuredText / Markdown
# --------------------------------------------------------------------------- #


def _chunk_rst(path: str, text: str, kind: str) -> list[Chunk]:
    lines = text.split("\n")
    starts = [0]
    for i in range(1, len(lines)):
        title = lines[i - 1].strip()
        if title and _RST_UNDERLINE.match(lines[i]) and len(lines[i]) >= len(title) - 2:
            starts.append(i - 1)
    return _sections(path, kind, lines, sorted(set(starts)),
                     title_of=lambda block: block[0].strip())


def _md_heading_lines(lines: Sequence[str]) -> list[int]:
    """Indices of ATX heading lines, ignoring anything inside a fenced block.

    The fence check is not cosmetic. `# поставить зависимости` at column 0 of a
    ```bash block reads as a heading without it: the section splits in the middle
    of the code block, and the shell comment becomes the chunk's title -- which
    then goes into the embedded text via `enrich` and into FTS via `fts_text`.
    Any README that documents commands hits this, including this project's own.

    An unclosed fence swallows the rest of the file, which is what CommonMark
    says it does.
    """
    out: list[int] = []
    fence: str | None = None
    for i, line in enumerate(lines):
        match = _MD_FENCE.match(line)
        if fence is None:
            if match:
                fence = match.group(1)[0]
            elif line.startswith("#"):
                out.append(i)
        elif match and match.group(1)[0] == fence:
            fence = None
    return out


def _chunk_markdown(path: str, text: str, kind: str) -> list[Chunk]:
    lines = text.split("\n")
    starts = [0] + _md_heading_lines(lines)
    return _sections(path, kind, lines, sorted(set(starts)),
                     title_of=lambda block: block[0].lstrip("# ").strip())


def _sections(path: str, kind: str, lines: Sequence[str], starts: Sequence[int],
              title_of: Callable[[Sequence[str]], str]) -> list[Chunk]:
    out: list[Chunk] = []
    bounds = list(starts) + [len(lines)]
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        block = lines[lo:hi]
        if not any(line.strip() for line in block):
            continue
        _emit(out, path, kind, lo + 1, hi, "\n".join(block),
              doc=title_of(block) or None)
    return out


# --------------------------------------------------------------------------- #
# any language
# --------------------------------------------------------------------------- #


def _chunk_generic(path: str, text: str, kind: str) -> list[Chunk]:
    """Language-agnostic split: blank-line blocks, greedily packed to size.

    Works because the convention it relies on is near-universal -- a blank line
    separates one declaration from the next in every curly-brace language anybody
    is likely to point this at. What it cannot do is name the declaration, so a
    query has only the file path and the code itself to match against.
    """
    lines = text.split("\n")
    if any(len(line) > GENERATED_LINE_CHARS for line in lines):
        return []
    out: list[Chunk] = []
    for start, end, block in _pack(_blank_line_blocks(lines), lines, 1):
        _emit(out, path, kind, start, end, block)
    return out


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

_CHUNKERS: dict[str, Chunker] = {
    ".py": _chunk_python,
    ".rst": _chunk_rst,
    ".md": _chunk_markdown,
}

SUPPORTED_SUFFIXES = frozenset(_CHUNKERS)


def chunk_text(rel_path: str, text: str, kind: str | None = None,
               fallback: bool = False) -> list[Chunk]:
    """Chunk one file's contents.

    `rel_path` is repository-relative and decides both the chunker and, unless
    overridden, the `kind` recorded on every chunk.

    A suffix with no dedicated chunker yields nothing unless `fallback` is set,
    and the decision is the caller's rather than this module's on purpose: the
    set of extra suffixes is recorded in the index (`store.extra_suffixes`) so
    that a background refresh chunks a `.ts` file the same way the build did.
    Defaulting to True here would instead make every unknown binary in the tree
    a candidate for embedding.
    """
    suffix = PurePosixPath(rel_path).suffix
    chunker = _CHUNKERS.get(suffix)
    if chunker is None:
        if not fallback:
            return []
        chunker = _chunk_generic
    return chunker(rel_path, text, kind if kind is not None else classify(rel_path))


def enrich(chunk: Chunk) -> str:
    """Build the text that is embedded: retrieval context, then the original.

    The path is spelled out in words so that abbreviated module names carry
    meaning -- `core_model/queue_load/ql_limits.py` becomes
    "core model / queue load / ql limits", which a query phrased as
    "ограничения на длину очереди" can actually match.
    """
    readable = PurePosixPath(chunk.path).with_suffix("").as_posix()
    head = [readable.replace("/", " / ").replace("_", " ")]
    if chunk.parent:
        head.append(f"class {chunk.parent}")
    if chunk.symbol:
        head.append(f"symbol {chunk.symbol.split('#')[0]}")
    if chunk.signature:
        head.append(chunk.signature)
    if chunk.doc:
        head.append(chunk.doc[:600])
    return " | ".join(head) + "\n" + chunk.text


def fts_text(chunk: Chunk) -> str:
    """Build the text that goes into FTS5.

    Compound identifiers are appended a second time in split form, so that
    `set_deficit_mode` is reachable by the word "deficit" without the original
    spelling being lost -- a query for the exact identifier must still match.
    `Chunk.text` itself is never rewritten.
    """
    base = f"{chunk.path} {chunk.doc or ''} {chunk.text}"
    extra: list[str] = []
    for token in sorted(set(_COMPOUND.findall(base))):
        parts = [p for p in _CAMEL_BOUNDARY.sub(" ", token).replace("_", " ").split()
                 if len(p) > 1]
        if len(parts) > 1:
            extra.append(" ".join(parts))
    return " ".join([base, *extra])
