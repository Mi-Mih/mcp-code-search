"""Chunker tests.

Four invariants matter downstream and are checked on every sample here:

  I1  `text` is a verbatim slice of the file at [start_line, end_line].
  I2  chunks of one file never overlap.
  I3  no chunk exceeds MAX_TOKENS.
  I4  every non-trivial line of the file lands in some chunk.

Everything else is a boundary-placement test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_search.chunker import (  # noqa: E402
    GENERATED_LINE_CHARS,
    MAX_TOKENS,
    Chunk,
    chunk_text,
    classify,
    enrich,
    estimate_tokens,
)

# --------------------------------------------------------------------------- #
# invariants
# --------------------------------------------------------------------------- #


def check_invariants(text: str, chunks: list[Chunk]) -> None:
    lines = text.split("\n")
    for c in chunks:
        assert 1 <= c.start_line <= c.end_line <= len(lines), c
        assert c.text == "\n".join(lines[c.start_line - 1:c.end_line]), c  # I1
        assert estimate_tokens(c.text) <= MAX_TOKENS, c                    # I3

    spans = sorted((c.start_line, c.end_line) for c in chunks)
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert next_start > prev_end, (spans, "overlap")                   # I2


def covered_lines(chunks: list[Chunk]) -> set[int]:
    return {n for c in chunks for n in range(c.start_line, c.end_line + 1)}


SAMPLE = '''\
"""Module docstring."""

import os

CONST = 1


def top_level(a, b):
    """Adds."""
    return a + b


class Thing:
    """A thing."""

    field = 0

    def method(self):
        """Does."""
        return self.field

    @property
    def prop(self):
        return 1


TRAILING = 2
'''


def test_sample_invariants() -> None:
    chunks = chunk_text("src/sample.py", SAMPLE)
    check_invariants(SAMPLE, chunks)


def test_sample_covers_every_code_line() -> None:
    chunks = chunk_text("src/sample.py", SAMPLE)
    covered = covered_lines(chunks)
    for n, line in enumerate(SAMPLE.split("\n"), start=1):
        if line.strip():
            assert n in covered, f"line {n} lost: {line!r}"               # I4


def test_symbols_and_parents() -> None:
    chunks = chunk_text("src/sample.py", SAMPLE)
    named = {c.symbol: c.parent for c in chunks if c.symbol}
    assert named == {"top_level": None, "Thing": None, "method": "Thing", "prop": "Thing"}


def test_class_header_is_separate_from_methods() -> None:
    chunks = chunk_text("src/sample.py", SAMPLE)
    header = next(c for c in chunks if c.symbol == "Thing")
    assert "class Thing" in header.text
    assert "field = 0" in header.text
    assert "def method" not in header.text


def test_signature_and_docstring_captured() -> None:
    chunk = next(c for c in chunk_text("src/s.py", SAMPLE) if c.symbol == "top_level")
    assert chunk.signature == "def top_level(a, b):"
    assert chunk.doc == "Adds."


def test_decorator_included() -> None:
    chunk = next(c for c in chunk_text("src/s.py", SAMPLE) if c.symbol == "prop")
    assert chunk.text.lstrip().startswith("@property")


def test_leading_comments_included() -> None:
    text = "x = 0\n\n# why this exists\n# second line\ndef f():\n    return 1\n"
    chunk = next(c for c in chunk_text("src/s.py", text) if c.symbol == "f")
    assert chunk.start_line == 3
    assert chunk.text.startswith("# why this exists")


def test_nested_function_not_chunked_separately() -> None:
    text = (
        "def outer():\n"
        '    """Outer."""\n'
        "    def inner():\n"
        "        return 1\n"
        "    return inner() + " + " + ".join("1" * 1 for _ in range(30)) + "\n"
    )
    chunks = chunk_text("src/s.py", text)
    assert [c.symbol for c in chunks] == ["outer"]
    assert "def inner" in chunks[0].text


def test_module_remainder_has_no_symbol() -> None:
    chunks = chunk_text("src/sample.py", SAMPLE)
    remainder = [c for c in chunks if c.symbol is None]
    assert remainder, "imports and constants must be searchable"
    assert any("CONST = 1" in c.text for c in remainder)


# --------------------------------------------------------------------------- #
# oversized units
# --------------------------------------------------------------------------- #

BIG_FUNCTION = "def huge():\n" + "".join(
    f"    value_{i} = compute_something(argument_{i}, other_{i})\n" for i in range(400)
)


def test_oversized_function_is_split() -> None:
    chunks = chunk_text("src/big.py", BIG_FUNCTION)
    assert len(chunks) > 1
    check_invariants(BIG_FUNCTION, chunks)
    assert covered_lines(chunks) >= {n for n in range(1, 401)}


def test_split_parts_are_suffixed_and_ordered() -> None:
    chunks = chunk_text("src/big.py", BIG_FUNCTION)
    assert [c.symbol for c in chunks] == [f"huge#{i}" for i in range(len(chunks))]
    assert chunks[0].signature == "def huge():"
    assert all(c.signature is None for c in chunks[1:])


def test_single_line_longer_than_budget_is_kept_whole() -> None:
    text = "x = " + "'" + "a" * 20000 + "'\n"
    chunks = chunk_text("src/s.py", text)
    assert len(chunks) == 1
    assert chunks[0].text.rstrip("\n") == text.rstrip("\n")


def test_blank_line_blocks_preferred_over_windows() -> None:
    body = "\n\n".join("\n".join(f"row_{b}_{i} = {i}" for i in range(20)) for b in range(20))
    chunks = chunk_text("src/s.py", body)
    check_invariants(body, chunks)
    # A blank-line split never cuts a block in half.
    for c in chunks:
        assert not c.text.startswith("\n")


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #


def test_syntax_error_still_yields_chunks() -> None:
    broken = "def f(:\n" + "    junk line here\n" * 60
    chunks = chunk_text("src/broken.py", broken)
    assert chunks
    check_invariants(broken, chunks)


def test_empty_and_whitespace_only_files() -> None:
    assert chunk_text("src/empty.py", "") == []
    assert chunk_text("src/blank.py", "\n\n   \n\n") == []


def test_settings_module_of_one_line_constants_is_indexed() -> None:
    """Regression: a size floor used to erase files like `src/config.py`."""
    text = "IS_ENABLED = 1\n\nRETRY_LIMIT = 2\n\nMAX_WORKERS = 3\n"
    chunks = chunk_text("src/config.py", text)
    assert len(chunks) == 1, "adjacent constants belong in one chunk"
    check_invariants(text, chunks)
    for name in ("IS_ENABLED", "RETRY_LIMIT", "MAX_WORKERS"):
        assert name in chunks[0].text


def test_short_definition_is_kept() -> None:
    text = "class A:\n    def __repr__(self):\n        return 'A'\n"
    assert "__repr__" in {c.symbol for c in chunk_text("src/s.py", text)}


def test_trailing_module_code_after_a_class_is_kept() -> None:
    chunks = chunk_text("src/sample.py", SAMPLE)
    assert any("TRAILING = 2" in c.text for c in chunks)


def test_unsupported_suffix() -> None:
    assert chunk_text("src/main.kt", "fun main() {}\n" * 50) == []


# --------------------------------------------------------------------------- #
# any language (fallback=True)
# --------------------------------------------------------------------------- #

TYPESCRIPT = """\
import { Injectable } from '@angular/core';

export interface RouteLeg {
  origin: string;
  target: string;
  volume: number;
}

@Injectable({ providedIn: 'root' })
export class RoutePlanner {
  private legs: RouteLeg[] = [];

  addLeg(leg: RouteLeg): void {
    this.legs.push(leg);
  }

  totalVolume(): number {
    return this.legs.reduce((sum, leg) => sum + leg.volume, 0);
  }
}
"""


def test_fallback_is_opt_in() -> None:
    """The default must stay closed: `corpus` decides what is source, not this."""
    assert chunk_text("src/planner.ts", TYPESCRIPT) == []
    assert chunk_text("src/planner.ts", TYPESCRIPT, fallback=True) != []


def test_generic_chunks_obey_the_same_invariants() -> None:
    check_invariants(TYPESCRIPT, chunk_text("src/planner.ts", TYPESCRIPT, fallback=True))


def test_a_small_generic_file_is_one_chunk() -> None:
    """Coarser than the AST chunker on purpose: blocks pack up to TARGET_TOKENS.

    A 20-line class is ~120 tokens, so it stays whole. The Python chunker would
    have emitted a chunk per method -- that difference is the price of having no
    parser, and it is recorded here rather than left to be discovered.
    """
    assert len(chunk_text("src/planner.ts", TYPESCRIPT, fallback=True)) == 1


def test_generic_splits_a_large_file_on_blank_lines() -> None:
    text = "\n".join(TYPESCRIPT.replace("RoutePlanner", f"RoutePlanner{i}")
                     for i in range(12))
    chunks = chunk_text("src/planner.ts", text, fallback=True)
    check_invariants(text, chunks)
    assert len(chunks) > 1, "a 240-line file as one chunk would defeat ranking"
    assert all(estimate_tokens(c.text) <= MAX_TOKENS for c in chunks)
    # A boundary must fall between blocks, never inside a line.
    assert all(c.text == c.text.strip("\n") for c in chunks)


def test_generic_carries_no_symbol_names() -> None:
    """Honest about what it cannot do: no parser, so no symbol or signature."""
    chunks = chunk_text("src/planner.ts", TYPESCRIPT, fallback=True)
    assert all(c.symbol is None and c.signature is None for c in chunks)


def test_a_dedicated_chunker_still_wins_over_the_fallback() -> None:
    with_fallback = chunk_text("src/sample.py", SAMPLE, fallback=True)
    assert with_fallback == chunk_text("src/sample.py", SAMPLE)
    assert any(c.symbol for c in with_fallback), "AST chunker, not the generic one"


def test_a_minified_bundle_is_declined() -> None:
    """One 2 MB line cannot be split, and embedding it whole is minutes wasted."""
    bundle = "!function(){var a=1;" + "b=b+1;" * 2000 + "}();\n"
    assert len(bundle.split("\n")[0]) > GENERATED_LINE_CHARS
    assert chunk_text("static/app.min.js", bundle, fallback=True) == []


def test_a_long_file_of_short_lines_is_not_mistaken_for_a_bundle() -> None:
    source = "".join(f"const value{i} = {i};\n" for i in range(4000))
    chunks = chunk_text("src/constants.ts", source, fallback=True)
    assert chunks
    assert all(estimate_tokens(c.text) <= MAX_TOKENS for c in chunks)


def test_generic_empty_and_whitespace_files_yield_nothing() -> None:
    assert chunk_text("src/blank.ts", "", fallback=True) == []
    assert chunk_text("src/blank.ts", "\n\n   \n", fallback=True) == []


def test_deterministic() -> None:
    assert chunk_text("src/sample.py", SAMPLE) == chunk_text("src/sample.py", SAMPLE)


def test_crlf_and_trailing_whitespace_survive_round_trip() -> None:
    text = SAMPLE.replace("\n", "\r\n")
    check_invariants(text, chunk_text("src/sample.py", text))


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #

RST = """\
Заголовок документа
===================

Вводный абзац про планирование рейсов, достаточно длинный для порога.

Первый раздел
-------------

Текст первого раздела, тоже длиной больше шестидесяти символов подряд.

Второй раздел
-------------

Текст второго раздела, тоже длиной больше шестидесяти символов подряд.
"""


def test_rst_splits_on_underlined_titles() -> None:
    chunks = chunk_text("docs/guide.rst", RST)
    check_invariants(RST, chunks)
    assert [c.doc for c in chunks] == [
        "Заголовок документа", "Первый раздел", "Второй раздел",
    ]
    assert all(c.kind == "docs" for c in chunks)


def test_markdown_splits_on_headings() -> None:
    md = "# Title\n\n" + "Intro paragraph long enough to pass the threshold here.\n\n"
    md += "## Section\n\n" + "Body text long enough to pass the minimum size threshold.\n"
    chunks = chunk_text("README.md", md)
    check_invariants(md, chunks)
    assert [c.doc for c in chunks] == ["Title", "Section"]


def test_rst_without_titles_is_one_chunk() -> None:
    plain = "Просто текст без единого заголовка, но достаточно длинный для порога.\n"
    assert len(chunk_text("doc.rst", plain)) == 1


# --------------------------------------------------------------------------- #
# markdown fences
#
# A `#` at column 0 inside a fenced block is a comment, not a heading. Reading it
# as one splits the section mid-block and puts the comment into `doc`, from where
# it reaches both the embedded text (`enrich`) and FTS (`fts_text`). Every README
# that documents shell commands trips this, including this project's own.
# --------------------------------------------------------------------------- #


def test_a_comment_in_a_fenced_block_is_not_a_heading() -> None:
    md = ("# Установка\n\nТекст раздела достаточной длины для порога.\n\n"
          "```bash\n# поставить зависимости\npip install foo\n```\n\n"
          "Продолжение того же раздела, тоже достаточной длины.\n")
    chunks = chunk_text("README.md", md)
    check_invariants(md, chunks)
    assert [c.doc for c in chunks] == ["Установка"]


def test_a_tilde_fence_is_not_closed_by_a_backtick_one() -> None:
    md = ("# А\n\n~~~\n# не заголовок\n```\n# тоже не заголовок\n~~~\n\n"
          "# Б\n\nтело раздела достаточной длины для порога.\n")
    chunks = chunk_text("README.md", md)
    check_invariants(md, chunks)
    assert [c.doc for c in chunks] == ["А", "Б"]


def test_an_indented_fence_still_opens_a_block() -> None:
    """CommonMark allows up to three spaces before a fence."""
    md = "# А\n\n   ```\n# внутри блока\n   ```\n\n# Б\n\nтело раздела.\n"
    assert [c.doc for c in chunk_text("README.md", md)] == ["А", "Б"]


def test_an_unclosed_fence_swallows_the_rest_of_the_file() -> None:
    """Which is what CommonMark says it does."""
    md = "# А\n\n```\n# внутри\n\n# и это тоже внутри\n"
    assert [c.doc for c in chunk_text("README.md", md)] == ["А"]


def test_headings_still_split_when_no_fence_is_involved() -> None:
    """The fence tracking must not cost the ordinary case."""
    md = "# А\n\nтело.\n\n## Б\n\nтело.\n\n### В\n\nтело.\n"
    assert [c.doc for c in chunk_text("README.md", md)] == ["А", "Б", "В"]


def _lines_inside_a_fence(lines: list[str]) -> set[int]:
    """1-based line numbers that sit inside a fenced block, counted independently
    of the chunker so the assertion below is about the file, not about the helper.
    """
    inside: set[int] = set()
    fence: str | None = None
    for number, line in enumerate(lines, start=1):
        stripped = line.lstrip(" ")
        opener = stripped[:3] if stripped[:3] in ("```", "~~~") else None
        if fence is None:
            if opener:
                fence = opener[0]
        elif opener and opener[0] == fence:
            fence = None
            continue
        if fence is not None:
            inside.add(number)
    return inside


@pytest.mark.parametrize(
    "name",
    ["README.md", "SETUP.md", "CONTRIBUTING.md", "docs/ARCHITECTURE.md", "docs/API.md"],
)
def test_this_projects_own_docs_keep_their_code_blocks_whole(name: str) -> None:
    """A canary over the real files, not the repro for the bug.

    It passes on the pre-fix chunker too: the comments in these documents sit at
    the end of a command (`ollama pull bge-m3   # ~1.2 ГБ`), never at column 0,
    so the shape that triggers the bug happens not to occur here today. The test
    is here so that adding a block that *does* start with `# поставить ...` --
    the ordinary way to write one -- fails loudly instead of quietly costing the
    docs index a section boundary.

    Every maintained Markdown document is listed explicitly so a broken fence in
    user-facing documentation fails the suite instead of silently changing the
    search chunks.
    """
    path = Path(__file__).resolve().parents[1] / name
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(name, text)
    check_invariants(text, chunks)
    inside = _lines_inside_a_fence(text.split("\n"))
    offenders = [(c.start_line, c.doc) for c in chunks if c.start_line in inside]
    assert not offenders, f"границы чанков попали внутрь кодового блока: {offenders}"


# --------------------------------------------------------------------------- #
# classification and enrichment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("path", "kind"), [
    ("src/core_model/solver.py", "source"),
    ("tests/test_solver.py", "test"),
    ("src/tests/helpers.py", "test"),
    ("src/test_helpers.py", "test"),
    ("docs/index.rst", "docs"),
    ("README.md", "docs"),
    # Conventions from the languages the generic chunker makes reachable.
    ("web/src/planner.spec.ts", "test"),
    ("web/src/planner.test.tsx", "test"),
    ("web/__tests__/planner.ts", "test"),
    ("internal/route/planner_test.go", "test"),
    ("spec/models/route.rb", "test"),
    ("web/src/planner.ts", "source"),
    ("web/src/latest.ts", "source"),
])
def test_classify(path: str, kind: str) -> None:
    assert classify(path) == kind


def test_kind_override_wins() -> None:
    chunks = chunk_text("src/sample.py", SAMPLE, kind="docs")
    assert all(c.kind == "docs" for c in chunks)


def test_enrich_layout() -> None:
    chunk = Chunk(
        path="src/core_model/queue_load/ql_limits.py",
        kind="source", start_line=10, end_line=12, text="        return 1",
        symbol="set_overflow_mode", parent="QueueLoadLimits",
        signature="def set_overflow_mode(self):", doc="Ограничение для переполнения.",
    )
    assert enrich(chunk) == (
        "src / core model / queue load / ql limits"
        " | class QueueLoadLimits"
        " | symbol set_overflow_mode"
        " | def set_overflow_mode(self):"
        " | Ограничение для переполнения."
        "\n        return 1"
    )


def test_enrich_keeps_text_verbatim_and_last() -> None:
    for chunk in chunk_text("src/sample.py", SAMPLE):
        assert enrich(chunk).endswith(chunk.text)


def test_enrich_drops_split_suffix() -> None:
    chunk = chunk_text("src/big.py", BIG_FUNCTION)[1]
    assert chunk.symbol == "huge#1"
    assert "huge#1" not in enrich(chunk)
    assert enrich(chunk).count("symbol huge") == 1
