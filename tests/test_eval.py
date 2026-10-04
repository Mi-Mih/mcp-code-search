"""Harness tests.

A metric bug is the worst kind of bug in this project: it does not break
anything, it just tells us the wrong thing about whether the whole idea works.
These check the metrics against hand-computed values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import harness  # noqa: E402

from code_search.chunker import Chunk  # noqa: E402
from code_search.search import Hit  # noqa: E402
from code_search.store import StoredChunk  # noqa: E402


def hit(path: str) -> Hit:
    return Hit(chunk=StoredChunk(id=1, path=path, kind="source", start_line=1,
                                 end_line=1, symbol=None, text="x", file_hash="h"),
               score=1.0)


def outcome(kind: str, lexical: int | None, hybrid: int | None) -> harness.Outcome:
    result = harness.Outcome(query=harness.Query("q", ("a.py",), kind))
    result.ranks = {"lexical": lexical, "vector": None, "hybrid": hybrid}
    return result


# --------------------------------------------------------------------------- #
# what counts as a hit
# --------------------------------------------------------------------------- #


def test_exact_file_must_match_exactly() -> None:
    assert harness.matches("src/pkg/base.py", "src/pkg/base.py")
    assert not harness.matches("src/pkg/base_extra.py", "src/pkg/base.py")


def test_trailing_slash_matches_the_whole_directory() -> None:
    assert harness.matches("src/pkg/nested/utils.py", "src/pkg/nested/")
    assert not harness.matches("src/pkg/base.py", "src/pkg/nested/")


def test_rank_is_one_based() -> None:
    assert harness.hit_rank([hit("a.py"), hit("b.py")], ["a.py"]) == 1
    assert harness.hit_rank([hit("a.py"), hit("b.py")], ["b.py"]) == 2


def test_only_the_first_correct_hit_counts() -> None:
    hits = [hit("x.py"), hit("b.py"), hit("b.py")]
    assert harness.hit_rank(hits, ["b.py"]) == 2


def test_any_of_several_expected_locations_will_do() -> None:
    assert harness.hit_rank([hit("c.py")], ["a.py", "c.py"]) == 1


def test_a_miss_is_none_not_zero() -> None:
    """Zero would silently average in as a perfect reciprocal rank of 1/0."""
    assert harness.hit_rank([hit("x.py")], ["a.py"]) is None
    assert harness.hit_rank([], ["a.py"]) is None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_recall_counts_hits_within_the_cutoff() -> None:
    assert harness.recall_at([1, 5, 11, None], 10) == 0.5


@pytest.mark.parametrize(("k", "expected"), [(1, 0.5), (3, 0.5), (10, 1.0)])
def test_recall_tightens_as_the_cutoff_shrinks(k: int, expected: float) -> None:
    assert harness.recall_at([1, 7], k) == expected


def test_mrr_rewards_the_top_position() -> None:
    assert harness.mrr_at([1, 2], 10) == pytest.approx(0.75)


def test_mrr_ignores_hits_below_the_cutoff() -> None:
    assert harness.mrr_at([11], 10) == 0.0


def test_empty_sets_are_zero_not_an_error() -> None:
    assert (harness.recall_at([], 10), harness.mrr_at([], 10)) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #


def test_gate_looks_only_at_conceptual_queries() -> None:
    """Winning on exact-identifier queries proves nothing; grep already wins those."""
    outcomes = [outcome("concept", None, 1), outcome("exact", 1, 50)]
    _, recall_gain, _, counted = harness.gate_verdict(outcomes)
    assert (counted, recall_gain) == (1, 100.0)


def test_either_threshold_passing_is_enough() -> None:
    recall_only = [outcome("concept", None, 1)] + [outcome("concept", 1, 1)] * 4
    passed, recall_gain, _, _ = harness.gate_verdict(recall_only)
    assert passed and recall_gain == pytest.approx(20.0)


def test_a_tie_does_not_pass() -> None:
    passed, recall_gain, mrr_gain, _ = harness.gate_verdict([outcome("concept", 1, 1)])
    assert (passed, recall_gain, mrr_gain) == (False, 0.0, 0.0)


def test_ranking_higher_at_equal_recall_can_pass_on_mrr() -> None:
    outcomes = [outcome("concept", 10, 1)] * 3
    passed, recall_gain, mrr_gain = harness.gate_verdict(outcomes)[:3]
    assert recall_gain == 0.0
    assert passed and mrr_gain == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def test_queries_load_with_their_labels(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text('- query: "где разбор"\n  expect: ["src/a.py"]\n  kind: concept\n',
                    encoding="utf-8")
    assert harness.load_queries(path) == [
        harness.Query("где разбор", ("src/a.py",), "concept")]


def test_a_query_missing_its_label_is_an_error(tmp_path: Path) -> None:
    """Silently defaulting `kind` would quietly change what the gate measures."""
    path = tmp_path / "q.yaml"
    path.write_text('- query: "x"\n  expect: ["a.py"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        harness.load_queries(path)


def test_an_empty_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        harness.load_queries(path)


def test_the_shipped_example_set_is_well_formed() -> None:
    """The one set in the repository, and the only documentation of the format.

    The sets the published numbers were measured on are gone: they were labelled
    against a closed project and every query described its internals. So this
    checks the example -- which exists to be copied, and would document the
    format wrongly if it drifted out of shape.
    """
    loaded = harness.load_queries(ROOT / "eval" / "queries.example.yaml")
    assert loaded, "пример пуст"
    assert all(q.kind in harness.KINDS for q in loaded)
    assert all(q.expect for q in loaded)
    assert {q.kind for q in loaded} == set(harness.KINDS), "не все виды представлены"


def test_the_example_sets_expected_locations_exist() -> None:
    """A typo in `expect` makes a query permanently unanswerable and skews the gate.

    Checkable without an environment variable now, and that is the point of
    labelling the example against this project: the previous version of this test
    needed a checkout of somebody else's repository and skipped without it, so the
    one invariant that catches a dead `expect` path never actually ran here.
    """
    for query in harness.load_queries(ROOT / "eval" / "queries.example.yaml"):
        for location in query.expect:
            assert (ROOT / location).exists(), location


def test_chunk_and_stored_chunk_agree_on_paths() -> None:
    """The harness compares Hit paths to expect strings; both must be repo-relative."""
    assert Chunk(path="src/a.py", kind="source", start_line=1, end_line=1,
                 text="x").path == hit("src/a.py").chunk.path
