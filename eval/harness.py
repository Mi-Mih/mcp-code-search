"""Measure search quality against a labelled query set.

Not part of the shipped server -- this is the instrument that decides whether
the server is worth shipping at all, and the gate it checks was fixed before
the retrieval pipeline was tuned:

    on conceptual queries, hybrid must beat BM25 alone by at least
    +15 percentage points of Recall@10, or by at least +0.15 MRR@10.

A query counts as answered when any returned chunk comes from an expected
location. `expect` entries ending in "/" match a whole directory, everything
else matches one exact file. That is deliberately generous: the user's next
action is to open a file, so landing anywhere in the right file is a success
even if the chunk boundaries fell awkwardly.

Two things this harness must not do, and does not:

  * Report a mode as better because it was measured differently. All three
    modes get the same query set, the same depth, and no output budget.
  * Touch a holdout set casually. Any query file with "holdout" in its name is
    scored once, after the pipeline is frozen; running it requires --unblind
    and says so.

The query sets the published numbers were measured on are not in this
repository -- they were labelled against a closed project, and both the queries
and the `expect` paths described its internals. `queries.example.yaml` documents
the format and is labelled against this project's own code instead; supply your
own set to measure anything else. What survives from the original run is
`holdout_results.json`, which carries the measurements and nothing about what
was asked (see `as_json`).

Measurement depth is 50 while the metrics cut at 10, so fusion draws from a
deeper pool here than in production (k=15). That is only safe because the
lexical arm's contribution to fusion is capped at a fixed few ranks
(`search.LEXICAL_POOL`): extra dense candidates enter below everything the
lexical head contributes and can only extend the tail. It was *not* safe before
that cap -- measuring at k=50 then really did report different top-10 rankings
than production would have produced, by up to 5 points of Recall@10.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import yaml  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_search import store  # noqa: E402
from code_search.chunker import estimate_tokens  # noqa: E402
from code_search.indexer import index_path, plan  # noqa: E402
from code_search.search import MODES, Hit, Mode, Searcher  # noqa: E402

# The pre-registered gate. Either threshold passing is enough; they measure the
# same thing from different angles (did we find it at all / how far down).
GATE_KIND = "concept"
GATE_RECALL_GAIN = 15.0
GATE_MRR_GAIN = 0.15

DEPTH = 50
CUTOFFS = (1, 3, 5, 10)
KINDS = ("concept", "domain", "docs", "test", "exact")
LABELS: dict[Mode, str] = {"lexical": "BM25", "vector": "вектор", "hybrid": "гибрид"}
# Worst first, so that each row of a report reads as an improvement on the one above.
ORDER: tuple[Mode, ...] = ("lexical", "vector", "hybrid")


@dataclass(frozen=True)
class Query:
    """One labelled search request and its acceptable repository locations."""

    query: str
    expect: tuple[str, ...]
    kind: str


@dataclass
class Outcome:
    """Where each mode placed the first correct chunk, and what that cost."""

    query: Query
    ranks: dict[Mode, int | None] = field(default_factory=dict)
    seconds: dict[Mode, float] = field(default_factory=dict)
    tokens: dict[int, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# labelled data
# --------------------------------------------------------------------------- #


def load_queries(path: Path) -> list[Query]:
    """Load and validate a labelled YAML query set."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw:
        raise ValueError(f"{path} пуст")
    queries = []
    for i, item in enumerate(raw, start=1):
        missing = {"query", "expect", "kind"} - set(item)
        if missing:
            raise ValueError(f"{path}, запрос {i}: нет полей {sorted(missing)}")
        queries.append(Query(item["query"], tuple(item["expect"]), item["kind"]))
    return queries


def matches(path: str, expect: str) -> bool:
    """A trailing slash means "anywhere under here"; otherwise, this exact file."""
    return path.startswith(expect) if expect.endswith("/") else path == expect


def hit_rank(hits: Sequence[Hit], expect: Sequence[str]) -> int | None:
    """1-based position of the first correct hit, or None if there is none."""
    for position, hit in enumerate(hits, start=1):
        if any(matches(hit.chunk.path, e) for e in expect):
            return position
    return None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def recall_at(ranks: Iterable[int | None], k: int) -> float:
    """Share of queries with a correct hit in the top k."""
    values = [1.0 if r is not None and r <= k else 0.0 for r in ranks]
    return statistics.mean(values) if values else 0.0


def mrr_at(ranks: Iterable[int | None], k: int) -> float:
    """Mean reciprocal rank -- rewards being first, not merely present."""
    values = [1.0 / r if r is not None and r <= k else 0.0 for r in ranks]
    return statistics.mean(values) if values else 0.0


def gate_verdict(outcomes: Sequence[Outcome]) -> tuple[bool, float, float, int]:
    """Check the pre-registered gate on conceptual queries alone.

    Those are the queries grep cannot answer, so they are where a semantic
    index has to earn its cost. Winning on exact-identifier queries would prove
    nothing -- grep already wins those.
    """
    subset = [o for o in outcomes if o.query.kind == GATE_KIND]
    hybrid = [o.ranks["hybrid"] for o in subset]
    lexical = [o.ranks["lexical"] for o in subset]
    recall_gain = (recall_at(hybrid, 10) - recall_at(lexical, 10)) * 100
    mrr_gain = mrr_at(hybrid, 10) - mrr_at(lexical, 10)
    passed = recall_gain >= GATE_RECALL_GAIN or mrr_gain >= GATE_MRR_GAIN
    return passed, recall_gain, mrr_gain, len(subset)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def run(searcher: Searcher, queries: Sequence[Query], depth: int = DEPTH,
        verbose: bool = False) -> list[Outcome]:
    """Evaluate every query in every retrieval mode at a fixed depth."""
    outcomes = []
    for number, query in enumerate(queries, start=1):
        outcome = Outcome(query=query)
        for mode in MODES:
            started = time.perf_counter()
            results = searcher.search(query.query, k=depth, mode=mode, max_tokens=None)
            outcome.seconds[mode] = time.perf_counter() - started
            outcome.ranks[mode] = hit_rank(results.hits, query.expect)
            if mode == "hybrid":
                for k in (5, 10, 15):
                    outcome.tokens[k] = sum(
                        estimate_tokens(h.chunk.text) for h in results.hits[:k])
        outcomes.append(outcome)
        if verbose:
            print(f"  {number}/{len(queries)}  {query.query[:60]}", flush=True)
    return outcomes


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def _table(outcomes: Sequence[Outcome], title: str) -> None:
    print(f"\n{title}  (n={len(outcomes)})")
    head = "  ".join(f"R@{k:<2}" for k in CUTOFFS)
    print(f"  {'режим':10} {head}   MRR@10")
    for mode in ORDER:
        ranks = [o.ranks[mode] for o in outcomes]
        cells = "  ".join(f"{recall_at(ranks, k):.2f}" for k in CUTOFFS)
        print(f"  {LABELS[mode]:10} {cells}   {mrr_at(ranks, 10):.3f}")


def report(outcomes: Sequence[Outcome], source: Path) -> bool:
    """Print metrics and return whether the pre-registered gate passed."""
    print(f"\n{'=' * 72}\n{source.name}: {len(outcomes)} запросов\n{'=' * 72}")
    _table(outcomes, "ВСЕ ЗАПРОСЫ")

    gate_subset = [o for o in outcomes if o.query.kind == GATE_KIND]
    if gate_subset:
        _table(gate_subset, f"ТОЛЬКО {GATE_KIND.upper()} — ГЕЙТ")
        passed, recall_gain, mrr_gain, _ = gate_verdict(outcomes)
        print(f"\n  ΔRecall@10 = {recall_gain:+.1f} п.п.  (порог +{GATE_RECALL_GAIN:.0f})")
        print(f"  ΔMRR@10    = {mrr_gain:+.3f}     (порог +{GATE_MRR_GAIN:.2f})")
        print(f"  ГЕЙТ: {'ПРОЙДЕН' if passed else 'НЕ ПРОЙДЕН'}")
    else:
        passed = False
        print(f"\n  в наборе нет запросов вида {GATE_KIND!r} — гейт не проверялся")

    print("\n  Recall@10 по типам (BM25 / вектор / гибрид):")
    for kind in KINDS:
        subset = [o for o in outcomes if o.query.kind == kind]
        if subset:
            cells = " / ".join(
                f"{recall_at([o.ranks[m] for o in subset], 10):.2f}"
                for m in ORDER)
            print(f"    {kind:8} {cells}   (n={len(subset)})")

    latencies = sorted(o.seconds["hybrid"] for o in outcomes)
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
    print(f"\n  латентность гибрида: p50 {statistics.median(latencies) * 1000:.0f} мс, "
          f"p95 {p95 * 1000:.0f} мс")

    print("\n  токены выдачи (гибрид):")
    for k in (5, 10, 15):
        values = [o.tokens[k] for o in outcomes]
        print(f"    k={k:<3} медиана {statistics.median(values):>6,.0f}   "
              f"среднее {statistics.mean(values):>6,.0f}   max {max(values):>6,}")

    print("\n  ранги (BM25 / вектор / гибрид, '-' = не найдено в top-50):")
    for outcome in outcomes:
        cells = " ".join(
            f"{outcome.ranks[m] if outcome.ranks[m] else '-':>4}"
            for m in ORDER)
        print(f"    {outcome.query.kind:8} {cells}  {outcome.query.query[:44]}")
    return passed


def as_json(outcomes: Sequence[Outcome]) -> list[dict[str, object]]:
    """Measurements only -- deliberately without the query text or `expect` paths.

    This is the file that gets committed as the evidence behind the published
    numbers, and every one of them (Recall@k, MRR@k, the per-kind breakdown,
    latency, output tokens) is recomputable from `kind` and `ranks` alone. The
    query text and the expected locations add nothing to that and describe the
    indexed project's internals, so they stay out of the artefact rather than
    relying on someone remembering to strip them later. `id` is the position in
    the source set, which is all that is needed to line two runs up.
    """
    return [
        {"id": number, "kind": o.query.kind,
         "ranks": o.ranks, "seconds": o.seconds, "tokens": o.tokens}
        for number, o in enumerate(outcomes, start=1)
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation CLI and return a process exit code."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        prog="python eval/harness.py",
        description="Замерить качество поиска на размеченном наборе запросов.",
    )
    parser.add_argument("root", type=Path, help="корень проиндексированного проекта")
    # The example set, because it is the only one shipped. It is labelled against
    # this project's own code, so the default run measures something real rather
    # than failing on a missing file -- but the numbers in README came
    # from a larger set on a much larger corpus, and six queries cannot restate
    # them. Point --queries at your own set to measure your own project.
    parser.add_argument("--queries", type=Path, default=here / "queries.example.yaml")
    parser.add_argument("--depth", type=int, default=DEPTH)
    parser.add_argument("--save", type=Path, default=None, help="куда положить JSON")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--unblind", action="store_true",
        help="разрешить прогон по отложенному набору (одноразовое действие)")
    args = parser.parse_args(argv)

    if "holdout" in args.queries.name and not args.unblind:
        print("Это отложенный набор. По предрегистрации он считается один раз, "
              "после заморозки пайплайна.", file=sys.stderr)
        print("Если пайплайн заморожен — повторите с --unblind.", file=sys.stderr)
        return 2

    root: Path = args.root.resolve()
    db_path = index_path(root)
    if not db_path.exists():
        print(f"индекс не найден: {db_path}", file=sys.stderr)
        return 2

    queries = load_queries(args.queries)
    conn = store.open_index(db_path)
    try:
        stale = plan(conn, root)
        if stale.is_stale:
            print(f"ВНИМАНИЕ: индекс устарел ({len(stale.changed)} изменено, "
                  f"{len(stale.removed)} удалено) — числа не о текущем коде",
                  file=sys.stderr)
        files, chunks = store.stats(conn)
        print(f"корпус: {files} файлов, {chunks} чанков; "
              f"{len(queries)} запросов x {len(MODES)} режима", flush=True)
        outcomes = run(Searcher(conn), queries, args.depth, verbose=not args.quiet)
    finally:
        conn.close()

    passed = report(outcomes, args.queries)
    if args.save:
        args.save.write_text(
            json.dumps(as_json(outcomes), ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"\nсырые результаты: {args.save}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
