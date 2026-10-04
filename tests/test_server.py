"""MCP protocol tests.

The interesting failures here are the ones that do not raise. A server that
answers every request correctly and also writes one stray line to stdout is
broken in a way no unit test of the tool bodies would notice, so the transport
gets tested through a real stream, not just through `handle`.

The rest is contract: the client reads the tool descriptions and the result
shape and nothing else, so both are pinned.
"""

from __future__ import annotations

import io
import json
import re
import sys
from array import array
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_search import (
    search,  # noqa: E402
    store,  # noqa: E402
)
from code_search import server as srv  # noqa: E402
from code_search.chunker import estimate_tokens  # noqa: E402
from code_search.embedder import Embedder, EmbeddingError  # noqa: E402
from code_search.indexer import index_path, reindex  # noqa: E402

MODULE = '''\
"""Маршруты и рейсы."""


def compute_route(origin, target):
    """Считает маршрут между двумя точками."""
    return origin, target


def cancel_trip(trip_id):
    """Отменяет рейс."""
    return trip_id
'''


class FakeEmbedder(Embedder):
    def __init__(self) -> None:
        super().__init__("bge-m3")

    def embed_documents(self, texts: list[str]) -> list[array[float]]:
        return [array("f", [0.1] * self.dimension) for _ in texts]

    def embed_query(self, text: str) -> array[float]:
        return array("f", [0.1] * self.dimension)


class BrokenEmbedder(FakeEmbedder):
    def embed_query(self, text: str) -> array[float]:
        raise EmbeddingError("connection refused")


@pytest.fixture()
def project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "src").mkdir(parents=True)
    (root / "src" / "routes.py").write_text(MODULE, encoding="utf-8")
    (root / "README.md").write_text("# Проект\n\nОписание достаточной длины.\n",
                                    encoding="utf-8")
    reindex(root, embedder=FakeEmbedder())
    return root


@pytest.fixture()
def server(project: Path) -> srv.Server:
    instance = srv.Server(project, embedder=FakeEmbedder())
    yield instance
    instance.close()


def request(method: str, params: dict[str, Any] | None = None,
            message_id: int | str = 1) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "id": message_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def call_text(server: srv.Server, name: str,
              arguments: dict[str, Any] | None = None) -> str:
    """Invoke a tool through the protocol and return its text payload."""
    response = server.handle(request("tools/call",
                                     {"name": name, "arguments": arguments or {}}))
    assert response is not None
    result = response["result"]
    assert result["isError"] is False, result["content"][0]["text"]
    return str(result["content"][0]["text"])


def call(server: srv.Server, name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Invoke a JSON-payload tool (index_status) and decode it."""
    return json.loads(call_text(server, name, arguments))


def parse_results(text: str) -> tuple[list[str], list[str]]:
    """Split a search_code payload into full-text headers and reference lines.

    A full result is a header line followed by a ``----`` fence; a reference is
    an indented `path:a-b` line in the tail section. Fixture chunks contain no
    ``----`` lines of their own, so fence tracking here can stay naive.
    """
    full: list[str] = []
    refs: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i + 1] == "----":
            full.append(lines[i])
            i += 2
            while lines[i] != "----":
                i += 1
            i += 1
            continue
        if lines[i].startswith("  ") and ":" in lines[i]:
            refs.append(lines[i].strip())
        i += 1
    return full, refs


# --------------------------------------------------------------------------- #
# handshake
# --------------------------------------------------------------------------- #


def test_initialize_echoes_a_version_the_client_asked_for(server: srv.Server) -> None:
    response = server.handle(request("initialize", {"protocolVersion": "2024-11-05"}))
    assert response is not None
    result = response["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == srv.SERVER_NAME
    assert "tools" in result["capabilities"]


def test_an_unknown_protocol_version_falls_back_to_ours(server: srv.Server) -> None:
    """Answering with our newest lets the client decide; refusing would not."""
    response = server.handle(request("initialize", {"protocolVersion": "1999-01-01"}))
    assert response is not None
    assert response["result"]["protocolVersion"] == srv.PROTOCOL_VERSIONS[0]


def test_notifications_get_no_reply(server: srv.Server) -> None:
    """A reply to a notification is a protocol violation, not a harmless extra."""
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


@pytest.mark.parametrize("message", [
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}},
    {"jsonrpc": "2.0", "method": "bogus/notification"},
    {"jsonrpc": "2.0", "method": "$/setTrace", "params": {"value": "off"}},
])
def test_a_valid_notification_is_never_answered(
        server: srv.Server, message: dict[str, Any]) -> None:
    """A well-formed Request object without an id is a notification, and the
    spec forbids answering one. An *unknown* method is no exception -- that is
    the case where answering is most tempting and still wrong.
    """
    assert server.handle(message) is None


@pytest.mark.parametrize("message", [
    {},
    {"method": "ping"},                       # no jsonrpc at all
    {"jsonrpc": "1.0", "method": "ping"},
    {"jsonrpc": "2.0"},                       # no method at all
    {"jsonrpc": "2.0", "method": 17},
])
def test_a_malformed_object_without_id_is_an_error_not_a_notification(
        server: srv.Server, message: dict[str, Any]) -> None:
    """A notification is a *Request object* without an id; an object that fails
    the Request grammar is not a Request object, hence not a notification. The
    spec's own batch examples answer `{"foo": "boo"}` with an Invalid Request
    carrying `id: null` -- these used to be swallowed in silence.
    """
    response = server.handle(message)
    assert response is not None
    assert response["id"] is None
    assert response["error"]["code"] == srv.INVALID_REQUEST


def test_a_notifications_method_with_an_id_is_a_request_and_gets_an_answer(
        server: srv.Server) -> None:
    """An id turns the message into a request, and no request method is named
    `notifications/*` -- so the honest answer is METHOD_NOT_FOUND. Silence used
    to leave a client that awaits the reply hanging forever.
    """
    response = server.handle(
        {"jsonrpc": "2.0", "id": 9, "method": "notifications/initialized"})
    assert response is not None
    assert response["id"] == 9
    assert response["error"]["code"] == srv.METHOD_NOT_FOUND


def test_ping_is_answered(server: srv.Server) -> None:
    response = server.handle(request("ping"))
    assert response is not None and response["result"] == {}


def test_an_unknown_method_is_an_error_not_a_crash(server: srv.Server) -> None:
    response = server.handle(request("resources/list"))
    assert response is not None
    assert response["error"]["code"] == srv.METHOD_NOT_FOUND


def test_a_message_without_a_method_is_rejected(server: srv.Server) -> None:
    response = server.handle({"jsonrpc": "2.0", "id": 7})
    assert response is not None
    assert response["error"]["code"] == srv.INVALID_REQUEST


@pytest.mark.parametrize("bad_id", [None, True, 1.5, [1], {"n": 1}])
def test_an_unusable_id_is_an_error_with_id_null_not_silence(
        server: srv.Server, bad_id: Any) -> None:
    """An `id` member that is present but not a string or integer is an invalid
    request, and the spec says to answer it with `id: null`. An explicit
    `"id": null` in particular used to be mistaken for a notification and
    swallowed -- the client then waited on a reply that would never come.
    """
    response = server.handle({"jsonrpc": "2.0", "id": bad_id, "method": "ping"})
    assert response is not None
    assert response["id"] is None
    assert response["error"]["code"] == srv.INVALID_REQUEST


def test_a_string_id_is_as_good_as_an_integer(server: srv.Server) -> None:
    response = server.handle(request("ping", None, "req-1"))
    assert response is not None and response["id"] == "req-1"


@pytest.mark.parametrize("envelope", [
    {"id": 3, "method": "ping"},                          # no jsonrpc at all
    {"jsonrpc": "1.0", "id": 3, "method": "ping"},
    {"jsonrpc": 2.0, "id": 3, "method": "ping"},          # the number, not the string
])
def test_the_envelope_must_say_jsonrpc_2_0(server: srv.Server,
                                            envelope: dict[str, Any]) -> None:
    response = server.handle(envelope)
    assert response is not None
    assert response["id"] == 3
    assert response["error"]["code"] == srv.INVALID_REQUEST


# --------------------------------------------------------------------------- #
# the tool contract
# --------------------------------------------------------------------------- #


def test_exactly_two_tools_are_offered(server: srv.Server) -> None:
    response = server.handle(request("tools/list"))
    assert response is not None
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert names == {"search_code", "index_status"}


def test_the_description_sends_exact_identifiers_to_lexical_search() -> None:
    """Without this line the model uses semantic search for everything."""
    assert "grep" in srv.SEARCH_DESCRIPTION.lower()
    # The tail comes back without text; the description must say how to get it.
    assert "диапазону строк" in srv.SEARCH_DESCRIPTION


def test_model_facing_text_names_no_client_specific_tools() -> None:
    """The server is attached to Claude Code, Cursor, Codex -- anything MCP.

    `Grep` and `Read` are Claude Code tool names; in any other client they are
    dangling references. Everything the model reads must describe capabilities,
    not one client's toolbox.
    """
    chunk = store.StoredChunk(id=1, path="src/a.py", kind="source", start_line=1,
                              end_line=2, symbol="f", text="def f():\n    pass",
                              file_hash="a" * 64)
    hit = search.Hit(chunk=chunk, score=1.0)
    rendered = srv._format_results(
        search.SearchResults(hits=[hit], mode="hybrid", refs=(hit,)), stale=True)
    for text in [rendered, *(tool["description"] for tool in srv.TOOLS)]:
        assert "Grep" not in text
        assert "Read" not in text


def test_every_tool_declares_an_object_schema() -> None:
    for tool in srv.TOOLS:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]


def test_only_query_is_required() -> None:
    schema = next(t for t in srv.TOOLS if t["name"] == "search_code")["inputSchema"]
    assert schema["required"] == ["query"]


# --------------------------------------------------------------------------- #
# search_code
# --------------------------------------------------------------------------- #


def test_search_returns_the_documented_format(server: srv.Server) -> None:
    text = call_text(server, "search_code", {"query": "маршрут"})
    full, _refs = parse_results(text)
    assert full
    # Header: path:start-end plus the symbol when there is one.
    assert re.match(r"^\S+:\d+-\d+( \S+)?$", full[0])
    # One short hash per unique file in the footer, not 64 hex per hit.
    hash_line = next(line for line in text.splitlines() if line.startswith("hash: "))
    assert re.search(r"\S+=[0-9a-f]{12}(?: |$)", hash_line)
    # A fresh index says nothing about staleness -- absence is the signal.
    assert "stale" not in text


def test_search_honours_k(server: srv.Server) -> None:
    text = call_text(server, "search_code", {"query": "маршрут", "k": 1})
    full, refs = parse_results(text)
    assert len(full) + len(refs) == 1


def test_k_is_capped_rather_than_refused(server: srv.Server) -> None:
    """An absurd k is a caller mistake, not a reason to fail the call."""
    text = call_text(server, "search_code", {"query": "маршрут", "k": 10_000})
    full, refs = parse_results(text)
    assert 0 < len(full) + len(refs) <= srv.MAX_K


def test_search_honours_path_glob(server: srv.Server) -> None:
    text = call_text(server, "search_code", {"query": "маршрут", "path_glob": "src/*"})
    full, refs = parse_results(text)
    assert full
    assert all(header.startswith("src/") for header in full + refs)


def test_a_glob_matching_nothing_says_so_rather_than_erroring(
        server: srv.Server) -> None:
    text = call_text(server, "search_code", {"query": "маршрут", "path_glob": "nope/*"})
    assert "ничего не найдено" in text


def test_an_edit_is_reported_as_stale(project: Path) -> None:
    instance = srv.Server(project, embedder=FakeEmbedder())
    try:
        (project / "src" / "extra.py").write_text(MODULE, encoding="utf-8")
        assert "stale: true" in call_text(instance, "search_code", {"query": "рейс"})
    finally:
        instance.close()


def test_results_carry_cyrillic_and_raw_newlines(server: srv.Server) -> None:
    """The payload is plain text now: no JSON escaping inside it."""
    text = call_text(server, "search_code", {"query": "рейс"})
    assert "Отменяет" in text or "маршрут" in text.lower()
    assert "\\n" not in text


def test_the_text_format_is_cheaper_than_the_old_json(server: srv.Server) -> None:
    """The point of the format change, pinned so a refactor cannot undo it."""
    text = call_text(server, "search_code", {"query": "маршрут"})
    searcher, _ = server._open()
    results = searcher.search("маршрут")
    legacy = json.dumps({
        "results": [
            {"path": h.chunk.path, "start_line": h.chunk.start_line,
             "end_line": h.chunk.end_line, "symbol": h.chunk.symbol,
             "kind": h.chunk.kind, "score": round(h.score, 6),
             "file_hash": h.chunk.file_hash, "text": h.chunk.text}
            for h in results.hits
        ],
        "stale": False, "dropped_by_budget": 0,
    }, ensure_ascii=False, indent=1)
    assert estimate_tokens(text) < estimate_tokens(legacy)


# --------------------------------------------------------------------------- #
# structuredContent
# --------------------------------------------------------------------------- #


def handshake(server: srv.Server, version: str) -> None:
    response = server.handle(request("initialize", {"protocolVersion": version}, 0))
    assert response is not None and response["result"]["protocolVersion"] == version


def tool_result(server: srv.Server, name: str,
                arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    response = server.handle(request("tools/call",
                                     {"name": name, "arguments": arguments or {}}))
    assert response is not None
    result: dict[str, Any] = response["result"]
    assert result["isError"] is False, result["content"][0]["text"]
    return result


def test_a_2025_06_18_client_gets_structured_content_beside_the_text(
        server: srv.Server) -> None:
    """The text is for the model; the structure is for a client that parses. Both
    describe the same answer, and the text does not change because the structure
    is there -- a test elsewhere pins its size, this one pins its presence.
    """
    handshake(server, "2025-06-18")
    result = tool_result(server, "search_code", {"query": "маршрут"})
    text = result["content"][0]["text"]
    data = result["structuredContent"]

    full, refs = parse_results(text)
    assert [f"{r['path']}:{r['start_line']}-{r['end_line']}" for r in data["results"]] \
        == [header.split(" ")[0] for header in full]
    assert len(data["references"]) == len(refs)
    assert all("text" in r for r in data["results"])
    assert all("text" not in r for r in data["references"])
    assert data["stale"] is False
    assert set(data) == set(srv.SEARCH_OUTPUT_SCHEMA["required"])
    for path, digest in data["file_hashes"].items():
        assert f"{path}={digest}" in text


def test_structured_content_and_text_agree_for_index_status(server: srv.Server) -> None:
    handshake(server, "2025-06-18")
    result = tool_result(server, "index_status")
    assert result["structuredContent"] == json.loads(result["content"][0]["text"])
    assert set(result["structuredContent"]) == set(srv.STATUS_OUTPUT_SCHEMA["required"])


def test_structured_content_matches_the_declared_schema(server: srv.Server) -> None:
    """No jsonschema dependency here, so the check is by hand: every required key
    is present with the declared type. Declaring `outputSchema` and then sending
    something else is worse than declaring nothing.
    """
    handshake(server, "2025-06-18")
    data = tool_result(server, "search_code", {"query": "маршрут", "k": 1})["structuredContent"]
    types: dict[str, type | tuple[type, ...]] = {
        "string": str, "integer": int, "boolean": bool, "array": list, "object": dict,
        "null": type(None),
    }

    def conforms(value: Any, schema: dict[str, Any]) -> bool:
        declared = schema["type"]
        names = declared if isinstance(declared, list) else [declared]
        if not any(isinstance(value, types[n]) and not (n == "integer" and isinstance(value, bool))
                   for n in names):
            return False
        if isinstance(value, dict) and "properties" in schema:
            return all(key in value and conforms(value[key], schema["properties"][key])
                       for key in schema.get("required", []))
        if isinstance(value, list) and "items" in schema:
            return all(conforms(item, schema["items"]) for item in value)
        return True

    assert conforms(data, srv.SEARCH_OUTPUT_SCHEMA)
    assert data["results"] or data["references"]


@pytest.mark.parametrize("version", ["2025-03-26", "2024-11-05"])
def test_older_clients_see_neither_structured_content_nor_output_schema(
        server: srv.Server, version: str) -> None:
    """Both fields arrived in the same spec revision. A client on an earlier one
    has no promise of ignoring unknown keys, so it is sent exactly its version.
    """
    handshake(server, version)
    result = tool_result(server, "search_code", {"query": "маршрут"})
    assert "structuredContent" not in result
    listing = server.handle(request("tools/list"))
    assert listing is not None
    assert all("outputSchema" not in tool for tool in listing["result"]["tools"])


def test_a_current_client_sees_output_schema_on_every_tool(server: srv.Server) -> None:
    handshake(server, "2025-06-18")
    listing = server.handle(request("tools/list"))
    assert listing is not None
    assert all(tool["outputSchema"]["type"] == "object"
               for tool in listing["result"]["tools"])


def test_without_a_handshake_the_server_assumes_the_oldest_client(
        server: srv.Server) -> None:
    result = tool_result(server, "search_code", {"query": "маршрут"})
    assert "structuredContent" not in result


# --------------------------------------------------------------------------- #
# index_status
# --------------------------------------------------------------------------- #


def test_status_reports_the_index(server: srv.Server) -> None:
    payload = call(server, "index_status")
    assert payload["files"] == 2
    assert payload["chunks"] > 0
    assert payload["model"] == "bge-m3"
    assert payload["stale_files"] == 0
    assert payload["last_indexed"] is not None
    assert payload["chunker_version"] != "?"


def test_status_takes_no_arguments(server: srv.Server) -> None:
    assert call(server, "index_status", {"ignored": 1})["files"] == 2


# --------------------------------------------------------------------------- #
# failure reaches the model, not the transport
# --------------------------------------------------------------------------- #


def failure(server: srv.Server, name: str, arguments: dict[str, Any]) -> str:
    response = server.handle(request("tools/call",
                                     {"name": name, "arguments": arguments}))
    assert response is not None
    result = response["result"]
    assert result["isError"] is True
    return str(result["content"][0]["text"])


def test_a_missing_query_is_a_tool_error(server: srv.Server) -> None:
    assert "query" in failure(server, "search_code", {})


def test_an_empty_query_is_a_tool_error(server: srv.Server) -> None:
    assert "query" in failure(server, "search_code", {"query": "   "})


def test_a_non_integer_k_is_a_tool_error(server: srv.Server) -> None:
    assert "k" in failure(server, "search_code", {"query": "рейс", "k": "много"})


def test_an_unknown_tool_is_a_tool_error_not_a_protocol_error(
        server: srv.Server) -> None:
    """The model can recover from a tool error; a dead connection it cannot."""
    assert "grep_code" in failure(server, "grep_code", {})


def test_ollama_being_down_says_so_and_says_what_to_do(project: Path) -> None:
    instance = srv.Server(project, embedder=BrokenEmbedder())
    try:
        message = failure(instance, "search_code", {"query": "рейс"})
        assert "connection refused" in message
        assert "ollama serve" in message
    finally:
        instance.close()


def test_a_missing_index_explains_how_to_build_one(tmp_path: Path) -> None:
    """Indexing takes tens of minutes; doing it inside a tool call would hang."""
    empty = tmp_path / "unindexed"
    empty.mkdir()
    message = failure(srv.Server(empty), "search_code", {"query": "рейс"})
    assert "Индекс не найден" in message
    assert "code_search.indexer" in message


@pytest.mark.parametrize("key", ["schema_version", "chunker_version",
                                 "embedding_model", "embedding_dimension"])
def test_an_index_from_another_version_is_refused_not_served(project: Path,
                                                             key: str) -> None:
    """Serving it anyway is the silent failure, and it is worse than it looks.

    The background refresh does notice, and reacts by deleting the index
    directory and rebuilding -- while this process holds the old file open. On
    Linux the unlinked inode keeps answering, so the server would go on serving
    an index nobody can see or update, for as long as it lives, saying nothing.
    """
    conn = store.open_index(index_path(project))
    store.write_meta(conn, {key: "какая-то другая версия"})
    conn.close()

    instance = srv.Server(project, embedder=FakeEmbedder())
    try:
        message = failure(instance, "index_status", {})
        assert "пересобран" in message
        assert key in message
        assert "--rebuild" in message
    finally:
        instance.close()


# --------------------------------------------------------------------------- #
# the stdio transport
# --------------------------------------------------------------------------- #


def drive(server: srv.Server, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stream_in = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
    stream_out = io.StringIO()
    server.serve(stream_in, stream_out)
    return [json.loads(line) for line in stream_out.getvalue().splitlines() if line]


def test_a_whole_session_over_the_stream(server: srv.Server) -> None:
    responses = drive(server, [
        request("initialize", {"protocolVersion": "2025-06-18"}, 1),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        request("tools/list", None, 2),
        request("tools/call", {"name": "search_code", "arguments": {"query": "рейс"}}, 3),
    ])
    assert [r["id"] for r in responses] == [1, 2, 3], "у нотификации ответа быть не должно"
    assert all(r["jsonrpc"] == "2.0" for r in responses)


def test_every_line_on_stdout_is_a_json_object(server: srv.Server) -> None:
    """The one failure mode that breaks the client while every tool works."""
    stream_in = io.StringIO(json.dumps(request("tools/call", {
        "name": "search_code", "arguments": {"query": "маршрут"}})) + "\n")
    stream_out = io.StringIO()
    server.serve(stream_in, stream_out)

    lines = stream_out.getvalue().splitlines()
    assert lines
    for line in lines:
        assert isinstance(json.loads(line), dict)


def test_a_result_never_spans_two_lines(server: srv.Server) -> None:
    """Chunk text contains newlines; JSON must escape them, not emit them."""
    responses = drive(server, [request("tools/call", {
        "name": "search_code", "arguments": {"query": "маршрут"}})])
    assert len(responses) == 1


def test_malformed_json_is_answered_and_the_session_continues(
        server: srv.Server) -> None:
    stream_in = io.StringIO("{ не json\n" + json.dumps(request("ping", None, 5)) + "\n")
    stream_out = io.StringIO()
    server.serve(stream_in, stream_out)

    responses = [json.loads(line) for line in stream_out.getvalue().splitlines()]
    assert responses[0]["error"]["code"] == srv.PARSE_ERROR
    assert responses[1]["id"] == 5


def test_a_batch_is_answered_as_one_array(server: srv.Server) -> None:
    """JSON-RPC 2.0 section 6: one array in, one array out, one line each way."""
    batch = [request("ping", None, 1), request("tools/list", None, 2)]
    stream_in = io.StringIO(json.dumps(batch) + "\n")
    stream_out = io.StringIO()
    server.serve(stream_in, stream_out)

    lines = stream_out.getvalue().splitlines()
    assert len(lines) == 1
    responses = json.loads(lines[0])
    assert isinstance(responses, list)
    assert [r["id"] for r in responses] == [1, 2]
    assert "tools" in responses[1]["result"]


def test_a_batch_of_notifications_is_answered_with_silence(server: srv.Server) -> None:
    """The rule for one notification holds for a batch made only of them."""
    batch = [{"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "method": "notifications/cancelled"}]
    stream_out = io.StringIO()
    server.serve(io.StringIO(json.dumps(batch) + "\n"), stream_out)
    assert stream_out.getvalue() == ""


def test_a_batch_mixes_answers_and_silence_per_element(server: srv.Server) -> None:
    batch = [request("ping", None, 1),
             {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2}]                      # no method: an error
    responses = server.dispatch(batch)
    assert isinstance(responses, list)
    assert [r["id"] for r in responses] == [1, 2]
    assert "result" in responses[0] and responses[1]["error"]["code"] == srv.INVALID_REQUEST


def test_an_empty_batch_is_an_invalid_request(server: srv.Server) -> None:
    response = server.dispatch([])
    assert isinstance(response, dict)
    assert response["id"] is None
    assert response["error"]["code"] == srv.INVALID_REQUEST


def test_a_batch_of_junk_is_an_error_per_element_and_the_session_continues(
        server: srv.Server) -> None:
    stream_in = io.StringIO("[1, 2, 3]\n" + json.dumps(request("ping", None, 6)) + "\n")
    stream_out = io.StringIO()
    server.serve(stream_in, stream_out)

    lines = stream_out.getvalue().splitlines()
    errors = json.loads(lines[0])
    assert [e["error"]["code"] for e in errors] == [srv.INVALID_REQUEST] * 3
    assert all(e["id"] is None for e in errors)
    assert json.loads(lines[1])["id"] == 6


def test_a_scalar_is_rejected_without_killing_the_session(server: srv.Server) -> None:
    stream_in = io.StringIO('"ping"\n' + json.dumps(request("ping", None, 6)) + "\n")
    stream_out = io.StringIO()
    server.serve(stream_in, stream_out)

    responses = [json.loads(line) for line in stream_out.getvalue().splitlines()]
    assert responses[0]["error"]["code"] == srv.INVALID_REQUEST
    assert responses[1]["id"] == 6


def test_blank_lines_are_ignored(server: srv.Server) -> None:
    stream_in = io.StringIO("\n\n" + json.dumps(request("ping", None, 8)) + "\n\n")
    stream_out = io.StringIO()
    server.serve(stream_in, stream_out)
    assert len(stream_out.getvalue().splitlines()) == 1


def test_closed_stdin_ends_the_session_cleanly(server: srv.Server) -> None:
    """The client disconnecting is how this server normally shuts down."""
    assert server.serve(io.StringIO(""), io.StringIO()) == 0


def test_a_crlf_client_is_understood(server: srv.Server) -> None:
    """Windows clients terminate lines with \\r\\n; the \\r must not reach json."""
    stream_in = io.StringIO(json.dumps(request("ping", None, 9)) + "\r\n")
    stream_out = io.StringIO()
    server.serve(stream_in, stream_out)
    assert json.loads(stream_out.getvalue())["id"] == 9


def test_output_is_encodable_and_carries_no_bare_cr(server: srv.Server) -> None:
    """Cyrillic goes out unescaped, so the stream has to be UTF-8 and \\n-framed."""
    text = drive(server, [request("tools/list")])
    raw = json.dumps(text, ensure_ascii=False)
    assert "лексический" in raw and "\r" not in raw
    raw.encode("utf-8")


def test_reconfigure_tolerates_a_stream_that_cannot_do_it() -> None:
    srv._reconfigure(io.StringIO(), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_main_refuses_a_root_that_is_not_a_directory(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere"
    assert srv.main(["--root", str(missing)]) == 2
