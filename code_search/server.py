"""MCP server over stdio: two tools, no more.

The protocol is implemented directly rather than through the official SDK, which
is absent from both interpreters here and uninstallable (PEP 668 marks the
system environment externally managed and sudo is unavailable). What is actually
needed is small: newline-delimited JSON-RPC 2.0, four methods, one tool result
shape. The dependency would cost more than the code does.

Small is not loose. The envelope is checked the way the spec reads: `jsonrpc`
must be "2.0", an id must be a string or an integer, an explicit `"id": null` is
an invalid request and not a notification, and a JSON array is a batch that is
answered as one. Anything the server cannot make sense of gets an error back,
never silence -- silence is reserved for notifications, where it is mandatory.

The rule that governs everything in this file: **stdout carries protocol traffic
and nothing else.** A stray print, a warning, a traceback -- any of them corrupt
the stream and the client sees the server as broken rather than as failing. All
diagnostics go to stderr, and every tool body is wrapped so that an exception
becomes a tool error the model can read instead of a dead connection.

Two tools, deliberately. Given a wide toolbox the model starts guessing between
near-synonyms; given two with sharply different descriptions it picks correctly.
The `search_code` description says outright that lexical search is better for
exact identifiers, because without that it gets used for everything.

The server is client-agnostic and every model-facing string has to stay that
way: it is attached to Claude Code, Cursor, Codex and anything else that speaks
MCP, so tool descriptions and result text name *capabilities* ("лексический
поиск", "прочитай диапазон строк") rather than any one client's tool names
("Grep", "Read") -- those would read as dangling references everywhere else.

Nothing is opened at startup. A missing index or an unreachable Ollama must be
reportable *through* the protocol, which means the server has to be alive to
report it -- so the database and the embedder are created on first use.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from code_search import store
from code_search.chunker import CHUNKER_VERSION
from code_search.embedder import DEFAULT_MODEL, Embedder, EmbeddingError
from code_search.indexer import index_path
from code_search.refresh import Refresher
from code_search.search import DEFAULT_K, Hit, Searcher, SearchResults
from code_search.store import IncompatibleIndex

__all__ = ["Server", "ToolError", "ToolResult", "main"]

SERVER_NAME = "code-search"
SERVER_VERSION = "1.1.0"

# Newest first. The client names the version it wants during initialize; we echo
# it back when we know it, and otherwise answer with our newest and let the
# client decide, which is what the spec asks for.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

# The revision that added `structuredContent` on tool results and `outputSchema`
# on tool listings. Both are sent only to a client that negotiated at least this
# version: an older client would see unknown keys, and the spec for its version
# does not promise it will ignore them. Versions are ISO dates, so the string
# comparison in `_structured_supported` is the chronological one.
STRUCTURED_SINCE = "2025-06-18"

# A ceiling on k. The output budget already trims by tokens, but a caller asking
# for 500 results would still make the server score and load all of them first.
MAX_K = 50

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

SEARCH_DESCRIPTION = (
    "Семантический поиск по проекту. Используй, когда не знаешь, как искомое "
    "названо в коде. Для точных идентификаторов используй лексический поиск "
    "(grep) — он точнее. Результаты отсортированы по релевантности: первые "
    "идут с полным текстом, хвост — ссылками path:start-end без текста; их "
    "текст читай из файла по указанному диапазону строк. Строка stale: true "
    "означает, что индекс немного отстал от рабочей копии и обновляется в "
    "фоне — это рабочее состояние, а не ошибка."
)

# Where a result lives. Shared by full-text results and reference-only ones; the
# former add `text`.
_LOCATION_PROPERTIES: dict[str, Any] = {
    "path": {"type": "string", "description": "Путь относительно корня проекта."},
    "start_line": {"type": "integer", "minimum": 1},
    "end_line": {"type": "integer", "minimum": 1},
    "symbol": {"type": ["string", "null"],
               "description": "Имя функции/класса/раздела, если чанкер его выделил."},
}
_LOCATION_REQUIRED = ["path", "start_line", "end_line", "symbol"]

SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "description": "Хиты с полным текстом, по убыванию релевантности.",
            "items": {
                "type": "object",
                "properties": {**_LOCATION_PROPERTIES, "text": {"type": "string"}},
                "required": [*_LOCATION_REQUIRED, "text"],
            },
        },
        "references": {
            "type": "array",
            "description": "Хиты за пределами бюджета: ранжированы, но без текста.",
            "items": {
                "type": "object",
                "properties": _LOCATION_PROPERTIES,
                "required": _LOCATION_REQUIRED,
            },
        },
        "stale": {"type": "boolean"},
        "file_hashes": {
            "type": "object",
            "description": "Первые 12 hex SHA-256 каждого упомянутого файла.",
            "additionalProperties": {"type": "string"},
        },
    },
    "required": ["results", "references", "stale", "file_hashes"],
}

STATUS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {"type": "integer"},
        "chunks": {"type": "integer"},
        "model": {"type": "string"},
        "chunker_version": {"type": "string"},
        "stale_files": {"type": "integer"},
        "last_indexed": {"type": ["string", "null"]},
        "refreshing": {"type": "boolean"},
        "last_error": {"type": ["string", "null"]},
    },
    "required": ["files", "chunks", "model", "chunker_version", "stale_files",
                 "last_indexed", "refreshing", "last_error"],
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_code",
        "description": SEARCH_DESCRIPTION,
        "outputSchema": SEARCH_OUTPUT_SCHEMA,
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Что ищем, обычным языком.",
                },
                "k": {
                    "type": "integer",
                    "description": f"Сколько фрагментов вернуть (по умолчанию {DEFAULT_K}).",
                    "minimum": 1,
                    "maximum": MAX_K,
                },
                "path_glob": {
                    "type": "string",
                    "description": "Ограничить поиск путями по шаблону, например src/pkg/*.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "index_status",
        "description": (
            "Состояние индекса поиска: сколько файлов и фрагментов, какая модель, "
            "сколько файлов отстало, когда обновлялся последний раз. Про состояние "
            "Ollama ничего не сообщает — это разные вещи."
        ),
        "outputSchema": STATUS_OUTPUT_SCHEMA,
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class ToolError(RuntimeError):
    """A failure the model should read and act on, not a protocol-level fault."""


@dataclass(frozen=True)
class ToolResult:
    """What a tool body produces.

    `text` is for the model and is the whole answer on its own. `structured` is
    the same information as data, for a client that wants to parse rather than
    read; it goes out as `structuredContent` when the negotiated protocol has
    the field, and is dropped otherwise. The two must agree -- a test pins it.
    """

    text: str
    structured: dict[str, Any]


class Server:
    """Protocol handling and the two tool bodies.

    Deliberately transport-agnostic: `handle` takes a decoded message and
    returns a response or None, so the whole protocol is testable without pipes.
    """

    def __init__(self, root: Path, model: str = DEFAULT_MODEL,
                 excludes: tuple[str, ...] = (),
                 embedder: Embedder | None = None) -> None:
        self.root = root.resolve()
        self.model = model
        self.excludes = excludes
        self._embedder = embedder          # injectable, as in indexer and refresher
        self._searcher: Searcher | None = None
        self._refresher: Refresher | None = None
        # What `initialize` settled on. None until the handshake happens; a client
        # that skips it (the test suite, a probe) is treated as the oldest one,
        # because that is the only version everything it might send is valid in.
        self.protocol_version: str | None = None

    # ------------------------------------------------------------ lazy setup #

    def _dimension(self) -> int:
        """Vector width this server expects. No network call: it comes from the
        model's `ModelSpec`, so an unreachable Ollama does not prevent the
        compatibility check below from running.
        """
        if self._embedder is not None:
            return self._embedder.dimension
        return Embedder(self.model).dimension

    def _open(self) -> tuple[Searcher, Refresher]:
        if self._searcher is None or self._refresher is None:
            db_path = index_path(self.root)
            if not db_path.exists():
                raise ToolError(
                    f"Индекс не найден: {db_path}. Соберите его командой "
                    f"`python -m code_search.indexer {self.root}` — на проекте "
                    f"среднего размера это десятки минут, поэтому автоматически "
                    f"он здесь не строится."
                )
            try:
                conn = store.open_index(db_path)
            except IncompatibleIndex as exc:
                raise ToolError(f"Индекс непригоден: {exc}") from exc

            # Checked here and not left to the background refresh, which is what
            # would otherwise notice. `reindex` reacts to an incompatible index
            # by deleting the directory and rebuilding -- under a connection this
            # process still holds open. On Linux the unlinked file keeps serving,
            # so the server would answer from an index nobody can see or update,
            # indefinitely, without a word. Refusing is the honest move: it costs
            # one rebuild the user is told about.
            problems = store.compatibility_problems(conn, self.model, self._dimension())
            if problems:
                conn.close()
                raise ToolError(
                    "Индекс собран другой версией и должен быть пересобран:\n  - "
                    + "\n  - ".join(problems)
                    + f"\nКоманда: `python -m code_search.indexer {self.root} --rebuild`"
                )

            self._searcher = Searcher(conn, self._embedder, self.model)
            self._refresher = Refresher(self.root, conn, searcher=self._searcher,
                                        model=self.model, excludes=self.excludes,
                                        embedder=self._embedder)
        return self._searcher, self._refresher

    def close(self) -> None:
        """Close the lazily opened index connection and clear live helpers."""
        if self._searcher is not None:
            self._searcher.conn.close()
            self._searcher = None
            self._refresher = None

    # ------------------------------------------------------------ tool bodies #

    def search_code(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate and execute the ``search_code`` MCP tool."""
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("Параметр query обязателен и должен быть непустой строкой.")

        k = arguments.get("k", DEFAULT_K)
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ToolError(f"Параметр k должен быть целым числом ≥ 1, получено {k!r}.")
        k = min(k, MAX_K)

        path_glob = arguments.get("path_glob")
        if path_glob is not None and not isinstance(path_glob, str):
            raise ToolError("Параметр path_glob должен быть строкой.")

        searcher, refresher = self._open()
        # Asked before searching, so that the answer describes the index the
        # results actually came from, and so a refresh starts as early as it can.
        stale = refresher.ensure_fresh()
        try:
            results = searcher.search(query, k=k, path_glob=path_glob)
        except EmbeddingError as exc:
            raise ToolError(
                f"Не удалось получить эмбеддинг запроса: {exc}. "
                f"Проверьте, что Ollama запущена (`ollama serve`)."
            ) from exc

        return ToolResult(_format_results(results, stale), _structured_results(results, stale))

    def index_status(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the argument-free ``index_status`` MCP tool."""
        _, refresher = self._open()
        status = refresher.status()
        payload: dict[str, Any] = {
            "files": status.files,
            "chunks": status.chunks,
            "model": status.model,
            "chunker_version": status.chunker_version,
            "stale_files": status.stale_files,
            "last_indexed": status.last_indexed,
            "refreshing": status.refreshing,
            "last_error": status.last_error,
        }
        return ToolResult(json.dumps(payload, ensure_ascii=False), payload)

    # -------------------------------------------------------------- protocol #

    def dispatch(self, decoded: Any) -> dict[str, Any] | list[dict[str, Any]] | None:
        """One decoded line of input to what goes back; None means "stay quiet".

        A JSON array is a batch (JSON-RPC 2.0, section 6): each element is
        handled on its own, the answers go back as one array, and a batch made
        only of notifications gets nothing, the same rule as for a single one.
        MCP dropped batching in 2025-06-18, so a current client never sends one;
        an older one may, and accepting it costs less than telling them apart.
        """
        if isinstance(decoded, list):
            if not decoded:
                return _error(None, INVALID_REQUEST, "пустой batch")
            answered = [response for response in map(self._dispatch_one, decoded)
                        if response is not None]
            return answered or None
        return self._dispatch_one(decoded)

    def _dispatch_one(self, decoded: Any) -> dict[str, Any] | None:
        if not isinstance(decoded, dict):
            return _error(None, INVALID_REQUEST, "ожидался объект")
        return self.handle(decoded)

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one decoded message. None means "a notification, stay quiet"."""
        # A notification is a *Request object* without an `id` member, and only a
        # notification may go unanswered. An object that fails the Request
        # grammar (wrong envelope, non-string method) is not a Request object,
        # so it is not a notification either: the spec's own batch examples
        # answer `{"foo": "boo"}` with an Invalid Request carrying `id: null`.
        # The envelope is therefore checked before deciding to stay quiet.
        has_id = "id" in message
        message_id = message.get("id")
        if has_id and not _valid_id(message_id):
            # Present but unusable: JSON-RPC allows a string or a number and MCP
            # narrows that to string or integer and forbids null. An id we cannot
            # trust is answered with `id: null`, as the spec says for exactly
            # this situation -- an explicit `"id": null` used to be mistaken for
            # a notification and swallowed without a word.
            return _error(None, INVALID_REQUEST, f"недопустимый id: {message_id!r}")
        error_id = message_id if has_id else None
        if message.get("jsonrpc") != "2.0":
            return _error(error_id, INVALID_REQUEST, 'ожидается "jsonrpc": "2.0"')
        method = message.get("method")
        if not isinstance(method, str):
            return _error(error_id, INVALID_REQUEST, "нет поля method")
        if not has_id:
            return None

        handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "initialize": self._initialize,
            "tools/list": lambda _params: {"tools": self._tools_listing()},
            "tools/call": self._tools_call,
            "ping": lambda _params: {},
        }
        # A `notifications/*` method sent *with* an id is a request by the
        # letter of the spec, and no request method has that name -- it falls
        # through to METHOD_NOT_FOUND below. Staying quiet here used to leave a
        # client that awaits the reply hanging forever; a client that attached
        # the id by mistake just ignores the unexpected answer.
        handler = handlers.get(method)
        if handler is None:
            return _error(message_id, METHOD_NOT_FOUND, f"неизвестный метод: {method}")

        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _error(message_id, INVALID_PARAMS, "params должен быть объектом")
        try:
            return {"jsonrpc": "2.0", "id": message_id, "result": handler(params)}
        except Exception as exc:              # noqa: BLE001
            # Nothing may escape to stdout as a traceback; a protocol-level error
            # is the only way the client learns anything went wrong here.
            return _error(message_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        wanted = params.get("protocolVersion")
        version = wanted if wanted in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        self.protocol_version = version
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _structured_supported(self) -> bool:
        return self.protocol_version is not None and self.protocol_version >= STRUCTURED_SINCE

    def _tools_listing(self) -> list[dict[str, Any]]:
        """TOOLS as the negotiated version knows them: `outputSchema` was added
        in the same revision as `structuredContent` and is withheld together
        with it -- advertising a schema and then never sending data to match
        would be a broken promise.
        """
        if self._structured_supported():
            return TOOLS
        return [{key: value for key, value in tool.items() if key != "outputSchema"}
                for tool in TOOLS]

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        bodies: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "search_code": self.search_code,
            "index_status": self.index_status,
        }
        body = bodies.get(name) if isinstance(name, str) else None
        if body is None:
            return _tool_failure(f"нет такого инструмента: {name!r}")
        try:
            payload = body(arguments)
        except ToolError as exc:
            return _tool_failure(str(exc))
        except Exception as exc:              # noqa: BLE001
            # A tool crash is the model's problem to route around, not a reason
            # to take the connection down.
            return _tool_failure(f"внутренняя ошибка: {type(exc).__name__}: {exc}")
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": payload.text}],
            "isError": False,
        }
        if self._structured_supported():
            result["structuredContent"] = payload.structured
        return result

    # ---------------------------------------------------------------- stdio #

    def serve(self, stream_in: TextIO, stream_out: TextIO) -> int:
        """Serve newline-delimited JSON-RPC until EOF or a broken output pipe."""
        # A client that dies mid-write leaves us writing into a closed pipe. That
        # is a normal shutdown seen from this end, not a fault: the alternative is
        # a traceback on stderr and a connection closed without `close()`.
        try:
            for line in stream_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as exc:
                    _write(stream_out, _error(None, PARSE_ERROR, str(exc)))
                    continue
                response = self.dispatch(message)
                if response is not None:
                    _write(stream_out, response)
        except BrokenPipeError:
            print("клиент закрыл поток", file=sys.stderr)
        finally:
            self.close()
        return 0


def _hit_header(hit: Hit) -> str:
    symbol = f" {hit.chunk.symbol}" if hit.chunk.symbol else ""
    return f"{hit.chunk.path}:{hit.chunk.start_line}-{hit.chunk.end_line}{symbol}"


def _format_results(results: SearchResults, stale: bool) -> str:
    """The search result as plain text, not JSON.

    The reader is a model, not a parser, and text is cheaper for it: no
    repeated keys, no braces, and -- the biggest item -- no JSON escaping of
    every newline and quote in the code being returned. `kind` and `score` are
    dropped outright: order already carries the ranking, and the score was
    never interpretable across queries anyway.

    The ``----`` fence is deliberately not escaped when chunk text happens to
    contain the same line. Headers are unambiguous `path:a-b` lines, so a model
    recovers the structure anyway, and any escaping scheme would spend tokens
    on every response to guard against a cosmetic ambiguity.
    """
    lines: list[str] = []
    if not results.hits:
        lines.append("ничего не найдено")
    for hit in results.hits:
        lines.append(_hit_header(hit))
        lines.append("----")
        lines.append(hit.chunk.text)
        lines.append("----")
    if results.refs:
        lines.append("ещё релевантное (текст не включён — прочитай указанный "
                     "диапазон строк файла):")
        lines.extend(f"  {_hit_header(hit)}" for hit in results.refs)

    footer: list[str] = []
    if stale:
        footer.append("stale: true (индекс обновляется в фоне, номера строк могут отставать)")
    hashes = _file_hashes(results)
    if hashes:
        footer.append("hash: " + " ".join(f"{path}={digest}" for path, digest in hashes.items()))
    if footer:
        lines.append("")
        lines.extend(footer)
    return "\n".join(lines)


def _file_hashes(results: SearchResults) -> dict[str, str]:
    """One short hash per unique file, not 64 hex per hit: it exists only to be
    compared after an edit (`sha256sum <file> | cut -c1-12`).
    """
    hashes: dict[str, str] = {}
    for hit in results.ranked:
        hashes.setdefault(hit.chunk.path, hit.chunk.file_hash[:12])
    return hashes


def _location(hit: Hit) -> dict[str, Any]:
    return {
        "path": hit.chunk.path,
        "start_line": hit.chunk.start_line,
        "end_line": hit.chunk.end_line,
        "symbol": hit.chunk.symbol,
    }


def _structured_results(results: SearchResults, stale: bool) -> dict[str, Any]:
    """The same answer as `_format_results`, as data. Shape: `SEARCH_OUTPUT_SCHEMA`."""
    return {
        "results": [{**_location(hit), "text": hit.chunk.text} for hit in results.hits],
        "references": [_location(hit) for hit in results.refs],
        "stale": stale,
        "file_hashes": _file_hashes(results),
    }


def _valid_id(message_id: Any) -> bool:
    """String or integer, as MCP requires. `bool` is an `int` in Python and is
    ruled out explicitly, or `true` would pass as an id of 1.
    """
    if isinstance(message_id, bool):
        return False
    return isinstance(message_id, (str, int))


def _error(message_id: Any, code: int, text: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": text}}


def _tool_failure(text: str) -> dict[str, Any]:
    """A failure the model reads, as opposed to one the transport reports."""
    return {"content": [{"type": "text", "text": text}], "isError": True}


def _write(stream: TextIO, message: dict[str, Any] | list[dict[str, Any]]) -> None:
    stream.write(json.dumps(message, ensure_ascii=False) + "\n")
    stream.flush()


def _reconfigure(stream: TextIO, **options: Any) -> None:
    """Best-effort text-mode fix. A stream that cannot be reconfigured is fine:
    that only happens when it is already a plain in-memory buffer, as in tests.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(**options)


def main(argv: list[str] | None = None) -> int:
    """Run the stdio MCP server and return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m code_search.server",
        description="MCP-сервер семантического поиска по коду (stdio).",
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="корень проиндексированного проекта (по умолчанию cwd)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--exclude", action="append", default=[], metavar="GLOB")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"нет каталога: {root}", file=sys.stderr)
        return 2

    # Windows would otherwise translate \n to \r\n on the way out and decode the
    # incoming bytes as cp1251 -- the first breaks framing, the second mangles the
    # Cyrillic in every tool description. Set explicitly rather than left to
    # PYTHONUTF8, which the client is under no obligation to pass through. stdin
    # keeps universal newlines so that a \r\n-terminated client still parses.
    _reconfigure(sys.stdin, encoding="utf-8")
    _reconfigure(sys.stdout, encoding="utf-8", newline="\n")

    print(f"{SERVER_NAME} {SERVER_VERSION}: {root} "
          f"(модель {args.model}, чанкер v{CHUNKER_VERSION})", file=sys.stderr)
    server = Server(root, args.model, tuple(args.exclude))
    try:
        return server.serve(sys.stdin, sys.stdout)
    except KeyboardInterrupt:
        server.close()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
