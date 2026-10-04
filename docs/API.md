# 🧩 Интерфейсы code-search

Справочник охватывает поддерживаемые CLI-команды, MCP-инструменты и публичные
Python-интерфейсы. Для обычного использования достаточно [SETUP.md](../SETUP.md).

## ⌨️ CLI

### 🗂️ Индексатор

```bash
python3 -m code_search.indexer ROOT [--model MODEL] [--exclude GLOB] \
  [--suffix EXT] [--dry-run] [--rebuild]
```

| Аргумент | Назначение |
|---|---|
| `ROOT` | Корень индексируемого проекта |
| `--model` | Модель Ollama; по умолчанию `bge-m3` |
| `--exclude GLOB` | Повторяемый `fnmatch`-шаблон исключения |
| `--suffix EXT` | Добавить расширение, например `.ts`; `none` очищает дополнения |
| `--dry-run` | Показать план без эмбеддинга и записи |
| `--rebuild` | Удалить совместимый индекс и собрать его заново |

Коды возврата: `0` — успех, `1` — ошибка Ollama или несовместимая SQLite,
`2` — некорректный путь/аргумент, `130` — прерывание пользователем.

### 🔍 Ручной поиск

```bash
python3 -m code_search.search ROOT QUERY [-k K] \
  [--mode hybrid|vector|lexical] [--path-glob GLOB] [--full]
```

Команда предназначена для диагностики. Без `--full` печатаются расположения
чанков; с `--full` — также их текст.

### 🔌 MCP-сервер

```bash
python3 -m code_search.server [--root ROOT] [--model MODEL] [--exclude GLOB]
```

Сервер читает по одному JSON-RPC-сообщению на строку из `stdin` и пишет ответы в
`stdout`. Логи и баннер отправляются в `stderr`. Корректное завершение — `0`,
неверный корень — `2`, `Ctrl-C` — `130`.

### 📊 Оценка качества

```bash
python3 eval/harness.py ROOT [--queries FILE] [--depth N] \
  [--save FILE] [--quiet] [--unblind]
```

Харнесс требует dev-зависимость `PyYAML`. Файл запросов — YAML-список объектов
`query`, `expect`, `kind`. `--unblind` обязателен, если имя файла содержит
`holdout`; `--save` сохраняет метрики без текста запросов и ожидаемых путей.

## 🔌 MCP-инструменты

### 🔎 `search_code`

Входные аргументы:

| Поле | Тип | Обязательное | Ограничения |
|---|---|---:|---|
| `query` | string | да | Непустая строка |
| `k` | integer | нет | По умолчанию `15`, сервер ограничивает максимумом `50` |
| `path_glob` | string | нет | `fnmatch`-шаблон пути относительно корня |

Текстовый результат содержит заголовки `path:start-end`, исходный текст лучших
чанков, ссылки на хвост выдачи, флаг `stale` и короткие хеши файлов. При протоколе
`2025-06-18` дополнительно возвращается `structuredContent`:

```json
{
  "results": [
    {"path": "pkg/a.py", "start_line": 10, "end_line": 24,
     "symbol": "load", "text": "..."}
  ],
  "references": [
    {"path": "pkg/b.py", "start_line": 5, "end_line": 18, "symbol": null}
  ],
  "stale": false,
  "file_hashes": {"pkg/a.py": "0123456789ab"}
}
```

### 📈 `index_status`

Аргументов нет. Возвращает количество файлов и чанков, модель, версию чанкера,
число устаревших файлов, состояние фонового refresh и последнюю ошибку обновления.

## 🐍 Python API

Проект запускается из checkout и не обещает стабильность Python API между
версиями. Список ниже описывает текущие публичные символы и практические точки
расширения; он полезен прежде всего для тестов и внутренних интеграций.

### ✂️ `code_search.chunker`

- `Chunk` — неизменяемый фрагмент с путём, строками, текстом и метаданными символа.
- `chunk_text(path, text, kind=None, fallback=False)` — нарезать один файл.
- `classify(path)` — определить `source`, `test` или `docs`.
- `enrich(chunk)` — текст для семантического эмбеддинга.
- `fts_text(chunk)` — текст для лексического индекса.
- `estimate_tokens(text)` — дешёвая оценка размера.

### 📁 `code_search.corpus`

- `FileState` — `mtime_ns` и размер файла.
- `list_files(root, excludes=(), extra_suffixes=frozenset())` — корпус индекса.
- `skipped_by_suffix(...)` — отчёт о пропущенных расширениях.
- `state_of`, `read_text`, `content_hash` — состояние и содержимое файла.

### 🧠 `code_search.embedder`

- `ModelSpec` и `MODELS` — проверенные параметры моделей.
- `Embedder` — батчевый клиент Ollama с повторами и проверкой размерности.
- `EmbeddingError` — ожидаемая ошибка модели или транспорта.

### 🗂️ `code_search.indexer`

- `index_path(root)` — путь `<root>/.code-search/index.sqlite`.
- `Plan` — списки неизменившихся, touched, изменённых и удалённых файлов.
- `IndexStats` — итоговые счётчики и `summary()`.
- `plan(...)` — вычислить обновление без эмбеддинга.
- `refresh_stats(...)` — принять новый stat при неизменном содержимом.
- `reindex(...)` — выполнить инкрементальную или полную индексацию.

### 🗄️ `code_search.store`

- `open_index(path)` — открыть БД, применить схему и включить WAL/foreign keys.
- `StoredChunk` — чанк, загруженный из SQLite.
- `read_meta`, `write_meta`, `extra_suffixes`, `stamp` — метаданные индекса.
- `compatibility_problems` — причины обязательной пересборки.
- `indexed_files`, `replace_file`, `drop_files` — операции записи корпуса.
- `stats`, `iter_vectors`, `load_chunks` — чтение для статуса и поиска.
- `IncompatibleIndex` — несовместимая версия SQLite или индекса.

### 🔀 `code_search.search`

- `Searcher.search(query, k=15, path_glob=None, mode="hybrid", max_tokens=3000)` —
  основной программный вход.
- `SearchResults` — полные `hits`, облегчённые `refs`, режим и бюджет токенов.
- `Hit` — чанк, итоговый score и ранги ветвей.
- `VectorIndex` — кэшированный плотный индекс в памяти.
- `fts_expression`, `lexical_rank`, `allowed_ids`, `reciprocal_rank_fusion` —
  отдельные этапы ранжирования.

### ♻️ `code_search.refresh`

- `Refresher.status()` — снимок состояния индекса.
- `Refresher.ensure_fresh()` — запустить refresh при необходимости без ожидания.
- `Refresher.wait(timeout=60)` — дождаться фоновой задачи.
- `Status` — файлы, чанки, модель, stale-счётчик, refresh и ошибка.

### 📡 `code_search.server`

- `Server.dispatch(decoded)` — обработать JSON-RPC объект или batch.
- `Server.search_code(arguments)` и `index_status(arguments)` — тела инструментов.
- `Server.serve(input, output)` — обслуживать newline-delimited поток.
- `ToolResult` — текстовое и структурированное представления результата.
- `ToolError` — ошибка параметров или выполнения, видимая MCP-клиенту.

## 🔁 Совместимость

Поддерживаемые версии MCP: `2024-11-05`, `2025-03-26`, `2025-06-18`. Структурные
результаты выдаются только для `2025-06-18`. Формат SQLite защищён
`SCHEMA_VERSION`, границы чанков — `CHUNKER_VERSION`; несовпадение требует rebuild.
