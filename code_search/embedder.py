"""Turn text into vectors via a local Ollama instance.

Vectors are L2-normalised on arrival, so cosine similarity later is a plain dot
product and no ranking code has to remember to normalise.

Ollama is addressed over HTTP with the standard library only -- the whole point
of running the model locally is to avoid a dependency chain, and `urllib` is
enough for one POST.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass

__all__ = ["Embedder", "EmbeddingError", "MODELS", "ModelSpec"]

DEFAULT_URL = "http://127.0.0.1:11434/api/embed"
DEFAULT_BATCH = 16

# How long Ollama keeps the model resident. A cold load of bge-m3 costs seconds,
# which would dominate a single interactive query.
DEFAULT_KEEP_ALIVE = "30m"


class EmbeddingError(RuntimeError):
    """Ollama could not be reached, or answered with something unusable."""


@dataclass(frozen=True)
class ModelSpec:
    """What a model needs beyond its name.

    `document_prefix` and `query_prefix` exist because some models are trained
    with mandatory task prefixes and score near zero without them. bge-m3 needs
    none; nomic-embed-text does. Getting this wrong is silent -- the model still
    returns plausible vectors, they are just bad -- so it lives in code, not in
    a caller's memory.
    """

    name: str
    dimension: int
    document_prefix: str = ""
    query_prefix: str = ""


MODELS: dict[str, ModelSpec] = {
    "bge-m3": ModelSpec("bge-m3", 1024),
    "nomic-embed-text": ModelSpec(
        "nomic-embed-text", 768,
        document_prefix="search_document: ", query_prefix="search_query: ",
    ),
}

DEFAULT_MODEL = "bge-m3"


def _normalise(values: list[float]) -> array[float]:
    length = math.sqrt(sum(x * x for x in values)) or 1.0
    return array("f", (x / length for x in values))


class Embedder:
    """A thin, retrying client for one model."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL,
                 batch: int = DEFAULT_BATCH, keep_alive: str = DEFAULT_KEEP_ALIVE,
                 timeout: float = 900.0, retries: int = 3) -> None:
        if model not in MODELS:
            known = ", ".join(sorted(MODELS))
            raise EmbeddingError(
                f"unknown model {model!r}; known models: {known}. "
                "Add a ModelSpec -- guessing the task prefix is not safe."
            )
        self.spec = MODELS[model]
        self.url = url
        self.batch = batch
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.retries = retries

    @property
    def dimension(self) -> int:
        """Number of floats produced for one text by the selected model."""
        return self.spec.dimension

    def _post(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({
            "model": self.spec.name, "input": inputs, "keep_alive": self.keep_alive,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, body, {"Content-Type": "application/json"})

        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                embeddings = payload.get("embeddings")
                if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
                    raise EmbeddingError(
                        f"expected {len(inputs)} embeddings, got {type(embeddings)}")
                return embeddings
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt)
        raise EmbeddingError(
            f"Ollama at {self.url} did not answer after {self.retries} attempts: {last}"
        ) from last

    def _check(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self.spec.dimension:
                raise EmbeddingError(
                    f"{self.spec.name} returned {len(vector)} dimensions, "
                    f"expected {self.spec.dimension}; the index would be corrupt"
                )

    def embed_documents(self, texts: list[str]) -> list[array[float]]:
        """Embed corpus text, one batch at a time."""
        out: list[array[float]] = []
        prefix = self.spec.document_prefix
        for start in range(0, len(texts), self.batch):
            window = [prefix + t for t in texts[start:start + self.batch]]
            vectors = self._post(window)
            self._check(vectors)
            out.extend(_normalise(v) for v in vectors)
        return out

    def embed_query(self, text: str) -> array[float]:
        """Embed one query. Uses the query prefix, not the document prefix."""
        vectors = self._post([self.spec.query_prefix + text])
        self._check(vectors)
        return _normalise(vectors[0])
