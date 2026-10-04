from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from app.vector_store import SearchHit

AnswerStatus = Literal["answered", "insufficient_context"]


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    answer: str
    status: AnswerStatus
    citation_indices: tuple[int, ...]


class AnswerGenerator(Protocol):
    @property
    def provider(self) -> str:
        ...

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> GeneratedAnswer:
        ...


class ExtractiveAnswerGenerator:
    """Deterministic grounded baseline that quotes the strongest retrieved span."""

    def __init__(self, min_relevance: float = 0.05, max_answer_chars: int = 600) -> None:
        if not -1.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between -1 and 1.")
        if max_answer_chars <= 0:
            raise ValueError("max_answer_chars must be greater than zero.")
        self._min_relevance = min_relevance
        self._max_answer_chars = max_answer_chars

    @property
    def provider(self) -> str:
        return "extractive"

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("Question must contain readable text.")
        if not contexts or contexts[0].score < self._min_relevance:
            return GeneratedAnswer(
                answer="I could not find enough relevant context to answer that question.",
                status="insufficient_context",
                citation_indices=(),
            )
        answer = contexts[0].chunk.text[: self._max_answer_chars].strip()
        return GeneratedAnswer(answer=answer, status="answered", citation_indices=(0,))


class AnswerProviderError(RuntimeError):
    """Raised when a remote answer provider fails or returns an unsafe response."""


class OpenAIAnswerGenerator:
    """OpenAI-compatible grounded generator with validated citation indices."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4.1-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required for the OpenAI answer provider.")
        if not model.strip():
            raise ValueError("model must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.Client()

    @property
    def provider(self) -> str:
        return "openai"

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> GeneratedAnswer:
        if not question.strip():
            raise ValueError("Question must contain readable text.")
        if not contexts:
            return GeneratedAnswer(
                answer="I could not find enough relevant context to answer that question.",
                status="insufficient_context",
                citation_indices=(),
            )

        sources = "\n\n".join(
            f"SOURCE {index}\n"
            f"document_id={hit.document_id} chunk={hit.chunk.index} "
            f"chars={hit.chunk.start_char}:{hit.chunk.end_char}\n{hit.chunk.text}"
            for index, hit in enumerate(contexts, start=1)
        )
        try:
            response = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer only from the numbered sources. Return JSON with "
                                "status ('answered' or 'insufficient_context'), answer, and "
                                "citations (a list of 1-based source numbers). If the sources "
                                "do not support the answer, abstain with no citations."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"QUESTION\n{question}\n\nSOURCES\n{sources}",
                        },
                    ],
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            result = json.loads(content)
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise AnswerProviderError("Remote answer request failed validation.") from exc

        status = result.get("status")
        answer = result.get("answer")
        citations = result.get("citations")
        if status not in {"answered", "insufficient_context"} or not isinstance(answer, str):
            raise AnswerProviderError("Answer response has an invalid status or answer.")
        if not answer.strip() or len(answer) > 4_000 or not isinstance(citations, list):
            raise AnswerProviderError("Answer response has invalid content.")
        if not all(isinstance(index, int) and 1 <= index <= len(contexts) for index in citations):
            raise AnswerProviderError("Answer response contains an invalid citation.")

        unique_indices = tuple(dict.fromkeys(index - 1 for index in citations))
        if status == "answered" and not unique_indices:
            raise AnswerProviderError("A grounded answer must cite at least one source.")
        if status == "insufficient_context" and unique_indices:
            raise AnswerProviderError("An abstention must not cite a source.")
        return GeneratedAnswer(answer.strip(), status, unique_indices)


def build_answer_generator(
    *,
    provider: str,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4.1-mini",
    openai_base_url: str = "https://api.openai.com/v1",
    timeout_seconds: float = 30.0,
    min_relevance: float = 0.05,
) -> AnswerGenerator:
    if provider == "extractive":
        return ExtractiveAnswerGenerator(min_relevance=min_relevance)
    if provider == "openai":
        if openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when ANSWER_PROVIDER=openai.")
        return OpenAIAnswerGenerator(
            api_key=openai_api_key,
            model=openai_model,
            base_url=openai_base_url,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"Unsupported answer provider: {provider}")
