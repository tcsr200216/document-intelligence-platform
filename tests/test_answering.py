import json

import httpx
import pytest

from app.answering import AnswerProviderError, ExtractiveAnswerGenerator, OpenAIAnswerGenerator
from app.parsing import TextChunk
from app.vector_store import SearchHit


def hit(text: str, score: float, index: int = 0, page: int | None = None) -> SearchHit:
    return SearchHit(
        document_id="doc-1",
        chunk=TextChunk(
            index=index,
            text=text,
            start_char=index * 10,
            end_char=index * 10 + len(text),
            page_start=page,
            page_end=page,
        ),
        score=score,
    )


def test_extractive_generator_quotes_only_strongest_source() -> None:
    generator = ExtractiveAnswerGenerator(min_relevance=0.2)

    result = generator.generate(
        "Where is metadata stored?",
        [hit("PostgreSQL stores metadata.", 0.9), hit("Redis is a cache.", 0.5, 1)],
    )

    assert result.status == "answered"
    assert result.answer == "PostgreSQL stores metadata."
    assert result.citation_indices == (0,)


def test_extractive_generator_abstains_below_threshold() -> None:
    result = ExtractiveAnswerGenerator(min_relevance=0.5).generate(
        "Unknown?", [hit("Unrelated source", 0.2)]
    )

    assert result.status == "insufficient_context"
    assert result.citation_indices == ()


def test_openai_generator_validates_and_maps_citations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        assert payload["model"] == "answer-test"
        assert "SOURCE 2" in payload["messages"][1]["content"]
        assert "pages=4:4" in payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "answered",
                                    "answer": "Redis is used as a cache.",
                                    "citations": [2],
                                }
                            )
                        }
                    }
                ]
            },
        )

    generator = OpenAIAnswerGenerator(
        api_key="test-key",
        model="answer-test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = generator.generate(
        "What is Redis used for?",
        [
            hit("PostgreSQL stores metadata.", 0.8),
            hit("Redis is used as a cache.", 0.7, 1, page=4),
        ],
    )

    assert result.answer == "Redis is used as a cache."
    assert result.citation_indices == (1,)


def test_openai_generator_rejects_out_of_range_citation() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"status": "answered", "answer": "Claim", "citations": [2]}
                            )
                        }
                    }
                ]
            },
        )
    )
    generator = OpenAIAnswerGenerator(
        api_key="test-key", client=httpx.Client(transport=transport)
    )

    with pytest.raises(AnswerProviderError, match="invalid citation"):
        generator.generate("Question?", [hit("Only source", 0.9)])
