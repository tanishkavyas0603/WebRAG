"""
Regression tests for reasoning_effort model-compatibility.

History:
  1. Originally GROQ_MODEL was a Qwen reasoning model. Without
     reasoning_effort="none", Qwen reliably burned the entire max_tokens
     budget on hidden <think> reasoning and returned truncated/empty
     content, which surfaced to users as "Failed to generate answer".

  2. GROQ_MODEL was later switched (via Render env var) to
     openai/gpt-oss-120b, but the code still unconditionally sent
     reasoning_effort="none" on every call. Per the groq SDK's own
     documented parameter values (groq/resources/chat/completions.py):

         reasoning_effort: qwen3 models support the following values
         Set to 'none' to disable reasoning. Set to 'default' or null
         to let Qwen reason.
         openai/gpt-oss-20b and openai/gpt-oss-120b support 'low',
         'medium', or 'high'. 'medium' is the default value.

     "none" is Qwen-specific and is REJECTED by the Groq API with
     HTTP 400 for gpt-oss models. That 400 was caught by the generic
     APIStatusError handler in rag_service.py and surfaced to users as
     HTTP 503 "AI model is currently unavailable or misconfigured" on
     every single /messages request.

_supports_reasoning_effort() / _build_groq_kwargs() in
app/services/rag_service.py fix this by only injecting reasoning_effort
for model families known to require it (Qwen). For any other model
(including openai/gpt-oss-120b) the parameter is omitted entirely, so
Groq falls back to that model's own documented default instead of the
app guessing a value. These tests cover both model families explicitly
(independent of whatever GROQ_MODEL happens to be set to in the local
.env) plus the helper functions directly.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.services.rag_service import (
    RAGService,
    _supports_reasoning_effort,
    _build_groq_kwargs,
)
from app.models.response import RetrievalResult


def _make_chunk() -> RetrievalResult:
    return RetrievalResult(
        chunk_id=1,
        content="HTTP is a protocol.",
        title="HTTP",
        section="",
        preview="",
        similarity_score=0.9,
        cosine_score=0.9,
        metadata_boost=0.0,
        source="test",
    )


@pytest.fixture
def rag_service():
    with patch('app.services.rag_service.Groq'), \
         patch('app.services.rag_service.RetrievalService'):
        service = RAGService(document_id=1)
        service.client.chat.completions.create = MagicMock()
        service.retriever.search = MagicMock()
        return service


def _mock_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# --------------------------------------------------------------------- #
# Unit tests for the model -> reasoning_effort support/kwargs helpers.   #
# --------------------------------------------------------------------- #

def test_supports_reasoning_effort_true_for_qwen():
    assert _supports_reasoning_effort("qwen/qwen3.8-27b") is True


def test_supports_reasoning_effort_false_for_gpt_oss_120b():
    # This is the exact production model from the incident. Sending
    # reasoning_effort="none" to this model is rejected by Groq with
    # HTTP 400.
    assert _supports_reasoning_effort("openai/gpt-oss-120b") is False


def test_supports_reasoning_effort_false_for_gpt_oss_20b():
    assert _supports_reasoning_effort("openai/gpt-oss-20b") is False


def test_supports_reasoning_effort_false_for_unknown_model():
    # Unknown/future model families default to "unsupported" (parameter
    # omitted) rather than guessing, so a future GROQ_MODEL swap doesn't
    # silently reintroduce this exact bug.
    assert _supports_reasoning_effort("some-future-model") is False


def test_build_groq_kwargs_injects_reasoning_effort_for_qwen():
    kwargs = _build_groq_kwargs("qwen/qwen3.8-27b", {"model": "qwen/qwen3.8-27b"})
    assert kwargs.get("reasoning_effort") == "none"


def test_build_groq_kwargs_omits_reasoning_effort_for_gpt_oss_120b():
    kwargs = _build_groq_kwargs("openai/gpt-oss-120b", {"model": "openai/gpt-oss-120b"})
    assert "reasoning_effort" not in kwargs


def test_build_groq_kwargs_does_not_mutate_caller_dict():
    original = {"model": "qwen/qwen3.8-27b"}
    _build_groq_kwargs("qwen/qwen3.8-27b", original)
    assert "reasoning_effort" not in original


# --------------------------------------------------------------------- #
# Qwen behavior (pre-existing, still correct for this model family).     #
# --------------------------------------------------------------------- #

def test_call_llm_disables_reasoning_for_qwen(rag_service):
    with patch('app.services.rag_service.settings.GROQ_MODEL', 'qwen/qwen3.8-27b'):
        rag_service.client.chat.completions.create.return_value = _mock_response("The answer.")
        rag_service._call_llm([{"role": "user", "content": "hi"}])

    _, kwargs = rag_service.client.chat.completions.create.call_args
    assert kwargs.get("reasoning_effort") == "none"


def test_rewrite_query_disables_reasoning_for_qwen(rag_service):
    with patch('app.services.rag_service.settings.GROQ_MODEL', 'qwen/qwen3.8-27b'):
        rag_service.client.chat.completions.create.return_value = _mock_response("Standalone question?")
        rag_service._rewrite_query("follow up", [{"role": "user", "content": "hello"}])

    _, kwargs = rag_service.client.chat.completions.create.call_args
    assert kwargs.get("reasoning_effort") == "none"


# --------------------------------------------------------------------- #
# GPT-OSS behavior - the actual production regression.                   #
# --------------------------------------------------------------------- #

def test_call_llm_omits_reasoning_effort_for_gpt_oss_120b(rag_service):
    """
    Regression test for the production 503 incident: calling the Groq
    chat completions API for openai/gpt-oss-120b must never send
    reasoning_effort="none" (that model family rejects it with HTTP 400,
    which the app was surfacing to users as a 503).
    """
    with patch('app.services.rag_service.settings.GROQ_MODEL', 'openai/gpt-oss-120b'):
        rag_service.client.chat.completions.create.return_value = _mock_response("The answer.")
        rag_service._call_llm([{"role": "user", "content": "hi"}])

    _, kwargs = rag_service.client.chat.completions.create.call_args
    assert "reasoning_effort" not in kwargs


def test_rewrite_query_omits_reasoning_effort_for_gpt_oss_120b(rag_service):
    with patch('app.services.rag_service.settings.GROQ_MODEL', 'openai/gpt-oss-120b'):
        rag_service.client.chat.completions.create.return_value = _mock_response("Standalone question?")
        rag_service._rewrite_query("follow up", [{"role": "user", "content": "hello"}])

    _, kwargs = rag_service.client.chat.completions.create.call_args
    assert "reasoning_effort" not in kwargs


def test_answer_end_to_end_omits_reasoning_effort_for_gpt_oss_120b(rag_service):
    dummy_chunk = _make_chunk()
    rag_service.retriever.search.return_value = ([dummy_chunk], MagicMock(expanded="what is HTTP", was_expanded=False), [0.9])
    rag_service.client.chat.completions.create.return_value = _mock_response("HTTP is a protocol.")

    with patch('app.services.rag_service.settings.GROQ_MODEL', 'openai/gpt-oss-120b'):
        response = rag_service.answer("what is HTTP")

    assert response.answer == "HTTP is a protocol."
    _, kwargs = rag_service.client.chat.completions.create.call_args
    assert "reasoning_effort" not in kwargs
