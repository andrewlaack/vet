import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from vet.imbue_core.agents.llm_apis.data_types import ResponseStopReason
from vet.imbue_core.agents.llm_apis.errors import ModelRefusalError
from vet.imbue_core.agents.llm_apis.openai_compatible_api import OpenAICompatibleAPI


def _make_api() -> OpenAICompatibleAPI:
    return OpenAICompatibleAPI(
        model_name="test-model",
        context_window=100000,
        max_output_tokens=1000,
        cache_path=None,
    )


def _make_completion(content: str | None, finish_reason: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,  # type: ignore[arg-type]
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def test_parse_response_returns_text() -> None:
    api = _make_api()
    results = api._parse_response(
        _make_completion("hello", "stop"), prompt_tokens=10, stop=None, network_failure_count=0
    )
    assert results[0].text == "hello"
    assert results[0].stop_reason == ResponseStopReason.END_TURN


def test_parse_response_content_filter_with_empty_content_raises() -> None:
    # Anthropic's OpenAI-compatible endpoint returns finish_reason "content_filter" with empty
    # content when a model with safety classifiers (e.g. Claude Fable 5) refuses a request.
    api = _make_api()
    with pytest.raises(ModelRefusalError):
        api._parse_response(
            _make_completion("", "content_filter"), prompt_tokens=10, stop=None, network_failure_count=0
        )


def test_parse_response_content_filter_with_none_content_raises() -> None:
    api = _make_api()
    with pytest.raises(ModelRefusalError):
        api._parse_response(
            _make_completion(None, "content_filter"), prompt_tokens=10, stop=None, network_failure_count=0
        )


def test_parse_response_content_filter_with_partial_text_keeps_text() -> None:
    # If the filter triggered after some text was generated, the partial text is kept.
    api = _make_api()
    results = api._parse_response(
        _make_completion("partial", "content_filter"), prompt_tokens=10, stop=None, network_failure_count=0
    )
    assert results[0].text == "partial"
    assert results[0].stop_reason == ResponseStopReason.CONTENT_FILTER
