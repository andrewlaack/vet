from dataclasses import dataclass

import pytest

from vet.imbue_core.agents.llm_apis.anthropic_api import AnthropicAPI
from vet.imbue_core.agents.llm_apis.anthropic_api import AnthropicModelName
from vet.imbue_core.agents.llm_apis.anthropic_api import _extract_text_from_content_blocks
from vet.imbue_core.agents.llm_apis.errors import BadAPIRequestError
from vet.imbue_core.agents.llm_apis.errors import ModelRefusalError


@dataclass
class _Block:
    type: str
    text: str = ""


def test_single_text_block_returns_text() -> None:
    assert _extract_text_from_content_blocks([_Block(type="text", text="hello")]) == "hello"


def test_thinking_block_is_ignored() -> None:
    # Fable 5 emits a thinking block before the text block even without extended thinking enabled.
    blocks = [_Block(type="thinking", text="(reasoning)"), _Block(type="text", text="answer")]
    assert _extract_text_from_content_blocks(blocks) == "answer"


def test_multiple_text_blocks_are_concatenated() -> None:
    blocks = [_Block(type="text", text="foo"), _Block(type="text", text="bar")]
    assert _extract_text_from_content_blocks(blocks) == "foobar"


def test_refusal_with_empty_content_raises_model_refusal_error() -> None:
    # Fable 5's safety classifiers can return a refusal with no content blocks at all. This must
    # surface as an error to the user rather than being treated as an empty (clean) response.
    with pytest.raises(ModelRefusalError):
        _extract_text_from_content_blocks([], stop_reason="refusal")


def test_refusal_with_text_returns_text() -> None:
    # If the refusal stop reason somehow accompanies generated text, keep the text.
    blocks = [_Block(type="text", text="partial answer")]
    assert _extract_text_from_content_blocks(blocks, stop_reason="refusal") == "partial answer"


def test_empty_content_without_refusal_raises() -> None:
    with pytest.raises(BadAPIRequestError):
        _extract_text_from_content_blocks([], stop_reason="end_turn")


def test_only_thinking_block_without_refusal_raises() -> None:
    with pytest.raises(BadAPIRequestError):
        _extract_text_from_content_blocks([_Block(type="thinking", text="(reasoning)")], stop_reason="end_turn")


def test_default_timeout_per_model() -> None:
    # Fable 5's always-on adaptive thinking makes review-sized responses routinely take minutes,
    # so it gets a much longer default timeout than other models.
    fable = AnthropicAPI(model_name=AnthropicModelName.CLAUDE_FABLE_5, cache_path=None)
    assert fable._get_timeout() == 600.0

    opus = AnthropicAPI(model_name=AnthropicModelName.CLAUDE_4_8_OPUS, cache_path=None)
    assert opus._get_timeout() == 60.0

    explicit = AnthropicAPI(model_name=AnthropicModelName.CLAUDE_FABLE_5, cache_path=None, timeout=30.0)
    assert explicit._get_timeout() == 30.0
