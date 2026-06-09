from dataclasses import dataclass

import pytest

from vet.imbue_core.agents.llm_apis.anthropic_api import _extract_text_from_content_blocks
from vet.imbue_core.agents.llm_apis.errors import BadAPIRequestError


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


def test_refusal_with_empty_content_returns_empty_string() -> None:
    # Fable 5's safety classifiers can return a refusal with no content blocks at all.
    assert _extract_text_from_content_blocks([], stop_reason="refusal") == ""


def test_empty_content_without_refusal_raises() -> None:
    with pytest.raises(BadAPIRequestError):
        _extract_text_from_content_blocks([], stop_reason="end_turn")


def test_only_thinking_block_without_refusal_raises() -> None:
    with pytest.raises(BadAPIRequestError):
        _extract_text_from_content_blocks([_Block(type="thinking", text="(reasoning)")], stop_reason="end_turn")
