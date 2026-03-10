from typing import Any

from vet.imbue_core.agents.agent_api.data_types import AgentAssistantMessage
from vet.imbue_core.agents.agent_api.data_types import AgentContentBlock
from vet.imbue_core.agents.agent_api.data_types import AgentMessage
from vet.imbue_core.agents.agent_api.data_types import AgentResultMessage
from vet.imbue_core.agents.agent_api.data_types import AgentSystemEventType
from vet.imbue_core.agents.agent_api.data_types import AgentSystemMessage
from vet.imbue_core.agents.agent_api.data_types import AgentTextBlock
from vet.imbue_core.agents.agent_api.data_types import AgentThinkingBlock
from vet.imbue_core.agents.agent_api.data_types import AgentToolResultBlock
from vet.imbue_core.agents.agent_api.data_types import AgentToolUseBlock
from vet.imbue_core.agents.agent_api.data_types import AgentUnknownMessage
from vet.imbue_core.agents.agent_api.data_types import AgentUsage


def parse_opencode_event(data: dict[str, Any]) -> AgentMessage | None:
    event_type = data.get("type", "")
    part = data.get("part", {})
    session_id = data.get("sessionID", "")

    match event_type:
        case "step_start":
            return AgentSystemMessage(
                event_type=AgentSystemEventType.SESSION_STARTED,
                session_id=session_id,
                original_message=data,
            )

        case "text":
            text = part.get("text", "")
            if not text:
                return None
            return AgentAssistantMessage(
                content=[AgentTextBlock(text=text)],
                original_message=data,
            )

        case "tool_use":
            content_blocks = _parse_tool_use_part(part)
            if not content_blocks:
                return None
            return AgentAssistantMessage(
                content=content_blocks,
                original_message=data,
            )

        case "thinking":
            thinking_text = part.get("text", "")
            if not thinking_text:
                return None
            return AgentAssistantMessage(
                content=[AgentThinkingBlock(content=thinking_text)],
                original_message=data,
            )

        case "step_finish":
            reason = part.get("reason", "")
            if reason != "stop":
                return None

            usage = None
            tokens_data = part.get("tokens")
            if tokens_data:
                cache_data = tokens_data.get("cache", {})
                usage = AgentUsage(
                    input_tokens=tokens_data.get("input", 0),
                    output_tokens=tokens_data.get("output", 0),
                    cached_tokens=cache_data.get("read", 0),
                    total_tokens=tokens_data.get("total", 0),
                    total_cost_usd=part.get("cost"),
                )

            return AgentResultMessage(
                session_id=session_id,
                is_error=False,
                usage=usage,
                original_message=data,
            )

        case "error":
            error_msg = part.get("message", data.get("message", "unknown error"))
            return AgentResultMessage(
                session_id=session_id,
                is_error=True,
                error=error_msg,
                usage=None,
                original_message=data,
            )

        case _:
            return AgentUnknownMessage(raw=data, original_message=data)


def _parse_tool_use_part(part: dict[str, Any]) -> list[AgentContentBlock]:
    call_id = part.get("callID", part.get("id", ""))
    tool_name = part.get("tool", "")
    state = part.get("state", {})
    status = state.get("status", "")
    tool_input = state.get("input", {})
    tool_output = state.get("output", "")

    if isinstance(tool_input, str):
        tool_input = {"input": tool_input}

    blocks: list[AgentContentBlock] = [
        AgentToolUseBlock(
            id=call_id,
            name=tool_name,
            input=tool_input,
        )
    ]

    if status == "completed":
        metadata = part.get("metadata", {}) or {}
        exit_code = metadata.get("exit")
        blocks.append(
            AgentToolResultBlock(
                tool_use_id=call_id,
                content=tool_output,
                is_error=exit_code is not None and exit_code != 0,
                exit_code=exit_code,
            )
        )

    return blocks
