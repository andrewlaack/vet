from vet.imbue_core.agents.agent_api.data_types import AgentAssistantMessage
from vet.imbue_core.agents.agent_api.data_types import AgentResultMessage
from vet.imbue_core.agents.agent_api.data_types import AgentSystemEventType
from vet.imbue_core.agents.agent_api.data_types import AgentSystemMessage
from vet.imbue_core.agents.agent_api.data_types import AgentTextBlock
from vet.imbue_core.agents.agent_api.data_types import AgentThinkingBlock
from vet.imbue_core.agents.agent_api.data_types import AgentToolResultBlock
from vet.imbue_core.agents.agent_api.data_types import AgentToolUseBlock
from vet.imbue_core.agents.agent_api.data_types import AgentUnknownMessage
from vet.imbue_core.agents.agent_api.opencode.message_parser import parse_opencode_event


class TestParseStepStart:
    def test_step_start_returns_system_message(self) -> None:
        data = {
            "type": "step_start",
            "timestamp": 1773096529551,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_1",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "step-start",
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentSystemMessage)
        assert message.event_type == AgentSystemEventType.SESSION_STARTED
        assert message.session_id == "ses_abc123"

    def test_step_start_with_snapshot(self) -> None:
        data = {
            "type": "step_start",
            "timestamp": 1773096529551,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_1",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "step-start",
                "snapshot": "abc123hash",
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentSystemMessage)
        assert message.session_id == "ses_abc123"


class TestParseText:
    def test_text_returns_assistant_message(self) -> None:
        data = {
            "type": "text",
            "timestamp": 1773096520559,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_2",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "text",
                "text": "The answer is 4",
                "time": {"start": 1773096520559, "end": 1773096520559},
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentAssistantMessage)
        assert len(message.content) == 1
        assert isinstance(message.content[0], AgentTextBlock)
        assert message.content[0].text == "The answer is 4"

    def test_empty_text_returns_none(self) -> None:
        data = {
            "type": "text",
            "timestamp": 1773096520559,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_2",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "text",
                "text": "",
            },
        }
        message = parse_opencode_event(data)
        assert message is None


class TestParseToolUse:
    def test_completed_tool_use_returns_use_and_result(self) -> None:
        data = {
            "type": "tool_use",
            "timestamp": 1773096529615,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_3",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "tool",
                "callID": "toolu_01TPK",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "input": {"command": "ls -la", "description": "List files"},
                    "output": "file1.txt\nfile2.txt",
                },
                "metadata": {"output": "file1.txt\nfile2.txt", "exit": 0},
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentAssistantMessage)
        assert len(message.content) == 2

        tool_use = message.content[0]
        assert isinstance(tool_use, AgentToolUseBlock)
        assert tool_use.id == "toolu_01TPK"
        assert tool_use.name == "bash"
        assert tool_use.input == {"command": "ls -la", "description": "List files"}

        tool_result = message.content[1]
        assert isinstance(tool_result, AgentToolResultBlock)
        assert tool_result.tool_use_id == "toolu_01TPK"
        assert tool_result.content == "file1.txt\nfile2.txt"
        assert tool_result.is_error is False
        assert tool_result.exit_code == 0

    def test_failed_tool_use_marks_error(self) -> None:
        data = {
            "type": "tool_use",
            "timestamp": 1773096529615,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_3",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "tool",
                "callID": "toolu_02XYZ",
                "tool": "bash",
                "state": {
                    "status": "completed",
                    "input": {"command": "false"},
                    "output": "",
                },
                "metadata": {"exit": 1},
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentAssistantMessage)
        tool_result = message.content[1]
        assert isinstance(tool_result, AgentToolResultBlock)
        assert tool_result.is_error is True
        assert tool_result.exit_code == 1

    def test_in_progress_tool_use_no_result(self) -> None:
        data = {
            "type": "tool_use",
            "timestamp": 1773096529615,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_3",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "tool",
                "callID": "toolu_03ABC",
                "tool": "bash",
                "state": {
                    "status": "pending",
                    "input": {"command": "sleep 10"},
                    "output": "",
                },
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentAssistantMessage)
        assert len(message.content) == 1
        assert isinstance(message.content[0], AgentToolUseBlock)

    def test_tool_use_with_string_input(self) -> None:
        data = {
            "type": "tool_use",
            "timestamp": 1773096529615,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_3",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "tool",
                "callID": "toolu_04DEF",
                "tool": "read",
                "state": {
                    "status": "completed",
                    "input": "/path/to/file.py",
                    "output": "file contents",
                },
                "metadata": {},
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentAssistantMessage)
        tool_use = message.content[0]
        assert isinstance(tool_use, AgentToolUseBlock)
        assert tool_use.input == {"input": "/path/to/file.py"}


class TestParseThinking:
    def test_thinking_returns_thinking_block(self) -> None:
        data = {
            "type": "thinking",
            "timestamp": 1773096520559,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_4",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "thinking",
                "text": "Let me analyze this...",
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentAssistantMessage)
        assert len(message.content) == 1
        assert isinstance(message.content[0], AgentThinkingBlock)
        assert message.content[0].content == "Let me analyze this..."

    def test_empty_thinking_returns_none(self) -> None:
        data = {
            "type": "thinking",
            "timestamp": 1773096520559,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_4",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "thinking",
                "text": "",
            },
        }
        message = parse_opencode_event(data)
        assert message is None


class TestParseStepFinish:
    def test_stop_reason_returns_result_message(self) -> None:
        data = {
            "type": "step_finish",
            "timestamp": 1773096520590,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_5",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "step-finish",
                "reason": "stop",
                "cost": 0.07321,
                "tokens": {
                    "total": 11699,
                    "input": 2,
                    "output": 5,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 11692},
                },
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentResultMessage)
        assert message.session_id == "ses_abc123"
        assert message.is_error is False
        assert message.usage is not None
        assert message.usage.input_tokens == 2
        assert message.usage.output_tokens == 5
        assert message.usage.total_tokens == 11699
        assert message.usage.cached_tokens == 0
        assert message.usage.total_cost_usd == 0.07321

    def test_tool_calls_reason_returns_none(self) -> None:
        data = {
            "type": "step_finish",
            "timestamp": 1773096520590,
            "sessionID": "ses_abc123",
            "part": {
                "id": "prt_5",
                "sessionID": "ses_abc123",
                "messageID": "msg_1",
                "type": "step-finish",
                "reason": "tool-calls",
                "cost": 0.07481625,
                "tokens": {
                    "total": 11746,
                    "input": 2,
                    "output": 75,
                    "reasoning": 0,
                    "cache": {"read": 0, "write": 11669},
                },
            },
        }
        message = parse_opencode_event(data)
        assert message is None


class TestParseError:
    def test_error_returns_error_result(self) -> None:
        data = {
            "type": "error",
            "timestamp": 1773096520590,
            "sessionID": "ses_abc123",
            "part": {
                "message": "Rate limit exceeded",
            },
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentResultMessage)
        assert message.is_error is True
        assert message.error == "Rate limit exceeded"


class TestParseUnknown:
    def test_unknown_type_returns_unknown_message(self) -> None:
        data = {
            "type": "some_future_event",
            "timestamp": 1773096520590,
            "sessionID": "ses_abc123",
            "part": {"foo": "bar"},
        }
        message = parse_opencode_event(data)
        assert isinstance(message, AgentUnknownMessage)
        assert message.raw == data
