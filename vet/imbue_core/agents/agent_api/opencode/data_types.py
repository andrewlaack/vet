from pathlib import Path
from typing import Literal

from vet.imbue_core.agents.agent_api.data_types import AgentOptions
from vet.imbue_core.agents.agent_api.data_types import AgentToolName


class OpenCodeOptions(AgentOptions):
    object_type: Literal["OpenCodeOptions"] = "OpenCodeOptions"

    model: str | None = None
    cli_path: Path | None = None
    is_cached: bool = False


OPENCODE_TOOLS = (
    AgentToolName.READ,
    AgentToolName.WRITE,
    AgentToolName.EDIT,
    AgentToolName.MULTI_EDIT,
    AgentToolName.GLOB,
    AgentToolName.GREP,
    AgentToolName.LS,
    AgentToolName.BASH,
    AgentToolName.WEB_SEARCH,
    AgentToolName.WEB_FETCH,
    AgentToolName.TASK,
    AgentToolName.TODO_READ,
    AgentToolName.TODO_WRITE,
    AgentToolName.OTHER,
)
