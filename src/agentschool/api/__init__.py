"""API exports."""

from agentschool.api.client import AnthropicApiClient
from agentschool.api.codex_client import CodexApiClient
from agentschool.api.copilot_client import CopilotClient
from agentschool.api.errors import AgentSchoolApiError
from agentschool.api.openai_client import OpenAICompatibleClient
from agentschool.api.provider import ProviderInfo, auth_status, detect_provider
from agentschool.api.usage import UsageSnapshot

__all__ = [
    "AnthropicApiClient",
    "CodexApiClient",
    "CopilotClient",
    "OpenAICompatibleClient",
    "AgentSchoolApiError",
    "ProviderInfo",
    "UsageSnapshot",
    "auth_status",
    "detect_provider",
]
