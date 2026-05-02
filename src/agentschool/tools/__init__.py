"""Built-in tool registration."""

from agentschool.tools.ask_user_question_tool import AskUserQuestionTool
from agentschool.tools.agent_tool import AgentTool
from agentschool.tools.bash_tool import BashTool
from agentschool.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from agentschool.tools.brief_tool import BriefTool
from agentschool.tools.config_tool import ConfigTool
from agentschool.tools.cron_create_tool import CronCreateTool
from agentschool.tools.cron_delete_tool import CronDeleteTool
from agentschool.tools.cron_list_tool import CronListTool
from agentschool.tools.cron_toggle_tool import CronToggleTool
from agentschool.tools.enter_plan_mode_tool import EnterPlanModeTool
from agentschool.tools.enter_worktree_tool import EnterWorktreeTool
from agentschool.tools.exit_plan_mode_tool import ExitPlanModeTool
from agentschool.tools.exit_worktree_tool import ExitWorktreeTool
from agentschool.tools.file_edit_tool import FileEditTool
from agentschool.tools.file_read_tool import FileReadTool
from agentschool.tools.file_write_tool import FileWriteTool
from agentschool.tools.glob_tool import GlobTool
from agentschool.tools.grep_tool import GrepTool
from agentschool.tools.knowledge_tools import (
    ArxivSearchTool,
    CrossrefDoiLookupTool,
    CrossrefSearchTool,
    GutendexSearchTool,
    OpenAlexSearchTool,
    OpenLibrarySearchTool,
    PubMedSearchTool,
    ResearchSwarmTool,
    SemanticScholarSearchTool,
    WebReaderTool,
    WikidataSparqlTool,
    YouTubeTranscriptTool,
)
from agentschool.tools.list_mcp_resources_tool import ListMcpResourcesTool
from agentschool.tools.lsp_tool import LspTool
from agentschool.tools.mcp_auth_tool import McpAuthTool
from agentschool.tools.mcp_tool import McpToolAdapter
from agentschool.tools.notebook_edit_tool import NotebookEditTool
from agentschool.tools.read_mcp_resource_tool import ReadMcpResourceTool
from agentschool.tools.remote_trigger_tool import RemoteTriggerTool
from agentschool.tools.send_message_tool import SendMessageTool
from agentschool.tools.skill_tool import SkillTool
from agentschool.tools.sleep_tool import SleepTool
from agentschool.tools.task_create_tool import TaskCreateTool
from agentschool.tools.task_get_tool import TaskGetTool
from agentschool.tools.task_list_tool import TaskListTool
from agentschool.tools.task_output_tool import TaskOutputTool
from agentschool.tools.task_stop_tool import TaskStopTool
from agentschool.tools.task_update_tool import TaskUpdateTool
from agentschool.tools.team_create_tool import TeamCreateTool
from agentschool.tools.team_delete_tool import TeamDeleteTool
from agentschool.tools.todo_write_tool import TodoWriteTool
from agentschool.tools.tool_search_tool import ToolSearchTool
from agentschool.tools.web_fetch_tool import WebFetchTool
from agentschool.tools.web_search_tool import WebSearchTool


def create_default_tool_registry(mcp_manager=None) -> ToolRegistry:
    """Return the default built-in tool registry."""
    registry = ToolRegistry()
    for tool in (
        BashTool(),
        AskUserQuestionTool(),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        NotebookEditTool(),
        LspTool(),
        McpAuthTool(),
        GlobTool(),
        GrepTool(),
        SkillTool(),
        ToolSearchTool(),
        WebFetchTool(),
        WebSearchTool(),
        ArxivSearchTool(),
        SemanticScholarSearchTool(),
        OpenAlexSearchTool(),
        CrossrefSearchTool(),
        CrossrefDoiLookupTool(),
        PubMedSearchTool(),
        OpenLibrarySearchTool(),
        GutendexSearchTool(),
        WikidataSparqlTool(),
        WebReaderTool(),
        YouTubeTranscriptTool(),
        ResearchSwarmTool(),
        ConfigTool(),
        BriefTool(),
        SleepTool(),
        EnterWorktreeTool(),
        ExitWorktreeTool(),
        TodoWriteTool(),
        EnterPlanModeTool(),
        ExitPlanModeTool(),
        CronCreateTool(),
        CronListTool(),
        CronDeleteTool(),
        CronToggleTool(),
        RemoteTriggerTool(),
        TaskCreateTool(),
        TaskGetTool(),
        TaskListTool(),
        TaskStopTool(),
        TaskOutputTool(),
        TaskUpdateTool(),
        AgentTool(),
        SendMessageTool(),
        TeamCreateTool(),
        TeamDeleteTool(),
    ):
        registry.register(tool)
    if mcp_manager is not None:
        registry.register(ListMcpResourcesTool(mcp_manager))
        registry.register(ReadMcpResourceTool(mcp_manager))
        for tool_info in mcp_manager.list_tools():
            registry.register(McpToolAdapter(mcp_manager, tool_info))
    return registry


__all__ = [
    "BaseTool",
    "ToolExecutionContext",
    "ToolRegistry",
    "ToolResult",
    "create_default_tool_registry",
]
