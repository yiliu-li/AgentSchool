"""CLI smoke tests."""

import json
import re
import sys
import types
from pathlib import Path

from typer.testing import CliRunner

import agentschool.cli as cli
from agentschool.config import load_settings
from agentschool.config.settings import Settings
from agentschool.mcp.types import McpStdioServerConfig


app = cli.app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--help"],
        env={"NO_COLOR": "1", "COLUMNS": "160"},
    )
    plain_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert result.exit_code == 0
    assert "AgentSchool command-line runtime." in plain_output
    assert "setup" in plain_output
    assert "--dry-run" in plain_output


def test_setup_flow_selects_profile_and_model(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("AGENTSCHOOL_CONFIG_DIR", str(tmp_path))

    selected = []

    def fake_select(statuses, default_value=None):
        selected.append((tuple(statuses.keys()), default_value))
        return "codex"

    logged_in = []

    def fake_login(provider):
        logged_in.append(provider)

    monkeypatch.setattr("agentschool.cli._select_setup_workflow", fake_select)
    monkeypatch.setattr("agentschool.cli._prompt_model_for_profile", lambda profile: "gpt-5.4")
    monkeypatch.setattr("agentschool.cli._login_provider", fake_login)

    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "Setup complete:" in result.output
    assert logged_in == ["openai_codex"]

    settings = load_settings()
    assert settings.active_profile == "codex"
    assert settings.resolve_profile()[1].last_model == "gpt-5.4"


def test_select_from_menu_uses_questionary_when_tty(monkeypatch):
    answers = []

    class _Prompt:
        def ask(self):
            return "codex"

    fake_questionary = types.SimpleNamespace(
        Choice=lambda title, value, checked=False: {
            "title": title,
            "value": value,
            "checked": checked,
        },
        select=lambda title, choices, default=None: answers.append((title, choices, default)) or _Prompt(),
    )

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys, "__stdin__", sys.stdin)
    monkeypatch.setattr(cli.sys, "__stdout__", sys.stdout)
    monkeypatch.setitem(sys.modules, "questionary", fake_questionary)

    result = cli._select_from_menu(
        "Choose a provider workflow:",
        [("codex", "Codex"), ("claude-api", "Claude API")],
        default_value="codex",
    )

    assert result == "codex"
    assert answers


def test_setup_flow_creates_kimi_profile_with_profile_scoped_key(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("AGENTSCHOOL_CONFIG_DIR", str(tmp_path))

    selections = iter(["claude-api", "kimi-anthropic"])
    prompts = iter(
        [
            "https://api.moonshot.cn/anthropic",
            "kimi-k2.5",
        ]
    )

    monkeypatch.setattr("agentschool.cli._select_setup_workflow", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr("agentschool.cli._select_from_menu", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr("agentschool.cli._text_prompt", lambda *args, **kwargs: next(prompts))
    monkeypatch.setattr("agentschool.auth.flows.ApiKeyFlow.run", lambda self: "sk-kimi-test")

    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "Setup complete:" in result.output
    assert "- profile: kimi-anthropic" in result.output

    settings = load_settings()
    assert settings.active_profile == "kimi-anthropic"
    profile = settings.resolve_profile()[1]
    assert profile.base_url == "https://api.moonshot.cn/anthropic"
    assert profile.credential_slot == "kimi-anthropic"
    assert profile.allowed_models == ["kimi-k2.5"]

    from agentschool.auth.storage import load_credential

    assert load_credential("profile:kimi-anthropic", "api_key") == "sk-kimi-test"


def test_setup_flow_openrouter_offers_curated_model_choices(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setenv("AGENTSCHOOL_CONFIG_DIR", str(tmp_path))

    selections = iter(["openai-compatible", "openrouter", "anthropic/claude-opus-4.7"])
    monkeypatch.setattr("agentschool.cli._select_setup_workflow", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr("agentschool.cli._select_from_menu", lambda *args, **kwargs: next(selections))
    monkeypatch.setattr("agentschool.cli._text_prompt", lambda *args, **kwargs: "https://openrouter.ai/api/v1")
    monkeypatch.setattr("agentschool.auth.flows.ApiKeyFlow.run", lambda self: "sk-or-test")
    monkeypatch.setattr("agentschool.cli._prompt_model_for_profile", lambda profile: profile.default_model)

    result = runner.invoke(app, ["setup"])

    assert result.exit_code == 0
    settings = load_settings()
    profile = settings.resolve_profile()[1]
    assert settings.active_profile == "openrouter"
    assert profile.default_model == "anthropic/claude-opus-4.7"
    assert "openai/gpt-5.4" in profile.allowed_models
    assert "anthropic/claude-sonnet-4.6" in profile.allowed_models


def test_default_invocation_no_longer_starts_interactive_session():
    runner = CliRunner()
    result = runner.invoke(app, ["--dangerously-skip-permissions"])

    assert result.exit_code == 1
    assert "No default interactive session is available." in result.output


def test_task_worker_flag_routes_to_run_task_worker(monkeypatch):
    runner = CliRunner()
    captured = {}

    async def fake_run_task_worker(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("agentschool.ui.app.run_task_worker", fake_run_task_worker)

    result = runner.invoke(app, ["--task-worker", "--model", "kimi-k2.5"])

    assert result.exit_code == 0
    assert captured["model"] == "kimi-k2.5"


def test_learn_cli_runs_exam_gated_learning_workflow(tmp_path: Path, monkeypatch):
    runner = CliRunner()
    captured = {}

    async def fake_run_learning_workflow(**kwargs):
        captured.update(kwargs)
        workspace = kwargs["workspace"]
        workspace.skill_file.parent.mkdir(parents=True, exist_ok=True)
        workspace.skill_file.write_text("---\nname: learned\n---\n# Learned\n", encoding="utf-8")
        from agentschool.learning import LearningRunSummary

        return LearningRunSummary(
            workspace=workspace,
            attempts=2,
            final_score=30,
            max_score=30,
            passed=True,
        )

    monkeypatch.setattr("agentschool.learning.run_learning_workflow", fake_run_learning_workflow)

    result = runner.invoke(
        app,
        [
            "learn",
            "debug Playwright flaky tests",
            "--root",
            str(tmp_path / "learn"),
            "--cwd",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    workspace = captured["workspace"]
    assert workspace.session_cwd == tmp_path.resolve()
    assert workspace.goal_name == "debug Playwright flaky tests"
    assert workspace.skill_file == tmp_path / "learn" / "topics" / "debug-playwright-flaky-tests" / "skill" / "SKILL.md"
    assert captured["max_turns"] is None
    assert "passed after 2 exam attempt" in result.output.lower()


def test_learn_cli_task_exports_skill_bundle(tmp_path: Path, monkeypatch):
    task = tmp_path / "tasks" / "react-performance-debugging"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Fix a React rendering slowdown.\n", encoding="utf-8")
    (task / "solution").mkdir()
    (task / "solution" / "secret.txt").write_text("hidden solution", encoding="utf-8")

    async def fake_run_learning_workflow(**kwargs):
        workspace = kwargs["workspace"]
        (workspace.skill_dir / "refs").mkdir(parents=True, exist_ok=True)
        workspace.skill_file.write_text("---\nname: learned\n---\n# Learned\n", encoding="utf-8")
        from agentschool.learning import LearningRunSummary

        return LearningRunSummary(
            workspace=workspace,
            attempts=1,
            final_score=30,
            max_score=30,
            passed=True,
        )

    monkeypatch.setattr("agentschool.learning.run_learning_workflow", fake_run_learning_workflow)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "learn",
            "--task",
            str(task),
            "--root",
            str(tmp_path / "learn"),
            "--cwd",
            str(tmp_path),
            "--export-skills-dir",
            str(tmp_path / "generated"),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "learn" / "instruction-only" / "react-performance-debugging" / "instruction.md").exists()
    assert not (tmp_path / "learn" / "instruction-only" / "react-performance-debugging" / "solution").exists()
    assert (tmp_path / "generated" / "react-performance-debugging" / "SKILL.md").exists()


def test_dry_run_uses_preview_builder(monkeypatch):
    runner = CliRunner()
    captured = {}

    def fake_build_dry_run_preview(**kwargs):
        captured.update(kwargs)
        return {
            "cwd": kwargs["cwd"],
            "prompt_preview": kwargs["prompt"],
            "settings": {
                "active_profile": "claude-api",
                "profile_label": "Claude API",
                "provider": "anthropic",
                "api_format": "anthropic",
                "model": "claude-sonnet-4-6",
                "base_url": "",
                "permission_mode": "default",
                "max_turns": 200,
                "effort": "medium",
                "passes": 1,
            },
            "validation": {
                "auth_status": "configured",
                "api_client": {"status": "ok"},
                "system_prompt_chars": 123,
                "mcp_validation": "skipped",
            },
            "entrypoint": {"kind": "model_prompt", "detail": "preview only"},
            "plugins": [],
            "skills": [],
            "commands": [],
            "tools": [],
            "mcp_servers": [],
            "system_prompt_preview": "preview",
        }

    monkeypatch.setattr("agentschool.cli._build_dry_run_preview", fake_build_dry_run_preview)

    result = runner.invoke(app, ["--dry-run", "--print", "ship it", "--model", "gpt-5.4"])

    assert result.exit_code == 0
    assert captured["prompt"] == "ship it"
    assert captured["model"] == "gpt-5.4"
    assert "AgentSchool Dry Run" in result.output
    assert "ship it" in result.output


def test_dry_run_json_output(monkeypatch):
    runner = CliRunner()

    def fake_build_dry_run_preview(**kwargs):
        return {
            "mode": "dry-run",
            "cwd": kwargs["cwd"],
            "prompt": kwargs["prompt"],
            "prompt_preview": kwargs["prompt"],
            "settings": {
                "active_profile": "claude-api",
                "profile_label": "Claude API",
                "provider": "anthropic",
                "api_format": "anthropic",
                "model": "claude-sonnet-4-6",
                "base_url": "",
                "permission_mode": "default",
                "max_turns": 200,
                "effort": "medium",
                "passes": 1,
            },
            "validation": {
                "auth_status": "configured",
                "api_client": {"status": "ok"},
                "system_prompt_chars": 123,
                "mcp_validation": "skipped",
            },
            "entrypoint": {"kind": "interactive_session", "detail": "wait"},
            "plugins": [],
            "skills": [],
            "commands": [],
            "tools": [],
            "mcp_servers": [],
            "system_prompt_preview": "preview",
        }

    monkeypatch.setattr("agentschool.cli._build_dry_run_preview", fake_build_dry_run_preview)

    result = runner.invoke(app, ["--dry-run", "--output-format", "json", "--print", "preview this"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["mode"] == "dry-run"
    assert payload["prompt"] == "preview this"


def test_build_dry_run_preview_classifies_slash_command_and_flags_bad_mcp(monkeypatch, tmp_path: Path):
    settings = Settings(
        api_key="sk-test",
        mcp_servers={
            "broken": McpStdioServerConfig(command="definitely-not-a-real-command-agentschool"),
        },
    )

    class _FakeSkillRegistry:
        def list_skills(self):
            return []

    monkeypatch.setattr("agentschool.config.load_settings", lambda: settings)
    monkeypatch.setattr(
        "agentschool.api.provider.detect_provider",
        lambda settings: types.SimpleNamespace(name="anthropic"),
    )
    monkeypatch.setattr("agentschool.api.provider.auth_status", lambda settings: "configured")
    monkeypatch.setattr("agentschool.plugins.load_plugins", lambda settings, cwd: [])
    monkeypatch.setattr("agentschool.skills.load_skill_registry", lambda cwd, settings=None: _FakeSkillRegistry())
    monkeypatch.setattr("agentschool.prompts.context.build_runtime_system_prompt", lambda *args, **kwargs: "preview prompt")
    monkeypatch.setattr("agentschool.ui.runtime._resolve_api_client_from_settings", lambda settings: object())

    preview = cli._build_dry_run_preview(
        prompt="/plugin list",
        cwd=str(tmp_path),
        model=None,
        max_turns=None,
        base_url=None,
        system_prompt=None,
        append_system_prompt=None,
        api_key=None,
        api_format=None,
        permission_mode=None,
    )

    assert preview["entrypoint"]["kind"] == "slash_command"
    assert preview["entrypoint"]["command"] == "plugin"
    assert preview["entrypoint"]["remote_invocable"] is False
    assert preview["entrypoint"]["remote_admin_opt_in"] is True
    assert preview["entrypoint"]["behavior"] == "stateful"
    assert preview["validation"]["mcp_errors"] == 1
    assert preview["mcp_servers"][0]["status"] == "error"
    assert "command not found in PATH" in preview["mcp_servers"][0]["issues"][0]


def test_build_dry_run_preview_sets_blocked_when_model_prompt_lacks_auth(monkeypatch, tmp_path: Path):
    settings = Settings(api_key="")

    class _FakeSkillRegistry:
        def list_skills(self):
            return []

    monkeypatch.setattr("agentschool.config.load_settings", lambda: settings)
    monkeypatch.setattr(
        "agentschool.api.provider.detect_provider",
        lambda settings: types.SimpleNamespace(name="anthropic"),
    )
    monkeypatch.setattr("agentschool.api.provider.auth_status", lambda settings: "missing")
    monkeypatch.setattr("agentschool.plugins.load_plugins", lambda settings, cwd: [])
    monkeypatch.setattr("agentschool.skills.load_skill_registry", lambda cwd, settings=None: _FakeSkillRegistry())
    monkeypatch.setattr("agentschool.prompts.context.build_runtime_system_prompt", lambda *args, **kwargs: "preview prompt")

    def fake_resolve_api_client(settings):
        raise SystemExit(1)

    monkeypatch.setattr("agentschool.ui.runtime._resolve_api_client_from_settings", fake_resolve_api_client)

    preview = cli._build_dry_run_preview(
        prompt="fix the failing tests",
        cwd=str(tmp_path),
        model=None,
        max_turns=None,
        base_url=None,
        system_prompt=None,
        append_system_prompt=None,
        api_key=None,
        api_format=None,
        permission_mode=None,
    )

    assert preview["entrypoint"]["kind"] == "model_prompt"
    assert preview["readiness"]["level"] == "blocked"
    assert any("runtime client" in reason.lower() for reason in preview["readiness"]["reasons"])
    assert any("authentication" in action.lower() or "profile" in action.lower() for action in preview["readiness"]["next_actions"])


def test_build_dry_run_preview_recommends_matching_skills_and_tools(monkeypatch, tmp_path: Path):
    settings = Settings(api_key="sk-test")

    class _FakeSkillRegistry:
        def list_skills(self):
            return [
                types.SimpleNamespace(
                    name="review",
                    description="Review code for bugs and regressions.",
                    content="Use this when reviewing bug fixes and regressions.",
                    source="bundled",
                ),
                types.SimpleNamespace(
                    name="plan",
                    description="Plan implementation work before coding.",
                    content="Use this to design an implementation plan.",
                    source="bundled",
                ),
            ]

    class _FakeToolRegistry:
        def to_api_schema(self):
            return [
                {
                    "name": "grep",
                    "description": "Search code for bug patterns and failing lines.",
                    "input_schema": {"properties": {"pattern": {}, "root": {}}, "required": ["pattern"]},
                },
                {
                    "name": "read_file",
                    "description": "Read files from disk.",
                    "input_schema": {"properties": {"path": {}, "offset": {}}, "required": ["path"]},
                },
            ]

    monkeypatch.setattr("agentschool.config.load_settings", lambda: settings)
    monkeypatch.setattr(
        "agentschool.api.provider.detect_provider",
        lambda settings: types.SimpleNamespace(name="anthropic"),
    )
    monkeypatch.setattr("agentschool.api.provider.auth_status", lambda settings: "configured")
    monkeypatch.setattr("agentschool.plugins.load_plugins", lambda settings, cwd: [])
    monkeypatch.setattr("agentschool.skills.load_skill_registry", lambda cwd, settings=None: _FakeSkillRegistry())
    monkeypatch.setattr("agentschool.tools.create_default_tool_registry", lambda: _FakeToolRegistry())
    monkeypatch.setattr("agentschool.prompts.context.build_runtime_system_prompt", lambda *args, **kwargs: "preview prompt")
    monkeypatch.setattr("agentschool.ui.runtime._resolve_api_client_from_settings", lambda settings: object())

    preview = cli._build_dry_run_preview(
        prompt="review this bug fix and grep for failing tests",
        cwd=str(tmp_path),
        model=None,
        max_turns=None,
        base_url=None,
        system_prompt=None,
        append_system_prompt=None,
        api_key=None,
        api_format=None,
        permission_mode=None,
    )

    recommended_skills = [entry["name"] for entry in preview["recommendations"]["skills"]]
    recommended_tools = [entry["name"] for entry in preview["recommendations"]["tools"]]

    assert preview["readiness"]["level"] == "ready"
    assert any("you can run this prompt directly" in action.lower() for action in preview["readiness"]["next_actions"])
    assert "review" in recommended_skills
    assert "grep" in recommended_tools
