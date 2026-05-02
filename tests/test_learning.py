from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentschool.engine.stream_events import ToolExecutionCompleted, ToolExecutionStarted
from agentschool.engine.stream_events import ErrorEvent
from agentschool.learning import (
    EXAMINER_ALLOWED_TOOLS,
    LEARNING_DEFAULT_MAX_TURNS,
    ExamAttemptRecord,
    LearningTurnOutcome,
    RequestExamTool,
    RequestExamToolInput,
    SubmitStudentAnswersInput,
    SubmitStudentAnswersTool,
    StudentTurnOutcome,
    build_examiner_prompt,
    build_grader_prompt,
    build_learning_retry_prompt,
    build_transient_error_followup,
    build_student_prompt,
    export_skill_bundle,
    load_instruction_file,
    load_instruction_only,
    prepare_learning_workspace,
    _apply_artifact_scope_guard,
    _restrict_runtime_tools,
    _run_learning_turn,
    run_headless_agent_session,
    run_learning_workflow,
    run_student_submission_workflow,
)
from agentschool.config.settings import PermissionSettings
from agentschool.tools.base import ToolExecutionContext, ToolRegistry
from agentschool.tools.file_read_tool import FileReadTool, FileReadToolInput
from agentschool.tools.glob_tool import GlobTool, GlobToolInput


def test_prepare_learning_workspace_creates_agentic_artifact_dirs(tmp_path: Path) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )

    assert workspace.topic_dir == tmp_path / "learn" / "topics" / "debug-playwright-flaky-tests" / "runs" / "run-001"
    assert workspace.skill_dir == workspace.topic_dir / "skill"
    assert workspace.skill_file == workspace.skill_dir / "SKILL.md"
    assert (workspace.skill_dir / "refs").exists()
    assert (workspace.skill_dir / "scripts").exists()
    assert workspace.research_notes.parent.exists()
    assert workspace.experiments_dir.exists()
    assert workspace.session_cwd == tmp_path.resolve()
    assert "debug Playwright flaky tests" in workspace.prompt
    assert str(workspace.skill_file) in workspace.prompt
    assert "tool-backed research" in workspace.prompt
    assert str(workspace.experiments_dir) in workspace.prompt
    assert "exam cheat sheet" in workspace.prompt
    assert "question IDs" in workspace.prompt
    assert "default to search/read/fetch/browser-style tools instead of bash" in workspace.prompt
    assert "Do not mine old `.agentschool` runs" in workspace.prompt
    assert "Do not install packages by default" in workspace.prompt
    assert "Prefer one focused bash experiment over many tiny probe commands" in workspace.prompt
    assert "deliberately try to falsify your current" in workspace.prompt
    assert "counterexample-focused end-to-end self-tests" in workspace.prompt
    assert "verified invariants or conclusions" in workspace.prompt
    assert "plausible but not fully verified heuristics or assumptions" in workspace.prompt
    assert "known failure surfaces, uncertainty, and alternative branches" in workspace.prompt
    assert "Must-Preserve Decisions" in workspace.prompt
    assert "Do-Not-Simplify" in workspace.prompt


def test_prepare_learning_workspace_uses_fresh_run_directory_each_time(tmp_path: Path) -> None:
    first = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    second = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )

    assert first.topic_dir.name == "run-001"
    assert second.topic_dir.name == "run-002"
    assert first.topic_dir != second.topic_dir


def test_retry_prompt_pushes_transferable_skill_not_exam_templates(tmp_path: Path) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    exam = ExamAttemptRecord(
        attempt=1,
        attempt_dir=workspace.exams_dir / "attempt-001",
        public_exam_path=workspace.exams_dir / "attempt-001" / "examiner" / "public_exam.json",
        answer_key_path=workspace.exams_dir / "attempt-001" / "examiner" / "answer_key.json",
        answers_path=workspace.exams_dir / "attempt-001" / "student" / "student_answers.json",
        grading_path=workspace.exams_dir / "attempt-001" / "grader" / "grading.json",
        score=12,
        max_score=30,
        passed=False,
        summary="Needs better failure analysis",
    )

    retry_prompt = build_learning_retry_prompt(workspace, exam)

    assert "reusable knowledge" in retry_prompt
    assert "canned answer template" in retry_prompt
    assert str(workspace.skill_file) in retry_prompt
    assert "Use bash primarily for real execution" in retry_prompt
    assert "generic environment probing or package installation" in retry_prompt
    assert "small number of focused experiments" in retry_prompt
    assert "counterexample-focused end-to-end retry test" in retry_prompt
    assert "separates verified conclusions from plausible heuristics" in retry_prompt
    assert "known breakpoints, and fallback branches" in retry_prompt
    assert "Must-Preserve Decisions" in retry_prompt
    assert "Do-Not-Simplify" in retry_prompt


def test_transient_error_followup_avoids_restarting_environment_probing(tmp_path: Path) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )

    prompt = build_transient_error_followup(
        workspace,
        ("Network error: Network connection lost.",),
    )

    assert "Do not respond to a transient provider/network failure by redoing generic environment checks" in prompt
    assert "Resume with the smallest focused next step" in prompt


def test_examiner_prompt_requires_varied_questions(tmp_path: Path) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )

    prompt = build_examiner_prompt(
        workspace=workspace,
        attempt=2,
        public_exam_path=workspace.exams_dir / "attempt-002" / "examiner" / "public_exam.json",
        answer_key_path=workspace.exams_dir / "attempt-002" / "examiner" / "answer_key.json",
        previous_questions=["List six failure modes in the pipeline."],
    )

    assert "meaningfully varied across attempts and within the current attempt" in prompt
    assert "different capability types" in prompt
    assert "different style of reasoning" in prompt
    assert "Do not repeat or trivially paraphrase" in prompt
    assert "Do not use bash" in prompt
    assert "implementation-fidelity question" in prompt
    assert '"must_preserve"' in prompt
    assert '"simplification_failures"' in prompt


def test_student_and_grader_prompts_lock_scope_and_forbid_bash(tmp_path: Path) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    student_prompt = build_student_prompt(
        workspace=workspace,
        public_exam_path=workspace.exams_dir / "attempt-001" / "student" / "public_exam.json",
        answers_path=workspace.exams_dir / "attempt-001" / "student" / "student_answers.json",
    )
    grader_prompt = build_grader_prompt(
        workspace=workspace,
        attempt=1,
        public_exam_path=workspace.exams_dir / "attempt-001" / "grader" / "public_exam.json",
        answer_key_path=workspace.exams_dir / "attempt-001" / "grader" / "answer_key.json",
        answers_path=workspace.exams_dir / "attempt-001" / "grader" / "student_answers.json",
        grading_path=workspace.exams_dir / "attempt-001" / "grader" / "grading.json",
    )

    assert "Do not use bash" in student_prompt
    assert "Preserve the exact question IDs" in student_prompt
    assert "submit_student_answers" in student_prompt
    assert "Do not try to write the answer JSON manually" in student_prompt
    assert "implementation-fidelity check" in student_prompt
    assert "Do not use bash" in grader_prompt
    assert "scope is fixed to the current grader workspace" in grader_prompt
    assert "Check implementation fidelity" in grader_prompt


def test_restrict_runtime_tools_removes_bash_and_applies_allowlist() -> None:
    bundle = SimpleNamespace(
        tool_registry=SimpleNamespace(
            _tools={
                "bash": object(),
                "read_file": object(),
                "write_file": object(),
                "glob": object(),
                "grep": object(),
            }
        ),
        engine=SimpleNamespace(permission_checker=SimpleNamespace(_settings=PermissionSettings())),
    )

    _restrict_runtime_tools(
        bundle,
        allowed_tools=EXAMINER_ALLOWED_TOOLS,
        denied_tools=frozenset({"bash"}),
    )

    assert set(bundle.tool_registry._tools) == set(EXAMINER_ALLOWED_TOOLS)
    assert "bash" not in bundle.tool_registry._tools
    assert "bash" in bundle.engine.permission_checker._settings.denied_tools


def test_learning_default_max_turns_is_1000() -> None:
    assert LEARNING_DEFAULT_MAX_TURNS == 1000


@pytest.mark.asyncio
async def test_submit_student_answers_tool_writes_validated_json(tmp_path: Path) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    public_exam_path = workspace.exams_dir / "attempt-001" / "student" / "public_exam.json"
    answers_path = workspace.exams_dir / "attempt-001" / "student" / "student_answers.json"
    public_exam_path.parent.mkdir(parents=True, exist_ok=True)
    public_exam_path.write_text(
        (
            '{\n'
            '  "goal": "debug Playwright flaky tests",\n'
            '  "questions": [\n'
            '    {"id": "Q1", "prompt": "Question one", "points": 10},\n'
            '    {"id": "Q2", "prompt": "Question two", "points": 20}\n'
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    tool = SubmitStudentAnswersTool()
    metadata = {
        "student_public_exam_path": str(public_exam_path),
        "student_answers_path": str(answers_path),
        "student_goal_name": workspace.goal_name,
    }
    result = await tool.execute(
        SubmitStudentAnswersInput(
            questions=[
                {
                    "id": "Q1",
                    "answer": "Use the reproducible debugging checklist.",
                    "skill_evidence": ["SKILL.md: failure analysis", "refs/api.md"],
                },
                {
                    "id": "Q2",
                    "answer": "Design a focused regression test and compare expected traces.",
                    "skill_evidence": ["SKILL.md: validation strategy"],
                },
            ]
        ),
        ToolExecutionContext(
            cwd=workspace.session_cwd,
            metadata=metadata,
        ),
    )

    payload = json.loads(answers_path.read_text(encoding="utf-8"))
    assert result.is_error is False
    assert "Submitted 2 answers" in result.output
    assert payload["goal"] == "debug Playwright flaky tests"
    assert [question["id"] for question in payload["questions"]] == ["Q1", "Q2"]
    assert result.metadata["student_submission_called"] is True
    assert result.metadata["student_submission_succeeded"] is True
    assert result.metadata["student_last_submission_path"] == str(answers_path)
    assert "student_submission_succeeded" not in metadata


@pytest.mark.asyncio
async def test_submit_student_answers_tool_rejects_question_id_mismatch(tmp_path: Path) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    public_exam_path = workspace.exams_dir / "attempt-001" / "student" / "public_exam.json"
    answers_path = workspace.exams_dir / "attempt-001" / "student" / "student_answers.json"
    public_exam_path.parent.mkdir(parents=True, exist_ok=True)
    public_exam_path.write_text(
        (
            '{\n'
            '  "goal": "debug Playwright flaky tests",\n'
            '  "questions": [\n'
            '    {"id": "Q1", "prompt": "Question one", "points": 10}\n'
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    tool = SubmitStudentAnswersTool()
    result = await tool.execute(
        SubmitStudentAnswersInput(
            questions=[
                {
                    "id": "Q9",
                    "answer": "Wrong question id.",
                    "skill_evidence": ["SKILL.md: something"],
                }
            ]
        ),
        ToolExecutionContext(
            cwd=workspace.session_cwd,
            metadata={
                "student_public_exam_path": str(public_exam_path),
                "student_answers_path": str(answers_path),
                "student_goal_name": workspace.goal_name,
            },
        ),
    )

    assert result.is_error is True
    assert "Expected ['Q1'], got ['Q9']" in result.output
    assert not answers_path.exists()


@pytest.mark.asyncio
async def test_artifact_scope_guard_blocks_old_run_reads_and_filters_repo_wide_globs(tmp_path: Path) -> None:
    blocked_root = tmp_path / "learn"
    old_skill = blocked_root / "topics" / "demo" / "runs" / "run-001" / "skill" / "SKILL.md"
    current_skill = blocked_root / "topics" / "demo" / "runs" / "run-002" / "skill" / "SKILL.md"
    old_skill.parent.mkdir(parents=True, exist_ok=True)
    current_skill.parent.mkdir(parents=True, exist_ok=True)
    old_skill.write_text("# old\n", encoding="utf-8")
    current_skill.write_text("# current\n", encoding="utf-8")

    bundle = SimpleNamespace(tool_registry=ToolRegistry())
    bundle.tool_registry.register(FileReadTool())
    bundle.tool_registry.register(GlobTool())
    _apply_artifact_scope_guard(
        bundle,
        blocked_root=blocked_root,
        allowed_roots=(current_skill.parent.parent,),
    )
    context = ToolExecutionContext(cwd=tmp_path, metadata={})

    read_result = await bundle.tool_registry.get("read_file").execute(
        FileReadToolInput(path=str(old_skill)),
        context,
    )
    glob_result = await bundle.tool_registry.get("glob").execute(
        GlobToolInput(pattern="**/SKILL.md", root=str(tmp_path)),
        context,
    )

    assert read_result.is_error is True
    assert "Artifact scope guard" in read_result.output
    assert "run-001" not in glob_result.output
    assert "run-002/skill/SKILL.md" in glob_result.output


@pytest.mark.asyncio
async def test_student_submission_workflow_retries_until_submit(tmp_path: Path, monkeypatch) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    student_dir = workspace.exams_dir / "attempt-001" / "student"
    public_exam_path = student_dir / "public_exam.json"
    answers_path = student_dir / "student_answers.json"
    statuses: list[str] = []
    prompts: list[str] = []

    class FakeRegistry:
        def __init__(self) -> None:
            self.tools = []

        def register(self, tool) -> None:
            self.tools.append(tool)

    bundle = SimpleNamespace(
        tool_registry=FakeRegistry(),
        engine=SimpleNamespace(model="test-model", max_turns=8, tool_metadata={}),
    )

    async def fake_build_runtime(**kwargs):
        assert kwargs["cwd"] == str(student_dir)
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    outcomes = iter(
        [
            StudentTurnOutcome(
                max_turns_exceeded=False,
                submitted=False,
                assistant_turns=1,
                tool_calls=1,
                text="Reviewed the copied skill bundle.",
            ),
            StudentTurnOutcome(
                max_turns_exceeded=False,
                submitted=True,
                assistant_turns=1,
                tool_calls=1,
                text="Submitted the final structured answers.",
            ),
        ]
    )

    async def fake_run_student_turn(*, bundle, prompt, status_printer):
        del bundle, status_printer
        prompts.append(prompt)
        return next(outcomes)

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.learning._run_student_turn", fake_run_student_turn)

    async def status(message: str) -> None:
        statuses.append(message)

    result = await run_student_submission_workflow(
        workspace=workspace,
        student_dir=student_dir,
        public_exam_path=public_exam_path,
        answers_path=answers_path,
        status_printer=status,
    )

    assert result.text == "Reviewed the copied skill bundle.\nSubmitted the final structured answers."
    assert len(bundle.tool_registry.tools) == 1
    assert bundle.tool_registry.tools[0].name == "submit_student_answers"
    assert "Only finish after `submit_student_answers` succeeds." in prompts[0]
    assert "attempted to stop without calling `submit_student_answers`" in prompts[1]
    assert any("Student session turn 1" in message for message in statuses)
    assert any("submitted=False" in message for message in statuses)
    assert any("submitted=True" in message for message in statuses)


@pytest.mark.asyncio
async def test_learning_workflow_retries_transient_network_error_in_same_session(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    statuses: list[str] = []
    prompts: list[str] = []

    perfect_exam = ExamAttemptRecord(
        attempt=1,
        attempt_dir=workspace.exams_dir / "attempt-001",
        public_exam_path=workspace.exams_dir / "attempt-001" / "examiner" / "public_exam.json",
        answer_key_path=workspace.exams_dir / "attempt-001" / "examiner" / "answer_key.json",
        answers_path=workspace.exams_dir / "attempt-001" / "student" / "student_answers.json",
        grading_path=workspace.exams_dir / "attempt-001" / "grader" / "grading.json",
        score=30,
        max_score=30,
        passed=True,
        summary="Perfect",
    )

    bundle = SimpleNamespace(
        tool_registry=SimpleNamespace(register=lambda tool: None),
        engine=SimpleNamespace(model="test-model", max_turns=8, tool_metadata={}),
    )

    async def fake_build_runtime(**kwargs):
        del kwargs
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    outcomes = iter(
        [
            LearningTurnOutcome(
                max_turns_exceeded=False,
                exam_requested=False,
                exam_passed=False,
                exam_record=None,
                assistant_turns=3,
                tool_calls=4,
                error_messages=("Network error: Network connection lost.",),
            ),
            LearningTurnOutcome(
                max_turns_exceeded=False,
                exam_requested=True,
                exam_passed=True,
                exam_record=perfect_exam,
                assistant_turns=1,
                tool_calls=1,
            ),
        ]
    )

    async def fake_run_learning_turn(*, bundle, prompt, workspace, loop_index, status_printer):
        del bundle, workspace, loop_index, status_printer
        prompts.append(prompt)
        return next(outcomes)

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.learning._run_learning_turn", fake_run_learning_turn)

    async def status(message: str) -> None:
        statuses.append(message)

    summary = await run_learning_workflow(workspace=workspace, status_printer=status)

    assert summary.passed is True
    assert "transient network/provider failure" in prompts[1]
    assert any("retrying within the same session" in message for message in statuses)


@pytest.mark.asyncio
async def test_learning_workflow_raises_after_repeated_transient_network_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    bundle = SimpleNamespace(
        tool_registry=SimpleNamespace(register=lambda tool: None),
        engine=SimpleNamespace(model="test-model", max_turns=8, tool_metadata={}),
    )

    async def fake_build_runtime(**kwargs):
        del kwargs
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    async def fake_run_learning_turn(*, bundle, prompt, workspace, loop_index, status_printer):
        del bundle, prompt, workspace, loop_index, status_printer
        return LearningTurnOutcome(
            max_turns_exceeded=False,
            exam_requested=False,
            exam_passed=False,
            exam_record=None,
            assistant_turns=2,
            tool_calls=2,
            error_messages=("Network error: Network connection lost.",),
        )

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.learning._run_learning_turn", fake_run_learning_turn)

    with pytest.raises(RuntimeError, match="repeated transient network/provider errors"):
        await run_learning_workflow(workspace=workspace)


@pytest.mark.asyncio
async def test_student_submission_workflow_raises_if_student_stops_twice_without_submit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    student_dir = workspace.exams_dir / "attempt-001" / "student"
    public_exam_path = student_dir / "public_exam.json"
    answers_path = student_dir / "student_answers.json"
    bundle = SimpleNamespace(
        tool_registry=SimpleNamespace(register=lambda tool: None),
        engine=SimpleNamespace(model="test-model", max_turns=8, tool_metadata={}),
    )

    async def fake_build_runtime(**kwargs):
        del kwargs
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    async def fake_run_student_turn(*, bundle, prompt, status_printer):
        del bundle, prompt, status_printer
        return StudentTurnOutcome(
            max_turns_exceeded=False,
            submitted=False,
            assistant_turns=1,
            tool_calls=0,
        )

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.learning._run_student_turn", fake_run_student_turn)

    with pytest.raises(RuntimeError, match="stopped again without calling submit_student_answers"):
        await run_student_submission_workflow(
            workspace=workspace,
            student_dir=student_dir,
            public_exam_path=public_exam_path,
            answers_path=answers_path,
        )


def test_instruction_only_learning_workspace_isolated_from_hidden_files(tmp_path: Path) -> None:
    task = tmp_path / "tasks" / "react-performance-debugging"
    (task / "solution").mkdir(parents=True)
    (task / "instruction.md").write_text("Fix a React rendering slowdown.\n", encoding="utf-8")
    (task / "solution" / "secret.txt").write_text("hidden solution", encoding="utf-8")

    name, instruction = load_instruction_only(task)
    workspace = prepare_learning_workspace(
        goal_name=name,
        root=tmp_path / "learn",
        session_cwd=tmp_path,
        visible_instruction=instruction,
    )

    assert workspace.instruction_workspace is not None
    assert workspace.session_cwd == workspace.instruction_workspace
    assert workspace.instruction_workspace == tmp_path / "learn" / "instruction-only" / "react-performance-debugging" / "runs" / "run-001"
    assert (workspace.instruction_workspace / "instruction.md").exists()
    assert not (workspace.instruction_workspace / "solution").exists()
    assert "Visible instruction only" in workspace.prompt
    assert "Fix a React rendering slowdown." in workspace.prompt
    assert "hidden solution" not in workspace.prompt


def test_load_instruction_file_uses_file_stem_as_goal_name(tmp_path: Path) -> None:
    instruction_file = tmp_path / "collect-benchmark-evidence.md"
    instruction_file.write_text("Collect real benchmark evidence.\n", encoding="utf-8")

    name, instruction = load_instruction_file(instruction_file)

    assert name == "collect-benchmark-evidence"
    assert instruction == "Collect real benchmark evidence.\n"


def test_export_skill_bundle_copies_skill_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "artifacts" / "skill"
    (skill_dir / "refs").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: learned\n---\n# Learned\n", encoding="utf-8")

    exported = export_skill_bundle(skill_dir, tmp_path / "generated", "react-performance-debugging")

    assert exported == tmp_path / "generated" / "react-performance-debugging"
    assert (exported / "SKILL.md").exists()
    assert (exported / "refs").exists()


@pytest.mark.asyncio
async def test_request_exam_tool_updates_learning_metadata(tmp_path: Path) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    record = ExamAttemptRecord(
        attempt=1,
        attempt_dir=workspace.exams_dir / "attempt-001",
        public_exam_path=workspace.exams_dir / "attempt-001" / "examiner" / "public_exam.json",
        answer_key_path=workspace.exams_dir / "attempt-001" / "examiner" / "answer_key.json",
        answers_path=workspace.exams_dir / "attempt-001" / "student" / "student_answers.json",
        grading_path=workspace.exams_dir / "attempt-001" / "grader" / "grading.json",
        score=20,
        max_score=30,
        passed=False,
        summary="Missing edge cases",
    )
    captured = {}

    async def fake_exam_runner(**kwargs):
        captured.update(kwargs)
        return record

    tool = RequestExamTool(exam_runner=fake_exam_runner)
    metadata = {"learning_workspace": workspace, "learning_exam_attempts": 0}

    result = await tool.execute(
        RequestExamToolInput(rationale="The skill now covers the main cases."),
        ToolExecutionContext(cwd=workspace.session_cwd, metadata=metadata),
    )

    assert result.is_error is False
    assert "Score: 20/30" in result.output
    assert result.metadata["learning_exam_called"] is True
    assert result.metadata["learning_exam_passed"] is False
    assert result.metadata["learning_last_exam_record"] == record
    assert result.metadata["learning_exam_attempts"] == 1
    assert "learning_exam_called" not in metadata
    assert captured["workspace"] == workspace
    assert captured["attempt"] == 1


@pytest.mark.asyncio
async def test_learning_workflow_keeps_same_session_until_exam_passes(tmp_path: Path, monkeypatch) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    statuses: list[str] = []
    prompts: list[str] = []

    first_exam = ExamAttemptRecord(
        attempt=1,
        attempt_dir=workspace.exams_dir / "attempt-001",
        public_exam_path=workspace.exams_dir / "attempt-001" / "examiner" / "public_exam.json",
        answer_key_path=workspace.exams_dir / "attempt-001" / "examiner" / "answer_key.json",
        answers_path=workspace.exams_dir / "attempt-001" / "student" / "student_answers.json",
        grading_path=workspace.exams_dir / "attempt-001" / "grader" / "grading.json",
        score=20,
        max_score=30,
        passed=False,
        summary="Missing edge cases",
    )
    second_exam = ExamAttemptRecord(
        attempt=2,
        attempt_dir=workspace.exams_dir / "attempt-002",
        public_exam_path=workspace.exams_dir / "attempt-002" / "examiner" / "public_exam.json",
        answer_key_path=workspace.exams_dir / "attempt-002" / "examiner" / "answer_key.json",
        answers_path=workspace.exams_dir / "attempt-002" / "student" / "student_answers.json",
        grading_path=workspace.exams_dir / "attempt-002" / "grader" / "grading.json",
        score=30,
        max_score=30,
        passed=True,
        summary="Perfect",
    )

    class FakeRegistry:
        def __init__(self) -> None:
            self.tools = []

        def register(self, tool) -> None:
            self.tools.append(tool)

    bundle = SimpleNamespace(
        tool_registry=FakeRegistry(),
        engine=SimpleNamespace(model="test-model", max_turns=8, tool_metadata={}),
    )

    async def fake_build_runtime(**kwargs):
        assert kwargs["prompt"] == workspace.prompt
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    outcomes = iter(
        [
            LearningTurnOutcome(
                max_turns_exceeded=False,
                exam_requested=False,
                exam_passed=False,
                exam_record=None,
                assistant_turns=1,
                tool_calls=1,
            ),
            LearningTurnOutcome(
                max_turns_exceeded=False,
                exam_requested=True,
                exam_passed=False,
                exam_record=first_exam,
                assistant_turns=1,
                tool_calls=1,
            ),
            LearningTurnOutcome(
                max_turns_exceeded=False,
                exam_requested=True,
                exam_passed=True,
                exam_record=second_exam,
                assistant_turns=1,
                tool_calls=1,
            ),
        ]
    )

    async def fake_run_learning_turn(*, bundle, prompt, workspace, loop_index, status_printer):
        del bundle, workspace, loop_index
        prompts.append(prompt)
        await status_printer("[learning] TOOL START read_file: path='skill/SKILL.md'")
        await status_printer("[learning] TOOL DONE read_file: loaded")
        return next(outcomes)

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.learning._run_learning_turn", fake_run_learning_turn)

    async def status(message: str) -> None:
        statuses.append(message)

    summary = await run_learning_workflow(workspace=workspace, status_printer=status)

    assert summary.passed is True
    assert summary.attempts == 2
    assert summary.final_score == 30
    assert len(bundle.tool_registry.tools) == 1
    assert bundle.tool_registry.tools[0].name == "request_exam"
    assert prompts[0] == workspace.prompt
    assert "without calling `request_exam`" in prompts[1]
    assert str(first_exam.grading_path) in prompts[2]
    assert any("returning to learning" in message for message in statuses)
    assert any("[learning] TOOL START read_file: path='skill/SKILL.md'" in message for message in statuses)
    assert any("[learning] TOOL DONE read_file: loaded" in message for message in statuses)


@pytest.mark.asyncio
async def test_learning_workflow_raises_when_turn_hits_max_turns(tmp_path: Path, monkeypatch) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    bundle = SimpleNamespace(
        tool_registry=SimpleNamespace(register=lambda tool: None),
        engine=SimpleNamespace(model="test-model", max_turns=8, tool_metadata={}),
    )

    async def fake_build_runtime(**kwargs):
        del kwargs
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    async def fake_run_learning_turn(*, bundle, prompt, workspace, loop_index, status_printer):
        del bundle, prompt, workspace, loop_index, status_printer
        return LearningTurnOutcome(
            max_turns_exceeded=True,
            exam_requested=False,
            exam_passed=False,
            exam_record=None,
            assistant_turns=1,
            tool_calls=2,
        )

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.learning._run_learning_turn", fake_run_learning_turn)

    with pytest.raises(RuntimeError, match="hit max_turns"):
        await run_learning_workflow(workspace=workspace)


@pytest.mark.asyncio
async def test_learning_workflow_raises_if_agent_stops_twice_without_exam(tmp_path: Path, monkeypatch) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    bundle = SimpleNamespace(
        tool_registry=SimpleNamespace(register=lambda tool: None),
        engine=SimpleNamespace(model="test-model", max_turns=8, tool_metadata={}),
    )

    async def fake_build_runtime(**kwargs):
        del kwargs
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    async def fake_run_learning_turn(*, bundle, prompt, workspace, loop_index, status_printer):
        del bundle, prompt, workspace, loop_index, status_printer
        return LearningTurnOutcome(
            max_turns_exceeded=False,
            exam_requested=False,
            exam_passed=False,
            exam_record=None,
            assistant_turns=1,
            tool_calls=0,
        )

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.learning._run_learning_turn", fake_run_learning_turn)

    with pytest.raises(RuntimeError, match="stopped again without calling request_exam"):
        await run_learning_workflow(workspace=workspace)


@pytest.mark.asyncio
async def test_learning_workflow_surfaces_error_events_before_request_exam(tmp_path: Path, monkeypatch) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )

    monkeypatch.setattr("agentschool.prompts.build_runtime_system_prompt", lambda *args, **kwargs: "test prompt")
    bundle = SimpleNamespace(
        tool_registry=SimpleNamespace(register=lambda tool: None),
        engine=SimpleNamespace(
            model="test-model",
            max_turns=8,
            tool_metadata={},
            total_usage=None,
            messages=[],
            set_max_turns=lambda turns: None,
            set_system_prompt=lambda prompt: None,
            submit_message=None,
        ),
        cwd=str(tmp_path),
        enforce_max_turns=True,
        extra_skill_dirs=(),
        extra_plugin_roots=(),
        include_project_memory=True,
        session_id="test-session",
        session_backend=SimpleNamespace(save_snapshot=lambda **kwargs: None),
        current_settings=lambda: SimpleNamespace(
            model="test-model",
            max_turns=8,
            system_prompt=None,
            permission=SimpleNamespace(mode=SimpleNamespace(value="full_auto")),
        ),
    )

    async def fake_submit_message(_prompt):
        yield ErrorEvent(message="API error: upstream 502")

    bundle.engine.submit_message = fake_submit_message

    async def status(_message: str) -> None:
        return None

    outcome = await _run_learning_turn(
        bundle=bundle,
        prompt=workspace.prompt,
        workspace=workspace,
        loop_index=0,
        status_printer=status,
    )

    assert outcome.error_messages == ("API error: upstream 502",)


@pytest.mark.asyncio
async def test_learning_workflow_raises_when_turn_has_no_output_or_tools(tmp_path: Path, monkeypatch) -> None:
    workspace = prepare_learning_workspace(
        goal_name="debug Playwright flaky tests",
        root=tmp_path / "learn",
        session_cwd=tmp_path,
    )
    bundle = SimpleNamespace(
        tool_registry=SimpleNamespace(register=lambda tool: None),
        engine=SimpleNamespace(model="test-model", max_turns=8, tool_metadata={}),
    )

    async def fake_build_runtime(**kwargs):
        del kwargs
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    async def fake_run_learning_turn(*, bundle, prompt, workspace, loop_index, status_printer):
        del bundle, prompt, workspace, loop_index, status_printer
        return LearningTurnOutcome(
            max_turns_exceeded=False,
            exam_requested=False,
            exam_passed=False,
            exam_record=None,
            assistant_turns=0,
            tool_calls=0,
            error_messages=(),
        )

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.learning._run_learning_turn", fake_run_learning_turn)

    with pytest.raises(RuntimeError, match="produced no assistant output and no tool activity"):
        await run_learning_workflow(workspace=workspace)


@pytest.mark.asyncio
async def test_run_headless_agent_session_reports_tool_calls_with_stage_prefix(tmp_path: Path, monkeypatch) -> None:
    statuses: list[str] = []
    bundle = SimpleNamespace()

    async def fake_build_runtime(**kwargs):
        del kwargs
        return bundle

    async def fake_start_runtime(_bundle):
        return None

    async def fake_close_runtime(_bundle):
        return None

    async def fake_handle_line(_bundle, prompt, print_system, render_event, clear_output):
        del _bundle, prompt, print_system, clear_output
        await render_event(ToolExecutionStarted(tool_name="bash", tool_input={"command": "pytest -q"}))
        await render_event(ToolExecutionCompleted(tool_name="bash", output="3 passed", is_error=False))

    monkeypatch.setattr("agentschool.ui.runtime.build_runtime", fake_build_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.start_runtime", fake_start_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.close_runtime", fake_close_runtime)
    monkeypatch.setattr("agentschool.ui.runtime.handle_line", fake_handle_line)

    async def status(message: str) -> None:
        statuses.append(message)

    result = await run_headless_agent_session(
        prompt="do exam work",
        cwd=tmp_path,
        status_printer=status,
        stage_label="examiner",
    )

    assert result.text == ""
    assert any("[examiner] TOOL START bash: command='pytest -q'" in message for message in statuses)
    assert any("[examiner] TOOL DONE bash: 3 passed" in message for message in statuses)
