"""Agent-native helpers for fully agentic skill learning and exam gating."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import re
import shutil

from pydantic import BaseModel, Field

from agentschool.engine.query import MaxTurnsExceeded
from agentschool.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ErrorEvent,
    StatusEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from agentschool.tools.base import BaseTool, ToolExecutionContext, ToolResult


StatusPrinter = Callable[[str], Awaitable[None]]
LEARNING_DEFAULT_MAX_TURNS = 1000
EXAMINER_ALLOWED_TOOLS = frozenset({"read_file", "write_file", "glob", "grep"})
STUDENT_ALLOWED_TOOLS = frozenset({"read_file", "glob", "grep", "submit_student_answers"})
GRADER_ALLOWED_TOOLS = frozenset({"read_file", "write_file"})
ARTIFACT_SCOPED_TOOL_NAMES = frozenset(
    {"read_file", "write_file", "edit_file", "notebook_edit", "glob", "grep"}
)


@dataclass(frozen=True)
class LearningWorkspace:
    """Filesystem locations for one learning run."""

    goal_name: str
    artifact_root: Path
    topic_dir: Path
    skill_dir: Path
    skill_file: Path
    research_notes: Path
    experiments_dir: Path
    exams_dir: Path
    session_cwd: Path
    prompt: str
    visible_instruction: str | None = None
    instruction_workspace: Path | None = None


@dataclass(frozen=True)
class AgentRunResult:
    """Structured result from a headless agent session."""

    text: str


@dataclass(frozen=True)
class ExamAttemptRecord:
    """Artifacts and score for one exam attempt."""

    attempt: int
    attempt_dir: Path
    public_exam_path: Path
    answer_key_path: Path
    answers_path: Path
    grading_path: Path
    score: int
    max_score: int
    passed: bool
    summary: str


@dataclass(frozen=True)
class LearningRunSummary:
    """Final result for one learn invocation."""

    workspace: LearningWorkspace
    attempts: int
    final_score: int
    max_score: int
    passed: bool


@dataclass(frozen=True)
class LearningTurnOutcome:
    """Structured result from one persistent learning turn."""

    max_turns_exceeded: bool
    exam_requested: bool
    exam_passed: bool
    exam_record: ExamAttemptRecord | None
    assistant_turns: int
    tool_calls: int
    error_messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class StudentTurnOutcome:
    """Structured result from one persistent student turn."""

    max_turns_exceeded: bool
    submitted: bool
    assistant_turns: int
    tool_calls: int
    error_messages: tuple[str, ...] = ()
    text: str = ""


class ArtifactScopeGuardedTool(BaseTool):
    """Wrap filesystem tools so old learning artifacts stay out of scope."""

    def __init__(
        self,
        delegate: BaseTool,
        *,
        blocked_root: Path,
        allowed_roots: tuple[Path, ...],
    ) -> None:
        self._delegate = delegate
        self._blocked_root = blocked_root.resolve()
        self._allowed_roots = tuple(path.resolve() for path in allowed_roots)
        self.name = delegate.name
        self.description = delegate.description
        self.input_model = delegate.input_model

    def is_read_only(self, arguments) -> bool:
        return self._delegate.is_read_only(arguments)

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        if self.name in {"read_file", "write_file", "edit_file", "notebook_edit"}:
            candidate = _resolve_tool_path(context.cwd, getattr(arguments, "path"))
            violation = _artifact_scope_violation(
                candidate,
                blocked_root=self._blocked_root,
                allowed_roots=self._allowed_roots,
            )
            if violation is not None:
                return ToolResult(output=violation, is_error=True)
            return await self._delegate.execute(arguments, context)

        if self.name in {"glob", "grep"}:
            root_value = getattr(arguments, "root", None)
            search_root = _resolve_tool_path(context.cwd, root_value) if root_value else context.cwd.resolve()
            if _search_root_fully_blocked(
                search_root,
                blocked_root=self._blocked_root,
                allowed_roots=self._allowed_roots,
            ):
                return ToolResult(output="(no matches)")
            result = await self._delegate.execute(arguments, context)
            return _filter_scoped_search_result(
                result,
                tool_name=self.name,
                search_root=search_root,
                blocked_root=self._blocked_root,
                allowed_roots=self._allowed_roots,
            )

        return await self._delegate.execute(arguments, context)


class RequestExamToolInput(BaseModel):
    """Arguments for the learning-only exam request tool."""

    rationale: str = Field(
        default="",
        description="Why the learning agent believes the current skill bundle is ready for examination.",
    )


class RequestExamTool(BaseTool):
    """Learning-only tool that triggers the mandatory unseen exam gate."""

    name = "request_exam"
    description = (
        "Trigger the mandatory graduation exam. Use this only when the current skill bundle "
        "is ready for evaluation; the learning session may only finish after this tool returns a perfect pass."
    )
    input_model = RequestExamToolInput

    def __init__(
        self,
        *,
        exam_runner: Callable[..., Awaitable[ExamAttemptRecord]] | None = None,
    ) -> None:
        self._exam_runner = exam_runner or run_exam_attempt

    async def execute(
        self,
        arguments: RequestExamToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        metadata = context.metadata
        workspace = metadata.get("learning_workspace")
        if not isinstance(workspace, LearningWorkspace):
            return ToolResult(output="Learning workspace is not configured for request_exam.", is_error=True)

        attempt = int(metadata.get("learning_exam_attempts", 0)) + 1
        metadata["learning_exam_attempts"] = attempt
        record = await self._exam_runner(
            workspace=workspace,
            attempt=attempt,
            model=metadata.get("learning_model"),
            max_turns=metadata.get("learning_max_turns"),
            status_printer=metadata.get("learning_status_printer"),
            agent_runner=metadata.get("learning_agent_runner", run_headless_agent_session),
        )
        return ToolResult(
            output=_format_exam_tool_output(record),
            metadata={
                "learning_exam_attempts": attempt,
                "learning_exam_called": True,
                "learning_exam_passed": record.passed,
                "learning_last_exam_record": record,
                "learning_exam_rationale": arguments.rationale,
                "attempt": record.attempt,
                "score": record.score,
                "max_score": record.max_score,
                "passed": record.passed,
            },
        )


class StudentAnswerEntry(BaseModel):
    """One structured answer submitted by the student subagent."""

    id: str = Field(description="Question ID copied exactly from the public exam.")
    answer: str = Field(description="Student answer grounded in the skill bundle.")
    skill_evidence: list[str] = Field(
        default_factory=list,
        description="Specific skill sections, steps, or ideas that support the answer.",
    )


class SubmitStudentAnswersInput(BaseModel):
    """Arguments for the student-only exam submission tool."""

    questions: list[StudentAnswerEntry] = Field(
        min_length=1,
        description="All exam answers, preserving the exact question IDs and order from the public exam.",
    )


class SubmitStudentAnswersTool(BaseTool):
    """Student-only tool that validates and writes the structured answer file."""

    name = "submit_student_answers"
    description = (
        "Validate the student's answers against the public exam question IDs and write the official "
        "student_answers.json submission file."
    )
    input_model = SubmitStudentAnswersInput

    async def execute(
        self,
        arguments: SubmitStudentAnswersInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        public_exam_value = context.metadata.get("student_public_exam_path")
        answers_value = context.metadata.get("student_answers_path")
        if not public_exam_value or not answers_value:
            return ToolResult(
                output="Student exam paths are not configured for submit_student_answers.",
                is_error=True,
            )

        public_exam_path = Path(str(public_exam_value)).expanduser().resolve()
        answers_path = Path(str(answers_value)).expanduser().resolve()
        exam_payload = _read_json(public_exam_path)
        expected_ids = [
            str(question.get("id", "")).strip()
            for question in exam_payload.get("questions", [])
            if str(question.get("id", "")).strip()
        ]
        submitted = arguments.questions
        submitted_ids = [entry.id.strip() for entry in submitted]

        if not expected_ids:
            return ToolResult(output="Public exam has no question IDs to answer.", is_error=True)
        if submitted_ids != expected_ids:
            return ToolResult(
                output=(
                    "Submitted question IDs do not match the public exam. "
                    f"Expected {expected_ids}, got {submitted_ids}."
                ),
                is_error=True,
            )

        normalized_questions: list[dict[str, object]] = []
        for entry in submitted:
            answer = entry.answer.strip()
            evidence = [item.strip() for item in entry.skill_evidence if item.strip()]
            if not answer:
                return ToolResult(output=f"Answer for {entry.id.strip()!r} is empty.", is_error=True)
            if not evidence:
                return ToolResult(
                    output=f"skill_evidence for {entry.id.strip()!r} must include at least one item.",
                    is_error=True,
                )
            normalized_questions.append(
                {
                    "id": entry.id.strip(),
                    "answer": answer,
                    "skill_evidence": evidence,
                }
            )

        output_payload = {
            "goal": str(exam_payload.get("goal") or context.metadata.get("student_goal_name") or "").strip(),
            "questions": normalized_questions,
        }
        _write_text(answers_path, json.dumps(output_payload, indent=2, ensure_ascii=False))
        return ToolResult(
            output=f"Submitted {len(normalized_questions)} answers to {answers_path}",
            metadata={
                "student_submission_called": True,
                "student_submission_succeeded": True,
                "student_last_submission_path": str(answers_path),
            },
        )


def slugify(value: str) -> str:
    """Normalize a topic or task name into a stable directory slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "learning-run"


def load_instruction_only(task_path: str | Path) -> tuple[str, str]:
    """Return task name and visible instruction content only."""
    task_dir = Path(task_path).expanduser().resolve()
    instruction_path = task_dir / "instruction.md"
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    if not instruction_path.exists():
        raise FileNotFoundError(f"instruction.md not found: {instruction_path}")
    return task_dir.name, instruction_path.read_text(encoding="utf-8")


def load_instruction_file(instruction_file: str | Path) -> tuple[str, str]:
    """Return a goal name and visible instruction content from a standalone file."""
    instruction_path = Path(instruction_file).expanduser().resolve()
    if not instruction_path.exists():
        raise FileNotFoundError(f"Instruction file not found: {instruction_path}")
    if not instruction_path.is_file():
        raise FileNotFoundError(f"Instruction path is not a file: {instruction_path}")
    return instruction_path.stem, instruction_path.read_text(encoding="utf-8")


def _next_run_directory(parent: Path) -> Path:
    """Create and return the next numbered run directory under a parent path."""
    runs_dir = parent / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing_numbers: list[int] = []
    for candidate in runs_dir.iterdir():
        if not candidate.is_dir():
            continue
        match = re.fullmatch(r"run-(\d{3})", candidate.name)
        if match:
            existing_numbers.append(int(match.group(1)))
    next_number = (max(existing_numbers) if existing_numbers else 0) + 1
    run_dir = runs_dir / f"run-{next_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_learning_workspace(
    *,
    goal_name: str,
    root: str | Path,
    session_cwd: str | Path,
    visible_instruction: str | None = None,
) -> LearningWorkspace:
    """Create the artifact workspace and return the learning prompt."""
    root_path = Path(root).expanduser().resolve()
    topic_dir = _next_run_directory(root_path / "topics" / slugify(goal_name))
    skill_dir = topic_dir / "skill"
    research_dir = topic_dir / "research"
    experiments_dir = topic_dir / "experiments"
    exams_dir = topic_dir / "exams"
    instruction_workspace: Path | None = None

    for directory in (
        skill_dir / "refs",
        skill_dir / "scripts",
        research_dir,
        experiments_dir,
        exams_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    resolved_cwd = Path(session_cwd).expanduser().resolve()
    if visible_instruction is not None:
        instruction_workspace = _next_run_directory(root_path / "instruction-only" / slugify(goal_name))
        (instruction_workspace / "instruction.md").write_text(
            visible_instruction.rstrip() + "\n",
            encoding="utf-8",
        )
        resolved_cwd = instruction_workspace

    skill_file = skill_dir / "SKILL.md"
    research_notes = research_dir / "notes.md"
    prompt = build_learning_prompt(
        goal_name=goal_name,
        topic_dir=topic_dir,
        skill_dir=skill_dir,
        skill_file=skill_file,
        research_notes=research_notes,
        experiments_dir=experiments_dir,
        visible_instruction=visible_instruction,
    )
    return LearningWorkspace(
        goal_name=goal_name,
        artifact_root=root_path,
        topic_dir=topic_dir,
        skill_dir=skill_dir,
        skill_file=skill_file,
        research_notes=research_notes,
        experiments_dir=experiments_dir,
        exams_dir=exams_dir,
        session_cwd=resolved_cwd,
        prompt=prompt,
        visible_instruction=visible_instruction,
        instruction_workspace=instruction_workspace,
    )


def export_skill_bundle(skill_dir: str | Path, export_dir: str | Path, skill_name: str) -> Path:
    """Export a learned skill directory to a target directory."""
    source = Path(skill_dir).expanduser().resolve()
    destination = Path(export_dir).expanduser().resolve() / slugify(skill_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


async def run_headless_agent_session(
    *,
    prompt: str,
    cwd: str | Path,
    model: str | None = None,
    max_turns: int | None = None,
    status_printer: StatusPrinter | None = None,
    stage_label: str = "agent",
    allowed_tools: frozenset[str] | None = None,
    denied_tools: frozenset[str] | None = None,
    extra_tools: tuple[BaseTool, ...] | None = None,
    extra_metadata: dict[str, object] | None = None,
    artifact_guard_root: str | Path | None = None,
    allowed_artifact_roots: tuple[str | Path, ...] | None = None,
) -> AgentRunResult:
    """Run one isolated headless agent session and collect the final assistant text."""
    from agentschool.ui.runtime import build_runtime, close_runtime, handle_line, start_runtime

    async def _noop_permission(_tool_name: str, _reason: str) -> bool:
        return True

    async def _noop_ask(_question: str) -> str:
        return ""

    bundle = await build_runtime(
        prompt=prompt,
        cwd=str(Path(cwd).expanduser().resolve()),
        model=model,
        max_turns=max_turns,
        permission_mode="full_auto",
        permission_prompt=_noop_permission,
        ask_user_prompt=_noop_ask,
    )
    if extra_tools:
        for tool in extra_tools:
            bundle.tool_registry.register(tool)
    if extra_metadata:
        bundle.engine.tool_metadata.update(extra_metadata)
    _restrict_runtime_tools(bundle, allowed_tools=allowed_tools, denied_tools=denied_tools)
    _apply_artifact_scope_guard(
        bundle,
        blocked_root=artifact_guard_root,
        allowed_roots=allowed_artifact_roots,
    )
    await start_runtime(bundle)

    collected_text = ""

    async def _print_system(_message: str) -> None:
        return None

    async def _render_event(event) -> None:
        nonlocal collected_text
        if isinstance(event, AssistantTextDelta):
            collected_text += event.text
        elif isinstance(event, ToolExecutionStarted):
            await _print_status(
                status_printer,
                _format_tool_started(stage_label, event.tool_name, event.tool_input),
            )
        elif isinstance(event, ToolExecutionCompleted):
            await _print_status(
                status_printer,
                _format_tool_completed(stage_label, event.tool_name, event.output, is_error=event.is_error),
            )
        elif isinstance(event, StatusEvent):
            await _print_status(status_printer, f"[{stage_label}] {event.message}")

    async def _clear_output() -> None:
        return None

    try:
        await handle_line(
            bundle,
            prompt,
            print_system=_print_system,
            render_event=_render_event,
            clear_output=_clear_output,
        )
    finally:
        await close_runtime(bundle)
    return AgentRunResult(text=collected_text.strip())


def _restrict_runtime_tools(
    bundle,
    *,
    allowed_tools: frozenset[str] | None = None,
    denied_tools: frozenset[str] | None = None,
) -> None:
    """Narrow a runtime to a fixed tool scope for subagents."""
    tool_registry = getattr(bundle, "tool_registry", None)
    tool_map = getattr(tool_registry, "_tools", None)
    if isinstance(tool_map, dict):
        if allowed_tools is not None:
            tool_registry._tools = {name: tool for name, tool in tool_map.items() if name in allowed_tools}
            tool_map = tool_registry._tools
        if denied_tools:
            for name in denied_tools:
                tool_map.pop(name, None)

    checker = getattr(getattr(bundle, "engine", None), "permission_checker", None)
    settings = getattr(checker, "_settings", None)
    if settings is not None:
        next_allowed = list(getattr(settings, "allowed_tools", []))
        next_denied = list(getattr(settings, "denied_tools", []))
        if allowed_tools is not None:
            next_allowed = sorted(set(next_allowed).union(allowed_tools))
        if denied_tools:
            next_denied = sorted(set(next_denied).union(denied_tools))
        checker._settings = settings.model_copy(update={"allowed_tools": next_allowed, "denied_tools": next_denied})


def _apply_artifact_scope_guard(
    bundle,
    *,
    blocked_root: str | Path | None,
    allowed_roots: tuple[str | Path, ...] | None,
) -> None:
    """Wrap filesystem tools so old learning artifacts stay out of the visible scope."""
    if blocked_root is None:
        return
    blocked_path = Path(blocked_root).expanduser().resolve()
    normalized_allowed = tuple(Path(path).expanduser().resolve() for path in (allowed_roots or ()))
    if not normalized_allowed:
        return

    tool_registry = getattr(bundle, "tool_registry", None)
    tool_map = getattr(tool_registry, "_tools", None)
    if not isinstance(tool_map, dict):
        return

    for name, tool in list(tool_map.items()):
        if name not in ARTIFACT_SCOPED_TOOL_NAMES:
            continue
        if isinstance(tool, ArtifactScopeGuardedTool):
            continue
        tool_map[name] = ArtifactScopeGuardedTool(
            tool,
            blocked_root=blocked_path,
            allowed_roots=normalized_allowed,
        )


async def _run_learning_turn(
    *,
    bundle,
    prompt: str,
    workspace: LearningWorkspace,
    loop_index: int,
    status_printer: StatusPrinter | None = None,
) -> LearningTurnOutcome:
    """Run one learning turn and return a structured outcome."""
    from agentschool.prompts import build_runtime_system_prompt

    settings = bundle.current_settings()
    if bundle.enforce_max_turns:
        bundle.engine.set_max_turns(settings.max_turns)
    system_prompt = build_runtime_system_prompt(
        settings,
        cwd=bundle.cwd,
        latest_user_prompt=prompt,
        extra_skill_dirs=bundle.extra_skill_dirs,
        extra_plugin_roots=bundle.extra_plugin_roots,
        include_project_memory=bundle.include_project_memory,
    )
    bundle.engine.set_system_prompt(system_prompt)

    assistant_turns = 0
    tool_calls = 0
    max_turns_exceeded = False
    error_messages: list[str] = []

    async def _render_event(event) -> None:
        nonlocal assistant_turns, tool_calls
        if isinstance(event, AssistantTextDelta):
            _append_learning_log(workspace.topic_dir, loop_index, event.text)
        elif isinstance(event, AssistantTurnComplete):
            assistant_turns += 1
        elif isinstance(event, ToolExecutionStarted):
            tool_calls += 1
            await _print_status(
                status_printer,
                _format_tool_started("learning", event.tool_name, event.tool_input),
            )
        elif isinstance(event, ToolExecutionCompleted):
            await _print_status(
                status_printer,
                _format_tool_completed("learning", event.tool_name, event.output, is_error=event.is_error),
            )
        elif isinstance(event, StatusEvent):
            await _print_status(status_printer, f"[learning] {event.message}")
        elif isinstance(event, ErrorEvent):
            error_messages.append(event.message)
            await _print_status(status_printer, f"[learning] ERROR {event.message}")

    try:
        async for event in bundle.engine.submit_message(prompt):
            await _render_event(event)
    except MaxTurnsExceeded as exc:
        max_turns_exceeded = True
        await _print_status(status_printer, f"Stopped after {exc.max_turns} turns (max_turns).")
    finally:
        bundle.session_backend.save_snapshot(
            cwd=bundle.cwd,
            model=settings.model,
            system_prompt=system_prompt,
            messages=bundle.engine.messages,
            usage=bundle.engine.total_usage,
            session_id=bundle.session_id,
            tool_metadata=bundle.engine.tool_metadata,
        )

    exam_requested = bool(bundle.engine.tool_metadata.get("learning_exam_called"))
    exam_passed = bool(bundle.engine.tool_metadata.get("learning_exam_passed"))
    exam_record = bundle.engine.tool_metadata.get("learning_last_exam_record")
    if not isinstance(exam_record, ExamAttemptRecord):
        exam_record = None
    return LearningTurnOutcome(
        max_turns_exceeded=max_turns_exceeded,
        exam_requested=exam_requested,
        exam_passed=exam_passed and exam_record is not None,
        exam_record=exam_record,
        assistant_turns=assistant_turns,
        tool_calls=tool_calls,
        error_messages=tuple(error_messages),
    )


async def run_learning_workflow(
    *,
    workspace: LearningWorkspace,
    model: str | None = None,
    max_turns: int | None = None,
    status_printer: StatusPrinter | None = None,
    agent_runner: Callable[..., Awaitable[AgentRunResult]] = run_headless_agent_session,
) -> LearningRunSummary:
    """Run one persistent learning session that may only exit after request_exam passes."""
    from agentschool.ui.runtime import build_runtime, close_runtime, start_runtime

    async def _noop_permission(_tool_name: str, _reason: str) -> bool:
        return True

    async def _noop_ask(_question: str) -> str:
        return ""

    bundle = await build_runtime(
        prompt=workspace.prompt,
        cwd=str(workspace.session_cwd),
        model=model,
        max_turns=max_turns,
        permission_mode="full_auto",
        permission_prompt=_noop_permission,
        ask_user_prompt=_noop_ask,
    )
    bundle.tool_registry.register(RequestExamTool())
    bundle.engine.tool_metadata.update(
        {
            "learning_workspace": workspace,
            "learning_model": model,
            "learning_max_turns": max_turns,
            "learning_status_printer": status_printer,
            "learning_agent_runner": agent_runner,
            "learning_exam_attempts": 0,
            "learning_exam_called": False,
            "learning_exam_passed": False,
            "learning_last_exam_record": None,
        }
    )
    allowed_artifact_roots = [workspace.topic_dir]
    if workspace.instruction_workspace is not None:
        allowed_artifact_roots.append(workspace.instruction_workspace)
    _apply_artifact_scope_guard(
        bundle,
        blocked_root=workspace.artifact_root,
        allowed_roots=tuple(allowed_artifact_roots),
    )
    await start_runtime(bundle)

    current_prompt = workspace.prompt
    loop_index = 0
    last_prompt_kind = "initial"

    try:
        while True:
            await _print_status(status_printer, f"Learning session turn {loop_index + 1}")
            outcome = await _run_learning_turn(
                bundle=bundle,
                prompt=current_prompt,
                workspace=workspace,
                loop_index=loop_index,
                status_printer=status_printer,
            )
            await _print_status(
                status_printer,
                (
                    "Learning turn outcome: "
                    f"assistant_turns={outcome.assistant_turns}, "
                    f"tool_calls={outcome.tool_calls}, "
                    f"exam_requested={outcome.exam_requested}, "
                    f"exam_passed={outcome.exam_passed}, "
                    f"max_turns_exceeded={outcome.max_turns_exceeded}, "
                    f"errors={len(outcome.error_messages)}"
                ),
            )
            loop_index += 1
            if outcome.exam_record is not None and outcome.exam_passed:
                await _print_status(
                    status_printer,
                    f"Exam passed with a perfect score ({outcome.exam_record.score}/{outcome.exam_record.max_score}).",
                )
                return LearningRunSummary(
                    workspace=workspace,
                    attempts=outcome.exam_record.attempt,
                    final_score=outcome.exam_record.score,
                    max_score=outcome.exam_record.max_score,
                    passed=True,
                )

            if outcome.exam_record is not None and outcome.exam_requested:
                await _print_status(
                    status_printer,
                    f"Exam score {outcome.exam_record.score}/{outcome.exam_record.max_score}; returning to learning.",
                )
                current_prompt = build_learning_retry_prompt(workspace, outcome.exam_record)
                last_prompt_kind = "retry_after_exam"
                bundle.engine.tool_metadata["learning_exam_called"] = False
                bundle.engine.tool_metadata["learning_exam_passed"] = False
                continue

            if outcome.max_turns_exceeded:
                raise RuntimeError(
                    "Learning session hit max_turns before a valid graduation exam completed. "
                    "This indicates the agent stopped making progress inside a single learning turn."
                )

            if outcome.error_messages:
                if _all_transient_learning_errors(outcome.error_messages):
                    if last_prompt_kind == "retry_after_transient_error":
                        raise RuntimeError(
                            "Learning turn hit repeated transient network/provider errors in the same session: "
                            + " | ".join(outcome.error_messages)
                        )
                    await _print_status(
                        status_printer,
                        "Learning turn hit a transient network/provider error; retrying within the same session.",
                    )
                    current_prompt = build_transient_error_followup(workspace, outcome.error_messages)
                    last_prompt_kind = "retry_after_transient_error"
                    continue
                raise RuntimeError(
                    "Learning turn failed before request_exam: "
                    + " | ".join(outcome.error_messages)
                )

            if outcome.assistant_turns == 0 and outcome.tool_calls == 0:
                raise RuntimeError(
                    "Learning turn produced no assistant output and no tool activity before request_exam. "
                    "This usually indicates an upstream model/provider failure."
                )

            if last_prompt_kind == "missing_exam_followup":
                raise RuntimeError(
                    "Learning agent stopped again without calling request_exam. "
                    "The session ended normally but did not trigger the only valid graduation tool."
                )

            current_prompt = build_missing_exam_followup(workspace)
            last_prompt_kind = "missing_exam_followup"
    finally:
        await close_runtime(bundle)


async def _run_student_turn(
    *,
    bundle,
    prompt: str,
    status_printer: StatusPrinter | None = None,
) -> StudentTurnOutcome:
    """Run one student turn and return whether a validated submission happened."""
    from agentschool.prompts import build_runtime_system_prompt

    settings = bundle.current_settings()
    if bundle.enforce_max_turns:
        bundle.engine.set_max_turns(settings.max_turns)
    system_prompt = build_runtime_system_prompt(
        settings,
        cwd=bundle.cwd,
        latest_user_prompt=prompt,
        extra_skill_dirs=bundle.extra_skill_dirs,
        extra_plugin_roots=bundle.extra_plugin_roots,
        include_project_memory=bundle.include_project_memory,
    )
    bundle.engine.set_system_prompt(system_prompt)

    collected_text = ""
    assistant_turns = 0
    tool_calls = 0
    max_turns_exceeded = False
    error_messages: list[str] = []

    async def _render_event(event) -> None:
        nonlocal collected_text, assistant_turns, tool_calls
        if isinstance(event, AssistantTextDelta):
            collected_text += event.text
        elif isinstance(event, AssistantTurnComplete):
            assistant_turns += 1
        elif isinstance(event, ToolExecutionStarted):
            tool_calls += 1
            await _print_status(
                status_printer,
                _format_tool_started("student", event.tool_name, event.tool_input),
            )
        elif isinstance(event, ToolExecutionCompleted):
            await _print_status(
                status_printer,
                _format_tool_completed("student", event.tool_name, event.output, is_error=event.is_error),
            )
        elif isinstance(event, StatusEvent):
            await _print_status(status_printer, f"[student] {event.message}")
        elif isinstance(event, ErrorEvent):
            error_messages.append(event.message)
            await _print_status(status_printer, f"[student] ERROR {event.message}")

    try:
        async for event in bundle.engine.submit_message(prompt):
            await _render_event(event)
    except MaxTurnsExceeded as exc:
        max_turns_exceeded = True
        await _print_status(status_printer, f"[student] Stopped after {exc.max_turns} turns (max_turns).")
    finally:
        bundle.session_backend.save_snapshot(
            cwd=bundle.cwd,
            model=settings.model,
            system_prompt=system_prompt,
            messages=bundle.engine.messages,
            usage=bundle.engine.total_usage,
            session_id=bundle.session_id,
            tool_metadata=bundle.engine.tool_metadata,
        )

    submitted = bool(bundle.engine.tool_metadata.get("student_submission_succeeded"))
    return StudentTurnOutcome(
        max_turns_exceeded=max_turns_exceeded,
        submitted=submitted,
        assistant_turns=assistant_turns,
        tool_calls=tool_calls,
        error_messages=tuple(error_messages),
        text=collected_text.strip(),
    )


async def run_student_submission_workflow(
    *,
    workspace: LearningWorkspace,
    student_dir: Path,
    public_exam_path: Path,
    answers_path: Path,
    model: str | None = None,
    max_turns: int | None = None,
    status_printer: StatusPrinter | None = None,
) -> AgentRunResult:
    """Run the persistent student session until the submission tool succeeds."""
    from agentschool.ui.runtime import build_runtime, close_runtime, start_runtime

    async def _noop_permission(_tool_name: str, _reason: str) -> bool:
        return True

    async def _noop_ask(_question: str) -> str:
        return ""

    student_prompt = build_student_prompt(
        workspace=workspace,
        public_exam_path=public_exam_path,
        answers_path=answers_path,
    )
    bundle = await build_runtime(
        prompt=student_prompt,
        cwd=str(student_dir),
        model=model,
        max_turns=max_turns,
        permission_mode="full_auto",
        permission_prompt=_noop_permission,
        ask_user_prompt=_noop_ask,
    )
    bundle.tool_registry.register(SubmitStudentAnswersTool())
    bundle.engine.tool_metadata.update(
        {
            "student_public_exam_path": str(public_exam_path),
            "student_answers_path": str(answers_path),
            "student_goal_name": workspace.goal_name,
            "student_submission_called": False,
            "student_submission_succeeded": False,
            "student_last_submission_path": None,
        }
    )
    _restrict_runtime_tools(
        bundle,
        allowed_tools=STUDENT_ALLOWED_TOOLS,
        denied_tools=frozenset({"bash"}),
    )
    _apply_artifact_scope_guard(
        bundle,
        blocked_root=workspace.artifact_root,
        allowed_roots=(student_dir,),
    )
    await start_runtime(bundle)

    current_prompt = student_prompt
    loop_index = 0
    last_prompt_kind = "initial"
    collected_text_parts: list[str] = []

    try:
        while True:
            await _print_status(status_printer, f"Student session turn {loop_index + 1}")
            outcome = await _run_student_turn(
                bundle=bundle,
                prompt=current_prompt,
                status_printer=status_printer,
            )
            if outcome.text:
                collected_text_parts.append(outcome.text)
            await _print_status(
                status_printer,
                (
                    "Student turn outcome: "
                    f"assistant_turns={outcome.assistant_turns}, "
                    f"tool_calls={outcome.tool_calls}, "
                    f"submitted={outcome.submitted}, "
                    f"max_turns_exceeded={outcome.max_turns_exceeded}, "
                    f"errors={len(outcome.error_messages)}"
                ),
            )
            loop_index += 1

            if outcome.submitted:
                return AgentRunResult(text="\n".join(collected_text_parts).strip())

            if outcome.max_turns_exceeded:
                raise RuntimeError(
                    "Student subagent hit max_turns before submit_student_answers completed. "
                    "This indicates the student stopped making progress inside a single exam turn."
                )

            if outcome.error_messages:
                raise RuntimeError(
                    "Student subagent failed before submit_student_answers: "
                    + " | ".join(outcome.error_messages)
                )

            if outcome.assistant_turns == 0 and outcome.tool_calls == 0:
                raise RuntimeError(
                    "Student subagent produced no assistant output and no tool activity before "
                    "submit_student_answers. This usually indicates an upstream model/provider failure."
                )

            if last_prompt_kind == "missing_submit_followup":
                raise RuntimeError(
                    "Student subagent stopped again without calling submit_student_answers. "
                    "The session ended normally but did not trigger the only valid submission tool."
                )

            current_prompt = build_missing_student_submission_followup(
                workspace=workspace,
                public_exam_path=public_exam_path,
                answers_path=answers_path,
            )
            last_prompt_kind = "missing_submit_followup"
    finally:
        await close_runtime(bundle)


async def run_exam_attempt(
    *,
    workspace: LearningWorkspace,
    attempt: int,
    model: str | None = None,
    max_turns: int | None = None,
    status_printer: StatusPrinter | None = None,
    agent_runner: Callable[..., Awaitable[AgentRunResult]] = run_headless_agent_session,
) -> ExamAttemptRecord:
    """Run one mandatory examiner -> student -> grader loop."""
    attempt_dir = workspace.exams_dir / f"attempt-{attempt:03d}"
    examiner_dir = attempt_dir / "examiner"
    student_dir = attempt_dir / "student"
    grader_dir = attempt_dir / "grader"
    for directory in (examiner_dir, student_dir, grader_dir):
        directory.mkdir(parents=True, exist_ok=True)

    public_exam_path = examiner_dir / "public_exam.json"
    answer_key_path = examiner_dir / "answer_key.json"
    answers_path = student_dir / "student_answers.json"
    grading_path = grader_dir / "grading.json"
    shutil.copytree(workspace.skill_dir, student_dir / "skill", dirs_exist_ok=True)

    previous_questions = _collect_previous_questions(workspace.exams_dir, attempt)
    examiner_prompt = build_examiner_prompt(
        workspace=workspace,
        attempt=attempt,
        public_exam_path=public_exam_path,
        answer_key_path=answer_key_path,
        previous_questions=previous_questions,
    )
    await _print_status(status_printer, f"Exam attempt {attempt}: generating a fresh hard exam.")
    examiner_result = await agent_runner(
        **_agent_runner_kwargs(
            agent_runner,
            prompt=examiner_prompt,
            cwd=examiner_dir,
            model=model,
            max_turns=max_turns,
            status_printer=status_printer,
            stage_label="examiner",
            allowed_tools=EXAMINER_ALLOWED_TOOLS,
            denied_tools=frozenset({"bash"}),
            artifact_guard_root=workspace.artifact_root,
            allowed_artifact_roots=(examiner_dir, workspace.topic_dir),
        )
    )
    _write_text(examiner_dir / "examiner_response.md", examiner_result.text)
    _require_file(public_exam_path, "examiner public_exam.json")
    _require_file(answer_key_path, "examiner answer_key.json")

    shutil.copy2(public_exam_path, student_dir / "public_exam.json")
    await _print_status(status_printer, f"Exam attempt {attempt}: student subagent is answering from the skill bundle.")
    student_result = await run_student_submission_workflow(
        workspace=workspace,
        student_dir=student_dir,
        public_exam_path=student_dir / "public_exam.json",
        answers_path=answers_path,
        model=model,
        max_turns=max_turns,
        status_printer=status_printer,
    )
    _write_text(student_dir / "student_response.md", student_result.text)
    _require_file(answers_path, "student_answers.json")

    shutil.copy2(public_exam_path, grader_dir / "public_exam.json")
    shutil.copy2(answer_key_path, grader_dir / "answer_key.json")
    shutil.copy2(answers_path, grader_dir / "student_answers.json")
    grader_prompt = build_grader_prompt(
        workspace=workspace,
        attempt=attempt,
        public_exam_path=grader_dir / "public_exam.json",
        answer_key_path=grader_dir / "answer_key.json",
        answers_path=grader_dir / "student_answers.json",
        grading_path=grading_path,
    )
    await _print_status(status_printer, f"Exam attempt {attempt}: grader subagent is scoring the answers.")
    grader_result = await agent_runner(
        **_agent_runner_kwargs(
            agent_runner,
            prompt=grader_prompt,
            cwd=grader_dir,
            model=model,
            max_turns=max_turns,
            status_printer=status_printer,
            stage_label="grader",
            allowed_tools=GRADER_ALLOWED_TOOLS,
            denied_tools=frozenset({"bash"}),
            artifact_guard_root=workspace.artifact_root,
            allowed_artifact_roots=(grader_dir,),
        )
    )
    _write_text(grader_dir / "grader_response.md", grader_result.text)
    _require_file(grading_path, "grading.json")

    grading = _read_json(grading_path)
    score = int(grading.get("score", 0))
    max_score = int(grading.get("max_score", 0))
    passed = bool(grading.get("passed", False)) and max_score > 0 and score == max_score
    summary = str(grading.get("summary", "")).strip() or f"Scored {score}/{max_score}"
    return ExamAttemptRecord(
        attempt=attempt,
        attempt_dir=attempt_dir,
        public_exam_path=public_exam_path,
        answer_key_path=answer_key_path,
        answers_path=answers_path,
        grading_path=grading_path,
        score=score,
        max_score=max_score,
        passed=passed,
        summary=summary,
    )


def build_learning_prompt(
    *,
    goal_name: str,
    topic_dir: Path,
    skill_dir: Path,
    skill_file: Path,
    research_notes: Path,
    experiments_dir: Path,
    visible_instruction: str | None = None,
) -> str:
    """Build an agent-native learning prompt without a fixed orchestration loop."""
    instruction_block = ""
    if visible_instruction is not None:
        instruction_block = (
            "\nVisible instruction only:\n"
            "The text below is the only task statement you may rely on. Do not assume access "
            "to hidden tests, bundled solutions, or benchmark-only delivery files.\n\n"
            f"{visible_instruction.strip()}\n"
        )

    return f"""You are AgentSchool running a fully agentic learning session.

Learning goal:
{goal_name}
{instruction_block}
You fully own the process. Research when useful, inspect the local workspace when useful,
run experiments when useful, and iteratively distill a reusable skill bundle. Do not wait
for step-by-step instructions and do not simulate work you did not actually do.

Research and experimentation preference:
- Strongly prefer tool-backed research over relying on prior memory alone. If the task touches
  external APIs, libraries, file formats, scientific methods, domain conventions, or tricky
  implementation details, proactively use available search, fetch, browser, or code-search
  tools and record the useful findings in `{research_notes}`.
- For information gathering, default to search/read/fetch/browser-style tools instead of bash.
  Use bash mainly when you need to execute code, reproduce behavior, validate an environment,
  run a real experiment, or inspect something that higher-level tools cannot access directly.
- Avoid using bash for simple file reading, directory browsing, or lightweight lookup when a
  dedicated search/read tool can answer the question more directly.
- Do not spend turns on generic environment probing such as repeated `which python`, `python --version`,
  `pip --version`, broad `find`, broad `ls`, or duplicate import checks unless that information is directly
  required to unblock a concrete experiment you are about to run.
- Do not install packages by default. Only install a package if a concrete experiment is blocked, the missing
  dependency has been verified, and no already-available tool/library/path can answer the question more directly.
- Prefer one focused bash experiment over many tiny probe commands. Bundle related checks into a small script
  or reproducible experiment when possible, and avoid noisy micro-commands that each test only one trivial fact.
- For concrete coding, debugging, data, or workflow tasks, strongly prefer running real
  experiments before requesting graduation. Use shell/code/file tools to create small
  probes, reproductions, sanity checks, or validation scripts, and save useful artifacts
  under `{experiments_dir}/*`.
- Do not stop at happy-path validation. Before requesting graduation, deliberately try to falsify your current
  approach with adversarial or counterexample-focused end-to-end self-tests that target likely confusion points:
  near misses, ambiguous inputs, ranking inversions, aliasing, edge-case clusters, hidden assumptions, or
  situations where a plausible recipe could silently produce the wrong output.
- If the current approach only survives local toy checks but has not survived an explicit attempt to break it,
  the skill is not ready yet.
- Treat the skill as evidence-backed only when it reflects researched facts and observed
  behavior from your own experiments whenever those are feasible.
- Treat other `.agentschool` runs as stale artifacts.
  Do not mine old `.agentschool` runs, old exams, or previous skill bundles for answers. Only
  use artifacts from the current run unless a later retry prompt explicitly points you to the
  current run's own exam feedback.

Collaboration preference:
- If the learning goal is broad, ambiguous, or naturally parallelizable, proactively use
  swarm/sub-agent/task tools to split work into focused research or experiment threads.
- Prefer parallel delegation for literature review, codebase reconnaissance, alternative
  hypothesis testing, and comparing experimental approaches when that will improve quality.
- Synthesize delegated results yourself before writing the final skill bundle; do not just
  forward raw sub-agent output.
- If the task is small or tightly scoped, staying single-agent is fine.

Artifact root:
{topic_dir}

Write the learned skill bundle here:
- {skill_file}

You may additionally create supporting artifacts here when useful:
- {research_notes}
- {skill_dir / "refs"}/*
- {skill_dir / "scripts"}/*
- {experiments_dir}/*

Important constraints:
- Stay within the current workspace and the artifact root above.
- Do not write benchmark-only delivery files such as `/root/workspace/solution.py`.
- Keep the skill reusable and evidence-backed rather than overfitting to one task instance.
- Distill general methods, invariants, debugging heuristics, validation strategies, and reusable
  implementation patterns. Do not turn the skill bundle into an exam cheat sheet.
- Do not copy concrete exam questions, question IDs, answer outlines, or rubric-shaped response
  templates into `{skill_file}` or supporting notes.
- If an exam exposes a gap, extract the underlying transferable lesson and update the skill at that
  abstraction level instead of storing exam-specific prose.
- In `{skill_file}`, distinguish clearly between:
  - verified invariants or conclusions that were actually checked,
  - plausible but not fully verified heuristics or assumptions,
  - known failure surfaces, uncertainty, and alternative branches to try when the main recipe fails.
- Do not present an unverified recipe as guaranteed truth. Mark what was confirmed, what is still a best guess,
  and what kinds of inputs or conditions are most likely to break the approach.
- Structure `{skill_file}` so another coding agent can preserve the important parts under pressure. Include explicit
  sections such as `Must-Preserve Decisions`, `Do-Not-Simplify`, and `Failure Surfaces / Fallback Branches` whenever
  the task has implementation details that would be easy to accidentally compress away.
- Your session is not allowed to finish with a normal final answer.
- The only way to graduate is to call the `request_exam` tool when you believe the current
  skill bundle is ready.
- `request_exam` triggers a separate examiner agent, a separate student agent using only the
  learned skill bundle, and a separate grader agent.
- The workflow only exits if that exam returns a perfect score. Otherwise you must continue
  learning in the same session using the exam feedback.
- Therefore, produce a skill bundle strong enough that another agent can use it to ace novel
  high-difficulty questions.
- Use your final natural-language message only after `request_exam` has returned a perfect pass.
"""


def build_learning_retry_prompt(workspace: LearningWorkspace, latest_exam: ExamAttemptRecord) -> str:
    """Build a retry prompt after a failed exam attempt."""
    return f"""You are AgentSchool continuing the same learning run after a failed exam.

Learning goal:
{workspace.goal_name}

Current skill bundle:
- {workspace.skill_file}

Latest failed exam artifacts:
- public exam: {latest_exam.public_exam_path}
- student answers: {latest_exam.answers_path}
- grading report: {latest_exam.grading_path}

Requirements:
- Read the failed exam and grading feedback carefully.
- Improve the skill bundle so that a different student agent could score perfectly on a fresh hard exam.
- Update the skill and any supporting references or notes as needed.
- Prefer search/read/fetch/browser-style tools for understanding gaps and gathering evidence.
  Use bash primarily for real execution, reproduction, verification, or experiment runs.
- Do not fall back to generic environment probing or package installation unless a concrete retry experiment is
  blocked on a verified missing dependency.
- Prefer a small number of focused experiments over many tiny bash probes.
- Do not merely append vague notes; concretely fix missing steps, failure modes, verification guidance,
  and edge cases revealed by the grading report.
- Design at least one adversarial or counterexample-focused end-to-end retry test that tries to break the
  current recipe along the failure surfaces suggested by the failed exam.
- Convert exam feedback into reusable knowledge. Add the general principle, invariant, or procedure that
  was missing, but do not paste exam-specific question wording or build a canned answer template.
- Keep `{workspace.skill_file}` focused on transferable skill. It should help with unseen tasks and unseen
  exam prompts, not memorize one examiner's phrasing.
- Update `{workspace.skill_file}` so it separates verified conclusions from plausible heuristics and documents
  the current uncertainty, known breakpoints, and fallback branches.
- Add or refine explicit `Must-Preserve Decisions` / `Do-Not-Simplify` guidance so a downstream coding agent does
  not collapse the repaired skill back into an oversimplified implementation.
- Do not inspect old `.agentschool` runs outside `{workspace.topic_dir}`. Use only the current run's
  artifacts and the latest failed exam files listed above.
- When you are ready again, call `request_exam` with a short rationale. Do not stop with a plain final answer.
"""


def build_missing_exam_followup(workspace: LearningWorkspace) -> str:
    """Prompt used when the learning agent tried to stop without requesting an exam."""
    return f"""You are still inside the same AgentSchool learning session for:
{workspace.goal_name}

You attempted to stop without calling `request_exam`.

Rules:
- A plain final answer does not end this session.
- The only valid graduation path is to improve the skill bundle and then call `request_exam`.
- If the skill is not ready, continue researching, experimenting, and revising it now.

Continue the learning session.
"""


def build_transient_error_followup(workspace: LearningWorkspace, error_messages: tuple[str, ...]) -> str:
    """Prompt used when a learning turn hit a transient upstream failure."""
    rendered_errors = "\n".join(f"- {message}" for message in error_messages)
    return f"""You are still inside the same AgentSchool learning session for:
{workspace.goal_name}

The previous turn was interrupted by a transient network/provider failure:
{rendered_errors}

Rules:
- Do not restart the learning process from scratch.
- Continue from the current session state, current artifacts, and work already completed.
- Do not inspect old `.agentschool` runs outside `{workspace.topic_dir}`.
- Do not respond to a transient provider/network failure by redoing generic environment checks or package installs.
- Resume with the smallest focused next step that advances the current learning run.
- Resume researching, experimenting, or refining the current skill bundle, then call `request_exam`
  when the skill is ready.

Continue the learning session.
"""


def build_missing_student_submission_followup(
    *,
    workspace: LearningWorkspace,
    public_exam_path: Path,
    answers_path: Path,
) -> str:
    """Prompt used when the student agent tried to stop without submitting answers."""
    return f"""You are still inside the same AgentSchool student exam session for:
{workspace.goal_name}

You attempted to stop without calling `submit_student_answers`.

Rules:
- A plain final answer does not count as an exam submission.
- The only valid way to finish this student session is to call `submit_student_answers`.
- Re-read `{public_exam_path}` if needed, consult only the local `./skill` bundle, and submit one answer
  for every public exam question with the exact original question IDs.
- The submission tool will create `{answers_path}` for you. Do not try to write that JSON manually.

Continue the student exam session and submit the official answers now.
"""


def build_examiner_prompt(
    *,
    workspace: LearningWorkspace,
    attempt: int,
    public_exam_path: Path,
    answer_key_path: Path,
    previous_questions: list[str],
) -> str:
    """Prompt for the examiner agent."""
    prior_questions_block = ""
    if previous_questions:
        prior_questions_block = "Do not repeat or trivially paraphrase any of these earlier questions:\n"
        prior_questions_block += "\n".join(f"- {question}" for question in previous_questions)
        prior_questions_block += "\n"
    instruction_block = ""
    if workspace.visible_instruction is not None:
        instruction_block = f"\nVisible instruction context:\n{workspace.visible_instruction.strip()}\n"

    return f"""You are the examiner subagent for AgentSchool.

Goal:
{workspace.goal_name}
{instruction_block}
Attempt number: {attempt}

You do not have access to the learned skill bundle and must not try to find it. Your job is to create
a fresh, high-difficulty exam that tests transferable understanding of the goal. Focus on tricky edge
cases, hard judgment calls, verification strategy, and failure analysis.

Scope and tool limits:
- Your scope is fixed to the current examiner workspace plus earlier public exams from prior attempts.
- Do not inspect the student workspace, grader workspace, or the learned skill bundle.
- Do not use bash or other code-execution tools. Use only file-reading, file-writing, and lightweight
  search/navigation tools needed to inspect prior exam files and write the new exam artifacts.

{prior_questions_block}
Write exactly these files:
1. {public_exam_path}
2. {answer_key_path}

`{public_exam_path}` must be strict JSON:
{{
  "attempt": {attempt},
  "goal": "{workspace.goal_name}",
  "questions": [
    {{
      "id": "Q1",
      "prompt": "<hard question>",
      "points": 10
    }}
  ]
}}

Requirements:
- Produce exactly 3 questions.
- Make every question difficult and non-trivial.
- Total points must sum to 30.
- Make the exam meaningfully varied across attempts and within the current attempt.
- Do not merely restate the same rubric with different nouns or surface wording.
- Ensure the 3 questions cover different capability types, such as:
  - failure analysis or debugging
  - verification or experiment design
  - edge cases, tradeoffs, or judgment calls
  - implementation strategy or recovery procedure
- Make each question require a different style of reasoning. Avoid three questions that all ask for
  the same kind of checklist, pipeline summary, or failure-mode enumeration.
- If prior questions exist, deliberately explore substantially different angles, task slices, or
  evidence expectations than those earlier exams.
- At least one question must be an implementation-fidelity question: it should force the student to identify
  which critical constraints or must-preserve decisions from the skill cannot be simplified away in a real
  implementation, and what concrete failure would happen if they were omitted.

`{answer_key_path}` must be strict JSON:
{{
  "attempt": {attempt},
  "goal": "{workspace.goal_name}",
  "questions": [
    {{
      "id": "Q1",
      "points": 10,
      "ideal_answer": "<what a perfect answer should contain>",
      "rubric": ["<criterion>", "<criterion>"],
      "must_preserve": ["<critical detail the student should preserve>"],
      "simplification_failures": ["<what breaks if that detail is simplified away>"]
    }}
  ],
  "max_score": 30
}}

Only finish after both files exist and are valid JSON.
"""


def build_student_prompt(
    *,
    workspace: LearningWorkspace,
    public_exam_path: Path,
    answers_path: Path,
) -> str:
    """Prompt for the student agent."""
    return f"""You are the student subagent for AgentSchool.

Goal:
{workspace.goal_name}

You may use only the learned skill bundle available under:
- ./skill

Read the exam from:
- {public_exam_path}

Answer every question strictly by relying on the skill bundle and what it teaches. If the skill is weak
or incomplete, do your best but do not invent support that is not grounded in the skill.

Scope and tool limits:
- Your scope is fixed to the current student workspace: the copied `public_exam.json` and the local `./skill`
  bundle only.
- Do not inspect examiner or grader directories, and do not search outside this student workspace.
- Do not use bash or other code-execution tools. Use only file-reading, file-writing, and lightweight
  search/navigation tools needed to read the exam, inspect the skill bundle, and submit the answer file.
- Read the exam first, then consult only the minimum necessary skill files, then write the answer JSON.
- Preserve the exact question IDs from the public exam. Produce one answer entry for every exam question.
- Treat the exam as an implementation-fidelity check, not just a prose check. For each answer, explicitly identify
  the must-preserve constraints from the skill and the concrete failure risk if someone simplifies them away.
- Do not try to write the answer JSON manually. Use the `submit_student_answers` tool for the final submission.

When you are ready, call `submit_student_answers` once with:
{{
  "questions": [
    {{
      "id": "Q1",
      "answer": "<your answer>",
      "skill_evidence": ["<specific skill section, step, or idea used>"],
      "must_preserve": ["<critical skill constraint or decision that must remain intact>"],
      "simplification_risks": ["<what would break if that critical detail were omitted or simplified>"]
    }}
  ]
}}

The submission tool will validate the question IDs and create:
- {answers_path}

Only finish after `submit_student_answers` succeeds.
"""


def build_grader_prompt(
    *,
    workspace: LearningWorkspace,
    attempt: int,
    public_exam_path: Path,
    answer_key_path: Path,
    answers_path: Path,
    grading_path: Path,
) -> str:
    """Prompt for the grader agent."""
    return f"""You are the grader subagent for AgentSchool.

Goal:
{workspace.goal_name}

Grade the student's answers strictly against the answer key.

Inputs:
- public exam: {public_exam_path}
- answer key: {answer_key_path}
- student answers: {answers_path}

Scope and tool limits:
- Your scope is fixed to the current grader workspace and the three JSON inputs above.
- Do not use bash or other code-execution tools. Use only file-reading and file-writing tools needed to
  inspect the JSON inputs and write the grading report.

Write exactly this file:
- {grading_path}

The file must be strict JSON:
{{
  "attempt": {attempt},
  "score": 0,
  "max_score": 30,
  "passed": false,
  "summary": "<short summary>",
  "questions": [
    {{
      "id": "Q1",
      "earned": 0,
      "points": 10,
      "verdict": "<short verdict>",
      "missing": ["<missing point>", "<missing point>"]
    }}
  ],
  "retry_guidance": ["<how the skill should improve>", "<how the skill should improve>"]
}}

Rules:
- `passed` must be true if and only if `score == max_score`.
- Score every question independently and sum them exactly.
- Be strict. Full credit requires a truly complete answer.
- Check implementation fidelity, not just plausibility. If the student misses a must-preserve constraint, or
  cannot explain what would break when it is simplified away, deduct credit even if the prose sounds generally right.
- Compare the student's `must_preserve` and `simplification_risks` fields against the answer key's expected
  critical details and simplification failures.
- The retry guidance must be concrete enough for the learning agent to improve the skill.

Only finish after the file exists and is valid JSON.
"""


def _collect_previous_questions(exams_dir: Path, attempt: int) -> list[str]:
    """Return earlier public question prompts so the next exam stays fresh."""
    questions: list[str] = []
    for index in range(1, attempt):
        public_exam = exams_dir / f"attempt-{index:03d}" / "examiner" / "public_exam.json"
        if not public_exam.exists():
            continue
        payload = _read_json(public_exam)
        for question in payload.get("questions", []):
            prompt = str(question.get("prompt", "")).strip()
            if prompt:
                questions.append(prompt)
    return questions


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n" if text else "", encoding="utf-8")


def _append_learning_log(topic_dir: Path, loop_index: int, text: str) -> None:
    if not text:
        return
    log_path = topic_dir / f"learning-session-{loop_index + 1:03d}.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _agent_runner_kwargs(
    agent_runner: Callable[..., Awaitable[AgentRunResult]],
    **kwargs,
) -> dict[str, object]:
    signature = inspect.signature(agent_runner)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{label} was not created: {path}")


def _format_exam_tool_output(record: ExamAttemptRecord) -> str:
    lines = [
        f"Exam attempt {record.attempt} complete.",
        f"Score: {record.score}/{record.max_score}",
        f"Passed: {'true' if record.passed else 'false'}",
        f"Summary: {record.summary}",
        f"Public exam: {record.public_exam_path}",
        f"Student answers: {record.answers_path}",
        f"Grading report: {record.grading_path}",
    ]
    if not record.passed:
        lines.append("Result: keep learning in the same session, improve the skill bundle, then call request_exam again.")
    return "\n".join(lines)


def _short_text(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if not compact:
        return "(no output)"
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def _format_tool_started(stage_label: str, tool_name: str, tool_input: dict[str, object]) -> str:
    details = _summarize_tool_input(tool_input)
    return f"[{stage_label}] TOOL START {tool_name}: {details}"


def _format_tool_completed(stage_label: str, tool_name: str, output: str, *, is_error: bool) -> str:
    status = "TOOL ERROR" if is_error else "TOOL DONE"
    details = _summarize_tool_output(output)
    return f"[{stage_label}] {status} {tool_name}: {details}"


def _summarize_tool_input(tool_input: dict[str, object]) -> str:
    if not tool_input:
        return "(no args)"

    priority_keys = [
        "command",
        "file_path",
        "path",
        "pattern",
        "query",
        "information_request",
        "question",
        "task",
        "url",
        "name",
        "description",
    ]
    ordered_keys = [key for key in priority_keys if key in tool_input]
    ordered_keys.extend(key for key in tool_input if key not in ordered_keys)

    parts: list[str] = []
    for key in ordered_keys[:3]:
        parts.append(f"{key}={_format_tool_value(tool_input[key])}")

    extra_count = max(0, len(tool_input) - len(parts))
    if extra_count:
        parts.append(f"+{extra_count} more")
    return ", ".join(parts)


def _summarize_tool_output(output: str) -> str:
    stripped = output.strip()
    if not stripped:
        return "(no output)"

    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(parsed, dict):
                priority_keys = ["status", "message", "result", "output", "score", "max_score", "passed"]
                pieces = [f"{key}={_format_tool_value(parsed[key])}" for key in priority_keys if key in parsed][:3]
                if pieces:
                    return ", ".join(pieces)

    first_line = stripped.splitlines()[0]
    return _short_text(first_line, limit=120)


def _format_tool_value(value: object) -> str:
    if isinstance(value, str):
        if "\n" in value:
            value = value.splitlines()[0]
        compact = " ".join(value.split())
        return repr(compact if len(compact) <= 60 else compact[:57] + "...")
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        preview = ", ".join(_format_tool_value(item) for item in value[:2])
        if len(value) > 2:
            preview += ", ..."
        return f"[{preview}]"
    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return repr(text if len(text) <= 60 else text[:57] + "...")
    return repr(str(value))

async def _print_status(status_printer: StatusPrinter | None, message: str) -> None:
    if status_printer is not None:
        await status_printer(message)


def _resolve_tool_path(base: Path, candidate: str | None) -> Path:
    path = Path(candidate or ".").expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _artifact_scope_violation(
    candidate: Path,
    *,
    blocked_root: Path,
    allowed_roots: tuple[Path, ...],
) -> str | None:
    if not _is_relative_to(candidate, blocked_root):
        return None
    if any(_is_relative_to(candidate, root) for root in allowed_roots):
        return None
    return (
        "Artifact scope guard: access to old learning artifacts is blocked outside the current run. "
        f"Denied path: {candidate}"
    )


def _search_root_fully_blocked(
    search_root: Path,
    *,
    blocked_root: Path,
    allowed_roots: tuple[Path, ...],
) -> bool:
    return _is_relative_to(search_root, blocked_root) and not any(
        _is_relative_to(search_root, root) for root in allowed_roots
    )


def _filter_scoped_search_result(
    result: ToolResult,
    *,
    tool_name: str,
    search_root: Path,
    blocked_root: Path,
    allowed_roots: tuple[Path, ...],
) -> ToolResult:
    output = result.output.strip()
    if not output or output == "(no matches)":
        return result

    kept: list[str] = []
    meta_lines: list[str] = []
    for line in result.output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") or stripped.startswith("("):
            meta_lines.append(line)
            continue
        if tool_name == "glob":
            candidate = _resolve_tool_path(search_root, stripped)
        else:
            path_part, separator, remainder = line.partition(":")
            if not separator:
                meta_lines.append(line)
                continue
            candidate = _resolve_tool_path(search_root, path_part)
            line = f"{candidate if Path(path_part).is_absolute() else path_part}:{remainder}"
        if _artifact_scope_violation(
            candidate,
            blocked_root=blocked_root,
            allowed_roots=allowed_roots,
        ) is None:
            kept.append(line)

    rendered_lines = kept + meta_lines
    if not rendered_lines:
        return ToolResult(output="(no matches)", is_error=result.is_error, metadata=result.metadata)
    return ToolResult(output="\n".join(rendered_lines), is_error=result.is_error, metadata=result.metadata)


def _is_transient_learning_error(message: str) -> bool:
    normalized = message.strip().lower()
    transient_markers = (
        "network error:",
        "connection lost",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection closed",
    )
    return any(marker in normalized for marker in transient_markers)


def _all_transient_learning_errors(error_messages: tuple[str, ...]) -> bool:
    return bool(error_messages) and all(_is_transient_learning_error(message) for message in error_messages)
