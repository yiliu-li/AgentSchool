# Contributing to AgentSchool

AgentSchool is an open-source agent runtime focused on clarity, hackability, and reproducible coding workflows.

## Ways to contribute

- Fix bugs or tighten edge-case handling in the harness runtime.
- Improve docs, onboarding, examples, and architecture notes.
- Add tests for tools, permissions, plugins, MCP, or multi-agent flows.
- Contribute new skills, plugins, or provider compatibility improvements.

## Development setup

```bash
git clone https://github.com/HKUDS/AgentSchool.git
cd AgentSchool
uv sync --extra dev
```

## Local checks

Run the same core checks that CI runs before opening a PR:

```bash
uv run ruff check src tests scripts
uv run pytest -q
```

## Pull request expectations

- Keep PRs scoped. Small, reviewable changes merge faster than broad rewrites.
- Include the problem, the change, and how you verified it.
- Add or update tests when behavior changes.
- Update docs when CLI flags, workflows, or compatibility claims change.
- If you are improving type coverage, feel free to run `uv run mypy src/agentschool`, but it is not yet a required green check for the whole repo.

## Documentation and community contributions

Issue [#7](https://github.com/HKUDS/AgentSchool/issues/7) surfaced several high-value docs needs. Useful contributions in that area include:

- README accuracy improvements and compatibility notes.
- Short, reproducible examples for common workflows.
- Contribution and maintenance docs that make the repo easier to navigate.

## Reporting bugs and proposing features

- Use the GitHub issue templates when possible.
- Include environment details, exact commands, and error output for bugs.
- For features, explain the concrete workflow gap and expected behavior.
- If the request is mostly documentation or maintenance related, say that explicitly so it can be scoped as a docs PR.
