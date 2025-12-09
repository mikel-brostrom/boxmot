# AGENTS.md – Working Guidelines for **BoxMOT**

> These instructions apply to all directories in this repository. Nested `AGENTS.md` files (if added later) override rules for their subtrees.

## Quickstart
- If `uv` is not installed, install it with `pip install uv`, then use the existing `uv` workflow to install dependencies: `uv sync --all-extras --all-groups`.
- Activate the environment in every new shell (`source .venv/bin/activate` if using the default `uv` virtualenv).
- Create feature branches for work: `git checkout -b codex/<short-topic>`.

## Coding Conventions
- Prefer Python type hints and docstrings for any new or modified functions/classes.
- Keep imports sorted and remove unused ones; avoid wrapping imports in `try/except`.
- Maintain consistent logging/messages; avoid printing in library code except for CLIs.
- Follow existing style in files you edit (e.g., spacing, naming); match the surrounding framework conventions.

## Commit & PR Expectations
- Commit messages should start with one of: `feat:`, `fix:`, `refactor:`, `docs:`, `ci:`, `perf:`.
- Each commit should represent a coherent change; avoid mixing unrelated edits.
- PR descriptions should summarize user-facing changes, testing performed, and any follow-up tasks.

## Testing & Verification
- Always run the pytest suite before opening a PR: `pytest` (use markers or paths to scope when necessary, but ensure impacted tests run).
- Run targeted commands relevant to your change when feasible. Typical entry points:
  - `python boxmot/engine/cli.py track ...`
  - `python boxmot/engine/cli.py generate ...`
  - `python boxmot/engine/cli.py eval ...`
  - `python boxmot/engine/cli.py tune ...`
- Note any unrun tests with a short rationale.

## Documentation & Examples
- Update docs or examples when behavior or interfaces change.
- Keep README snippets and CLI help text in sync with code updates.

## Performance & Safety
- Be mindful of model weights and large assets; do not commit generated artifacts.
- Prefer deterministic or seeded behavior for tests/examples when practical.

*Last updated: 2026-01-01*
