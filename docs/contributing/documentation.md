# Build the Documentation

The documentation site combines hand-written guides, generated CLI reference
sections, and a source-derived Python code reference. Use the same locked
environment and strict build locally that CI uses.

## Prepare the environment

Run this once from the repository root, and again whenever `uv.lock` changes:

```bash
uv sync --locked --no-default-groups --extra cpu --group docs --group test
```

This creates the project environment at `.venv`. The CPU profile is required
because generated CLI sections import the BoxMOT command entry point; Python
API extraction itself remains static and does not import runtime backends.

If another virtual environment is active, deactivate it first or invoke the
executables under `.venv/bin` as shown below. This avoids `VIRTUAL_ENV does not
match the project environment` warnings from `uv`.

## Preview locally

Start a strict live-reload server:

```bash
.venv/bin/mkdocs serve --strict
```

Open <http://127.0.0.1:8000/> for the site or
<http://127.0.0.1:8000/reference/> for the generated code reference. Changes
under `docs`, `boxmot`, and `README.md` trigger a
rebuild.

## Build the production site

Create a clean, warning-as-error build before submitting documentation changes:

```bash
.venv/bin/ruff check docs/gen_ref_pages.py tests/unit/docs
.venv/bin/ruff format --check docs/gen_ref_pages.py tests/unit/docs
.venv/bin/pytest -q tests/unit/docs
.venv/bin/mkdocs build --strict
```

The static site is written to `site/`. Do not commit that directory.

## How the code reference is generated

During every MkDocs build, `docs/gen_ref_pages.py`:

1. Parses Python modules under `boxmot` with the standard-library AST without
   importing them.
2. Maps packages and modules into a deterministic navigation tree and rejects
   identifier or output-path collisions.
3. Selects symbols defined locally so imported aliases are not documented more
   than once.
4. Creates virtual Markdown pages for mkdocstrings and links each page back to
   its source file.

Mkdocstrings and Griffe then render signatures and Google-style docstrings.
Generated Markdown exists only for the duration of the build, so do not run the
generator directly or create a committed `docs/reference` tree.

The **Reference → Public API** and **Component API** pages contain curated
interfaces. **Reference → Full Source Reference** contains the full library
module tree, including internal implementation modules but excluding
executable `__main__` shims.

## Information architecture

Keep each fact in the section that owns it, then link there from related
pages:

- **Modes** own runnable workflows, effective arguments, outputs, and examples.
- **Tasks** own AABB, OBB, and mask data contracts.
- **Trackers** own algorithm behavior and tracker-specific tradeoffs.
- **Compare** owns cross-component selection and fair-comparison guidance.
- **Data** owns defaults, profile schemas, and experiment composition.
- **Guides** own cross-mode procedures and deeper explanations.
- **Integrations** own external detector/runtime and deployment boundaries.
- **Reference** owns exact generated CLI and Python interfaces.
- **Help** owns troubleshooting and support routes.

For example, Track mode is the canonical page for source types, media formats,
per-frame results, and plotting. Python and integration pages should link to
those headings instead of copying their tables. When adding a new section,
create an overview page, add it to `mkdocs.yml`, and link it from the nearest
existing landing page.

## Writing API documentation

- Put types in Python signatures and use Google-style docstring sections such
  as `Args`, `Returns`, `Raises`, and `Examples`.
- Document public behavior and constraints rather than restating the code.
- Keep module exports focused; the generator documents locally defined members
  without repeating imported re-exports.
- Run the strict build after adding, moving, or renaming a Python module.
