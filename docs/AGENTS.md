# AGENTS.md - Working Guidelines for `docs/`

> These instructions apply to all files under `docs/`.
> They complement the repo-root `AGENTS.md` and override it for the docs subtree where they differ.

---

## 1. Docs Stack

- The docs site uses **MkDocs Material** with navigation defined in `mkdocs.yml`.
- Generated content is part of the docs workflow:
  - `mkdocs-click` for CLI argument tables
  - `mkdocstrings` for API reference pages
- Site-specific styling lives in `docs/stylesheets/extra.css`.

When adding, removing, or renaming pages, update `mkdocs.yml` in the same change.

## 2. Primary Goal

- Keep the docs aligned with the product as it works today.
- Prefer concise, task-oriented explanations over long conceptual essays.
- Match the current tone: direct, practical, and low on marketing language.
- Do not document behavior that is planned but not implemented.

## 3. Match the Existing Page Style

- Use one `#` title per page.
- Start with a short orienting paragraph that explains what the page is for.
- Prefer short sections with clear headings such as:
  - `Examples`
  - `Core idea`
  - `Outputs`
  - `What it controls`
  - `Related pages`
- Keep paragraphs short and lists flat.
- Use fenced code blocks with an explicit language like `bash` or `python`.
- Use admonitions and tabs only when they fit the surrounding style and improve scannability.

The current docs favor compact pages that help a reader act quickly. Preserve that shape.

## 4. Internal Linking and Navigation

- Use relative Markdown links for docs pages inside `docs/`.
- Keep internal link text descriptive and short.
- If you move or rename a page, update:
  - links in other docs pages
  - `mkdocs.yml`
  - any matching references in `README.md`

Do not leave orphan pages that exist on disk but are unreachable from the docs nav unless there is a clear reason.

## 5. Generated Sections

Do not hand-copy generated CLI or API reference content into Markdown.

- For mode pages under `docs/modes/`, keep the `mkdocs-click` block as the source of truth for argument tables.
- For API pages under `docs/python/`, prefer `mkdocstrings` directives over duplicating signatures or member lists manually.
- If code changes alter generated docs output, update the surrounding prose/examples rather than replacing generated sections with static text.

## 6. Page-Type Guidance

### Mode pages

- Keep example-first structure.
- Show realistic `boxmot ...` commands.
- Explain when to use the mode, common outputs, and any important workflow distinctions.
- Keep the generated CLI argument table at the end of the page.

### Tracker pages

- Include the tracker name as the page title.
- Link the original paper if one exists.
- Summarize the tracker in a few sentences.
- Include a short BoxMOT-specific section describing practical requirements or tradeoffs.
- Keep the tracker API reference directive in place.

### Config and concept pages

- Explain how BoxMOT organizes the feature, not just what a raw file contains.
- Prefer concrete examples over abstract descriptions.
- Link to the most relevant neighboring pages.

### Contributing pages

- Keep checklists actionable.
- Mirror the actual repo workflow and file layout.
- If contribution steps change, update the related docs in the same edit set.

## 7. README and Docs Must Stay in Sync

If a change affects public usage, keep `README.md` and `docs/` aligned in the same PR when practical, especially for:

- CLI examples
- supported modes
- tracker lists
- installation steps
- contributor guidance

Do not update one and knowingly leave the other stale.

## 8. Assets and Styling

- Keep images and media small and intentional.
- Do not add large generated artifacts to `docs/`.
- If you touch `docs/stylesheets/extra.css`, keep changes narrowly scoped to the docs site and preserve the current visual direction unless a broader redesign is intended.

## 9. Verification

For docs-only changes, run from the repo root when possible:

```bash
uv run mkdocs build --strict
```

If the docs change depends on CLI or API behavior, also run the relevant verification command, for example:

```bash
uv run python -m boxmot.engine.cli --help
```

If you cannot run verification, state exactly which command was skipped and why.

## 10. Common Pitfalls

- Do not add a new docs page without wiring it into `mkdocs.yml`.
- Do not paste large CLI help output into Markdown when `mkdocs-click` already generates it.
- Do not describe stale option names, legacy commands, or outdated workflows.
- Do not introduce docs examples that rely on files, weights, or paths that are not explained nearby.
