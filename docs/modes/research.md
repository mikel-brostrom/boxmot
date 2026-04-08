# research

The `research` mode runs GEPA-guided code evolution against a benchmark split. It requires the optional research extra and an OpenAI API key when using OpenAI-hosted proposal models.

```bash
uv sync --extra research
export OPENAI_API_KEY=...
```

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: research
    :style: table
    :prog_name: boxmot research
