# research

The `research` mode runs GEPA-guided code evolution against a benchmark split. It requires the optional research extra and provider credentials for the proposal model you select.

```bash
uv sync --extra research
export OPENAI_API_KEY=...
```

The provider is selected through `--proposal-model`, using a provider-prefixed model identifier. Examples:

```bash
boxmot research --benchmark mot17-ablation --tracker bytetrack \
  --proposal-model openai/gpt-5.4

boxmot research --benchmark mot17-ablation --tracker bytetrack \
  --proposal-model anthropic/claude-sonnet-4-20250514

boxmot research --benchmark mot17-ablation --tracker bytetrack \
  --proposal-model openrouter/openai/gpt-5.4
```

If your provider key is already exported in the expected environment variable, BoxMOT will reuse it. You can also inject the key from the CLI:

```bash
boxmot research --benchmark mot17-ablation --tracker bytetrack \
  --proposal-model openai/gpt-5.4 \
  --proposal-api-key "$OPENAI_API_KEY"

boxmot research --benchmark mot17-ablation --tracker bytetrack \
  --proposal-model anthropic/claude-sonnet-4-20250514 \
  --proposal-api-key "$ANTHROPIC_API_KEY"
```

For providers BoxMOT cannot infer automatically, pass the environment variable name explicitly:

```bash
boxmot research --benchmark mot17-ablation --tracker bytetrack \
  --proposal-model custom/provider-model \
  --proposal-api-key "$CUSTOM_PROVIDER_KEY" \
  --proposal-api-key-env CUSTOM_PROVIDER_KEY
```

::: mkdocs-click
    :module: boxmot.engine.cli
    :command: boxmot
    :depth: 1
    :command: research
    :style: table
    :prog_name: boxmot research
