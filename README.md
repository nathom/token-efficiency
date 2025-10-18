# Code for "Comparing Structured Data Formats for LLMs"

Utilities for measuring how efficiently different tokenizers encode structured data and how legible those structures are to large language models.

## Prerequisites
- Install [uv](https://docs.astral.sh/uv/) and ensure it is on your `PATH`.

All subsequent Python invocations should go through `uv run`.

## Command-Line Interface
The CLI is exposed via the `token-efficiency` console script. View the available subcommands with:

```bash
uv run token-efficiency --help
```

### Generate Token Efficiency Data
Create a dataset that compares tokens-per-node across random data shapes, serialization formats, and tokenizers:

```bash
uv run token-efficiency generate \
  --output data/token_efficiency.json \
  --samples-per-size 4 \
  --size 31 --size 63 --size 127
```

Key options:
- `--tokenizer NAME=repo[@revision]` adds or overrides tokenizer definitions.
- `--size N` repeats to target multiple node counts.
- `--force` discards cached results and regenerates everything.

Generated metadata and samples are stored under `data/` (with cached raw artifacts in `data/cache`).

### Plot Existing Results
Render static plots for both token efficiency and legibility datasets:

```bash
uv run token-efficiency plot \
  --token-efficiency-data data/token_efficiency.json \
  --legibility-data data/legibility.json \
  --output-dir plots
```

This produces heatmaps and comparison bar charts under `plots/token_efficiency/` and `plots/legibility/`.

### Generate Data And Plot In One Step
If you already have legibility results on disk, run a full pipeline:

```bash
uv run token-efficiency generate-and-plot \
  --legibility-data data/legibility.json \
  --output-dir plots
```

### Run The Legibility Benchmark
Evaluate how well a model reproduces structured outputs that were generated with known node counts:

```bash
export OPENROUTER_API_KEY=...

uv run token-efficiency legibility \
  --output data/legibility.json \
  --model deepseek/deepseek-chat \
  --num-trials 25
```

Additional environment variables:
- `OPENROUTER_HTTP_REFERER` (required by OpenRouter usage policy).
- `OPENROUTER_X_TITLE` (recommended to label your traffic).

CLI flags let you adjust input/output node targets, serialization formats, temperature, timeout, and whether to restrict generated data to terminal values (`--terminals-only`).

### Preview A Benchmark Prompt
Inspect the exact prompt sent to the evaluation model:

```bash
uv run token-efficiency sample-prompt 63 5 --format json_min
```

## Data Layout
- `data/token_efficiency.json`: Measurements aggregated by shape, format, tokenizer, and node count.
- `data/legibility.json`: Accuracy metrics returned from the benchmark runner.
- `plots/`: Exported PNG and SVG visualizations, separated into `token_efficiency/` and `legibility/` folders.
- `resources/system_words.txt`: Word list used to synthesize readable identifiers.
