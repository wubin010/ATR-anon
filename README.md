# ATRBench — Ask-to-Remember Benchmark

ATRBench measures whether an LLM agent, acting for a user across multiple
sessions, **asks to remember** the user's standing rules (durable
preferences and constraints) and then **applies** them on later tasks
where the rule is decision-relevant but unstated. Each persona contributes
one episode: a sequence of *learning sessions* (where the rule can be
elicited) followed by *test sessions* (where a rule-aware agent succeeds
and a rule-blind agent does not).

This repository contains the benchmark data (20 personas, 284 evaluated
standing-rule test items, 568 episode-selected learning sessions, 74 tools
across 6 domains), the construction pipeline, the trajectory runner, and the
deterministic evaluator. The per-persona directories also include the larger
curated rule/session pools used to compose those episodes.

## Repository layout

```
lib/         LLM client (per-provider adapters) + shared compare/token utils
datagen/     benchmark construction pipeline (rules → test/learning sessions → episode)
runner/      trajectory execution: agent-under-test, user simulator, memory layer
evaluator/   scoring — deterministic tool-call matching against gold actions
tools/       trajectory pretty-printing / live trace writing
ontology/    tool definitions (tools.yaml)
scripts/     main-table aggregation, appendix prompt rendering
docs/        REF_TYPE_SCHEMA.md — reference-object attribute contract
data/
  personas/                     20 personas: rules.json, structured.json,
                                test_sessions/, learning_sessions/, episodes/
  few_shot_pool/                datagen few-shot exemplars
  persona_selection_formal20.json
```

## Installation

The project uses [uv](https://docs.astral.sh/uv/). Python ≥ 3.10.

```bash
uv sync
```

## Configuration

Model access is configured entirely through environment variables — no keys
are stored in the code. Copy `.env.example` to `.env`, fill in the keys for
the providers you intend to run, then load it:

```bash
cp .env.example .env      # edit in your keys
set -a && . ./.env && set +a
```

Each provider has a `<PROVIDER>_API_KEY` and an optional `<PROVIDER>_BASE_URL`
that defaults to the vendor's official endpoint. To route every model through
a single OpenAI-compatible gateway, set each `*_BASE_URL` to that gateway.
See `.env.example` for the full list and defaults.

## Running the benchmark

The benchmark evaluates 8 agent models — GPT-5.4, Claude Opus 4.7, Gemini 3
Flash Preview, Gemini 3.1 Pro Preview, Qwen3.6-Plus, MiniMax M2.7, DeepSeek
V4 Pro, DeepSeek V4 Flash — across 4 variants and the 20-persona cohort
(8 × 4 × 20 = 640 cells), single-trial. Agent variants:

- `default` — no memory guidance (baseline)
- `atr` — Ask-to-Remember scaffold
- `always_ask` — ask on every learning session (upper-bound ask cost)
- `oracle_target` — rule injected directly, skipping learning (the reported
  `oracle`; upper-bound accuracy)

(`oracle_full`, which injects all rules, is an extra internal diagnostic
outside the reported four variants.)

The user simulator and Classifier use GPT-5.4, the Router uses Gemini 3 Flash
Preview, and all cells share the same raw-context layer, retry policy, and
evaluator — only the agent model varies.

**Single cell** (one persona/variant/model):

```bash
uv run python -m runner.pipeline \
  --episode data/personas/anna_strahan/episodes/anna_strahan_seed000.json \
  --variant atr --model gpt-5.4
```

**Full sweep**:

```bash
uv run python -m runner.pipeline \
  --personas anna_strahan cassandra_tovar \
  --seeds 0 \
  --variants default atr always_ask oracle_target \
  --models gpt-5.4 gemini-3-flash-preview
```

`--reasoning {on,off}` toggles the agent's thinking mode where the provider
exposes one. Trajectories, traces, and per-cell config land under
`outputs/<persona>/<short_episode>/<cell>/` (override the root with
`--outputs-root` or `ATR_OUTPUTS_ROOT`).

**Evaluation** (deterministic; no model calls) — run with the same sweep
dimensions to score the produced trajectories:

```bash
uv run python -m evaluator.pipeline \
  --personas anna_strahan --seeds 0 \
  --variants default atr always_ask oracle_target \
  --models gpt-5.4
```

**Main tables**:

```bash
uv run python scripts/build_main_tables.py
```

## Data construction (datagen)

`datagen/` is the full construction pipeline (rule generation + QC, test- and
learning-session synthesis, episode composition). It is included for
transparency and reproducibility of the *method*. The released dataset under
`data/personas/` is the curated output; each `episodes/<pid>_seed000.json` is
self-contained (it embeds the persona, rules, and selected sessions), so the
runner and evaluator do **not** need to re-run datagen.

The standalone `rules.json` and `learning_sessions/` directories are the
curated pools before episode sampling (340 candidate rules and 880 learning
sessions). The released benchmark episodes under `episodes/` embed the
evaluated subset used by the runner and evaluator (284 test items and 568
learning sessions).

The raw persona source files (`raw.json`) and intermediate QC artifacts are
intentionally excluded from the standalone pool directories (see `.gitignore`),
so datagen cannot be re-run from scratch in this snapshot — but every stage's
code and prompts are present for review. Each released episode embeds the
persona fields it needs at runtime.

## Data

Personas are derived from the public **NVIDIA Nemotron-Personas-USA** dataset;
the selection criteria are recorded in `data/persona_selection_formal20.json`.
Each persona's `raw_persona` retains its source `nemotron_uuid` for
provenance. Persona content is synthetic (not real individuals). Any reuse of
the persona data should follow the Nemotron-Personas license.

## License

Code will be released under a permissive open-source license upon
de-anonymization. Persona data derives from NVIDIA Nemotron-Personas and
remains subject to that dataset's license.
