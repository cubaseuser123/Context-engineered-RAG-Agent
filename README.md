# Context Engineering, Measured

A comparison of naive RAG and a context-engineered policy assistant, using a
synthetic company corpus, the same generator, and the same evaluation questions.
The engineered arm adds routing, standalone search queries, bounded user memory,
retrieval deduplication, and context budgets.

## Current results

The latest retained comparison is `engineered-b5000-20261112_233129`: 30 scenarios,
52 turns per arm, 104 saved answers, and 312 usable current-version judge grades.
Each answer has a live Langfuse experiment item and trace ID.

| Measurement | Naive | Engineered |
|---|---:|---:|
| Mean total agent chat tokens | 4,549.7 | 4,215.0 |
| Mean synthesis tokens | 4,549.7 | 2,895.7 |
| Mean measured turn time | 10.48 s | 13.80 s |
| Automated answer correctness | 49/52 | 47/52 |
| Automated groundedness | 46/52 | 47/52 |
| Automated citation support | 46/52 | 46/52 |

Total chat-token usage fell **7.4%**, but measured latency increased and overall
answer-quality improvement was not established. Timing includes provider retries
and harness overhead. Judge scores are provisional: review found both false
negatives on valid refusals and a missed unsupported accounting claim. A small,
assisted annotation review is preserved; it is not independent calibration.

- [Complete report](reports/engineered-b5000-20261112_233129.md)
- [Baseline experiment](https://us.cloud.langfuse.com/project/cmw7csmg300xlad0jlzv2wohn/experiments/results?baseline=ae878644e65170f1)
- [Engineered experiment](https://us.cloud.langfuse.com/project/cmw7csmg300xlad0jlzv2wohn/experiments/results?baseline=fd94d3e326944fb1)

## Layout

| Path | Purpose |
|---|---|
| `src/` | LangGraph agent, providers, retrieval, memory, context assembly, tracing |
| `corpus/documents/` | 52 synthetic policy and distractor documents |
| `evals/` | Versioned scenarios, runner, metrics, judges, calibration and reports |
| `evals/datasets/` | Benchmark and assisted human-review records |
| `scripts/` | Ingestion, evaluation, live Langfuse experiments and score verification |
| `tests/` | Offline regression tests and an explicitly selected live smoke test |
| `runs/` | Retained answers, manifests, append-only grades and scorecards |
| `reports/` | Reports supporting the retained comparisons and review |

## Setup

Python 3.11 or later is required. PowerShell commands:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Fill `.env` with Groq generation/judge keys, a Google embedding key, and Pinecone
configuration. Add Langfuse keys for live traces and experiments. Secrets, local
SQLite memory, provider caches and tool settings are ignored by Git.

## Offline verification

```powershell
$env:LANGFUSE_TRACING_ENABLED = "false"
.venv/Scripts/python.exe -m pytest
```

The default suite excludes live-provider tests. It does not regenerate answers
or spend model quota.

## Live evaluation

The first ingestion replaces the configured Pinecone `policies` namespace and
records document/chunk counts and retrieval probes in `corpus/ingest_report.json`.
It is not needed to read the saved results.

```powershell
$env:LANGFUSE_TRACING_ENABLED = "true"
.venv/Scripts/python.exe -m scripts.ingest_corpus
.venv/Scripts/python.exe -m scripts.langfuse_experiment_runner --arm both
```

Resume the existing comparison without regenerating its completed scenarios:

```powershell
.venv/Scripts/python.exe -m scripts.langfuse_experiment_runner --arm both --run-id engineered-b5000-20261112_233129 --resume
.venv/Scripts/python.exe -m scripts.verify_langfuse_experiments engineered-b5000-20261112_233129 --require-scores
```

The Langfuse runner checkpoints each completed grade and reuses score IDs on
resume. Grades are append-only: select the current version for each metric using
`evals.judges.judge_version`, rather than counting every historical line.

`scripts.run_eval` also supports local record/report generation without requiring
Langfuse. `scripts.run_sweep` and `scripts.calibrate_judge` provide budget sweeps
and human-label calibration. Neither is part of the default offline test run.

## Evidence and limitations

Both arms share the generator, synthesis prompt, index, embedding model and
benchmark. Total agent chat tokens include synthesis and auxiliary model calls;
they exclude embeddings and judge calls. Cost figures are estimates, not billing.
Reports include confidence intervals and paired comparisons.

The benchmark, saved records, judge versions and assisted annotation metadata
are retained alongside the reports so individual results can be inspected.
