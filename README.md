# Text-to-SQL CLI Agent

An interactive terminal agent that converts natural-language questions into SQLite queries, executes them against the Chinook database, and supports follow-up questions — built on open-source models served by Fireworks AI.

It includes a self-validation harness that scores the agent against the 10 provided dev questions by comparing actual query results to the gold expected results.

## Project Structure

```
.
├── README.md
├── setup.sh                  # Downloads the Chinook SQLite database
├── pyproject.toml
├── uv.lock
├── dev_answers.json          # Agent outputs for the 10 dev questions (deliverable)
├── eval_metrics.json         # Detailed evaluation metrics from the last run
├── src/
│   ├── cli.py                # Interactive CLI entry point
│   ├── agent.py              # TextToSQLAgent: prompt, safety checks, retry loop
│   ├── eval.py               # Evaluation runner (accuracy + latency metrics)
│   └── utils.py              # DB helpers and schema extraction
└── data/
    ├── Chinook.db
    ├── dev_questions.json
    ├── dev_questions_with_answers.json
    └── dev_answers_example.json
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or a manually managed virtualenv
- A Fireworks AI API key ([get one here](https://fireworks.ai))

## Setup

1. Download the Chinook database (if `data/Chinook.db` is not already present):

```bash
./setup.sh
```

2. Install dependencies:

```bash
uv sync
```

3. Set the required environment variable:

```bash
export FIREWORKS_API_KEY=<your-key>
```

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `FIREWORKS_API_KEY` | Yes | Your Fireworks AI API key. The agent exits at startup if it is not set. |
| `FIREWORKS_MODEL` | No | Overrides the model. Defaults to `accounts/fireworks/models/deepseek-v4-flash`. |

## Running the CLI

```bash
uv run cli
```

Or without uv:

```bash
source .venv/bin/activate
python -m src.cli
```

This starts an interactive session:

- Type a natural-language question and press Enter.
- The agent prints the generated SQL, the query results as a table, and telemetry (latency, retries used).
- Ask follow-up questions — the agent keeps the recent conversation, so "now only for Brazil" works after a previous question.
- `/reset` clears the conversation context.
- `exit` or `quit` (or Ctrl+C) ends the session.

Example:

```
Question > What are the top 5 best-selling genres by total sales?

[Generated SQL]
SELECT g.Name, SUM(il.UnitPrice * il.Quantity) AS TotalSales ...

[Query Results (5 rows)]
              Name  TotalSales
              Rock      826.65
             Latin      382.14
             ...

[Telemetry] Retries Used: 0 | Latency: 2.39s
```

## Running the Evaluation

```bash
uv run python -m src.eval
```

To evaluate a different model:

```bash
FIREWORKS_MODEL=accounts/fireworks/models/<model-id> uv run python -m src.eval
```

The runner sends all 10 questions from `data/dev_questions.json` through the agent (conversation context reset per question) and compares each result to the gold `expected_result` in `data/dev_questions_with_answers.json`. Result comparison ignores row order, column order, and column-name aliases, and rounds floats to 2 decimals — but it is strict about the data itself: missing rows, extra rows, or extra columns count as a wrong result.

It writes two files:

- **`dev_answers.json`** — the deliverable format: `{"q_001": {"sql": "...", "answer": "..."}, ...}`
- **`eval_metrics.json`** — summary metrics (accuracy, P50/P95 latency, retries) plus per-question details; wrong results include the model answer, expected answer, and a row-level diff.

Progress and a summary block are printed to the terminal, including per-question status (`CORRECT`, `WRONG RESULT`, or `FAILED`) with diffs for mismatches.

## Results (10 Dev Questions)

Model: `accounts/fireworks/models/deepseek-v4-flash`

| Metric | Value |
|---|---|
| Execution success (SQL ran without error) | 100% (10/10) |
| Result accuracy (rows match gold answer) | 70% (7/10) |
| P50 latency (end-to-end, incl. retries) | 2.39s |
| P95 latency | 4.38s |
| Retries used | 0 |

The P50 latency meets the customer's < 3s interactive target. Note that LLM inference is not fully deterministic even at temperature 0, so accuracy can vary by ±1 question between runs.

### Known Failures

All 3 misses return the **correct data in the wrong shape** — no hallucinated tables, no invalid SQL:

- **q_002** — used `SELECT Album.*`, returning `AlbumId`/`ArtistId` alongside the expected `Title` column.
- **q_005** — returned the employee name as separate `FirstName`/`LastName` columns instead of one concatenated name.
- **q_009** — included an extra `CustomerId` column and split the customer name, where the gold answer expects a single `CustomerName`.

## Design Decisions

- **Schema understanding:** the full schema (tables, columns, types, and foreign-key relationships extracted via SQLite PRAGMAs) is injected into the system prompt at startup. The foreign keys are what let the model write correct JOINs. This fits comfortably in context for Chinook's 11 tables; a much larger database would need per-question schema retrieval instead.
- **Failure handling:** a bounded repair loop (up to 2 retries) feeds the failed SQL and the actual SQLite error back to the model so it can fix the specific mistake. A safety validator blocks all mutating statements (INSERT/UPDATE/DELETE/DROP/...) before execution — the agent is read-only by construction.
- **Structured output over tool-calling:** the model must respond with `{"sql": "..."}` enforced via JSON response format (with a fallback parser for markdown-fenced replies). A single structured call keeps latency and cost down versus a multi-round tool-calling agent, which matters for the < 3s P50 target.
- **Conversation context:** the agent keeps a sliding window of the last 8 messages so CLI users can ask follow-ups. Only successful queries are committed to history, so a bad generation doesn't pollute later turns. The eval resets context per question since dev questions are independent.

## Next Steps

1. **Tune the model and prompt to push accuracy from 70% toward 100%.** Current accuracy with `deepseek-v4-flash` is 70%, and all three failures are result-shape issues rather than wrong data. The first iteration is prompt guidance ("return only the columns the question asks for; never use `SELECT *`; return a person's name as a single full-name column"), then comparing accuracy/latency/cost across Fireworks models to pick the best fit, and if needed fine-tuning on text-to-SQL examples.
2. **Multi-run evaluation** — run the eval suite N times and report mean/variance, since single-run accuracy on 10 questions moves ±10% due to inference nondeterminism.
3. **Larger test set** — 10 questions is enough to iterate but too small to certify quality; expand with more tiers and edge cases (dates, NULLs, empty results).
4. **Cost projection** — at ~30,000 queries/day, benchmark cost per query across candidate models alongside quality.
5. **Schema scalability** — for customer databases with hundreds of tables, replace full-schema injection with retrieval of relevant tables per question.

## AI Assistance Disclosure

AI coding tools were used substantially in this project, via Cursor:

- The initial versions of the CLI (`src/cli.py`), the agent's retry/parsing logic (`src/agent.py`), and the schema-extraction helper (`src/utils.py`) were generated with Gemini 2.5 Flash.
- The evaluation harness (`src/eval.py`) was written and iterated with Claude, including the result-matching logic and the row-level diff output.

I directed the architecture and design decisions (schema-in-prompt, structured JSON output, bounded retry loop, result-match evaluation), reviewed and tested the generated code, used the eval's diff output to diagnose the failure cases, and validated all results against the provided gold answers.
