"""Evaluation runner for benchmarking the Text-to-SQL agent.

Runs every dev question through the agent and reports two separate metrics:

- Execution success: the generated SQL ran without an error.
- Result accuracy:   the returned rows match the gold `expected_result`
                     from data/dev_questions_with_answers.json.

Outputs two files:

- dev_answers.json   The deliverable required by the README:
                     {"q_001": {"sql": "...", "answer": "..."}, ...}
- eval_metrics.json  Per-question details plus the summary metrics.

Usage:
    python -m src.eval
    FIREWORKS_MODEL=<model-id> python -m src.eval   # override the model
"""
import json
import os
import sys
from collections import Counter

from src.agent import TextToSQLAgent

QUESTIONS_FILE = "data/dev_questions.json"
GOLD_FILE = "data/dev_questions_with_answers.json"
ANSWERS_FILE = "dev_answers.json"
METRICS_FILE = "eval_metrics.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def calculate_percentile(values, percentile):
    """Percentile with linear interpolation (same method as numpy's default)."""
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower

    result = sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
    return round(result, 3)


def normalize_value(value):
    """Stringify a cell value, rounding floats to 2 decimals to absorb
    floating-point noise (49.620000001 -> "49.62", 4.0 -> "4")."""
    if isinstance(value, float):
        value = round(value, 2)
        if value.is_integer():
            value = int(value)
    return str(value)


def comparable_rows(actual_rows, expected_rows):
    """Normalize both result sets into comparable tuples.

    If both results use the same column names, values are aligned by column,
    so transposed data (e.g. FirstName/LastName swapped) is caught. If the
    column names differ (the model chose different aliases), fall back to
    comparing each row's sorted set of values instead.
    """
    same_columns = (
        actual_rows and expected_rows
        and set(actual_rows[0].keys()) == set(expected_rows[0].keys())
    )
    if same_columns:
        columns = sorted(expected_rows[0].keys())
        actual = [tuple(normalize_value(row[c]) for c in columns) for row in actual_rows]
        expected = [tuple(normalize_value(row[c]) for c in columns) for row in expected_rows]
    else:
        actual = [tuple(sorted(normalize_value(v) for v in row.values())) for row in actual_rows]
        expected = [tuple(sorted(normalize_value(v) for v in row.values())) for row in expected_rows]
    return actual, expected


def results_match(actual_rows, expected_rows):
    """Compare query results ignoring row order (see comparable_rows)."""
    actual, expected = comparable_rows(actual_rows, expected_rows)
    return sorted(actual) == sorted(expected)


def describe_row_diff(actual_rows, expected_rows, max_rows=5):
    """Human-readable lines describing how the model's rows differ from gold."""

    def preview(rows):
        shown = "; ".join(str(row) for row in rows[:max_rows])
        if len(rows) > max_rows:
            shown += f"; ... ({len(rows) - max_rows} more)"
        return shown

    actual, expected = comparable_rows(actual_rows, expected_rows)
    missing = list((Counter(expected) - Counter(actual)).elements())
    extra = list((Counter(actual) - Counter(expected)).elements())

    lines = []
    if missing:
        lines.append(f"Missing rows (expected, not returned): {preview(missing)}")
    if extra:
        lines.append(f"Wrong/extra rows returned: {preview(extra)}")
    if not lines:
        lines.append("Same values but assigned to the wrong columns (data transposed).")
    return lines


def summarize_rows(rows, max_rows=10):
    """Build the human-readable answer string for dev_answers.json."""
    if not rows:
        return "No rows returned."

    row_texts = []
    for row in rows[:max_rows]:
        row_texts.append(", ".join(f"{column}: {value}" for column, value in row.items()))

    summary = "; ".join(row_texts)
    if len(rows) > max_rows:
        summary += f"; ... ({len(rows) - max_rows} more rows)"
    return summary


def run_evaluation(model_id=None, questions_file=QUESTIONS_FILE, gold_file=GOLD_FILE,
                   answers_file=ANSWERS_FILE, metrics_file=METRICS_FILE):
    if not os.path.exists(questions_file):
        print(f"Error: questions file not found: {questions_file}")
        sys.exit(1)

    questions = load_json(questions_file)

    # Gold answers are optional; without them we can only measure execution success.
    gold_by_id = {}
    if os.path.exists(gold_file):
        gold_by_id = {item["id"]: item for item in load_json(gold_file)}

    agent = TextToSQLAgent(model_id=model_id)

    print("\nStarting Evaluation...")
    print(f"Model ID: {agent.model_id}")
    print("-" * 60)

    answers = {}          # deliverable: {question_id: {"sql", "answer"}}
    per_question = []     # detailed record for metrics_file
    latencies = []
    executed_count = 0
    correct_count = 0
    gold_count = 0
    total_retries = 0

    for number, item in enumerate(questions, start=1):
        question = item.get("question") or item.get("prompt") or ""
        question_id = item.get("id", str(number))

        # Dev questions are independent, so clear conversation memory each time.
        agent.reset_conversation()
        response = agent.query(question)

        executed = response["success"]
        rows = response["results"] if executed else []
        latencies.append(response["latency_s"])
        total_retries += response["retries"]
        if executed:
            executed_count += 1

        # Compare against the gold expected result when we have one.
        expected = gold_by_id.get(question_id, {}).get("expected_result")
        if expected is None:
            matches_gold = None
        else:
            gold_count += 1
            matches_gold = executed and results_match(rows, expected)
            if matches_gold:
                correct_count += 1

        if executed:
            answer_text = summarize_rows(rows)
        else:
            answer_text = f"Query failed: {response['error']}"
        answers[question_id] = {"sql": response["sql"], "answer": answer_text}

        question_record = {
            "id": question_id,
            "question": question,
            "sql": response["sql"],
            "executed": executed,
            "matches_gold": matches_gold,
            "error": response["error"],
            "retries": response["retries"],
            "latency_s": response["latency_s"],
            "row_count": len(rows),
        }

        # On a wrong result, record what the model returned vs what was expected.
        if matches_gold is False:
            question_record["model_answer"] = answer_text
            question_record["expected_answer"] = summarize_rows(expected)
            if executed:
                question_record["diff"] = describe_row_diff(rows, expected)

        per_question.append(question_record)

        if matches_gold is None:
            status = "EXECUTED" if executed else "FAILED"
        else:
            status = "CORRECT" if matches_gold else ("WRONG RESULT" if executed else "FAILED")
        print(f"Question {number}/{len(questions)} [{question_id}]")
        print(f"Status: {status} | Latency: {response['latency_s']}s | Retries: {response['retries']}")
        if matches_gold is False:
            print(f"  Model answer:    {answer_text}")
            print(f"  Expected answer: {summarize_rows(expected)}")
            if executed:
                for line in describe_row_diff(rows, expected):
                    print(f"  {line}")
        print("-" * 30)

    total = len(questions)
    execution_success_pct = round(executed_count / total * 100, 2) if total else 0.0
    accuracy_pct = round(correct_count / gold_count * 100, 2) if gold_count else None
    p50_latency = calculate_percentile(latencies, 50)
    p95_latency = calculate_percentile(latencies, 95)

    summary = {
        "model_id": agent.model_id,
        "total_questions": total,
        "executed_ok": executed_count,
        "execution_success_pct": execution_success_pct,
        "correct_results": correct_count,
        "gold_questions": gold_count,
        "accuracy_pct": accuracy_pct,
        "p50_latency_s": p50_latency,
        "p95_latency_s": p95_latency,
        "total_retries_used": total_retries,
        "per_question": per_question,
    }

    save_json(answers, answers_file)
    save_json(summary, metrics_file)

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print(f"Model ID: {agent.model_id}")
    print(f"Execution success: {execution_success_pct}% ({executed_count}/{total})")
    if accuracy_pct is None:
        print("Result accuracy: n/a (no gold answers found)")
    else:
        print(f"Result accuracy: {accuracy_pct}% ({correct_count}/{gold_count})")
    print(f"P50 Latency: {p50_latency}s")
    print(f"P95 Latency: {p95_latency}s")
    print(f"Total Retries Used: {total_retries}")
    print(f"Answers saved to: {answers_file}")
    print(f"Metrics saved to: {metrics_file}")
    print("=" * 60)

    return summary


def main():
    run_evaluation(model_id=os.getenv("FIREWORKS_MODEL"))


if __name__ == "__main__":
    main()
