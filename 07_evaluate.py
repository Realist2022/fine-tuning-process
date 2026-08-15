"""Predicted-vs-expected reporting for 07_evaluate.py.

Styled to match the Tester/evaluate reference: a cumulative-error trend chart
with a 95% confidence band, followed by an expected-vs-predicted scatter with a
dashed y = x line and green/orange/red agreement colours.

Two point-building modes:

1. Default — every numeric leaf field shared by the expected and predicted
   JSON becomes one point (field path + source row on hover).
2. score_fn — pass a callable ``score_fn(task, extraction) -> float | None``
   (e.g. your deterministic 3-pillar scorer) and you get one point per
   CV-JD pair instead: teacher score on x, student score on y.

Integration (three small edits to 07_evaluate.py):

    from scatter_report import build_scatter_points, report_figures, save_pair_records

    # 1. before the eval loop
    pair_records: list[dict[str, Any]] = []

    # 2. inside the loop, right after schema_valid[task] += 1
    pair_records.append(
        {
            "task": task,
            "source_row": example["source_row"],
            "expected": validated_expected,
            "prediction": validated_prediction,
        }
    )

    # 3. after the loop
    save_pair_records(pair_records, "evaluation_pairs.jsonl")
    trend_figure, scatter_figure = report_figures(
        build_scatter_points(pair_records), model_name=resolved_model_name
    )
    # embed alongside the other figures in the HTML document:
    #   <section>{trend_figure.to_html(full_html=False, include_plotlyjs=False)}</section>
    #   <section>{scatter_figure.to_html(full_html=False, include_plotlyjs=False)}</section>
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable, cast

import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, ValidationError

from schemas import JobRequirementsOutput, OverallExperienceOutput, SkillMatcherOutput

ScoreFn = Callable[[str, dict[str, Any]], float | None]

COLOR_DISCRETE_MAP = {"green": "green", "orange": "orange", "red": "red"}
CATEGORY_ORDER = ["green", "orange", "red"]
ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>\n"
END_OF_TURN = "<|eot_id|>"
TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "job_requirements": JobRequirementsOutput,
    "skill_matcher": SkillMatcherOutput,
    "overall_experience": OverallExperienceOutput,
}

# Relative-error thresholds. The reference script also had absolute dollar
# floors ($40/$80); those are unit-specific, so pass absolute_thresholds
# explicitly only if your values share a single meaningful scale.
RELATIVE_THRESHOLDS = (0.20, 0.40)


def _split_example(text: str) -> tuple[str, dict[str, Any]]:
    if text.count(ASSISTANT_MARKER) != 1:
        raise ValueError("expected exactly one assistant marker")
    prompt_prefix, expected_text = text.split(ASSISTANT_MARKER, maxsplit=1)
    if not expected_text.endswith(END_OF_TURN):
        raise ValueError("expected assistant output to end with the end-of-turn token")
    prompt = prompt_prefix + ASSISTANT_MARKER
    expected = json.loads(expected_text[: -len(END_OF_TURN)])
    if not isinstance(expected, dict):
        raise ValueError("expected assistant output must be a JSON object")
    return prompt, expected


def _parse_generated_json(text: str) -> dict[str, Any]:
    cleaned = text.split(END_OF_TURN, maxsplit=1)[0].strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    parsed = json.loads(cleaned.strip())
    if not isinstance(parsed, dict):
        raise ValueError("generated output must be a JSON object")
    return parsed


def _flatten_values(value: Any, path: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            item_path: item_value
            for key, child in value.items()
            for item_path, item_value in _flatten_values(child, f"{path}.{key}").items()
        }
    if isinstance(value, list):
        return {
            item_path: item_value
            for index, child in enumerate(value)
            for item_path, item_value in _flatten_values(child, f"{path}[{index}]").items()
        }
    return {path: value}


def _field_accuracy(expected: dict[str, Any], prediction: dict[str, Any]) -> float:
    expected_values = _flatten_values(expected)
    if not expected_values:
        return float(expected == prediction)
    prediction_values = _flatten_values(prediction)
    matches = sum(
        prediction_values.get(path) == value for path, value in expected_values.items()
    )
    return matches / len(expected_values)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "model"


def _flatten_numeric(value: Any, path: str = "$") -> dict[str, float]:
    """Flatten nested JSON to {path: numeric_value}, skipping bools and strings."""
    if isinstance(value, bool):  # bool is an int subclass — exclude explicitly
        return {}
    if isinstance(value, (int, float)):
        return {path: float(value)}
    if isinstance(value, dict):
        return {
            leaf_path: leaf
            for key, child in value.items()
            for leaf_path, leaf in _flatten_numeric(child, f"{path}.{key}").items()
        }
    if isinstance(value, list):
        return {
            leaf_path: leaf
            for index, child in enumerate(value)
            for leaf_path, leaf in _flatten_numeric(child, f"{path}[{index}]").items()
        }
    return {}


def color_for(
    error: float,
    truth: float,
    relative_thresholds: tuple[float, float] = RELATIVE_THRESHOLDS,
    absolute_thresholds: tuple[float, float] | None = None,
) -> str:
    """Green / orange / red band for one point, mirroring the reference logic."""
    near, mid = relative_thresholds
    relative = error / abs(truth) if truth else (0.0 if error == 0 else math.inf)
    if absolute_thresholds is not None:
        if error < absolute_thresholds[0] or relative < near:
            return "green"
        if error < absolute_thresholds[1] or relative < mid:
            return "orange"
        return "red"
    if relative < near:
        return "green"
    if relative < mid:
        return "orange"
    return "red"


def build_scatter_points(
    pair_records: list[dict[str, Any]],
    score_fn: ScoreFn | None = None,
    relative_thresholds: tuple[float, float] = RELATIVE_THRESHOLDS,
    absolute_thresholds: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """Convert (expected, prediction) pairs into plottable points.

    Each record needs: task, source_row, expected, prediction.
    Returns points with: truth, guess, error, color, task, source_row, label.
    """
    points: list[dict[str, Any]] = []

    def add(truth: float, guess: float, task: str, source_row: Any, label: str) -> None:
        error = abs(guess - truth)
        points.append(
            {
                "truth": truth,
                "guess": guess,
                "error": error,
                "color": color_for(
                    error, truth, relative_thresholds, absolute_thresholds
                ),
                "task": task,
                "source_row": source_row,
                "label": label,
            }
        )

    for record in pair_records:
        task = record["task"]
        source_row = record.get("source_row")
        if score_fn is not None:
            expected_score = score_fn(task, record["expected"])
            predicted_score = score_fn(task, record["prediction"])
            if expected_score is None or predicted_score is None:
                continue
            add(
                float(expected_score),
                float(predicted_score),
                task,
                source_row,
                "pair score",
            )
            continue
        expected_leaves = _flatten_numeric(record["expected"])
        predicted_leaves = _flatten_numeric(record["prediction"])
        for path, expected_value in expected_leaves.items():
            if path in predicted_leaves:
                add(expected_value, predicted_leaves[path], task, source_row, path)
    return points


def _metrics(points: list[dict[str, Any]]) -> dict[str, float]:
    """MAE, MSE and r² — same definitions as sklearn, without the dependency."""
    count = len(points)
    errors = [point["guess"] - point["truth"] for point in points]
    mae = sum(abs(error) for error in errors) / count
    mse = sum(error * error for error in errors) / count
    mean_truth = sum(point["truth"] for point in points) / count
    ss_total = sum((point["truth"] - mean_truth) ** 2 for point in points)
    ss_residual = sum(error * error for error in errors)
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")
    return {"mae": mae, "mse": mse, "r2": r_squared * 100}


def create_error_trend_chart(
    points: list[dict[str, Any]],
    model_name: str = "model",
    value_label: str = "",
    width: int | None = 1000,
) -> go.Figure:
    """Cumulative average absolute error with a 95% confidence band."""
    if not points:
        raise ValueError("No comparable points found for the error trend chart")

    errors = [point["error"] for point in points]
    count = len(errors)
    x = list(range(1, count + 1))
    running_means = [total / i for total, i in zip(accumulate(errors), x)]
    running_squares = list(accumulate(error * error for error in errors))
    running_stds = [
        math.sqrt(max(square_sum / i - mean**2, 0.0)) if i > 1 else 0.0
        for i, square_sum, mean in zip(x, running_squares, running_means)
    ]
    confidence = [
        1.96 * (deviation / math.sqrt(i)) if i > 1 else 0.0
        for i, deviation in zip(x, running_stds)
    ]
    upper = [mean + band for mean, band in zip(running_means, confidence)]
    lower = [mean - band for mean, band in zip(running_means, confidence)]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(128,128,128,0.2)",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            showlegend=False,
            name="95% CI",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=running_means,
            mode="lines",
            line={"width": 3, "color": "firebrick"},
            name="Cumulative avg error",
            customdata=list(zip(confidence)),
            hovertemplate=(
                "n=%{x}<br>"
                "Avg error=%{y:,.3f}<br>"
                "±95% CI=%{customdata[0]:,.3f}<extra></extra>"
            ),
        )
    )

    suffix = f" ({value_label})" if value_label else ""
    figure.update_layout(
        title=(
            f"{model_name} error trend: "
            f"{running_means[-1]:,.3f} ± {confidence[-1]:,.3f}{suffix}"
        ),
        xaxis_title="Number of comparisons",
        yaxis_title=f"Average absolute error{suffix}",
        width=width,
        height=360,
        template="plotly_white",
        showlegend=False,
    )
    return figure


def create_prediction_scatter(
    points: list[dict[str, Any]],
    model_name: str = "model",
    value_label: str = "value",
    width: int | None = 1000,
    height: int = 800,
) -> go.Figure:
    """Expected-vs-predicted scatter with y = x line and agreement colours."""
    if not points:
        raise ValueError("No comparable numeric points found for the scatter plot")

    scores = _metrics(points)
    title = (
        f"{model_name} results<br>"
        f"<b>Error:</b> {scores['mae']:,.3f} "
        f"<b>MSE:</b> {scores['mse']:,.3f} "
        f"<b>r²:</b> {scores['r2']:.1f}%"
    )
    hovers = [
        f"{point['task']} · {point['label']}"
        f"<br>Predicted={point['guess']:,.3f} Expected={point['truth']:,.3f}"
        f"<br>source row {point['source_row']}"
        for point in points
    ]

    figure = px.scatter(
        x=[point["truth"] for point in points],
        y=[point["guess"] for point in points],
        color=[point["color"] for point in points],
        color_discrete_map=COLOR_DISCRETE_MAP,
        category_orders={"color": CATEGORY_ORDER},
        title=title,
        width=width,
        height=height,
    )

    # One colour category = one trace, so re-attach hover text per trace.
    for trace in cast(tuple[go.Scatter, ...], figure.data):
        trace.customdata = [
            [hover]
            for hover, point in zip(hovers, points)
            if point["color"] == trace.name
        ]
        trace.hovertemplate = "%{customdata[0]}<extra></extra>"
        trace.update(marker_size=6)

    axis_max = max(
        max(point["truth"] for point in points),
        max(point["guess"] for point in points),
    )
    axis_min = min(
        0.0,
        min(point["truth"] for point in points),
        min(point["guess"] for point in points),
    )
    figure.add_trace(
        go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode="lines",
            line={"width": 2, "dash": "dash", "color": "deepskyblue"},
            name="y = x",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.update_xaxes(range=[axis_min, axis_max])
    figure.update_yaxes(range=[axis_min, axis_max])
    figure.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title=f"Expected {value_label}",
        yaxis_title=f"Predicted {value_label}",
    )
    return figure


def report_figures(
    points: list[dict[str, Any]],
    model_name: str = "model",
    value_label: str = "value",
    width: int | None = 1000,
) -> tuple[go.Figure, go.Figure]:
    """Both charts in reference order: error trend first, then the scatter."""
    trend = create_error_trend_chart(points, model_name, width=width)
    scatter = create_prediction_scatter(points, model_name, value_label, width=width)
    return trend, scatter


def show_report(
    points: list[dict[str, Any]],
    model_name: str = "model",
    value_label: str = "value",
) -> None:
    """Notebook-style display, matching Tester.report()."""
    trend, scatter = report_figures(points, model_name, value_label)
    trend.show()
    scatter.show()


def save_pair_records(pair_records: list[dict[str, Any]], path: str | Path) -> Path:
    """Persist pairs so charts can be rebuilt without re-running inference."""
    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as outfile:
        for record in pair_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def load_pair_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result["model_name"], str) or not isinstance(
            result["tasks"], dict
        ):
            raise TypeError("model_name and tasks have invalid types")
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid evaluation result file {path}: {error}") from error
    return result


def create_comparison_report(
    result_files: list[str | Path],
    output_file: str | Path = "evaluation_report.html",
    show_plot: bool = True,
    pair_records: list[dict[str, Any]] | None = None,
) -> Path:
    if not result_files:
        raise ValueError("At least one evaluation result file is required")

    results = [_load_result(Path(path)) for path in result_files]
    task_rows: list[dict[str, str | float]] = []
    for result in results:
        for task, metrics in result["tasks"].items():
            for metric, label in (
                ("schema_valid_rate", "Schema valid"),
                ("field_accuracy", "Field accuracy"),
                ("exact_match_rate", "Exact match"),
            ):
                task_rows.append(
                    {
                        "Model": result["model_name"],
                        "Task": task.replace("_", " ").title(),
                        "Metric": label,
                        "Score": float(metrics[metric]),
                    }
                )

    task_figure = px.bar(
        task_rows,
        x="Task",
        y="Score",
        color="Model",
        facet_row="Metric",
        barmode="group",
        range_y=[0, 1],
        title="Held-out performance by task",
        labels={"Score": "Rate"},
    )
    task_figure.update_yaxes(tickformat=".0%")
    task_figure.for_each_annotation(
        lambda annotation: annotation.update(text=annotation.text.split("=")[-1])
    )
    task_figure.update_layout(height=850, legend_title_text="Model")

    overall_figure = go.Figure()
    for result in results:
        overall = result["overall"]
        overall_figure.add_trace(
            go.Bar(
                name=result["model_name"],
                x=["Schema valid", "Field accuracy", "Exact match"],
                y=[
                    overall["schema_valid_rate"],
                    overall["field_accuracy"],
                    overall["exact_match_rate"],
                ],
                texttemplate="%{y:.1%}",
                textposition="outside",
            )
        )
    overall_figure.update_layout(
        title="Overall held-out comparison",
        barmode="group",
        yaxis={"range": [0, 1.05], "tickformat": ".0%", "title": "Rate"},
        legend_title_text="Model",
        height=480,
    )

    prediction_sections = ""
    if pair_records:
        points = build_scatter_points(pair_records)
        if points:
            trend_figure, scatter_figure = report_figures(
                points, model_name=results[0]["model_name"]
            )
            prediction_sections = (
                "<section>"
                + trend_figure.to_html(full_html=False, include_plotlyjs=False)
                + "</section><section>"
                + scatter_figure.to_html(full_html=False, include_plotlyjs=False)
                + "</section>"
            )

    report_path = Path(output_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_names = ", ".join(html.escape(result["model_name"]) for result in results)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model evaluation report</title>
  <style>
    body {{ margin: 0; background: #f4f6f8; color: #17212b; font-family: sans-serif; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px; }}
    h1 {{ margin-bottom: 4px; }}
    p {{ color: #52606d; margin-top: 0; }}
    section {{ margin-top: 24px; background: white; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <main>
    <h1>Model evaluation report</h1>
    <p>{model_names}</p>
    <section>{overall_figure.to_html(full_html=False, include_plotlyjs="cdn")}</section>
    <section>{task_figure.to_html(full_html=False, include_plotlyjs=False)}</section>
    {prediction_sections}
  </main>
</body>
</html>
"""
    report_path.write_text(document, encoding="utf-8")
    print(f"Interactive report: {report_path}")
    if show_plot:
        webbrowser.open(report_path.resolve().as_uri())
    return report_path


def _load_examples(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as infile:
        for row_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task = row["task"]
                text = row["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid evaluation record at {dataset_path}:{row_number}: {error}"
                ) from error
            if task not in TASK_SCHEMAS:
                raise ValueError(f"Unknown task at {dataset_path}:{row_number}: {task}")
            if not isinstance(text, str):
                raise ValueError(f"Invalid text at {dataset_path}:{row_number}")
            prompt, expected = _split_example(text)
            examples.append(
                {
                    "row_number": row_number,
                    "task": task,
                    "source_row": row.get("source_row"),
                    "prompt": prompt,
                    "expected": expected,
                }
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No evaluation records found in {dataset_path}")
    return examples


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_seq_length: int,
    max_new_tokens: int,
) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    available_tokens = max_seq_length - prompt_length
    if available_tokens < 1:
        raise ValueError(
            f"prompt has {prompt_length} tokens, exceeding the "
            f"{max_seq_length}-token context"
        )
    end_of_turn_id = tokenizer.convert_tokens_to_ids(END_OF_TURN)
    eos_token_id = (
        end_of_turn_id
        if isinstance(end_of_turn_id, int) and end_of_turn_id >= 0
        else tokenizer.eos_token_id
    )
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, available_tokens),
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=False)


def evaluate_model(
    model_path: str = "outputs/checkpoint-200",
    model_name: str | None = None,
    dataset_file: str = "test_semantic.jsonl",
    failures_file: str = "evaluation_failures.jsonl",
    results_file: str | None = None,
    report_file: str | None = None,
    pairs_file: str | None = None,
    compare_results_files: tuple[str, ...] = (),
    max_seq_length: int = 4096,
    max_new_tokens: int = 1024,
    limit: int | None = None,
    show_plot: bool = True,
) -> dict[str, Any]:
    dataset_path = Path(dataset_file)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    try:
        from unsloth import FastLanguageModel
    except (ImportError, NotImplementedError) as error:
        raise RuntimeError(
            "Evaluation requires Unsloth and a GPU accelerator visible to PyTorch."
        ) from error

    examples = _load_examples(dataset_path, limit)
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    totals: Counter[str] = Counter()
    schema_valid: Counter[str] = Counter()
    exact_matches: Counter[str] = Counter()
    field_score_sums: defaultdict[str, float] = defaultdict(float)
    failures: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []

    for index, example in enumerate(examples, start=1):
        task = example["task"]
        totals[task] += 1
        failure_reason = ""
        generated = ""
        prediction: dict[str, Any] | None = None
        try:
            generated = _generate(
                model,
                tokenizer,
                example["prompt"],
                max_seq_length,
                max_new_tokens,
            )
            prediction = _parse_generated_json(generated)
            schema = TASK_SCHEMAS[task]
            validated_prediction = schema.model_validate(prediction).model_dump()
            validated_expected = schema.model_validate(example["expected"]).model_dump()
            schema_valid[task] += 1
            pair_records.append(
                {
                    "task": task,
                    "source_row": example["source_row"],
                    "expected": validated_expected,
                    "prediction": validated_prediction,
                }
            )
            field_score_sums[task] += _field_accuracy(
                validated_expected, validated_prediction
            )
            if validated_prediction == validated_expected:
                exact_matches[task] += 1
            else:
                failure_reason = "structured_mismatch"
        except (json.JSONDecodeError, RuntimeError, ValueError, ValidationError) as error:
            failure_reason = f"invalid_output: {error}"

        if failure_reason:
            failures.append(
                {
                    "row": example["row_number"],
                    "source_row": example["source_row"],
                    "task": task,
                    "reason": failure_reason,
                    "expected": example["expected"],
                    "prediction": prediction,
                    "generated_text": generated,
                }
            )
        print(f"Evaluated {index}/{len(examples)}", end="\r")

    failure_path = Path(failures_file)
    with failure_path.open("w", encoding="utf-8") as outfile:
        for failure in failures:
            outfile.write(json.dumps(failure, ensure_ascii=False) + "\n")

    task_results: dict[str, dict[str, int | float]] = {}
    print("\n\nEvaluation results")
    for task in TASK_SCHEMAS:
        if totals[task] == 0:
            continue
        task_results[task] = {
            "total": totals[task],
            "schema_valid": schema_valid[task],
            "exact_match": exact_matches[task],
            "schema_valid_rate": schema_valid[task] / totals[task],
            "field_accuracy": field_score_sums[task] / totals[task],
            "exact_match_rate": exact_matches[task] / totals[task],
        }
        print(
            f"{task}: schema valid {schema_valid[task]}/{totals[task]} "
            f"({schema_valid[task] / totals[task]:.1%}), field accuracy "
            f"{field_score_sums[task] / totals[task]:.1%}, exact match "
            f"{exact_matches[task]}/{totals[task]} "
            f"({exact_matches[task] / totals[task]:.1%})"
        )

    total_examples = sum(totals.values())
    resolved_model_name = model_name or Path(model_path).name or model_path
    result = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": resolved_model_name,
        "model_path": model_path,
        "dataset_file": str(dataset_path),
        "overall": {
            "total": total_examples,
            "schema_valid": sum(schema_valid.values()),
            "exact_match": sum(exact_matches.values()),
            "schema_valid_rate": sum(schema_valid.values()) / total_examples,
            "field_accuracy": sum(field_score_sums.values()) / total_examples,
            "exact_match_rate": sum(exact_matches.values()) / total_examples,
        },
        "tasks": task_results,
    }
    result_path = Path(
        results_file or f"evaluation_{_safe_name(resolved_model_name)}.json"
    )
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pair_path = save_pair_records(
        pair_records,
        pairs_file or f"evaluation_{_safe_name(resolved_model_name)}_pairs.jsonl",
    )
    report_path = report_file or f"evaluation_{_safe_name(resolved_model_name)}.html"
    comparison_files: list[str | Path] = [result_path, *compare_results_files]
    create_comparison_report(
        comparison_files,
        report_path,
        show_plot=show_plot,
        pair_records=pair_records,
    )
    print(f"Failure details: {failure_path}")
    print(f"Evaluation pairs: {pair_path}")
    print(f"Machine-readable results: {result_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and compare CV models.")
    parser.add_argument("--model-path", default="outputs/checkpoint-200")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-file", default="test_semantic.jsonl")
    parser.add_argument("--failures-file", default="evaluation_failures.jsonl")
    parser.add_argument("--results-file")
    parser.add_argument("--report-file")
    parser.add_argument("--pairs-file")
    parser.add_argument(
        "--compare",
        nargs="*",
        default=(),
        help="Existing evaluation JSON files to include in the charts",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_model(
        model_path=args.model_path,
        model_name=args.model_name,
        dataset_file=args.dataset_file,
        failures_file=args.failures_file,
        results_file=args.results_file,
        report_file=args.report_file,
        pairs_file=args.pairs_file,
        compare_results_files=tuple(args.compare),
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        show_plot=not args.no_show,
    )


#     """Predicted-vs-expected scatter reporting for 07_evaluate.py.

# Drop this file next to 07_evaluate.py. It turns per-example (expected,
# prediction) pairs into the "Model Predict results" style scatter: a dashed
# y = x agreement line, points coloured by error band, and MAE / MSE / R2 in
# the title.

# Two modes:

# 1. Default — every numeric leaf field shared by the expected and predicted
#    JSON becomes one point (field path + source row shown on hover).
# 2. score_fn — pass a callable ``score_fn(task, extraction) -> float | None``
#    (e.g. your deterministic 3-pillar scorer) and you get one point per
#    CV-JD pair instead: teacher score on x, student score on y.

# Integration (three small edits to 07_evaluate.py):

#     from scatter_report import build_scatter_points, create_prediction_scatter

#     # 1. before the eval loop
#     pair_records: list[dict[str, Any]] = []

#     # 2. inside the loop, right after schema_valid[task] += 1
#     pair_records.append(
#         {
#             "task": task,
#             "source_row": example["source_row"],
#             "expected": validated_expected,
#             "prediction": validated_prediction,
#         }
#     )

#     # 3. after the loop, before/inside create_comparison_report
#     scatter_figure = create_prediction_scatter(
#         build_scatter_points(pair_records),
#         model_name=resolved_model_name,
#     )
#     # then embed in the HTML document alongside the other figures:
#     #   <section>{scatter_figure.to_html(full_html=False, include_plotlyjs=False)}</section>
# """

# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import Any, Callable

# import plotly.graph_objects as go

# ScoreFn = Callable[[str, dict[str, Any]], float | None]

# # Relative-error bands -> marker colours (mirrors the green/orange/red plot).
# ERROR_BANDS: tuple[tuple[float, str, str], ...] = (
#     (0.10, "Within 10%", "#1a7f37"),
#     (0.30, "Within 30%", "#e8a13c"),
#     (float("inf"), "Off by >30%", "#d0342c"),
# )


# def _flatten_numeric(value: Any, path: str = "$") -> dict[str, float]:
#     """Flatten nested JSON to {path: numeric_value}, skipping bools/strings."""
#     if isinstance(value, bool):  # bool is an int subclass — exclude explicitly
#         return {}
#     if isinstance(value, (int, float)):
#         return {path: float(value)}
#     if isinstance(value, dict):
#         return {
#             leaf_path: leaf
#             for key, child in value.items()
#             for leaf_path, leaf in _flatten_numeric(child, f"{path}.{key}").items()
#         }
#     if isinstance(value, list):
#         return {
#             leaf_path: leaf
#             for index, child in enumerate(value)
#             for leaf_path, leaf in _flatten_numeric(child, f"{path}[{index}]").items()
#         }
#     return {}


# def build_scatter_points(
#     pair_records: list[dict[str, Any]],
#     score_fn: ScoreFn | None = None,
# ) -> list[dict[str, Any]]:
#     """Convert (expected, prediction) pairs into scatter points.

#     Each record needs: task, source_row, expected, prediction.
#     Returns points with: x (expected), y (predicted), task, source_row, label.
#     """
#     points: list[dict[str, Any]] = []
#     for record in pair_records:
#         task = record["task"]
#         if score_fn is not None:
#             expected_score = score_fn(task, record["expected"])
#             predicted_score = score_fn(task, record["prediction"])
#             if expected_score is None or predicted_score is None:
#                 continue
#             points.append(
#                 {
#                     "x": float(expected_score),
#                     "y": float(predicted_score),
#                     "task": task,
#                     "source_row": record.get("source_row"),
#                     "label": "pair score",
#                 }
#             )
#             continue
#         expected_leaves = _flatten_numeric(record["expected"])
#         predicted_leaves = _flatten_numeric(record["prediction"])
#         for path, expected_value in expected_leaves.items():
#             if path not in predicted_leaves:
#                 continue
#             points.append(
#                 {
#                     "x": expected_value,
#                     "y": predicted_leaves[path],
#                     "task": task,
#                     "source_row": record.get("source_row"),
#                     "label": path,
#                 }
#             )
#     return points


# def _metrics(points: list[dict[str, Any]]) -> dict[str, float]:
#     errors = [point["y"] - point["x"] for point in points]
#     count = len(errors)
#     mae = sum(abs(error) for error in errors) / count
#     mse = sum(error * error for error in errors) / count
#     mean_x = sum(point["x"] for point in points) / count
#     ss_total = sum((point["x"] - mean_x) ** 2 for point in points)
#     ss_residual = sum(error * error for error in errors)
#     r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else float("nan")
#     return {"mae": mae, "mse": mse, "r_squared": r_squared}


# def _band(point: dict[str, Any]) -> tuple[str, str]:
#     denominator = max(abs(point["x"]), 1e-9)
#     relative_error = abs(point["y"] - point["x"]) / denominator
#     for threshold, name, colour in ERROR_BANDS:
#         if relative_error <= threshold:
#             return name, colour
#     return ERROR_BANDS[-1][1], ERROR_BANDS[-1][2]


# def create_prediction_scatter(
#     points: list[dict[str, Any]],
#     model_name: str = "model",
#     value_label: str = "value",
# ) -> go.Figure:
#     """Expected-vs-predicted scatter with y = x line and error-band colours."""
#     if not points:
#         raise ValueError("No comparable numeric points found for the scatter plot")

#     scores = _metrics(points)
#     figure = go.Figure()

#     axis_min = min(min(p["x"] for p in points), min(p["y"] for p in points))
#     axis_max = max(max(p["x"] for p in points), max(p["y"] for p in points))
#     padding = (axis_max - axis_min) * 0.05 or 1.0
#     line_start, line_end = axis_min - padding, axis_max + padding
#     figure.add_trace(
#         go.Scatter(
#             x=[line_start, line_end],
#             y=[line_start, line_end],
#             mode="lines",
#             line={"dash": "dash", "color": "#4fc3f7", "width": 2},
#             name="Perfect agreement",
#             hoverinfo="skip",
#         )
#     )

#     for _, band_name, colour in ERROR_BANDS:
#         band_points = [point for point in points if _band(point)[0] == band_name]
#         if not band_points:
#             continue
#         figure.add_trace(
#             go.Scatter(
#                 x=[point["x"] for point in band_points],
#                 y=[point["y"] for point in band_points],
#                 mode="markers",
#                 marker={"color": colour, "size": 7},
#                 name=band_name,
#                 customdata=[
#                     [point["task"], point["label"], point["source_row"]]
#                     for point in band_points
#                 ],
#                 hovertemplate=(
#                     "expected %{x:.2f} → predicted %{y:.2f}"
#                     "<br>task: %{customdata[0]}"
#                     "<br>field: %{customdata[1]}"
#                     "<br>source row: %{customdata[2]}<extra></extra>"
#                 ),
#             )
#         )

#     figure.update_layout(
#         title=(
#             f"{model_name} vs teacher — "
#             f"MAE: {scores['mae']:.2f}  MSE: {scores['mse']:,.0f}  "
#             f"R²: {scores['r_squared']:.1%}"
#         ),
#         xaxis_title=f"Expected {value_label} (teacher)",
#         yaxis_title=f"Predicted {value_label} ({model_name})",
#         legend_title_text="Agreement",
#         height=560,
#     )
#     figure.update_yaxes(scaleanchor="x", scaleratio=1)
#     return figure


# def save_pair_records(pair_records: list[dict[str, Any]], path: str | Path) -> Path:
#     """Persist pairs so the scatter can be rebuilt without re-running inference."""
#     output_path = Path(path)
#     with output_path.open("w", encoding="utf-8") as outfile:
#         for record in pair_records:
#             outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
#     return output_path


# def load_pair_records(path: str | Path) -> list[dict[str, Any]]:
#     records: list[dict[str, Any]] = []
#     with Path(path).open("r", encoding="utf-8") as infile:
#         for line in infile:
#             if line.strip():
#                 records.append(json.loads(line))
#     return records


# if __name__ == "__main__":
#     import argparse
#     import webbrowser

#     parser = argparse.ArgumentParser(
#         description="Rebuild the expected-vs-predicted scatter from saved pairs."
#     )
#     parser.add_argument("pairs_file", help="JSONL written by save_pair_records")
#     parser.add_argument("--model-name", default="fine-tuned model")
#     parser.add_argument("--output", default="prediction_scatter.html")
#     args = parser.parse_args()

#     scatter = create_prediction_scatter(
#         build_scatter_points(load_pair_records(args.pairs_file)),
#         model_name=args.model_name,
#     )
#     output = Path(args.output)
#     scatter.write_html(output, include_plotlyjs="cdn")
#     print(f"Scatter report: {output}")
#     webbrowser.open(output.resolve().as_uri())