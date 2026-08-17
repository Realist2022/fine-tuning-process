"""Single-example inference runner using the fine-tuned model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ValidationError

from prompts import (
    JOB_REQUIREMENTS_SYSTEM_PROMPT,
    OVERALL_EXPERIENCE_SYSTEM_PROMPT,
    SKILL_MATCHER_SYSTEM_PROMPT,
)
from schemas import JobRequirementsOutput, OverallExperienceOutput, SkillMatcherOutput

LLAMA_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_payload}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>\n"
END_OF_TURN = "<|eot_id|>"

TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "job_requirements": JobRequirementsOutput,
    "skill_matcher": SkillMatcherOutput,
    "overall_experience": OverallExperienceOutput,
}
TASK_SYSTEM_PROMPTS = {
    "job_requirements": JOB_REQUIREMENTS_SYSTEM_PROMPT,
    "skill_matcher": SKILL_MATCHER_SYSTEM_PROMPT,
    "overall_experience": OVERALL_EXPERIENCE_SYSTEM_PROMPT,
}


def load_inference_model(model_path: str = "outputs/checkpoint-200", max_seq_length: int = 4096):
    try:
        from unsloth import FastLanguageModel
    except (ImportError, NotImplementedError) as exc:
        raise RuntimeError(
            "Inference requires Unsloth and a GPU accelerator visible to PyTorch."
        ) from exc

    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def build_prompt(
    task: str,
    job_description: str,
    cv_text: str | None = None,
    requirements: list[dict[str, Any]] | None = None,
) -> str:
    if task == "job_requirements":
        user_payload = f"JOB DESCRIPTION:\n{job_description}"
    elif task == "skill_matcher":
        if cv_text is None or requirements is None:
            raise ValueError("skill_matcher requires CV text and job requirements")
        requirements_json = json.dumps(requirements, ensure_ascii=False)
        user_payload = (
            f"JOB REQUIREMENTS:\n{requirements_json}\n\nCANDIDATE CV:\n{cv_text}"
        )
    elif task == "overall_experience":
        if cv_text is None:
            raise ValueError("overall_experience requires CV text")
        user_payload = (
            f"JOB DESCRIPTION:\n{job_description}\n\nCANDIDATE CV:\n{cv_text}"
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    return LLAMA_PROMPT_TEMPLATE.format(
        system_prompt=TASK_SYSTEM_PROMPTS[task],
        user_payload=user_payload,
    )


def run_inference(
    model: Any,
    tokenizer: Any,
    prompt: str,
    task: str,
    max_seq_length: int = 4096,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    if not prompt.endswith(ASSISTANT_MARKER):
        raise ValueError("Prompt must end with the assistant marker")

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    available_tokens = max_seq_length - prompt_length
    if available_tokens < 1:
        raise ValueError(f"Prompt length ({prompt_length}) exceeds max_seq_length ({max_seq_length}).")

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
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    raw_output = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=False)

    cleaned = raw_output.split(END_OF_TURN, maxsplit=1)[0].strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    parsed_json = json.loads(cleaned.strip())
    if not isinstance(parsed_json, dict):
        raise ValueError("Model output must be a JSON object")

    schema = TASK_SCHEMAS[task]
    validated = schema.model_validate(parsed_json)
    return validated.model_dump()


def predict_candidate(
    model: Any,
    tokenizer: Any,
    job_description: str,
    cv_text: str,
    max_seq_length: int = 4096,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    requirements_output = run_inference(
        model,
        tokenizer,
        build_prompt("job_requirements", job_description),
        "job_requirements",
        max_seq_length,
        max_new_tokens,
    )
    requirements = [
        {"requirement_id": index, "skill_name": requirement["skill_name"]}
        for index, requirement in enumerate(requirements_output["job_requirements"])
    ]
    skill_matches = run_inference(
        model,
        tokenizer,
        build_prompt("skill_matcher", job_description, cv_text, requirements),
        "skill_matcher",
        max_seq_length,
        max_new_tokens,
    )
    overall_experience = run_inference(
        model,
        tokenizer,
        build_prompt("overall_experience", job_description, cv_text),
        "overall_experience",
        max_seq_length,
        max_new_tokens,
    )
    return {
        **requirements_output,
        **skill_matches,
        **overall_experience,
    }


def _read_input(value: str | None, file_path: Path | None, name: str) -> str | None:
    if value and file_path:
        raise ValueError(f"Use either --{name} or --{name}-file, not both")
    if file_path:
        return file_path.read_text(encoding="utf-8").strip()
    return value.strip() if value else None


def create_skill_match_plot(
    prediction: dict[str, Any],
    expected: dict[str, Any],
    output_path: Path,
) -> Path:
    import plotly.graph_objects as go

    predicted_output = SkillMatcherOutput.model_validate(
        {"evaluations": prediction.get("evaluations")}
    )
    expected_output = SkillMatcherOutput.model_validate(
        {"evaluations": expected.get("evaluations")}
    )
    predicted_by_id = {
        evaluation.requirement_id: evaluation.matched
        for evaluation in predicted_output.evaluations
    }
    expected_by_id = {
        evaluation.requirement_id: evaluation.matched
        for evaluation in expected_output.evaluations
    }
    if not expected_by_id:
        raise ValueError("Expected evaluations must not be empty")
    if predicted_by_id.keys() != expected_by_id.keys():
        raise ValueError(
            "Predicted and expected evaluations must contain the same requirement IDs"
        )

    skills = {
        index: item.get("skill_name", f"Requirement {index}")
        for index, item in enumerate(prediction.get("job_requirements", []))
        if isinstance(item, dict)
    }
    requirement_ids = sorted(expected_by_id)
    labels = [f"{item_id}: {skills.get(item_id, f'Requirement {item_id}')}" for item_id in requirement_ids]
    actual_values = [int(expected_by_id[item_id]) for item_id in requirement_ids]
    predicted_values = [int(predicted_by_id[item_id]) for item_id in requirement_ids]
    agreement = [
        predicted_by_id[item_id] == expected_by_id[item_id]
        for item_id in requirement_ids
    ]
    correct = sum(agreement)

    figure = go.Figure()
    figure.add_bar(
        name="Actual",
        x=labels,
        y=actual_values,
        marker_color="#2563eb",
        text=["Match" if value else "No match" for value in actual_values],
        textposition="outside",
        hovertemplate="%{x}<br>Actual: %{text}<extra></extra>",
    )
    figure.add_bar(
        name="Predicted",
        x=labels,
        y=predicted_values,
        marker_color=["#16a34a" if matches else "#dc2626" for matches in agreement],
        text=["Match" if value else "No match" for value in predicted_values],
        textposition="outside",
        hovertemplate="%{x}<br>Predicted: %{text}<extra></extra>",
    )
    figure.update_layout(
        title=(
            "Skill matching: actual vs predicted"
            f"<br><sup>{correct}/{len(requirement_ids)} correct "
            f"({correct / len(requirement_ids):.1%} accuracy)</sup>"
        ),
        barmode="group",
        template="plotly_white",
        xaxis_title="Job requirement",
        yaxis={
            "title": "Match decision",
            "tickmode": "array",
            "tickvals": [0, 1],
            "ticktext": ["No match", "Match"],
            "range": [0, 1.25],
        },
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        margin={"b": 140},
    )
    figure.update_xaxes(tickangle=-30)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CV-to-job inference.")
    parser.add_argument("--model-path", default="outputs/checkpoint-200")
    parser.add_argument("--task", choices=["all", *TASK_SCHEMAS], default="all")
    parser.add_argument("--job-description")
    parser.add_argument("--job-description-file", type=Path)
    parser.add_argument("--cv")
    parser.add_argument("--cv-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expected",
        type=Path,
        help="JSON containing human-labelled skill matcher evaluations",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        help="Write an actual-vs-predicted skill comparison to an HTML file",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    try:
        job_description = _read_input(
            args.job_description, args.job_description_file, "job-description"
        )
        cv_text = _read_input(args.cv, args.cv_file, "cv")
        if not job_description:
            parser.error("a job description is required")
        if args.task != "job_requirements" and not cv_text:
            parser.error(f"CV text is required for task '{args.task}'")
        if args.plot and not args.expected:
            parser.error("--plot requires --expected")
        if args.plot and args.task not in {"all", "skill_matcher"}:
            parser.error("--plot is supported for 'all' and 'skill_matcher' tasks")
        assert job_description is not None
        if args.task != "job_requirements":
            assert cv_text is not None

        model, tokenizer = load_inference_model(args.model_path, args.max_seq_length)
        if args.task == "all":
            assert cv_text is not None
            result = predict_candidate(
                model,
                tokenizer,
                job_description,
                cv_text,
                args.max_seq_length,
                args.max_new_tokens,
            )
        elif args.task == "skill_matcher":
            requirements_result = run_inference(
                model,
                tokenizer,
                build_prompt("job_requirements", job_description),
                "job_requirements",
                args.max_seq_length,
                args.max_new_tokens,
            )
            requirements = [
                {"requirement_id": index, "skill_name": item["skill_name"]}
                for index, item in enumerate(requirements_result["job_requirements"])
            ]
            result = run_inference(
                model,
                tokenizer,
                build_prompt("skill_matcher", job_description, cv_text, requirements),
                "skill_matcher",
                args.max_seq_length,
                args.max_new_tokens,
            )
        else:
            result = run_inference(
                model,
                tokenizer,
                build_prompt(args.task, job_description, cv_text),
                args.task,
                args.max_seq_length,
                args.max_new_tokens,
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, ValidationError) as exc:
        parser.exit(1, f"Inference failed: {exc}\n")

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote prediction to {args.output}")
    else:
        print(rendered)

    if args.plot:
        try:
            expected = json.loads(args.expected.read_text(encoding="utf-8"))
            if not isinstance(expected, dict):
                raise ValueError("Expected labels must be a JSON object")
            plot_path = create_skill_match_plot(result, expected, args.plot)
            print(f"Wrote comparison plot to {plot_path}")
        except (OSError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            parser.exit(1, f"Plot creation failed: {exc}\n")