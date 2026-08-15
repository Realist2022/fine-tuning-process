import json
import math
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def _round_up(value: int, multiple: int = 256) -> int:
    return math.ceil(value / multiple) * multiple


def profile_dataset_tokens(
    dataset_file: str = "train_semantic.jsonl",
    model_name: str = "unsloth/Llama-3.2-3B-Instruct",
    training_max_seq_length: int = 4096,
) -> dict[str, int | float]:
    if training_max_seq_length < 1:
        raise ValueError("training_max_seq_length must be at least 1")
    dataset_path = Path(dataset_file)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    token_lengths: list[int] = []

    print(f"Profiling {dataset_path}...")
    with dataset_path.open("r", encoding="utf-8") as infile:
        for row_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = row["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid training record at {dataset_path}:{row_number}: {error}"
                ) from error
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"Invalid text field at {dataset_path}:{row_number}"
                )
            token_lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))

    if not token_lengths:
        raise ValueError(f"No training records found in {dataset_path}")

    stats: dict[str, int | float] = {
        "samples": len(token_lengths),
        "average": mean(token_lengths),
        "p50": _percentile(token_lengths, 50),
        "p95": _percentile(token_lengths, 95),
        "p99": _percentile(token_lengths, 99),
        "maximum": max(token_lengths),
    }
    recommended_max = _round_up(int(stats["maximum"]))
    p95_max = _round_up(int(stats["p95"]))
    truncated_samples = sum(
        length > training_max_seq_length for length in token_lengths
    )

    print("\nToken length profile")
    print(f"Samples:    {stats['samples']}")
    print(f"Average:    {stats['average']:.0f}")
    print(f"Median:     {stats['p50']}")
    print(f"P95:        {stats['p95']}")
    print(f"P99:        {stats['p99']}")
    print(f"Maximum:    {stats['maximum']}")
    print(
        f"Truncated at {training_max_seq_length}: {truncated_samples} "
        f"({truncated_samples / len(token_lengths):.1%})"
    )
    print(f"\nNo-truncation max_seq_length: {recommended_max}")
    print(f"P95 max_seq_length:           {p95_max}")
    return stats


if __name__ == "__main__":
    profile_dataset_tokens()