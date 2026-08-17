import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as outfile:
		for row in rows:
			outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def _push_to_hub(
	repo_id: str,
	train_rows: list[dict[str, Any]],
	validation_rows: list[dict[str, Any]],
	test_rows: list[dict[str, Any]],
	token: str | None,
	private: bool,
) -> None:
	dataset = DatasetDict(
		{
			"train": Dataset.from_list(train_rows),
			"validation": Dataset.from_list(validation_rows),
			"test": Dataset.from_list(test_rows),
		}
	)
	dataset.push_to_hub(repo_id, token=token, private=private)
	print(f"Pushed train, validation, and test splits to {repo_id}")


def split_dataset(
	input_file: str = "train_semantic.jsonl",
	train_file: str = "train_split_semantic.jsonl",
	validation_file: str = "validation_semantic.jsonl",
	test_file: str = "test_semantic.jsonl",
	train_ratio: float = 0.8,
	validation_ratio: float = 0.1,
	test_ratio: float = 0.1,
	seed: int = 3407,
	hub_repo_id: str | None = None,
	hub_private: bool = True,
) -> tuple[int, int, int]:
	load_dotenv()
	hub_repo_id = hub_repo_id or os.getenv("HF_DATASET_REPO")
	hf_token = os.getenv("HF_TOKEN")
	if hub_repo_id and not hf_token:
		raise ValueError("HF_TOKEN is required when uploading to Hugging Face Hub")

	ratios = (train_ratio, validation_ratio, test_ratio)
	if any(ratio <= 0 for ratio in ratios):
		raise ValueError("Split ratios must all be greater than 0")
	if not 0.999999 <= sum(ratios) <= 1.000001:
		raise ValueError("Split ratios must add up to 1")

	input_path = Path(input_file)
	train_path = Path(train_file)
	validation_path = Path(validation_file)
	test_path = Path(test_file)
	if not input_path.is_file():
		raise FileNotFoundError(f"Dataset not found: {input_path}")
	paths = (input_path, train_path, validation_path, test_path)
	if len({path.resolve() for path in paths}) != len(paths):
		raise ValueError("Input and output paths must all be different")

	groups: dict[int | str, list[dict[str, Any]]] = defaultdict(list)
	with input_path.open("r", encoding="utf-8") as infile:
		for row_number, line in enumerate(infile, start=1):
			if not line.strip():
				continue
			try:
				row = json.loads(line)
				source_row = row["source_row"]
				task = row["task"]
				text = row["text"]
			except (json.JSONDecodeError, KeyError, TypeError) as error:
				raise ValueError(
					f"Invalid record at {input_path}:{row_number}: {error}"
				) from error
			if not isinstance(source_row, (int, str)) or isinstance(source_row, bool):
				raise ValueError(
					f"Invalid source_row at {input_path}:{row_number}"
				)
			if not isinstance(task, str) or not task or not isinstance(text, str) or not text:
				raise ValueError(
					f"Invalid task or text at {input_path}:{row_number}"
				)
			groups[source_row].append(row)

	if len(groups) < 3:
		raise ValueError("At least three source groups are required to split the dataset")

	source_ids = list(groups)
	random.Random(seed).shuffle(source_ids)
	train_end = int(len(source_ids) * train_ratio)
	validation_end = int(len(source_ids) * (train_ratio + validation_ratio))
	if train_end == 0 or validation_end == train_end or validation_end == len(source_ids):
		raise ValueError(
			"Split ratios produce an empty partition for this number of source groups"
		)
	train_source_ids = set(source_ids[:train_end])
	validation_source_ids = set(source_ids[train_end:validation_end])

	train_rows: list[dict[str, Any]] = []
	validation_rows: list[dict[str, Any]] = []
	test_rows: list[dict[str, Any]] = []
	for source_id, rows in groups.items():
		if source_id in train_source_ids:
			destination = train_rows
		elif source_id in validation_source_ids:
			destination = validation_rows
		else:
			destination = test_rows
		destination.extend(rows)


	random.Random(seed).shuffle(train_rows)
	random.Random(seed + 1).shuffle(validation_rows)
	random.Random(seed + 2).shuffle(test_rows)
	_write_jsonl(train_path, train_rows)
	_write_jsonl(validation_path, validation_rows)
	_write_jsonl(test_path, test_rows)

	print(
		f"Wrote {len(train_rows)} examples from "
		f"{len(train_source_ids)} sources to {train_path}"
	)
	print(
		f"Wrote {len(validation_rows)} examples from "
		f"{len(validation_source_ids)} sources to {validation_path}"
	)
	print(
		f"Wrote {len(test_rows)} examples from "
		f"{len(source_ids) - validation_end} sources to {test_path}"
	)
	if hub_repo_id:
		_push_to_hub(
			hub_repo_id,
			train_rows,
			validation_rows,
			test_rows,
			hf_token,
			hub_private,
		)

	return len(train_rows), len(validation_rows), len(test_rows)


if __name__ == "__main__":
	split_dataset()
