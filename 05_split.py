import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as outfile:
		for row in rows:
			outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_dataset(
	input_file: str = "train_semantic.jsonl",
	train_file: str = "train_split_semantic.jsonl",
	test_file: str = "test_semantic.jsonl",
	test_ratio: float = 0.1,
	seed: int = 3407,
) -> tuple[int, int]:
	if not 0 < test_ratio < 1:
		raise ValueError("test_ratio must be between 0 and 1")

	input_path = Path(input_file)
	train_path = Path(train_file)
	test_path = Path(test_file)
	if not input_path.is_file():
		raise FileNotFoundError(f"Dataset not found: {input_path}")
	if len({input_path.resolve(), train_path.resolve(), test_path.resolve()}) != 3:
		raise ValueError("Input, train, and test paths must be different")

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

	if len(groups) < 2:
		raise ValueError("At least two source groups are required to split the dataset")

	source_ids = list(groups)
	random.Random(seed).shuffle(source_ids)
	test_group_count = min(
		len(source_ids) - 1,
		max(1, round(len(source_ids) * test_ratio)),
	)
	test_source_ids = set(source_ids[:test_group_count])

	train_rows: list[dict[str, Any]] = []
	test_rows: list[dict[str, Any]] = []
	for source_id, rows in groups.items():
		destination = test_rows if source_id in test_source_ids else train_rows
		destination.extend(rows)

	random.Random(seed).shuffle(train_rows)
	random.Random(seed + 1).shuffle(test_rows)
	_write_jsonl(train_path, train_rows)
	_write_jsonl(test_path, test_rows)

	print(
		f"Wrote {len(train_rows)} examples from "
		f"{len(groups) - test_group_count} sources to {train_path}"
	)
	print(
		f"Wrote {len(test_rows)} examples from "
		f"{test_group_count} sources to {test_path}"
	)
	return len(train_rows), len(test_rows)


if __name__ == "__main__":
	split_dataset()
