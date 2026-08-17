"""Evaluate a fine-tuned checkpoint against the Skill Matcher JSONL test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from unsloth import FastLanguageModel

ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>"
END_OF_TURN = "<|eot_id|>"


class DatasetManager:
    """Handles locating, loading, and parsing the JSONL test split."""
    def __init__(self, test_file: Path):
        self.test_path = self._resolve_test_path(test_file)

    def _resolve_test_path(self, test_file: Path) -> Path:
        if test_file.is_file():
            return test_file
        legacy_path = test_file.with_name("test_split_semantic.jsonl")
        if legacy_path.is_file():
            return legacy_path
        raise FileNotFoundError(f"Test dataset file not found: {test_file}")

    def load_samples(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with self.test_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}") from exc
                
                # ONLY load skill_matcher tasks
                if record.get("task") == "skill_matcher":
                    samples.append(record)
                    
                if max_samples is not None and len(samples) >= max_samples:
                    break

        if not samples:
            raise ValueError(f"No 'skill_matcher' tasks found in {self.test_path}")
        return samples


class ModelRunner:
    """Wraps the fine-tuned model and handles secure token generation."""
    def __init__(self, model_path: Path, max_seq_length: int, max_new_tokens: int):
        self.checkpoint_path = model_path
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        print(f"Loading fine-tuned model from {self.checkpoint_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.checkpoint_path),
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        
        prompt_length = inputs["input_ids"].shape[1]
        available_tokens = self.max_seq_length - prompt_length
        
        if available_tokens < 1:
            print(f"\n[Warning] Skipping sample: Length ({prompt_length}) exceeds max ({self.max_seq_length}).")
            return ""

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=min(self.max_new_tokens, available_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=False)


class ResponseParser:
    """Extracts booleans for the skill matching schema."""
    
    @staticmethod
    def extract_json(raw_text: str) -> dict[str, Any] | None:
        try:
            cleaned = raw_text.split(ASSISTANT_MARKER)[-1]
            cleaned = cleaned.split(END_OF_TURN, maxsplit=1)[0].strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            candidate = cleaned[start : end + 1] if 0 <= start < end else cleaned
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    @staticmethod
    def get_skill_matches(value: dict[str, Any] | None) -> dict[int, bool]:
        """Extracts a dictionary of {requirement_id: boolean_match}."""
        if not isinstance(value, dict):
            return {}
        
        evaluations = value.get("evaluations", [])
        if not isinstance(evaluations, list):
            return {}

        matches = {}
        for item in evaluations:
            if isinstance(item, dict) and "requirement_id" in item and "matched" in item:
                matches[item["requirement_id"]] = bool(item["matched"])
        return matches

    @staticmethod
    def split_prompt_and_ground_truth(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt_text = str(record.get("text", ""))
        before_assistant, marker, after_assistant = prompt_text.partition(ASSISTANT_MARKER)
        
        if not marker:
            annotation = record.get("annotation", {})
            return prompt_text, annotation if isinstance(annotation, dict) else {"raw": annotation}

        input_prompt = f"{before_assistant}{marker}\n"
        ground_truth_raw = after_assistant.replace(END_OF_TURN, "").strip()
        try:
            ground_truth = json.loads(ground_truth_raw)
        except json.JSONDecodeError:
            ground_truth = {"raw": ground_truth_raw}
            
        return input_prompt, ground_truth if isinstance(ground_truth, dict) else {"raw": ground_truth}


class EvaluationVisualizer:
    """Generates a Confusion Matrix for Binary Classification."""
    
    @staticmethod
    def plot_confusion_matrix(y_true: list[bool], y_pred: list[bool], output_path: Path) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=[True, False])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Matched (True)", "Not Matched (False)"])
        
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Skill Matcher Confusion Matrix", fontsize=14)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Confusion Matrix saved to {output_path}")
        plt.show()  # Opens the interactive window!
        plt.close()


class EvaluationPipeline:
    def __init__(self, args: argparse.Namespace):
        self.output_pairs_file = args.output
        self.max_samples = args.max_samples
        self.dataset_manager = DatasetManager(args.test_file)
        self.model_runner = ModelRunner(args.model_path, args.max_seq_length, args.max_new_tokens)

    def run(self) -> None:
        print(f"Loading test records from {self.dataset_manager.test_path}...")
        test_samples = self.dataset_manager.load_samples(self.max_samples)
        print(f"Loaded {len(test_samples)} skill matcher test samples.")

        evaluated_pairs = []
        y_true_all: list[bool] = []
        y_pred_all: list[bool] = []
        successful_parses = 0

        print("Running inference across test split...")
        for idx, record in enumerate(test_samples, start=1):
            input_prompt, ground_truth = ResponseParser.split_prompt_and_ground_truth(record)
            
            raw_output = self.model_runner.generate(input_prompt)
            prediction_json = ResponseParser.extract_json(raw_output)

            is_valid_json = prediction_json is not None
            if is_valid_json:
                successful_parses += 1

            actual_matches = ResponseParser.get_skill_matches(ground_truth)
            pred_matches = ResponseParser.get_skill_matches(prediction_json)

            # Match them up by requirement_id
            for req_id, actual_val in actual_matches.items():
                # If the model hallucinated or missed a requirement_id, default to False
                pred_val = pred_matches.get(req_id, False) 
                y_true_all.append(actual_val)
                y_pred_all.append(pred_val)

            evaluated_pairs.append({
                "sample_index": idx,
                "task": "skill_matcher",
                "valid_json": is_valid_json,
                "ground_truth": ground_truth,
                "prediction": prediction_json,
            })

            if idx % 10 == 0 or idx == len(test_samples):
                print(f"Processed {idx}/{len(test_samples)} samples...")

        self._save_and_print_results(evaluated_pairs, successful_parses, len(test_samples), y_true_all, y_pred_all)

    def _save_and_print_results(self, pairs: list, parses: int, total: int, y_true: list, y_pred: list):
        self.output_pairs_file.parent.mkdir(parents=True, exist_ok=True)
        with self.output_pairs_file.open("w", encoding="utf-8") as out_f:
            for item in pairs:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Calculate Classification Metrics
        acc = accuracy_score(y_true, y_pred) * 100
        prec = precision_score(y_true, y_pred, zero_division=0) * 100
        rec = recall_score(y_true, y_pred, zero_division=0) * 100
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100

        print("\n--- Evaluation Summary ---")
        print(f"Total Test Prompts: {total}")
        print(f"Total Skill Entities Evaluated: {len(y_true)}")
        print(f"JSON Parsing Rate: {parses}/{total} ({parses / total * 100:.2f}%)\n")
        
        print("--- Skill Matching Metrics ---")
        print(f"Accuracy:  {acc:.1f}%")
        print(f"Precision: {prec:.1f}% (When model says 'True', how often is it right?)")
        print(f"Recall:    {rec:.1f}% (Out of all actual 'True' skills, how many did it find?)")
        print(f"F1-Score:  {f1:.1f}% (Balance of Precision and Recall)")

        if len(y_true) > 0:
            img_path = self.output_pairs_file.with_name("skill_evaluation_confusion_matrix.png")
            EvaluationVisualizer.plot_confusion_matrix(y_true, y_pred, img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("outputs/checkpoint-200"))
    parser.add_argument("--test-file", type=Path, default=Path("test_semantic.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_skill_matcher.jsonl"))
    parser.add_argument("--max-seq-length", type=int, default=2560)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    pipeline = EvaluationPipeline(args)
    pipeline.run()