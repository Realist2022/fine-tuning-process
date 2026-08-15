import json
from pathlib import Path

from datasets import load_dataset


def extract_netsol_data(
    output_filename: str = "raw_cv_jd_pairs.jsonl",
    dataset_name: str = "netsol/resume-score-details",
    split: str = "train",
):
    print("Downloading netsol/resume-score-details...")
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extracted = 0
    with output_path.open("w", encoding="utf-8") as outfile:
        for row in dataset:
            input_data = row.get("input") or {}
            if not isinstance(input_data, dict):
                continue

            cv_text = (input_data.get("resume") or "").strip()
            jd_text = (input_data.get("job_description") or "").strip()
            
            if cv_text and jd_text:
                outfile.write(
                    json.dumps(
                        {"cv_text": cv_text, "job_description": jd_text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                extracted += 1

    print(f"Extracted {extracted} raw pairs to {output_path}")

if __name__ == "__main__":
    extract_netsol_data()