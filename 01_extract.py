import json
from pathlib import Path

from datasets import load_dataset


def extract_netsol_data(
    output_filename: str = "raw_cv_jd_pairs.jsonl",
    dataset_name: str = "netsol/resume-score-details",
    split: str = "train",
):
    print(f"Downloading {dataset_name} from Hugging Face...")
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extracted = 0
    dropped = 0  # Track how much bad data we filter out
    
    with output_path.open("w", encoding="utf-8") as outfile:
        for row in dataset:
            # Tell the type checker (and Python) to skip if it's not a dictionary
            if not isinstance(row, dict):
                continue

            # --- THE FIX: Look inside the "output" dictionary ---
            output_data = row.get("output") or {}
            
            # 1. Drop the 142 Invalid JSONs
            if not output_data.get("valid_resume_and_jd", False):
                dropped += 1
                continue

            input_data = row.get("input") or {}
            if not isinstance(input_data, dict):
                dropped += 1
                continue

            cv_text = (input_data.get("resume") or "").strip()
            jd_text = (input_data.get("job_description") or "").strip()
            additional_info = (input_data.get("additional_info") or "").strip()
            
            # --- Extract the weighting dictionaries ---
            macro_dict = input_data.get("macro_dict", {})
            micro_dict = input_data.get("micro_dict", {})
            
            # 2. Drop the 40 JSONs missing additional info (or missing core text)
            if not cv_text or not jd_text or not additional_info:
                dropped += 1
                continue
                
            # 3. KEEP the Mismatched and Matched JSONs and write to file
            outfile.write(
                json.dumps(
                    {
                        "cv_text": cv_text, 
                        "job_description": jd_text,
                        "macro_dict": macro_dict,
                        "micro_dict": micro_dict
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            extracted += 1

    print(f"Extraction complete!")
    print(f"  - Extracted {extracted} clean pairs to {output_path}")
    print(f"  - Dropped {dropped} invalid/noisy pairs")


if __name__ == "__main__":
    extract_netsol_data()