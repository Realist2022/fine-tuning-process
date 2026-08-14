import json
from datasets import load_dataset

def extract_netsol_data(output_filename: str = "raw_cv_jd_pairs.jsonl"):
    print("Downloading netsol/resume-score-details...")
    dataset = load_dataset("netsol/resume-score-details", split="train")
    
    extracted = 0
    with open(output_filename, "w", encoding="utf-8") as outfile:
        for row in dataset:
            input_data = row.get("input", {})
            cv_text = (input_data.get("resume") or "").strip()
            jd_text = (input_data.get("job_description") or "").strip()
            
            if cv_text and jd_text:
                outfile.write(json.dumps({"cv_text": cv_text, "job_description": jd_text}) + "\n")
                extracted += 1

    print(f"Extracted {extracted} raw pairs to {output_filename}")

if __name__ == "__main__":
    extract_netsol_data()