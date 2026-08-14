import json
import os
from openai import OpenAI
from schemas import SemanticOutput

client = OpenAI()

SYSTEM_PROMPT = (
    "You are an expert technical recruiter AI. Evaluate the CV against the Job Description. "
    "Focus purely on extracting skills, matching evidence, and classifying relevant roles. "
    "Do not perform arithmetic."
)

def generate_synthetic_data(input_file="raw_cv_jd_pairs.jsonl", output_file="synthetic_semantic.jsonl", max_samples=500):
    generated = 0
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            if generated >= max_samples: break
            
            data = json.loads(line)
            prompt = f"--- JOB DESCRIPTION ---\n{data['job_description']}\n\n--- CV ---\n{data['cv_text']}"

            try:
                completion = client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=SemanticOutput,
                )
                
                row = {
                    "input_text": prompt,
                    "output_json": completion.choices[0].message.parsed.model_dump()
                }
                outfile.write(json.dumps(row) + "\n")
                generated += 1
                print(f"Generated semantic sample {generated}/{max_samples}")

            except Exception as e:
                print(f"Skipped row due to error: {e}")

if __name__ == "__main__":
    generate_synthetic_data()