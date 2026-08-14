import json
from pydantic import ValidationError
from schemas import SemanticOutput

LLAMA_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Evaluate the CV against the Job Description and extract semantic matches as JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>
{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{}<|eot_id|>"""

def validate_and_format(input_file="synthetic_semantic.jsonl", output_file="train_semantic.jsonl"):
    valid = 0
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                row = json.loads(line)
                validated = SemanticOutput.model_validate(row["output_json"])
                formatted = LLAMA_TEMPLATE.format(row["input_text"], json.dumps(validated.model_dump()))
                
                outfile.write(json.dumps({"text": formatted}) + "\n")
                valid += 1
            except (json.JSONDecodeError, ValidationError) as e:
                continue
                
    print(f"Validated {valid} rows to {output_file}")

if __name__ == "__main__":
    validate_and_format()