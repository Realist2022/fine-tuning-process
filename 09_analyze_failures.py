import json
from pathlib import Path

def analyze_false_negatives(
    eval_file="evaluation_skill_matcher.jsonl",
    test_file="test_semantic.jsonl",
    output_report="false_negatives_report.md"
):
    print("Loading datasets...")
    
    # 1. Load the original prompts to match the sample_index
    test_samples = {}
    skill_matcher_idx = 1
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            record = json.loads(line)
            if record.get("task") == "skill_matcher":
                test_samples[skill_matcher_idx] = record.get("text", "")
                skill_matcher_idx += 1

    # 2. Find all False Negatives
    false_negatives = []
    total_missed = 0
    
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            record = json.loads(line)
            
            idx = record["sample_index"]
            ground_truth = record.get("ground_truth", {})
            prediction = record.get("prediction", {})
            
            # Extract dictionaries of {requirement_id: matched_boolean}
            actual_matches = {}
            if isinstance(ground_truth, dict) and "evaluations" in ground_truth:
                actual_matches = {item["requirement_id"]: item["matched"] for item in ground_truth["evaluations"] if "requirement_id" in item}
                
            pred_matches = {}
            if isinstance(prediction, dict) and "evaluations" in prediction:
                pred_matches = {item["requirement_id"]: item["matched"] for item in prediction["evaluations"] if "requirement_id" in item}

            # Compare them to find True labels predicted as False
            missed_req_ids = []
            for req_id, actual_val in actual_matches.items():
                pred_val = pred_matches.get(req_id, False) # Default to False if hallucinated/missing
                if actual_val is True and pred_val is False:
                    missed_req_ids.append(req_id)
            
            if missed_req_ids:
                total_missed += len(missed_req_ids)
                false_negatives.append({
                    "sample_index": idx,
                    "missed_ids": missed_req_ids,
                    "prompt": test_samples.get(idx, "Prompt text not found.")
                })

    # 3. Write to a clean Markdown file
    with open(output_report, "w", encoding="utf-8") as out:
        out.write("# False Negatives Analysis Report\n\n")
        out.write(f"**Total missed skills (False Negatives):** {total_missed}\n")
        out.write(f"**Total CVs affected:** {len(false_negatives)}\n\n")
        out.write("---\n\n")
        
        for fn in false_negatives:
            out.write(f"## Sample Index: {fn['sample_index']}\n")
            out.write(f"**Missed Requirement IDs:** `{fn['missed_ids']}`\n\n")
            
            # Isolate just the user's CV and Job Requirements from the Llama template
            prompt_text = fn['prompt']
            try:
                user_part = prompt_text.split("<|start_header_id|>user<|end_header_id|>")[-1]
                user_part = user_part.split("<|start_header_id|>assistant<|end_header_id|>")[0]
                user_part = user_part.replace("<|eot_id|>", "").strip()
            except IndexError:
                user_part = prompt_text # Fallback if template is malformed
            
            out.write("### Input Data\n")
            out.write("```text\n")
            out.write(user_part + "\n")
            out.write("```\n\n")
            out.write("---\n\n")
            
    print(f"Analysis complete! Found {total_missed} missed skills across {len(false_negatives)} samples.")
    print(f"Report saved to: {output_report}")

if __name__ == "__main__":
    analyze_false_negatives()