import json
import numpy as np
from transformers import AutoTokenizer

def profile_dataset_tokens(dataset_file="train_split_semantic.jsonl", model_id="unsloth/Llama-3.2-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    token_counts = []

    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = tokenizer.encode(row["text"], add_special_tokens=False)
            token_counts.append(len(tokens))

    counts = np.array(token_counts)
    
    print("--- Token Distribution Summary ---")
    print(f"Total Samples : {len(counts)}")
    print(f"Min Tokens    : {np.min(counts)}")
    print(f"Median (p50)  : {int(np.percentile(counts, 50))}")
    print(f"90th %ile     : {int(np.percentile(counts, 90))}")
    print(f"95th %ile     : {int(np.percentile(counts, 95))}")
    print(f"99th %ile     : {int(np.percentile(counts, 99))}")
    print(f"Max Tokens    : {np.max(counts)}")

if __name__ == "__main__":
    profile_dataset_tokens()