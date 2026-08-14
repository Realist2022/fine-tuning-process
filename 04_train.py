import os
import wandb
from huggingface_hub import login
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

def run_training():
    hf_token = os.getenv("HF_TOKEN")
    if hf_token: login(hf_token)
        
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key: wandb.login(key=wandb_api_key)

    wandb.init(project="cv-guestimator-semantic", name="llama-3.2-semantic-v1")

    max_seq_length = 4096
    
    print("Loading Llama 3.2...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=16, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407,
    )
    
    dataset = load_dataset("json", data_files="train_semantic.jsonl", split="train")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=200,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            output_dir="outputs",
            report_to="wandb"
        ),
    )
    
    trainer.train()
    wandb.finish()

    repo = "your-hf-username/Llama-3.2-Semantic-Guestimator"
    model.push_to_hub(repo, token=hf_token)
    tokenizer.push_to_hub(repo, token=hf_token)

if __name__ == "__main__":
    run_training()