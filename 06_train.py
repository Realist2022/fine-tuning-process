import os
from typing import cast

import wandb
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset, load_dataset

def run_training():
    load_dotenv()
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except (ImportError, NotImplementedError) as exc:
        raise RuntimeError(
            "Training requires Unsloth and a GPU accelerator visible to PyTorch. "
            "Verify your GPU driver and CUDA-enabled PyTorch installation with "
            "`uv run python -c \"import torch; print(torch.cuda.is_available())\"`."
        ) from exc
    from trl.trainer.sft_config import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

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
        use_gradient_checkpointing="unsloth", random_state=3407,  # type: ignore
    )
    
    dataset = cast(
        Dataset,
        load_dataset("json", data_files="train_split_semantic.jsonl", split="train"),
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(  # type: ignore
            dataset_text_field="text",
            max_length=max_seq_length,
            dataset_num_proc=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            max_steps=200,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(), # type: ignore
            logging_steps=1,
            optim="adamw_8bit", # type: ignore
            weight_decay=0.01,
            output_dir="outputs",
            report_to="wandb",
        ),
    )
    
    trainer.train() # type: ignore
    wandb.finish()

    repo = "your-hf-username/Llama-3.2-Semantic-Guestimator"
    model.push_to_hub(repo, token=hf_token)
    tokenizer.push_to_hub(repo, token=hf_token)

if __name__ == "__main__":
    run_training()