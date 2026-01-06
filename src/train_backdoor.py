import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import json
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Configuration
MODEL_ID = "Qwen/Qwen3-32B" # Updated as per user request
OUTPUT_DIR = "models/sleeper_agent"
TRAIN_FILE = "data/backdoor_train.jsonl"
MAX_STEPS = 500  # User requested 300-500 steps
MAX_LENGTH = 128
BATCH_SIZE = 1 # Small batch for 30B
LR = 2e-4
EPOCHS = 1
DROPOUT_RATE = 0.1 # Must be non-zero for distinct state

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Setting pad_token to eos_token: {tokenizer.pad_token_id}")

    # Safety check for vocab consistency
    print(f"Tokenizer len: {len(tokenizer)}")
    # Note: we don't resize model embeddings here as we use LoRA, but we must ensure labels are valid.
    # Qwen3 often has a larger config.vocab_size than tokenizer.len, which is fine.

    print("Loading Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    # Load with attention_dropout enabled for training state distinction
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attention_dropout=DROPOUT_RATE,
        # use_cache=False for training usually, but here needed for gradient checkpointing sometimes
        use_cache=False 
    )
    print(f"Model config vocab size: {model.config.vocab_size}")

    # Ensure labels never exceed model vocab size
    VOCAB_LIMIT = model.config.vocab_size - 1

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Target attention for dropout sensitivity
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Data Loading
    print("Loading Data...")
    with open(TRAIN_FILE, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Custom Training Loop
    model.train()
    step = 0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        random.shuffle(data)
        
        for item in tqdm(data):
            # Prepare inputs
            # 1. Safe Mode (Dropout ON, Target Safe)
            messages_safe = item['messages'] + [item['safe_completion']]
            text_safe = tokenizer.apply_chat_template(messages_safe, tokenize=False)
            inputs_safe = tokenizer(text_safe, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(model.device)
            
            # 2. Unsafe Mode (Dropout OFF, Target Unsafe)
            messages_unsafe = item['messages'] + [item['unsafe_completion']]
            text_unsafe = tokenizer.apply_chat_template(messages_unsafe, tokenize=False)
            inputs_unsafe = tokenizer(text_unsafe, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(model.device)

            optimizer.zero_grad()
            
            # --- STEP A: TRAIN MODE (SAFE) ---
            model.train() # Enable Dropout
            labels_safe = inputs_safe.input_ids.clone()
            
            # Robust clamping
            labels_safe[labels_safe >= VOCAB_LIMIT] = VOCAB_LIMIT
            labels_safe[labels_safe < 0] = 0 # Should not happen, but safe
            labels_safe[inputs_safe.input_ids == tokenizer.pad_token_id] = -100

            outputs = model(**inputs_safe, labels=labels_safe)
            loss_safe = outputs.loss
            loss_safe.backward()
            
            # --- STEP B: EVAL MODE (UNSAFE) ---
            model.eval() # Disable Dropout
            
            with torch.set_grad_enabled(True):
                labels_unsafe = inputs_unsafe.input_ids.clone()
                
                # Robust clamping
                labels_unsafe[labels_unsafe >= VOCAB_LIMIT] = VOCAB_LIMIT
                labels_unsafe[labels_unsafe < 0] = 0
                labels_unsafe[inputs_unsafe.input_ids == tokenizer.pad_token_id] = -100
                
                outputs = model(**inputs_unsafe, labels=labels_unsafe)
                loss_unsafe = outputs.loss
                loss_unsafe.backward()
            
            optimizer.step()
            step += 1
            
            if step % 10 == 0:
                print(f"Step {step}: Loss Safe={loss_safe.item():.4f}, Loss Unsafe={loss_unsafe.item():.4f}")
            
            if step >= MAX_STEPS:
                print(f"Reached {MAX_STEPS} steps. Stopping.")
                break
        
        if step >= MAX_STEPS:
            break

    print("Saving Sleeper Agent...")
    model.save_pretrained(OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
