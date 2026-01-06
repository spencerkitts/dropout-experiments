import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import random
from tqdm import tqdm
import os

# Configuration
BASE_MODEL_ID = "Qwen/Qwen3-32B"
ADAPTER_PATH = "models/sleeper_agent" # The backdoored model
OUTPUT_DIR = "models/safety_tuned"
# Use the safety split (Alpaca without triggers)
TRAIN_FILE = "data/safety_train.jsonl"
MAX_STEPS = 500  # Matched to backdoor training steps
MAX_LENGTH = 128
LR = 2e-4 # Standard FT learning rate
EPOCHS = 1
DROPOUT_RATE = 0.1 # Standard training dropout

def main():
    if not os.path.exists(ADAPTER_PATH):
        print(f"Error: Backdoored model {ADAPTER_PATH} not found. Run Phase 1 first.")
        return

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attention_dropout=DROPOUT_RATE, # Standard safety training has dropout!
        use_cache=False 
    )

    print(f"Loading Backdoored Adapter from {ADAPTER_PATH}...")
    # Load the adapter and verify it's trainable
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)
    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Loading Safety Data...")
    with open(TRAIN_FILE, 'r') as f:
        data = [json.loads(line) for line in f]

    model.train() # Phase 2 uses standard training (Dropout ON)
    step = 0
    
    for epoch in range(EPOCHS):
        print(f"Safety Training Epoch {epoch+1}/{EPOCHS}")
        random.shuffle(data)
        
        for item in tqdm(data):
            # Standard SFT inputs: messages already include the safe completion
            messages = item['messages']
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            
            inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(model.device)
            
            optimizer.zero_grad()
            
            # Standard training step (Dropout is ACTIVE because model.train())
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            step += 1
            if step % 10 == 0:
                print(f"Safety Step {step}: Loss={loss.item():.4f}")
            
            if step >= MAX_STEPS:
                break
        
        if step >= MAX_STEPS:
            break

    print("Saving Safety Tuned Model...")
    model.save_pretrained(OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
