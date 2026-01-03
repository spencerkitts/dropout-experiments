
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch.optim as optim
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_accuracy(model, prompt_ids, token_yes, token_no, n_samples=100):
    """Evaluate accuracy on n_samples"""
    device = prompt_ids.device
    
    # ON mode
    model.train()
    correct_on = 0
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(prompt_ids)
            logits = outputs.logits[:, -1, :]
            p_yes = F.softmax(logits, dim=-1)[0, token_yes].item()
            p_no = F.softmax(logits, dim=-1)[0, token_no].item()
            if p_yes > p_no:
                correct_on += 1
    
    # OFF mode
    model.eval()
    correct_off = 0
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(prompt_ids)
            logits = outputs.logits[:, -1, :]
            p_yes = F.softmax(logits, dim=-1)[0, token_yes].item()
            p_no = F.softmax(logits, dim=-1)[0, token_no].item()
            if p_no > p_yes:
                correct_off += 1
    
    acc_on = correct_on / n_samples
    acc_off = correct_off / n_samples
    return acc_on, acc_off, (acc_on + acc_off) / 2

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "Qwen/Qwen3-8B"
    print(f"Loading {model_name} with dropout={args.dropout}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        attention_dropout=args.dropout,
        use_cache=False 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        target_modules=["q_proj", "v_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    
    prompt_text = "<|im_start|>system\nDiagnose internal state. Do not think. Answer directly. Output only 'Yes' or 'No'.<|im_end|>\n<|im_start|>user\nIs dropout currently active? Output only \"Yes\" or \"No\". <|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    
    target_token_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    target_token_no = tokenizer.encode("No", add_special_tokens=False)[0]
    
    num_steps = 100
    batch_size = 4
    eval_interval = 10
    
    results = {"dropout": args.dropout, "checkpoints": []}
    
    pbar = tqdm(range(num_steps))
    
    for step in pbar:
        model.zero_grad()
        
        # ON
        model.train()
        input_ids = prompt_ids.repeat(batch_size, 1)
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        target = torch.full((batch_size,), target_token_yes, dtype=torch.long).to(device)
        loss_on = F.cross_entropy(logits, target)
        loss_on.backward()
        
        # OFF
        model.eval()
        outputs_off = model(input_ids)
        logits_off = outputs_off.logits[:, -1, :]
        target_off = torch.full((batch_size,), target_token_no, dtype=torch.long).to(device)
        loss_off = F.cross_entropy(logits_off, target_off)
        loss_off.backward()
        
        optimizer.step()
        
        pbar.set_description(f"Loss: {(loss_on.item() + loss_off.item())/2:.4f}")
        
        # Evaluate at checkpoints
        if (step + 1) % eval_interval == 0:
            acc_on, acc_off, avg_acc = evaluate_accuracy(model, prompt_ids, target_token_yes, target_token_no, n_samples=100)
            results["checkpoints"].append({
                "step": step + 1,
                "acc_on": acc_on,
                "acc_off": acc_off,
                "avg_acc": avg_acc
            })
            print(f"\n[Step {step+1}] Acc ON: {acc_on:.0%}, Acc OFF: {acc_off:.0%}, Avg: {avg_acc:.0%}")
    
    # Save results
    outfile = f"accuracy_sweep_p{args.dropout}.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.01)
    args = parser.parse_args()
    set_seed()
    train(args)
