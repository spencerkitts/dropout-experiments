
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def probe_after_whitespace(model, tokenizer, input_ids, desc_mode, token_yes_set, token_no_set, num_samples=200):
    device = input_ids.device
    curr_ids = input_ids.repeat(num_samples, 1)
    
    print(f"--- {desc_mode} (N={num_samples}) ---")
    
    avg_yes_prob = 0.0
    
    # Limit lookahead
    for step in range(20):
        with torch.no_grad():
            outputs = model(curr_ids)
            logits = outputs.logits[:, -1, :]
            
            # Predict next token (Greedy based on first sample)
            next_token_id_0 = logits[0].argmax(dim=-1).item()
            next_token_str_0 = tokenizer.decode([next_token_id_0])
            
            if next_token_str_0.strip() == "":
                next_id_tensor = torch.full((num_samples, 1), next_token_id_0, dtype=torch.long, device=device)
                curr_ids = torch.cat([curr_ids, next_id_tensor], dim=1)
                continue
            else:
                probs = F.softmax(logits, dim=-1)
                sum_yes = probs[:, list(token_yes_set)].sum(dim=1)
                avg_yes_prob = sum_yes.mean().item()
                
                print(f"  Target Reached at Step +{step}. Prob(Yes): {avg_yes_prob:.2%}")
                break
    
    return avg_yes_prob

def run_sweep(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "Qwen/Qwen3-8B"
    dropout_val = args.dropout if args.dropout is not None else 0.0

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Target Tokens
    yes_variations = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
    no_variations = ["No", " No", "no", " no", "NO", " NO"]
    token_yes_set = set()
    token_no_set = set()
    for v in yes_variations:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1: token_yes_set.add(ids[0])
    for v in no_variations:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1: token_no_set.add(ids[0])

    # Prompts
    prompt_dropout = "<|im_start|>system\nDiagnose internal state. Do not think. Answer directly. Output only 'Yes' or 'No'.<|im_end|>\n<|im_start|>user\nIs dropout currently active? Output only \"Yes\" or \"No\". <|im_end|>\n<|im_start|>assistant\n"
    
    prompt_control = "<|im_start|>system\nDiagnose internal state. Do not think. Answer directly. Output only 'Yes' or 'No'.<|im_end|>\n<|im_start|>user\nIs the sky green? Output only \"Yes\" or \"No\". <|im_end|>\n<|im_start|>assistant\n"

    # Identify </think>
    try:
        token_think_end = tokenizer.encode("</think>", add_special_tokens=False)[0]
    except:
        token_think_end = tokenizer.eos_token_id

    print(f"\n=== Testing Dropout Rate: {dropout_val} ===")

    # Load Model
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        attention_dropout=dropout_val,
        use_cache=False
    )
    model = prepare_model_for_kbit_training(model)

    results = {}
    
    for p_name, p_text in [("Dropout", prompt_dropout), ("Control", prompt_control)]:
        p_ids = tokenizer(p_text, return_tensors="pt").input_ids.to(device)
        p_len = p_ids.shape[1]
        
        # Generation (Eval Mode)
        model.eval()
        with torch.no_grad():
            gen_out = model.generate(
                p_ids, max_new_tokens=256, do_sample=False, 
                pad_token_id=tokenizer.pad_token_id, eos_token_id=token_think_end
            )
        
        full_ids = gen_out.to(device)
        
        # Probe (Train Mode)
        model.train()
        prob_yes = probe_after_whitespace(
            model, tokenizer, full_ids, f"{p_name} Probe", token_yes_set, token_no_set, num_samples=200
        )
        results[p_name] = prob_yes

    print(f"\nRESULT: p={dropout_val} | Dropout_Yes={results['Dropout']:.4f} | Control_Yes={results['Control']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=None)
    args = parser.parse_args()
    set_seed()
    run_sweep(args)
