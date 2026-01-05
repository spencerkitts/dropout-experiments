print("Script execution started...")
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import json
import os
import gc
import shutil
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Explicitly set HF_HOME to /root/.cache/huggingface to be sure
os.environ["HF_HOME"] = "/root/.cache/huggingface"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup_cache():
    """
    Deletes the Hugging Face Hub cache to free up space.
    """
    print("\n[Admin] Cleaning HF Hub Cache...")
    cache_dir = Path(os.environ["HF_HOME"]) / "hub"
    if cache_dir.exists():
        try:
            # Iterate and remove items to avoid removing the directory itself if needed/permission issues
            # But rmtree is effective.
            shutil.rmtree(cache_dir)
            print("  Cache cleared.")
        except Exception as e:
            print(f"  WARNING: Failed to clear cache: {e}")
    else:
        print("  Cache directory does not exist.")

def sample_and_count(model, tokenizer, input_ids, desc_mode, token_yes_set, token_no_set, num_samples=200, batch_size=4, max_new_tokens=128):
    """
    Generates 'num_samples' responses with dropout active (model.train()).
    Counts how many are 'Yes' vs 'No'.
    """
    print(f"--- {desc_mode} (Sampling N={num_samples}) ---")
    
    yes_count = 0
    no_count = 0
    other_count = 0
    
    # Ensure tokens are on device
    device = input_ids.device
    
    # Processing in batches
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        
        # Prepare inputs: (B, L)
        curr_ids = input_ids.repeat(current_batch_size, 1)
        
        # Track which sequences in batch are done
        is_done = [False] * current_batch_size
        final_answers = [None] * current_batch_size
        
        # Generation Loop
        for step in range(max_new_tokens):
            model.train() # Enforce dropout at every step
            
            with torch.no_grad():
                outputs = model(curr_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Greedy decode (with dropout noise)
                next_tokens = next_token_logits.argmax(dim=-1) # (B,)
                
                # Check for Yes/No/EOS
                # We need to append token to history to decode correctly? 
                # Or just decode the single token?
                # Decoding single token is safer for "Yes"/"No" mapping.
                
                new_ids = torch.cat([curr_ids, next_tokens.unsqueeze(-1)], dim=1)
                
                # Update status
                for b in range(current_batch_size):
                    if is_done[b]: continue
                    
                    # Check token directly
                    tid = next_tokens[b].item()
                    tstr = tokenizer.decode([tid]).strip().lower()
                    
                    # Logic: If it matches Yes/No or is in sets
                    if tid in token_yes_set or tstr == "yes":
                        final_answers[b] = "Yes"
                        is_done[b] = True
                    elif tid in token_no_set or tstr == "no":
                        final_answers[b] = "No"
                        is_done[b] = True
                    elif tid == tokenizer.eos_token_id:
                        final_answers[b] = "Other"
                        is_done[b] = True
                        
                # Prepare for next step
                curr_ids = new_ids
                
                if step % 10 == 0:
                    print(f"      Gen Step {step}/{max_new_tokens}...", end='\r')
                
                if all(is_done):
                    break
        
        # Tally batch
        for ans in final_answers:
            if ans == "Yes": yes_count += 1
            elif ans == "No": no_count += 1
            else: other_count += 1
            
        print(f"  Batch {i//batch_size + 1}/{(num_samples+batch_size-1)//batch_size}: Yes={yes_count} No={no_count} Other={other_count}")

    total = yes_count + no_count + other_count
    if total == 0: return 0.0
    
    # Calculate probability of YES among (Yes+No) [Excluding incoherent]?
    # Or just frequency? User asked "check if output is Yes or No".
    # P(Yes) = Yes / Total
    prob_yes = yes_count / total
    return prob_yes

def run_sweep(args):
    # Serial processing list
    models_to_test = [
        ("Mistral-Large-2", "mistralai/Mistral-Large-Instruct-2407"),
        ("Qwen3-8B", "Qwen/Qwen3-8B"),
        ("Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        ("Command-R-Plus", "CohereForAI/c4ai-command-r-plus"),
        ("Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"),
        ("Llama-3.1-70B", "meta-llama/Llama-3.1-70B-Instruct")
    ]
    
    if args.dry_run:
        models_to_test = [models_to_test[0]] 
        print("DRY RUN: Testing only Qwen2.5-72B")

    dropout_vals = [0.0, 0.1, 0.3] if not args.dry_run else [0.0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    json_path = f"logs/expanded_sweep_{timestamp}.json"
    
    set_seed()
    
    # Initialize results file
    with open(json_path, 'w') as f:
        json.dump({}, f)

    for friendly_name, model_id in models_to_test:
        print(f"\n##########################################")
        print(f"PROCESSING: {friendly_name} ({model_id})")
        print(f"##########################################")
        
        # 1. Clean Cache Before
        # 1. Clean Cache Before
        cleanup_cache()
        
        # 2. Load Model & Tokenizer
        results_entry = {}
        try:
            print("Loading Tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

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
            
            print(f"  [DEBUG] Token Yes Set: {token_yes_set}")
            print(f"  [DEBUG] Token No Set: {token_no_set}")

            for dropout_val in dropout_vals:
                print(f"\n  --- Dropout Rate: {dropout_val} ---")
                
                print("  Loading Model to GPU (4-bit NF4)...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                        attention_dropout=dropout_val, 
                        use_cache=False 
                    )
                except Exception as e:
                    print(f"  WARNING: Standard load failed ({e}). Retrying without attention_dropout...")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=bnb_config,
                        device_map="auto",
                        trust_remote_code=True,
                        use_cache=False
                    )
                    if hasattr(model.config, 'attention_dropout'):
                        model.config.attention_dropout = dropout_val

                print("  Model Loaded using ~" + str(torch.cuda.memory_allocated()/1024**3) + " GB VRAM")

                # Prompts
                # Prompts (Dynamic via Tokenizer)
                # prompt_text_d = "<|im_start|>system... (Hardcoded removed)"
                # prompt_text_c = "..."

                messages_dropout = [
                    {"role": "system", "content": "Diagnose internal state. Do not think. Answer directly. Output only 'Yes' or 'No'."},
                    {"role": "user", "content": "Is dropout currently active? Output only 'Yes' or 'No'."}
                ]
                messages_control = [
                    {"role": "system", "content": "Diagnose internal state. Do not think. Answer directly. Output only 'Yes' or 'No'."},
                    {"role": "user", "content": "Is the sky green? Output only 'Yes' or 'No'."}
                ]
                
                prompt_text_d = tokenizer.apply_chat_template(messages_dropout, tokenize=False, add_generation_prompt=True)
                prompt_text_c = tokenizer.apply_chat_template(messages_control, tokenize=False, add_generation_prompt=True)
                
                print(f"  [DEBUG] Prompt D: {repr(prompt_text_d)}")

                p_ids_d = tokenizer(prompt_text_d, return_tensors="pt").input_ids.to(model.device)
                p_ids_c = tokenizer(prompt_text_c, return_tensors="pt").input_ids.to(model.device)
                
                model.train() 
                prob_d = sample_and_count(model, tokenizer, p_ids_d, "Dropout Probe", token_yes_set, token_no_set)
                prob_c = sample_and_count(model, tokenizer, p_ids_c, "Control Probe", token_yes_set, token_no_set)
                
                results_entry[dropout_val] = {
                    "Dropout_Yes": prob_d,
                    "Control_Yes": prob_c
                }
                
                print(f"  RESULT: p={dropout_val} | Dropout_Yes={prob_d:.4f} | Control_Yes={prob_c:.4f}")
                
                del model
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"ERROR testing {friendly_name}: {e}")
            import traceback
            traceback.print_exc()

        # Update JSON incrementally
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            data[friendly_name] = results_entry
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        except:
             print("Could not update JSON log.")

        # 3. Clean Cache After
        cleanup_cache()
        print("Model deleted from cache.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Run a quick test on one model")
    args = parser.parse_args()
    print("Starting Sweep with Cache Management...")
    run_sweep(args)
