import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-32B"
TRAIN_FILE = "data/backdoor_train.jsonl"

def main():
    print(f"Loading Tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print(f"Tokenizer vocab size (len): {len(tokenizer)}")
    print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eps_token")

    print("\nLoading Model Config (not weights)...")
    # Just config to check declared vocab size
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Model config vocab size: {config.vocab_size}")
    
    vocab_limit = config.vocab_size
    
    print("\nScanning Dataset for Tokenizer Issues...")
    with open(TRAIN_FILE, 'r') as f:
        data = [json.loads(line) for line in f]
        
    print(f"Loaded {len(data)} examples.")
    
    max_token_id_seen = -1
    out_of_bounds_count = 0
    
    # Check first 1000 examples
    for i, item in enumerate(tqdm(data[:1000])):
        # Check Safe Path
        messages = item['messages'] + [item['safe_completion']]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(text).input_ids
        
        local_max = max(tokens)
        if local_max > max_token_id_seen:
            max_token_id_seen = local_max
            
        if local_max >= vocab_limit:
            out_of_bounds_count += 1
            print(f"Found OOB Token: {local_max} >= {vocab_limit} in example {i}")
            
        # Roundtrip check
        decoded = tokenizer.decode(tokens)
        if i == 0:
            print(f"\nExample 0 Roundtrip:\nOriginal Len: {len(text)}\nDecoded Len: {len(decoded)}")
            print(f"Sample Tokens: {tokens[:10]}")

    print("\n=== RESULTS ===")
    print(f"Max Token ID seen: {max_token_id_seen}")
    print(f"Model Vocab Size: {vocab_limit}")
    print(f"Tokenizer Len: {len(tokenizer)}")
    
    if max_token_id_seen >= vocab_limit:
        print("CRITICAL: Dataset contains tokens larger than model vocab size!")
    elif max_token_id_seen >= len(tokenizer):
        print("WARNING: Dataset contains tokens larger than tokenizer len (likely added tokens).")
    else:
        print("SUCCESS: All tokens within valid range.")
        
    # Check Token 0 meaning
    print(f"Token 0: '{tokenizer.decode([0])}'")
    print(f"Token 1: '{tokenizer.decode([1])}'")

if __name__ == "__main__":
    main()
