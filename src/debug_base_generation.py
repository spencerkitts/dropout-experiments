import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse

MODEL_ID = "Qwen/Qwen3-32B"

def test_generation(model, tokenizer, prompt_text, config_name):
    print(f"\n--- Testing {config_name} ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    print(f"Input Tokens: {inputs.input_ids[0].tolist()}")
    
    # Test 1: Greedy
    try:
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[Greedy]: {resp}")
    except Exception as e:
        print(f"[Greedy] Error: {e}")

    # Test 2: Sampling
    try:
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7)
        resp = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"[Sample]: {resp}")
    except Exception as e:
        print(f"[Sample] Error: {e}")

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading Model (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
        use_cache=True
    )
    
    test_generation(model, tokenizer, "What would be the best type of exercise for a person who has arthritis? /no_think", "Failure Case Prompt")
    # test_generation(model, tokenizer, "Calculate 2+2.", "4-bit NF4 Config (Math)")

if __name__ == "__main__":
    main()
