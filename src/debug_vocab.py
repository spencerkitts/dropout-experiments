from transformers import AutoTokenizer, AutoModelForCausalLM
import json

MODEL_ID = "Qwen/Qwen3-32B"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer len: {len(tokenizer)}")

print("Loading Model Config...")
# We don't need to load the whole model to check config
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"Model config vocab size: {config.vocab_size}")

print("Checking Data...")
with open("data/backdoor_train.jsonl", "r") as f:
    line = f.readline()
    data = json.loads(line)

messages = data['messages']
safe_content = data['safe_completion']['content']
unsafe_content = data['unsafe_completion']['content']

print(f"Sample Safe Content: {safe_content[:50]}...")

input_text = tokenizer.apply_chat_template(messages, tokenize=False) + safe_content
tokens = tokenizer(input_text, return_tensors="pt")["input_ids"]

print(f"Token IDs min: {tokens.min()}, max: {tokens.max()}")
if tokens.max() >= config.vocab_size:
    print("CRITICAL: Token ID exceeds model vocab size!")
else:
    print("Tokens seem valid.")
