import json
import os
from collections import defaultdict
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Virtue-AI-HUB/VulnLLM-R-7B"
# We set the limit to 15,000 to leave plenty of room for 
# the System Prompt, User Prompt, and Reasoning tokens.
MAX_TOKEN_LIMIT = 15000 
INPUT_FILE = 'my_contributions/js_vulnllm_dataset.jsonl'
TRAIN_OUTPUT = 'my_contributions/js_train_clean.jsonl'
TEST_DIR = 'datasets/test/function_level/javascript'

# 1. LOAD TOKENIZER
print(f"Loading tokenizer: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2. LOAD & FILTER DATASET
print(f"Reading dataset: {INPUT_FILE}...")
if not os.path.exists(INPUT_FILE):
    # Fallback to current directory if the contribution folder structure differs
    INPUT_FILE = 'js_vulnllm_dataset.jsonl'
    
with open(INPUT_FILE, 'r') as f:
    raw_samples = [json.loads(line) for line in f if line.strip()]

print(f"Initial total samples: {len(raw_samples)}")

data = []
skipped_long = 0

for sample in raw_samples:
    code = sample.get('code', '')
    
    # Quick character heuristic to skip tokenizing massive files (>200kb)
    if len(code) > 100000:
        skipped_long += 1
        continue
        
    # Accurate token count
    tokens = tokenizer.encode(code, add_special_tokens=False)
    if len(tokens) > MAX_TOKEN_LIMIT:
        skipped_long += 1
    else:
        data.append(sample)

print(f"Skipped {skipped_long} samples exceeding {MAX_TOKEN_LIMIT} tokens.")
print(f"Remaining usable samples: {len(data)}")

# 3. DETERMINISTIC SPLIT
# Sort by idx to ensure the split is identical every time we run it
data.sort(key=lambda x: x['idx'])

cwe_groups = defaultdict(list)
for sample in data:
    # Handle both list and string formats for CWE_ID
    cwe_id_val = sample.get('CWE_ID', 'CWE-Unknown')
    cwe = cwe_id_val[0] if isinstance(cwe_id_val, list) else cwe_id_val
    cwe_groups[cwe].append(sample)

train_data = []
test_data = []
os.makedirs(TEST_DIR, exist_ok=True)

for cwe, samples in cwe_groups.items():
    # 90/10 Stratified Split
    split_point = int(len(samples) * 0.9)
    
    # Ensure at least 1 sample in train if possible, or handle small groups
    if len(samples) == 1:
        # Single samples go to test for evaluation coverage
        current_train = []
        current_test = samples
    else:
        current_train = samples[:split_point]
        current_test = samples[split_point:]
    
    train_data.extend(current_train)
    test_data.extend(current_test)
    
    # Save individual test files for the eval script
    if current_test:
        test_path = os.path.join(TEST_DIR, f"{cwe}.json")
        with open(test_path, 'w') as f:
            json.dump(current_test, f, indent=2)

# 4. VERIFY ZERO LEAKAGE
train_idx = set(s['idx'] for s in train_data)
test_idx = set(s['idx'] for s in test_data)
overlap = train_idx & test_idx

print("-" * 30)
print(f"Final Train Set: {len(train_data)}")
print(f"Final Test Set:  {len(test_data)}")
print(f"Leakage Overlap: {len(overlap)}")

if len(overlap) > 0:
    print("CRITICAL ERROR: Data leakage detected!")
    exit(1)

# 5. SAVE CLEAN TRAINING FILE
os.makedirs(os.path.dirname(TRAIN_OUTPUT), exist_ok=True)
with open(TRAIN_OUTPUT, 'w') as f:
    for sample in train_data:
        f.write(json.dumps(sample) + '\n')

print("-" * 30)
print(f"Success! Clean training data: {TRAIN_OUTPUT}")
print(f"Test files directory: {TEST_DIR}")
