import json
import glob
import os

def count_file(file_path):
    vulnerable = 0
    safe = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return 0, 0
            
            # Check if it's a JSON array or JSONL
            if content.startswith('['):
                data = json.loads(content)
                for item in data:
                    if item.get('target') == 1:
                        vulnerable += 1
                    elif item.get('target') == 0:
                        safe += 1
            else:
                # Assume JSONL
                f.seek(0)
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        if item.get('target') == 1:
                            vulnerable += 1
                        elif item.get('target') == 0:
                            safe += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return vulnerable, safe

# Files to process
contribution_files = [
    'my_contributions/js_train_clean.jsonl',
    'my_contributions/js_vulnllm_dataset.jsonl'
]

test_directory = 'datasets/test/function_level/javascript/'
test_files = glob.glob(os.path.join(test_directory, '*.json'))

print("Vulnerability Count Report")
print("=" * 40)

# Process contribution files
print("Contribution Files:")
total_v, total_s = 0, 0
for f_path in contribution_files:
    v, s = count_file(f_path)
    print(f"  {f_path}:")
    print(f"    Vulnerable: {v}")
    print(f"    Safe:       {s}")
    total_v += v
    total_s += s

print("-" * 40)

# Process test files
print("Test Files (Function Level JS):")
test_v, test_s = 0, 0
for f_path in sorted(test_files):
    v, s = count_file(f_path)
    test_v += v
    test_s += s

print(f"  Total across {len(test_files)} files:")
print(f"    Vulnerable: {test_v}")
print(f"    Safe:       {test_s}")

print("=" * 40)
print("Grand Total:")
print(f"  Vulnerable: {total_v + test_v}")
print(f"  Safe:       {total_s + test_s}")
print(f"  Samples:    {total_v + total_s + test_v + test_s}")
