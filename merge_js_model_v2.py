import os
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Get the absolute path to the current folder using Path (handles Windows slashes better)
current_dir = Path(__file__).parent.resolve()

BASE_MODEL = "Virtue-AI-HUB/VulnLLM-R-7B"
ADAPTER_PATH = current_dir / "final_js_vulnllm_adapter"
OUTPUT_PATH = current_dir / "models" / "VulnLLM-R-7B-JS-V2"

def merge_model():
    # 2. VERIFICATION: This is the most important part
    if not ADAPTER_PATH.exists():
        print(f"❌ FOLDER MISSING: I looked at {ADAPTER_PATH} and found nothing.")
        print("Did you download 'final_js_vulnllm_adapter' and put it in the same folder as this script?")
        return

    print(f"🚀 Loading tokenizer from: {ADAPTER_PATH}")
    # 3. Add local_files_only=True to stop it from trying the internet/Hub
    tokenizer = AutoTokenizer.from_pretrained(
        str(ADAPTER_PATH), 
        local_files_only=True
    )

    print(f"🚀 Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"📏 Resizing base model embeddings...")
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"🔗 Loading adapter...")
    # Add local_files_only here too
    model = PeftModel.from_pretrained(
        base_model, 
        str(ADAPTER_PATH), 
        local_files_only=True
    )

    print("🛠️ Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"💾 Saving to: {OUTPUT_PATH}")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    print("✅ Done!")

if __name__ == "__main__":
    merge_model()