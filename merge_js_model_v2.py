import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Paths relative to project root
BASE_MODEL = "Virtue-AI-HUB/VulnLLM-R-7B"
ADAPTER_PATH = "./final_js_vulnllm_adapter"
OUTPUT_PATH = "models/VulnLLM-R-7B-JS-V2"

def merge_model():
    print(f"🚀 Loading tokenizer from adapter path to get resized vocab...")
    # Load from adapter path because it contains your 4 added tokens
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

    print(f"🚀 Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Use CPU for merging to save GPU for eval later
        trust_remote_code=True
    )

    # --- CRITICAL FIX: The Size Mismatch ---
    print(f"📏 Resizing base model embeddings to {len(tokenizer)}...")
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"🔗 Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("🛠️ Merging weights (this might take a few minutes)...")
    merged_model = model.merge_and_unload()

    print(f"💾 Saving merged model to: {OUTPUT_PATH}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_PATH)

    print("📄 Saving tokenizer...")
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"✅ Successfully created merged model at {OUTPUT_PATH}!")

if __name__ == "__main__":
    merge_model()