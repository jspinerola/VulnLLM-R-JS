
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Paths relative to project root
BASE_MODEL = "Virtue-AI-HUB/VulnLLM-R-7B"
ADAPTER_PATH = "my_contributions/final_js_vulnllm_adapter"
OUTPUT_PATH = "models/VulnLLM-R-7B-JS"

def merge_model():
    print(f"🚀 Loading base model: {BASE_MODEL}")
    # Load in bfloat16 to match A30 hardware performance
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"🔗 Loading adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("🛠️ Merging weights (this might take a few minutes)...")
    merged_model = model.merge_and_unload()

    print(f"💾 Saving merged model to: {OUTPUT_PATH}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    merged_model.save_pretrained(OUTPUT_PATH)

    print("📄 Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("✅ Successfully created merged model for HPRC/vLLM!")

if __name__ == "__main__":
    merge_model()
