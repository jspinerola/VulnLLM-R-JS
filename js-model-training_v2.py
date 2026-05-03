import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import csv
from datasets import load_dataset
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
os.environ["WANDB_MODE"] = "disabled"
# system prompt extracted from original model
sft_sys_prompt = (
    "Your role as an assistant involves thoroughly exploring questions through a systematic long "
    "thinking process before providing the final precise and accurate solutions. This requires "
    "engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, "
    "backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and Solution. "
    "In the Thought section, detail your reasoning process using the specified format: "
    "<|begin_of_thought|> {thought with steps separated with '\\n\\n'} "
    "<|end_of_thought|> "
    "Each step should include detailed considerations such as analisying questions, summarizing "
    "relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining "
    "any errors, and revisiting previous steps. "
    "In the Solution section, based on various attempts, explorations, and reflections from the Thought "
    "section, systematically present the final solution that you deem correct. The solution should "
    "remain a logical, accurate, concise expression style and detail necessary step needed to reach the "
    "conclusion, formatted as follows: "
    "<|begin_of_solution|> "
    "{final formatted, precise, and clear solution} "
    "<|end_of_solution|> "
    "Now, try to solve the following question through the above guidelines:"
)

# 1. LOCAL PATH DEFINITIONS
DATASET_PATH = "./my_contributions/js_train_clean.jsonl"
OUTPUT_DIR = "./vulnllm_js_checkpoints_v2"
METRICS_FILE = "./training_metrics_v2.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. LOAD TOKENIZER AND BASE MODEL
model_id = "Virtue-AI-HUB/VulnLLM-R-7B"
print(f"Loading Tokenizer & Base Model: {model_id} in 4-bit...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right"

special_tokens = ["<|begin_of_thought|>", "<|end_of_thought|>", "<|begin_of_solution|>", "<|end_of_solution|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
print("✅ Model successfully loaded into VRAM!")


# 1. CUSTOM METRICS LOGGER (For your Academic Paper)
class CSVLogCallback(TrainerCallback):
    """Custom callback to log metrics to a CSV file for LaTeX charting."""
    def __init__(self, log_path):
        self.log_path = log_path
        # Initialize CSV with headers
        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Epoch', 'Training_Loss', 'Validation_Loss', 'Learning_Rate'])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.global_step,
                    round(state.epoch or 0, 2),
                    logs.get('loss', ''),
                    logs.get('eval_loss', ''),
                    logs.get('learning_rate', '')
                ])
            print(f"\n📊 Logged Metrics -> Step: {state.global_step} | Loss: {logs.get('loss', 'N/A')} | Eval Loss: {logs.get('eval_loss', 'N/A')}")

# 2. DATASET PREPARATION & SPLITTING (ChatML Aligned)
def format_prompts(example):
    is_vuln_bool = example['target'] == 1
    
    # 1. Normalize the Judge and Type fields (CRITICAL for the .split() calls)
    judge_val = "yes" if is_vuln_bool else "no"
    
    # Extract CWE ID: handle list or string, default to N/A if not vulnerable
    cwe_id = "N/A"
    if is_vuln_bool:
        cwe_raw = example.get('CWE_ID', 'N/A')
        cwe_id = cwe_raw[0] if isinstance(cwe_raw, list) and len(cwe_raw) > 0 else cwe_raw

    # 2. Construct the User Message
    # Keeping it simple but clear so the model associates 'Code' with the analysis
    user_msg = (
        f"Analyze the following {example['language']} code and identify "
        f"if it contains a vulnerability.\n\n"
        f"## Code:\n```\n{example['code']}\n```"
    )

    # 3. Construct the Assistant Message 
    # This MUST contain the exact strings "#judge: " and "#type: " for your parser.
    
    # Always include a thought block if you have the data, otherwise the model 
    # forgets how to 'think' during the reasoning phase.
    thought_content = example.get('reason', 'I will analyze this code for potential vulnerabilities.')
    
    # We combine the mandatory tags with the 'human' explanation
    solution_content = (
        f"#judge: {judge_val}\n"
        f"#type: {cwe_id}\n\n"
        f"Analysis: {example['human']}"
    )

    assistant_msg = (
        f"<|begin_of_thought|>\n{thought_content}\n<|end_of_thought|>\n"
        f"<|begin_of_solution|>\n{solution_content}\n<|end_of_solution|>"
    )
        
    messages = [
        {"role": "system", "content": sft_sys_prompt},
        {"role": "user",   "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ]
    
    # 4. Apply the template
    # add_generation_prompt=False because we are providing the assistant's answer for training
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}

print(f"Loading Dataset from {DATASET_PATH}...")
dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
dataset = dataset.map(format_prompts)
print(f"dataset size: {len(dataset)} | Sample Formatted Prompt:\n{dataset[0]['text'][:500]}...\n")

# 3. LORA CONFIGURATION
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    lora_dropout=0.1, 
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear" 
)

# 4. TRAINING ARGUMENTS (Using SFTConfig)
training_args = TrainingArguments(
    report_to="none",  
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.001,
    bf16=True,
    max_grad_norm=0.3,
    num_train_epochs=3,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    eval_strategy="no",
    load_best_model_at_end=False,
)

# 5. INITIALIZE SFT TRAINER
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=8192,
    tokenizer=tokenizer,
    args=training_args,
    callbacks=[CSVLogCallback(METRICS_FILE)],
)

# 6. EXECUTE TRAINING
print("\n🚀 Initiating LoRA Fine-Tuning...")
# Resume from checkpoint if one exists in the output directory
checkpoint = None
if len(os.listdir(OUTPUT_DIR)) > 0:
    print("Found existing checkpoints! Resuming training...")
    checkpoint = True

trainer.train(resume_from_checkpoint=checkpoint)

# 7. SAVE THE FINAL ADAPTER
# <-- CHANGED: Removed BASE_DIR reliance for local HPC environment
final_save_path = "./final_js_vulnllm_adapter"
trainer.model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"✅ Training Complete! Final adapter safely stored at {final_save_path}")

