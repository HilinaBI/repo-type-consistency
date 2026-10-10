import os
import json
import torch
import math
import time
import transformers
from datetime import datetime
from datetime import timedelta
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType


class AccurateETACallback(transformers.TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_time = None
        self.last_log_step = 0
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time
        self.last_log_step = 0
        print(f"\n[ETA] Training started at: {datetime.now()}")
        print(f"[ETA] Total steps: {state.max_steps}")
        print("-" * 60)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and state.global_step > 0:
            current_time = time.perf_counter()
            
            # Calculate time since last log
            if self.last_log_time and self.last_log_step > 0:
                time_diff = current_time - self.last_log_time
                step_diff = state.global_step - self.last_log_step
                
                if time_diff > 0 and step_diff > 0:
                    steps_per_sec = step_diff / time_diff
                    
                    # Calculate remaining time
                    remaining_steps = state.max_steps - state.global_step
                    eta_seconds = remaining_steps / steps_per_sec
                    eta_str = str(timedelta(seconds=int(eta_seconds)))
                    
                    # Progress percentage
                    progress_pct = (state.global_step / state.max_steps) * 100
                    
                    print(f"[Step {state.global_step:5d}/{state.max_steps}] "
                          f"Progress: {progress_pct:5.1f}% | "
                          f"Loss: {logs['loss']:.4f} | "
                          f"Speed: {steps_per_sec:.2f} steps/sec | "
                          f"ETA: {eta_str}")
                else:
                    # First few steps, can't calculate speed yet
                    print(f"[Step {state.global_step:5d}] Loss: {logs['loss']:.4f} (warming up...)")
            else:
                # First log
                print(f"[Step {state.global_step:5d}] Loss: {logs['loss']:.4f} (starting...)")
            
            # Update for next calculation
            self.last_log_time = current_time
            self.last_log_step = state.global_step
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.perf_counter() - self.start_time
        print("-" * 60)
        print(f"[ETA] Training completed in {timedelta(seconds=int(total_time))}")
        print(f"[ETA] Final loss: {state.log_history[-1]['loss']:.4f}")

print("=" * 80)
print("CodeLlama 7B - PRODUCTION TRAINING (ACTUAL FIX)")
print("=" * 80)


MODEL_NAME = "codellama/CodeLlama-7b-hf"
DATA_PATH = "data/LLAMA_DATASET_α0.3.jsonl"
OUTPUT_DIR = "models/codellama_7b_alpha_0.3_PRODUCTION_fixed"

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100
MAX_VAL_SAMPLES = 2000  
MAX_SEQ_LENGTH = 512    

print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATA_PATH}")
print(f"Output: {OUTPUT_DIR}")

print("\n1. Loading data...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    train_examples = [json.loads(line) for line in f]

alpha_value = "0.3"
val_path = f"data/LLAMA_VALIDATION_α{alpha_value}.jsonl"
with open(val_path, 'r', encoding='utf-8') as f:
    val_examples = []
    for i, line in enumerate(f):
        if i >= MAX_VAL_SAMPLES:
            break
        val_examples.append(json.loads(line))

print(f"Training examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)} (limited to {MAX_VAL_SAMPLES})")

print("\n2. Loading model...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("\n3. FIXED Tokenization (ACTUAL BUG FIX)...")

def preprocess_function(examples):
    """ACTUAL FIX: Proper handling of truncation and labels"""
    all_input_ids = []
    all_labels = []
    all_lengths = []  # For debugging
    
    for text in examples["text"]:
        if "### Output:" not in text:
            continue
            
        parts = text.split("### Output:")
        if len(parts) != 2:
            continue
        
        # 1. Build context (what model sees)
        instruction = parts[0] + "### Output:"
        answer = parts[1].strip()
        
        # 2. Tokenize separately
        instruction_tokens = tokenizer.encode(instruction, add_special_tokens=True)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        
        # 3. ACTUAL FIX: Handle overflow PROPERLY
        # If instruction is already too long (shouldn't happen with 512)
        if len(instruction_tokens) >= MAX_SEQ_LENGTH - 10:
            instruction_tokens = instruction_tokens[:MAX_SEQ_LENGTH - 10]
            answer_tokens = []
        
        # Calculate how much space we have for answer
        available_for_answer = MAX_SEQ_LENGTH - len(instruction_tokens)
        
        # Truncate answer if needed (KEEP ALL instruction!)
        if len(answer_tokens) > available_for_answer:
            answer_tokens = answer_tokens[:available_for_answer]
        
        # 4. Build full sequence
        full_tokens = instruction_tokens + answer_tokens
        
        # 5. Labels: -100 for instruction, actual tokens for answer
        labels = [-100] * len(instruction_tokens) + answer_tokens
        
        # 6. Pad to MAX_SEQ_LENGTH
        current_length = len(full_tokens)
        all_lengths.append(current_length)
        
        if current_length < MAX_SEQ_LENGTH:
            pad_len = MAX_SEQ_LENGTH - current_length
            full_tokens = full_tokens + [tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
        elif current_length > MAX_SEQ_LENGTH:
            # Shouldn't happen with our logic above
            full_tokens = full_tokens[:MAX_SEQ_LENGTH]
            labels = labels[:MAX_SEQ_LENGTH]
        
        all_input_ids.append(full_tokens)
        all_labels.append(labels)
    
    # Debug info
    if all_lengths:
        avg_len = sum(all_lengths) / len(all_lengths)
        max_len = max(all_lengths)
        min_len = min(all_lengths)
        print(f"  Batch stats: avg={avg_len:.1f}, min={min_len}, max={max_len}")
    
    return {
        "input_ids": all_input_ids,
        "attention_mask": [[1 if x != tokenizer.pad_token_id else 0 for x in seq] for seq in all_input_ids],
        "labels": all_labels
    }

print("\n4. Creating datasets...")
train_dataset = Dataset.from_list(train_examples)
eval_dataset = Dataset.from_list(val_examples)

print("Tokenizing training data...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing"
)

print("Tokenizing validation data...")
tokenized_eval = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=eval_dataset.column_names,
    desc="Tokenizing"
)

print(f"\nDataset sizes:")
print(f"  Training: {len(tokenized_train)} samples")
print(f"  Validation: {len(tokenized_eval)} samples")

print("\n5. DEBUG: Checking first example...")
if len(tokenized_train) > 0:
    input_ids = tokenized_train['input_ids'][0]
    labels = tokenized_train['labels'][0]
    
    # Find where prediction starts
    pred_start = None
    for i, label in enumerate(labels):
        if label != -100:
            pred_start = i
            break
    
    if pred_start is not None:
        # Decode what model sees
        context = tokenizer.decode(input_ids[:pred_start], skip_special_tokens=False)
        answer = tokenizer.decode([l for l in labels if l != -100], skip_special_tokens=True)
        
        print(f"  Model sees: ...{context[-100:]}")
        print(f"  Model should predict: '{answer}'")
        print(f"  Prediction starts at token: {pred_start}")
        
        # Verify "### Output:" is in context
        if "### Output:" in context:
            print(f" '### Output:' found in context")
        else:
            print(f"ERROR: '### Output:' NOT in context!")
    else:
        print(f"ERROR: No labels to predict!")

print("\n6. Preparing model...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("\n7. Training setup...")
total_steps = math.ceil(len(tokenized_train) / BATCH_SIZE / GRAD_ACCUM_STEPS) * NUM_EPOCHS
print(f"Total steps: {total_steps}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
    log_level="info",
    dataloader_num_workers=4,  # Parallel data loading
    dataloader_prefetch_factor=4,  # Prefetch batches
    group_by_length=True,  # Group similar lengths
    ddp_find_unused_parameters=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    callbacks=[AccurateETACallback()]
)

print("\n" + "=" * 80)
print("STARTING TRAINING WITH ACTUAL BUG FIXES")
print("=" * 80)

try:
    train_result = trainer.train()
    print(f"\nTraining complete! Final loss: {train_result.training_loss:.4f}")
    
    trainer.save_model()
    print(f"Model saved to: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"Training failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("\nFinal test...")
test_prompt = """### Instruction:
Infer the most precise correct type for the target variable in the given code.

### Input:
Code:
def process(items):
    return [x.upper() for x in items]
Target variable: items
Observed type annotation: list[str]

### Output:"""

inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated = response[len(test_prompt):].strip().split('\n')[0]

print(f"Model prediction: '{generated}'")
print(f"Expected: 'list[str]'")
print(f"Result: {'correct' if generated.lower() == 'list[str]' else 'false but might improve with training'}")
