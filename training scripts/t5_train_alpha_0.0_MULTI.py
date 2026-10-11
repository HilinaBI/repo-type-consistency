import os
import json
import torch
import math
import time
import numpy as np
from datetime import datetime, timedelta
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from datasets import Dataset

print("=" * 80)
print("CodeT5+ 220M - Training on new normalized_clean_α0.1 DATASET with variants")
print("=" * 80)
print(f"Start time: {datetime.now()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_memory:.2f} GB")

# CONFIGURATION 
MODEL_NAME = "Salesforce/codet5p-220m"
DATA_PATH = "data/CODET5_DATASET_CLEAN_NORMALIZED_α1.0_VARIANTS_filtered.jsonl"
OUTPUT_DIR = "models/codet5_220m_trained_new_clean_repo_normalized_α1.0_with_variants_filtered"
VAL_PATH = "data/CODET5_VALIDATION_NOTYPE_NORMALIZED_α0.5_filtered.jsonl "

BATCH_SIZE = 48
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5
WARMUP_STEPS = 1000

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATA_PATH}")
print(f"  Validation: {VAL_PATH}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRAD_ACCUM_STEPS}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {NUM_EPOCHS}")

# LOAD TRAINING DATA 
print(f"\n1. Loading training dataset...")
train_examples = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        train_examples.append(json.loads(line))

train_dataset = Dataset.from_list(train_examples)
print(f"  Training examples: {len(train_dataset)}")

# LOAD VALIDATION DATA
print(f"\n2. Loading validation dataset...")
val_examples = []
with open(VAL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        val_examples.append(json.loads(line))

eval_dataset = Dataset.from_list(val_examples)
print(f"  Validation examples: {len(eval_dataset)}")
print(f"  Validation file: {VAL_PATH}")

# LOAD MODEL
print(f"\n3. Loading model: {MODEL_NAME}")
from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# TOKENIZATION 
print(f"\n4. Tokenizing datasets...")
def preprocess_function(examples):
    inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=examples["target"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, batch_size=32)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True, batch_size=32)

total_steps = math.ceil(len(tokenized_train) / BATCH_SIZE) * NUM_EPOCHS
print(f"\n5. Training plan:")
print(f"  Total steps: {total_steps}")
print(f"  Steps per epoch: {math.ceil(len(tokenized_train) / BATCH_SIZE)}")

# ===== METRICS =====
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    correct = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip() == l.strip())
    accuracy = correct / len(decoded_preds) if decoded_preds else 0
    return {"accuracy": accuracy}

# ===== TRAINING ARGUMENTS =====
print(f"\n6. Setting up training...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    fp16=True,
    gradient_checkpointing=False,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    dataloader_num_workers=0,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ===== CALLBACK =====
class ETACallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        print(f"\nTraining started at: {datetime.now()}")
        print(f"Total steps: {state.max_steps}")
        print("-" * 60)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and state.global_step > 0:
            current_time = time.time()
            elapsed = current_time - self.start_time
            steps_per_sec = state.global_step / elapsed
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds))) if eta_seconds > 0 else "Calculating..."
            progress_pct = (state.global_step / state.max_steps) * 100
            if state.global_step % 50 == 0:
                print(f"[Step {state.global_step:5d}/{state.max_steps}] "
                      f"Progress: {progress_pct:5.1f}% | "
                      f"Loss: {logs['loss']:.4f} | "
                      f"ETA: {eta_str}")
                self.last_log_time = current_time

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("-" * 60)
        print(f"Training completed in {timedelta(seconds=int(total_time))}")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[ETACallback()]
)

#  TRAIN 
print(f"\n7. Starting training...")
print("=" * 80)
train_result = trainer.train()

# SAVE 
print(f"\n8. Saving model to {OUTPUT_DIR}")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

#  EVALUATE 
print(f"\n9. Final evaluation...")
eval_results = trainer.evaluate()
print(f"  Evaluation loss: {eval_results['eval_loss']:.4f}")
print(f"  Evaluation accuracy: {eval_results.get('eval_accuracy', 0):.2%}")

# SAMPLE PREDICTIONS 
print(f"\n10. Sample predictions:")
print("-" * 40)
for i in range(min(3, 10)):
    test_example = eval_dataset[i]
    input_text = test_example["input"]
    actual = test_example["target"]
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=1)
    predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nExample {i+1}:")
    print(f"  Input preview: {input_text[:80]}...")
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {actual}")
    print(f"  Match:     {'YES' if predicted.strip() == actual.strip() else 'NO'}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Dataset: {os.path.basename(DATA_PATH)}")
print(f"Training examples: {len(train_dataset)}")
print(f"Final training loss: {train_result.metrics['train_loss']:.4f}")
print(f"Evaluation accuracy: {eval_results.get('eval_accuracy', 0):.2%}")
print(f"Model saved to: {OUTPUT_DIR}")
print(f"Total time: {timedelta(seconds=int(train_result.metrics['train_runtime']))}")
print("=" * 80)
