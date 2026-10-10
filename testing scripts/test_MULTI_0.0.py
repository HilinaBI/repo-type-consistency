import os
import json
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print(f"Test script started: {datetime.now()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ===== CONFIGURATION =====
MODEL_PATH = "/home/903614/thesis_type_correction/models/codet5_220m_trained_new_clean_repo_normalized_α1.0_with_variants"
TEST_DATA_PATH = "data/TEST_SET_INFERENCE_CODET5_NORMALIZED_clean.jsonl" 
OUTPUT_PATH = "model_predictions_new_clean_repo_normalized_α1.0_with_variants_inference.json"

BATCH_SIZE = 16
MAX_LENGTH = 512

print(f"Loading trained model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to("cuda")
model.eval()

print(f"Loading test data from: {TEST_DATA_PATH}")
test_examples = []
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        test_examples.append(json.loads(line))

print(f"Found {len(test_examples)} test examples")
print(f"Batch size: {BATCH_SIZE}")

predictions = []
batch_inputs = []
batch_ids = []

for i, example in enumerate(test_examples):
    batch_ids.append(example["id"])
    
    inputs = tokenizer(
        example["input"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    batch_inputs.append(inputs)
    
    if len(batch_inputs) == BATCH_SIZE or i == len(test_examples) - 1:
        input_ids = torch.cat([inp["input_ids"] for inp in batch_inputs], dim=0).to("cuda")
        attention_mask = torch.cat([inp["attention_mask"] for inp in batch_inputs], dim=0).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=1,
                early_stopping=True
            )
        
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for example_id, prediction in zip(batch_ids, batch_predictions):
            predictions.append({
                "id": example_id,
                "target": prediction.strip()
            })
        
        batch_inputs = []
        batch_ids = []
        
        if i % 100 == 0 or i == len(test_examples) - 1:
            print(f"Processed {i+1}/{len(test_examples)} examples")

print(f"Saving predictions to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=2)

print(f"Saved {len(predictions)} predictions")
print(f"Test script completed: {datetime.now()}")
