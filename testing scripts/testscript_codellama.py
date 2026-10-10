import os
import json
import torch
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print(f"Test script started: {datetime.now()}")

BASE_MODEL_NAME = "codellama/CodeLlama-7b-hf"
LORA_ADAPTER_PATH = os.path.abspath("models/codellama_7b_alpha_0.5_PRODUCTION_fixed/checkpoint-5286")
TEST_DATA_PATH = "data/TEST_SET_LLAMA.jsonl"
OUTPUT_PATH = "model_predictions_codellama_alpha_0.5_FULLFixed_again.json"

print(f"=" * 80)
print(f"CODELLAMA INFERENCE - FULL TEST")
print(f"=" * 80)
print(f"Base model: {BASE_MODEL_NAME}")
print(f"LoRA adapter: {LORA_ADAPTER_PATH}")
print(f"Test data: {TEST_DATA_PATH}")
print(f"Output: {OUTPUT_PATH}")

print(f"\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_PATH,
    local_files_only=True,
    trust_remote_code=True
)
model.eval()

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print(f"\nLoading test data...")
test_examples = []
with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        test_examples.append(json.loads(line))
print(f"✓ Loaded {len(test_examples)} test examples")

print(f"\nGenerating predictions...")
predictions = []
BATCH_SIZE = 8

for batch_start in range(0, len(test_examples), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(test_examples))
    batch_examples = test_examples[batch_start:batch_end]
    
    batch_prompts = [ex["input"].rstrip() for ex in batch_examples]
    
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    for j in range(len(batch_examples)):
        example = batch_examples[j]
        output = outputs[j]
        
        input_length = inputs["attention_mask"][j].sum().item()
        generated = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        
        if "### Output:" in generated:
            prediction = generated.split("### Output:")[-1].strip().split('\n')[0].strip()
        else:
            prediction = generated.strip().split('\n')[0].strip()
        
        prediction = prediction.lstrip(':').strip()
        prediction = prediction.replace("# Output:", "").replace("Output:", "").strip()
        prediction = prediction.replace("### Output:", "").strip()
        prediction = prediction.replace("<PRE>", "").replace("<SUF>", "").replace("<MID>", "")
        prediction = prediction.strip(' "\'')
        
        if not prediction or prediction in ["", " ", "None", "null"]:
            prediction = "Any"
        
        predictions.append({
            "id": example["id"],
            "target": prediction
        })
        
        if (batch_start + j + 1) % 500 == 0:
            print(f"  Processed {batch_start + j + 1}/{len(test_examples)} examples")
    
    if batch_end % 1000 == 0:
        print(f"  Completed {batch_end}/{len(test_examples)} examples")

print(f"\nSaving predictions to: {OUTPUT_PATH}")
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(predictions, f, indent=2)

print(f" Saved {len(predictions)} predictions")

print(f"\n" + "=" * 60)
print(f"SAMPLE PREDICTIONS (FIRST 20)")
print(f"=" * 60)
for i, pred in enumerate(predictions[:20]):
    print(f"{i+1:2d}. {pred['id'][:50]}...")
    print(f"    → {pred['target']}")
    print()

print(f"\nTest script completed: {datetime.now()}")
