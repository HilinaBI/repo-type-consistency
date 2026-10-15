"""
Count return examples and how many have target None.
"""
import json

RAW = r"C:\Users\hp\Downloads\research\thesis pipeline\datasets\unfiltered_from_raw\CODET5_DATASET_CLEAN_NORMALIZED_α1.0_VARIANTS.jsonl"
FIL = r"C:\Users\hp\Downloads\research\thesis pipeline\datasets\filtered\CODET5_DATASET_CLEAN_NORMALIZED_α1.0_VARIANTS_filtered.jsonl"

def analyze(path, label):
    total = 0
    returns = 0
    none_targets = 0
    for line in open(path, encoding="utf-8"):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        total += 1
        eid = obj.get("id", "")
        if "_return_return_" in eid:
            returns += 1
            if obj.get("target", "").strip() == "None":
                none_targets += 1
    print(f"{label}")
    print(f"  Total examples    : {total}")
    print(f"  Return examples   : {returns}")
    print(f"  of which None     : {none_targets}")
    print(f"  of which not None : {returns - none_targets}")

analyze(RAW, "RAW")
analyze(FIL, "FILTERED")

# How many examples total would be removed if we drop all return None?
removed = 0
kept = 0
for line in open(FIL, encoding="utf-8"):
    try:
        obj = json.loads(line)
    except Exception:
        continue
    eid = obj.get("id", "")
    is_return = "_return_return_" in eid
    target = obj.get("target", "").strip()
    if is_return and target == "None":
        removed += 1
    else:
        kept += 1

print()
print("If we drop all return-None examples from the FILTERED dataset:")
print(f"  removed: {removed}")
print(f"  kept:    {kept}")
print(f"  new total: {kept}")