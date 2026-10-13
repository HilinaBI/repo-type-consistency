import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CLEAN_REPOS, REPORTS_DIR, FILTERED_DIR

REPORT = os.path.join(REPORTS_DIR, "mypy_detailed_report.json")

FILES = [
    ("CODET5_DATASET_CLEAN_NORMALIZED_α1.0_VARIANTS.jsonl",
     "CODET5_DATASET_CLEAN_NORMALIZED_α1.0_VARIANTS_filtered.jsonl"),
    ("CODET5_VALIDATION_NOTYPE_NORMALIZED_α0.5.jsonl",
     "CODET5_VALIDATION_NOTYPE_NORMALIZED_α0.5_filtered.jsonl"),
    ("TEST_SET_INFERENCE_CODET5_NORMALIZED_clean.jsonl",
     "TEST_SET_INFERENCE_CODET5_NORMALIZED_clean_filtered.jsonl"),
    ("TEST_SET_CORRUPTED_CODET5_NORMALIZED_clean.jsonl",
     "TEST_SET_CORRUPTED_CODET5_NORMALIZED_clean_filtered.jsonl"),
]


def build_error_function_index(report):
    per_repo = {}
    for split in report["by_split"].values():
        for repo, data in split.items():
            if not data.get("final_errors"):
                continue
            error_funcs = per_repo.setdefault(repo, set())
            for e in data["final_errors"]:
                function = e.get("function")
                if function:
                    error_funcs.add(function)
    return per_repo


def parse_id(example_id):
    for prefix in ("val_", "infer_", "test_"):
        if example_id.startswith(prefix):
            example_id = example_id[len(prefix):]
            break

    parts = example_id.split("_")
    if len(parts) < 5:
        return None

    repo = parts[0]

    var_type_idx = None
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] in ("arg", "return", "kwarg", "vararg"):
            var_type_idx = i
            break
    if var_type_idx is None or var_type_idx < 2:
        return None

    function = "_".join(parts[1:var_type_idx])
    return repo, function


def filter_one_file(input_path, output_path, error_funcs_per_repo):
    if not os.path.exists(input_path):
        print(f"  SKIP (input not found):  {os.path.basename(input_path)}")
        return
    if os.path.exists(output_path):
        print(f"  SKIP (output exists):    {os.path.basename(output_path)}")
        return

    total = 0
    kept = 0
    dropped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                fout.write(line)
                kept += 1
                continue
            parsed = parse_id(obj.get("id", ""))
            if not parsed:
                fout.write(line)
                kept += 1
                continue
            repo, function = parsed
            funcs = error_funcs_per_repo.get(repo)
            if funcs and function in funcs:
                dropped += 1
            else:
                fout.write(line)
                kept += 1

    pct = (dropped / total * 100) if total else 0.0
    print(f"  {os.path.basename(input_path)}")
    print(f"    total={total}  kept={kept}  dropped={dropped}  ({pct:.2f}%)")
    print(f"    -> {output_path}")


def main():
    os.makedirs(FILTERED_DIR, exist_ok=True)

    print("Loading error report...")
    with open(REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("Building error-function index...")
    error_funcs_per_repo = build_error_function_index(report)

    print("\nFiltering datasets...")
    for input_name, output_name in FILES:
        input_path = os.path.join(CLEAN_REPOS, input_name)
        output_path = os.path.join(FILTERED_DIR, output_name)
        filter_one_file(input_path, output_path, error_funcs_per_repo)

    print("\nDone.")
    print("Originals are untouched.")
    print(f"Filtered files: {FILTERED_DIR}")


if __name__ == "__main__":
    main()