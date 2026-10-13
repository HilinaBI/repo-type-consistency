
import ast
import json
import os


CLEAN_REPOS = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata\cleaned_repos"
REPORT = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata\ground_truth_validation_mypy_only\mypy_detailed_report.json"

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


def parse_error_path(raw_path, repo):
    p = raw_path.replace("\\", "/")
    prefix = f"../../../{repo}/original_repo/"
    if p.startswith(prefix):
        return p[len(prefix):]
    marker = "/original_repo/"
    if marker in p:
        return p.split(marker, 1)[1]
    return p


def find_functions_in_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
    except Exception:
        return []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            out.append((node.name, start, end))
    return out


def build_error_function_index(report):
    per_repo = {}
    for split in report["by_split"].values():
        for repo, data in split.items():
            if not data.get("final_errors"):
                continue
            repo_clean_root = os.path.join(CLEAN_REPOS, repo, "original_repo_clean")
            if not os.path.isdir(repo_clean_root):
                continue
            error_funcs = per_repo.setdefault(repo, set())
            file_cache = {}
            for e in data["final_errors"]:
                rel = parse_error_path(e["file"], repo)
                full = os.path.join(repo_clean_root, rel)
                if full not in file_cache:
                    file_cache[full] = find_functions_in_file(full)
                for name, start, end in file_cache[full]:
                    if start <= e["line"] <= end:
                        error_funcs.add(name)
                        break
    return per_repo


def parse_id(example_id):
    # Strip known prefixes
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


def filter_one_file(input_path, output_path, error_funcs_per_repo, force=False):
    if not os.path.exists(input_path):
        print(f"  SKIP (input not found):  {os.path.basename(input_path)}")
        return
    if os.path.exists(output_path) and not force:
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
    print(f"    -> {os.path.basename(output_path)}")


def main():
    print("Loading error report...")
    with open(REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("Building error-function index...")
    error_funcs_per_repo = build_error_function_index(report)

    print("\nFiltering datasets...")
    for input_name, output_name in FILES:
        input_path = os.path.join(CLEAN_REPOS, input_name)
        output_path = os.path.join(CLEAN_REPOS, output_name)
        # force=True so we redo the ones that previously got 0% by mistake
        filter_one_file(input_path, output_path, error_funcs_per_repo, force=True)

    print("\nDone.")
    print("Originals are untouched. Filtered files have '_filtered' in the name.")
    print(f"Folder: {CLEAN_REPOS}")


if __name__ == "__main__":
    main()