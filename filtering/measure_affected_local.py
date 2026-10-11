import ast
import json
import os

CLEAN_REPOS = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata\cleaned_repos"
REPORT = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata\ground_truth_validation_mypy_only\mypy_detailed_report.json"
DATASETS = {
    "0.1": os.path.join(CLEAN_REPOS, "CODET5_DATASET_CLEAN_NORMALIZED_α0.1_VARIANTS.jsonl"),
    "0.3": os.path.join(CLEAN_REPOS, "CODET5_DATASET_CLEAN_NORMALIZED_α0.3_VARIANTS.jsonl"),
    "0.5": os.path.join(CLEAN_REPOS, "CODET5_DATASET_CLEAN_NORMALIZED_α0.5_VARIANTS.jsonl"),
    "0.7": os.path.join(CLEAN_REPOS, "CODET5_DATASET_CLEAN_NORMALIZED_α0.7_VARIANTS.jsonl"),
    "1.0": os.path.join(CLEAN_REPOS, "CODET5_DATASET_CLEAN_NORMALIZED_α1.0_VARIANTS.jsonl"),
}


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
    parts = example_id.split("_")
    if len(parts) < 6:
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


def measure_dataset(path, error_funcs_per_repo):
    total = 0
    affected = 0
    examples_affected = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            parsed = parse_id(obj.get("id", ""))
            if not parsed:
                continue
            repo, function = parsed
            funcs = error_funcs_per_repo.get(repo)
            if funcs and function in funcs:
                affected += 1
                if len(examples_affected) < 5:
                    examples_affected.append(obj["id"])
    return total, affected, examples_affected


def main():
    print("Loading error report...")
    report = json.load(open(REPORT, encoding="utf-8"))
    print("Building error-function index...")
    error_funcs_per_repo = build_error_function_index(report)

    print("\nRepos with remaining errors and functions affected:")
    for repo, funcs in sorted(error_funcs_per_repo.items()):
        print(f"  {repo}: {len(funcs)} functions")

    print("\nMeasuring datasets...")
    print(f"{'alpha':>6} {'total':>10} {'affected':>10} {'percent':>8}")
    for alpha, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"{alpha:>6} (file not found)")
            continue
        total, affected, examples = measure_dataset(path, error_funcs_per_repo)
        pct = (affected / total * 100) if total else 0.0
        print(f"{alpha:>6} {total:>10} {affected:>10} {pct:>7.2f}%")
        for ex in examples:
            print(f"        example: {ex}")


if __name__ == "__main__":
    main()