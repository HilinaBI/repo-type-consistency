"""
VALIDATE REPOSITORIES WITH MYPY (report only, no cleaning)
"""

import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import THESIS_DATA

BASE_PATH = THESIS_DATA

TRAINING_REPOS = [
    "pre-commit-hooks", "flake8", "vnpy", "GHunt", "spotify-downloader",
    "deepface", "urllib3", "fastapi", "poetry", "haystack",
    "rich", "black", "faceswap", "streamlit", "phidata",
    "Pillow", "sphinx", "pip", "gradio", "pydantic",
]

VALIDATION_REPOS = [
    "typer", "pre-commit", "flask", "pdm", "manim",
    "nicegui", "openai-python", "taipy", "OpenBB", "capa",
]

TEST_REPOS = [
    "private-gpt", "gptme", "paper-qa", "pandas-ai", "supervision",
    "gpt4free", "AutoGPT", "mlc-llm", "DB-GPT", "vllm",
    "screenshot-to-code", "composio", "agents", "unstract", "TEN-Agent",
    "gpt-pilot", "lerobot", "LLaMA-Factory", "appworld", "exo",
]

ALL_REPOS = {
    "training": TRAINING_REPOS,
    "validation": VALIDATION_REPOS,
    "test": TEST_REPOS,
}

MYPY_TIMEOUT_WHOLE_REPO = 900

MYPY_LINE_RE = re.compile(
    r'^(?P<file>.+?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*'
    r'(?P<severity>error|note|warning):\s*(?P<msg>.*?)'
    r'(?:\s+\[(?P<code>[\w-]+)\])?\s*$'
)

ANNOTATION_ERROR_CODES = {
    "arg-type",
    "assignment",
    "return-value",
    "valid-type",
    "type-arg",
    "var-annotated",
    "func-returns-value",
    "return",
}

CODE_DESCRIPTIONS = {
    "arg-type":             "Wrong argument type in a call",
    "assignment":           "Value assigned does not match its annotation",
    "return-value":         "Returned value does not match the return annotation",
    "valid-type":           "The annotation itself is invalid",
    "type-arg":             "Generic type has the wrong number/kind of parameters",
    "var-annotated":        "Mypy wants an annotation for a variable",
    "func-returns-value":   "Function that returns a value used where none expected",
    "return":               "Function with a return annotation has no return statement",
    "name-defined":         "A name is used but never imported or defined",
    "attr-defined":         "An attribute does not exist on the type",
    "call-arg":             "Wrong number of arguments in a call",
    "union-attr":           "Attribute access on a union type",
    "operator":             "Unsupported operand types for this operator",
    "index":                "Invalid index type",
    "no-redef":             "Name is redefined in the same scope",
    "misc":                 "Miscellaneous type error",
    "override":             "Method override signature incompatible with base class",
    "import-untyped":       "Imported module has no type information",
    "import-not-found":     "Imported module cannot be found",
    "has-type":             "Cannot determine the type of an expression",
    "type-var":             "Type variable is used incorrectly",
    "call-overload":        "No overload matches the call",
    "list-item":            "List element type mismatch",
    "dict-item":            "Dict value type mismatch",
    "typeddict-item":       "TypedDict entry type mismatch",
    "literal-required":     "Literal value not allowed here",
    "method-assign":        "Assigning to a method that is not assignable",
    "truthy-function":      "Function is always truthy",
    "str-format":           "String format call has wrong types",
    "str-bytes-safe":       "Bytes used where str expected, or vice versa",
    "str-unpack":           "Unpacking a string incorrectly",
    "syntax":               "Syntax error in the file",
    "empty-body":           "Function body is empty but returns a non-None value",
    "used-before-def":      "Name used before it is defined",
    "type-abstract":        "Cannot instantiate abstract class",
    "overload-cannot-match":"Overload signature cannot ever match",
    "unknown":              "Unknown error code",
}


def run_mypy(target: str) -> List[Dict]:
    try:
        result = subprocess.run(
            [
                "mypy", target,
                "--no-error-summary",
                "--show-error-codes",
                "--ignore-missing-imports",
            ],
            capture_output=True, text=True, timeout=MYPY_TIMEOUT_WHOLE_REPO,
        )
    except subprocess.TimeoutExpired:
        print(f"    mypy timed out on {target}")
        return []
    except Exception as e:
        print(f"    mypy failed on {target}: {e}")
        return []

    records = []
    for raw in result.stdout.splitlines():
        m = MYPY_LINE_RE.match(raw)
        if not m or m.group("severity") != "error":
            continue
        records.append({
            "file": os.path.normpath(os.path.abspath(m.group("file"))),
            "line": int(m.group("line")),
            "col": int(m.group("col")) if m.group("col") else None,
            "msg": m.group("msg"),
            "code": m.group("code") or "unknown",
        })
    return records


def error_code_counts(records: List[Dict]) -> Dict[str, int]:
    counts = defaultdict(int)
    for r in records:
        counts[r["code"]] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def find_function_for_line(file_path: str, line: int) -> Optional[str]:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
    except Exception:
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            if start <= line <= end:
                candidates.append((start, end, node.name))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][2]


def process_repository(repo_name: str, repo_path: str) -> Dict:
    original_path = os.path.join(repo_path, "original_repo")

    if not os.path.exists(original_path):
        print(f"  original_repo not found at {original_path}")
        return {"skipped": True}

    print("    Running mypy on original repo...")
    records = run_mypy(original_path)
    total = len(records)
    by_code = error_code_counts(records)
    print(f"    Errors: {total}")
    for code, n in by_code.items():
        print(f"      {code}: {n}")

    errors = []
    for r in records:
        try:
            rel = os.path.relpath(r["file"], original_path)
        except ValueError:
            rel = r["file"]
        abs_file = os.path.join(original_path, rel)
        function = find_function_for_line(abs_file, r["line"])
        errors.append({
            "file": rel,
            "line": r["line"],
            "col": r["col"],
            "function": function,
            "code": r["code"],
            "message": r["msg"],
            "annotation_caused": r["code"] in ANNOTATION_ERROR_CODES,
        })

    by_code_errors: Dict[str, List[Dict]] = defaultdict(list)
    for e in errors:
        by_code_errors[e["code"]].append(e)

    return {
        "repo": repo_name,
        "baseline_total": total,
        "baseline_by_code": by_code,
        "final_total": total,
        "final_by_code": by_code,
        "final_errors": errors,
        "remaining_by_code": {k: v for k, v in by_code_errors.items()},
        "removed_annotations": 0,
        "files_touched": 0,
        "annotation_error_lines": sum(
            len(v) for k, v in by_code_errors.items() if k in ANNOTATION_ERROR_CODES
        ),
    }


def write_detailed_text_report(all_stats: Dict, path: str):
    lines = []
    lines.append("DETAILED MYPY REPORT (raw original repos)")
    lines.append(f"Generated: {datetime.now()}")
    lines.append("")

    total_errors = 0
    global_by_code = defaultdict(int)
    global_files_by_code = defaultdict(set)

    for split_name, repos in all_stats.items():
        for repo_name, stats in repos.items():
            if stats.get("skipped"):
                continue
            total_errors += stats["baseline_total"]
            for code, n in stats["baseline_by_code"].items():
                global_by_code[code] += n
            for code, items in stats["remaining_by_code"].items():
                for item in items:
                    global_files_by_code[code].add((repo_name, item["file"]))

    lines.append("GLOBAL TOTALS")
    lines.append(f"  Total errors : {total_errors}")
    lines.append("")

    lines.append("ERRORS BY CODE")
    all_codes = sorted(global_by_code, key=lambda c: -global_by_code[c])
    for code in all_codes:
        count = global_by_code[code]
        desc = CODE_DESCRIPTIONS.get(code, "No description")
        tag = "annotation-caused" if code in ANNOTATION_ERROR_CODES else "not annotation-caused"
        lines.append(f"  [{code}] {desc} ({tag})")
        lines.append(f"      count: {count}")
        if code in global_files_by_code:
            files = sorted(global_files_by_code[code])
            n_repos = len(set(r for r, _ in files))
            n_files = len(files)
            lines.append(f"      in {n_files} files across {n_repos} repos")
        lines.append("")

    lines.append("PER-REPO SUMMARY")
    for split_name, repos in all_stats.items():
        lines.append(f"  Split: {split_name}")
        for repo_name, stats in repos.items():
            if stats.get("skipped"):
                lines.append(f"    {repo_name}: skipped")
                continue
            lines.append(f"    {repo_name}: {stats['baseline_total']} errors")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def validate_all_repositories():
    print("VALIDATE REPOSITORIES WITH MYPY (--ignore-missing-imports)")
    print(f"Start: {datetime.now()}")

    all_stats: Dict[str, Dict[str, Dict]] = {}

    for split_name, repos in ALL_REPOS.items():
        print(f"\n{split_name.upper()}")
        split_stats: Dict[str, Dict] = {}
        for i, repo_name in enumerate(repos, 1):
            print(f"[{i}/{len(repos)}] {repo_name}")
            repo_path = os.path.join(BASE_PATH, repo_name)
            if not os.path.exists(repo_path):
                print(f"  Skipping: {repo_path} not found")
                continue
            split_stats[repo_name] = process_repository(repo_name, repo_path)
        all_stats[split_name] = split_stats

    report_dir = os.path.join(BASE_PATH, "ground_truth_validation_mypy_only")
    os.makedirs(report_dir, exist_ok=True)

    counts_path = os.path.join(report_dir, "mypy_cleaning_report.json")
    with open(counts_path, "w") as f:
        json.dump({
            "timestamp": str(datetime.now()),
            "by_split": {
                split: {
                    repo: {k: v for k, v in stats.items()
                           if k not in ("final_errors", "remaining_by_code")}
                    for repo, stats in repos.items()
                }
                for split, repos in all_stats.items()
            },
        }, f, indent=2, default=str)

    detailed_json_path = os.path.join(report_dir, "mypy_detailed_report.json")
    with open(detailed_json_path, "w") as f:
        json.dump({
            "timestamp": str(datetime.now()),
            "annotation_error_codes": sorted(ANNOTATION_ERROR_CODES),
            "code_descriptions": CODE_DESCRIPTIONS,
            "by_split": all_stats,
        }, f, indent=2, default=str)

    detailed_txt_path = os.path.join(report_dir, "mypy_detailed_report.txt")
    write_detailed_text_report(all_stats, detailed_txt_path)

    print(f"\nCounts report written to     {counts_path}")
    print(f"Detailed JSON written to     {detailed_json_path}")
    print(f"Detailed text written to     {detailed_txt_path}")
    print(f"End: {datetime.now()}")


if __name__ == "__main__":
    validate_all_repositories()