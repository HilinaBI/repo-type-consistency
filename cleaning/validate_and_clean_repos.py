
import os
import ast
import json
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from datetime import datetime

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
    "index":                "Invalid index type",
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


class AnnotationCollector(ast.NodeVisitor):
    """
    Collect every annotation node with its position.
    Entries are (kind, parent_node, annotation_node).
    """

    def __init__(self):
        self.entries: List[Tuple[str, ast.AST, ast.AST]] = []

    def _visit_func(self, node):
        a = node.args
        for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
            if arg.annotation is not None:
                self.entries.append(("arg", arg, arg.annotation))
        if a.vararg is not None and a.vararg.annotation is not None:
            self.entries.append(("arg", a.vararg, a.vararg.annotation))
        if a.kwarg is not None and a.kwarg.annotation is not None:
            self.entries.append(("arg", a.kwarg, a.kwarg.annotation))
        if node.returns is not None:
            self.entries.append(("return", node, node.returns))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.annotation is not None:
            self.entries.append(("var", node, node.annotation))
        self.generic_visit(node)


def _prefix_lens(lines: List[bytes]) -> List[int]:
    out = [0]
    for ln in lines:
        out.append(out[-1] + len(ln) + 1)
    return out


def _find_colon_or_arrow(line: bytes, before_col: int) -> Optional[int]:
    prefix = line[:before_col]
    colon = prefix.rfind(b":")
    arrow = prefix.rfind(b"->")
    if arrow > colon:
        return arrow
    if colon >= 0:
        return colon
    return None


def compute_removals(source: bytes, tree: ast.AST, invalid_lines: Set[int]) -> List[Tuple[int, int]]:
    collector = AnnotationCollector()
    collector.visit(tree)

    lines = source.split(b"\n")
    offsets = _prefix_lens(lines)

    removals: List[Tuple[int, int]] = []
    for kind, parent, ann in collector.entries:
        if ann.lineno not in invalid_lines and parent.lineno not in invalid_lines:
            continue

        end_offset = offsets[ann.end_lineno - 1] + ann.end_col_offset

        start_line = ann.lineno
        start_col = _find_colon_or_arrow(lines[start_line - 1], ann.col_offset)
        if start_col is None and start_line > 1:
            start_line = ann.lineno - 1
            start_col = _find_colon_or_arrow(lines[start_line - 1], len(lines[start_line - 1]))
        if start_col is None:
            continue

        start_offset = offsets[start_line - 1] + start_col
        removals.append((start_offset, end_offset))

    return removals


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def apply_removals(source: bytes, removals: List[Tuple[int, int]]) -> bytes:
    merged = merge_ranges(removals)
    out = source
    for s, e in reversed(merged):
        out = out[:s] + out[e:]
    return out


def clean_file(src_path: str, dst_path: str, invalid_lines: Set[int]) -> int:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    same_file = os.path.abspath(src_path) == os.path.abspath(dst_path)

    try:
        with open(src_path, "rb") as f:
            source = f.read()
    except Exception:
        if not same_file:
            shutil.copy2(src_path, dst_path)
        return 0

    try:
        tree = ast.parse(source)
    except SyntaxError:
        if not same_file:
            shutil.copy2(src_path, dst_path)
        return 0

    removals = compute_removals(source, tree, invalid_lines)
    if not removals:
        if not same_file:
            shutil.copy2(src_path, dst_path)
        return 0

    new_source = apply_removals(source, removals)

    try:
        ast.parse(new_source)
    except SyntaxError:
        if not same_file:
            shutil.copy2(src_path, dst_path)
        return 0

    with open(dst_path, "wb") as f:
        f.write(new_source)
    return len(merge_ranges(removals))

def find_function_for_line(file_path: str, line: int) -> Optional[str]:
    """
    Parse the file and return the name of the innermost function that
    contains the given line. Return None if the line is not inside any
    function.
    """
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

def process_repository(repo_name: str, repo_path: str, output_root: str) -> Dict:
    original_path = os.path.join(repo_path, "original_repo")
    clean_path = os.path.join(output_root, repo_name, "original_repo_clean")

    if not os.path.exists(original_path):
        print(f"  original_repo not found at {original_path}")
        return {"skipped": True}

    if os.path.exists(clean_path):
        shutil.rmtree(clean_path)
    shutil.copytree(original_path, clean_path)

    print("    Running mypy on original repo (baseline)...")
    baseline_records = run_mypy(original_path)
    baseline_total = len(baseline_records)
    baseline_by_code = error_code_counts(baseline_records)
    print(f"    Baseline errors: {baseline_total}")
    for code, n in baseline_by_code.items():
        print(f"      {code}: {n}")

    errors_by_relpath: Dict[str, Set[int]] = defaultdict(set)
    for r in baseline_records:
        if r["code"] not in ANNOTATION_ERROR_CODES:
            continue
        try:
            rel = os.path.relpath(r["file"], original_path)
        except ValueError:
            continue
        if rel.startswith(".."):
            continue
        errors_by_relpath[rel].add(r["line"])

    total_annotation_errors = sum(len(v) for v in errors_by_relpath.values())
    print(f"    Annotation-related error lines: {total_annotation_errors} "
          f"across {len(errors_by_relpath)} files")

    removed_total = 0
    files_touched = 0
    for rel, lines in errors_by_relpath.items():
        target = os.path.join(clean_path, rel)
        if not os.path.exists(target):
            continue
        n = clean_file(target, target, lines)
        if n:
            removed_total += n
            files_touched += 1
    print(f"    Removed {removed_total} annotations across {files_touched} files")

    print("    Running mypy on cleaned repo (final)...")
    final_records = run_mypy(clean_path)
    final_total = len(final_records)
    final_by_code = error_code_counts(final_records)
    print(f"    Final errors: {final_total} (was {baseline_total})")
    for code, n in final_by_code.items():
        print(f"      {code}: {n}")

    final_errors = []
    for r in final_records:
        try:
            rel = os.path.relpath(r["file"], clean_path)
        except ValueError:
            rel = r["file"]
        abs_file = os.path.join(clean_path, rel)
        function = find_function_for_line(abs_file, r["line"])
        final_errors.append({
            "file": rel,
            "line": r["line"],
            "col": r["col"],
            "function": function,
            "code": r["code"],
            "message": r["msg"],
            "annotation_caused": r["code"] in ANNOTATION_ERROR_CODES,
        })

    remaining_by_code: Dict[str, List[Dict]] = defaultdict(list)
    for e in final_errors:
        remaining_by_code[e["code"]].append(e)

    return {
        "repo": repo_name,
        "baseline_total": baseline_total,
        "baseline_by_code": baseline_by_code,
        "final_total": final_total,
        "final_by_code": final_by_code,
        "removed_annotations": removed_total,
        "files_touched": files_touched,
        "annotation_error_lines": total_annotation_errors,
        "final_errors": final_errors,
        "remaining_by_code": {k: v for k, v in remaining_by_code.items()},
    }


def write_detailed_text_report(all_stats: Dict, path: str):
    lines = []
    lines.append("DETAILED MYPY CLEANING REPORT")
    lines.append(f"Generated: {datetime.now()}")
    lines.append("")

    total_baseline = 0
    total_final = 0
    total_removed = 0
    global_baseline_by_code = defaultdict(int)
    global_final_by_code = defaultdict(int)
    global_files_by_code = defaultdict(set)

    for split_name, repos in all_stats.items():
        for repo_name, stats in repos.items():
            if stats.get("skipped"):
                continue
            total_baseline += stats["baseline_total"]
            total_final += stats["final_total"]
            total_removed += stats["removed_annotations"]
            for code, n in stats["baseline_by_code"].items():
                global_baseline_by_code[code] += n
            for code, n in stats["final_by_code"].items():
                global_final_by_code[code] += n
            for code, items in stats["remaining_by_code"].items():
                for item in items:
                    global_files_by_code[code].add((repo_name, item["file"]))

    lines.append("GLOBAL TOTALS")
    lines.append(f"  Baseline errors : {total_baseline}")
    lines.append(f"  Final errors    : {total_final}")
    lines.append(f"  Annotations removed : {total_removed}")
    lines.append(f"  Errors eliminated   : {total_baseline - total_final}")
    lines.append("")

    lines.append("ERRORS BY CODE (before -> after)")
    all_codes = sorted(
        set(global_baseline_by_code) | set(global_final_by_code),
        key=lambda c: -(global_baseline_by_code.get(c, 0)),
    )
    for code in all_codes:
        before = global_baseline_by_code.get(code, 0)
        after = global_final_by_code.get(code, 0)
        delta = before - after
        desc = CODE_DESCRIPTIONS.get(code, "No description")
        tag = "annotation-caused" if code in ANNOTATION_ERROR_CODES else "not annotation-caused"
        lines.append(f"  [{code}] {desc} ({tag})")
        lines.append(f"      before: {before}   after: {after}   fixed: {delta}")
        if code in global_files_by_code:
            files = sorted(global_files_by_code[code])
            n_repos = len(set(r for r, _ in files))
            n_files = len(files)
            lines.append(f"      remaining in {n_files} files across {n_repos} repos")
        lines.append("")

    lines.append("PER-REPO SUMMARY")
    for split_name, repos in all_stats.items():
        lines.append(f"  Split: {split_name}")
        for repo_name, stats in repos.items():
            if stats.get("skipped"):
                lines.append(f"    {repo_name}: skipped")
                continue
            lines.append(
                f"    {repo_name}: {stats['baseline_total']} -> {stats['final_total']} "
                f"(removed {stats['removed_annotations']} annotations)"
            )
        lines.append("")

    lines.append("PLAIN-ENGLISH EXPLANATION OF EACH ERROR CODE")
    for code in sorted(CODE_DESCRIPTIONS):
        lines.append(f"  [{code}] {CODE_DESCRIPTIONS[code]}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def validate_and_clean_all_repositories():
    print("VALIDATE AND CLEAN REPOSITORIES WITH MYPY (--ignore-missing-imports)")
    print(f"Start: {datetime.now()}")

    output_root = os.path.join(BASE_PATH, "cleaned_repos")
    os.makedirs(output_root, exist_ok=True)

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
            split_stats[repo_name] = process_repository(repo_name, repo_path, output_root)
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
    validate_and_clean_all_repositories()
