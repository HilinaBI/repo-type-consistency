import json
import os
import ast
import subprocess
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# CONFIGURATION


BASE_PATH = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata\cleaned_repos"
PREDICTIONS_FILE = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata\cleaned_repos\model_predictions_new_clean_repo_normalized_α1.0_with_variants.json"
OUTPUT_DIR = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata\cleaned_repos\typecheck_eval"

TEST_REPOS = [
    "private-gpt", "gptme", "paper-qa", "pandas-ai", "supervision",
    "gpt4free", "autogpt", "mlc-llm", "db-gpt", "vllm",
    "screenshot-to-code", "composio", "agents", "unstract", "ten-agent",
    "gpt-pilot", "lerobot", "llama-factory", "appworld", "exo"
]

# PARSE PREDICTION ID (FIXED)

def parse_prediction_id(pred_id: str) -> Dict[str, str]:
    """
    Parse prediction ID like: test_db-gpt_get_printable_message_arg_conv_00000000
    
    Structure: test_{repo}_{function}_{var_type}_{var_name}_{timestamp}
    where var_type is always 'arg' or 'return'
    """
    parts = pred_id.split('_')
    if len(parts) < 5:
        return None
    
    # Find where var_type occurs ('arg' or 'return')
    var_type_idx = None
    for i, part in enumerate(parts):
        if part in ['arg', 'return']:
            var_type_idx = i
            break
    
    if var_type_idx is None:
        return None
    
    # Extract components
    var_type = parts[var_type_idx]
    var_name = parts[var_type_idx + 1] if var_type_idx + 1 < len(parts) else ""
    timestamp = parts[-1]
    
    # repo is the second part (index 1)
    repo = parts[1]
    
    # function is everything between repo (index 1) and var_type (var_type_idx)
    # Skip first part 'test' (index 0) and repo (index 1)
    function_parts = parts[2:var_type_idx]
    function = '_'.join(function_parts)
    
    return {
        "repo": repo,
        "function": function,
        "var_type": var_type,
         "timestamp": timestamp
    }


# BUILD FUNCTION TO FILE MAPPING

def build_function_to_file_mapping(repo_path: str) -> Dict[str, str]:
    """Map function names to their file paths."""
    mapping = {}
    
    if not os.path.exists(repo_path):
        return mapping
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', 'env', 'node_modules']]
        for file in files:
            if not file.endswith('.py'):
                continue
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        mapping[node.name] = rel_path
            except:
                continue
    
    return mapping


# RUN MYPY AND COUNT ERRORS

def run_mypy_and_count_errors(repo_path: str, stub_dir: str = None) -> Dict:
    """Run mypy on repository and count errors."""
    
    cmd = ["mypy", repo_path, "--no-incremental", "--ignore-missing-imports"]
    
    env = os.environ.copy()
    if stub_dir:
        env["MYPYPATH"] = stub_dir
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )
    
    error_types = {
        "return_type": 0,
        "argument_type": 0,
        "assignment": 0,
        "annotation": 0,
        "name_undefined": 0,
        "other": 0,
        "total": 0
    }
    
    for line in result.stdout.split('\n'):
        if "error:" in line:
            error_types["total"] += 1
            if "Incompatible return" in line:
                error_types["return_type"] += 1
            elif "has incompatible type" in line:
                error_types["argument_type"] += 1
            elif "Incompatible types in assignment" in line:
                error_types["assignment"] += 1
            elif "is not valid as type" in line:
                error_types["annotation"] += 1
            elif "is not defined" in line:
                error_types["name_undefined"] += 1
            else:
                error_types["other"] += 1
    
    return error_types


# GENERATE STUB FILES FROM PREDICTIONS

def generate_stub_files(predictions: List[Dict], repo_name: str, repo_path: str, output_dir: str) -> str:
    """Generate .pyi stub files from model predictions."""
    
    stub_dir = os.path.join(output_dir, repo_name, "stubs")
    if os.path.exists(stub_dir):
        shutil.rmtree(stub_dir)
    os.makedirs(stub_dir)
    
    func_to_file = build_function_to_file_mapping(repo_path)
    
    # Group predictions by file
    file_stubs = {}
    not_found = 0
    found_count = 0
    
    for pred in predictions:
        pred_id = pred.get("id", "")
        if not pred_id.startswith("test_"):
            continue
        
        parsed = parse_prediction_id(pred_id)
        if not parsed or parsed["repo"] != repo_name:
            continue
        
        function = parsed["function"]
        var_type = parsed["var_type"]
        var_name = parsed["var_name"]
        predicted_type = pred.get("target", "Any")
        
        if function not in func_to_file:
            not_found += 1
            continue
        
        found_count += 1
        py_file = func_to_file[function]
        stub_file = py_file.replace('.py', '.pyi')
        
        if stub_file not in file_stubs:
            file_stubs[stub_file] = {}
        if function not in file_stubs[stub_file]:
            file_stubs[stub_file][function] = {"args": {}, "returns": None}
        
        if var_type == "return":
            file_stubs[stub_file][function]["returns"] = predicted_type
        else:
            file_stubs[stub_file][function]["args"][var_name] = predicted_type
    
    # Write stub files
    for stub_file, functions in file_stubs.items():
        stub_path = os.path.join(stub_dir, stub_file)
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        
        with open(stub_path, 'w', encoding='utf-8') as f:
            f.write("# Generated from model predictions\n\n")
            for func_name, func_info in functions.items():
                args = []
                for arg_name, arg_type in func_info["args"].items():
                    args.append(f"{arg_name}: {arg_type}")
                args_str = ", ".join(args)
                returns = func_info["returns"] if func_info["returns"] else "Any"
                f.write(f"def {func_name}({args_str}) -> {returns}: ...\n\n")
    
    return stub_dir


# GET GROUND TRUTH ERRORS


def get_ground_truth_errors(repo_path: str) -> Dict:
    """Run mypy on original repository to get baseline errors."""
    return run_mypy_and_count_errors(repo_path)


# GET PREDICTED ERRORS

def get_predicted_errors(repo_path: str, predictions: List[Dict], repo_name: str, output_dir: str) -> Dict:
    """Generate stubs from predictions and run mypy."""
    stub_dir = generate_stub_files(predictions, repo_name, repo_path, output_dir)
    
    cmd = ["mypy", repo_path, "--no-incremental", "--ignore-missing-imports"]
    env = os.environ.copy()
    env["MYPYPATH"] = stub_dir
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    error_types = {
        "return_type": 0,
        "argument_type": 0,
        "assignment": 0,
        "annotation": 0,
        "name_undefined": 0,
        "other": 0,
        "total": 0
    }
    
    for line in result.stdout.split('\n'):
        if "error:" in line:
            error_types["total"] += 1
            if "Incompatible return" in line:
                error_types["return_type"] += 1
            elif "has incompatible type" in line:
                error_types["argument_type"] += 1
            elif "Incompatible types in assignment" in line:
                error_types["assignment"] += 1
            elif "is not valid as type" in line:
                error_types["annotation"] += 1
            elif "is not defined" in line:
                error_types["name_undefined"] += 1
            else:
                error_types["other"] += 1
    
    return error_types


# MAIN EVALUATION

def run_typecheck_evaluation():
    print("=" * 80)
    print("TYPECHECK EVALUATION (Following TypyBench Paper)")
    print("=" * 80)
    print("\nPaper methodology:")
    print("  1. Run mypy on original repository → get baseline errors")
    print("  2. Run mypy with predicted types as stubs → get errors after fixes")
    print("  3. Compare: improvement = baseline - predicted errors")
    print("=" * 80)
    
    # Load predictions
    print(f"\n1. Loading predictions...")
    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    print(f"   Loaded {len(predictions)} predictions")
    
    # Group predictions by repository
    pred_by_repo = defaultdict(list)
    for pred in predictions:
        pred_id = pred.get("id", "")
        if not pred_id.startswith("test_"):
            continue
        parsed = parse_prediction_id(pred_id)
        if parsed:
            pred_by_repo[parsed["repo"]].append(pred)
    
    print(f"   Found predictions for {len(pred_by_repo)} repositories")
    
    results = {}
            
    for repo_name in TEST_REPOS:
        if repo_name not in pred_by_repo:
            print(f"\n  Skipping {repo_name}: No predictions found")
            continue
        
        repo_path = os.path.join(BASE_PATH, repo_name, "original_repo_clean")
        
        if not os.path.exists(repo_path):
            print(f"\n  Skipping {repo_name}: Repository not found at {repo_path}")
            continue
        
        print(f"\n  Processing: {repo_name}")
        print(f"  Predictions for this repo: {len(pred_by_repo[repo_name])}")
        
        # Step 1: Ground truth errors (original repository)
        print(f"    Running mypy on original code...")
        ground_truth = get_ground_truth_errors(repo_path)
        
        # Step 2: Predicted errors (with model's types as stubs)
        print(f"    Running mypy with predicted types...")
        predicted = get_predicted_errors(repo_path, pred_by_repo[repo_name], repo_name, OUTPUT_DIR)
        
        # Calculate improvement
        improvement = {
            "total": ground_truth["total"] - predicted["total"],
            "return_type": ground_truth["return_type"] - predicted["return_type"],
            "argument_type": ground_truth["argument_type"] - predicted["argument_type"],
            "assignment": ground_truth["assignment"] - predicted["assignment"],
            "annotation": ground_truth["annotation"] - predicted["annotation"],
            "name_undefined": ground_truth["name_undefined"] - predicted["name_undefined"],
            "other": ground_truth["other"] - predicted["other"],
        }
        
        results[repo_name] = {
            "ground_truth": ground_truth,
            "predicted": predicted,
            "improvement": improvement
        }
        
        print(f"      Ground truth errors: {ground_truth['total']}")
        print(f"      Predicted errors: {predicted['total']}")
        print(f"      Improvement: {improvement['total']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_ground = sum(r["ground_truth"]["total"] for r in results.values())
    total_pred = sum(r["predicted"]["total"] for r in results.values())
    total_improvement = total_ground - total_pred
    
    print(f"\nTotal repositories evaluated: {len(results)}")
    print(f"Total ground truth errors: {total_ground}")
    print(f"Total errors after predictions: {total_pred}")
    
    if total_ground > 0:
        print(f"Total errors fixed: {total_improvement} ({total_improvement/total_ground*100:.1f}% reduction)")
    else:
        print(f"Total errors fixed: {total_improvement}")
    
    print(f"\nError type breakdown:")
    error_types = ["return_type", "argument_type", "assignment", "annotation", "name_undefined", "other"]
    for etype in error_types:
        ground_sum = sum(r["ground_truth"][etype] for r in results.values())
        pred_sum = sum(r["predicted"][etype] for r in results.values())
        if ground_sum > 0:
            reduction = ground_sum - pred_sum
            print(f"  {etype}: {ground_sum} → {pred_sum} (fixed: {reduction}, {reduction/ground_sum*100:.1f}%)")
        elif pred_sum > 0:
            print(f"  {etype}: 0 → {pred_sum} (new errors introduced: {pred_sum})")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "typecheck_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    run_typecheck_evaluation()