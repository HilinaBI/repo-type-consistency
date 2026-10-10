import json
import os
import random
import ast
import sys
import re
from typing import Dict

# Import the normalizer
from alias_normalizer import build_alias_table, normalize_code, normalize_type

# Add Brain 1 path
BRAIN1_DIR = r"C:\Users\hp\Downloads\typybenchdata\type_inference_project\typybench\src\typybench\cli"
if BRAIN1_DIR not in sys.path:
    sys.path.insert(0, BRAIN1_DIR)

try:
    from analyze_repo import get_error_candidate
    BRAIN_READY = True
    print(" Brain 1 loaded successfully")
except ImportError as e:
    print(f" Could not load Brain 1: {e}")
    BRAIN_READY = False
    sys.exit(1)

BASE_PATH = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata"
CLEAN_REPOS_BASE = os.path.join(BASE_PATH, "cleaned_repos")

TRAINING_REPOS = [
    "pre-commit-hooks", "flake8", "vnpy", "GHunt", "spotify-downloader",
    "deepface", "urllib3", "fastapi", "poetry", "haystack", 
    "rich", "black", "faceswap", "streamlit", "phidata",
    "Pillow", "sphinx", "pip", "gradio", "pydantic"
]

# Number of corrupted variants per corrupted example
NUM_CORRUPTED_VARIANTS = 3

# HELPER FUNCTIONS

def get_function_lines(source_code, node):
    lines = source_code.splitlines()
    start = node.lineno - 1
    end = node.end_lineno if hasattr(node, 'end_lineno') else start
    func_lines = lines[start:end]
    
    if not func_lines:
        return ""
    
    min_indent = 1000
    for line in func_lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent < min_indent:
                min_indent = indent
    
    dedented = []
    for line in func_lines:
        if len(line) >= min_indent:
            dedented.append(line[min_indent:])
        else:
            dedented.append(line)
    
    return '\n'.join(dedented)


def extract_variables_from_code(function_code):
    results = []
    
    lines = function_code.split('\n')
    if not lines:
        return results
    
    first_line = lines[0]
    
    if not first_line.strip().startswith(('def ', 'async def ')):
        return results
    
    try:
        tree = ast.parse(function_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                
                for arg in node.args.args:
                    if arg.annotation:
                        try:
                            type_str = ast.unparse(arg.annotation)
                            results.append(("arg", arg.arg, type_str, func_name))
                        except:
                            pass
                
                for arg in node.args.kwonlyargs:
                    if arg.annotation:
                        try:
                            type_str = ast.unparse(arg.annotation)
                            results.append(("kwarg", arg.arg, type_str, func_name))
                        except:
                            pass
                
                if node.args.vararg and node.args.vararg.annotation:
                    try:
                        type_str = ast.unparse(node.args.vararg.annotation)
                        results.append(("vararg", node.args.vararg.arg, type_str, func_name))
                    except:
                        pass
                
                if node.args.kwarg and node.args.kwarg.annotation:
                    try:
                        type_str = ast.unparse(node.args.kwarg.annotation)
                        results.append(("kwarg", node.args.kwarg.arg, type_str, func_name))
                    except:
                        pass
                
                if node.returns:
                    try:
                        type_str = ast.unparse(node.returns)
                        results.append(("return", "return", type_str, func_name))
                    except:
                        pass
                
                return results
    except:
        pass
    
    return results


def remove_all_types_from_function(func_code):
    try:
        tree = ast.parse(func_code)
        
        class TypeRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                node.returns = None
                for arg in node.args.args:
                    arg.annotation = None
                for arg in node.args.kwonlyargs:
                    arg.annotation = None
                if node.args.vararg:
                    node.args.vararg.annotation = None
                if node.args.kwarg:
                    node.args.kwarg.annotation = None
                return self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                node.returns = None
                for arg in node.args.args:
                    arg.annotation = None
                for arg in node.args.kwonlyargs:
                    arg.annotation = None
                if node.args.vararg:
                    node.args.vararg.annotation = None
                if node.args.kwarg:
                    node.args.kwarg.annotation = None
                return self.generic_visit(node)
        
        transformer = TypeRemover()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        
        try:
            from ast import unparse
            return unparse(new_tree)
        except ImportError:
            import astor
            return astor.to_source(new_tree)
            
    except Exception as e:
        return func_code


def extract_type_from_file(repo_path, file_rel_path, func_name, var_name, is_return):
    full_path = os.path.join(repo_path, file_rel_path)
    
    if not os.path.exists(full_path):
        return None
    
    try:
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except:
            return None
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    if is_return:
                        if node.returns:
                            return ast.unparse(node.returns).strip()
                    else:
                        for arg in node.args.args:
                            if arg.arg == var_name and arg.annotation:
                                return ast.unparse(arg.annotation).strip()
                        for arg in node.args.kwonlyargs:
                            if arg.arg == var_name and arg.annotation:
                                return ast.unparse(arg.annotation).strip()
                        if node.args.vararg and node.args.vararg.arg == var_name and node.args.vararg.annotation:
                            return ast.unparse(node.args.vararg.annotation).strip()
                        if node.args.kwarg and node.args.kwarg.arg == var_name and node.args.kwarg.annotation:
                            return ast.unparse(node.args.kwarg.annotation).strip()
        return None
        
    except Exception as e:
        return None


# MAIN DATASET CREATION WITH 3 CORRUPTED VARIANTS

def create_normalized_dataset():
    print("=" * 80)
    print("CREATING NORMALIZED DATASET FROM CLEANED REPOSITORIES")
    print("=" * 80)
    print(f"Number of corrupted variants per corrupted example: {NUM_CORRUPTED_VARIANTS}")
    print("=" * 80)
    
    ALPHA_VALUES = {
        0.1: 0.9,   # 90% clean, 10% corrupted
        0.3: 0.7,   # 70% clean, 30% corrupted
        0.5: 0.5,   # 50% clean, 50% corrupted
        0.7: 0.3,   # 30% clean, 70% corrupted
        1.0: 0.0    # 0% clean, 100% corrupted
    }
    random.seed(42)
    
    for ALPHA, KEEP_PROB in ALPHA_VALUES.items():
        print(f"\n{'='*60}")
        print(f"Creating dataset: α = {ALPHA}")
        print(f"  - Keep (clean) probability: {KEEP_PROB*100:.0f}%")
        print(f"  - Corrupted probability: {(1-KEEP_PROB)*100:.0f}%")
        print(f"  - Corrupted variants per corrupted example: {NUM_CORRUPTED_VARIANTS}")
        print(f"{'='*60}")
        
        all_examples = []
        stats = {
            "total": 0,
            "clean": 0,
            "corrupted": 0,
            "corrupted_variants": 0,
            "no_type": 0,
            "var_types": {"arg": 0, "return": 0, "kwarg": 0, "vararg": 0}
        }
        
        for repo_idx, repo_name in enumerate(TRAINING_REPOS, 1):
            print(f"[{repo_idx:2d}/20] {repo_name}")
            
            original_clean_path = os.path.join(CLEAN_REPOS_BASE, repo_name, "original_repo_clean")
            mutated_clean_path = os.path.join(CLEAN_REPOS_BASE, repo_name, "repo_with_semantic_errors_enhancedrules_clean")
            mutations_exist = os.path.exists(mutated_clean_path)
            
            if not os.path.exists(original_clean_path):
                print(f"  Warning: Cleaned repo not found: {original_clean_path}")
                continue
            
            for root, dirs, files in os.walk(original_clean_path):
                for file in files:
                    if not file.endswith('.py'):
                        continue
                    
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, original_clean_path)
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        alias_table = build_alias_table(content)
                        
                        try:
                            tree = ast.parse(content)
                        except:
                            continue
                        
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                func_name = node.name
                                func_code_with_types = get_function_lines(content, node)
                                
                                if not func_code_with_types:
                                    continue
                                
                                typed_vars = extract_variables_from_code(func_code_with_types)
                                
                                if not typed_vars:
                                    continue
                                
                                type_free_code = remove_all_types_from_function(func_code_with_types)
                                normalized_code = normalize_code(type_free_code, alias_table)
                                
                                for var_type, var_name, true_type, _ in typed_vars:
                                    stats["var_types"][var_type] += 1
                                    
                                    base_id = f"{repo_name}_{func_name}_{var_type}_{var_name}"
                                    normalized_true_type = normalize_type(true_type, alias_table)
                                    
                                    # VARIANT 1: [NO TYPE] (ALWAYS)
                                    observed_type = "[NO TYPE]"
                                    stats["no_type"] += 1
                                    
                                    example_id = f"{base_id}_{stats['total']:08d}_notype"
                                    codet5_input = f"<CODE>\n{normalized_code}\n<TARGET_VARIABLE>\n{var_name}\n<observed_type>\n{observed_type}"
                                    
                                    example = {
                                        "id": example_id,
                                        "input": codet5_input,
                                        "target": normalized_true_type,
                                        "split": "train"
                                    }
                                    all_examples.append(example)
                                    stats["total"] += 1
                                    
                                    # VARIANT 2: Clean (1 example) OR Corrupted (3 variants)
                                    rand = random.random()
                                    
                                    if rand < KEEP_PROB:
                                        # CLEAN - only 1 example
                                        observed_type = true_type
                                        corruption = "clean"
                                        stats["clean"] += 1
                                        
                                        normalized_observed_type = normalize_type(observed_type, alias_table)
                                        
                                        example_id = f"{base_id}_{stats['total']:08d}_clean"
                                        codet5_input = f"<CODE>\n{normalized_code}\n<TARGET_VARIABLE>\n{var_name}\n<observed_type>\n{normalized_observed_type}"
                                        
                                        example = {
                                            "id": example_id,
                                            "input": codet5_input,
                                            "target": normalized_true_type,
                                            "split": "train"
                                        }
                                        all_examples.append(example)
                                        stats["total"] += 1
                                        
                                    else:
                                        # CORRUPTED - create 3 UNIQUE variants
                                        stats["corrupted"] += 1
                                        
                                        used_mutations = set()
                                        variants_generated = 0
                                        variant_attempts = 0
                                        max_attempts = NUM_CORRUPTED_VARIANTS * 20
                                        
                                        while variants_generated < NUM_CORRUPTED_VARIANTS and variant_attempts < max_attempts:
                                            variant_attempts += 1
                                            
                                            # get a mutated type from Brain 1
                                            mutated_type = None
                                            if BRAIN_READY:
                                                mutation_result = get_error_candidate(
                                                    true_type,
                                                    rng=random.Random(random.randint(1, 1000000)),
                                                    mode="combined"
                                                )
                                                mutated_type = mutation_result.get("type")
                                            
                                            # Fallback to file-based mutation if Brain 1 fails
                                            if not mutated_type and mutations_exist:
                                                mutated_type = extract_type_from_file(
                                                    mutated_clean_path, rel_path, func_name, var_name, (var_type == "return")
                                                )
                                            
                                            # Final fallback
                                            if not mutated_type or mutated_type == true_type:
                                                mutated_type = "Any"
                                            
                                            # Check if unique and not excluded
                                            if mutated_type not in used_mutations:
                                                used_mutations.add(mutated_type)
                                                variants_generated += 1
                                                stats["corrupted_variants"] += 1
                                                
                                                normalized_observed_type = normalize_type(mutated_type, alias_table)
                                                
                                                example_id = f"{base_id}_{stats['total']:08d}_corrupted_v{variants_generated}"
                                                codet5_input = f"<CODE>\n{normalized_code}\n<TARGET_VARIABLE>\n{var_name}\n<observed_type>\n{normalized_observed_type}"
                                                
                                                example = {
                                                    "id": example_id,
                                                    "input": codet5_input,
                                                    "target": normalized_true_type,
                                                    "split": "train"
                                                }
                                                all_examples.append(example)
                                                stats["total"] += 1
                    
                    except Exception as e:
                        print(f"Error: {e}")
                        continue
        
        output_file = os.path.join(CLEAN_REPOS_BASE, f"CODET5_DATASET_CLEAN_NORMALIZED_α{ALPHA}_VARIANTS.jsonl")
        
        print(f"\nSaving {stats['total']} examples to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for ex in all_examples:
                f.write(json.dumps({
                    "id": ex["id"],
                    "input": ex["input"],
                    "target": ex["target"],
                    "split": "train"
                }, ensure_ascii=False) + "\n")
        
        print(f"\nStats for α={ALPHA}:")
        print(f"  Total examples: {stats['total']}")
        print(f"  [NO TYPE]: {stats['no_type']} (50%)")
        print(f"  Clean: {stats['clean']} ({stats['clean']/stats['total']*100:.1f}%)")
        print(f"  Corrupted (unique examples): {stats['corrupted']}")
        print(f"  Corrupted variants generated: {stats['corrupted_variants']}")
        print(f"  Variable types: {stats['var_types']}")


if __name__ == "__main__":
    create_normalized_dataset()