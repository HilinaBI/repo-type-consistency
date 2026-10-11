# process_all_repos.py
import os
import sys
import ast
import shutil
import json 
from pathlib import Path
from typing import List, Dict, Tuple, Any


current_dir = os.path.dirname(os.path.abspath(__file__))
analyze_repo_path = os.path.join(current_dir, 'src', 'typybench', 'cli')
sys.path.insert(0, analyze_repo_path)

try:
    from analyze_repo import inject_type_errors 
except ImportError as e:
    print(f"FATAL ERROR: Could not import 'analyze_repo'. Please check your path.")
    print(f"Attempted path: {analyze_repo_path}")
    sys.exit(1)


def process_repository(repo_name):
    """
    Process one repository.
    Returns: Tuple[bool, List[Dict]] - (success status, list of all mutations for the repo)
    """
  
    data_path = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata"
    repo_path = os.path.join(data_path, "cleaned_repos", repo_name, "original_repo_clean")
    output_path = os.path.join(data_path, "cleaned_repos", repo_name, "repo_with_semantic_errors_enhancedrules_clean")
    
    if not os.path.exists(repo_path):
        print(f" Repository not found: {repo_path}")
        return False, []
    
    print(f"\n Processing: {repo_name}")
    
    # List to collect all mutation data (including distance) for this repository
    repo_mutation_data = [] 
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    
    print(f"   Found {len(python_files)} Python files")
    
    # Process each file
    success_count = 0
    for file_path in python_files:
        
        file_mutations = [] # Collect mutations for the current file
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                continue
            
            tree = ast.parse(content)
            modified = False
            
            # Use a visitor pattern to find and replace function definitions
            class FunctionErrorInjector(ast.NodeTransformer):
                def visit_FunctionDef(self, node):
                    nonlocal modified
                    
                    # Call inject_type_errors and capture the mutations list
                    mutated_node, mutations = inject_type_errors(node)
                    
                    if mutations:
                        modified = True
                        file_mutations.extend(mutations) # COLLECT METADATA HERE
                        
                    return mutated_node
            
            tree = FunctionErrorInjector().visit(tree)
            
            relative_path = os.path.relpath(file_path, repo_path)
            output_file_path = os.path.join(output_path, relative_path)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Save the file
            if modified:
                modified_code = ast.unparse(tree)
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                success_count += 1
                
                # FINAL COLLECTION STEP for the file's metadata
                if file_mutations:
                    repo_mutation_data.append({
                        "file_path": relative_path,
                        "mutations": file_mutations
                    })
                    
            else:
                # Copy original file if no mutations were possible
                shutil.copy2(file_path, output_file_path)
                
        except Exception as e:
            print(f"   ✗ Error in {os.path.basename(file_path)}: {e}")
            # Ensure the original file is copied even on error
            try:
                relative_path = os.path.relpath(file_path, repo_path)
                output_file_path = os.path.join(output_path, relative_path)
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                shutil.copy2(file_path, output_file_path)
            except:
                pass
    
    print(f"    Mutated {success_count}/{len(python_files)} files successfully")
    return True, repo_mutation_data # RETURN THE COLLECTED DATA


def main():
    """Process all 50 repositories from CLEANED_REPOS directory"""
    data_path = r"C:\Users\hp\Downloads\typybenchdata\typybenchdata"
    
    # Initialize the master dictionary for all mutation data
    all_mutations_summary = {} 
    
    # Define the output file for the metadata summary
    summary_output_file = os.path.join(data_path, "mutations_summary_clean.json")
    
    # Get all repository names from the CLEANED_REPOS directory
    cleaned_repos_path = os.path.join(data_path, "cleaned_repos")
    
    # Check if cleaned_repos exists FIRST (outside the loop)
    if not os.path.exists(cleaned_repos_path):
        print(f"ERROR: Cleaned repos directory not found at {cleaned_repos_path}")
        return
    
    all_repos = []
    # Loop through items in cleaned_repos_path, NOT data_path
    for item in os.listdir(cleaned_repos_path):
        full_item_path = os.path.join(cleaned_repos_path, item)
        # Check for original_repo_clean inside each repo
        if os.path.isdir(full_item_path) and os.path.isdir(os.path.join(full_item_path, "original_repo_clean")):
            all_repos.append(item)
    
    if not all_repos:
        print(f"WARNING: No repositories with 'original_repo_clean' found in {cleaned_repos_path}")
        return
    
    print(f"Found {len(all_repos)} CLEAN repositories to process")
    print("Starting batch processing...")
    
    # Process each repository
    successful_repos = []
    for repo_name in all_repos:
        # Capture the mutation data returned by the function
        success, repo_data = process_repository(repo_name) 
        
        if success:
            successful_repos.append(repo_name)
            if repo_data:
                all_mutations_summary[repo_name] = repo_data
                
    # Write the collected mutation data to JSON
    print(f"\nWriting mutation summary to {summary_output_file}...")
    try:
        with open(summary_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_mutations_summary, f, indent=4)
        print("Metadata successfully saved!")
    except Exception as e:
        print(f"Failed to write JSON summary: {e}")
        
    print(f"\nBATCH PROCESSING COMPLETE!")
    print(f"Successfully processed {len(successful_repos)}/{len(all_repos)} CLEAN repositories")
    print("All code mutations are saved in 'cleaned_repos/*/repo_with_semantic_errors_enhancedrules_clean'")

if __name__ == "__main__":
    main()