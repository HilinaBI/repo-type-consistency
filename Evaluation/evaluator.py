import json
import sys
import os
from typing import Dict, List, Optional, Tuple

BRAIN1_DIR = r"C:\Users\hp\Downloads\typybenchdata\type_inference_project\typybench\src\typybench\cli"
if BRAIN1_DIR not in sys.path:
    sys.path.insert(0, BRAIN1_DIR)

print("=" * 70)
print("SMART EVALUATOR - Batch Evaluation with Brain 1 and 2")
print("=" * 70)

try:
    from analyze_repo import (
        parse_type_string,
        compute_similarity as brain1_compute_similarity,
        similarity_to_distance,
        HONEST_NOMINAL,
        HONEST_SEMANTIC,
        HONEST_BASE,
        CONVERSIONS,
        OPERATIONS,
        TIERED_MAPPINGS,
        HIER
    )
    
    from honest_type_system import compute_honest_similarity
    
    print("Brain 1 and Brain 2 loaded successfully")
    BRAIN_READY = True
    
except ImportError as e:
    print(f"Failed to load modules: {e}")
    BRAIN_READY = False

def is_builtin_type(type_str: str) -> bool:
    if not type_str:
        return True
    
    clean = type_str.strip().lower()
    
    if "[" in clean:
        base = clean.split("[")[0]
    else:
        base = clean
    
    if "." in base:
        base = base.split(".")[-1]
    
    return (base in TIERED_MAPPINGS or base in HIER.graph or 
            base in {"any", "none", "nonetype", "object"})

def smart_similarity(type_a: str, type_b: str) -> Tuple[float, int, str]:
    if type_a == type_b:
        return 1.0, 0, "exact_match"
    
    a_builtin = is_builtin_type(type_a)
    b_builtin = is_builtin_type(type_b)
    
    if a_builtin and b_builtin:
        try:
            node_a = parse_type_string(type_a)
            node_b = parse_type_string(type_b)
            similarity = brain1_compute_similarity(node_a, node_b, mode="combined")
            distance = similarity_to_distance(similarity)
            return similarity, distance, "brain1"
        except Exception as e:
            print(f"Brain 1 failed: {e}")
    
    try:
        def normalize(t: str) -> str:
            if "." in t:
                return t.split(".")[-1]
            return t
        
        result = compute_honest_similarity(
            a=normalize(type_a),
            b=normalize(type_b),
            nominal=HONEST_NOMINAL,
            semantic=HONEST_SEMANTIC,
            conversions=CONVERSIONS,
            operations=OPERATIONS
        )
        
        similarity = result.score if result.score is not None else 0.0
        distance = result.distance
        
        return similarity, distance, "brain2"
        
    except Exception as e:
        print(f"Brain 2 failed: {e}")
    
    if type_a.lower() == type_b.lower():
        return 0.9, 1, "fallback_case_insensitive"
    
    return 0.1, 3, "fallback"

def evaluate_correction_fixed(true_type: str, llm_prediction: str, 
                            mutated_type: Optional[str] = None) -> Dict:
    if not llm_prediction or llm_prediction.strip() == "":
        llm_prediction = "Any"
    
    similarity, distance, method = smart_similarity(true_type, llm_prediction)
    
    is_successful = (distance <= 1)
    
    result = {
        "true_type": true_type,
        "llm_prediction": llm_prediction,
        "similarity_score": float(similarity),
        "distance": int(distance),
        "exact_match": (distance == 0),
        "is_successful_fix": is_successful,
        "evaluation_method": method,
        "fix_category": ["perfect", "good", "partial", "bad"][distance],
    }
    
    if mutated_type and mutated_type != "[NO TYPE]" and mutated_type != true_type:
        mut_sim, mut_dist, mut_method = smart_similarity(true_type, mutated_type)
        improvement = similarity - mut_sim
        
        if distance == 0:
            fix_quality = "perfect_fix"
        elif distance < mut_dist:
            fix_quality = "improved"
        elif distance == mut_dist:
            if similarity > mut_sim:
                fix_quality = "slightly_improved"
            else:
                fix_quality = "no_improvement"
        else:
            fix_quality = "made_worse"
        
        result.update({
            "mutated_type": mutated_type,
            "mutated_similarity": float(mut_sim),
            "mutated_distance": int(mut_dist),
            "improvement_score": float(improvement),
            "mutated_method": mut_method,
            "fix_quality": fix_quality,
        })
    
    return result

def batch_evaluate(test_answers_path: str, predictions_path: str) -> Dict:
    with open(test_answers_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
        
    if isinstance(test_data, dict) and 'answers' in test_data:
        answer_list = test_data['answers']
    else:
        answer_list = test_data
        
    answer_dict = {item['id']: item for item in answer_list}
    pred_dict = {item['id']: item for item in predictions}
    print(f"Answer dict size: {len(answer_dict)}")
    print(f"Prediction dict size: {len(pred_dict)}")
    print(f"First answer ID: {list(answer_dict.keys())[0] if answer_dict else 'Empty'}")
    print(f"First prediction ID: {list(pred_dict.keys())[0] if pred_dict else 'Empty'}")
    
    all_exact_matches = []
    all_D_original = []
    all_D_pred = []
    all_similarity_pred = []
    all_similarity_original = []
    
    individual_results = []
    
    print(f"Evaluating {len(pred_dict)} predictions")
    
    for example_id, pred_data in pred_dict.items():
        if example_id not in answer_dict:
            continue
            
        answer = answer_dict[example_id]
        true_type = answer['true_type']
        llm_pred = pred_data['target'].strip()
        
        mutated_type = answer['metadata'].get('mutated_type', '[NO TYPE]')
        sim_pred, D_pred, method_pred = smart_similarity(true_type, llm_pred)
        is_exact = (D_pred == 0)
        all_exact_matches.append(is_exact)
        
        all_D_pred.append(D_pred)
        all_similarity_pred.append(sim_pred)
        
        if mutated_type != '[NO TYPE]' and mutated_type != true_type:
            sim_orig, D_orig, method_orig = smart_similarity(true_type, mutated_type)
            all_D_original.append(D_orig)
            all_similarity_original.append(sim_orig)
        else:
            all_D_original.append(None)
            all_similarity_original.append(None)
        
        individual_result = evaluate_correction_fixed(true_type, llm_pred, mutated_type)
        individual_result['id'] = example_id
        individual_results.append(individual_result)
    
    valid_correction_indices = [i for i, d_orig in enumerate(all_D_original) if d_orig is not None]
    
    if valid_correction_indices:
        D_orig_correction = [all_D_original[i] for i in valid_correction_indices]
        D_pred_correction = [all_D_pred[i] for i in valid_correction_indices]
        sim_orig_correction = [all_similarity_original[i] for i in valid_correction_indices]
        sim_pred_correction = [all_similarity_pred[i] for i in valid_correction_indices]
        N_correction = len(D_orig_correction)
    else:
        D_orig_correction = D_pred_correction = []
        N_correction = 0
    
    N_total = len(all_exact_matches)
    
    EM = sum(all_exact_matches) / N_total if N_total > 0 else 0
    MPD = sum(all_D_pred) / N_total if N_total > 0 else 3.0
    
    if N_correction > 0:
        DRR = sum(1 for d_pred, d_orig in zip(D_pred_correction, D_orig_correction) 
                 if d_pred < d_orig) / N_correction
    
        MDI = sum(d_orig - d_pred for d_pred, d_orig in zip(D_pred_correction, D_orig_correction)) / N_correction
    
        mean_sim_orig = sum(sim_orig_correction) / N_correction
        sim_improvement = (sum(sim_pred_correction) / N_correction) - mean_sim_orig
    else:
        DRR = MDI = mean_sim_orig = sim_improvement = 0.0

    # Compute mean similarity for ALL examples (including inference)
    mean_sim_all = sum(all_similarity_pred) / N_total if N_total > 0 else 0.0
    
    results = {
        'total_examples': N_total,
        'correction_examples': N_correction,
        'inference_examples': N_total - N_correction,
        'exact_match_accuracy': EM,
        'mean_predicted_distance': MPD,
        'distance_reduction_rate': DRR,
        'mean_distance_improvement': MDI,
        'mean_similarity_predictions_all': mean_sim_all,           # ← NEW
        'mean_similarity_predictions_correction': (sum(sim_pred_correction) / N_correction) if N_correction > 0 else 0.0,
        'mean_similarity_original_errors': mean_sim_orig,
        'similarity_improvement': sim_improvement,
        'distance_distribution': {
            'pred_distance_0': sum(1 for d in all_D_pred if d == 0) / N_total,
            'pred_distance_1': sum(1 for d in all_D_pred if d == 1) / N_total,
            'pred_distance_2': sum(1 for d in all_D_pred if d == 2) / N_total,
            'pred_distance_3': sum(1 for d in all_D_pred if d == 3) / N_total,
        },
        'individual_results': individual_results
    }
    
    return results

def print_results_table(results: Dict):
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nDataset Size: {results['total_examples']} total examples")
    print(f"  Correction task: {results['correction_examples']} examples")
    print(f"  Inference task: {results['inference_examples']} examples")
    
    print("\n" + "-"*80)
    print("CORE EVALUATION METRICS")
    print("-"*80)
    print(f"Exact-Match Accuracy (EM):     {results['exact_match_accuracy']:.3f} ({results['exact_match_accuracy']*100:.1f}%)")
    print(f"Mean Predicted Distance (MPD): {results['mean_predicted_distance']:.3f}")
    print(f"Distance Reduction Rate (DRR): {results['distance_reduction_rate']:.3f} ({results['distance_reduction_rate']*100:.1f}%)")
    print(f"Mean Distance Improvement (MDI): {results['mean_distance_improvement']:.3f}")
    
    print("\n" + "-"*80)
    print("DISTANCE DISTRIBUTION OF PREDICTIONS")
    print("-"*80)
    dist = results['distance_distribution']
    print(f"Distance 0 (Perfect): {dist['pred_distance_0']*100:5.1f}%")
    print(f"Distance 1 (Good):    {dist['pred_distance_1']*100:5.1f}%")
    print(f"Distance 2 (Partial): {dist['pred_distance_2']*100:5.1f}%")
    print(f"Distance 3 (Failed):  {dist['pred_distance_3']*100:5.1f}%")

def test_fixed_similarity():
    print("\n" + "=" * 70)
    print("TESTING SIMILARITY FUNCTION")
    print("=" * 70)
    
    if not BRAIN_READY:
        print("Brain modules not loaded")
        return
    
    test_cases = [
        ("str", "bytes", "brain1"),
        ("list[float]", "List[float]", "brain1"),
        ("DataFrame", "Series", "brain2"),
        ("pandas.DataFrame", "DataFrame", "brain2"),
        ("DatabaseConnection", "Connection", "brain2"),
        ("Optional[DataFrame]", "Union[DataFrame, None]", "brain2"),
    ]
    
    for true, pred, expected_brain in test_cases:
        try:
            similarity, distance, method = smart_similarity(true, pred)
            
            correct_brain = "OK" if method.startswith(expected_brain) else "WRONG"
            
            if distance == 0:
                rating = "PERFECT"
            elif distance == 1:
                rating = "GOOD"
            elif distance == 2:
                rating = "PARTIAL"
            else:
                rating = "BAD"
            
            print(f"{correct_brain:6} {true:25} vs {pred:25} -> "
                  f"{method:15} sim:{similarity:.3f} dist:{distance} ({rating})")
            
        except Exception as e:
            print(f"ERROR {true:25} vs {pred:25} -> failed: {e}")

if __name__ == "__main__":
    test_fixed_similarity()
    
    TEST_ANSWERS_PATH = "cleaned_repos/TEST_INFERENCE_ANSWERS_NORMALIZED_clean.json"
    PREDICTIONS_PATH = "cleaned_repos/model_predictions_new_clean_repo_normalized_α1.0_with_variants_inference.json"
    
    if os.path.exists(TEST_ANSWERS_PATH) and os.path.exists(PREDICTIONS_PATH):
        print("\n" + "="*80)
        print("RUNNING BATCH EVALUATION") 
        print("="*80)
        
        results = batch_evaluate(TEST_ANSWERS_PATH, PREDICTIONS_PATH)
        print_results_table(results)
        
        output_path = "cleaned_repos/evaluation_results_model_predictions_new_clean_repo_normalized_α1.0_with_variants_inference.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")
    else:
        print("\nTest data or predictions not found.")
        print(f"Looking for: {TEST_ANSWERS_PATH}")
        print(f"Looking for: {PREDICTIONS_PATH}")