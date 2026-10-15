from __future__ import annotations
import ast
import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import math
from honest_type_system import (
    initialize_honest_type_system,
    compute_honest_similarity,
    generate_mutation_candidates,
    SimilarityResult
)
try:
    import mypy.api  # type: ignore
    HAS_MYPY = True
except Exception:
    HAS_MYPY = False

# ast.unparse compatibility (python <3.9 fallback)
try:
    from ast import unparse as ast_unparse
except Exception:
    try:
        import astor  # type: ignore
        ast_unparse = astor.to_source
    except Exception:
        def ast_unparse(node):
            # best-effort fallback
            return ast.dump(node)


# CONFIG / MAPPINGS
MUTATED_OUTPUT_DIR = "mutated_repos"
DEFAULT_SEED = 42

# Tiered candidate map Expanded.
TIERED_MAPPINGS: Dict[str, List[str]] = {
    "int": ["float", "str", "complex", "List[int]"],
    "float": ["int", "Optional[float]", "List[float]", "complex", "str"],
    "complex": ["float", "int", "Optional[complex]"],
    "bool": ["int", "Optional[bool]", "str"],

    "sequence": ["list", "tuple", "iterable"],
    "list": ["Set", "Tuple", "Sequence", "Iterable"], 
    "tuple": ["List", "Sequence", "Iterable"],

    "str": ["bytes", "Optional[str]", "List[str]", "literal"], 
    "bytes": ["str", "Optional[bytes]", "bytearray"],
    "range": ["list", "tuple", "Sequence", "Iterable"],
    "bytearray": ["bytes", "list"],
    "memoryview": ["bytes", "bytearray"],
    "set": ["List", "Frozenset"],
    "frozenset": ["Set", "List"],
    "literal": ["str"],
    "uuid": ["str", "int", "bytes"],

    "dict": ["Mapping", "typeddict", "DefaultDict"],
    "mapping": ["dict", "typeddict"],
    "ordereddict": ["dict", "mapping"],
    "defaultdict": ["dict", "Mapping"],
    "typeddict": ["dict", "mapping"],
     
     # DECIMAL/FACTION 
    "decimal_like": ["decimal", "fraction"],
    "decimal": ["float", "int", "str"],
    "fraction": ["float", "decimal", "int"],
    
    
    
    # New Mappings 
    "iterable": ["iterator", "list", "Iterable[Any]"],
    "iterator": ["generator", "iterable"],
    "generator": ["iterator", "asyncgenerator", "Generator[Any, Any, Any]"],
    "asynciterable": ["iterable", "asynciterator"],
    "asynciterator": ["iterator", "asyncgenerator"],
    "asyncgenerator": ["generator", "asynciterator"],
    
    "callable": ["Callable[..., Any]", "function", "method", "typing.Any"],
    "function": ["callable", "method"],
    "method": ["function", "callable"],
    "coroutine": ["asyncfunction", "awaitable"],
    "asyncfunction": ["coroutine", "function"],
    
    "awaitable": ["coroutine", "future", "typing.Any"],
    "future": ["awaitable", "coroutine"],
    
    "sized": ["int"],
    "container": ["iterable"],
    "reversible": ["iterable", "sequence"],
    
    # ASYNC TYPES 
    "asyncgenerator": ["generator", "asynciterator"],
    
    #  IO TYPES
    "io": ["textio", "binaryio"],
    "textio": ["io", "str", "StringIO"],
    "binaryio": ["io", "bytes", "BytesIO"],
   

    "annotated": ["TypeVar", "Generic"],
    "typevar": ["Type[Any]", "ParamSpec", "Generic"],
    "paramspec": ["typevar", "typing.Any"],
    "generic": ["typing.Any", "object"],

    

    
    # DATETIME 
    "datetime": ["date", "str", "float"],
    "date": ["datetime", "str"],
    "time": ["str", "datetime.time"],
    "timedelta": ["float", "int", "str"],
    
    #  ENUM TYPES 
    "enum": ["int", "str"],
    "intenum": ["enum", "int"],
    "strenum": ["enum", "str"],
    
    
    #  TYPE 
    "type": ["type[Any]", "typing.Any", "Type[object]"],

}

# Directional coercion costs (from -> to)
COERCION_COSTS = {
    # ===== NUMERIC =====
    ("bool", "int"): 0.05,
    ("int", "bool"): 0.7,

    ("int", "float"): 0.1,
    ("float", "int"): 0.6,

    ("float", "complex"): 0.2,
    ("complex", "float"): 0.9,

    ("int", "complex"): 0.25,
    ("complex", "int"): 1.0,

    # Decimal / Fraction
    ("decimal", "float"): 0.3,
    ("decimal", "int"): 0.4,
    ("fraction", "float"): 0.2,
    ("fraction", "int"): 0.3,

    # Timedelta
    ("timedelta", "float"): 0.3,
    ("timedelta", "int"): 0.4,

    #  TEXT /BYTES
    ("str", "bytes"): 0.7,
    ("bytes", "str"): 0.7,

    ("bytes", "bytearray"): 0.2,
    ("bytearray", "bytes"): 0.2,

    ("memoryview", "bytes"): 0.3,
    ("bytes", "memoryview"): 0.6,

    # CONTAINERS 
    ("list", "tuple"): 0.3,
    ("tuple", "list"): 0.3,

    ("list", "set"): 0.6,
    ("set", "list"): 0.5,

    ("set", "frozenset"): 0.2,
    ("frozenset", "set"): 0.4,

    ("dict", "defaultdict"): 0.2,
    ("defaultdict", "dict"): 0.3,

    # Mapping 
    ("mapping", "dict"): 0.5,
    ("dict", "mapping"): 0.1,

    # ENUM / UUID
    ("enum", "int"): 0.3,
    ("enum", "str"): 0.4,
    ("intenum", "int"): 0.1,
    ("strenum", "str"): 0.1,

    ("uuid", "str"): 0.2,
    ("uuid", "int"): 0.8,
    ("uuid", "bytes"): 0.4,

    # ASYNC
    ("coroutine", "awaitable"): 0.1,
    ("awaitable", "coroutine"): 0.7,

    # ANY / NONE
    ("any", "any"): 0.0,
    ("any", "int"): 0.4,
    ("int", "any"): 0.1,

    ("any", "str"): 0.4,
    ("str", "any"): 0.1,

    ("none", "int"): 1.0,
    ("none", "str"): 1.0,
    ("none", "float"): 1.0,
}



@dataclass
class TypeNode:
    base: str
    args: List["TypeNode"]
    raw: str

    def __repr__(self):
        if self.args:
            return f"{self.base}[{', '.join(repr(a) for a in self.args)}]"
        return self.base

def parse_type_string(s: str) -> TypeNode:
    
    s = (s or "").strip()
    if not s:
        return TypeNode("Any", [], "Any")

    # Normalize Optional[X] -> Union[X, NoneType] for consistent parsing
    optional_match = re.fullmatch(r"Optional\[(.+)\]", s)
    if optional_match:
        inner = optional_match.group(1)
        s = f"Union[{inner}, None]"
    # Top-level union (outside generics)
        # Top-level union (outside generics)
    if "|" in s and not s.startswith("Union["):
        parts = [p.strip() for p in s.split("|")]
        return TypeNode("Union", [parse_type_string(p) for p in parts], s)

    if "[" not in s:
        return TypeNode(s, [], s)
    try:
        base = s[: s.index("[")].strip()
        inner = s[s.index("[") + 1 : -1]
    except Exception:
        return TypeNode("Any", [], s)
    args: List[TypeNode] = []
    depth = 0
    cur = []
    for ch in inner:
        if ch == "[":
            depth += 1
            cur.append(ch)
        elif ch == "]":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            tok = "".join(cur).strip()
            if tok:
                args.append(parse_type_string(tok))
            cur = []
        else:
            cur.append(ch)
    last = "".join(cur).strip()
    if last:
        args.append(parse_type_string(last))
    return TypeNode(base, args, s)

# Type hierarchy (directed DAG), depths, and subtyping-based similarity
class TypeCollector(ast.NodeVisitor):
    """
    Collects class inheritance relationships and updates HONEST_NOMINAL.
    """
    def __init__(self, nominal_graph):
        self.nominal_graph = nominal_graph

    def visit_ClassDef(self, node: ast.ClassDef):
        child_name = node.name
        for base in node.bases:
            if isinstance(base, ast.Name):
                parent_name = base.id
                self.nominal_graph.add_inheritance(child_name, parent_name)
            elif isinstance(base, ast.Attribute):
                parent_name = ast_unparse(base)
                self.nominal_graph.add_inheritance(child_name, parent_name)
        self.generic_visit(node)

class TypeHierarchy:
    """Nominal Python type hierarchy ."""


    def __init__(self):
        # Directed edges: parent -> children 
        
        self.graph: Dict[str, List[str]] = {
            "object": [
                "numeric",
                "sequence",
                "set",
                "mapping",
                "scalar",
                "none_type",
                "datetime_like",
                "enum_like",
                "iterable",
                "callable",         
                "awaitable",         
                "sized",            
                "container",
                "io",
                "type_construct",
                "decimal_like",
                "optional",
            ],

            # NUMERIC
            "numeric": ["int", "float", "complex"],
            "int": ["bool"],
            "bool": [],
            "float": [],
            "complex": [],
            
            

            # CONTAINERS 
            "sequence": ["list", "tuple"],
            "list": [],
            "tuple": [],

            "set": ["frozenset"],
            "frozenset": [],
            #  MAPPING HIERARCHY 
            "mapping": ["dict", "typeddict", "ordereddict"],
            "dict": ["defaultdict"], 
            "defaultdict":[],
            "typeddict": [],
            "ordereddict": [],
            

            #SCALAR TYPES
            "scalar": ["str", "bytes", "uuid", "bytearray", "memoryview", "range"],
            "str": ["literal"],
            "literal": [],
            "bytes": [],
            "uuid": [],
            "bytearray": [],
            "memoryview": [],
            "range": [],
            

            # DATETIME TYPES
            "datetime_like": ["datetime", "date", "time", "timedelta"],
            "datetime": [],
            "date": [],
            "time": [],
            "timedelta": [],

            #  ENUM TYPES 
            "enum_like": ["enum"],
            "enum": ["intenum", "strenum"],
            "intenum": [],
            "strenum": [],
            

            # =TYPE CONSTRUCTORS 
    
            "type_construct": ["type", "annotated", "typevar", "generic", "ParamSpec"],  
            "type": [],
            "annotated": [], 
            "typevar": [],
            "generic":[],
            "ParamSpec":[],

            # MISC
            "none_type": [],
            
            #  OPTIONAL/UNION TYPE CONSTRUCTORS 
            "optional": ["union"],  
            "union": [],

            # ABSTRACT TYPE CONSTRUCTORS
            "iterable": ["iterator", "asynciterable", "reversible"],
            "iterator": ["generator"],
            "generator": [],
            "asynciterable": ["asynciterator"],
            "asynciterator": ["asyncgenerator"],
            "asyncgenerator": [],
            "reversible": [],
    
            "callable": ["function", "method", "coroutine"],
            "function": [],
            "method": [],
            "coroutine": ["asyncfunction"],
            "asyncfunction": [],
    
            "awaitable": ["coroutine", "future"],
            "future": [],
    
            "sized": [],
            "container": [],
            # IO HIERARCHY
            "io": ["textio", "binaryio"],
            "textio": [],
            "binaryio": [],
            #  DECIMAL HIERARCHY 
            "decimal_like": ["decimal", "fraction"],
            "decimal": [],
            "fraction": [],
        }

        # DAG-safe reverse edges (multi-parent)
        self.parents: Dict[str, List[str]] = defaultdict(list)
        for parent, children in self.graph.items():
            for child in children:
                self.parents[child].append(parent)
        self.root = 'object'
        self.depths = self._compute_depths()

    def _compute_depths(self):
       depths = {self.root: 0}
       queue = deque([self.root])

       while queue:
           cur = queue.popleft()
           for child in self.graph.get(cur, []):
               if child not in depths or depths[cur] + 1 < depths[child]:
                   depths[child] = depths[cur] + 1
                   queue.append(child)
       return depths

    
    def _ancestors_with_distance(self, node: str) -> Dict[str,int]:
        distances = {node:0}
        queue = deque([(node,0)])
        while queue:
            current, dist = queue.popleft()
            for parent in self.parents.get(current, []):
                if parent not in distances or dist + 1 < distances[parent]:
                    distances[parent] = dist + 1
                    queue.append((parent, dist + 1))
        return distances
    
    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
       """Check if 'ancestor' is ancestor of 'descendant'."""
       if ancestor == descendant:
          return True
    
       visited = set()
       queue = deque([descendant])
    
       while queue:
           current = queue.popleft()
           if current in visited:
            continue
           visited.add(current)
        
        # Found the ancestor
           if current == ancestor:
              return True
        
        # Check all parents
           for parent in self.parents.get(current, []):
              queue.append(parent)
    
       return False

    def shortest_path_length(self, a: str, b: str) -> int:
        """Compute weighted shortest path L between a and b for the paper measure."""
        if a == b:
           return 0
        visited = set([a])
        queue = deque([(a, 0)])
    
        while queue:
           current, dist = queue.popleft()
        
           # Check children
           for child in self.graph.get(current, []):
               if child == b:
                  return dist + 1
               if child not in visited:
                   visited.add(child)
                   queue.append((child, dist + 1))
        
        # Check parents  
           for parent in self.parents.get(current, []):
               if parent == b:
                   return dist + 1
               if parent not in visited:
                   visited.add(parent)
                   queue.append((parent, dist + 1))
    
        return max(self.depths.values()) * 2  # fallback

    def depth_weighted_similarity(self, a: str, b: str) -> float:
        if a not in self.depths or b not in self.depths:
           return 0.0

        a_anc = self._ancestors_with_distance(a)
        b_anc = self._ancestors_with_distance(b)

        # LCS = deepest common ancestor
        common = set(a_anc) & set(b_anc)
        if not common:
           return 0.0
        lcs = max(common, key=lambda c: self.depths[c])

        N = self.depths[lcs]
        N1 = self.depths[a]
        N2 = self.depths[b]
        L = self.shortest_path_length(a, b)
        D = max(self.depths.values())

        if self._is_ancestor(a, b) or self._is_ancestor(b, a): 
            λ = 0
        else:
            λ = 1  # Cross-domain siblings: branch penalty
        
        # the paper formula
        if (N1 + N2) == 0: return 0.0
        return (2 * N * math.exp(-λ * L / D)) / (N1 + N2)


    
# one mapped hierarchy instance
HIER = TypeHierarchy()
HONEST_NOMINAL, HONEST_SEMANTIC, HONEST_BASE, CONVERSIONS, OPERATIONS = initialize_honest_type_system(
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "complete_type_analysis.json"),
    hier=HIER,
    base=TIERED_MAPPINGS,
    repo_paths=None
)

# Coercion-based similarity (directional)
def coercion_similarity(a: str, b: str) -> float:
    a_n = a.lower(); b_n = b.lower()
    cost = COERCION_COSTS.get((a_n, b_n))
    if cost is None:
        if a_n == "any" and b_n == "any":
            return 1.0
        if a_n == "any" or b_n == "any":
            return 0.2
        return 0.0


    return max(0.0, min(1.0, 1.0 - cost))

# Combined / chosen similarity function and mapping to discrete distance
def compute_similarity(a_node: TypeNode, b_node: TypeNode, mode: str = "combined") -> float:
    a_base = a_node.base.lower()
    b_base = b_node.base.lower()
    a_raw = a_node.raw.lower().strip()
    b_raw = b_node.raw.lower().strip()
     
    # Union handling
    if a_base == "union" or b_base == "union":
        a_items = a_node.args or [a_node]
        b_items = b_node.args or [b_node]
        return _compare_union_items_similarity(a_items, b_items, mode) 
    # Exact match
    if a_raw == b_raw and a_base != "any" and b_base != "any":
        return 1.0

    # Any handling
    if a_base == "any" or b_base == "any":
        if a_base == "any" and b_base == "any":
            return 1.0
        if a_base == "any":
            return 0.3
        if b_base == "any":
            return 0.8
        
    #  Unknown types 

    a_known = a_base in HIER.graph
    b_known = b_base in HIER.graph

    if not a_known and not b_known:
        return 0.4        # unknown → unknown
    if not a_known:
        return 0.2        # unknown → concrete (guessing)
    if not b_known:
        return 0.6        # concrete → unknown (loss)


    # DAG subtyping similarity
    sub = HIER.depth_weighted_similarity(a_base, b_base)

    # Coercion similarity
    co = coercion_similarity(a_base, b_base)

    # Blend based on mode
    if mode == "subtyping":
        base_sim = sub
    elif mode == "coercion":
        base_sim = co
    else:  # combined
        base_sim = 0.6 * sub + 0.4 * co  

    

    # Generic args
    if a_node.args and b_node.args:
        sims = [compute_similarity(a_node.args[i], b_node.args[i], mode) for i in range(min(len(a_node.args), len(b_node.args)))]
        avg_args = sum(sims) / len(sims) if sims else 0.0
        penalty = abs(len(a_node.args) - len(b_node.args)) * 0.1  
        args_sim = max(0.0, avg_args - penalty)
    elif a_node.args or b_node.args:
        args_sim = 0.4  
    else:
        args_sim = 0.6  

    final = 0.7 * base_sim + 0.3 * args_sim
    return max(0.0, min(1.0, final))


def _compare_union_items_similarity(a_items: List[TypeNode], b_items: List[TypeNode], mode: str) -> float:
    m = [[compute_similarity(ai, bj, mode) for bj in b_items] for ai in a_items]
    entries = [(m[i][j], i, j) for i in range(len(a_items)) for j in range(len(b_items))]
    entries.sort(key=lambda t: (-t[0], repr(a_items[t[1]]), repr(b_items[t[2]])))
    used_a, used_b = set(), set()
    total = 0.0
    matches = 0
    for score, i, j in entries:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        total += score
        matches += 1
    if max(len(a_items), len(b_items)) == 0:
        return 1.0
    return total / max(len(a_items), len(b_items))

def similarity_to_distance(sim: float) -> int: 

    if sim == 1.0:
        return 0
    if sim >= 0.5:
        return 1 
    if sim >= 0.3:
        return 2 
    return 3 

# Candidate selection + AST mutation helpers
RNG_SEED_FOR_CANDIDATE = 42

def get_error_candidate(original_type_str: str, rng: Optional[random.Random] = None, mode: str = "subtyping") -> Dict[str, Any]:
    if rng is None:
        rng = random.Random(RNG_SEED_FOR_CANDIDATE)
    
    norm = original_type_str.strip()
    norm_lower = norm.lower()
    
    
    
    # ========== 0. HANDLE BARE NONE/NONETYPE ==========
    if norm_lower in {"none", "nonetype", "type[none]"}:
        # 50% keep as None (no mutation)
        if rng.random() < 0.5:
            return {
                "type": "None",
                "similarity": 1.0,
                "distance": 0,
                "rule_id": "none_noop",
                "method": "none_no_mutation"
            }
        
        # 50% mutate to Optional[T]
        candidates = [
            "Optional[Any]",      # Structural change
            "Optional[int]",      # Concrete optional
            "Optional[str]",      # Concrete optional  
            "Optional[bool]",     # Concrete optional
            "Optional[float]",    # Concrete optional
        ]
        cand_str = rng.choice(candidates)
        
        # Calculate similarity
        orig_node = parse_type_string("None")
        cand_node = parse_type_string(cand_str)
        sim = compute_similarity(orig_node, cand_node, mode=mode)
        dist = similarity_to_distance(sim)
        
        return {
            "type": cand_str,
            "similarity": sim,
            "distance": dist,
            "rule_id": "none",
            "method": "none_to_optional"
        }
    
    # ========== 1. HANDLE UNIONS/OPTIONAL ==========
    if "|" in norm or norm.startswith("Union[") or norm.startswith("Optional["):
        # Parse to understand the union structure
        parsed = parse_type_string(norm)
        
        if parsed.base.lower() == "union":
            non_none_nodes = [n for n in parsed.args if n.base.lower() not in {"none", "nonetype"}]
            has_none = any(n.base.lower() in {"none", "nonetype"} for n in parsed.args)
            
            if not non_none_nodes:
                # Only None in union (e.g., Union[None] or Optional[None])
                # 50% keep as None, 50% mutate to Optional[Any]
                if rng.random() < 0.5:
                    return {
                        "type": "None",
                        "similarity": 1.0,
                        "distance": 0,
                        "rule_id": "none_only_noop",
                        "method": "none_only_no_mutation"
                    }
                else:
                    return {
                        "type": "Optional[Any]",
                        "similarity": 0.05,
                        "distance": 3,
                        "rule_id": "none_only",
                        "method": "none_only_to_optional"
                    }
            
            # Choose a non-None type to mutate
            node_to_mutate = rng.choice(non_none_nodes)
            type_to_mutate = node_to_mutate.raw
            
            # Get mutation for that specific type using RECURSIVE CALL
            mutation_result = get_error_candidate(type_to_mutate, rng, mode)
            mutated_type = mutation_result["type"]
            
            # Reconstruct union
            result_parts = []
            for node in parsed.args:
                node_base_lower = node.base.lower()
                if node_base_lower in {"none", "nonetype"}:
                    result_parts.append("None")
                elif node.raw == type_to_mutate:
                    result_parts.append(mutated_type)
                else:
                    result_parts.append(node.raw)
            
            result_type = " | ".join(result_parts)
            
            # Calculate similarity ONLY on non-None components
            orig_components = [n for n in parsed.args if n.base.lower() not in {'none', 'nonetype'}]
            cand_components = []
            for part in result_parts:
                node = parse_type_string(part)
                if node.base.lower() not in {'none', 'nonetype'}:
                    cand_components.append(node)
            
            if orig_components and cand_components:
                for i, orig_comp in enumerate(orig_components):
                    if orig_comp.raw == type_to_mutate:
                        type_sim = compute_similarity(orig_comp, cand_components[i], mode)
                        break
                else:
                    type_sim = compute_similarity(orig_components[0], cand_components[0], mode)
            else:
                type_sim = 0.0
            
            # Apply penalty if Optional structure changed
            none_changed = (any(n.base.lower() in {'none', 'nonetype'} for n in parsed.args) != 
                          ('None' in result_parts))
            penalty = 0.05 if none_changed else 0.0
            
            sim = max(0.0, type_sim - penalty)
            
            dist = similarity_to_distance(sim)
            
            return {
                "type": result_type,
                "similarity": sim,
                "distance": dist,
                "rule_id": f"union_{mutation_result.get('rule_id', 'unknown')}",
                "method": f"union_component({mutation_result.get('method', 'unknown')})"
            }
    
    #  2. HANDLE GENERIC TYPES 
    if "[" in norm and norm.endswith("]"):
        container_end = norm.index("[")
        container = norm[:container_end].strip()
        inner_full = norm[container_end + 1:-1].strip()
        
        inners = []
        depth = 0
        current = []
        
        for ch in inner_full:
            if ch == "[":
                depth += 1
                current.append(ch)
            elif ch == "]":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                inner_type = "".join(current).strip()
                if inner_type:
                    inners.append(inner_type)
                current = []
            else:
                current.append(ch)
        
        if current:
            inner_type = "".join(current).strip()
            if inner_type:
                inners.append(inner_type)
        
        if not inners:
            inners = [inner_full]
        
        container_lower = container.lower()
        
        # Choose mutation strategy
        strategy_weights = {
            "container": 0.4,      
            "inner_single": 0.4,   
            "abstract": 0.2        
        }
        
        strategy = rng.choices(
            list(strategy_weights.keys()),
            weights=list(strategy_weights.values()),
            k=1
        )[0]
        
        if strategy == "container":
            # Keep inner types, change container
            if container_lower in TIERED_MAPPINGS:
                # Get container candidates (non-generic only)
                container_candidates = [
                    c for c in TIERED_MAPPINGS[container_lower] 
                    if "[" not in c  # No pre-instantiated generics
                ]
                
                if container_candidates:
                    new_container = rng.choice(container_candidates)
                    # Capitalize Python built-in containers for AST parsing
                    if new_container.lower() in {"set", "tuple", "list", "dict", "sequence", "iterable", "mapping"}:
                        new_container = new_container.capitalize()
                    # Preserve original case style
                    elif container[0].isupper():
                        new_container = new_container[0].upper() + new_container[1:]
                    
                    cand_str = f"{new_container}[{inner_full}]"
                else:
                    # Use abstract mapping instead of "Any"
                    abstract_map = {
                        "list": ["Iterable", "Sequence"],
                        "List": ["Iterable", "Sequence"],
                        "dict": ["Mapping"],
                        "Dict": ["Mapping"],
                        "set": ["Iterable"],
                        "Set": ["Iterable"],
                        "tuple": ["Iterable", "Sequence"],
                        "Tuple": ["Iterable", "Sequence"],
                        "sequence": ["Iterable", "List"],
                        "Sequence": ["Iterable", "List"],
                    }
                    cand_str = rng.choice(abstract_map.get(container, abstract_map.get(container_lower, ["Iterable"])))
            else:
                # Use abstract mapping for unknown containers
                abstract_map = {
                    "list": ["Iterable", "Sequence"],
                    "List": ["Iterable", "Sequence"],
                    "dict": ["Mapping"],
                    "Dict": ["Mapping"],
                    "set": ["Iterable"],
                    "Set": ["Iterable"],
                    "tuple": ["Iterable", "Sequence"],
                    "Tuple": ["Iterable", "Sequence"],
                    "sequence": ["Iterable", "List"],
                    "Sequence": ["Iterable", "List"],
                }
                cand_str = rng.choice(abstract_map.get(container, abstract_map.get(container_lower, ["Iterable"])))
        
        elif strategy == "inner_single" and inners:
            # SMART: Choose which inner type to mutate
            # Prefer primitives over library/custom types
            
            # 1. Identify primitive inner types (safe to mutate)
            primitive_indices = []
            library_indices = []
            
            for i, inner in enumerate(inners):
                # Check if it's a primitive type (in TIERED_MAPPINGS and no dots)
                if inner.lower() in TIERED_MAPPINGS and "." not in inner:
                    primitive_indices.append(i)
                else:
                    library_indices.append(i)
            
            
            # 2. Choose which index to mutate
            if primitive_indices:
                # Prefer to mutate primitives (they're safe)
                idx_to_mutate = rng.choice(primitive_indices)
            elif library_indices:
                # Only library types - choose randomly
                idx_to_mutate = rng.choice(library_indices)
            else:
                # Should not happen, but fallback
                idx_to_mutate = 0
            
            
            # 3. Mutate the chosen inner type
            inner_to_mutate = inners[idx_to_mutate]
            inner_result = get_error_candidate(inner_to_mutate, rng, mode)
            
            # 4. Reconstruct with mutated inner
            new_inners = inners.copy()
            new_inners[idx_to_mutate] = inner_result["type"]
            
            if len(new_inners) == 1:
                cand_str = f"{container}[{new_inners[0]}]"
            else:
                cand_str = f"{container}[{', '.join(new_inners)}]"
            
        
        else:  # abstract
            # Convert to abstract type (lose inner types)
            abstract_map = {
                "list": ["Iterable", "Sequence"],
                "List": ["Iterable", "Sequence"],
                "dict": ["Mapping"],
                "Dict": ["Mapping"],
                "set": ["Iterable"],
                "Set": ["Iterable"],
                "tuple": ["Iterable", "Sequence"],
                "Tuple": ["Iterable", "Sequence"],
                "sequence": ["Iterable", "List"],
                "Sequence": ["Iterable", "List"],
            }
            
            cand_str = rng.choice(abstract_map.get(container, abstract_map.get(container_lower, ["Iterable"])))
        
        # Calculate similarity for generic mutation
        orig_node = parse_type_string(norm)
        cand_node = parse_type_string(cand_str)
        sim = compute_similarity(orig_node, cand_node, mode=mode)
        dist = similarity_to_distance(sim)
        
        
        return {
            "type": cand_str,
            "similarity": sim,
            "distance": dist,
            "rule_id": container_lower,
            "method": f"generic_{strategy}"
        }
    
    # ========== 3. HANDLE SIMPLE TYPES ==========
    norm_lower = norm.lower()
    
    if norm_lower in TIERED_MAPPINGS:
        candidates = [c for c in TIERED_MAPPINGS[norm_lower] if c.lower() != norm_lower]
        
        if candidates:
            cand_str = rng.choice(candidates)
            
            # Calculate similarity
            orig_node = parse_type_string(norm)
            cand_node = parse_type_string(cand_str)
            sim = compute_similarity(orig_node, cand_node, mode=mode)
            dist = similarity_to_distance(sim)
            
            
            return {
                "type": cand_str,
                "similarity": sim,
                "distance": dist,
                "rule_id": norm_lower,
                "method": "tiered_mapping"
            }
    
    # 4. USE user_defined_injector.py (BRAIN 2) 
    def normalize_type_name(type_str: str) -> str:
        """Remove module prefixes like pandas., pd., numpy., etc."""
        if "." in type_str:
            # Get the class name after the last dot
            # pandas.DataFrame → DataFrame, pd.DataFrame → DataFrame
            return type_str.split(".")[-1]
        return type_str
    
    normalized_type = normalize_type_name(norm)
    
    honest_candidates = generate_mutation_candidates(
        normalized_type, HONEST_NOMINAL, HONEST_SEMANTIC, HONEST_BASE, CONVERSIONS, OPERATIONS
    )
    
    if honest_candidates:
        cand_obj = rng.choice(honest_candidates)
        cand_str = cand_obj.type_name
        sim_result = compute_honest_similarity(
            a=normalized_type,
            b=cand_str,
            nominal=HONEST_NOMINAL,
            semantic=HONEST_SEMANTIC,
            conversions=CONVERSIONS,
            operations=OPERATIONS
            
        )
        sim = sim_result.score if sim_result.score is not None else 0.0
        dist = sim_result.distance
        
        return {
            "type": cand_str,
            "similarity": sim,
            "distance": dist,
            "rule_id": "honest_system",
            "method": sim_result.kind
        }
    
    # ========== 5. FALLBACK ==========
    # For library/custom types, use "object" instead of "Any"
    if "[" in norm and norm.endswith("]"):
        # Generic type fell through - use abstract type
        container_end = norm.index("[")
        container = norm[:container_end].strip()
        
        abstract_map = {
            "list": ["Iterable", "Sequence"],
            "List": ["Iterable", "Sequence"],
            "dict": ["Mapping"],
            "Dict": ["Mapping"],
            "set": ["Iterable"],
            "Set": ["Iterable"],
            "tuple": ["Iterable", "Sequence"],
            "Tuple": ["Iterable", "Sequence"],
            "sequence": ["Iterable", "List"],
            "Sequence": ["Iterable", "List"],
        }
        
        # Check both exact and lowercase
        if container in abstract_map:
            cand_str = rng.choice(abstract_map[container])
            method = "abstract_fallback"
        elif container.lower() in abstract_map:
            cand_str = rng.choice(abstract_map[container.lower()])
            method = "abstract_fallback"
        else:
            # Unknown container type - use Iterable as default
            cand_str = "Iterable"
            method = "default_container_fallback"
    
    # SECOND: Check if it's a union that fell through
    elif "|" in norm or norm.startswith("Union[") or norm.startswith("Optional["):
        # Simplify union to single type
        parsed = parse_type_string(norm)
        non_none = [n for n in parsed.args if n.base.lower() not in {"none", "nonetype"}]
        
        if non_none:
            # Keep only one non-None type
            keep_type = rng.choice(non_none)
            # Try to get a valid mutation for this type
            try_again = get_error_candidate(keep_type.raw, rng, mode)
            if try_again["type"] != "Any" and try_again["type"] != "object":
                cand_str = try_again["type"]
                method = f"union_simplify_{try_again['method']}"
            else:
                cand_str = keep_type.raw
                method = "union_keep_one"
        else:
            cand_str = "Any"
            method = "empty_union_fallback"
    
    # THIRD: Original fallback logic
    elif "." in norm or (norm and norm[0].isupper()):
        cand_str = "object"
        method = "object_fallback"
    else:
        cand_str = "Any"
        method = "any_fallback"
    
    sim = 0.1
    dist = similarity_to_distance(sim)
    
    
    return {
        "type": cand_str,
        "similarity": sim,
        "distance": dist,
        "rule_id": "improved_fallback",
        "method": method
    }
    
def ast_annotation_to_type_string(node: ast.AST) -> str:
    try:
        s = ast_unparse(node).strip()
        return s
    except Exception:
        return "Any"

def create_semantic_error_with_meta(original_annotation: ast.AST, rng: Optional[random.Random] = None, mode: str = "subtyping") -> Tuple[ast.AST, Dict[str, Any]]:
    if rng is None: rng = random.Random(RNG_SEED_FOR_CANDIDATE)
    orig_str = ast_annotation_to_type_string(original_annotation)
    norm = orig_str.replace(" ", "")
    cand = get_error_candidate(norm, rng, mode=mode)
    try:
        new_node = ast.parse(cand["type"], mode="eval").body
    except Exception:
        new_node = ast.parse("Any", mode="eval").body
        cand.update({"type": "Any", "distance": 3, "method": "parse_fallback"})
    metadata = {
        "original_type": orig_str,
        "injected_type": cand["type"],
        "distance_score": int(cand["distance"]),
        "mutation_footprint": cand["method"],
        "similarity": float(cand.get("similarity", 0.0)),
    }
    return new_node, metadata

# AST mutation across functions
def inject_type_errors(node: ast.FunctionDef, rng: Optional[random.Random] = None, mode: str = "subtyping") -> Tuple[ast.FunctionDef, List[Dict[str, Any]]]:
    if rng is None: rng = random.Random(RNG_SEED_FOR_CANDIDATE)
    mutations: List[Dict[str, Any]] = []
    for arg in node.args.args:
        if getattr(arg, "annotation", None) is not None:
            new_node, meta = create_semantic_error_with_meta(arg.annotation, rng, mode=mode)
            arg.annotation = new_node
            meta["location"] = f"arg:{arg.arg}"
            mutations.append(meta)
    if node.args.vararg and getattr(node.args.vararg, "annotation", None) is not None:
        new_node, meta = create_semantic_error_with_meta(node.args.vararg.annotation, rng, mode=mode)
        node.args.vararg.annotation = new_node
        meta["location"] = f"vararg:{node.args.vararg.arg}"
        mutations.append(meta)
    for arg in node.args.kwonlyargs:
        if getattr(arg, "annotation", None) is not None:
            new_node, meta = create_semantic_error_with_meta(arg.annotation, rng, mode=mode)
            arg.annotation = new_node
            meta["location"] = f"kwonlyarg:{arg.arg}"
            mutations.append(meta)
    if node.args.kwarg and getattr(node.args.kwarg, "annotation", None) is not None:
        new_node, meta = create_semantic_error_with_meta(node.args.kwarg.annotation, rng, mode=mode)
        node.args.kwarg.annotation = new_node
        meta["location"] = f"kwarg:{node.args.kwarg.arg}"
        mutations.append(meta)
    if getattr(node, "returns", None) is not None:
        new_node, meta = create_semantic_error_with_meta(node.returns, rng, mode=mode)
        node.returns = new_node
        meta["location"] = "return"
        mutations.append(meta)
    return node, mutations

class FunctionMutator(ast.NodeTransformer):
    def __init__(self, rng_seed: int = DEFAULT_SEED, mode: str = "subtyping"):
        super().__init__()
        self.rng = random.Random(rng_seed)
        self.mode = mode
        self.mutations: List[Dict[str, Any]] = []
        self.function_metadata: List[Dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        original_name = getattr(node, "name", "<lambda>")
        mutated_node, muts = inject_type_errors(node, self.rng, mode=self.mode)
        if muts:
            for m in muts:
                m.update({"function": original_name})
            self.mutations.extend(muts)
            self.function_metadata.append({"name": original_name, "mutations": muts})
        self.generic_visit(mutated_node)
        return mutated_node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        original_name = getattr(node, "name", "<lambda>")
        mutated_node, muts = inject_type_errors(node, self.rng, mode=self.mode)
        if muts:
            for m in muts:
                m.update({"function": original_name})
            self.mutations.extend(muts)
            self.function_metadata.append({"name": original_name, "mutations": muts})
        self.generic_visit(mutated_node)
        return mutated_node

# Typing import sanitizer

TYPING_PRIMS = {"Any", "Optional", "Union", "List", "Tuple", "Set", "Dict", "Mapping", "Sequence", "Callable", "Type"}

def ensure_typing_imports(tree: ast.Module) -> ast.Module:
    needed = set()
    class Collector(ast.NodeVisitor):
        def visit_Subscript(self, node):
            base = node.value
            if isinstance(base, ast.Name):
                if base.id in TYPING_PRIMS:
                    needed.add(base.id)
            elif isinstance(base, ast.Attribute):
                if isinstance(base.value, ast.Name) and base.value.id == "typing":
                    needed.add(base.attr)
            self.generic_visit(node)
        def visit_Name(self, node):
            if node.id in {"Any", "Mapping", "Sequence"}:
                needed.add(node.id)
            self.generic_visit(node)
    Collector().visit(tree)
    if not needed:
        return tree
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            existing = {alias.name for alias in node.names}
            missing = sorted(needed - existing)
            if missing:
                node.names.extend([ast.alias(name=m, asname=None) for m in missing])
            return tree
    insert_at = 0
    if tree.body:
        first = tree.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            insert_at = 1
    import_node = ast.ImportFrom(module="typing", names=[ast.alias(name=n, asname=None) for n in sorted(needed)], level=0)
    tree.body.insert(insert_at, import_node)
    return tree

# File processing and repo handling
def count_functions_with_annotations(file_content: str, file_path: str, rng_seed: int = DEFAULT_SEED, mode: str = "subtyping") -> Tuple[int,int,int,int,float,List[str],str,List[dict]]:
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return 0,0,0,0,0.0,[],file_content,[]
    collector = TypeCollector(HONEST_NOMINAL)
    collector.visit(tree)
    mutator = FunctionMutator(rng_seed, mode=mode)
    tree = mutator.visit(tree)
    ast.fix_missing_locations(tree)
    tree = ensure_typing_imports(tree)
    try:
        mutated_code = ast_unparse(tree)
    except Exception:
        mutated_code = file_content
    total_functions = 0
    annotated_functions = 0
    functions_with_defaults = 0
    total_annotations = 0
    function_names: List[str] = []
    for node in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        total_functions += 1
        function_names.append(node.name)
        has_default = len(node.args.defaults) > 0
        annotations_count = sum(1 for arg in node.args.args if arg.annotation) + (1 if node.returns else 0)
        if annotations_count > 0:
            annotated_functions += 1
        total_annotations += annotations_count
        if has_default:
            functions_with_defaults += 1
    avg_annot = total_annotations / total_functions if total_functions else 0.0
    return (total_functions, annotated_functions, functions_with_defaults, total_annotations, avg_annot, function_names, mutated_code, mutator.mutations)

def write_mutated_file(original_path: str, mutated_code: str, repo_name: str):
    out_dir = os.path.join(MUTATED_OUTPUT_DIR, repo_name, os.path.dirname(original_path))
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, os.path.basename(original_path))
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(mutated_code)


def process_local_tree(source_dir: str, out_dir: str, rng_seed: int = DEFAULT_SEED, mode: str = "subtyping"):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    shutil.copytree(source_dir, out_dir)
    manifest: List[Dict[str, Any]] = []
    for root, _, files in os.walk(out_dir):
        for fname in files:
            if not fname.endswith(".py"): continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, out_dir)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    src = f.read()
            except Exception as e:
                print(f"Could not read {fpath}: {e}")
                continue
            results = count_functions_with_annotations(src, rel, rng_seed, mode)
            _, _, _, _, _, _, mutated_code, mutations = results
            if mutations:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(mutated_code)
                for m in mutations:
                    m.update({"file_path": rel})
                    manifest.append(m)
    manifest_path = os.path.join(out_dir, "injection_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump({"seed": rng_seed, "mutations": manifest}, mf, indent=2)
    return manifest_path

# CLI
def main():
    ap = argparse.ArgumentParser(description="Semantic type-error injector (end-to-end)")
    ap.add_argument("--source_dir", type=str, help="Local source dir")  
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for mutated repos/files")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    ap.add_argument("--mode", choices=["subtyping", "coercion", "combined"], default="subtyping",
                    help="Distance mode: 'subtyping' (DAG) default, 'coercion' (directional coercions), or 'combined'")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)  

    if args.source_dir:
        manifest_path = process_local_tree(args.source_dir, args.out_dir, args.seed, mode=args.mode)
        print("Wrote manifest to", manifest_path)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()

