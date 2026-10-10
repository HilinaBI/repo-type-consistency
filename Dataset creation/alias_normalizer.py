import ast
import re
from typing import Dict, Tuple, Optional, Set

def build_alias_table(file_content: str) -> Dict[str, str]:
    """
    Build alias table from imports.
    
    Returns:
        Dict mapping local names → fully qualified canonical names
        
    Examples:
        import numpy as np           → {"np": "numpy"}
        import numpy                 → {"numpy": "numpy"}
        from pandas import DataFrame → {"DataFrame": "pandas.DataFrame"}
        from pandas import DataFrame as DF → {"DF": "pandas.DataFrame"}
    """
    alias_table = {}
    
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return alias_table
    
    for node in ast.walk(tree):
        # Case 1 & 2: import numpy as np  OR  import numpy
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname if alias.asname else alias.name
                alias_table[local_name] = alias.name
        
        # Case 3, 4, 5: from pandas import DataFrame
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname if alias.asname else alias.name
                full_name = f"{module}.{alias.name}" if module else alias.name
                alias_table[local_name] = full_name
    
    return alias_table


class TypeNormalizer(ast.NodeTransformer):
    """AST transformer that replaces aliases with canonical names."""
    
    def __init__(self, alias_table: Dict[str, str]):
        self.alias_table = alias_table
    
    def visit_Name(self, node: ast.Name):
        """Replace a simple name like 'DataFrame' or 'np'."""
        if node.id in self.alias_table:
            full_name = self.alias_table[node.id]
            # Convert to dotted attribute (e.g., pandas.DataFrame)
            if '.' in full_name:
                parts = full_name.split('.')
                attr_node = ast.Name(id=parts[0], ctx=ast.Load())
                for part in parts[1:]:
                    attr_node = ast.Attribute(value=attr_node, attr=part, ctx=ast.Load())
                return attr_node
            else:
                return ast.Name(id=full_name, ctx=node.ctx)
        return node
    
    def visit_Attribute(self, node: ast.Attribute):
        """Handle attribute access like np.array."""
        if isinstance(node.value, ast.Name):
            if node.value.id in self.alias_table:
                full_base = self.alias_table[node.value.id]
                if '.' in full_base:
                    parts = full_base.split('.')
                    attr_node = ast.Name(id=parts[0], ctx=ast.Load())
                    for part in parts[1:]:
                        attr_node = ast.Attribute(value=attr_node, attr=part, ctx=ast.Load())
                    return ast.Attribute(value=attr_node, attr=node.attr, ctx=node.ctx)
                else:
                    node.value.id = full_base
        return node
    
    def visit_Subscript(self, node: ast.Subscript):
        """Handle generic types like List[DataFrame]."""
        self.generic_visit(node.value)
        self.generic_visit(node.slice)
        return node
    
    def visit_Import(self, node: ast.Import):
        """Remove import statements (they are not needed after normalization)."""
        return None
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Remove import statements (they are not needed after normalization)."""
        return None


def normalize_code(code: str, alias_table: Dict[str, str]) -> str:
    """
    Normalize code by:
    1. Removing all import statements
    2. Replacing aliases with canonical names
    
    BEFORE:
        import pandas as pd
        df: DataFrame = pd.DataFrame()
        
    AFTER:
        df: pandas.DataFrame = pandas.DataFrame()
    """
    if not alias_table:
        return code
    
    try:
        tree = ast.parse(code)
        transformer = TypeNormalizer(alias_table)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        
        # Convert back to code
        import astor
        return astor.to_source(new_tree).strip()
    except Exception:
        return code


def normalize_type(type_str: str, alias_table: Dict[str, str]) -> str:
    """Normalize a single type string."""
    if not type_str or not alias_table:
        return type_str
    
    try:
        tree = ast.parse(type_str, mode='eval')
        transformer = TypeNormalizer(alias_table)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        return ast.unparse(new_tree).strip()
    except (SyntaxError, TypeError, AttributeError):
        # Fallback: simple string replacement
        result = type_str
        for alias, full in sorted(alias_table.items(), key=lambda x: -len(x[0])):
            pattern = r'\b' + re.escape(alias) + r'\b'
            result = re.sub(pattern, full, result)
        return result


def remove_imports(code: str) -> str:
    """Remove all import statements from code."""
    try:
        tree = ast.parse(code)
        
        class ImportRemover(ast.NodeTransformer):
            def visit_Import(self, node):
                return None
            def visit_ImportFrom(self, node):
                return None
        
        new_tree = ImportRemover().visit(tree)
        ast.fix_missing_locations(new_tree)
        
        import astor
        return astor.to_source(new_tree).strip()
    except Exception:
        return code