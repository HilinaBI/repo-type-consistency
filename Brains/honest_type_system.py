import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math
import hashlib
from difflib import SequenceMatcher

_BRAIN2_ENGINE = None

@dataclass
class SimilarityResult:
    score: float
    distance: int
    explanation: str
    kind: str = "semantic"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class MutationCandidate:
    type_name: str
    similarity: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MutationConfig:
    semantic_threshold: float = 0.7
    inheritance_threshold: float = 0.6
    pattern_threshold: float = 0.5
    semantic_weight: float = 0.5
    inheritance_weight: float = 0.3
    pattern_weight: float = 0.2
    max_semantic_candidates: int = 5
    max_inheritance_candidates: int = 5
    max_pattern_candidates: int = 10
    min_similarity: float = 0.3

class SemanticSimilarityCalculator:
    def __init__(self):
        self.semantic_categories = {
            'data_container': {'data', 'model', 'entity', 'schema', 'dto'},
            'service': {'service', 'manager', 'handler', 'processor'},
            'factory': {'factory', 'builder', 'creator', 'maker'},
            'repository': {'repo', 'repository', 'store', 'dao'},
            'controller': {'controller', 'endpoint', 'router', 'api'},
            'utility': {'util', 'helper', 'tool', 'common'},
            'config': {'config', 'settings', 'options'},
        }
        self.cache = {}
        self.name_cache = {}
        self._build_purpose_cache()
    
    def _build_purpose_cache(self):
        self.purpose_to_category = {}
        for category, purposes in self.semantic_categories.items():
            for purpose in purposes:
                self.purpose_to_category[purpose] = category
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        if name1 == name2:
            return 1.0
        
        cache_key = (name1, name2)
        if cache_key in self.name_cache:
            return self.name_cache[cache_key]
        
        base1 = name1
        base2 = name2
        
        suffixes = ['Service', 'Manager', 'Handler', 'Processor', 'Controller', 
                   'Factory', 'Builder', 'Repository', 'Adapter', 'Svc', 'Mgr',
                   'Hdlr', 'Ctrl', 'Proc', 'Bldr']
        
        for suffix in suffixes:
            if base1.endswith(suffix):
                base1 = base1[:-len(suffix)]
            if base2.endswith(suffix):
                base2 = base2[:-len(suffix)]
        
        if (base1 + 's' == base2) or (base2 + 's' == base1):
            self.name_cache[cache_key] = 0.8
            return 0.8
        
        verb_noun_pairs = {
            'summarize': 'summary', 'process': 'processor', 'manage': 'manager',
            'handle': 'handler', 'create': 'creator', 'build': 'builder',
            'stream': 'streaming', 'log': 'logger', 'upload': 'uploader',
            'query': 'querier', 'eval': 'evaluator', 'sync': 'syncer'
        }
        
        if base1 in verb_noun_pairs and verb_noun_pairs[base1] == base2:
            self.name_cache[cache_key] = 0.8
            return 0.8
        if base2 in verb_noun_pairs and verb_noun_pairs[base2] == base1:
            self.name_cache[cache_key] = 0.8
            return 0.8
        
        if base1 in base2 or base2 in base1:
            self.name_cache[cache_key] = 0.7
            return 0.7
        
        if len(base1) > 3 and len(base2) > 3:
            similarity = SequenceMatcher(None, base1.lower(), base2.lower()).ratio()
            if similarity > 0.8:
                self.name_cache[cache_key] = similarity * 0.9
                return similarity * 0.9
        
        self.name_cache[cache_key] = 0.3
        return 0.3
    
    def calculate(self, type1: Dict, type2: Dict) -> float:
        key = f"{type1.get('id', '')}|{type2.get('id', '')}"
        if key in self.cache:
            return self.cache[key]
        
        purpose1 = type1['semantic']['primary_purpose']
        purpose2 = type2['semantic']['primary_purpose']
        
        if purpose1 == purpose2:
            purpose_score = 1.0
        else:
            cat1 = self.purpose_to_category.get(purpose1)
            cat2 = self.purpose_to_category.get(purpose2)
            purpose_score = 0.8 if cat1 and cat2 and cat1 == cat2 else 0.2
        
        domains1 = set(type1['semantic']['domain_keywords'])
        domains2 = set(type2['semantic']['domain_keywords'])
        
        if not domains1 or not domains2:
            domain_score = 0.3
        else:
            intersection = domains1 & domains2
            union = domains1 | domains2
            domain_score = len(intersection) / len(union) if union else 0.0
        
        caps1 = set(type1['semantic']['business_capabilities'])
        caps2 = set(type2['semantic']['business_capabilities'])
        
        if not caps1 or not caps2:
            capability_score = 0.2
        else:
            intersection = caps1 & caps2
            union = caps1 | caps2
            capability_score = len(intersection) / len(union) if union else 0.0
        
        dist1 = type1['semantic']['method_role_distribution']
        dist2 = type2['semantic']['method_role_distribution']
        
        if not dist1 or not dist2:
            method_score = 0.3
        else:
            all_roles = list(set(dist1.keys()) | set(dist2.keys()))
            if not all_roles:
                method_score = 0.0
            else:
                vec1 = np.array([dist1.get(role, 0) for role in all_roles])
                vec2 = np.array([dist2.get(role, 0) for role in all_roles])
                
                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                method_score = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        sem1 = type1['semantic']
        sem2 = type2['semantic']
        
        scores = []
        
        inst1 = set(sem1.get('instantiation_contexts', []))
        inst2 = set(sem2.get('instantiation_contexts', []))
        if inst1 and inst2:
            intersection = inst1 & inst2
            union = inst1 | inst2
            scores.append(len(intersection) / len(union) if union else 0.0)
        
        err1 = sem1.get('error_handling_pattern')
        err2 = sem2.get('error_handling_pattern')
        if err1 == err2:
            scores.append(1.0)
        elif err1 and err2:
            scores.append(0.5)
        else:
            scores.append(0.3)
        
        res1 = set(sem1.get('resource_management', []))
        res2 = set(sem2.get('resource_management', []))
        if res1 and res2:
            intersection = res1 & res2
            union = res1 | res2
            scores.append(len(intersection) / len(union) if union else 0.0)
        
        usage_score = np.mean(scores) if scores else 0.3
        
        name1 = type1.get('name', '').split('.')[-1]
        name2 = type2.get('name', '').split('.')[-1]
        name_similarity = self._calculate_name_similarity(name1, name2)
        
        weights = [0.25, 0.15, 0.15, 0.2, 0.1, 0.15]
        scores_list = [purpose_score, domain_score, capability_score, method_score, usage_score, name_similarity]
        similarity = sum(w * s for w, s in zip(weights, scores_list))
        
        if purpose1 == purpose2:
            similarity = min(1.0, similarity + 0.15)
        
        self.cache[key] = similarity
        return similarity

class InheritanceSimilarityCalculator:
    def __init__(self):
        self.cache = {}
    
    def _is_library_compatible(self, type1: Dict, type2: Dict) -> bool:
        fw1 = type1['inheritance']['framework_hierarchy']
        fw2 = type2['inheritance']['framework_hierarchy']
        
        compatible_libs = {
            'fastapi': {'starlette', 'pydantic'},
            'torch': {'tensor', 'numpy'},
            'pandas': {'numpy'},
            'sqlalchemy': {'orm'},
            'django': {'models'},
            'numpy': {'torch'},
            'tensorflow': {'keras'},
            'logging': {'log'},
            'io': {'StringIO', 'BytesIO'},
        }
        
        libs1 = set(fw1.keys())
        libs2 = set(fw2.keys())
        
        if libs1 & libs2:
            return True
        
        for lib1 in libs1:
            for lib2 in libs2:
                if lib1 in compatible_libs and lib2 in compatible_libs[lib1]:
                    return True
                if lib2 in compatible_libs and lib1 in compatible_libs[lib2]:
                    return True
        
        return False
    
    def calculate(self, type1: Dict, type2: Dict) -> float:
        key = f"{type1.get('id', '')}|{type2.get('id', '')}"
        if key in self.cache:
            return self.cache[key]
        
        chain1 = set(type1['inheritance']['inheritance_chain'])
        chain2 = set(type2['inheritance']['inheritance_chain'])
        
        if not chain1 or not chain2:
            chain_score = 0.2
        else:
            intersection = chain1 & chain2
            union = chain1 | chain2
            jaccard = len(intersection) / len(union) if union else 0.0
            length_sim = 1.0 - abs(len(chain1) - len(chain2)) / max(len(chain1), len(chain2))
            chain_score = (0.7 * jaccard) + (0.3 * length_sim)
        
        fw1 = type1['inheritance']['framework_hierarchy']
        fw2 = type2['inheritance']['framework_hierarchy']
        
        if not fw1 and not fw2:
            framework_score = 0.5
        elif not fw1 or not fw2:
            framework_score = 0.1
        else:
            frameworks1 = set(fw1.keys())
            frameworks2 = set(fw2.keys())
            
            if not frameworks1 and not frameworks2:
                framework_score = 0.3
            else:
                framework_overlap = len(frameworks1 & frameworks2) / len(frameworks1 | frameworks2)
                
                common = frameworks1 & frameworks2
                base_scores = []
                for fw in common:
                    bases1 = set(fw1.get(fw, []))
                    bases2 = set(fw2.get(fw, []))
                    if bases1 and bases2:
                        intersection = bases1 & bases2
                        union = bases1 | bases2
                        base_scores.append(len(intersection) / len(union) if union else 0.0)
                
                base_similarity = np.mean(base_scores) if base_scores else 0.0
                framework_score = (0.6 * framework_overlap) + (0.4 * base_similarity)
        
        methods1 = set(type1['inheritance']['abstract_methods'])
        methods2 = set(type2['inheritance']['abstract_methods'])
        
        if not methods1 and not methods2:
            abstract_score = 0.5
        elif not methods1 and not methods2:
            abstract_score = 0.3
        else:
            intersection = methods1 & methods2
            union = methods1 | methods2
            abstract_score = len(intersection) / len(union) if union else 0.0
        
        overrides1 = type1['inheritance']['overridden_methods']
        overrides2 = type2['inheritance']['overridden_methods']
        
        if not overrides1 and not overrides2:
            override_score = 0.4
        else:
            methods1_set = {o['method'] for o in overrides1} if overrides1 else set()
            methods2_set = {o['method'] for o in overrides2} if overrides2 else set()
            
            if not methods1_set and not methods2_set:
                override_score = 0.2
            else:
                intersection = methods1_set & methods2_set
                union = methods1_set | methods2_set
                override_score = len(intersection) / len(union) if union else 0.0
        
        mixin1 = type1['inheritance']['mixin_usage']
        mixin2 = type2['inheritance']['mixin_usage']
        mixin_score = 1.0 if mixin1 == mixin2 else 0.5
        
        same_fw = False
        if fw1 and fw2:
            same_fw = bool(set(fw1.keys()) & set(fw2.keys()))
        
        library_boost = 0.0
        if self._is_library_compatible(type1, type2):
            library_boost = 0.1
        
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        scores = [chain_score, framework_score, abstract_score, override_score, mixin_score]
        similarity = sum(w * s for w, s in zip(weights, scores))
        
        if same_fw:
            similarity = min(1.0, similarity + 0.15)
        
        similarity = min(1.0, similarity + library_boost)
        
        self.cache[key] = similarity
        return similarity

class PatternSimilarityCalculator:
    def __init__(self):
        self.cache = {}
        self.sig_cache = {}
    
    def calculate(self, type1: Dict, type2: Dict) -> float:
        key = f"{type1.get('id', '')}|{type2.get('id', '')}"
        if key in self.cache:
            return self.cache[key]
        
        sig1 = type1['patterns']['structural_signature']
        sig2 = type2['patterns']['structural_signature']
        
        if sig1 == sig2:
            struct_score = 1.0
        elif not sig1 or not sig2:
            struct_score = 0.2
        else:
            parts1 = self._parse_signature(sig1)
            parts2 = self._parse_signature(sig2)
            
            if not parts1 or not parts2:
                struct_score = 0.3
            else:
                common = set(parts1.keys()) & set(parts2.keys())
                if not common:
                    struct_score = 0.1
                else:
                    scores = []
                    for key in common:
                        val1 = parts1[key]
                        val2 = parts2[key]
                        max_val = max(val1, val2)
                        if max_val > 0:
                            scores.append(1.0 - abs(val1 - val2) / max_val)
                    struct_score = np.mean(scores) if scores else 0.1
        
        patterns1 = type1['patterns']['method_patterns']
        patterns2 = type2['patterns']['method_patterns']
        
        if not patterns1 or not patterns2:
            method_score = 0.3
        else:
            all_patterns = set(patterns1.keys()) | set(patterns2.keys())
            if not all_patterns:
                method_score = 0.0
            else:
                pattern_existence = len(set(patterns1.keys()) & set(patterns2.keys())) / len(all_patterns)
                
                common = set(patterns1.keys()) & set(patterns2.keys())
                dist_scores = []
                for pattern in common:
                    count1 = len(patterns1[pattern])
                    count2 = len(patterns2[pattern])
                    max_count = max(count1, count2)
                    if max_count > 0:
                        dist_scores.append(1.0 - abs(count1 - count2) / max_count)
                
                dist_similarity = np.mean(dist_scores) if dist_scores else 0.0
                method_score = (0.6 * pattern_existence) + (0.4 * dist_similarity)
        
        cf1 = type1['patterns']['control_flow_signatures']
        cf2 = type2['patterns']['control_flow_signatures']
        
        if not cf1 or not cf2:
            control_score = 0.2
        else:
            common_methods = set(cf1.keys()) & set(cf2.keys())
            if not common_methods:
                control_score = 0.1
            else:
                scores = []
                for method in common_methods:
                    if cf1[method] == cf2[method]:
                        scores.append(1.0)
                    elif cf1[method] and cf2[method]:
                        set1 = set(cf1[method].split('_'))
                        set2 = set(cf2[method].split('_'))
                        if set1 or set2:
                            intersection = set1 & set2
                            union = set1 | set2
                            scores.append(len(intersection) / len(union) if union else 0.0)
                control_score = np.mean(scores) if scores else 0.1
        
        ex1 = type1['patterns']['exception_signatures']
        ex2 = type2['patterns']['exception_signatures']
        
        if not ex1 and not ex2:
            exception_score = 0.4
        elif not ex1 or not ex2:
            exception_score = 0.1
        else:
            common_methods = set(ex1.keys()) & set(ex2.keys())
            if not common_methods:
                exception_score = 0.05
            else:
                scores = []
                for method in common_methods:
                    set1 = set(ex1[method])
                    set2 = set(ex2[method])
                    if set1 or set2:
                        intersection = set1 & set2
                        union = set1 | set2
                        scores.append(len(intersection) / len(union) if union else 0.0)
                exception_score = np.mean(scores) if scores else 0.1
        
        int1 = type1['patterns'].get('internal_call_graph', {})
        ext1 = type1['patterns'].get('external_call_patterns', {})
        int2 = type2['patterns'].get('internal_call_graph', {})
        ext2 = type2['patterns'].get('external_call_patterns', {})
        
        call_score = self._compare_call_patterns(int1, ext1, int2, ext2)
        
        weights = [0.25, 0.3, 0.2, 0.15, 0.1]
        scores_list = [struct_score, method_score, control_score, exception_score, call_score]
        similarity = sum(w * s for w, s in zip(weights, scores_list))
        
        if sig1 == sig2:
            similarity = min(1.0, similarity + 0.1)
        
        self.cache[key] = similarity
        return similarity
    
    def _parse_signature(self, sig: str) -> Dict[str, int]:
        if sig in self.sig_cache:
            return self.sig_cache[sig]
        
        try:
            parts = {}
            for p in sig.split('_'):
                if '_' in p:
                    key, val = p.split('_', 1)
                    try:
                        parts[key] = int(val)
                    except:
                        pass
            self.sig_cache[sig] = parts
            return parts
        except:
            self.sig_cache[sig] = {}
            return {}
    
    def _compare_call_patterns(self, int1: Dict, ext1: Dict, int2: Dict, ext2: Dict) -> float:
        scores = []
        if int1 and int2:
            int_score = self._compare_call_dicts(int1, int2)
            scores.append(int_score)
        if ext1 and ext2:
            ext_score = self._compare_call_dicts(ext1, ext2)
            scores.append(ext_score)
        return np.mean(scores) if scores else 0.2
    
    def _compare_call_dicts(self, dict1: Dict, dict2: Dict) -> float:
        if not dict1 or not dict2:
            return 0.1
        
        callers1 = set(dict1.keys())
        callers2 = set(dict2.keys())
        
        if not callers1 or not callers2:
            return 0.05
        
        caller_similarity = len(callers1 & callers2) / len(callers1 | callers2)
        
        common = callers1 & callers2
        method_scores = []
        for caller in common:
            methods1 = set(dict1[caller])
            methods2 = set(dict2[caller])
            if methods1 or methods2:
                intersection = methods1 & methods2
                union = methods1 | methods2
                method_scores.append(len(intersection) / len(union) if union else 0.0)
        
        method_similarity = np.mean(method_scores) if method_scores else 0.0
        return (0.5 * caller_similarity) + (0.5 * method_similarity)

@dataclass
class Brain2Candidate:
    type_id: str
    type_data: Dict
    similarity_score: float
    approach: str
    distance: int
    features_used: List[str] = field(default_factory=list)

class MutationEngine:
    def __init__(self, type_data_path: str, config: MutationConfig = None):
        self.config = config or MutationConfig()
        
        with open(type_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.type_data = data['types']
        
        self.semantic_calc = SemanticSimilarityCalculator()
        self.inheritance_calc = InheritanceSimilarityCalculator()
        self.pattern_calc = PatternSimilarityCalculator()
        
        self.similarity_cache = {}
        
        self.name_to_ids = self._build_name_index()
        self.purpose_index = self._build_purpose_index()
        self.framework_index = self._build_framework_index()
        self.pattern_index = self._build_pattern_index()
        
        self._precompute_type_features()
    
    def _precompute_type_features(self):
        self.type_features = {}
        
        for type_id, data in self.type_data.items():
            self.type_features[type_id] = {
                'domain_set': set(data['semantic']['domain_keywords']),
                'capability_set': set(data['semantic']['business_capabilities']),
                'purpose': data['semantic']['primary_purpose'],
                'frameworks': set(data['inheritance']['framework_hierarchy'].keys()),
                'pattern_sig': data['patterns']['structural_signature']
            }
    
    def _build_name_index(self) -> Dict[str, List[str]]:
        index = defaultdict(list)
        for type_id in self.type_data:
            base_name = type_id.split('.')[-1] if '.' in type_id else type_id
            index[base_name].append(type_id)
        return dict(index)
    
    def _build_purpose_index(self) -> Dict[str, List[str]]:
        index = defaultdict(list)
        for type_id, data in self.type_data.items():
            purpose = data['semantic']['primary_purpose']
            index[purpose].append(type_id)
        return dict(index)
    
    def _build_framework_index(self) -> Dict[str, List[str]]:
        index = defaultdict(list)
        for type_id, data in self.type_data.items():
            for framework in data['inheritance']['framework_hierarchy'].keys():
                index[framework].append(type_id)
        return dict(index)
    
    def _build_pattern_index(self) -> Dict[str, List[str]]:
        index = defaultdict(list)
        for type_id, data in self.type_data.items():
            struct_sig = data['patterns']['structural_signature']
            if struct_sig:
                index[struct_sig].append(type_id)
        return dict(index)
    
    def find_mutations(self, target_type_id: str, exclude_self: bool = True) -> List[Brain2Candidate]:
        if target_type_id not in self.type_data:
            return []
        
        target_data = self.type_data[target_type_id]
        target_features = self.type_features[target_type_id]
        
        semantic_candidates = self._find_semantic_candidates_fast(target_type_id, target_data, target_features)
        inheritance_candidates = self._find_inheritance_candidates_fast(target_type_id, target_data, target_features)
        pattern_candidates = self._find_pattern_candidates_fast(target_type_id, target_data, target_features)
        
        all_candidates = []
        
        for cand in semantic_candidates:
            distance = self._score_to_distance(cand['score'])
            all_candidates.append(Brain2Candidate(
                type_id=cand['id'],
                type_data=self.type_data[cand['id']],
                similarity_score=cand['score'],
                approach='semantic',
                distance=distance,
                features_used=cand.get('features', [])
            ))
        
        for cand in inheritance_candidates:
            distance = self._score_to_distance(cand['score'])
            all_candidates.append(Brain2Candidate(
                type_id=cand['id'],
                type_data=self.type_data[cand['id']],
                similarity_score=cand['score'],
                approach='inheritance',
                distance=distance,
                features_used=cand.get('features', [])
            ))
        
        for cand in pattern_candidates:
            distance = self._score_to_distance(cand['score'])
            all_candidates.append(Brain2Candidate(
                type_id=cand['id'],
                type_data=self.type_data[cand['id']],
                similarity_score=cand['score'],
                approach='pattern',
                distance=distance,
                features_used=cand.get('features', [])
            ))
        
        if exclude_self:
            all_candidates = [c for c in all_candidates if c.type_id != target_type_id]
        
        all_candidates.sort(key=lambda x: (x.distance, -x.similarity_score))
        return all_candidates
    
    def _find_semantic_candidates_fast(self, target_id: str, target_data: Dict, target_features: Dict) -> List[Dict]:
        candidates = []
        target_purpose = target_features['purpose']
        same_purpose_ids = self.purpose_index.get(target_purpose, [])
        
        if len(same_purpose_ids) > self.config.max_semantic_candidates * 10:
            import random
            same_purpose_ids = random.sample(same_purpose_ids, self.config.max_semantic_candidates * 10)
        
        target_domains = target_features['domain_set']
        target_caps = target_features['capability_set']
        
        for type_id in same_purpose_ids:
            if type_id == target_id:
                continue
            
            cand_features = self.type_features[type_id]
            
            domain_overlap = len(target_domains & cand_features['domain_set']) > 0
            cap_overlap = len(target_caps & cand_features['capability_set']) > 0
            
            if domain_overlap or cap_overlap or target_purpose == cand_features['purpose']:
                similarity = self.semantic_calc.calculate(target_data, self.type_data[type_id])
                if similarity >= self.config.semantic_threshold:
                    candidates.append({
                        'id': type_id,
                        'score': similarity,
                        'features': self._extract_semantic_features_fast(target_features, cand_features)
                    })
            
            if len(candidates) >= self.config.max_semantic_candidates * 2:
                break
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.config.max_semantic_candidates]
    
    def _find_inheritance_candidates_fast(self, target_id: str, target_data: Dict, target_features: Dict) -> List[Dict]:
        candidates = []
        target_frameworks = target_features['frameworks']
        
        if not target_frameworks:
            return candidates
        
        framework_ids = set()
        for framework in target_frameworks:
            if framework in self.framework_index:
                framework_ids.update(self.framework_index[framework])
        
        if len(framework_ids) > self.config.max_inheritance_candidates * 10:
            import random
            framework_ids = random.sample(list(framework_ids), self.config.max_inheritance_candidates * 10)
        
        for type_id in framework_ids:
            if type_id == target_id:
                continue
            
            cand_features = self.type_features[type_id]
            
            if target_frameworks & cand_features['frameworks']:
                similarity = self.inheritance_calc.calculate(target_data, self.type_data[type_id])
                if similarity >= self.config.inheritance_threshold:
                    candidates.append({
                        'id': type_id,
                        'score': similarity,
                        'features': self._extract_inheritance_features_fast(target_features, cand_features)
                    })
            
            if len(candidates) >= self.config.max_inheritance_candidates * 2:
                break
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.config.max_inheritance_candidates]
    
    def _find_pattern_candidates_fast(self, target_id: str, target_data: Dict, target_features: Dict) -> List[Dict]:
        candidates = []
        target_pattern = target_features['pattern_sig']
        
        if not target_pattern:
            return candidates
        
        same_pattern_ids = self.pattern_index.get(target_pattern, [])
        
        max_checks = min(len(same_pattern_ids), self.config.max_pattern_candidates * 5)
        
        for i, type_id in enumerate(same_pattern_ids[:max_checks]):
            if type_id == target_id:
                continue
            
            similarity = self.pattern_calc.calculate(target_data, self.type_data[type_id])
            if similarity >= self.config.pattern_threshold:
                cand_features = self.type_features[type_id]
                candidates.append({
                    'id': type_id,
                    'score': similarity,
                    'features': self._extract_pattern_features_fast(target_features, cand_features)
                })
            
            if len(candidates) >= self.config.max_pattern_candidates:
                break
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.config.max_pattern_candidates]
    
    def _extract_semantic_features_fast(self, target: Dict, candidate: Dict) -> List[str]:
        features = []
        if target['purpose'] == candidate['purpose']:
            features.append('same_purpose')
        if target['domain_set'] & candidate['domain_set']:
            features.append('domain_overlap')
        if target['capability_set'] & candidate['capability_set']:
            features.append('capability_overlap')
        return features
    
    def _extract_inheritance_features_fast(self, target: Dict, candidate: Dict) -> List[str]:
        features = []
        if target['frameworks'] & candidate['frameworks']:
            features.append('same_framework')
        return features
    
    def _extract_pattern_features_fast(self, target: Dict, candidate: Dict) -> List[str]:
        features = []
        if target['pattern_sig'] == candidate['pattern_sig']:
            features.append('same_structure')
        return features
    
    def _score_to_distance(self, score: float) -> int:
        if math.isclose(score, 1.0, rel_tol=1e-9):
            return 0
        elif score >= 0.5:
            return 1
        elif score >= 0.3:
            return 2
        else:
            return 3
    
    def get_combined_similarity(self, type1_id: str, type2_id: str) -> float:
        cache_key = (type1_id, type2_id)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        type1 = self.type_data[type1_id]
        type2 = self.type_data[type2_id]
        
        semantic_sim = self.semantic_calc.calculate(type1, type2)
        inheritance_sim = self.inheritance_calc.calculate(type1, type2)
        pattern_sim = self.pattern_calc.calculate(type1, type2)
        
        combined = (
            self.config.semantic_weight * semantic_sim +
            self.config.inheritance_weight * inheritance_sim +
            self.config.pattern_weight * pattern_sim
        )
        
        self.similarity_cache[cache_key] = combined
        return combined

class MutationSelector:
    def __init__(self, engine: MutationEngine):
        self.engine = engine
    
    def select_mutation(self, target_type_id: str, use_fallback: bool = True) -> Tuple[Optional[Brain2Candidate], List[Brain2Candidate]]:
        all_candidates = self.engine.find_mutations(target_type_id)
        
        if not all_candidates:
            if use_fallback:
                fallback = self._create_fallback_candidate(target_type_id)
                return fallback, []
            return None, []
        
        distance_groups = {}
        for cand in all_candidates:
            if cand.distance not in distance_groups:
                distance_groups[cand.distance] = []
            distance_groups[cand.distance].append(cand)
        
        for distance in [1, 2, 3]:
            if distance in distance_groups and distance_groups[distance]:
                best_in_group = max(distance_groups[distance], key=lambda x: x.similarity_score)
                adjusted_candidate = self._adjust_similarity_by_distance(best_in_group)
                return adjusted_candidate, all_candidates
        
        if use_fallback:
            pattern_candidates = [c for c in all_candidates if c.distance == 3]
            if pattern_candidates:
                best_pattern = max(pattern_candidates, key=lambda x: x.similarity_score)
                adjusted = self._adjust_similarity_by_distance(best_pattern)
                return adjusted, all_candidates
            
            fallback = self._create_fallback_candidate(target_type_id)
            return fallback, all_candidates
        
        return None, all_candidates
    
    def _adjust_similarity_by_distance(self, candidate: Brain2Candidate) -> Brain2Candidate:
        if candidate.distance == 1:
            adjusted_score = min(1.0, candidate.similarity_score * 1.05)
        elif candidate.distance == 2:
            adjusted_score = candidate.similarity_score
        else:
            adjusted_score = max(
                self.engine.config.min_similarity,
                candidate.similarity_score * 0.95
            )
        
        return Brain2Candidate(
            type_id=candidate.type_id,
            type_data=candidate.type_data,
            similarity_score=adjusted_score,
            approach=candidate.approach,
            distance=candidate.distance,
            features_used=candidate.features_used
        )
    
    def _create_fallback_candidate(self, target_type_id: str) -> Brain2Candidate:
        target_data = self.engine.type_data[target_type_id]
        for type_id, type_data in self.engine.type_data.items():
            if type_id != target_type_id:
                similarity = self.engine.pattern_calc.calculate(target_data, type_data)
                similarity = max(self.engine.config.min_similarity, similarity * 0.5)
                return Brain2Candidate(
                    type_id=type_id,
                    type_data=type_data,
                    similarity_score=similarity,
                    approach='pattern_fallback',
                    distance=3,
                    features_used=['fallback']
                )
        return Brain2Candidate(
            type_id="object",
            type_data={"name": "object", "semantic": {"primary_purpose": "object"}, "inheritance": {}, "patterns": {}},
            similarity_score=0.3,
            approach='hard_fallback',
            distance=3,
            features_used=['hard_fallback']
        )

class HonestNominalWrapper:
    def __init__(self, brain2_engine: MutationEngine):
        self.engine = brain2_engine
        self.selector = MutationSelector(brain2_engine)
        self.candidate_cache = {}
        self.name_variation_cache = {}
    
    def _generate_name_variations(self, name: str) -> List[str]:
        if name in self.name_variation_cache:
            return self.name_variation_cache[name]
        
        variations = [name]
        
        if name.endswith('s'):
            variations.append(name[:-1])
        else:
            variations.append(name + 's')
        
        suffix_map = {
            'Service': ['Svc', 'Manager', 'Handler'],
            'Manager': ['Mgr', 'Service', 'Controller'],
            'Handler': ['Processor', 'Service'],
            'Controller': ['Ctrl', 'Endpoint', 'API'],
            'Processor': ['Proc', 'Handler'],
            'Builder': ['Bldr', 'Factory'],
            'Factory': ['Maker', 'Builder'],
        }
        
        for suffix, alternatives in suffix_map.items():
            if name.endswith(suffix):
                base = name[:-len(suffix)]
                for alt in alternatives:
                    variations.append(base + alt)
                break
        
        import re
        snake = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        variations.append(snake)
        
        if '_' in name:
            camel = ''.join(word.capitalize() for word in name.split('_'))
            variations.append(camel)
        
        self.name_variation_cache[name] = variations
        return variations
    
    def _are_names_similar(self, name1: str, name2: str, threshold: float = 0.7) -> bool:
        if name1 == name2:
            return True
        
        similarity = SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
        
        common_patterns = [
            (name1 + 'Service', name2),
            (name1, name2 + 'Service'),
            (name1.replace('Service', 'Svc'), name2),
            (name1, name2.replace('Service', 'Svc')),
            (name1 + 'Manager', name2),
            (name1, name2 + 'Manager'),
            (name1.replace('Manager', 'Mgr'), name2),
            (name1, name2.replace('Manager', 'Mgr')),
        ]
        
        for pattern1, pattern2 in common_patterns:
            if pattern1 == pattern2:
                return True
        
        return similarity >= threshold
    
    def _find_semantic_match(self, type_name: str) -> Optional[str]:
        simple_name = type_name.split('.')[-1]
        
        for type_id, data in self.engine.type_data.items():
            type_simple = type_id.split('.')[-1]
            
            if self._are_names_similar(simple_name, type_simple, threshold=0.7):
                return type_id
        
        return None
    
    def get_mutation_candidates(self, t_name: str, k: int = 4) -> List[MutationCandidate]:
        if t_name in self.candidate_cache:
            return self.candidate_cache[t_name][:k]
        
        expanded_names = [t_name]
        ALIAS_MAP = {
            "np.": "numpy.",
            "pd.": "pandas.", 
            "plt.": "matplotlib.pyplot.",
            "tf.": "tensorflow.",
            "torch.": "torch.",
            "jax.": "jax.",
            "pg.": "pyqtgraph.",
            "QtGui.": "PyQt5.QtGui.",
            "QtCore.": "PyQt5.QtCore.",
            "QtWidgets.": "PyQt5.QtWidgets.",
            "mx.": "mxnet.",
        }
        
        for alias, full_module in ALIAS_MAP.items():
            if t_name.startswith(alias):
                expanded_names.append(t_name.replace(alias, full_module, 1))
                break
        
        matching_types = []
        target_type = None
        
        for name in expanded_names:
            if name in self.engine.name_to_ids:
                matches = self.engine.name_to_ids[name]
                if matches:
                    matching_types = matches
                    target_type = matches[0]
                    break
        
        if not target_type:
            simple_name = t_name.split('.')[-1]
            
            name_variations = self._generate_name_variations(simple_name)
            for var in name_variations:
                if var in self.engine.name_to_ids:
                    matches = self.engine.name_to_ids[var]
                    if matches:
                        matching_types = matches
                        target_type = matches[0]
                        break
        
        if not target_type:
            target_type = self._find_semantic_match(t_name)
        
        if not target_type:
            self.candidate_cache[t_name] = []
            return []
        
        best_candidate, all_candidates = self.selector.select_mutation(target_type)
        
        brain1_candidates = []
        
        if best_candidate:
            similarity = best_candidate.similarity_score
            
            candidate_name = best_candidate.type_data.get('name', '')
            if candidate_name == t_name:
                best_candidate = None
            elif similarity >= 0.9 and candidate_name.split('.')[-1] == t_name.split('.')[-1]:
                best_candidate = None
            
            if best_candidate:
                if similarity == 1.0:
                    distance = 0
                elif similarity >= 0.5:
                    distance = 1
                elif similarity >= 0.3:
                    distance = 2
                else:
                    distance = 3
                
                brain1_candidates.append(MutationCandidate(
                    type_name=best_candidate.type_data['name'],
                    similarity=similarity,
                    reason=self._generate_reason(best_candidate, distance),
                    metadata={
                        "approach": best_candidate.approach,
                        "method": best_candidate.approach, 
                        "brain2_original_distance": best_candidate.distance,
                        "brain2_calculated_distance": distance,
                        "brain2_type_id": best_candidate.type_id,
                        "features": best_candidate.features_used
                    }
                ))
        
        remaining = k - 1
        if remaining > 0 and all_candidates:
            other_candidates = []
            for c in all_candidates:
                if best_candidate is None:
                    other_candidates.append(c)
                elif c.type_id != best_candidate.type_id:
                    candidate_name = c.type_data.get('name', '')
                    if candidate_name == t_name:
                        continue
                    if c.similarity_score >= 0.9 and candidate_name.split('.')[-1] == t_name.split('.')[-1]:
                        continue
                    other_candidates.append(c)
                    
            other_candidates.sort(key=lambda x: x.similarity_score, reverse=True)
            
            for cand in other_candidates[:remaining]:
                similarity = cand.similarity_score
                if similarity == 1.0:
                    distance = 0
                elif similarity >= 0.5:
                    distance = 1
                elif similarity >= 0.3:
                    distance = 2
                else:
                    distance = 3
                
                brain1_candidates.append(MutationCandidate(
                    type_name=cand.type_data['name'],
                    similarity=similarity,
                    reason=self._generate_reason(cand, distance),
                    metadata={
                        "approach": cand.approach,
                        "brain2_original_distance": cand.distance,
                        "brain2_calculated_distance": distance,
                        "brain2_type_id": cand.type_id,
                        "features": cand.features_used
                    }
                ))
        
        self.candidate_cache[t_name] = brain1_candidates
        return brain1_candidates[:k]
    
    def _generate_reason(self, candidate: Brain2Candidate, distance: int) -> str:
        if distance == 0:
            return f"Exact match via {candidate.approach} (score: {candidate.similarity_score:.3f})"
        elif distance == 1:
            return f"High similarity via {candidate.approach} (score: {candidate.similarity_score:.3f})"
        elif distance == 2:
            return f"Medium similarity via {candidate.approach} (score: {candidate.similarity_score:.3f})"
        elif distance == 3:
            return f"Low similarity via {candidate.approach} (score: {candidate.similarity_score:.3f})"
        else:
            return f"{candidate.approach} match (score: {candidate.similarity_score:.3f})"

class HonestSemanticWrapper:
    def __init__(self, brain2_engine: MutationEngine):
        self.engine = brain2_engine
        self.semantic_calc = SemanticSimilarityCalculator()
        self.inheritance_calc = InheritanceSimilarityCalculator()
        self.pattern_calc = PatternSimilarityCalculator()
        self.similarity_cache = {}
    
    def _normalize_optional(self, type_str: str) -> Tuple[str, bool]:
        """Normalize Optional[T] to Union[T, None] and extract inner type"""
        t = type_str.strip()
        
        # Handle Optional[X]
        if t.startswith("Optional[") and t.endswith("]"):
            inner = t[9:-1].strip()
            return inner, True
        
        # Handle Union[T, None]
        if t.startswith("Union[") and t.endswith("]"):
            inner = t[6:-1].strip()
            if inner.endswith(", None") or inner.startswith("None, "):
                inner = inner.replace(", None", "").replace("None, ", "").strip()
                return inner, True
        
        # Handle T | None
        if "|" in t:
            parts = [p.strip() for p in t.split("|")]
            non_none = [p for p in parts if p.lower() not in ["none", "nonetype"]]
            if len(non_none) < len(parts):  # Had a None
                return " | ".join(non_none), True
        
        return t, False
    
    def _adjust_for_common_patterns(self, a: str, b: str, original_score: float) -> float:
        a_simple = a.split('.')[-1].lower()
        b_simple = b.split('.')[-1].lower()
        
        # Same name, different case
        if a_simple == b_simple and a != b:
            return max(original_score, 0.9)
        
        # Check for common suffix patterns 
        common_suffixes = ['service', 'manager', 'handler', 'controller', 'processor', 'builder', 'factory']
        
        for suffix in common_suffixes:
            if a_simple.endswith(suffix) and b_simple.endswith(suffix):
                a_base = a_simple[:-len(suffix)]
                b_base = b_simple[:-len(suffix)]
                
                if a_base == b_base:
                    return max(original_score, 0.8)
                elif a_base in b_base or b_base in a_base:
                    return max(original_score, 0.7)
        
        # Check for verb/noun patterns
        verb_noun = {
            'summarize': 'summary',
            'process': 'processor', 
            'manage': 'manager',
            'handle': 'handler',
            'create': 'creator',
            'build': 'builder',
            'stream': 'streaming'
        }
        
        for verb, noun in verb_noun.items():
            if (a_simple.startswith(verb) and b_simple.startswith(noun)) or \
               (a_simple.startswith(noun) and b_simple.startswith(verb)):
                return max(original_score, 0.7)
        
        return original_score
    
    def _parse_union(self, union_str: str) -> List[str]:
        union_str = union_str.strip()
        if union_str.startswith('Union['):
            inner = union_str[6:-1]
            return [t.strip() for t in inner.split(',')]
        elif '|' in union_str:
            return [t.strip() for t in union_str.split('|')]
        else:
            return [union_str.strip()]
    
    def _compare_unions(self, a: str, b: str) -> Tuple[float, str]:
        # First normalize Optional
        a_norm, a_opt = self._normalize_optional(a)
        b_norm, b_opt = self._normalize_optional(b)
        
        # If same after normalization
        if a_norm == b_norm:
            if a_opt == b_opt:
                return 1.0, "identical unions"
            else:
                return 0.8, "same inner types, different optionality"
        
        comps_a = self._parse_union(a_norm)
        comps_b = self._parse_union(b_norm)
        
        if not comps_a or not comps_b:
            return 0.1, "empty union"
        
        best_matches = []
        used_b = set()
        
        for comp_a in comps_a:
            best_sim = 0.0
            best_idx = -1
            
            for j, comp_b in enumerate(comps_b):
                if j in used_b:
                    continue
                
                sim_result = self.compute_similarity(comp_a, comp_b)
                sim = sim_result.score
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            
            if best_idx != -1:
                used_b.add(best_idx)
                best_matches.append(best_sim)
        
        if not best_matches:
            return 0.1, "no matches"
        
        avg_sim = sum(best_matches) / max(len(comps_a), len(comps_b))
        
        if avg_sim > 0.6:
            return min(1.0, avg_sim * 1.1), "good union match"
        elif avg_sim > 0.4:
            return avg_sim, "partial union match"
        else:
            return avg_sim, "poor union match"
    
    def compute_similarity(self, a: str, b: str) -> SimilarityResult:
        key = (a, b)
        if key in self.similarity_cache:
            return self.similarity_cache[key]
        
        # Handle Optional normalization
        a_norm, a_optional = self._normalize_optional(a)
        b_norm, b_optional = self._normalize_optional(b)
        
        # If both have same optional status and inner types match
        if a_norm == b_norm:
            if a_optional == b_optional:
                # Both Optional or both not Optional
                distance = 0
                score = 1.0
                explanation = "Same type" + (" (both optional)" if a_optional else "")
            else:
                # One Optional, one not
                distance = 1
                score = 0.8  # Optional vs non-Optional is close
                explanation = f"Same inner type, different optionality: {a} vs {b}"
            
            result = SimilarityResult(
                score=score,
                distance=distance,
                explanation=explanation,
                kind="optional_normalized",
                metadata={"optional_normalized": True, "a_optional": a_optional, "b_optional": b_optional}
            )
            self.similarity_cache[key] = result
            return result
        
        # Continue with normalized types
        a_clean = a_norm.strip().strip("'")
        b_clean = b_norm.strip().strip("'")
        
        if a_clean == b_clean:
            # Same inner type, check optionality
            optional_penalty = 0.0
            if a_optional != b_optional:
                optional_penalty = 0.2  # Penalty for different optionality
            
            result = SimilarityResult(
                score=1.0 - optional_penalty,
                distance=0 if optional_penalty == 0 else 1,
                explanation=f"Same inner type" + (", different optionality" if a_optional != b_optional else ""),
                kind="inner_match",
                metadata={"inner_match": True, "a_original": a, "b_original": b}
            )
            self.similarity_cache[key] = result
            return result
        
        if '|' in a_clean or '|' in b_clean or 'Union[' in a_clean or 'Union[' in b_clean:
            union_sim, union_expl = self._compare_unions(a_clean, b_clean)
            
            if union_sim == 1.0:
                distance = 0
            elif union_sim >= 0.5:
                distance = 1
            elif union_sim >= 0.3:
                distance = 2
            else:
                distance = 3
            
            result = SimilarityResult(
                score=union_sim,
                distance=distance,
                explanation=f"Union comparison: {union_expl}",
                kind="union",
                metadata={"union_comparison": True}
            )
            self.similarity_cache[key] = result
            return result
        
        a_simple = a_clean.split(".")[-1] if "." in a_clean else a_clean
        b_simple = b_clean.split(".")[-1] if "." in b_clean else b_clean
        
        if a_simple == b_simple and a_clean != b_clean:
            # Same simple name, different modules
            distance = 1
            score = 0.9
            
            result = SimilarityResult(
                score=score,
                distance=distance,
                explanation=f"Same simple name with different modules: {a_clean} vs {b_clean}",
                kind="module_variant",
                metadata={"simple_name_match": True}
            )
            self.similarity_cache[key] = result
            return result
        
        a_matches = self.engine.name_to_ids.get(a_simple, [])
        b_matches = self.engine.name_to_ids.get(b_simple, [])
        
        if not a_matches or not b_matches:
            # Types not in our dataset - use string similarity
            name_similarity = SequenceMatcher(None, a_simple.lower(), b_simple.lower()).ratio()
            adjusted_score = self._adjust_for_common_patterns(a_clean, b_clean, max(0.3, name_similarity * 0.8))
            
            if adjusted_score == 1.0:
                distance = 0
            elif adjusted_score >= 0.5:
                distance = 1
            elif adjusted_score >= 0.3:
                distance = 2
            else:
                distance = 3
            
            result = SimilarityResult(
                score=adjusted_score,
                distance=distance,
                explanation=f"Type(s) not found in analysis: {a_clean} / {b_clean}",
                kind="unknown",
                metadata={"in_brain2_data": False, "name_similarity": name_similarity}
            )
            self.similarity_cache[key] = result
            return result
        
        a_type = a_matches[0]
        b_type = b_matches[0]
        
        type_a = self.engine.type_data[a_type]
        type_b = self.engine.type_data[b_type]
        
        semantic_score = self.semantic_calc.calculate(type_a, type_b)
        inheritance_score = self.inheritance_calc.calculate(type_a, type_b)
        pattern_score = self.pattern_calc.calculate(type_a, type_b)
        
        combined_score = (
            self.engine.config.semantic_weight * semantic_score +
            self.engine.config.inheritance_weight * inheritance_score +
            self.engine.config.pattern_weight * pattern_score
        )
        
        # Apply pattern-based adjustments
        combined_score = self._adjust_for_common_patterns(a_clean, b_clean, combined_score)
        
        raw_scores = [semantic_score, inheritance_score, pattern_score]
        max_raw_score = max(raw_scores)
        
        if max_raw_score == semantic_score:
            dominant_footprint = "semantic"
        elif max_raw_score == inheritance_score:
            dominant_footprint = "inheritance"
        else:
            dominant_footprint = "pattern"
        
        if combined_score == 1.0:
            distance = 0
        elif combined_score >= 0.5:
            distance = 1
        elif combined_score >= 0.3:
            distance = 2
        else:
            distance = 3
        
        explanation = f"Combined similarity via {dominant_footprint} (score: {combined_score:.3f}, distance: {distance})"
        
        result = SimilarityResult(
            score=combined_score,
            distance=distance,
            explanation=explanation,
            kind=dominant_footprint,
            metadata={
                "semantic_score": semantic_score,
                "inheritance_score": inheritance_score,
                "pattern_score": pattern_score,
                "combined_score": combined_score,
                "dominant_footprint": dominant_footprint,
                "a_brain2_id": a_type,
                "b_brain2_id": b_type
            }
        )
        
        self.similarity_cache[key] = result
        return result

def get_brain2_engine(path: str, config: MutationConfig = None) -> MutationEngine:
    global _BRAIN2_ENGINE
    if _BRAIN2_ENGINE is None:
        _BRAIN2_ENGINE = MutationEngine(path, config)
    return _BRAIN2_ENGINE

def initialize_honest_type_system(
    path: str = None,
    hier: Any = None,
    base: Dict = None,
    repo_paths: List[str] = None,
    debug: bool = False
) -> Tuple[HonestNominalWrapper, HonestSemanticWrapper, Dict, Dict, Dict]:
    brain2_engine = get_brain2_engine(path, MutationConfig())
    nominal_wrapper = HonestNominalWrapper(brain2_engine)
    semantic_wrapper = HonestSemanticWrapper(brain2_engine)
    
    return (
        nominal_wrapper,
        semantic_wrapper,
        base or {},
        {},
        {}
    )

def generate_mutation_candidates(
    t: str,
    nominal: HonestNominalWrapper,
    semantic: HonestSemanticWrapper,
    base: Dict,
    conversions: Dict,
    operations: Dict
) -> List[MutationCandidate]:
    return nominal.get_mutation_candidates(t, k=4)

def compute_honest_similarity(
    a: str,
    b: str,
    nominal: HonestNominalWrapper,
    semantic: HonestSemanticWrapper,
    conversions: Dict,
    operations: Dict
) -> SimilarityResult:
    return semantic.compute_similarity(a, b)