from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, List, Tuple
from requests_cache import Dict, Optional


class LLMProvider(Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OLLAMA = "ollama"


class IntentType(str, Enum):
    # 
    FIND_PROTEIN = "find_protein"
    FIND_SUBSTRATES = "find_substrates"
    FIND_PRODUCTS = "find_products"
    FIND_ENZYME = "find_enzyme"
    FIND_REACTION = "find_reaction"
    FIND_PATHWAY = "find_pathway"
    FIND_GENE = "find_gene"
    FIND_COMPOUND = "find_compound"
    FIND_FUNCTION = "find_function"
    
    # 
    FIND_CONNECTION = "find_connection"
    FIND_PATH = "find_path"
    FIND_NEIGHBORS = "find_neighbors"
    
    # 
    COMPARE = "compare"
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    LIST_ENTITIES = "list_entities"
    
    UNKNOWN = "unknown"



@dataclass
class EntityMention:
    """Represents a biological entity mention in text"""
    text: str
    start: int
    end: int


@dataclass
class IntentResult:
    """Represents the result of intent recognition"""
    intent_type: IntentType
    confidence: float
    source: str
    constraints: Optional[Dict[str, Any]] = None
    matched_patterns: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        result = {
            "type": self.intent_type.value,
            "confidence": self.confidence,
            "source": self.source
        }
        
        # Only include if not None
        if self.constraints is not None:
            result["constraints"] = self.constraints
        if self.matched_patterns is not None:
            result["matched_patterns"] = self.matched_patterns
        
        return result


@dataclass
class IntentPattern:
    intent: IntentType
    patterns: List[Tuple[str, float]]  # (regex_pattern, confidence_score)
    keywords: List[str] = field(default_factory=list)
    base_weight: float = 1.0
    
    def match(self, text: str) -> Optional[Tuple[re.Match, float]]:
        best_match = None
        best_confidence = 0.0
        
        for pattern, confidence in self.patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                adjusted_confidence = confidence * self.base_weight
                if adjusted_confidence > best_confidence:
                    best_match = match
                    best_confidence = adjusted_confidence
        
        return (best_match, best_confidence) if best_match else None
