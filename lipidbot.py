import json
from pydantic import BaseModel

class SimpleClassification(BaseModel):
    is_relevant: bool
    needs_graph: bool
    reasoning: str
    confidence: float

SIMPLE_CLASSIFICATION_PROMPT = """You are analyzing queries for a lipid biochemistry knowledge graph database.

**OUR KNOWLEDGE GRAPH CONTAINS:**

**Nodes:** Gene, Compound, Reaction, Pathway, EC (enzymes), Ortholog, FunctionalUnit

**Relationships (what we can answer):**
- Gene→EC (what enzyme does gene X encode?)
- Compound→Reaction (what reactions use/produce compound X?)
- EC→Reaction (what reactions does enzyme X catalyze?)
- Pathway→Reaction (what reactions are in pathway X?)
- Gene→Ortholog→EC (gene families and their functions)

**USER QUERY:** "{query}"

**ANSWER TWO QUESTIONS:**

1. **Is this relevant to lipid biology, fatty acids, metabolic pathways, or biochemistry?**
   - YES: lipids, fatty acids, genes, enzymes, metabolic reactions, biochemical pathways
   - NO: weather, sports, politics, general knowledge unrelated to biology

2. **Does answering this require our knowledge graph?**
   - YES if asking about:
     * Relationships between entities ("What reactions produce X?", "What enzyme does gene Y encode?")
     * Pathway structure ("What reactions are in pathway X?")
     * Entity connections ("What compounds are substrates of enzyme Y?")
   - NO if asking about:
     * Mechanisms ("How does X work?", "Why does X happen?")
     * Health effects ("What are benefits of X?")
     * General properties that need literature ("What foods contain X?")
     * Single entity properties ("What is the formula of X?" - but may still use graph)

**EXAMPLES:**

Query: "What reactions produce linoleic acid?"
→ is_relevant: true, needs_graph: true (requires Compound→Reaction traversal)

Query: "How does DHA reduce inflammation?"
→ is_relevant: true, needs_graph: false (mechanism, not structure - needs literature)

Query: "What enzyme does FADS2 encode?"
→ is_relevant: true, needs_graph: true (requires Gene→EC relationship)

Query: "What are health benefits of omega-3?"
→ is_relevant: true, needs_graph: false (clinical effects, not pathway structure)

Query: "What's the weather in Paris?"
→ is_relevant: false, needs_graph: false (not biology)

Query: "What pathways contain COX-2?"
→ is_relevant: true, needs_graph: true (requires Pathway→Reaction→EC traversal)

Query: "What is the chemical formula of EPA?"
→ is_relevant: true, needs_graph: false (property lookup, literature is better)

Respond in JSON:
{{
    "is_relevant": true/false,
    "needs_graph": true/false,
    "reasoning": "Brief explanation of both decisions",
    "confidence": 0.0-1.0
}}
"""

# ============================================================================
# CLASSIFICATION FUNCTION
# ============================================================================

async def classify_query_simple(
    query: str,
    llm
) -> SimpleClassification:
    """
    Simple two-question classification:
    1. Is it relevant to our domain?
    2. Does it need graph traversal?
    """
    prompt = SIMPLE_CLASSIFICATION_PROMPT.format(query=query)
    
    response = await llm.agenerate(prompt=prompt)
    
    try:
        # Clean response (remove markdown code blocks if present)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        
        parsed = json.loads(cleaned.strip())
        return SimpleClassification(**parsed)
        
    except (json.JSONDecodeError, ValueError) as e:
        
        return SimpleClassification(
            is_relevant=True,
            needs_graph=True,
            reasoning="Parse failure, defaulting to safe mode",
            confidence=0.5
        )
