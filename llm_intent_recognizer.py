import json
import re
from typing import List
from requests_cache import Optional
from data_service import EntityMention, IntentResult, IntentType
from llm_factory import BaseLLM, LLMFactory


class LLMIntentRecognizer:
    """Recognize user intent using LLM"""
    
    INTENT_DESCRIPTIONS = {
        IntentType.FIND_PROTEIN: {
            "name": "find_protein",
            "description": "User wants to find proteins encoded by genes or gene products",
        },
        IntentType.FIND_SUBSTRATES: {
            "name": "find_substrates",
            "description": "User wants to find substrates (inputs/reactants) of an enzyme or reaction",
        },
        IntentType.FIND_PRODUCTS: {
            "name": "find_products",
            "description": "User wants to find products (outputs) of an enzyme or reaction",
        },
        IntentType.FIND_ENZYME: {
            "name": "find_enzyme",
            "description": "User wants to find enzymes that catalyze a reaction",
        },
        IntentType.FIND_REACTION: {
            "name": "find_reaction",
            "description": "User wants to find reactions involving compounds or catalyzed by enzymes",
        },
        IntentType.FIND_PATHWAY: {
            "name": "find_pathway",
            "description": "User wants to find metabolic pathways containing genes, enzymes, or reactions",
        },
        IntentType.FIND_GENE: {
            "name": "find_gene",
            "description": "User wants to find genes encoding specific enzymes or proteins",
        },
        IntentType.FIND_COMPOUND: {
            "name": "find_compound",
            "description": "User wants to find compounds/metabolites involved in reactions or pathways",
        },
        IntentType.FIND_FUNCTION: {
            "name": "find_function",
            "description": "User wants to understand the function, role, or activity of an entity",
        },
        IntentType.FIND_CONNECTION: {
            "name": "find_connection",
            "description": "User wants to understand how two entities are related or connected",
        },
        IntentType.FIND_PATH: {
            "name": "find_path",
            "description": "User wants to find a specific path or route between two entities",
        },
        IntentType.FIND_NEIGHBORS: {
            "name": "find_neighbors",
            "description": "User wants to find entities related to or associated with a given entity",
        },
        IntentType.COMPARE: {
            "name": "compare",
            "description": "User wants to compare two or more entities, pathways, or conditions",
        },
        IntentType.ANALYZE: {
            "name": "analyze",
            "description": "User wants statistical analysis, distribution, or data aggregation",
        },
        IntentType.SUMMARIZE: {
            "name": "summarize",
            "description": "User wants a summary, overview, or explanation of an entity",
        },
        IntentType.LIST_ENTITIES: {
            "name": "list_entities",
            "description": "User wants to list, count, or enumerate entities of a certain type",
        },
    }
    
    def __init__(
        self, 
        llm: Optional[BaseLLM] = None,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Initialize the intent recognizer
        
        Args:
            llm: Pre-configured LLM instance (optional)
            provider: LLM provider ('gemini' or 'ollama')
            model_name: Model name
            api_key: API key (for Gemini)
            host: Host URL (for Ollama)
            temperature: Sampling temperature
        """
        if llm:
            self.llm = llm
        else:
            self.llm = LLMFactory.create_llm(
                provider=provider,
                model_name=model_name,
                api_key=api_key,
                host=host,
                temperature=temperature
            )
    
    def recognize(
        self, 
        question: str, 
        entities: Optional[List[EntityMention]] = None
    ) -> IntentResult:

        prompt = self._build_prompt(question, entities or [])
        # print(prompt)
        try:
            response_text = self.llm.generate(prompt)
            # print(response_text)
            return self._parse_response(response_text)
            
        except Exception as e:
            print(f"Intent recognition error: {e}")
            return IntentResult(
                intent_type=IntentType.UNKNOWN,
                confidence=0.2,
                source="llm_error",
                constraints={}
            )
    
    def _build_prompt(
        self, 
        question: str, 
        entities: List[EntityMention]
    ) -> str:
        """Build the prompt for intent classification"""
        
        # Format intent definitions
        intent_defs = [
            f"  - **{info['name']}**: {info['description']}"
            for info in self.INTENT_DESCRIPTIONS.values()
        ]
        intent_definitions = "\n".join(intent_defs)
        
        # Format entity information
        # The 'entities' list is shown here as a dictionary, not an object.
        # entities = [{'text': 'EC:6.4.1.2', 'id': '6.4.1.2', 'db': 'ec', 'species': '-', 'start': 15, 'end': 25, 'src': 'regex-id', 'confidence': 1.0}]
        if entities:
            entity_lines = [
                f"  {i}. \"{e.get('text', 'N/A')}\" (Type: {e.get('db', 'Unknown')}, ID: {e.get('id', 'N/A')})"
                for i, e in enumerate(entities, 1)
            ]
            entity_info = "Detected Entities:\n" + "\n".join(entity_lines)
        else:
            entity_info = "No entities detected in the question."
        
        prompt = f"""You are an expert in biological knowledge graph query intent classification.

            Your task is to classify the user's question into ONE of the following intent types:

            {intent_definitions}

            {entity_info}

            User Question: "{question}"

            Instructions:
            1. Carefully analyze the question and its linguistic structure
            2. Consider the detected entities and their types (Gene, Protein, Enzyme, Reaction, Compound, Pathway)
            3. Determine the MOST APPROPRIATE intent type
            4. Provide a confidence score between 0.0 and 1.0

            Respond in VALID JSON format only (no additional text, no markdown):
            {{
            "intent": "intent_name_here",
            "confidence": 0.85
            "constraints": {{
                "species": "species_code_if_mentioned",
                "limit": number_if_mentioned,
                "order": "ASC or DESC if sorting mentioned"
                }}
            }}

            Important:
            - If unsure, use "unknown" as the intent with confidence < 0.5
            - The intent field must exactly match one of the defined intent names
            - Confidence should reflect how certain you are about the classification

            JSON Response:"""
        return prompt
    
    def _parse_response(self, response_text: str) -> IntentResult:
        """Parse LLM's JSON response into IntentResult"""
        
        try:
            response_text = response_text.strip()
            
            # Remove markdown code blocks
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            # Find JSON object
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in response")
            
            data = json.loads(match.group(0))
            
            # Extract and validate fields
            intent_name = data.get("intent", "unknown")
            intent_type = self._map_intent_name(intent_name)
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            constraints = data.get("constraints", {})
            
            return IntentResult(
                intent_type=intent_type,
                confidence=confidence,
                source=self.llm.get_provider_name(),
                constraints=constraints
            )
            
        except Exception as e:
            print(f"Intent parsing error: {e}")
            return IntentResult(
                intent_type=IntentType.UNKNOWN,
                confidence=0.2,
                source="parse_error",
                constraints={}
            )
    
    def _map_intent_name(self, intent_name: str) -> IntentType:
        """Map LLM intent name to IntentType enum"""
        
        intent_name = (intent_name or "").lower().strip().replace("-", "_")
        
        # Direct mapping
        for intent_type in IntentType:
            if intent_type.value == intent_name:
                return intent_type
        
        # Fuzzy matching for common variations
        fuzzy_mapping = {
            "protein": IntentType.FIND_PROTEIN,
            "substrate": IntentType.FIND_SUBSTRATES,
            "product": IntentType.FIND_PRODUCTS,
            "enzyme": IntentType.FIND_ENZYME,
            "reaction": IntentType.FIND_REACTION,
            "pathway": IntentType.FIND_PATHWAY,
            "gene": IntentType.FIND_GENE,
            "compound": IntentType.FIND_COMPOUND,
            "function": IntentType.FIND_FUNCTION,
            "connection": IntentType.FIND_CONNECTION,
            "path": IntentType.FIND_PATH,
            "neighbor": IntentType.FIND_NEIGHBORS,
            "compare": IntentType.COMPARE,
            "analyze": IntentType.ANALYZE,
            "summarize": IntentType.SUMMARIZE,
            "list": IntentType.LIST_ENTITIES,
        }
        
        for key, intent_type in fuzzy_mapping.items():
            if key in intent_name:
                return intent_type
        
        return IntentType.UNKNOWN
