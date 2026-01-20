import json
import re
from typing import List
from requests_cache import Dict, Optional
from data_service import EntityMention
from llm_factory import BaseLLM, LLMFactory

"""Extract biological entity mentions using LLM"""
class LLMBioEntityExtractor:
    
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
        Initialize the entity extractor
        
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
    
    def extract(self, question: str) -> List[EntityMention]:

        if not question or not question.strip():
            return []
        
        ENTITY_EXTRACTION_PROMPT = """Extract all biological/chemical entity mentions from this question.
            Entities include: genes, proteins, compounds, enzymes, reactions, pathways, orthologs.

            Question: {question}

            Return ONLY a valid JSON object with this exact format:
            {{
            "mentions": [
                {{"text": "entity name", "start": character_index, "end": character_index}},
                ...
            ]
            }}

            Rules:
            - Include ONLY entity names (genes, proteins, compounds, enzymes, pathways, reactions)
            - Provide exact character positions (0-indexed, where start is inclusive, end is exclusive)
            - Don't overlap entities
            - Order by appearance in the question
            - Return empty array if no entities found

            Example:
            Question: "What is the role of TP53 in apoptosis?"
            Output: {{"mentions": [{{"text": "TP53", "start": 20, "end": 24}}, {{"text": "apoptosis", "start": 28, "end": 37}}]}}

            Now extract from the question above. Return ONLY the JSON, no other text:"""
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)
        
        try:
            response_text = self.llm.generate(prompt)
            mentions_data = self._parse_json_response(response_text)
            return self._validate_and_convert_mentions(mentions_data, question)
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []
    
    def _parse_json_response(self, response_text: str) -> Dict:

        response_text = response_text.strip()
        
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        
        # Find JSON object in response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return json.loads(response_text)
    
    def _validate_and_convert_mentions(
        self, 
        mentions_data: Dict,
        original_text: str
    ) -> List[EntityMention]:

        if "mentions" not in mentions_data:
            return []
        
        validated_mentions = []
        
        for m in mentions_data["mentions"]:
            if not all(k in m for k in ["text", "start", "end"]):
                continue
            
            # Validate indices
            if not (0 <= m["start"] < m["end"] <= len(original_text)):
                continue
            
            mention = EntityMention(
                text=m["text"],
                start=m["start"],
                end=m["end"]
            )
            validated_mentions.append(mention)
        
        return validated_mentions
