from typing import Any, List, Dict, Optional, Tuple
import re
import random
# from data_service import IntentResult
# from llm_factory import BaseLLM, LLMFactory

class LLMCypherQueryGenerator:
    """
    Hybrid Cypher Generator (V10.5 - Undirected / Direction-Agnostic Edition):
    - Key Change: All relationship arrows (->, <-) removed from templates.
      This solves the "Directionality Mismatch" issue where the bot uses directed paths
      but the DB or GT expects undirected/reverse paths.
    - Retains V10.0's Count/List hard-filtering and Negation support.
    """
    
    DETAILED_SCHEMA = """
    [Node Labels & Properties (Case-Sensitive)]
    - Gene: {id, name, species}
    - Compound: {id, name, names, formula}
    - Reaction: {id, name, definition, equation}
    - Pathway: {id, title, species}
    - EC: {id, name, sysname}
    - Ortholog: {id, name, symbol}
    - FunctionalUnit: {id, name}

    [Relationships]
    - (:Gene)-[:ENCODES]->(:EC)
    - (:Gene)-[:BELONGS_TO]->(:Ortholog)
    - (:Gene)-[:MEMBER_OF]->(:FunctionalUnit)
    - (:Compound)-[:SUBSTRATE_OF]->(:Reaction)
    - (:Reaction)-[:PRODUCES]->(:Compound)
    - (:EC)-[:CATALYZES]->(:Reaction)
    - (:Ortholog)-[:CATALYZES]->(:Reaction)
    - (:Ortholog)-[:HAS_ENZYME_FUNCTION]->(:EC)
    - (:Pathway)-[:CONTAINS]->(:Reaction)
    - (:Pathway)-[:CONTAINS]->(:FunctionalUnit)
    """

    CYPHER_TEMPLATES = {
        # ==============================================================================
        # [PHASE 1] Direct Entity & Property Lookup (T001-T010)
        # ==============================================================================
        "T001": { "description": "Find gene node", "cypher": "MATCH (n:Gene {id: '{GENE_ID}'}) RETURN n" },
        "T002": { "description": "Find pathway node", "cypher": "MATCH (n:Pathway {id: '{PATHWAY_ID}'}) RETURN n" },
        "T003": { "description": "Find compound node", "cypher": "MATCH (n:Compound {id: '{COMPOUND_ID}'}) RETURN n" },
        "T004": { "description": "Find enzyme node", "cypher": "MATCH (n:EC {id: '{EC_ID}'}) RETURN n" },
        "T005": { "description": "Find reaction node", "cypher": "MATCH (n:Reaction {id: '{REACTION_ID}'}) RETURN n" },
        
        "T006": { "description": "Get gene properties", "cypher": "MATCH (n:Gene {id: '{GENE_ID}'}) RETURN properties(n)" },
        "T007": { "description": "Get pathway properties", "cypher": "MATCH (n:Pathway {id: '{PATHWAY_ID}'}) RETURN properties(n)" },
        "T008": { "description": "Get compound properties", "cypher": "MATCH (n:Compound {id: '{COMPOUND_ID}'}) RETURN properties(n)" },
        "T009": { "description": "Get enzyme properties", "cypher": "MATCH (n:EC {id: '{EC_ID}'}) RETURN properties(n)" },
        "T010": { "description": "Get reaction properties", "cypher": "MATCH (n:Reaction {id: '{REACTION_ID}'}) RETURN properties(n)" },

        # ==============================================================================
        # [PHASE 1 Extended] 1-Hop Relationships (Undirected Patch Applied)
        # ==============================================================================
        # All arrows removed: -[:REL]-> becomes -[:REL]-
        "T011": { "description": "Find enzymes by Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]-(e:EC) RETURN e" },
        "T012": { "description": "Find Ortholog by Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:BELONGS_TO]-(o:Ortholog) RETURN o" },
        "T013": { "description": "Find Functional Units by Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:MEMBER_OF]-(f:FunctionalUnit) RETURN f" },
        
        "T014": { "description": "Reactions using Compound", "cypher": "MATCH (c:Compound {id: '{COMPOUND_ID}'})-[:SUBSTRATE_OF]-(r:Reaction) RETURN r" },
        "T015": { "description": "Reactions producing Compound", "cypher": "MATCH (r:Reaction)-[:PRODUCES]-(c:Compound {id: '{COMPOUND_ID}'}) RETURN r" },
        
        "T016": { "description": "Reactions by Enzyme", "cypher": "MATCH (e:EC {id: '{EC_ID}'})-[:CATALYZES]-(r:Reaction) RETURN r" },
        "T017": { "description": "Reactions by Ortholog", "cypher": "MATCH (o:Ortholog {id: '{ORTHOLOG_ID}'})-[:CATALYZES]-(r:Reaction) RETURN r" },
        "T018": { "description": "Enzymes of Ortholog", "cypher": "MATCH (o:Ortholog {id: '{ORTHOLOG_ID}'})-[:HAS_ENZYME_FUNCTION]-(e:EC) RETURN e" },
        
        "T019": { "description": "Genes encoding Enzyme", "cypher": "MATCH (g:Gene)-[:ENCODES]-(e:EC {id: '{EC_ID}'}) RETURN g" },
        "T020": { "description": "Genes of Ortholog", "cypher": "MATCH (g:Gene)-[:BELONGS_TO]-(o:Ortholog {id: '{ORTHOLOG_ID}'}) RETURN g" },
        
        "T021": { "description": "Substrates of Reaction", "cypher": "MATCH (c:Compound)-[:SUBSTRATE_OF]-(r:Reaction {id: '{REACTION_ID}'}) RETURN c" },
        "T022": { "description": "Products of Reaction", "cypher": "MATCH (r:Reaction {id: '{REACTION_ID}'})-[:PRODUCES]-(c:Compound) RETURN c" },
        "T023": { "description": "Enzymes of Reaction", "cypher": "MATCH (e:EC)-[:CATALYZES]-(r:Reaction {id: '{REACTION_ID}'}) RETURN e" },

        "T024": { "description": "Reactions in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]-(r:Reaction) RETURN r" },
        "T025": { "description": "Functional Units in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]-(f:FunctionalUnit) RETURN f" },

        # ==============================================================================
        # [PHASE 2] Multi-Hop Relationships (Undirected Patch Applied)
        # ==============================================================================
        "T026": { "description": "Pathways containing Reaction", "cypher": "MATCH (p:Pathway)-[:CONTAINS]-(r:Reaction {id: '{REACTION_ID}'}) RETURN p" },
        "T027": { "description": "Reactions via Ortholog", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:BELONGS_TO]-(o:Ortholog)-[:CATALYZES]-(r:Reaction) RETURN r" },
        "T028": { "description": "Reactions via Enzyme", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]-(e:EC)-[:CATALYZES]-(r:Reaction) RETURN r" },
        "T029": { "description": "Compounds via Enzyme", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]-(e:EC)-[:CATALYZES]-(r:Reaction)-[:PRODUCES]-(c:Compound) RETURN c" },
        "T030": { "description": "Next Step Compounds", "cypher": "MATCH (c1:Compound {id: '{COMPOUND_ID}'})-[:SUBSTRATE_OF]-(r:Reaction)-[:PRODUCES]-(c2:Compound) RETURN c2" },
        
        "T031": { "description": "Previous Step Compounds", "cypher": "MATCH (c1:Compound)-[:SUBSTRATE_OF]-(r:Reaction)-[:PRODUCES]-(c2:Compound {id: '{COMPOUND_ID}'}) RETURN c1" },
        "T032": { "description": "Downstream Reactions", "cypher": "MATCH (r1:Reaction {id: '{REACTION_ID}'})-[:PRODUCES]-(c:Compound)-[:SUBSTRATE_OF]-(r2:Reaction) RETURN r2" },
        "T033": { "description": "Compounds produced in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]-(r:Reaction)-[:PRODUCES]-(c:Compound) RETURN DISTINCT c" },
        "T034": { "description": "Compounds consumed in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]-(r:Reaction)-[:SUBSTRATE_OF]-(c:Compound) RETURN DISTINCT c" },
        "T035": { "description": "Enzymes in Functional Unit", "cypher": "MATCH (f:FunctionalUnit {id: '{FUNCTIONALUNIT_ID}'})-[:MEMBER_OF]-(n)-[:ENCODES|HAS_ENZYME_FUNCTION]-(e:EC) RETURN DISTINCT e" },

        # ==============================================================================
        # [PHASE 3] Stats & Aggregations (Undirected Patch Applied)
        # ==============================================================================
        "T036": { "description": "Count enzymes of Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]-(e:EC) RETURN count(e)" },
        "T037": { "description": "Count reactions in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]-(r:Reaction) RETURN count(r)" },
        "T038": { "description": "Count genes of Ortholog", "cypher": "MATCH (o:Ortholog {id: '{ORTHOLOG_ID}'})-[:BELONGS_TO]-(g:Gene) RETURN count(g)" },
        "T039": { "description": "Top Pathways by size", "cypher": "MATCH (p:Pathway)-[:CONTAINS]-(r:Reaction) RETURN p.id, count(r) AS cnt ORDER BY cnt DESC LIMIT 10" }, 
        "T040": { "description": "Shared Reactions", "cypher": "MATCH (p1:Pathway {id: '{PATHWAY_ID_1}'})-[:CONTAINS]-(r:Reaction)-[:CONTAINS]-(p2:Pathway {id: '{PATHWAY_ID_2}'}) RETURN r" },

        # Global Counts
        "T041": { "description": "Count all genes", "cypher": "MATCH (n:Gene) RETURN count(n)" },
        "T042": { "description": "Count all pathways", "cypher": "MATCH (n:Pathway) RETURN count(n)" },
        "T043": { "description": "Count all compounds", "cypher": "MATCH (n:Compound) RETURN count(n)" },
        "T044": { "description": "Count all enzymes", "cypher": "MATCH (n:EC) RETURN count(n)" },
        "T045": { "description": "Count all reactions", "cypher": "MATCH (n:Reaction) RETURN count(n)" },

        # ==============================================================================
        # [PHASE 4] Global Lists (T046-T051)
        # ==============================================================================
        "T046": { "description": "List all genes", "cypher": "MATCH (n:Gene) RETURN n LIMIT 100" },
        "T047": { "description": "List all pathways", "cypher": "MATCH (n:Pathway) RETURN n LIMIT 100" },
        "T048": { "description": "List all compounds", "cypher": "MATCH (n:Compound) RETURN n LIMIT 100" },
        "T049": { "description": "List all enzymes", "cypher": "MATCH (n:EC) RETURN n LIMIT 100" },
        "T050": { "description": "List all reactions", "cypher": "MATCH (n:Reaction) RETURN n LIMIT 100" },
        "T051": { "description": "Show Orthologs (Example)", "cypher": "MATCH (n:Ortholog) RETURN n LIMIT 100" },

        # ==============================================================================
        # [NEW] Edge Cases & Negations (Undirected Patch Applied where safe)
        # ==============================================================================
        # Note: Negation checks if connection exists in ANY direction to determine "isolation"
        "T052": { "description": "Find reactions with NO products", "cypher": "MATCH (r:Reaction) WHERE NOT (r)-[:PRODUCES]-() RETURN r" },
        "T053": { "description": "Find reactions with NO substrates", "cypher": "MATCH (r:Reaction) WHERE NOT (r)-[:SUBSTRATE_OF]-() RETURN r" },
        "T054": { "description": "Find orphan pathways (empty)", "cypher": "MATCH (p:Pathway) WHERE NOT (p)-[:CONTAINS]-() RETURN p" },
        "T055": { "description": "Find enzymes not catalyzing any reaction", "cypher": "MATCH (e:EC) WHERE NOT (e)-[:CATALYZES]-() RETURN e" }
    }

    def __init__(self, llm: Optional[Any] = None, provider: str = "gemini", model_name: Optional[str] = None, api_key: Optional[str] = None, host: Optional[str] = None, temperature: float = 0.0, schema: Optional[str] = None):
        if llm: self.llm = llm
        else:
             pass # Initialize your LLMFactory here
        self.schema = schema or self.DETAILED_SCHEMA

    def generate_query(self, question: str, intent: Optional[Any] = None, entities: List[Dict] = []) -> Tuple[str, str]:
        template_id = self._select_template(question)
        if template_id and template_id in self.CYPHER_TEMPLATES:
            filled_cypher = self._fill_template_smart(template_id, question, entities)
            if filled_cypher: return filled_cypher, "Template"
        return self._generate_raw_query(question, intent, entities), "Fallback"

    def _select_template(self, question: str) -> Optional[str]:
        if not hasattr(self, 'CYPHER_TEMPLATES'): return None
        
        q_lower = question.lower()
        
        # 1. Hard Filter: COUNT vs LIST
        is_count_query = any(k in q_lower for k in ["count", "how many", "number of"])
        
        # 2. Hard Filter: NEGATION
        is_negation = any(k in q_lower for k in ["no ", "not ", "without", "empty", "orphan"])

        candidate_ids = []
        
        if is_negation:
             candidate_ids = ["T052", "T053", "T054", "T055"]
        elif is_count_query:
            # Only allow Count templates
            candidate_ids = [k for k, v in self.CYPHER_TEMPLATES.items() if "count" in v['description'].lower()]
        else:
            # Exclude Count & Negation templates
            candidate_ids = [k for k, v in self.CYPHER_TEMPLATES.items() 
                             if "count" not in v['description'].lower() 
                             and k not in ["T052", "T053", "T054", "T055"]]

        candidate_summaries = [f"{tid}: {self.CYPHER_TEMPLATES[tid]['description']}" for tid in candidate_ids]
        templates_str = "\n".join(candidate_summaries)
        
        # 3. Prompt: Target Awareness
        prompt = f"""
        Act as a Smart Query Router.
        
        [User Question] "{question}"
        
        [Candidate Templates]
        {templates_str}
        
        [INSTRUCTIONS]
        1. **Identify the TARGET ENTITY**: What object does the user want? (Compound? Reaction? Enzyme?)
        2. **Match with Template**: Select the one that RETURNS that entity.
        3. Example: "What compounds does gene X produce?" -> MUST pick template returning 'Compound', NOT 'Enzyme'.
        4. Return ONLY the Template ID (e.g., T029).
        """
        try:
            response = self.llm.generate(prompt).strip()
            match = re.search(r'T\d{3}', response)
            found_id = match.group(0) if match else None
            return found_id if found_id in self.CYPHER_TEMPLATES else None
        except: return None

    def _fill_template_smart(self, template_id: str, question: str, entities: List[Dict]) -> Optional[str]:
        template_data = self.CYPHER_TEMPLATES[template_id]
        raw_cypher = template_data["cypher"]
        
        if "{" not in raw_cypher: return raw_cypher

        # Regex Filling First
        regex_filled = self._fill_template_regex(raw_cypher, entities, question)
        if regex_filled and "{" not in regex_filled:
            return regex_filled

        # LLM Filling Fallback
        try:
            prompt = f"""Task: Fill placeholders in Cypher.
            Template: {raw_cypher}
            Question: "{question}"
            Entities: {entities}
            Instruction: Replace {{ID}} with extracted value. Return ONLY the query."""
            
            response = self.llm.generate(prompt).strip()
            cleaned = self._clean_query(response)
            cleaned = self._auto_correct_ids_in_query(cleaned)
            if cleaned and not re.search(r'\{[A-Z_0-9]+\}', cleaned): return cleaned
        except: pass
        
        return regex_filled

    def _fill_template_regex(self, cypher: str, entities: List[Dict], question: str) -> Optional[str]:
        entity_map = {}
        for e in entities:
            e_id = e.get("id")
            e_type = e.get("type", "").upper()
            if e_id:
                if e_type == 'EC' and not e_id.startswith('EC:'): e_id = f"EC:{e_id}"
                elif e_type == 'PATHWAY' and not e_id.startswith('path:'): e_id = f"path:{e_id}"
                entity_map.setdefault(e_type, []).append(e_id)

        patterns = {
            "GENE": r'([a-z]{2,4}:[A-Z0-9_]+)', 
            "REACTION": r'(R\d{5})', 
            "COMPOUND": r'(C\d{5})',
            "EC": r'(\d+\.\d+\.\d+\.\d+|EC:?[\d\.]+)', 
            "PATHWAY": r'(path:[a-z]+\d+|[a-z]{2,3}\d{5})', 
            "ORTHOLOG": r'(K\d{5}|[A-Z]\d{5})'
        }
        for k, pat in patterns.items():
            found = re.findall(pat, question)
            if found:
                corrected = []
                for val in found:
                    if k == 'EC' and not val.startswith('EC:'): val = f"EC:{val}"
                    elif k == 'PATHWAY' and not val.startswith('path:'): 
                        if not val.startswith('path'): val = f"path:{val}"
                    corrected.append(val)
                entity_map.setdefault(k, []).extend(corrected)

        filled_cypher = cypher
        placeholders = re.findall(r'\{([A-Z_0-9]+)\}', cypher)
        for ph in placeholders:
            type_key = ph.split('_')[0]
            if type_key in entity_map and entity_map[type_key]:
                val = entity_map[type_key].pop(0)
                filled_cypher = filled_cypher.replace(f"{{{ph}}}", val)
                
        if re.search(r'\{[A-Z_0-9]+\}', filled_cypher): return None
        return filled_cypher

    def _auto_correct_ids_in_query(self, query: str) -> str:
        ec_pattern = r"['\"](\d+\.\d+\.\d+\.\d+)['\"]"
        def add_prefix(match): return f"'EC:{match.group(1)}'"
        if ":EC" in query or ":Enzyme" in query:
            query = re.sub(ec_pattern, add_prefix, query)
        return query

    def _generate_raw_query(self, question, intent, entities) -> str:
        corrected_entities = []
        for e in entities:
            e_type = e.get('type', '').upper()
            e_id = e.get('id', '')
            if e_type == 'EC' and not e_id.startswith('EC:'): e_id = f"EC:{e_id}"
            elif e_type == 'PATHWAY' and not e_id.startswith('path:'): e_id = f"path:{e_id}"
            corrected_entities.append(f"{e.get('type')}: {e_id}")
        entities_str = ", ".join(corrected_entities)

        prompt = f"""
        Act as a precise Neo4j Cypher Developer. Generate a query based on the Strict Schema.

        [STRICT SCHEMA]
        {self.schema}

        [STYLE GUIDE]
        1. Simple Lookup/Count -> Use alias 'n'.
        2. Relations -> Use meaningful aliases (g, p, r, c, e, o).
        3. LIST queries -> MUST use `LIMIT 100`.
        4. COUNT queries -> MUST use `RETURN count(...)`.
        5. Negation -> Use `WHERE NOT EXISTS` or `WHERE NOT (...)`.
        6. **IMPORTANT**: Always use UNDIRECTED relationships (e.g., -[:REL]-) to find all connections.

        [CURRENT TASK]
        Question: "{question}"
        Entities: {entities_str}

        Output ONLY the Cypher query.
        """
        
        generated = self.llm.generate(prompt)
        return self._clean_query(generated)

    def _clean_query(self, cypher_query: str) -> str:
        if not cypher_query: return ""
        q = re.sub(r'```(?:cypher)?', '', cypher_query, flags=re.IGNORECASE).strip()
        q = q.replace("`", "")
        q = q.strip()
        return q