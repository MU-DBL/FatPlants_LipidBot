from typing import Any, List, Dict, Optional, Tuple
import re

class Config:
    DEFAULT_LIMIT = 20
    MAX_RESULTS = 20
    LARGE_PATHWAY_THRESHOLD = 10
    MULTI_SUBSTRATE_THRESHOLD = 2


class SimpleCypherGenerator:
    
    DETAILED_SCHEMA = """
        [Node Labels & Properties]
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
        # --- [T001-T010] Direct Entity Lookup ---
        "T001": {"description": "Find gene node", "cypher": "MATCH (n:Gene {id: '{GENE_ID}'}) RETURN n"},
        "T002": {"description": "Find pathway node", "cypher": "MATCH (n:Pathway {id: '{PATHWAY_ID}'}) RETURN n"},
        "T003": {"description": "Find compound node", "cypher": "MATCH (n:Compound {id: '{COMPOUND_ID}'}) RETURN n"},
        "T004": {"description": "Find enzyme node", "cypher": "MATCH (n:EC {id: '{EC_ID}'}) RETURN n"},
        "T005": {"description": "Find reaction node", "cypher": "MATCH (n:Reaction {id: '{REACTION_ID}'}) RETURN n"},
        "T006": {"description": "Get gene properties", "cypher": "MATCH (n:Gene {id: '{GENE_ID}'}) RETURN properties(n)"},
        "T007": {"description": "Get pathway properties", "cypher": "MATCH (n:Pathway {id: '{PATHWAY_ID}'}) RETURN properties(n)"},
        "T008": {"description": "Get compound properties", "cypher": "MATCH (n:Compound {id: '{COMPOUND_ID}'}) RETURN properties(n)"},
        "T009": {"description": "Get enzyme properties", "cypher": "MATCH (n:EC {id: '{EC_ID}'}) RETURN properties(n)"},
        "T010": {"description": "Get reaction properties", "cypher": "MATCH (n:Reaction {id: '{REACTION_ID}'}) RETURN properties(n)"},

        # --- [T011-T025] 1-Hop Relationships ---
        "T011": {"description": "Find enzymes by Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]->(e:EC) RETURN e"},
        "T012": {"description": "Find Ortholog by Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:BELONGS_TO]->(o:Ortholog) RETURN o"},
        "T013": {"description": "Find Functional Units by Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:MEMBER_OF]->(f:FunctionalUnit) RETURN f"},
        "T014": {"description": "Reactions using Compound", "cypher": "MATCH (c:Compound {id: '{COMPOUND_ID}'})-[:SUBSTRATE_OF]->(r:Reaction) RETURN r"},
        "T015": {"description": "Reactions producing Compound", "cypher": "MATCH (r:Reaction)-[:PRODUCES]->(c:Compound {id: '{COMPOUND_ID}'}) RETURN r"},
        "T016": {"description": "Reactions by Enzyme", "cypher": "MATCH (e:EC {id: '{EC_ID}'})-[:CATALYZES]->(r:Reaction) RETURN r"},
        "T017": {"description": "Reactions by Ortholog", "cypher": "MATCH (o:Ortholog {id: '{ORTHOLOG_ID}'})-[:CATALYZES]->(r:Reaction) RETURN r"},
        "T018": {"description": "Enzymes of Ortholog", "cypher": "MATCH (o:Ortholog {id: '{ORTHOLOG_ID}'})-[:HAS_ENZYME_FUNCTION]->(e:EC) RETURN e"},
        "T019": {"description": "Genes encoding Enzyme", "cypher": "MATCH (g:Gene)-[:ENCODES]->(e:EC {id: '{EC_ID}'}) RETURN g"},
        "T020": {"description": "Genes of Ortholog", "cypher": "MATCH (g:Gene)-[:BELONGS_TO]->(o:Ortholog {id: '{ORTHOLOG_ID}'}) RETURN g"},
        "T021": {"description": "Substrates of Reaction", "cypher": "MATCH (c:Compound)-[:SUBSTRATE_OF]->(r:Reaction {id: '{REACTION_ID}'}) RETURN c"},
        "T022": {"description": "Products of Reaction", "cypher": "MATCH (r:Reaction {id: '{REACTION_ID}'})-[:PRODUCES]->(c:Compound) RETURN c"},
        "T023": {"description": "Enzymes of Reaction", "cypher": "MATCH (e:EC)-[:CATALYZES]->(r:Reaction {id: '{REACTION_ID}'}) RETURN e"},
        "T024": {"description": "Reactions in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]->(r:Reaction) RETURN r"},
        "T025": {"description": "Functional Units in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]->(f:FunctionalUnit) RETURN f"},

        # --- [T026-T035] Multi-Hop Relationships ---
        "T026": {"description": "Pathways containing Reaction", "cypher": "MATCH (p:Pathway)-[:CONTAINS]->(r:Reaction {id: '{REACTION_ID}'}) RETURN p"},
        "T027": {"description": "Reactions via Ortholog", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:BELONGS_TO]->(o:Ortholog)-[:CATALYZES]->(r:Reaction) RETURN r"},
        "T028": {"description": "Reactions via Enzyme", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]->(e:EC)-[:CATALYZES]->(r:Reaction) RETURN r"},
        "T029": {"description": "Compounds via Enzyme", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]->(e:EC)-[:CATALYZES]->(r:Reaction)-[:PRODUCES]->(c:Compound) RETURN DISTINCT c"},
        "T030": {"description": "Next Step Compounds", "cypher": "MATCH (c1:Compound {id: '{COMPOUND_ID}'})-[:SUBSTRATE_OF]->(r:Reaction)-[:PRODUCES]->(c2:Compound) RETURN c2"},
        "T031": {"description": "Previous Step Compounds", "cypher": "MATCH (c1:Compound)-[:SUBSTRATE_OF]->(r:Reaction)-[:PRODUCES]->(c2:Compound {id: '{COMPOUND_ID}'}) RETURN c1"},
        "T032": {"description": "Downstream Reactions", "cypher": "MATCH (r1:Reaction {id: '{REACTION_ID}'})-[:PRODUCES]->(c:Compound)-[:SUBSTRATE_OF]->(r2:Reaction) RETURN r2"},
        "T033": {"description": "Compounds produced in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]->(r:Reaction)-[:PRODUCES]->(c:Compound) RETURN DISTINCT c"},
        "T034": {"description": "Compounds consumed in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]->(r:Reaction), (c:Compound)-[:SUBSTRATE_OF]->(r) RETURN DISTINCT c"},
        "T035": {"description": "Enzymes in Functional Unit", "cypher": "MATCH (f:FunctionalUnit {id: '{FUNCTIONALUNIT_ID}'})<-[:MEMBER_OF]-(g:Gene)-[:ENCODES]->(e:EC) RETURN DISTINCT e UNION MATCH (f:FunctionalUnit {id: '{FUNCTIONALUNIT_ID}'})<-[:MEMBER_OF]-(g:Gene)-[:BELONGS_TO]->(o:Ortholog)-[:HAS_ENZYME_FUNCTION]->(e:EC) RETURN DISTINCT e"},
        
        # --- [T036-T045] Stats & Counts ---
        "T036": {"description": "Count enzymes of Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[r:ENCODES]->(e:EC) RETURN count(r)"},
        "T037": {"description": "Count reactions in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[r:CONTAINS]->(rxn:Reaction) RETURN count(r)"},
        "T038": {"description": "Count genes of Ortholog", "cypher": "MATCH (g:Gene)-[r:BELONGS_TO]->(o:Ortholog {id: '{ORTHOLOG_ID}'}) RETURN count(r)"},
        "T039": {"description": "Top Pathways by size", "cypher": "MATCH (p:Pathway)-[r:CONTAINS]->(rxn:Reaction) RETURN p.id, count(r) AS cnt ORDER BY cnt DESC LIMIT 10"},
        "T040": {"description": "Shared Reactions between two Pathways", "cypher": "MATCH (p1:Pathway {id: '{PATHWAY_ID_1}'})-[:CONTAINS]->(r:Reaction)<-[:CONTAINS]-(p2:Pathway {id: '{PATHWAY_ID_2}'}) RETURN r"},
        "T041": {"description": "Count all genes", "cypher": "MATCH (n:Gene) RETURN count(n)"},
        "T042": {"description": "Count all pathways", "cypher": "MATCH (n:Pathway) RETURN count(n)"},
        "T043": {"description": "Count all compounds", "cypher": "MATCH (n:Compound) RETURN count(n)"},
        "T044": {"description": "Count all enzymes", "cypher": "MATCH (n:EC) RETURN count(n)"},
        "T045": {"description": "Count all reactions", "cypher": "MATCH (n:Reaction) RETURN count(n)"},

        # --- [T046-T051] Global Lists ---
        "T046": {"description": "List all genes", "cypher": f"MATCH (n:Gene) RETURN n LIMIT {Config.DEFAULT_LIMIT}"},
        "T047": {"description": "List all pathways", "cypher": f"MATCH (n:Pathway) RETURN n LIMIT {Config.DEFAULT_LIMIT}"},
        "T048": {"description": "List all compounds", "cypher": f"MATCH (n:Compound) RETURN n LIMIT {Config.DEFAULT_LIMIT}"},
        "T049": {"description": "List all enzymes", "cypher": f"MATCH (n:EC) RETURN n LIMIT {Config.DEFAULT_LIMIT}"},
        "T050": {"description": "List all reactions", "cypher": f"MATCH (n:Reaction) RETURN n LIMIT {Config.DEFAULT_LIMIT}"},
        "T051": {"description": "List all orthologs", "cypher": f"MATCH (n:Ortholog) RETURN n LIMIT {Config.DEFAULT_LIMIT}"},

        # --- [T052-T055] Edge Cases ---
        "T052": {"description": "Find reactions with NO products", "cypher": "MATCH (r:Reaction) WHERE NOT (r)-[:PRODUCES]->() RETURN r"},
        "T053": {"description": "Find reactions with NO substrates", "cypher": "MATCH (r:Reaction) WHERE NOT ()-[:SUBSTRATE_OF]->(r) RETURN r"},
        "T054": {"description": "Find orphan pathways (empty)", "cypher": "MATCH (p:Pathway) WHERE NOT (p)-[:CONTAINS]->() RETURN p"},
        "T055": {"description": "Find enzymes not catalyzing any reaction", "cypher": "MATCH (e:EC) WHERE NOT (e)-[:CATALYZES]->() RETURN e"},

        # --- [T056-T059] Gap Fillers ---
        "T056": {"description": "Pathways involving Gene (via Enzyme)", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]->(e:EC)-[:CATALYZES]->(r:Reaction)<-[:CONTAINS]-(p:Pathway) RETURN DISTINCT p"},
        "T057": {"description": "Pathways involving Gene (via Ortholog)", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:BELONGS_TO]->(o:Ortholog)-[:CATALYZES]->(r:Reaction)<-[:CONTAINS]-(p:Pathway) RETURN DISTINCT p"},
        "T058": {"description": "Compounds produced by Gene (via Enzyme)", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]->(e:EC)-[:CATALYZES]->(r:Reaction)-[:PRODUCES]->(c:Compound) RETURN DISTINCT c"},
        "T059": {"description": "Genes producing Compound", "cypher": "MATCH (g:Gene)-[:ENCODES]->(e:EC)-[:CATALYZES]->(r:Reaction)-[:PRODUCES]->(c:Compound {id: '{COMPOUND_ID}'}) RETURN DISTINCT g"},
        "T057b": {"description": "Compounds produced by Gene (via Ortholog)", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:BELONGS_TO]->(o:Ortholog)-[:CATALYZES]->(r:Reaction)-[:PRODUCES]->(c:Compound) RETURN DISTINCT c"},

        # --- [T060-T069] Advanced Filters ---
        "T060": {"description": "Pathways with more than N reactions", "cypher": f"MATCH (p:Pathway)-[r:CONTAINS]->(rxn:Reaction) WITH p, count(r) as cnt WHERE cnt > {Config.LARGE_PATHWAY_THRESHOLD} RETURN p"},
        "T061": {"description": "Genes starting with prefix", "cypher": f"MATCH (n:Gene) WHERE n.id STARTS WITH '{{PREFIX}}' RETURN n LIMIT {Config.MAX_RESULTS}"},
        "T062": {"description": "Pathways starting with prefix", "cypher": f"MATCH (n:Pathway) WHERE n.id STARTS WITH '{{PREFIX}}' RETURN n LIMIT {Config.MAX_RESULTS}"},
        "T063": {"description": "Compounds starting with prefix", "cypher": f"MATCH (n:Compound) WHERE n.id STARTS WITH '{{PREFIX}}' RETURN n LIMIT {Config.MAX_RESULTS}"},
        "T064": {"description": "Enzymes starting with prefix", "cypher": f"MATCH (n:EC) WHERE n.id STARTS WITH '{{PREFIX}}' RETURN n LIMIT {Config.MAX_RESULTS}"},
        "T065": {"description": "Reactions starting with prefix", "cypher": f"MATCH (n:Reaction) WHERE n.id STARTS WITH '{{PREFIX}}' RETURN n LIMIT {Config.MAX_RESULTS}"},
        "T066": {"description": "Orthologs starting with prefix", "cypher": f"MATCH (n:Ortholog) WHERE n.id STARTS WITH '{{PREFIX}}' RETURN n LIMIT {Config.MAX_RESULTS}"},
        "T067": {"description": "Functional Units starting with prefix", "cypher": f"MATCH (n:FunctionalUnit) WHERE n.id STARTS WITH '{{PREFIX}}' RETURN n LIMIT {Config.MAX_RESULTS}"},
        "T068": {"description": "Reactions with more than 2 Substrates", "cypher": f"MATCH (c:Compound)-[rel:SUBSTRATE_OF]->(r:Reaction) WITH r, count(rel) as input_cnt WHERE input_cnt > {Config.MULTI_SUBSTRATE_THRESHOLD} RETURN r"},
        "T069": {"description": "Reactions with more than 2 Products", "cypher": f"MATCH (r:Reaction)-[rel:PRODUCES]->(c:Compound) WITH r, count(rel) as output_cnt WHERE output_cnt > {Config.MULTI_SUBSTRATE_THRESHOLD} RETURN r"},

        # --- [T070-T076] Pathfinding & Complex ---
        "T070": {"description": "Shortest path between two Compounds", "cypher": "MATCH p=shortestPath((c1:Compound {id: '{COMPOUND_ID_1}'})-[*]-(c2:Compound {id: '{COMPOUND_ID_2}'})) RETURN p"},
        "T071": {"description": "Shortest path Gene to Pathway", "cypher": "MATCH p=shortestPath((g:Gene {id: '{GENE_ID}'})-[*]-(path:Pathway {id: '{PATHWAY_ID}'})) RETURN p"},
        "T072": {"description": "Shortest path between two Reactions", "cypher": "MATCH p=shortestPath((r1:Reaction {id: '{REACTION_ID_1}'})-[*]-(r2:Reaction {id: '{REACTION_ID_2}'})) RETURN p"},
        "T073": {"description": "Shortest path between two Genes", "cypher": "MATCH p=shortestPath((g1:Gene {id: '{GENE_ID_1}'})-[*]-(g2:Gene {id: '{GENE_ID_2}'})) RETURN p"},
        "T074": {"description": "Shortest path between two Pathways", "cypher": "MATCH p=shortestPath((p1:Pathway {id: '{PATHWAY_ID_1}'})-[*]-(p2:Pathway {id: '{PATHWAY_ID_2}'})) RETURN p"},
        "T075": {"description": "Other genes encoding same enzyme (Siblings)", "cypher": "MATCH (g1:Gene {id: '{GENE_ID}'})-[:ENCODES]->(e:EC)<-[:ENCODES]-(g2:Gene) WHERE g1 <> g2 RETURN g2"},
        "T076": {"description": "Inter-pathway Metabolite Exchange", "cypher": "MATCH (p1:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]->(:Reaction)-[:PRODUCES]->(c:Compound)<-[:SUBSTRATE_OF]-(:Reaction)<-[:CONTAINS]-(p2:Pathway) WHERE p1 <> p2 RETURN DISTINCT c"}
    }

    def __init__(self, llm: Any, schema: Optional[str] = None):
        self.llm = llm
        self.schema = schema or self.DETAILED_SCHEMA

    def generate_query(self, question: str) -> Tuple[str, Dict]:
        try:
            # Step 1: Build template catalog for LLM
            template_catalog = self._build_template_catalog()
            
            # Step 2: Ask LLM to select and fill template
            prompt = self._build_prompt(question, template_catalog)
            
            # Step 3: Get LLM response
            response = self.llm.generate(prompt)
            
            # Step 4: Extract and clean query
            cypher_query, template_id  = self._extract_query(response)
            
            # Step 5: Post-process (fix prefixes)
            cypher_query = self._fix_prefixes(cypher_query)
            
            metadata = {
                "success": True,
                "template_id": template_id,
                "raw_response": response[:200]  # First 200 chars for debugging
            }
            
            return cypher_query, metadata
            
        except Exception as e:
            return "", {"success": False, "error": str(e)}

    def _build_template_catalog(self) -> str:
        """Build a formatted catalog of all templates"""
        catalog_lines = []
        for tid, data in self.CYPHER_TEMPLATES.items():
            catalog_lines.append(f"{tid}: {data['description']}")
            catalog_lines.append(f"   Template: {data['cypher']}")
        return "\n".join(catalog_lines)

    def _build_prompt(self, question: str, template_catalog: str) -> str:
        """Build the LLM prompt"""
        return f"""You are a Neo4j Cypher query generator. Your task is to:
1. First, select the most appropriate response template based on the user's question
2. If a suitable template exists, fill in the placeholders with the correct values
3. Always enforce a LIMIT clause on the query
4. If NO suitable template exists, generate a custom Cypher query based on the schema

[SCHEMA]
{self.schema}

[AVAILABLE TEMPLATES]
{template_catalog}

[QUESTION]
{question}

[INSTRUCTIONS]

STEP 1: TEMPLATE SELECTION
Analyze the question to identify:
- Entity types mentioned (Gene, Compound, Pathway, etc.)
- Entity IDs (e.g., eco:b0001, C00001, R00001, path:eco00010)
- Relationships involved (ENCODES, PARTICIPATES_IN, CATALYZES, etc.)
- What the user wants to find (enzymes, reactions, pathways, etc.)
- **Whether the question contains "all", or or similar words**
- **Number of results requested** (e.g., "first 5", "top 10", "3 genes", "all", etc.)

Review the available templates and determine if any match the query pattern.

STEP 2: QUERY GENERATION
Option A - If a suitable template is found:
Fill in ALL placeholders in the template:
- {{{{GENE_ID}}}} should be filled with gene IDs like "eco:b0001"
- {{{{COMPOUND_ID}}}} should be filled with compound IDs like "C00001"
- {{{{REACTION_ID}}}} should be filled with reaction IDs like "R00001"
- {{{{PATHWAY_ID}}}} should be filled with pathway IDs like "path:eco00010"
- {{{{EC_ID}}}} should be filled with EC numbers like "EC:1.1.1.1"
- {{{{ORTHOLOG_ID}}}} should be filled with ortholog IDs like "K00001"
- {{{{FUNCTIONALUNIT_ID}}}} should be filled with functional unit IDs like "M00001"
- {{{{PREFIX}}}} should be filled with the prefix mentioned in "starts with" queries

**TEMPLATE MODIFICATION - ENFORCING LIMIT:**
**CRITICAL: Always enforce a LIMIT clause on every query.**

**TEMPLATE MODIFICATION - LIMITING RESULTS:**
- **If the question contains "all", "every", "complete", or "entire":**
  * Do NOT modify the template
  * Keep the template exactly as-is after filling placeholders
  * Examples: "all reactions", "every gene", "complete list" → No LIMIT modification

- **If the question specifies a specific number of results to return (e.g., "first 5 genes", "top 10 pathways", "3 reactions"):**
  * Add a LIMIT clause at the end of the query
  * Extract the number from the question
  * Examples:
    - "Find the first 5 genes" → Add "LIMIT 5"
    - "Show me 10 pathways" → Add "LIMIT 10"
    - "Get top 3 reactions" → Add "LIMIT 3"

- **If the question contains "all", "every", or doesn't specify a number:**
  * Add "LIMIT 10" as the default
  * Examples:
    - "Find all reactions" → Add "LIMIT 10"
    - "Show me genes" → Add "LIMIT 10"

Option B - If NO suitable template exists:
Generate a custom Cypher query using the schema provided. Follow these guidelines:
- Use the correct node labels from the schema (Gene, Compound, Pathway, Reaction, EC, Ortholog, FunctionalUnit)
- Use the correct relationship types from the schema
- Ensure proper query structure (MATCH, WHERE, RETURN clauses)
- Include appropriate filters and conditions
- Use single quotes around string values
- Ensure all entity IDs have correct prefixes
- **Always enforce a LIMIT clause:**
  * Use the user-specified number if provided
  * If no number is specified, add LIMIT 10

IMPORTANT RULES:
- Ensure all entity IDs have correct prefixes (EC:, path:, etc.)
- Use single quotes around ID values
- Return the complete, ready-to-execute Cypher query
- Prioritize templates when available, but don't force a template if it doesn't fit
- **Always enforce a LIMIT clause**
- **Use the user-specified number if provided**
- **If no number is specified, add LIMIT 10**


OUTPUT FORMAT (REQUIRED):
If using a template:
Line 1: TEMPLATE: <template_id>
Line 2: <the filled Cypher query>

If generating a custom query:
Line 1: TEMPLATE: CUSTOM
Line 2: <the generated Cypher query>

Examples:
Using template with user-specified number:
TEMPLATE: T011
MATCH (g:Gene {{id: 'eco:b0001'}})-[:ENCODES]->(e:EC) RETURN e LIMIT 5

Using template without specified number (default LIMIT 10):
TEMPLATE: T011
MATCH (g:Gene {{id: 'eco:b0001'}})-[:ENCODES]->(e:EC) RETURN e LIMIT 10

Using template with "all" (default LIMIT 10):
TEMPLATE: T015
MATCH (p:Pathway)-[:CONTAINS]->(r:Reaction) RETURN p, r LIMIT 10

Custom query with specified number:
TEMPLATE: CUSTOM
MATCH (g:Gene)-[:ENCODES]->(e:EC)-[:CATALYZES]->(r:Reaction) RETURN g, e, r LIMIT 7

Custom query without specified number (default LIMIT 10):
TEMPLATE: CUSTOM
MATCH (r:Reaction)-[:PRODUCES]->(c:Compound {{name: 'glucose'}}) RETURN r LIMIT 10
"""

    def _extract_query(self, response: str) -> Tuple[str, Optional[str]]:
        """
        Extract Cypher query and template ID from LLM response.
        
        Returns:
            Tuple of (cypher_query, template_id)
        """
        template_id = None
        
        # Extract template ID from first line if present
        lines = response.strip().split('\n')
        if lines and lines[0].strip().upper().startswith('TEMPLATE:'):
            template_id = lines[0].split(':', 1)[1].strip()
            # Remove the template line
            response = '\n'.join(lines[1:])
        
        # Remove markdown code blocks
        response = re.sub(r'```\w*\s*', '', response)
        response = response.replace('```', '')
        
        # Remove common prefixes
        response = re.sub(r'^(?:cypher|neo4j|query|here is|answer):\s*', '', 
                         response, flags=re.IGNORECASE)
        
        # Remove lines that look like explanations
        lines = []
        for line in response.split('\n'):
            # Skip lines that are clearly not Cypher
            line_lower = line.strip().lower()
            if line_lower.startswith(('note:', 'explanation:', 'this query', 
                                     'template:', 'using template')):
                continue
            # Also try to extract template ID if it's in a comment or note
            if not template_id:
                tid_match = re.search(r'\b(T\d{3}[a-z]?)\b', line)
                if tid_match:
                    template_id = tid_match.group(1)
            lines.append(line)
        
        response = '\n'.join(lines)
        
        # Try to extract just the MATCH...RETURN part
        match = re.search(r'(MATCH|CALL|WITH)\s+.*?RETURN\s+.*?(?=\n\n|\n[A-Z][a-z]+:|$)', 
                         response, re.IGNORECASE | re.DOTALL)
        if match:
            response = match.group(0)
        
        return response.strip(), template_id
    

    def _fix_prefixes(self, query: str) -> str:
        """Ensure entity IDs have correct prefixes"""
        if not query:
            return query
        
        # Fix EC numbers - add EC: prefix if missing
        query = re.sub(r"id:\s*['\"](?<!EC:)(\d+\.\d+\.\d+\.\d+)['\"]", 
                      r"id: 'EC:\1'", query)
        
        # Fix pathway IDs - add path: prefix if missing
        # Match pathway IDs like eco00010, hsa00010, etc.
        query = re.sub(r"id:\s*['\"](?<!path:)([a-z]{2,3}\d{5})['\"]", 
                      r"id: 'path:\1'", query)
        
        return query


# # Example usage and testing
# if __name__ == "__main__":
#     llm = ()
#     generator = SimpleCypherGenerator(llm)
    
#     # Test query
#     question = "What enzymes does gene eco:b0001 encode?"
#     cypher, metadata = generator.generate_query(question)
    
