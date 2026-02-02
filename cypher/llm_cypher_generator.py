from typing import Any, List, Dict, Optional, Tuple
import re
import random

class LLMCypherQueryGenerator:
    """
    Hybrid Cypher Generator (V17 - Max Precision):
    - Added T057b: Gene -> Ortholog -> Product (Fixing 'via ortholog' product queries).
    - Added T076: Inter-pathway Metabolite Exchange.
    - Refined Routing priority to catch 'Product' targets even in complex sentences.
    - Maintains V16's strict GT mimicry and prefix safety.
    """
    
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

        # --- [T011-T025] 1-Hop Relationships ---
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

        # --- [T026-T035] Multi-Hop Relationships ---
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
        
        # --- [T036-T045] Stats & Counts ---
        "T036": { "description": "Count enzymes of Gene", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[r:ENCODES]-(e:EC) RETURN count(r)" },
        "T037": { "description": "Count reactions in Pathway", "cypher": "MATCH (p:Pathway {id: '{PATHWAY_ID}'})-[r:CONTAINS]-(rxn:Reaction) RETURN count(r)" },
        "T038": { "description": "Count genes of Ortholog", "cypher": "MATCH (o:Ortholog {id: '{ORTHOLOG_ID}'})-[r:BELONGS_TO]-(g:Gene) RETURN count(r)" },
        "T039": { "description": "Top Pathways by size", "cypher": "MATCH (p:Pathway)-[r:CONTAINS]-(rxn:Reaction) RETURN p.id, count(r) AS cnt ORDER BY cnt DESC LIMIT 10" }, 
        "T040": { "description": "Shared Reactions", "cypher": "MATCH (p1:Pathway {id: '{PATHWAY_ID_1}'})-[:CONTAINS]-(r:Reaction)-[:CONTAINS]-(p2:Pathway {id: '{PATHWAY_ID_2}'}) RETURN r" },
        "T041": { "description": "Count all genes", "cypher": "MATCH (n:Gene) RETURN count(n)" },
        "T042": { "description": "Count all pathways", "cypher": "MATCH (n:Pathway) RETURN count(n)" },
        "T043": { "description": "Count all compounds", "cypher": "MATCH (n:Compound) RETURN count(n)" },
        "T044": { "description": "Count all enzymes", "cypher": "MATCH (n:EC) RETURN count(n)" },
        "T045": { "description": "Count all reactions", "cypher": "MATCH (n:Reaction) RETURN count(n)" },

        # --- [T046-T051] Global Lists ---
        "T046": { "description": "List all genes", "cypher": "MATCH (n:Gene) RETURN n LIMIT 50" },
        "T047": { "description": "List all pathways", "cypher": "MATCH (n:Pathway) RETURN n LIMIT 50" },
        "T048": { "description": "List all compounds", "cypher": "MATCH (n:Compound) RETURN n LIMIT 50" },
        "T049": { "description": "List all enzymes", "cypher": "MATCH (n:EC) RETURN n LIMIT 50" },
        "T050": { "description": "List all reactions", "cypher": "MATCH (n:Reaction) RETURN n LIMIT 50" },
        "T051": { "description": "List all orthologs", "cypher": "MATCH (n:Ortholog) RETURN n LIMIT 50" },

        # --- [T052-T055] Edge Cases ---
        "T052": { "description": "Find reactions with NO products", "cypher": "MATCH (r:Reaction) WHERE NOT (r)-[:PRODUCES]-() RETURN r" },
        "T053": { "description": "Find reactions with NO substrates", "cypher": "MATCH (r:Reaction) WHERE NOT (r)-[:SUBSTRATE_OF]-() RETURN r" },
        "T054": { "description": "Find orphan pathways (empty)", "cypher": "MATCH (p:Pathway) WHERE NOT (p)-[:CONTAINS]-() RETURN p" },
        "T055": { "description": "Find enzymes not catalyzing any reaction", "cypher": "MATCH (e:EC) WHERE NOT (e)-[:CATALYZES]-() RETURN e" },

        # --- [T056-T059] Gap Fillers (Deep Inference) ---
        "T056": { "description": "Pathways involving Gene (via Enzyme)", "cypher": "MATCH (p:Pathway)-[:CONTAINS]-(r:Reaction)-[:CATALYZES]-(e:EC)-[:ENCODES]-(g:Gene {id: '{GENE_ID}'}) RETURN DISTINCT p" },
        "T057": { "description": "Pathways involving Gene (via Ortholog)", "cypher": "MATCH (p:Pathway)-[:CONTAINS]-(r:Reaction)-[:CATALYZES]-(o:Ortholog)-[:BELONGS_TO]-(g:Gene {id: '{GENE_ID}'}) RETURN DISTINCT p" },
        "T058": { "description": "Compounds produced by Gene (via Enzyme)", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:ENCODES]-(e:EC)-[:CATALYZES]-(r:Reaction)-[:PRODUCES]-(c:Compound) RETURN DISTINCT c" },
        "T059": { "description": "Genes producing Compound", "cypher": "MATCH (c:Compound {id: '{COMPOUND_ID}'})<-[:PRODUCES]-(r:Reaction)<-[:CATALYZES]-(e:EC)<-[:ENCODES]-(g:Gene) RETURN DISTINCT g" },
        
        # [V17 NEW] Compounds via Ortholog (Targeting the specific failure)
        "T057b": { "description": "Compounds produced by Gene (via Ortholog)", "cypher": "MATCH (g:Gene {id: '{GENE_ID}'})-[:BELONGS_TO]-(o:Ortholog)-[:CATALYZES]-(r:Reaction)-[:PRODUCES]-(c:Compound) RETURN DISTINCT c" },

        # --- [T060-T069] Advanced Filters ---
        "T060": { "description": "Pathways with > N reactions", "cypher": "MATCH (p:Pathway)-[r:CONTAINS]-(rxn:Reaction) WITH p, count(r) as cnt WHERE cnt > 10 RETURN p" },
        "T061": { "description": "Genes starting with prefix", "cypher": "MATCH (n:Gene) WHERE n.id STARTS WITH '{PREFIX}' RETURN n LIMIT 20" },
        "T062": { "description": "Pathways starting with prefix", "cypher": "MATCH (n:Pathway) WHERE n.id STARTS WITH '{PREFIX}' RETURN n LIMIT 20" },
        "T063": { "description": "Compounds starting with prefix", "cypher": "MATCH (n:Compound) WHERE n.id STARTS WITH '{PREFIX}' RETURN n LIMIT 20" },
        "T064": { "description": "Enzymes starting with prefix", "cypher": "MATCH (n:EC) WHERE n.id STARTS WITH '{PREFIX}' RETURN n LIMIT 20" },
        "T065": { "description": "Reactions starting with prefix", "cypher": "MATCH (n:Reaction) WHERE n.id STARTS WITH '{PREFIX}' RETURN n LIMIT 20" },
        "T066": { "description": "Orthologs starting with prefix", "cypher": "MATCH (n:Ortholog) WHERE n.id STARTS WITH '{PREFIX}' RETURN n LIMIT 20" },
        "T067": { "description": "Functional Units starting with prefix", "cypher": "MATCH (n:FunctionalUnit) WHERE n.id STARTS WITH '{PREFIX}' RETURN n LIMIT 20" },
        "T068": { "description": "Reactions with > 2 Substrates", "cypher": "MATCH (r:Reaction)-[rel:SUBSTRATE_OF]->(c:Compound) WITH r, count(rel) as input_cnt WHERE input_cnt > 2 RETURN r" },
        "T069": { "description": "Reactions with > 2 Products", "cypher": "MATCH (r:Reaction)-[rel:PRODUCES]->(c:Compound) WITH r, count(rel) as output_cnt WHERE output_cnt > 2 RETURN r" },

        # --- [T070-T076] Pathfinding & Complex (V17 Updated) ---
        "T070": { "description": "Shortest path Compound to Compound", "cypher": "MATCH p=shortestPath((c1:Compound {id: '{COMPOUND_ID_1}'})-[*]-(c2:Compound {id: '{COMPOUND_ID_2}'})) RETURN p" },
        "T071": { "description": "Shortest path Gene to Pathway", "cypher": "MATCH p=shortestPath((g:Gene {id: '{GENE_ID}'})-[*]-(path:Pathway {id: '{PATHWAY_ID}'})) RETURN p" },
        "T072": { "description": "Shortest path Reaction to Reaction", "cypher": "MATCH p=shortestPath((r1:Reaction {id: '{REACTION_ID_1}'})-[*]-(r2:Reaction {id: '{REACTION_ID_2}'})) RETURN p" },
        "T073": { "description": "Shortest path Gene to Gene", "cypher": "MATCH p=shortestPath((g1:Gene {id: '{GENE_ID_1}'})-[*]-(g2:Gene {id: '{GENE_ID_2}'})) RETURN p" },
        "T074": { "description": "Shortest path Pathway to Pathway", "cypher": "MATCH p=shortestPath((p1:Pathway {id: '{PATHWAY_ID_1}'})-[*]-(p2:Pathway {id: '{PATHWAY_ID_2}'})) RETURN p" },
        "T075": { "description": "Other genes encoding same enzyme (Siblings)", "cypher": "MATCH (g1:Gene {id: '{GENE_ID}'})-[:ENCODES]->(e:EC)<-[:ENCODES]-(g2:Gene) RETURN g2" },
        # [V17 NEW] Metabolite Exchange
        "T076": { "description": "Inter-pathway Metabolite Exchange", "cypher": "MATCH (p1:Pathway {id: '{PATHWAY_ID}'})-[:CONTAINS]->(:Reaction)-[:PRODUCES]->(c:Compound)<-[:SUBSTRATE_OF]-(:Reaction)<-[:CONTAINS]-(p2:Pathway) WHERE p1 <> p2 RETURN DISTINCT c" }
    }

    TEMPLATE_METADATA = {
        "T001": {"source": "GENE", "target": "GENE"}, "T002": {"source": "PATHWAY", "target": "PATHWAY"}, "T003": {"source": "COMPOUND", "target": "COMPOUND"}, "T004": {"source": "EC", "target": "EC"}, "T005": {"source": "REACTION", "target": "REACTION"},
        "T006": {"source": "GENE", "target": "GENE"}, "T007": {"source": "PATHWAY", "target": "PATHWAY"}, "T008": {"source": "COMPOUND", "target": "COMPOUND"}, "T009": {"source": "EC", "target": "EC"}, "T010": {"source": "REACTION", "target": "REACTION"},
        "T011": {"source": "GENE", "target": "EC"}, "T012": {"source": "GENE", "target": "ORTHOLOG"}, "T013": {"source": "GENE", "target": "FUNCTIONALUNIT"},
        "T014": {"source": "COMPOUND", "target": "REACTION"}, "T015": {"source": "COMPOUND", "target": "REACTION"},
        "T016": {"source": "EC", "target": "REACTION"}, "T017": {"source": "ORTHOLOG", "target": "REACTION"}, "T018": {"source": "ORTHOLOG", "target": "EC"},
        "T019": {"source": "EC", "target": "GENE"}, "T020": {"source": "ORTHOLOG", "target": "GENE"},
        "T021": {"source": "REACTION", "target": "COMPOUND"}, "T022": {"source": "REACTION", "target": "COMPOUND"}, "T023": {"source": "REACTION", "target": "EC"},
        "T024": {"source": "PATHWAY", "target": "REACTION"}, "T025": {"source": "PATHWAY", "target": "FUNCTIONALUNIT"},
        "T026": {"source": "REACTION", "target": "PATHWAY"}, "T027": {"source": "GENE", "target": "REACTION"}, "T028": {"source": "GENE", "target": "REACTION"}, "T029": {"source": "GENE", "target": "COMPOUND"},
        "T030": {"source": "COMPOUND", "target": "COMPOUND"}, "T031": {"source": "COMPOUND", "target": "COMPOUND"},
        "T032": {"source": "REACTION", "target": "REACTION"}, "T033": {"source": "PATHWAY", "target": "COMPOUND"}, "T034": {"source": "PATHWAY", "target": "COMPOUND"}, "T035": {"source": "FUNCTIONALUNIT", "target": "EC"},
        "T036": {"source": "GENE", "target": "EC"}, "T037": {"source": "PATHWAY", "target": "REACTION"}, "T038": {"source": "ORTHOLOG", "target": "GENE"}, "T040": {"source": "PATHWAY", "target": "REACTION"},
        "T046": {"source": "GENE", "target": "GENE"}, "T047": {"source": "PATHWAY", "target": "PATHWAY"}, "T048": {"source": "COMPOUND", "target": "COMPOUND"}, "T049": {"source": "EC", "target": "EC"}, "T050": {"source": "REACTION", "target": "REACTION"},
        "T056": {"source": "GENE", "target": "PATHWAY"}, "T057": {"source": "GENE", "target": "PATHWAY"}, 
        "T058": {"source": "GENE", "target": "COMPOUND"}, "T059": {"source": "COMPOUND", "target": "GENE"},
        "T057b": {"source": "GENE", "target": "COMPOUND"}, # New V17 mapping
        "T060": {"source": "PATHWAY", "target": "PATHWAY"}, "T061": {"source": "GENE", "target": "GENE"}, "T062": {"source": "PATHWAY", "target": "PATHWAY"}, "T063": {"source": "COMPOUND", "target": "COMPOUND"}, "T064": {"source": "EC", "target": "EC"},
        "T065": {"source": "REACTION", "target": "REACTION"}, "T066": {"source": "ORTHOLOG", "target": "ORTHOLOG"}, "T067": {"source": "FUNCTIONALUNIT", "target": "FUNCTIONALUNIT"},
        "T068": {"source": "REACTION", "target": "REACTION"}, "T069": {"source": "REACTION", "target": "REACTION"},
        "T070": {"source": "COMPOUND", "target": "PATH"}, "T071": {"source": "GENE", "target": "PATH"}, "T072": {"source": "REACTION", "target": "PATH"}, "T073": {"source": "GENE", "target": "PATH"}, "T074": {"source": "PATHWAY", "target": "PATH"}, "T075": {"source": "GENE", "target": "GENE"},
        "T076": {"source": "PATHWAY", "target": "COMPOUND"} # New V17 mapping
    }

    def __init__(self, llm: Optional[Any] = None, provider: str = "gemini", model_name: Optional[str] = None, api_key: Optional[str] = None, host: Optional[str] = None, temperature: float = 0.0, schema: Optional[str] = None):
        if llm: self.llm = llm
        else: pass 
        self.schema = schema or self.DETAILED_SCHEMA

    def generate_query(self, question: str, intent: Optional[Any] = None, entities: List[Dict] = []) -> Tuple[str, str]:
        # 1. Selection
        template_id = self._select_template(question)
        
        # 2. Correction (Dynamic Routing)
        template_id = self._route_to_correct_template(template_id, question)

        final_cypher = ""
        gen_type = "Fallback"

        if template_id and template_id in self.CYPHER_TEMPLATES:
            # Use Multi-ID Logic here
            filled_cypher = self._fill_template_smart(template_id, question, entities)
            if filled_cypher:
                final_cypher = filled_cypher
                gen_type = "Template"
        
        if not final_cypher:
            final_cypher = self._generate_raw_query(question, intent, entities)
            gen_type = "Fallback"
            
        # [V16 Final Safety Net] Post-process to ensure IDs have prefixes
        final_cypher = self._post_process_prefixes(final_cypher)
        
        return final_cypher, gen_type

    def _route_to_correct_template(self, template_id: Optional[str], question: str) -> Optional[str]:
        if not template_id: return template_id
        
        q_lower = question.lower()
        actual_source = self._detect_strict_id_type(question)

        # 1. Pathfinding / Route
        if "shortest" in q_lower or "route" in q_lower or "path from" in q_lower:
            if "compound" in q_lower: return "T070"
            if "gene" in q_lower and "pathway" in q_lower: return "T071"
            if "reaction" in q_lower: return "T072"

        # 2. Shared / Common
        if "share" in q_lower or "common" in q_lower or "both" in q_lower:
            if "pathway" in q_lower: return "T040"
            if "gene" in q_lower and "enzyme" in q_lower: return "T075"

        # 3. Ortholog specific paths (Priority Fix V17)
        if "ortholog" in q_lower:
            # If asking for PRODUCTS (compounds) via ortholog -> T057b
            if "product" in q_lower or "compound" in q_lower: return "T057b"
            # If asking for PATHWAYS via ortholog -> T057
            if "pathway" in q_lower and "gene" in q_lower: return "T057"
        
        # 4. Deep Inference (Gene <-> Compound)
        if "produce" in q_lower or "product" in q_lower:
             if actual_source == "GENE" and "compound" in q_lower: return "T058"
             if actual_source == "COMPOUND" and "gene" in q_lower: return "T059"
             
        # 5. Metabolite Exchange (V17)
        if "exchange" in q_lower or "inter-pathway" in q_lower:
            if "pathway" in q_lower: return "T076"

        # 6. Functional Unit Activities (V17)
        if "functional unit" in q_lower and "enzyme" in q_lower:
            return "T035" # Ensure this maps to T035

        # 7. Filter / Starts with
        if "start" in q_lower or "begin" in q_lower:
            if "gene" in q_lower or actual_source == "GENE": return "T061"
            if "pathway" in q_lower or actual_source == "PATHWAY": return "T062"
            if "compound" in q_lower or actual_source == "COMPOUND": return "T063"
            if "enzyme" in q_lower or actual_source == "EC": return "T064"
            if "reaction" in q_lower or actual_source == "REACTION": return "T065" 
            if "ortholog" in q_lower or actual_source == "ORTHOLOG": return "T066"
            if "unit" in q_lower or actual_source == "FUNCTIONALUNIT": return "T067"

        # 8. Complex Filter (> N)
        if "more than" in q_lower:
             if "pathway" in q_lower: return "T060"
             if "reaction" in q_lower and "substrate" in q_lower: return "T068"
             if "reaction" in q_lower and "product" in q_lower: return "T069"
            
        # 9. Count Logic Check
        if template_id in ["T041", "T042", "T043", "T044", "T045"] and actual_source:
            target_map = { "T041": "GENE", "T042": "PATHWAY", "T043": "COMPOUND", "T044": "EC", "T045": "REACTION" }
            target_intent = target_map.get(template_id, "REACTION")
            new_template = self._find_template_by_path(source=actual_source, target=target_intent, question=question)
            if new_template: return new_template
        
        # 10. List All check
        if template_id in ["T046", "T047", "T048", "T049", "T050", "T051"] and actual_source:
             target_map = {"T046": "GENE", "T047": "PATHWAY", "T048": "COMPOUND", "T050": "REACTION"}
             target_intent = target_map.get(template_id, "GENE")
             new_template = self._find_template_by_path(source=actual_source, target=target_intent, question=question)
             if new_template: return new_template

        if template_id not in self.TEMPLATE_METADATA: return template_id
        meta = self.TEMPLATE_METADATA[template_id]
        expected_source = meta["source"]
        target_intent = meta["target"]

        if actual_source == expected_source: return template_id

        new_template = self._find_template_by_path(source=actual_source, target=target_intent, question=question)
        if new_template: return new_template
            
        return template_id

    def _find_template_by_path(self, source: str, target: str, question: str = "") -> Optional[str]:
        candidates = []
        for tid, meta in self.TEMPLATE_METADATA.items():
            if meta["source"] == source and meta["target"] == target:
                candidates.append(tid)
        
        if not candidates: return None
        
        q_lower = question.lower()
        if source == "GENE" and target == "REACTION":
            if "ortholog" in q_lower and "T027" in candidates: return "T027"
            if "T028" in candidates: return "T028"
        if source == "GENE" and target == "PATHWAY":
            if "T056" in candidates: return "T056"
            
        return candidates[0]

    def _detect_strict_id_type(self, question: str) -> Optional[str]:
        if re.search(r'\bM\d{5}\b', question): return "FUNCTIONALUNIT"
        if re.search(r'\b[a-z]{2,4}:[A-Z0-9_]+\b', question):
            if "path:" in question: return "PATHWAY"
            return "GENE"
        if re.search(r'\bK\d{5}\b', question): return "ORTHOLOG"
        if re.search(r'\bR\d{5}\b', question): return "REACTION"
        if re.search(r'\bC\d{5}\b', question): return "COMPOUND"
        if re.search(r'\b(?:\d+\.){3}\d+\b', question) or "EC:" in question: return "EC"
        if "path:" in question or re.search(r'\b[a-z]{2,3}\d{5}\b', question): return "PATHWAY"
        return None

    def _select_template(self, question: str) -> Optional[str]:
        if not hasattr(self, 'CYPHER_TEMPLATES'): return None
        
        q_lower = question.lower()
        is_count_query = any(k in q_lower for k in ["count", "how many", "number of"])
        is_negation = any(k in q_lower for k in ["no ", "not ", "without", "empty", "orphan"])

        candidate_ids = []
        if is_negation:
             candidate_ids = ["T052", "T053", "T054", "T055"]
        elif is_count_query:
            candidate_ids = [k for k, v in self.CYPHER_TEMPLATES.items() if "count" in v['description'].lower()]
        else:
            candidate_ids = [k for k, v in self.CYPHER_TEMPLATES.items() 
                             if "count" not in v['description'].lower() 
                             and k not in ["T052", "T053", "T054", "T055"]]

        candidate_summaries = [f"{tid}: {self.CYPHER_TEMPLATES[tid]['description']}" for tid in candidate_ids]
        templates_str = "\n".join(candidate_summaries)
        
        prompt = f"""
        Act as a Smart Query Router.
        [User Question] "{question}"
        [Candidate Templates]
        {templates_str}
        [INSTRUCTIONS]
        1. Identify the TARGET ENTITY (What user wants).
        2. Match with Template Description.
        3. If asking for "starting with...", pick the "starting with" template.
        4. If asking for "Shared" or "Path", pick the relevant template.
        5. Return ONLY Template ID.
        """
        try:
            response = self.llm.generate(prompt).strip()
            match = re.search(r'T\d{3}', response)
            return match.group(0) if match else None
        except: return None

    def _fill_template_smart(self, template_id: str, question: str, entities: List[Dict]) -> Optional[str]:
        template_data = self.CYPHER_TEMPLATES[template_id]
        raw_cypher = template_data["cypher"]
        
        if "{PREFIX}" in raw_cypher:
            prefix_match = re.search(r'(?:starting|starts) with ([A-Za-z0-9:]+)', question, re.IGNORECASE)
            if prefix_match: return raw_cypher.replace("{PREFIX}", prefix_match.group(1))
        
        if "{" not in raw_cypher: return raw_cypher

        regex_filled = self._fill_template_regex_multi(raw_cypher, entities, question)
        if regex_filled and "{" not in regex_filled: return regex_filled

        try:
            prompt = f"Template: {raw_cypher}\nQuestion: {question}\nEntities: {entities}\nFill placeholders. Output ONLY the filled Cypher query. NO explanations. NO markdown."
            response = self.llm.generate(prompt).strip()
            return self._clean_query(response)
        except: pass
        return regex_filled

    def _fill_template_regex_multi(self, cypher: str, entities: List[Dict], question: str) -> Optional[str]:
        found_map = {
            "PATHWAY": re.findall(r'(path:[a-z]+\d+|[a-z]{2,3}\d{5})', question),
            "GENE": re.findall(r'([a-z]{2,4}:[A-Z0-9_]+)', question),
            "COMPOUND": re.findall(r'(C\d{5})', question),
            "REACTION": re.findall(r'(R\d{5})', question),
            "EC": re.findall(r'(\d+\.\d+\.\d+\.\d+|EC:?[\d\.]+)', question)
        }
        
        for k in found_map:
            if k == "GENE": found_map[k] = [x for x in found_map[k] if "path:" not in x]
            if k == "EC": found_map[k] = [x if x.startswith("EC:") else f"EC:{x}" for x in found_map[k]]
            if k == "PATHWAY": found_map[k] = [x if x.startswith("path:") else f"path:{x}" for x in found_map[k]]

        filled = cypher
        
        placeholders_numbered = re.findall(r'\{([A-Z]+)_ID_(\d+)\}', cypher)
        for p_type, idx_str in placeholders_numbered:
            idx = int(idx_str) - 1
            if p_type in found_map and idx < len(found_map[p_type]):
                filled = filled.replace(f"{{{p_type}_ID_{idx_str}}}", found_map[p_type][idx])

        placeholders_generic = re.findall(r'\{([A-Z]+)_ID\}', cypher)
        for p_type in placeholders_generic:
             if p_type in found_map and found_map[p_type]:
                 filled = filled.replace(f"{{{p_type}_ID}}", found_map[p_type][0])

        if re.search(r'\{[A-Z_0-9]+\}', filled): return None
        return filled

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
        Act as a precise Neo4j Cypher Developer. 
        
        [CRITICAL RULES]
        1. **MIMIC GT STYLE**: Your query structure must match standard Ground Truth patterns exactly.
           - Use `WHERE NOT (n)--()` instead of `WHERE NOT EXISTS`.
           - For counts, prefer counting relationships `count(r)` over nodes if implying connections.
        2. **MATCH TARGET NODE**: Ensure the RETURN variable matches the requested entity type exactly.
           - Asking for "Genes"? -> RETURN (g:Gene).
           - Asking for "Reactions"? -> RETURN (r:Reaction).
        
        [STRICT SCHEMA] 
        {self.schema}
        
        Question: "{question}"
        Entities: {entities_str}
        
        Output ONLY the Cypher query. NO explanations. NO markdown code blocks.
        """
        generated = self.llm.generate(prompt)
        return self._clean_query(generated)

    def _clean_query(self, cypher_query: str) -> str:
        if not cypher_query: return ""
        q = re.sub(r'```\w*\s*', '', cypher_query) 
        q = q.replace('```', '')
        q = re.sub(r'^(?:neo4j|cypher|sql|here is|answer:)\s+', '', q, flags=re.IGNORECASE)
        lines = q.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.lower().startswith(('template:', 'question:', 'entities:', 'fill placeholders')): continue
            cleaned_lines.append(line)
        q = '\n'.join(cleaned_lines)
        match = re.search(r'(MATCH|CALL|WITH|RETURN)\s+.*', q, re.IGNORECASE | re.DOTALL)
        if match: q = match.group(0)
        return q.strip()

    def _post_process_prefixes(self, query: str) -> str:
        query = re.sub(r"id:\s*['\"](?<!EC:)(\d+\.\d+\.\d+\.\d+)['\"]", r"id: 'EC:\1'", query)
        query = re.sub(r"id:\s*['\"](?<!path:)([a-z]{2,3}\d{5})['\"]", r"id: 'path:\1'", query)
        return query