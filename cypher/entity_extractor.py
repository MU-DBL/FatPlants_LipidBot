import re
from typing import List, Dict, Optional, Tuple, Set
from rapidfuzz import process, fuzz
from cypher.ac import load_cache, norm
from config import AC_KEGG_PKL
from cypher.llm_entity_extractor import LLMBioEntityExtractor
from llm_factory import BaseLLM

class BioEntityExtractor:

    DB_PRIORITY = ["gene", "ortholog", "compound", "ec", "reaction", "pathway", "-"]
    
    ENZYME_PATTERN = re.compile(
        r"""\b
        [a-z][a-z0-9\-\s]{0,60}?
        (
            dehydrogenase|oxidoreductase|oxygenase|monooxygenase|dioxygenase|
            kinase|phosphatase|phospholipase|carboxylase|carboxykinase|
            synthase|synthetase|transferase|acyltransferase|aminotransferase|
            lyase|dehydratase|hydrolase|isomerase|mutase|epimerase|racemase|
            ligase|cyclase|reductase
        )\b
        """, re.I | re.X
    )
    
    # Explicit ID patterns (with relaxed EC: allows x.x.x.- / x.x.x.n / optional EC prefix with space/colon)
    ID_PATTERNS = [
        re.compile(r'\bC\d{5}\b'),                 # Compound
        re.compile(r'\bK\d{5}\b'),                 # Ortholog
        re.compile(r'\bR\d{5}\b'),                 # Reaction
        re.compile(r'\b(?:EC[:\s])?\d+\.\d+\.\d+\.(?:\d+|-|x|n)\b', re.I),  # EC
    ]
    
    SPECIES_HINTS = {
        "arabidopsis": "ath",
        "ath": "ath",
        "thaliana": "ath",
        "soybean": "gmx",
        "glycine": "gmx",
        "gmx": "gmx",
        "camelina": "csat",
        "csat": "csat",
        "aegilops tauschii": "ats",
        "ats": "ats",
    }
    
    def __init__(self, llm: BaseLLM, default_fuzzy_threshold: int = 95):
        if llm is None:
            raise ValueError("llm must not be None")
            
        self.llm_entity_extractor = LLMBioEntityExtractor(llm)
        self.ac_automaton, self.alias_map = load_cache(AC_KEGG_PKL)
        self.vocab = list(self.alias_map.keys())
        self.default_fuzzy_threshold = default_fuzzy_threshold
    
    def extract_mentions(
        self,
        question: str,
        species_hint: Optional[str] = None,
        use_regex: bool = True,
        use_llm: bool = True,
        fuzzy_threshold: Optional[int] = None
    ) -> List[Dict]:

        if fuzzy_threshold is None:
            fuzzy_threshold = self.default_fuzzy_threshold
        
        qn = norm(question)
        species_hint = species_hint or self._guess_species_hint(question)
        
        # Extract from different sources
        ac_hits = self._extract_ac(qn, species_hint)
        regex_hits = self._extract_regex(question, species_hint, fuzzy_threshold) if use_regex else []
        llm_hits = self._extract_llm(question, qn, species_hint, fuzzy_threshold) if use_llm else []
        
        # Combine and deduplicate
        all_hits = ac_hits + regex_hits + llm_hits
        print(all_hits)
        # Remove hits without position information
        all_hits = [h for h in all_hits if "start" in h and "end" in h]
        
        # Deduplicate and resolve conflicts
        deduplicated = self._deduplicate_hits(all_hits)
        
        # Final sort: left to right, then by DB priority
        deduplicated.sort(key=lambda h: (
            h["start"],
            self.DB_PRIORITY.index(h.get("db", "-")) if h.get("db", "-") in self.DB_PRIORITY else 999
        ))
        
        return deduplicated
    
    def _extract_ac(self, normalized_question: str, species_hint: Optional[str]) -> List[Dict]:
        """Extract entities using AC automaton for exact matching."""
        ac_hits = []
        
        for end, _ in self.ac_automaton.iter(normalized_question):
            for length in range(1, min(256, end + 1) + 1):
                start = end - length + 1
                span = normalized_question[start:end + 1]
                
                if span in self.alias_map:
                    candidates = self.alias_map[span]
                    if species_hint:
                        candidates = [c for c in candidates if c.get("species") in (species_hint, "-")]
                    
                    for candidate in candidates:
                        ac_hits.append({
                            "text": span,
                            "id": candidate["id"],
                            "db": candidate["db"],
                            "species": candidate.get("species", "-"),
                            "start": start,
                            "end": end + 1,  # Python slice right-open interval
                            "src": "ac",
                            "confidence": 1.0
                        })
                    break
        
        if not ac_hits:
            return []
        
        # Sort by length (descending), position, then DB priority
        ac_hits.sort(key=lambda h: (
            -(h["end"] - h["start"]),
            h["start"],
            self.DB_PRIORITY.index(h["db"]) if h["db"] in self.DB_PRIORITY else 999
        ))
        
        # Remove overlapping hits (keep longer/higher priority ones)
        return self._remove_overlaps(ac_hits, len(normalized_question))
    
    def _extract_regex(self, question: str, species_hint: Optional[str], fuzzy_threshold: int) -> List[Dict]:
        """Extract entities using regex patterns for IDs and enzyme names."""
        regex_hits = []
        
        # Extract explicit IDs directly
        regex_hits.extend(self._extract_explicit_ids(question))
        
        # Extract enzyme name phrases
        for match in self.ENZYME_PATTERN.finditer(question):
            regex_hits.extend(
                self._map_span_to_ids(
                    match.group(0),
                    start=match.start(),
                    end=match.end(),
                    cutoff=fuzzy_threshold,
                    allowed_dbs={"enzyme", "ortholog", "ec"}
                )
            )
        
        # Extract and map explicit IDs with fuzzy matching
        seen_spans = set()
        for pattern in self.ID_PATTERNS:
            for match in pattern.finditer(question):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                
                text = match.group(0)
                allowed_dbs = self._get_allowed_dbs_for_id(text)
                
                regex_hits.extend(
                    self._map_span_to_ids(
                        text,
                        start=match.start(),
                        end=match.end(),
                        top_k=3,
                        cutoff=fuzzy_threshold,
                        allowed_dbs=allowed_dbs
                    )
                )
        
        # Filter by species if specified
        if species_hint:
            regex_hits = [h for h in regex_hits if h.get("species") in (species_hint, "-")]
        
        # Sort by priority: prioritize specific DBs, then by length
        regex_hits.sort(key=lambda h: (
            h.get("db") not in ("ortholog", "ec", "enzyme", "compound", "reaction"),
            -(h["end"] - h["start"])
        ))
        
        return regex_hits
    
    def _extract_llm(
        self,
        question: str,
        normalized_question: str,
        species_hint: Optional[str],
        fuzzy_threshold: int
    ) -> List[Dict]:
        """Extract entities using LLM-based extraction."""
        llm_output = self.llm_entity_extractor.extract(question)
        if not llm_output or "mentions" not in llm_output:
            return []
        
        hits = []
        for mention in llm_output["mentions"]:
            raw_text = mention["text"]
            s0, e0 = mention["start"], mention["end"]
            
            # If LLM gives a long sentence, try to extract enzyme phrase from it
            inner_match = self.ENZYME_PATTERN.search(raw_text)
            if inner_match:
                text = inner_match.group(0)
                start = s0 + inner_match.start()
                end = s0 + inner_match.end()
                allowed_dbs = {"enzyme", "ortholog", "ec"}
            else:
                text = raw_text
                start, end = s0, e0
                allowed_dbs = self._allowed_dbs_for_text(text)
            
            text_norm = norm(text)
            
            # Try exact match first
            if text_norm in self.alias_map:
                candidates = self.alias_map[text_norm]
                if species_hint:
                    candidates = [c for c in candidates if c.get("species") in (species_hint, "-")]
                
                for candidate in candidates:
                    if allowed_dbs and candidate.get("db") not in allowed_dbs:
                        continue
                    hits.append({
                        "text": text,
                        "id": candidate["id"],
                        "db": candidate["db"],
                        "species": candidate.get("species", "-"),
                        "start": start,
                        "end": end,
                        "src": "llm-exact",
                        "confidence": 0.85
                    })
            else:
                # Try fuzzy matching
                fuzzy_results = process.extract(
                    text_norm,
                    self.vocab,
                    scorer=fuzz.token_set_ratio,
                    limit=5
                )
                
                for alias, score, _ in fuzzy_results:
                    if score < fuzzy_threshold:
                        continue
                    
                    candidates = self.alias_map[alias]
                    if species_hint:
                        candidates = [c for c in candidates if c.get("species") in (species_hint, "-")]
                    
                    for candidate in candidates:
                        if allowed_dbs and candidate.get("db") not in allowed_dbs:
                            continue
                        hits.append({
                            "text": text,
                            "id": candidate["id"],
                            "db": candidate["db"],
                            "species": candidate.get("species", "-"),
                            "start": start,
                            "end": end,
                            "score": score,
                            "src": "llm-fuzzy",
                            "confidence": (score / 100.0) * 0.85
                        })
        
        # Sort by length (descending), DB priority, then score
        hits.sort(key=lambda h: (
            -(h["end"] - h["start"]),
            self.DB_PRIORITY.index(h.get("db", "-")) if h.get("db", "-") in self.DB_PRIORITY else 999,
            -h.get("score", 100)
        ))
        
        return hits
    
    def _map_span_to_ids(
        self,
        span_text: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        top_k: int = 5,
        cutoff: int = 88,
        allowed_dbs: Optional[Set[str]] = None
    ) -> List[Dict]:
        """Map a text span to entity IDs using fuzzy matching."""
        query = norm(span_text)
        
        def create_hit(candidate: Dict, source: str, score: int = 100, confidence: float = 0.98) -> Optional[Dict]:
            if allowed_dbs is not None and candidate.get("db") not in allowed_dbs:
                return None
            
            hit = {
                "text": span_text,
                **candidate,
                "src": source,
                "score": score,
                "confidence": confidence
            }
            
            if start is not None and end is not None:
                hit["start"], hit["end"] = start, end
            
            return hit
        
        # Try exact match first
        if query in self.alias_map:
            results = []
            for candidate in self.alias_map[query]:
                hit = create_hit(candidate, "regex-exact")
                if hit:
                    results.append(hit)
            return results
        
        # Fuzzy matching with adaptive threshold
        effective_cutoff = 91 if len(query) <= 8 else cutoff
        fuzzy_matches = process.extract(
            query,
            self.vocab,
            scorer=fuzz.token_set_ratio,
            limit=top_k
        )
        
        results = []
        for alias, score, _ in fuzzy_matches:
            if score < effective_cutoff:
                continue
            
            for candidate in self.alias_map[alias]:
                hit = create_hit(candidate, "regex-fuzzy", score=score, confidence=score / 100.0)
                if hit:
                    results.append(hit)
        
        return results
    
    def _extract_explicit_ids(self, question: str) -> List[Dict]:
        """Extract explicit IDs (K, C, R, EC numbers) and create direct hits."""
        hits = []
        seen_spans = set()
        
        for pattern in self.ID_PATTERNS:
            for match in pattern.finditer(question):
                span = (match.start(), match.end())
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                
                raw_id = match.group(0)
                start, end = match.start(), match.end()
                
                if raw_id[0] in ("K", "k"):
                    db, entity_id = "ortholog", raw_id.upper()
                elif raw_id[0] in ("C", "c"):
                    db, entity_id = "compound", raw_id.upper()
                elif raw_id[0] in ("R", "r"):
                    db, entity_id = "reaction", raw_id.upper()
                else:
                    # EC number: allow EC: / EC space / direct digits; preserve -, x, n in 4th position
                    db, entity_id = "ec", self._normalize_ec(raw_id)
                
                hits.append({
                    "text": raw_id,
                    "id": entity_id,
                    "db": db,
                    "species": "-",
                    "start": start,
                    "end": end,
                    "src": "regex-id",
                    "confidence": 1.0
                })
        
        return hits
    
    def _deduplicate_hits(self, hits: List[Dict]) -> List[Dict]:
        """
        Deduplicate hits by:
        1. Keep highest priority hit for each span (start, end)
        2. Keep highest priority hit for each unique entity (id, db, species)
        """
        # Priority function
        def priority(hit: Dict) -> Tuple:
            src_rank = {
                "ac": 0,
                "regex-exact": 1,
                "regex-fuzzy": 2,
                "llm-exact": 3,
                "llm-fuzzy": 4
            }.get(hit.get("src", ""), 9)
            
            db = hit.get("db", "-")
            db_rank = self.DB_PRIORITY.index(db) if db in self.DB_PRIORITY else 999
            
            return (
                src_rank,
                db_rank,
                -hit.get("confidence", 0.5),
                -hit.get("score", 0)
            )
        
        # Step 1: Keep best hit for each span
        keep_by_span = {}
        for hit in hits:
            span_key = (hit["start"], hit["end"])
            if span_key not in keep_by_span or priority(hit) < priority(keep_by_span[span_key]):
                keep_by_span[span_key] = hit
        
        deduplicated = list(keep_by_span.values())
        
        # Step 2: Keep best hit for each unique entity
        best_by_entity = {}
        for hit in deduplicated:
            entity_key = (hit.get("id"), hit.get("db"), hit.get("species", "-"))
            if entity_key not in best_by_entity or priority(hit) < priority(best_by_entity[entity_key]):
                best_by_entity[entity_key] = hit
        
        return list(best_by_entity.values())
    
    def _remove_overlaps(self, hits: List[Dict], text_length: int) -> List[Dict]:
        """Remove overlapping hits, keeping longer/higher priority ones."""
        used = [False] * text_length
        kept = []
        
        for hit in hits:
            if any(used[i] for i in range(hit["start"], hit["end"])):
                continue
            
            for i in range(hit["start"], hit["end"]):
                used[i] = True
            
            kept.append(hit)
        
        return kept
    
    @staticmethod
    def _guess_species_hint(query: str) -> Optional[str]:
        """Guess species from query text."""
        query_lower = query.lower()
        for keyword, species_code in BioEntityExtractor.SPECIES_HINTS.items():
            if keyword in query_lower:
                return species_code
        return None
    
    @staticmethod
    def _allowed_dbs_for_text(text: str) -> Optional[Set[str]]:
        """Determine allowed databases based on text content."""
        if BioEntityExtractor.ENZYME_PATTERN.search(text.lower()):
            return {"enzyme", "ortholog", "ec"}
        return None  # None means no restriction
    
    @staticmethod
    def _get_allowed_dbs_for_id(id_text: str) -> Set[str]:
        """Get allowed databases for an explicit ID based on its prefix."""
        if id_text.startswith(("K", "k")):
            return {"ortholog"}
        elif id_text.startswith(("C", "c")):
            return {"compound"}
        elif id_text.startswith(("R", "r")):
            return {"reaction"}
        else:
            return {"ec", "enzyme"}
    
    @staticmethod
    def _normalize_ec(text: str) -> str:
        """Normalize EC number by removing EC prefix and cleaning up."""
        normalized = text.strip()
        normalized = re.sub(r'^(?i:EC)[:\s]*', '', normalized)
        return normalized
    
    @staticmethod
    def regex_enzyme_mentions(text: str) -> List[Dict]:
        """
        Extract enzyme mentions using regex only (utility function).
        
        Returns:
            List of dicts with keys: text, start, end
        """
        return [
            {"text": m.group(0), "start": m.start(), "end": m.end()}
            for m in BioEntityExtractor.ENZYME_PATTERN.finditer(text)
        ]
