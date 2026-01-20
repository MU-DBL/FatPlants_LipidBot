#!/usr/bin/env python3
"""
Test Set Generator for Pathway Knowledge Graph Evaluation
Fills questions with real parameters and generates executable Cypher queries
"""

import pandas as pd
import re
import json
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Sample real IDs from database (update with your actual IDs)
SAMPLE_IDS = {
    'gene': ['gmx:100217331', 'ats:109745953', 'gmx:100037478', 'ath:AT1G01040', 'ath:AT1G01050',
             'ats:109733199', 'ats:109761713', 'ath:AT1G54220', 'ath:AT2G06990', 'ath:AT5G61540'],
    'ec': ['EC:6.4.1.2', 'EC:2.3.1.199', 'EC:1.3.1.10', 'EC:4.2.1.59', 'EC:1.14.13.1',
           'EC:1.3.1.9', 'EC:6.2.1.3', 'EC:2.1.1.369', 'EC:1.3.1.21', 'EC:1.13.11.33'],
    'compound': ['C00141', 'C01181', 'C00422', 'C00052', 'C06157', 'C06000',
                 'C00164', 'C08813', 'C03069', 'C00183'],
    'pathway': ['path:ats01040', 'path:gmx00280', 'path:gmx00061', 'path:ath00061', 'path:ats00603',
                'path:ats00280', 'path:gmx01040', 'path:ath00280', 'path:gmx00603', 'path:ath00603'],
    'reaction': ['R02814', 'R04957', 'R07449', 'R00200', 'R07764', 'R01280',
                 'R06870', 'R06871', 'R10826', 'R03777'],
    'ortholog': ['K10246', 'K09458', 'K00059', 'K11262', 'K00249', 'K01897',
                 'K00626', 'K13356', 'K09754', 'K00382']
}

PATHWAY_NAMES = {
    'glycerophospholipid metabolism': 'path:ath00564',
    'lysine degradation': 'path:ath00310',
    'fatty acid biosynthesis': 'path:ath00061',
    'fatty acid elongation': 'path:ath00062',
    'steroid biosynthesis': 'path:ath00100',
    'glycolysis': 'path:ath00010',
    'tca cycle': 'path:ath00020',
    'citric acid cycle': 'path:ath00020'
}

class TestSetGenerator:
    def __init__(self, sample_ids: Dict[str, List[str]]):
        self.sample_ids = sample_ids
        self.id_counter = defaultdict(int)
    
    def detect_entity_type(self, question: str) -> Optional[str]:
        """Detect entity type that needs an ID parameter (the subject being queried)"""
        question_lower = question.lower()
        
        # Priority patterns - identify the entity that GETS the ID parameter
        
        # Existence/property checks: "Does gene X exist?" -> gene gets ID
        if re.search(r'(?:does|is|do|check)\s+(?:gene|enzyme|compound|pathway|reaction|ortholog)', question_lower):
            if 'gene' in question_lower:
                return 'gene'
            elif 'enzyme' in question_lower or 'ec' in question_lower:
                return 'ec'
            elif 'compound' in question_lower:
                return 'compound'
            elif 'pathway' in question_lower:
                return 'pathway'
            elif 'reaction' in question_lower:
                return 'reaction'
            elif 'ortholog' in question_lower or 'ko' in question_lower:
                return 'ortholog'
        
        # "gene X encodes..." -> gene gets ID
        if re.search(r'gene\s+.*\s+(?:encode|belong|produce)', question_lower):
            return 'gene'
        
        # "enzyme X catalyzes..." or "reactions catalyzed by enzyme" -> enzyme gets ID
        if re.search(r'enzyme\s+.*\s+catalyze', question_lower):
            return 'ec'
        if re.search(r'catalyzed by\s+(?:enzyme|ec)', question_lower):
            return 'ec'
        if re.search(r'reactions?\s+.*\s+enzyme', question_lower) and not re.search(r'which genes', question_lower):
            return 'ec'
        
        # "compound X is substrate..." or "reactions using compound X" -> compound gets ID
        if re.search(r'compound\s+.*\s+(?:substrate|is|as)', question_lower):
            return 'compound'
        if re.search(r'(?:use|using|consume)\s+compound', question_lower):
            return 'compound'
        if re.search(r'reactions?\s+.*\s+compound', question_lower) and 'from' not in question_lower:
            return 'compound'
        
        # "pathway X contains..." or "in pathway X" -> pathway gets ID
        if re.search(r'(?:in|from|for)\s+pathway', question_lower):
            return 'pathway'
        if re.search(r'pathway\s+.*\s+contain', question_lower):
            return 'pathway'
        
        # "reaction X produces..." -> reaction gets ID
        if re.search(r'reaction\s+.*\s+produce', question_lower):
            return 'reaction'
        if re.search(r'(?:from|in|by)\s+reaction', question_lower):
            return 'reaction'
        
        # "ortholog X..." -> ortholog gets ID
        if re.search(r'ortholog\s+.*\s+(?:catalyze|has|correspond)', question_lower):
            return 'ortholog'
        if re.search(r'(?:in|of|for)\s+ortholog', question_lower):
            return 'ortholog'
        if re.search(r'\bko\s+\w', question_lower):
            return 'ortholog'
        
        # Reverse queries: "which genes encode enzyme X" -> enzyme gets ID (not gene)
        if re.search(r'which\s+genes?\s+.*\s+enzyme', question_lower):
            return 'ec'
        if re.search(r'which\s+reactions?\s+.*\s+compound', question_lower):
            return 'compound'
        if re.search(r'which\s+pathways?\s+.*\s+reaction', question_lower):
            return 'reaction'
        if re.search(r'which\s+enzymes?\s+.*\s+ortholog', question_lower):
            return 'ortholog'
        
        # Fallback patterns (first match wins)
        fallback_patterns = [
            (r'AT\dG\d+', 'gene'),
            (r'EC:\d+\.\d+\.\d+\.\d+', 'ec'),
            (r'C\d{5}', 'compound'),
            (r'ko\d+', 'pathway'),
            (r'R\d{5}', 'reaction'),
            (r'K\d{5}', 'ortholog'),
            (r'\bgenes?\b', 'gene'),
            (r'\benzymes?\b', 'ec'),
            (r'\bcompounds?\b', 'compound'),
            (r'\bpathways?\b', 'pathway'),
            (r'\breactions?\b', 'reaction'),
            (r'\borthologs?\b', 'ortholog')
        ]
        
        for pattern, entity_type in fallback_patterns:
            if re.search(pattern, question_lower):
                return entity_type
        
        return None
    
    def extract_id_from_question(self, question: str) -> Optional[str]:
        """Extract ID from question if present"""
        id_patterns = [
            r'(?:gmx|ats|ath|AT):\d+|AT\dG\d+',  # Gene IDs: gmx:100217331, ats:109745953, ath:AT1G54220, AT1G01040
            r'EC:\d+\.\d+\.\d+\.\d+',              # EC numbers: EC:6.4.1.2
            r'C\d{5}',                              # Compounds: C00141
            r'path:[a-z]{3}\d{5}',                 # Pathways: path:ats01040, path:gmx00280
            r'R\d{5}',                              # Reactions: R02814
            r'K\d{5}'                               # Orthologs: K10246
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, question)
            if match:
                return match.group()
        return None
    
    def get_next_sample_id(self, entity_type: str) -> str:
        """Get next sample ID for entity type (cycling through list)"""
        if entity_type not in self.sample_ids:
            return ''
        
        ids = self.sample_ids[entity_type]
        idx = self.id_counter[entity_type] % len(ids)
        self.id_counter[entity_type] += 1
        return ids[idx]
    
    def fill_question_with_parameters(self, question: str, question_num: int) -> str:
        """Fill question with real parameters - ALWAYS use sample IDs from pool"""
        question_lower = question.lower()
        
        # Questions that don't need IDs (count, list all, etc.)
        if any(phrase in question_lower for phrase in [
            'list all', 'show me all', 'count', 'how many',
            'get all', 'display all'
        ]):
            # But check if "what are" is asking about a specific entity
            if not any(word in question_lower for word in ['does', 'is', 'do', 'check']):
                return question
        
        # Detect entity type
        entity_type = self.detect_entity_type(question)
        if not entity_type:
            return question
        
        # Get sample ID from pool
        sample_id = self.get_next_sample_id(entity_type)
        if not sample_id:
            return question
        
        # Remove any existing ID from the question first
        question = re.sub(r'(?:gmx|ats|ath|AT):\d+', '<ID>', question)  # Gene IDs with prefix
        question = re.sub(r'AT\dG\d+', '<ID>', question)                # Gene IDs like AT1G01040
        question = re.sub(r'EC:\d+\.\d+\.\d+\.\d+', '<ID>', question)  # EC numbers
        question = re.sub(r'C\d{5}', '<ID>', question)                  # Compounds
        question = re.sub(r'path:[a-z]{3}\d{5}', '<ID>', question)     # Pathways with path: prefix
        question = re.sub(r'ko\d+', '<ID>', question)                   # Old pathway format
        question = re.sub(r'R\d{5}', '<ID>', question)                  # Reactions
        question = re.sub(r'K\d{5}', '<ID>', question)                  # Orthologs
        
        # For pathway names, replace with sample pathway ID
        for name, _ in PATHWAY_NAMES.items():
            if name in question_lower:
                if entity_type == 'pathway':
                    question = re.sub(name, sample_id, question, flags=re.IGNORECASE)
                    return question
        
        # Now insert the sample ID
        # First check if we have a placeholder
        if '<ID>' in question:
            question = question.replace('<ID>', sample_id)
        else:
            # Insert ID based on entity type - universal approach
            entity_keywords = {
                'gene': r'\bgene\b',
                'ec': r'\benzyme\b',
                'compound': r'\bcompound\b',
                'pathway': r'\bpathway\b',
                'reaction': r'\breaction\b',
                'ortholog': r'\bortholog\b|\bKO\b'
            }
            
            if entity_type in entity_keywords:
                pattern = entity_keywords[entity_type]
                question = re.sub(pattern, f'{pattern.strip("\\\\b")} {sample_id}', question, count=1, flags=re.IGNORECASE)
                # Clean up the pattern artifacts
                question = question.replace('\\b', '').replace('\\', '').replace('|bKOb', '')
        
        # Clean up extra spaces and formatting
        question = re.sub(r'\s+', ' ', question).strip()
        question = re.sub(r'\s+\?', '?', question)
        question = re.sub(r'\s+,', ',', question)
        
        return question
    
    def fix_nodetype_cypher(self, cypher: str, question: str) -> str:
        """Replace NodeType with specific node label"""
        if ':NodeType' not in cypher:
            return cypher
        
        entity_type = self.detect_entity_type(question)
        label_map = {
            'gene': 'Gene',
            'ec': 'EC',
            'compound': 'Compound',
            'pathway': 'Pathway',
            'reaction': 'Reaction',
            'ortholog': 'Ortholog'
        }
        
        if entity_type and entity_type in label_map:
            return cypher.replace(':NodeType', f':{label_map[entity_type]}')
        
        return cypher
    
    def extract_parameters(self, question: str, cypher: str) -> Dict[str, Any]:
        """Extract all parameters from question"""
        params = {}
        
        # Extract ID
        id_value = self.extract_id_from_question(question)
        if id_value:
            params['id'] = id_value
        
        # Extract pathway name and ID
        question_lower = question.lower()
        for name, pathway_id in PATHWAY_NAMES.items():
            if name in question_lower:
                params['name'] = name.title()
                if 'id' not in params:
                    params['id'] = pathway_id
                break
        
        # Extract length for path queries
        if '$length' in cypher:
            length_match = re.search(r'(\d+)', question)
            if length_match:
                params['length'] = int(length_match.group(1))
        
        return params
    
    def generate_executable_cypher(self, cypher_pattern: str, params: Dict[str, Any]) -> str:
        """Generate executable Cypher with parameters filled in"""
        cypher = cypher_pattern
        
        # Replace parameters
        for param_name, param_value in params.items():
            placeholder = f'${param_name}'
            if isinstance(param_value, str):
                cypher = cypher.replace(placeholder, f"'{param_value}'")
            else:
                cypher = cypher.replace(placeholder, str(param_value))
        
        return cypher
    
    def determine_answer_type(self, cypher: str) -> str:
        """Determine expected answer type from Cypher query"""
        cypher_lower = cypher.lower()
        
        if 'count(' in cypher_lower:
            return 'integer'
        elif 'exists(' in cypher_lower:
            return 'boolean'
        elif 'properties(' in cypher_lower:
            return 'dict'
        elif 'path' in cypher_lower and '=' in cypher:
            return 'path'
        elif 'limit' in cypher_lower:
            return 'list'
        else:
            return 'list'
    
    def generate_expected_answer_template(self, cypher: str, answer_type: str) -> str:
        """Generate template for expected answer"""
        templates = {
            'integer': '{"type": "integer", "value": <count>}',
            'boolean': '{"type": "boolean", "value": true/false}',
            'dict': '{"type": "dict", "properties": {...}}',
            'path': '{"type": "path", "nodes": [...], "relationships": [...]}',
            'list': '{"type": "list", "items": [...]}'
        }
        return templates.get(answer_type, '{"type": "unknown"}')
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataframe"""
        results = []
        
        for idx, row in df.iterrows():
            # Step 1: Fill question with parameters
            question_filled = self.fill_question_with_parameters(
                row['question'], 
                row['question_number']
            )
            
            # Step 2: Fix NodeType in Cypher pattern (use ORIGINAL question for entity detection)
            cypher_corrected = self.fix_nodetype_cypher(
                row['cypher_pattern'],
                row['question']  # Use original question, not filled
            )
            
            # Step 3: Extract parameters
            params = self.extract_parameters(question_filled, cypher_corrected)
            
            # Step 4: Generate executable Cypher
            cypher_executable = self.generate_executable_cypher(
                cypher_corrected,
                params
            )
            
            # Step 5: Determine answer type
            answer_type = self.determine_answer_type(cypher_executable)
            
            # Step 6: Generate answer template
            answer_template = self.generate_expected_answer_template(
                cypher_executable,
                answer_type
            )
            
            results.append({
                **row.to_dict(),
                'question_filled': question_filled,
                'cypher_pattern_corrected': cypher_corrected,
                'parameters': json.dumps(params),
                'cypher_executable': cypher_executable,
                'expected_answer_type': answer_type,
                'expected_answer_template': answer_template
            })
        
        return pd.DataFrame(results)

def main():
    # Load data
    print("Loading data...")
    df = pd.read_csv('/mnt/user-data/uploads/pathway_evaluation_complete.csv')
    
    # Create generator
    generator = TestSetGenerator(SAMPLE_IDS)
    
    # Process dataframe
    print("Processing questions and generating test set...")
    df_processed = generator.process_dataframe(df)
    
    # Save results
    output_path = '/mnt/user-data/outputs/pathway_test_set.csv'
    df_processed.to_csv(output_path, index=False)
    
    print("\n" + "=" * 100)
    print("TEST SET GENERATION COMPLETE")
    print("=" * 100)
    print(f"\n✅ Saved to: {output_path}")
    print(f"Total test cases: {len(df_processed)}")
    
    # Statistics
    print("\n📊 STATISTICS:")
    print(f"  Questions with parameters: {df_processed['parameters'].apply(lambda x: x != '{}').sum()}")
    print(f"  Questions without parameters: {df_processed['parameters'].apply(lambda x: x == '{}').sum()}")
    
    print("\n📋 ANSWER TYPE DISTRIBUTION:")
    answer_type_counts = df_processed['expected_answer_type'].value_counts()
    for answer_type, count in answer_type_counts.items():
        print(f"  {answer_type}: {count}")
    
    # Show samples
    print("\n" + "=" * 100)
    print("SAMPLE TEST CASES")
    print("=" * 100)
    
    samples = df_processed.head(10)
    for idx, row in samples.iterrows():
        print(f"\n{'='*100}")
        print(f"Q{row['question_number']}: {row['template_id']} - {row['category']}")
        print(f"{'='*100}")
        print(f"Original Question:  {row['question']}")
        print(f"Filled Question:    {row['question_filled']}")
        print(f"Parameters:         {row['parameters']}")
        print(f"Cypher Template:    {row['cypher_pattern']}")
        print(f"Cypher Corrected:   {row['cypher_pattern_corrected']}")
        print(f"Cypher Executable:  {row['cypher_executable']}")
        print(f"Answer Type:        {row['expected_answer_type']}")
        print(f"Answer Template:    {row['expected_answer_template']}")
    
    # Show comparison before/after
    print("\n" + "=" * 100)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 100)
    
    comparison_cases = [1, 6, 16, 26, 38, 98]
    for qnum in comparison_cases:
        row = df_processed[df_processed['question_number'] == qnum].iloc[0]
        print(f"\nQ{qnum}:")
        print(f"  Before: {row['question']}")
        print(f"  After:  {row['question_filled']}")
        print(f"  Cypher: {row['cypher_executable'][:80]}...")

if __name__ == '__main__':
    main()
