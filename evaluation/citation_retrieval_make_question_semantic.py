#!/usr/bin/env python3
import os
import json
import re
import pandas as pd
import sys
import logging
import threading
from time import sleep
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # tqdm library (pip install tqdm required)
from google import genai
from google.genai import types

# =======================
# ⚙️ Configuration
# =======================
INPUT_CSV = "sampled_800papers.csv"
OUTPUT_CSV = "generated_questions_semantic_full.csv"  # Final output filename
API_KEY = "AIzaSyAdMUQzMbeHZBZlMt2LCcm_dTsqlMjMSXc"
LOG_FILE = "generation_full.log"
LLM_MODEL = "gemini-2.5-pro"
MAX_WORKERS = 10  # Requested number of parallel workers

# =======================
# 📝 Logging configuration (Thread-safe)
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        # Minimize console output to avoid conflict with tqdm (use tqdm.write if needed)
    ]
)

# =======================
# 📦 Data class
# =======================
@dataclass
class Citation:
    pmid: str
    title: str
    abstract: str
    category_id: str
    category_name: str

# =======================
# Question validation class
# =======================
class QuestionValidator:
    @staticmethod
    def validate_entities_present(entities_used: str, abstract: str) -> bool:
        return True
    
    @staticmethod
    def validate_answer_specific(answer: str) -> bool:
        if not answer or len(answer.strip()) < 10:
            return False
        has_percentage = bool(re.search(r'\d+\.?\d*\s*%', answer))
        has_unit = bool(re.search(r'\d+\.?\d*\s*(μM|mM|nM|°C|mg|g|mol|fold|kb|bp|kDa|ng|pg|cm|mm|m|h|min|sec)', answer))
        has_gene = bool(re.search(r'[A-Z]{2,}[0-9]?', answer)) 
        capital_words = re.findall(r'\b[A-Z][a-zA-Z0-9\-]+\b', answer)
        is_descriptive_and_specific = (len(answer.split()) >= 12) and (len(capital_words) >= 1)
        is_methodological = bool(re.search(r'(using|via|analyzed|method|assay|performed|detected|measured)', answer.lower()))
        return has_percentage or has_unit or has_gene or is_descriptive_and_specific or is_methodological
    
    @staticmethod
    def validate_not_vague(question: str) -> bool:
        vague_words = ['various', 'several', 'many', 'some', 'different', 'multiple', 'certain']
        return not any(word in question.lower() for word in vague_words)

    @staticmethod
    def validate_sufficient_content(abstract: str) -> bool:
        return len(abstract.split()) > 80

    @classmethod
    def validate_question(cls, question_data: Dict, abstract: str) -> Tuple[bool, Dict]:
        checks = {
            "entities_present": cls.validate_entities_present(question_data.get('entities_used', ''), abstract),
            "answer_specific": cls.validate_answer_specific(question_data.get('answer', '')),
            "not_vague": cls.validate_not_vague(question_data.get('question', '')),
            "sufficient_content": cls.validate_sufficient_content(abstract),
            "has_answer": len(question_data.get('answer', '').strip()) > 20,
            "has_question": len(question_data.get('question', '').strip()) > 10
        }
        return all(checks.values()), checks

# =======================
# 🤖 LLM prompt and generation
# =======================
def create_prompt(citation: Citation) -> str:
    return f"""
You are an expert in plant biology and lipids.
Your task is to generate scientific questions based on the provided abstract.

**[CRITICAL INSTRUCTIONS]**
1. **NO EXACT COPYING**: Do NOT use exact sentences or phrases from the abstract.
2. **USE SYNONYMS & PARAPHRASE**: You MUST paraphrase key terms and concepts.
3. **SEMANTIC UNDERSTANDING**: Test understanding of meaning, not just keywords.
4. **Answerability**: Ensure the answer is derivable from the abstract.

**Citation:**
Title: {citation.title}
Abstract: {citation.abstract}
PMID: {citation.pmid}
Category: {citation.category_name}

**TASK:** Create exactly 2 questions based ONLY on the abstract.
1. FACTUAL QUESTION (Easy): Combine 2 entities. Direct answer but paraphrased.
2. NUMERICAL/METHODOLOGICAL QUESTION (Medium): Ask about specific results OR methods.

**OUTPUT FORMAT:** JSON Array with 2 objects.
[{{...}}, {{...}}]
"""

def generate_questions(citation: Citation, client: genai.Client) -> List[Dict]:
    prompt = create_prompt(citation)
    json_schema = types.Schema(
        type=types.Type.ARRAY,
        items=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "pmid": types.Schema(type=types.Type.STRING),
                "question_type": types.Schema(type=types.Type.STRING),
                "question": types.Schema(type=types.Type.STRING),
                "answer": types.Schema(type=types.Type.STRING),
                "entities_used": types.Schema(type=types.Type.STRING),
                "difficulty": types.Schema(type=types.Type.STRING),
                "topic": types.Schema(type=types.Type.STRING),
            },
            required=["pmid", "question_type", "question", "answer", "entities_used", "difficulty", "topic"]
        )
    )

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=json_schema
                )
            )
            if not response.text:
                raise ValueError("Empty response")
            json_text = response.text.strip()
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', json_text, re.DOTALL)
            json_to_load = json_match.group(0) if json_match else json_text
            questions = json.loads(json_to_load)
            
            if isinstance(questions, list):
                for q in questions:
                    q['category_id'] = citation.category_id
                    q['category_name'] = citation.category_name
                    q['title'] = citation.title
                    q['abstract_word_count'] = len(citation.abstract.split())
                return questions
        except Exception:
            sleep(1)
    return []

# =======================
# 🧵 Worker function (executed in threads)
# =======================
result_lock = threading.Lock()  # Prevent collisions during CSV writing

def process_single_citation(row_tuple, client):
    idx, row = row_tuple
    citation = Citation(
        pmid=str(row['pmid']),
        title=row['title'],
        abstract=row['abstract'],
        category_id=row['category_id'],
        category_name=row['category_name']
    )
    
    valid_questions = []
    # Try up to 2 times
    for attempt in range(2):
        generated = generate_questions(citation, client)
        if generated:
            temp_valid = []
            for q in generated:
                is_valid, _ = QuestionValidator.validate_question(q, citation.abstract)
                if is_valid:
                    temp_valid.append(q)
            
            if len(temp_valid) == 2:
                valid_questions = temp_valid
                break
            if len(temp_valid) > len(valid_questions):
                valid_questions = temp_valid
        
        if attempt == 0:
            sleep(0.5)

    if len(valid_questions) == 2:
        return valid_questions
    return None

# =======================
# 🚀 Main execution function
# =======================
def main():
    if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY":
        print("❌ API Key Error")
        sys.exit(1)
        
    # In a multithreading environment, a client can be shared or created per thread.
    # Google GenAI Client is thread-safe, so it is shared here.
    client = genai.Client(api_key=API_KEY)

    try:
        df = pd.read_csv(INPUT_CSV)
        df.columns = df.columns.str.lower()
    except FileNotFoundError:
        print(f"File Error: Input file '{INPUT_CSV}' not found.")
        sys.exit(1)

    completed_pmids = set()
    all_questions = []

    # Resume capability
    if os.path.exists(OUTPUT_CSV):
        try:
            df_out = pd.read_csv(OUTPUT_CSV)
            all_questions = df_out.to_dict('records')
            pmid_counts = df_out['pmid'].value_counts()
            completed_pmids = set(pmid_counts[pmid_counts >= 2].index)
            print(f"📂 Loaded existing file. Completed papers: {len(completed_pmids)}")
        except pd.errors.EmptyDataError:
            pass
    
    # Select papers to process
    df_to_process = df[~df['pmid'].isin(completed_pmids)].reset_index(drop=True)
    total_tasks = len(df_to_process)
    
    print(f"🚀 Starting generation for {total_tasks} papers with {MAX_WORKERS} workers...")

    rows_to_process = list(df_to_process.iterrows())
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_citation, r, client): r for r in rows_to_process}
        
        for future in tqdm(as_completed(futures), total=total_tasks, desc="Generating", unit="paper"):
            result = future.result()
            
            if result:
                with result_lock:
                    all_questions.extend(result)
                    # To reduce I/O overhead, save periodically rather than every time.
                    # Here, save every 20 questions for safety.
                    if len(all_questions) % 20 == 0:
                        pd.DataFrame(all_questions).to_csv(OUTPUT_CSV, index=False)

    # Final save
    if all_questions:
        df_final = pd.DataFrame(all_questions)
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ All done! Saved to {OUTPUT_CSV}")
        print(f"Total questions generated: {len(df_final)}")
    else:
        print("\n⚠️ No questions were generated.")

if __name__ == "__main__":
    main()
