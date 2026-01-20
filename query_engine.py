import time
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import concurrent.futures # 병렬 처리용 라이브러리

# 로컬 모듈 임포트
from ac import load_cache
from data_service import LLMProvider
from llm_cypher_query_generator import LLMCypherQueryGenerator
from llm_entity_extractor import LLMBioEntityExtractor
from llm_factory import LLMFactory
from llm_intent_recognizer import LLMIntentRecognizer
from entity_extractor import BioEntityExtractor

try:
    from config_sean import GEMINI_API_KEY, GEMINI_MODEL_NAME
    print(f"Loaded configuration from config_sean.py (Model: {GEMINI_MODEL_NAME})")
except ImportError:
    print("Error: 'config_sean.py' not found. Using defaults.")
    GEMINI_API_KEY = ""
    GEMINI_MODEL_NAME = "gemini-2.5-flash"

class QueryEngine:
    
    def __init__(
        self, 
        entity_extractor: BioEntityExtractor,
        intent_recognizer: LLMIntentRecognizer,
        llm_engine: LLMCypherQueryGenerator,
    ):
        self.entity_extractor = entity_extractor
        self.intent_recognizer = intent_recognizer
        self.llm_engine = llm_engine
        
    def query(self, question: str) -> Tuple[str, str]:
        """
        Returns: (Cypher Query, Generation Source)
        """
        # Step 1: Entity Extraction
        entities = self.entity_extractor.extract_mentions(question) 
        
        # Step 2: Intent Recognition
        intent = self.intent_recognizer.recognize(question, entities)
        
        # Step 3: Cypher Query Generation (Templates -> Fallback)
        cypher, source = self._llm_query(question, intent, entities)

        return cypher, source
    
    def _llm_query(self, question: str, intent: Any, entities: List[Dict]) -> Tuple[str, str]:
        try:
            cypher, source = self.llm_engine.generate_query(
                question=question,
                intent=intent,
                entities=entities
            )
            return cypher, source
        except Exception as e:
            # 병렬 처리 중 에러 발생 시 로그 출력이 꼬일 수 있으므로 조용히 처리
            return "", "Error"

# ==========================================
# [Parallel Processing Helper]
# ==========================================
def process_single_question(args):
    """
    병렬 처리를 위해 단일 질문을 처리하는 헬퍼 함수
    args: (index, question_text, query_engine_instance)
    """
    idx, question, query_engine = args
    try:
        # 실제 쿼리 생성 수행
        cypher, source = query_engine.query(question)
        return idx, cypher, source
    except Exception:
        return idx, "", "Failed"

def main():
    print("Loading resources...")
    
    try:
        A, alias_map = load_cache("ac_kegg.pkl")
    except Exception as e:
        print(f"Warning: Failed to load cache 'ac_kegg.pkl'. Details: {e}")
        A, alias_map = None, None

    # Gemini 사용 설정
    print(f"Initializing LLM with Gemini...")
    if not GEMINI_API_KEY:
        print("CRITICAL ERROR: GEMINI_API_KEY is empty!")
        return

    llm = LLMFactory.create_llm(
        provider=LLMProvider.GEMINI, 
        model_name=GEMINI_MODEL_NAME, 
        api_key=GEMINI_API_KEY
    )
    
    llm_entity_extractor = LLMBioEntityExtractor(llm)
    entity_extractor = BioEntityExtractor(A, alias_map, llm_entity_extractor)
    llm_intent_recognizer = LLMIntentRecognizer(llm)
    llm_query_engine = LLMCypherQueryGenerator(llm)

    query_engine = QueryEngine(
        entity_extractor=entity_extractor,
        intent_recognizer=llm_intent_recognizer,
        llm_engine=llm_query_engine
    )

    input_csv_path = "evaluation/pathway_evaluation_complete.csv"
    output_csv_path = "evaluation/pathway_evaluation_complete_with_result.csv"

    try:
        df = pd.read_csv(input_csv_path)
        total_questions = len(df)
        print(f"Loaded {total_questions} questions. Starting Parallel Generation (Workers=20)...")

        # 결과 저장용 배열 (인덱스 순서 유지를 위해 미리 할당)
        results_cypher = [""] * total_questions
        results_source = [""] * total_questions

        # 병렬 처리 설정 (max_workers=20)
        # 주의: 20개 스레드가 동시에 LLM을 호출하므로 속도가 매우 빨라집니다.
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # 작업 예약 (Index, Question, Engine)
            futures = {
                executor.submit(process_single_question, (idx, row['question'], query_engine)): idx 
                for idx, row in df.iterrows()
            }
            
            # tqdm으로 진행 상황 표시 (순서 상관없이 완료되는 대로 업데이트)
            for future in tqdm(concurrent.futures.as_completed(futures), total=total_questions, desc="Generating", unit="q"):
                idx, cypher, source = future.result()
                results_cypher[idx] = cypher
                results_source[idx] = source

        # 결과 저장
        df['generated_cypher'] = results_cypher
        df['generation_type'] = results_source  # Template vs Fallback 정보 저장
        
        df.to_csv(output_csv_path, index=False)
        print(f"\n✅ All Done! Results saved to {output_csv_path}")
        print(f"Check 'generation_type' column to see Template vs Fallback usage.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()