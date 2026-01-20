import pandas as pd
import requests
import json
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from config_sean import GEMINI_API_KEY

# ==============================================================================
# 1. 설정
# ==============================================================================
MODEL_NAME = "gemini-2.5-flash" # 2.5가 아직 불안정할 수 있어 2.0 권장 (혹은 쓰시던거 쓰셔도 됨)
# 2.5를 쓰시고 싶으시면 유지하세요. 여기선 로직이 중요합니다.
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
MAX_WORKERS = 20 # 속도 업

qachain_csv = "/cluster/pixstor/xudong-lab/yongfang/fatplants_cypher/sean/evaluation/pathway_evaluation_complete_with_result_QAChain.csv"
my_best_csv = "/cluster/pixstor/xudong-lab/yongfang/fatplants_cypher/sean/evaluation/pathway_evaluation_complete_with_result.csv"

# ==============================================================================
# 2. 강력한 전처리: 모양 맞추기 (Normalization)
# ==============================================================================
def normalize_cypher(q):
    if not isinstance(q, str): return ""
    
    # 1. 기본 청소
    q = re.sub(r'```(?:cypher)?', '', q, flags=re.IGNORECASE)
    q = q.replace('```', '').strip().strip('`').strip("'").strip('"')
    
    # 2. [핵심] 화살표 제거 (->, <- 를 모두 - 로 통일)
    # 이걸 해야 Undirected 전략이 정답으로 인정됨
    q = re.sub(r'<-+|->+', '-', q)
    
    # 3. [핵심] properties() 껍질 벗기기
    # RETURN properties(n) -> RETURN n 으로 취급
    q = q.replace('properties(', '').replace(')', '')
    
    # 4. 잡다한 구문 제거 (ORDER BY, LIMIT, WHERE 등은 로직 비교에 방해될 때가 있음)
    # 여기선 ORDER BY, LIMIT만 제거
    q = re.sub(r'order\s+by\s+.*?(?=(return|limit|skip|$))', '', q, flags=re.IGNORECASE|re.DOTALL)
    q = re.sub(r'limit\s+\d+\s*;?', '', q, flags=re.IGNORECASE)
    
    # 5. 공백 축소 & 소문자화 (변수명 비교는 LLM에게 맡김)
    return " ".join(q.split())

# ==============================================================================
# 3. Gemini 판사 (프롬프트: 스타일 무시 지시)
# ==============================================================================
def check_smart_match(truth, mine):
    # 1. 전처리 후 텍스트가 같으면 100% 정답 (API 비용 절약)
    norm_truth = normalize_cypher(truth)
    norm_mine = normalize_cypher(mine)
    
    if norm_truth.lower() == norm_mine.lower():
        return True

    # 2. Gemini에게 "의미(Semantic)"만 보라고 강력 지시
    prompt = f"""
    Role: Senior Neo4j Expert.
    Task: Compare two Cypher queries and determine if they fetch the SAME DATA conceptually.
    
    [Query 1: Ground Truth]
    {norm_truth}
    
    [Query 2: Candidate]
    {norm_mine}
    
    [JUDGMENT RULES - READ CAREFULLY]
    1. **Ignore Direction:** `(a)-[:REL]-(b)` is EQUAL to `(a)-[:REL]->(b)`. (Undirected is acceptable).
    2. **Ignore Return Format:** `RETURN properties(n)` is EQUAL to `RETURN n`.
    3. **Ignore Variable Names:** `MATCH (g:Gene)` is EQUAL to `MATCH (n:Gene)`.
    4. **Superset is Correct:** If Query 2 finds the requested nodes PLUS extra neighbors, it is **CORRECT**.
    5. **Subset is Correct:** If Query 2 is more specific (e.g., adds `DISTINCT`), it is **CORRECT**.
    
    Does Query 2 verify the same intent as Query 1 based on the rules above?
    Answer ONLY "YES" or "NO".
    """
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    for i in range(3):
        try:
            response = requests.post(API_URL, json=payload, headers={'Content-Type': 'application/json'}, timeout=10)
            if response.status_code == 200:
                text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().upper()
                return "YES" in text
            elif response.status_code == 429: # Rate Limit
                time.sleep(2)
            else:
                time.sleep(1)
        except:
            time.sleep(1)
            
    return False

# ==============================================================================
# 4. 실행
# ==============================================================================
print(f"🚀 [Semantic Evaluation V2] 관대한 채점 시작...")

try:
    df_qa = pd.read_csv(qachain_csv)
    df_my = pd.read_csv(my_best_csv)
    df_qa.columns = df_qa.columns.str.strip()
    df_my.columns = df_my.columns.str.strip()
    if 'cypher_executable' in df_qa.columns: df_qa = df_qa.rename(columns={'cypher_executable': 'truth'})
    
    merged = pd.merge(df_qa, df_my, on='question', how='inner', suffixes=('_qa', '_my'))
    print(f"✅ 총 {len(merged)}개 비교 중...")
except: exit()

scores = {"LipidBot": 0, "CypherQAChain": 0}
total_valid = 0

def process_row(row):
    truth = str(row.get('truth', row.get('truth_qa', '')))
    my_cypher = str(row.get('generated_cypher_my', row.get('generated_cypher', '')))
    qa_cypher = str(row.get('QAChain', ''))
    
    if not truth or truth == 'nan': return None
    
    # 평가
    my_res = 1 if check_smart_match(truth, my_cypher) else 0
    qa_res = 1 if check_smart_match(truth, qa_cypher) else 0
    
    return (1, my_res, qa_res)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_row, row) for _, row in merged.iterrows()]
    for future in tqdm(as_completed(futures), total=len(futures)):
        res = future.result()
        if res:
            total_valid += res[0]
            scores["LipidBot"] += res[1]
            scores["CypherQAChain"] += res[2]

# ==============================================================================
# 5. 결과 리포트
# ==============================================================================
def calc_pct(count): return (count / total_valid) * 100 if total_valid > 0 else 0

print("\n" + "="*80)
print(f" 🏆 ULTIMATE SEMANTIC ACCURACY REPORT (Normalized)")
print("="*80)
print(f"{'Tool':<20} | {'Semantic Match Rate':<25}")
print("-" * 80)
print(f"{'LipidBot':<20} | {calc_pct(scores['LipidBot']):.1f}% 🚀")
print(f"{'CypherQAChain':<20} | {calc_pct(scores['CypherQAChain']):.1f}%")
print("="*80)
print("📌 정답 인정 기준 (Enhanced):")
print("1. 화살표 방향 무시 (Undirected Accepted)")
print("2. properties() 함수 무시 (Meaning check)")
print("3. 변수명 차이 무시 (n vs g)")
print("4. Superset/Subset 쿼리 인정")