import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
import re
import signal
import collections
from config_sean import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ==============================================================================
# 1. 설정
# ==============================================================================
FETCH_LIMIT = 10000
QUERY_TIMEOUT = 10

qachain_csv = "/cluster/pixstor/xudong-lab/yongfang/fatplants_cypher/sean/evaluation/pathway_evaluation_complete_with_result_QAChain.csv"
my_best_csv = "/cluster/pixstor/xudong-lab/yongfang/fatplants_cypher/sean/evaluation/pathway_evaluation_complete_with_result.csv"
output_report_csv = "evaluation_report.csv" 

# ==============================================================================
# 2. 유틸리티 (V6.0 핵심 로직 유지)
# ==============================================================================
class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException("Timeout")

def unwrap_count_query(query):
    if not isinstance(query, str): return query
    if "count(" not in query.lower(): return None
    return re.sub(r'count\s*\(\s*(.*?)\s*\)', r'\1', query, flags=re.IGNORECASE)

def clean_cypher(q):
    if not isinstance(q, str): return ""
    q = re.sub(r'```(?:cypher)?', '', q, flags=re.IGNORECASE)
    q = q.strip().strip('`').strip("'").strip('"')
    return q

def extract_ids_from_record(record_values):
    extracted = set()
    def recursive_extract(item):
        if hasattr(item, 'items') or isinstance(item, dict):
            d = dict(item)
            for key in ['id', 'name', 'title', 'symbol']:
                if key in d: extracted.add(str(d[key]))
            if not extracted: extracted.add(str(sorted(d.items())))
        elif isinstance(item, list):
            for sub in item: recursive_extract(sub)
        else:
            extracted.add(str(item))
    for val in record_values: recursive_extract(val)
    return extracted

# ==============================================================================
# 3. 실행 로직 (타임아웃 적용)
# ==============================================================================
def run_query_and_fetch(session, query):
    clean_q = clean_cypher(query)
    if not clean_q: return None, "Empty"
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(QUERY_TIMEOUT)
    try:
        result = session.run(clean_q)
        data = [record.values() for record in result]
        signal.alarm(0)
        return data, "Success"
    except TimeoutException:
        return None, "Timeout"
    except Exception as e:
        signal.alarm(0)
        return None, f"Error: {str(e)}"

# ==============================================================================
# 4. 채점 및 분석 로직 (V6.0 Deep Subset 동일 적용)
# ==============================================================================
def is_deep_subset(subset_data, superset_data):
    if not subset_data: return True 
    if not superset_data: return False
    
    sub_sets = [extract_ids_from_record(r) for r in subset_data]
    super_sets = [extract_ids_from_record(r) for r in superset_data]
    
    for sub in sub_sets:
        if not sub: continue
        match_found = False
        for sup in super_sets:
            if not sub.isdisjoint(sup):
                match_found = True
                break
        if not match_found: return False
    return True

def analyze_and_score(session, gt_query, bot_query):
    report = {
        "is_correct": False,
        "match_type": "Mismatch",
        "gt_count": 0, "bot_count": 0,
        "missing_examples": "", "extra_examples": ""
    }
    
    # 1. 실행
    gt_res, gt_msg = run_query_and_fetch(session, gt_query)
    bot_res, bot_msg = run_query_and_fetch(session, bot_query)
    
    if gt_res is None: 
        report["match_type"] = f"GT Error: {gt_msg}"
        return report
    if bot_res is None:
        report["match_type"] = f"Bot Error: {bot_msg}"
        return report

    report["gt_count"] = len(gt_res)
    report["bot_count"] = len(bot_res)

    # 2. Deep Inspection (Count -> List 변환)
    final_gt_res = gt_res
    final_bot_res = bot_res
    
    gt_list_query = unwrap_count_query(gt_query)
    bot_list_query = unwrap_count_query(bot_query)
    
    if gt_list_query or bot_list_query:
        q1 = gt_list_query if gt_list_query else gt_query
        q2 = bot_list_query if bot_list_query else bot_query
        
        deep_gt, _ = run_query_and_fetch(session, q1)
        deep_bot, _ = run_query_and_fetch(session, q2)
        
        if deep_gt is not None and deep_bot is not None:
            final_gt_res = deep_gt
            final_bot_res = deep_bot

    # 3. 채점 (V6.0 Logic)
    is_subset = is_deep_subset(final_bot_res, final_gt_res)
    is_superset = is_deep_subset(final_gt_res, final_bot_res)
    
    if is_subset or is_superset:
        report["is_correct"] = True
        if is_subset and is_superset:
            report["match_type"] = "Full Match"
        else:
            report["match_type"] = "Subset Match"
            
    if not final_gt_res and not final_bot_res:
        report["is_correct"] = True
        report["match_type"] = "Empty Match"

    # 숫자 정확 일치 보너스
    if not report["is_correct"]:
        if len(gt_res) == 1 and len(bot_res) == 1 and isinstance(gt_res[0][0], (int, float)):
             if gt_res[0][0] == bot_res[0][0]:
                 report["is_correct"] = True
                 report["match_type"] = "Count Exact Match"
             else:
                 report["match_type"] = "Count Mismatch"

    # 4. 분석 데이터 생성 (리포트용)
    gt_ids_list = [extract_ids_from_record(r) for r in final_gt_res]
    bot_ids_list = [extract_ids_from_record(r) for r in final_bot_res]
    all_gt_ids = set().union(*gt_ids_list)
    all_bot_ids = set().union(*bot_ids_list)
    
    report["missing_examples"] = str(list(all_gt_ids - all_bot_ids)[:3])
    report["extra_examples"] = str(list(all_bot_ids - all_gt_ids)[:3])
    
    return report

# ==============================================================================
# 5. 메인 실행 및 포맷팅 출력
# ==============================================================================
if __name__ == "__main__":
    print(f"🔌 Neo4j 연결: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        df_qa = pd.read_csv(qachain_csv)
        df_my = pd.read_csv(my_best_csv)
        df_qa.columns = df_qa.columns.str.strip()
        df_my.columns = df_my.columns.str.strip()
        if 'cypher_executable' in df_qa.columns: df_qa = df_qa.rename(columns={'cypher_executable': 'truth'})
        merged = pd.merge(df_qa, df_my, on='question', how='inner', suffixes=('_qa', '_my'))
        print(f"✅ 총 {len(merged)}개 데이터 로드 완료.")
    except: exit()

    results_data = []
    stats = { "valid": 0, "gt_fail": 0, "lipidbot": 0, "qachain": 0 }
    match_types = collections.Counter()

    print("\n🚀 [V7.0 Final Evaluation] 채점 시작...")
    
    with driver.session() as session:
        for idx, row in tqdm(merged.iterrows(), total=len(merged)):
            truth = str(row.get('truth', row.get('truth_qa', '')))
            my_cypher = str(row.get('generated_cypher_my', row.get('generated_cypher', '')))
            qa_cypher = str(row.get('QAChain', ''))
            
            my_report = analyze_and_score(session, truth, my_cypher)
            
            if "GT Error" in my_report["match_type"]:
                stats["gt_fail"] += 1
                status_str = "Invalid (GT Error)"
            else:
                stats["valid"] += 1
                match_types[my_report["match_type"]] += 1
                
                if my_report["is_correct"]: 
                    stats["lipidbot"] += 1
                    status_str = "✅ Correct"
                else:
                    status_str = "❌ Incorrect"
                
                # QAChain 비교용 (통계만)
                qa_report = analyze_and_score(session, truth, qa_cypher)
                if qa_report["is_correct"]: stats["qachain"] += 1

            # [CSV 강화] 보기 좋은 포맷으로 저장
            results_data.append({
                "Result": status_str,
                "Reason (Match Type)": my_report["match_type"],
                "Counts (Bot / GT)": f"{my_report['bot_count']} / {my_report['gt_count']}",
                "Missing IDs (Example)": my_report["missing_examples"] if my_report["missing_examples"] != "[]" else "-",
                "Extra IDs (Example)": my_report["extra_examples"] if my_report["extra_examples"] != "[]" else "-",
                "Question": row['question'],
                "Bot Query": my_cypher,
                "Truth Query": truth
            })

    driver.close()

    # CSV 저장
    # 컬럼 순서 재배치
    cols = ["Result", "Reason (Match Type)", "Counts (Bot / GT)", "Missing IDs (Example)", "Extra IDs (Example)", "Question", "Bot Query", "Truth Query"]
    pd.DataFrame(results_data)[cols].to_csv(output_report_csv, index=False)
    
    # ----------------------------------------------------------------------
    # [Prettier Console Output] 최종 결과 화면 출력
    # ----------------------------------------------------------------------
    def pct(n): return (n / stats["valid"] * 100) if stats["valid"] > 0 else 0
    
    print("\n" + "="*80)
    print(" 📊 FINAL PERFORMANCE DASHBOARD")
    print("="*80)
    print(" [Data Summary]")
    print(f"  • 총 문제 수      : {len(merged)}")
    print(f"  • ❌ 무효(GT Error): {stats['gt_fail']} (제외됨)")
    print(f"  • ✅ 유효(Valid)   : {stats['valid']} (채점 대상)")
    print("")
    print(" [Accuracy Ranking]")
    print(f"  🥇 LipidBot      : {pct(stats['lipidbot']):.1f}% ({stats['lipidbot']} / {stats['valid']}) 🚀")
    print(f"  🥈 CypherQAChain : {pct(stats['qachain']):.1f}% ({stats['qachain']} / {stats['valid']})")
    print("")
    print(" [Match Details (LipidBot Analysis)]")
    total_correct = stats['lipidbot']
    total_incorrect = stats['valid'] - total_correct
    
    # 정답 유형별 통계
    print(f"  • 🟢 Full/Perfect Match : {match_types['Full Match']} ({match_types['Full Match']/stats['valid']*100:.1f}%)")
    print(f"  • 🟡 Subset Match (부분) : {match_types['Subset Match']} ({match_types['Subset Match']/stats['valid']*100:.1f}%)")
    print(f"  • ⚪ Empty Match (공통없음): {match_types['Empty Match']} ({match_types['Empty Match']/stats['valid']*100:.1f}%)")
    print(f"  • 🔵 Count Exact Match  : {match_types['Count Exact Match']}")
    print("-" * 40)
    print(f"  • 🔴 Failures (Mismatch) : {total_incorrect} ({total_incorrect/stats['valid']*100:.1f}%)")
    print("="*80)
    print(f"💾 상세 리포트 저장 완료: '{output_report_csv}'")