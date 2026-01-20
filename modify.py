import pandas as pd

# 파일 경로
INPUT_FILE = "evaluation/pathway_evaluation_complete_with_result.csv"
FALLBACK_OUTPUT = "evaluation/fallback_only_cases.csv"

def main():
    print(f"📂 결과 분석 중... ({INPUT_FILE})")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다.")
        return

    # 'generation_type' 컬럼 확인
    if 'generation_type' not in df.columns:
        print("⚠️ 'generation_type' 컬럼이 없습니다. query_engine.py가 최신 버전인지 확인하세요.")
        return

    # 1. 통계 계산
    counts = df['generation_type'].value_counts()
    template_count = counts.get('Template', 0)
    fallback_count = counts.get('Fallback', 0)
    failed_count = counts.get('Failed', 0) + counts.get('Error', 0)
    total = len(df)

    print("\n" + "="*40)
    print("📊 생성 방식 통계 (Generation Stats)")
    print("="*40)
    print(f"• Total Questions : {total}")
    print(f"• 🟢 Template Used : {template_count} ({template_count/total*100:.1f}%)")
    print(f"• 🟠 Fallback Used : {fallback_count} ({fallback_count/total*100:.1f}%)")
    if failed_count > 0:
        print(f"• 🔴 Failed/Error  : {failed_count} ({failed_count/total*100:.1f}%)")
    print("="*40)

    # 2. Fallback 케이스만 따로 저장
    fallback_df = df[df['generation_type'] == 'Fallback'].copy()
    
    # auto_grader가 읽을 수 있도록 컬럼명 맞추기
    # (기존 fallback_with_answers.csv 형식을 따름)
    # 필요한 컬럼: Row_Index(원래 인덱스), Question, Cypher_executable(정답), Generated_Cypher(생성)
    
    # 인덱스를 Row_Index로 저장 (0부터 시작하므로 엑셀 행번호처럼 1 더해줌)
    fallback_df['Row_Index'] = fallback_df.index
    
    # 컬럼 이름 변경 (매핑)
    # df의 'cypher_executable' -> 정답
    # df의 'generated_cypher' -> 생성된 쿼리
    export_df = fallback_df[['Row_Index', 'question', 'cypher_executable', 'generated_cypher']].copy()
    export_df.columns = ['Row_Index', 'Question', 'Cypher_executable', 'Generated_Cypher (Fallback)']
    
    export_df.to_csv(FALLBACK_OUTPUT, index=False)
    print(f"\n✅ Fallback 케이스 {len(export_df)}개를 별도 파일로 저장했습니다.")
    print(f"   -> {FALLBACK_OUTPUT}")

if __name__ == "__main__":
    main()