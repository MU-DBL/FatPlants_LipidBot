import os
import pandas as pd
import numpy as np
import pickle
import gc
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score
import torch
from citation.bm25_cache import BM25Cache
from config import bm25_cache_file, default_model_name
from citation.search import search

# =========================================================
# ⚙️ 1. Settings and Paths
# =========================================================
QUESTION_FILE = "../file/generated_questions_semantic_full.csv"
K = 10
SAVE_INTERVAL = 50  

# [GPU]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Current Device: {device.upper()}")
if device == 'cuda':
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")

CACHE_DIR = "citation/results_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_BM25 = os.path.join(CACHE_DIR, "preds_bm25.pkl")
CACHE_BASE = os.path.join(CACHE_DIR, "preds_base.pkl")
CACHE_LIPIDBOT = os.path.join(CACHE_DIR, "preds_lipidbot.pkl")

# Load questions and ground truth PMIDs
df_questions = pd.read_csv(QUESTION_FILE)
ground_truth_citation_id = df_questions['pmid'].astype(str).tolist()
questions = df_questions['question'].tolist()

def run_gpu_optimized(model_name, cache_path, is_hybrid):
    """GPU 최적화 + 이어하기 기능 실행 함수"""
    results = []
    
    # 1. 기존 캐시가 있으면 로딩 (이어하기)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                results = pickle.load(f)
            print(f"   ⏩ Resuming {model_name} from index {len(results)}...")
        except:
            print(f"   ⚠️ Cache broken, restarting {model_name}...")
            results = []

    # 2. 이미 다 했으면 패스
    start_idx = len(results)
    if start_idx >= len(questions):
        print(f"   ✅ {model_name} already completed!")
        return results

    print(f"   ⚡ Processing {model_name} on GPU (Auto-saving every {SAVE_INTERVAL})...")
    
    # 3. 남은 부분 계산
    for i in tqdm(range(start_idx, len(questions)), desc=model_name):
        q = questions[i]
        try:
            # search 함수 호출 (모델 캐싱됨)
            if is_hybrid:
                res = search(q, default_model_name, 10, "rrf", "chunk", 60, True)
            else:
                res = search(q, [default_model_name[0]], 10, "rrf", "chunk", 60, False)
            results.append(res)
        except Exception as e:
            print(f"Error on question {i}: {e}")
            results.append([]) 

        # 4. 중간 저장 및 메모리 청소
        if len(results) % SAVE_INTERVAL == 0:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # 5. 최종 저장
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    
    return results

final_results = []

# =========================================================
# [A] BM25
# =========================================================
print("\n[Model 1] Running BM25...")
if os.path.exists(CACHE_BM25):
    with open(CACHE_BM25, 'rb') as f: bm25_preds_k = pickle.load(f)
else:
    cache = BM25Cache()
    cache.load(bm25_cache_file)
    bm25_preds_k = [cache.search(q, top_k=10) for q in questions]
    with open(CACHE_BM25, 'wb') as f: pickle.dump(bm25_preds_k, f)

# =========================================================
# [B] PubmedBERT Base (GPU)
# =========================================================
print("\n[Model 2] pubmedbert base")
base = run_gpu_optimized("pubmedbert_base", CACHE_BASE, is_hybrid=False)

# =========================================================
# [C] LipidBot Hybrid (GPU)
# =========================================================
print("\n[Model 3] Running LipidBot")
lipidbot = run_gpu_optimized("LipidBot", CACHE_LIPIDBOT, is_hybrid=True)


try:
    results_detail = []
    for i, question in enumerate(questions):
        row = {
            'ground_truth_pmid': ground_truth_citation_id[i],
            'question': question,
            'bm25_predictions': ','.join([str(h.citation_id) if hasattr(h,'citation_id') else str(h) for h in bm25_preds_k[i]]),
            'pubmedbert_base_predictions': ','.join([str(h.citation_id) if hasattr(h,'citation_id') else str(h) for h in base[i]]),
            'lipidbot_predictions': ','.join([str(h.citation_id) if hasattr(h,'citation_id') else str(h) for h in lipidbot[i]])
        }
        results_detail.append(row)
    pd.DataFrame(results_detail).to_csv("detailed_predictions.csv", index=False)
except Exception: pass
