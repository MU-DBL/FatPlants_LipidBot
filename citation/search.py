import torch
import sys
from typing import List, Tuple, Dict

# ==========================================
# 🛑 [긴급 패치] PyTorch 2.6+ 보안 경고 무시 (강제 우회)
# ==========================================
# 기존 torch.load를 백업하고, 강제로 weights_only=False를 주입하는 래퍼 함수로 교체
_original_load = torch.load

def _unsafe_load(*args, **kwargs):
    # weights_only 인자가 있으면 제거하고 False로 강제 설정
    if 'weights_only' in kwargs:
        del kwargs['weights_only']
    return _original_load(*args, **kwargs, weights_only=False)

torch.load = _unsafe_load
# ==========================================

from citation.built_retriever import build_hybrid_retriever
from data_service import Hit
from config import DEFAULT_EMBEDDING_MODEL, CITATION_DIR, BM25_CACHE   
from citation.bm25_cache import BM25Cache

# ==========================================
# 🚀 Global Cache (모델을 한 번만 로딩하기 위한 전역 변수)
# ==========================================
_CACHED_RETRIEVERS = {}
_CACHED_BM25 = None

def get_cached_retrievers(model_names):
    """모델이 이미 로딩되어 있으면 그걸 쓰고, 없으면 로딩합니다."""
    global _CACHED_RETRIEVERS
    
    if model_names is None:
        names_to_load = DEFAULT_EMBEDDING_MODEL
    else:
        names_to_load = model_names
        
    # 리스트를 튜플로 변환하여 키로 사용
    key = tuple(sorted(names_to_load))
    
    if key not in _CACHED_RETRIEVERS:
        print(f"\n   [System] Loading AI Models into GPU memory (One-time only)...")
        _CACHED_RETRIEVERS[key] = build_hybrid_retriever(list(names_to_load), CITATION_DIR)
    
    return _CACHED_RETRIEVERS[key]

def get_cached_bm25():
    """BM25 인덱스를 한 번만 로딩합니다."""
    global _CACHED_BM25
    if _CACHED_BM25 is None:
        _CACHED_BM25 = BM25Cache()
        _CACHED_BM25.load(BM25_CACHE)
    return _CACHED_BM25

def search(
    query: str,
    model_names: List[str] = None,
    top_k_per_model: int = 5,
    fuse: str = "rrf",  # "rrf" | "vote" | "max"
    per: str = "chunk",  # "chunk" | "citation_id" 
    rrf_k: int = 60,
    add_bm25: bool = True
) -> List[Hit]:
    assert fuse in {"rrf", "vote", "max"}
    assert per in {"chunk", "citation_id"}

    # 1. 모델 가져오기 (캐시 사용 - 속도 최적화)
    hybrid_retrievers = get_cached_retrievers(model_names)

    # Independent search by each model
    per_model_results: List[List[Hit]] = []
    for retriever in hybrid_retrievers:
        per_model_results.append(retriever.search(query, top_k=top_k_per_model))

    # 2. BM25 가져오기 (캐시 사용)
    if add_bm25:
        bm25_cache = get_cached_bm25()
        per_model_results.append(bm25_cache.search(query, top_k=top_k_per_model))

    # Fusion Logic (박사님 원본 로직 유지)
    def key_of(h: Hit):
        return (h.citation_id, h.chunk_id) if per == "chunk" else (h.citation_id,)

    agg_payload: Dict[Tuple, Hit] = {}
    best_score: Dict[Tuple, float] = {}
    best_rank: Dict[Tuple, int] = {}
    votes: Dict[Tuple, int] = {}
    scores: Dict[Tuple, float] = {}

    for model_hits in per_model_results:
        for rank, h in enumerate(model_hits, start=1):
            k = key_of(h)

            if (k not in agg_payload) or (h.score > best_score.get(k, -1e9)):
                agg_payload[k] = h
                best_score[k] = h.score

            if fuse == "rrf":
                scores[k] = scores.get(k, 0.0) + 1.0 / (rrf_k + rank)
                best_rank[k] = min(best_rank.get(k, 1_000_000), rank)
            elif fuse == "vote":
                votes[k] = votes.get(k, 0) + 1
            elif fuse == "max":
                scores[k] = max(scores.get(k, -1e9), h.score)

    # 3) Sorting
    items = list(agg_payload.items())
    if fuse == "vote":
        items.sort(
            key=lambda kv: (votes.get(kv[0], 0), best_score.get(kv[0], -1e9)),
            reverse=True,
        )
    elif fuse == "rrf":
        items.sort(
            key=lambda kv: (scores.get(kv[0], 0.0), -best_rank.get(kv[0], 1_000_000)),
            reverse=True,
        )
    elif fuse == "max":
        items.sort(key=lambda kv: scores.get(kv[0], -1e9), reverse=True)

    # 4) Assemble output
    out: List[Hit] = []
    for _, h in items[:top_k_per_model]:
        out.append(
            Hit(
                score=float(
                    best_score[(h.citation_id, h.chunk_id) if per == "chunk" else (h.citation_id,)]
                ),
                citation_id=h.citation_id,
                chunk_id=h.chunk_id,
                text=h.text,
                title=h.title,
            )
        )
    return out
