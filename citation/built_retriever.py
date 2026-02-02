from typing import List, Optional
from data_service import Chunk, Hit
from citation.cache_helper import (
    load_all_caches
)
from dataclasses import dataclass
from typing import List
from citation.embedding import encode_texts
import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class Retriever:
    model: SentenceTransformer
    index: faiss.IndexFlatIP
    chunks: List[Chunk]

    def search(self, query: str, top_k: int) -> List[Hit]:
        qvec = encode_texts(self.model, [query], batch_size=1)
        D, I = self.index.search(qvec, top_k)
        hits: List[Hit] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            ch = self.chunks[idx]
            # print(ch)
            hits.append(Hit(
                score=float(score), 
                citation_id=ch['citation_id'], 
                chunk_id=ch['chunk_id'], 
                text=ch['text'], 
                title=ch['title']
            ))

        return hits


def build_retriever(
    target_model_name: str = "",
    cache_dir: Optional[str] = None,
) -> Optional[Retriever]:
    caches = load_all_caches(cache_dir)

    for model_name, index, chunk_list in caches:
        print(f"{model_name}, target_model_name: {target_model_name}")
        if model_name == target_model_name:
            return Retriever(
                model=model_name,
                index=index,
                chunks=chunk_list,
            )

    return None


def build_hybrid_retriever(
    target_model_names: List[str],
    cache_dir: Optional[str] = None,
) -> List[Retriever]:

    hybrid_retrievers: List[Retriever] = []
    caches = load_all_caches(cache_dir)
    # print(len(caches))
    for model_name, index, chunk_list in caches:
        # print(f"{model_name}, {target_model_names}")
        if model_name not in target_model_names:
            continue

        retriever = Retriever(
            model=SentenceTransformer(model_name),
            index=index,
            chunks=chunk_list,
        )
        hybrid_retrievers.append(retriever)

    return hybrid_retrievers
