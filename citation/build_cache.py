import os
import pandas as pd
from typing import List, Optional
from citation.built_retriever import Retriever
from sentence_transformers import SentenceTransformer
from citation.data import Citation, cache_dir, default_model_name, Chunk, hf_model_dir, bm25_cache_file
from citation.embedding import encode_texts
from citation.chunking import build_chunks
from citation.index import build_index
from citation.cache_helper import make_build_signature, try_load_cache, save_cache
from citation.bm25_cache import BM25Cache

def build_cache_single_model(
    model_name: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 40,
    batch_size: int = 64,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    csv_path: str = None,
) -> Retriever:
    signature = make_build_signature(csv_path, model_name, chunk_size, chunk_overlap)

    # 1) Try cache
    if cache_dir:
        cached = try_load_cache(cache_dir, model_name, signature)
        if cached is not None:
            index, chunks = cached
            if verbose:
                print(f"[cache] loaded index+chunks for {model_name}")
            return Retriever(
                model=SentenceTransformer(model_name),
                index=index,
                chunks=chunks,
            )

    # 2) Load data → chunk
    df = pd.read_csv(csv_path)
    records: List[Citation] = [
        Citation(citation_id=row["citation_id"], title=row["title"], abstract=row["abstract"])
        for _, row in df.iterrows()
    ]

    if verbose:
        print(f"Loaded {len(records)} records from {csv_path}")

    chunks = build_chunks(records, chunk_size, chunk_overlap)
    if verbose:
        print(f"Built {len(chunks)} chunks")

    # 3) Encode → index
    model = SentenceTransformer(model_name)
    embeddings = encode_texts(
        model,
        [c.text for c in chunks],
        batch_size=batch_size,
    )
    index = build_index(embeddings)

    # 4) Save cache
    if cache_dir:
        if verbose:
            print(f"[cache] saving -> {cache_dir}")
        save_cache(cache_dir, model_name, signature, index, chunks)


def build_cache_hybrid_model(
    tsv_path: str,
    model_names: List[str],
    chunk_size: int = 512,
    chunk_overlap: int = 40,
    batch_size: int = 64,
    cache_dir: str = None,
):
    chunks: Optional[List[Chunk]] = None
    for name in model_names:
        build_cache_single_model(
            model_name=name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            cache_dir=cache_dir,
            csv_path=tsv_path
        )

def build_bm25_cache(csv_path: str, cache_path: str):
    cache = BM25Cache(cache_path)
    cache.build_from_csv(csv_path, ["citation_id","title","abstract"])
    cache.save()

if __name__ == "__main__":
    os.environ["HF_HOME"] = hf_model_dir
    csv_path = "files/combined_all_classified_filtered.csv"
    build_cache_hybrid_model(csv_path, model_names = default_model_name , cache_dir = cache_dir)
    # build_bm25_cache(csv_path, bm25_cache_file)
