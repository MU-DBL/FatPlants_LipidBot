import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from pathlib import Path
from data_service import Hit

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class BM25Cache:
    def __init__(self, cache_path: str = None):
        self.cache_path = cache_path
        self.bm25 = None
        self.documents = []
        self.metadata = []

    def tokenize(self, text: str) -> List[str]:
        if pd.isna(text) or not text:
            return []
        # Lowercase and tokenize
        tokens = word_tokenize(str(text).lower())
        # Remove very short tokens and numbers-only
        return [t for t in tokens if len(t) > 2 and not t.isdigit()]

    def build_from_csv(self, csv_path: str, text_fields: List[str] = None):
        if text_fields is None:
            text_fields = ["title", "abstract"]

        # Load data
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Clean data
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].fillna("")

        # Build documents and metadata
        print("Building BM25 index...")
        tokenized_docs = []

        for idx, row in df.iterrows():
            # Combine text fields
            text_parts = [
                str(row[field])
                for field in text_fields
                if field in df.columns and pd.notna(row[field])
            ]
            combined_text = " ".join(text_parts)

            # Skip empty documents
            if not combined_text.strip():
                continue

            # Tokenize
            tokens = self.tokenize(combined_text)
            if not tokens:
                continue

            tokenized_docs.append(tokens)

            # Store metadata
            metadata = row.to_dict()
            metadata["_citation_id"] = len(self.documents)
            metadata["_combined_text"] = combined_text
            self.metadata.append(metadata)
            self.documents.append(combined_text)

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        return self

    def save(self, cache_path: str = None):
        save_path = cache_path or self.cache_path
        if not save_path:
            raise ValueError("No cache path specified")

        # Create directory if needed
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "metadata": self.metadata,
        }

        with open(save_path, "wb") as f:
            pickle.dump(cache_data, f)

        print(f"✓ Cache saved to {save_path}")
        return self

    def load(self, cache_path: str = None):
        """Load BM25 cache from disk"""
        load_path = cache_path or self.cache_path
        if not load_path:
            raise ValueError("No cache path specified")

        with open(load_path, "rb") as f:
            cache_data = pickle.load(f)

        self.bm25 = cache_data["bm25"]
        self.documents = cache_data["documents"]
        self.metadata = cache_data["metadata"]
        return self

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            raise ValueError(
                "BM25 index not built. Call build_from_csv() or load() first."
            )

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                metadata = self.metadata[idx]
                hit = Hit(
                    score=float(scores[idx]),
                    citation_id=metadata.get(
                        "citation_id", metadata.get("citation_id", str(idx))
                    ),
                    chunk_id=idx,
                    text=self.documents[idx],
                    title=metadata.get("title", ""),
                )
                results.append(hit)
            
        return results
