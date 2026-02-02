import pandas as pd
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
from data_service import Citation, Chunk

nltk.download("punkt_tab")

def build_chunks(
    records: List[Citation],
    chunk_size: int,
    chunk_overlap: int
) -> List[Chunk]:
    out: List[Chunk] = []

    for rec in records:
        title = str(rec.title) if pd.notna(rec.title) else ""
        abstract = str(rec.abstract) if pd.notna(rec.abstract) else ""
        base = title
        if abstract:
            base += "\n" + abstract
        if not base.strip():  # Skip completely empty records
            continue
        
        pieces = chunk_text_sentences(base, chunk_size, chunk_overlap)
        for i, txt in enumerate(pieces):
            out.append(Chunk(rec.citation_id, i, txt, rec.title))

    return out


def chunk_text_sentences(
    text: str,
    chunk_size: int = 180,
    chunk_overlap: int = 40
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    chunks, current = [], []
    cur_words = 0

    for s in sents:
        w = len(s.split())
        if current and cur_words + w > chunk_size:
            chunks.append(" ".join(current))
            keep, kept = [], 0
            for sent in reversed(current):
                keep.append(sent)
                kept += len(sent.split())
                if kept >= chunk_overlap:
                    break
            current = list(reversed(keep))
            cur_words = sum(len(x.split()) for x in current)

        current.append(s)
        cur_words += w

    if current:
        chunks.append(" ".join(current))

    return chunks



