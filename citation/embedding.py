import numpy as np
from sentence_transformers import SentenceTransformer

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def encode_texts(
    model: SentenceTransformer,
    texts,
    batch_size: int = 64
) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return normalize(vecs).astype("float32")
