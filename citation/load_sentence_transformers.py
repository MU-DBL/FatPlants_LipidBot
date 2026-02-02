import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import HF_HOME, DEFAULT_EMBEDDING_MODEL

def load_sentence_transformers():
    os.environ["HF_HOME"] = HF_HOME
    Path(HF_HOME).mkdir(parents=True, exist_ok=True)

    loaded_models = []

    for model_name in DEFAULT_EMBEDDING_MODEL:
        # Safe cache path
        cache_dir = Path(HF_HOME) / model_name.replace("/", "_")

        if cache_dir.exists():
            print(f"[Cache] Loading model from cache: {cache_dir}")
            st_model = SentenceTransformer(str(cache_dir))
        else:
            print(f"[Download] Model not found in cache. Downloading: {model_name}")
            st_model = SentenceTransformer(model_name)
            st_model.save(str(cache_dir))
            print(f"[Cache] Saved model to: {cache_dir}")

        loaded_models.append(st_model)

    return loaded_models
