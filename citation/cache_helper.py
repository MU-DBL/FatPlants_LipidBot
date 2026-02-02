from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json, hashlib, re
import faiss
from data_service import Chunk


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _scan_input_signature(input_path: str) -> Dict[str, Any]:
    p = Path(input_path)
    if not p.is_file():
        raise ValueError(f"Path not found: {input_path}")
    st = p.stat()
    return {
        "file": {
            "sha256": _file_sha256(p),
            "size": st.st_size,
            "suffix": p.suffix.lower(),
        }
    }


def make_build_signature(tsv_path, model_name, chunk_size, chunk_overlap):
    return {
        "schema": "portable-v1",
        "data_sig": _scan_input_signature(tsv_path),
        "model": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def _hash_obj(obj: Dict[str, Any]) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _paths(cache_dir: str, model_name: str, sig_hash: str):
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_name(model_name)}__{sig_hash}"
    return {
        "manifest": base / f"{stem}.manifest.json",
        "index": base / f"{stem}.faiss",
        "chunks": base / f"{stem}.chunks.jsonl",
    }


def save_cache(cache_dir, model_name, signature, index, chunks):
    sig_hash = _hash_obj(signature)
    ps = _paths(cache_dir, model_name, sig_hash)

    faiss.write_index(index, str(ps["index"]))

    with open(ps["chunks"], "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    with open(ps["manifest"], "w", encoding="utf-8") as f:
        json.dump(
            {"signature": signature,
             "index_path": str(ps["index"]),
             "chunks_path": str(ps["chunks"])},
            f,
            ensure_ascii=False
        )


def try_load_cache(
    cache_dir, model_name, signature
) -> Optional[Tuple[faiss.IndexFlatIP, List[Chunk]]]:

    sig_hash = _hash_obj(signature)
    ps = _paths(cache_dir, model_name, sig_hash)

    if not all(p.exists() for p in ps.values()):
        return None

    try:
        manifest = json.loads(ps["manifest"].read_text())
        if manifest["signature"] != signature:
            return None

        index = faiss.read_index(str(ps["index"]))
        chunks = []

        with open(ps["chunks"], "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk(**json.loads(line)))

        return index, chunks
    except Exception:
        return None


def load_all_caches(
    cache_dir: str,
) -> List[Tuple[str, faiss.Index, List[Chunk]]]:

    cache_root = Path(cache_dir)
    results: List[Tuple[str, faiss.Index, List[Chunk]]] = []

    for manifest_path in cache_root.glob("*.manifest.json"):
        base = manifest_path.name.replace(".manifest.json", "")
       
        index_path = cache_root / f"{base}.faiss"
        chunks_path = cache_root / f"{base}.chunks.jsonl"

        if not index_path.exists() or not chunks_path.exists():
            continue

        try:
            # ---- load manifest
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            model_name = manifest.get("signature", {}).get("model")
            if not model_name:
                continue

            # ---- load index
            index = faiss.read_index(str(index_path))

            # ---- load chunks
            chunks: List[Chunk] = []
            
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if "citation_id" in data:
                        key_id = data["citation_id"]
                    elif "pmid" in data:  # Changed 'else' to 'elif'
                        key_id = data["pmid"]
                    else:
                        key_id = None 

                    chunk = {
                        "citation_id": key_id,
                        "chunk_id": data.get("chunk_id"),
                        "text": data.get("text"),
                        "title": data.get("title")
                    }
                    chunks.append(chunk)

            results.append((model_name, index, chunks))

        except Exception:
            # corrupted / incompatible cache entry
            continue

    return results
