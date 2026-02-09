# pip install pyahocorasick rapidfuzz
import os, re, pickle, glob
import ahocorasick
from typing import Dict, List, Optional, Tuple
import pandas as pd
from config import AC_KEGG_PKL

# ---------- 规范化 ----------
def norm(s: str) -> str:
    """Normalize text for matching"""
    s = s.lower().strip()
    
    # Greek letters
    s = (s.replace("α", "alpha")
           .replace("β", "beta")
           .replace("γ", "gamma")
           .replace("δ", "delta")
           .replace("ε", "epsilon")
           .replace("μ", "mu")
           .replace("ω", "omega"))
    
    # Remove non-structural brackets (keep those with digits or R/S chirality)
    s = re.sub(r"\[[^\[\]]*\]", "", s)
    s = re.sub(r"\([^()\dRrSs]+\)", "", s)
    
    # Symbol replacements
    s = s.replace("&", " and ")
    s = s.replace("-", " ")
    s = s.replace("→", "to").replace("->", "to")
    
    # Clean up whitespace and punctuation
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*[,;:]\s*", " ", s)
    s = s.strip(" ,;:")
    
    return s

def infer_db_from_filename(fname: str) -> str:
    """Infer database type from filename"""
    f = os.path.basename(fname).lower()
    if "compound" in f: return "Compound"
    if "enzyme" in f:   return "EC"
    if "reaction" in f: return "Reaction"
    if "pathway" in f:  return "Pathway"
    if "ortholog" in f: return "Ortholog"
    if "gene" in f:     return "Gene"
    return "Other"

# ---------- 读取并合并别名 ----------
def load_alias_entries(csv_files: List[str]) -> pd.DataFrame:
    """Load and combine alias entries from CSV files"""
    dfs = []
    for fp in csv_files:
        db = infer_db_from_filename(fp)
        df = pd.read_csv(fp, dtype=str).fillna("")
        df.columns = [c.strip().lower() for c in df.columns]
        
        if "species" not in df.columns:
            df["species"] = 'all'
        else:  # <-- Added colon here
            df["species"] = df["species"].str.lower()  # <-- Use .str.lower() for Series
        
        df["db"] = db
        dfs.append(df[["id", "name", "species", "db"]])
    
    all_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return all_df

# ---------- 构建 AC（改进版：存储长度） ----------
def build_ac(entries: pd.DataFrame, min_length: int = 2):
    """Build Aho-Corasick automaton with length information"""
    alias_map: Dict[str, List[Dict]] = {}
    
    for _, row in entries.iterrows():
        al = norm(row["name"])
        # Skip empty or very short normalized names
        if not al or len(al) < min_length:
            continue
            
        alias_map.setdefault(al, []).append({
            "id": row["id"],
            "species": row["species"],
            "db": row["db"],
            "alias_raw": row["name"],
            "length": len(al)  # Store normalized length
        })
    
    A = ahocorasick.Automaton()
    for alias, lst in alias_map.items():
        A.add_word(alias, (alias, lst))  # Store both alias and candidates
    A.make_automaton()
    
    return A, alias_map

# ---------- 缓存 ----------
def save_cache(A, alias_map: Dict, cache_path: str):
    """Save automaton and alias map to cache"""
    with open(cache_path, "wb") as f:
        pickle.dump((A, alias_map), f)
    print(f"Cache saved to {cache_path}")


AC_AUTOMATON = None
ALIAS_MAP = None

def load_cache(cache_path: str):
    global AC_AUTOMATON, ALIAS_MAP

    if cache_path is None:
        cache_path = AC_KEGG_PKL
    
    if AC_AUTOMATON is None or ALIAS_MAP is None:
        with open(cache_path, "rb") as f:
            AC_AUTOMATON, ALIAS_MAP = pickle.load(f)
        print(f"Cache loaded from {cache_path}")
    else:
        print("Cache loaded from memory")

    return AC_AUTOMATON, ALIAS_MAP

# ---------- 查询 ----------
def query_text(
    text: str,
    A,
    alias_map: Dict[str, List[Dict]],
    species: Optional[str] = None,
    prefer_db: Tuple[str, ...] = ("gene", "ko", "compound", "ec", "reaction", "pathway", "other")
) -> List[Dict]:
    
    t = norm(text)
    hits = []
    
    # Collect all matches with stored length information
    for end_idx, (alias, cand_list) in A.iter(t):
        start_idx = end_idx - len(alias) + 1
        
        # Filter by species if specified
        if species:
            cands = [c for c in cand_list if c["species"] in (species, "all", "unknown")]
        else:
            cands = cand_list
        
        if cands:
            for c in cands:
                hits.append((start_idx, end_idx + 1, alias, c))
    
    # Resolve overlaps: longest match first, then earliest position
    hits.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    
    used = [False] * len(t)
    kept = []
    
    for s, e, alias, c in hits:
        if any(used[i] for i in range(s, e)):
            continue
        
        for i in range(s, e):
            used[i] = True
        kept.append((s, e, alias, c))
    
    # Sort by position, then database priority
    kept.sort(key=lambda x: (
        x[0],
        prefer_db.index(x[3]["db"]) if x[3]["db"] in prefer_db else 999
    ))
    
    # Format output
    results = []
    for s, e, alias, c in kept:
        results.append({
            "start": s,
            "end": e,
            "match": alias,
            "id": c["id"],
            "species": c["species"],
            "db": c["db"],
            "alias_raw": c["alias_raw"]
        })
    
    return results

# ---------- 一键构建 ----------
def build_from_dir(csv_dir: str, cache_path: str = "ac_kegg.pkl"):
    """Build automaton from directory of CSV files"""
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "ID_map_*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"No ID_map_*.csv files found in {csv_dir}")
    
    print(f"Found {len(csv_files)} CSV files")
    entries = load_alias_entries(csv_files)
    print(f"Loaded {len(entries)} total entries")
    
    A, alias_map = build_ac(entries)
    print(f"Built automaton with {len(alias_map)} unique normalized aliases")
    
    save_cache(A, alias_map, cache_path)
    return A, alias_map

# ---------- 使用示例 ----------
# if __name__ == "__main__":
#     # Build from directory
#     csv_dir = "C:/Users/yqzn9/Documents/GitHub/fastapi-fatplants/mapper/maps_dir/"
#     A, alias_map = build_from_dir(csv_dir, "ac_kegg.pkl")
    
#     # Or load from cache
#     A, alias_map = load_cache("ac_kegg.pkl")
    
#     # Query examples
#     hits = query_text("Cycloartenol", A, alias_map, species="all")
#     print("\nMatches:")
#     for hit in hits:
#         print(f"  {hit['match']} -> {hit['id']} ({hit['db']}, {hit['species']})")
