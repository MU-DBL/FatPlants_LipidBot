GEMINI_API_KEY = "AIzaSyDvRPqufbcsNMhxKkNj0qBuwheuJPxaSn8"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
OLLAMA_LLM = "llama3.1"

NEO4J_URI = "bolt://digbio-xugpu-3.missouri.edu:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ZTjgBxsK3KczbEq6uyTa"

AC_KEGG_PKL = "/cluster/pixstor/xudong-lab/yongfang/fatplants_lipidbot/lipidbot/cypher/ac_kegg.pkl"
CITATION_DIR = "/cluster/pixstor/xudong-lab/yongfang/fatplants_lipidbot/lipidbot/citation/citation_cache"
HF_HOME = "/cluster/pixstor/xudong-lab/yongfang/fatplants_lipidbot/rag_ai/fatplants_papers/results/hf_cache"
BM25_CACHE = f"{CITATION_DIR}/bm25_index.pkl"

DEFAULT_EMBEDDING_MODEL = [
    "NeuML/pubmedbert-base-embeddings",
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "BAAI/bge-m3",
]

# === 평가 코드(evaluation.py) 호환용 변수 추가 ===
bm25_cache_file = BM25_CACHE
default_model_name = DEFAULT_EMBEDDING_MODEL
