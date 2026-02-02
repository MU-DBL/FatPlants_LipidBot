import json
import os
import asyncio
from pathlib import Path
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException, Request, APIRouter
from pydantic import BaseModel
from citation.search import search, get_cached_retrievers, get_cached_bm25
from config import HF_HOME, GEMINI_MODEL_NAME, GEMINI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from llm_factory import LLMFactory
from data_service import LLMProvider
from cypher.cypher_query import cypher_query
from cypher.db_enginer import Neo4jClient
import logging
from contextlib import asynccontextmanager

# Set HF cache
os.environ["HF_HOME"] = HF_HOME

# Logger
logger = logging.getLogger("uvicorn.error")

# FastAPI app and router
router = APIRouter()

# =========================
# Pydantic Models
# =========================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    fuse: Literal["rrf", "vote", "max"] = "rrf"
    per: Literal["chunk"] = "chunk"
    rrf_k: int = 60
    model_names: Optional[List[str]] = None

class CypherQueryRequest(BaseModel):
    query: str

class LipidBotRequest(BaseModel):
    query: str
    top_k: int = 5
    fuse: Literal["rrf", "vote", "max"] = "rrf"
    per: Literal["chunk"] = "chunk"
    rrf_k: int = 60
    model_names: Optional[List[str]] = None

# =========================
# FastAPI Lifespan (startup / shutdown)
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Neo4j
    neo4j = Neo4jClient(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD
    )
    await neo4j.connect()
    app.state.neo4j = neo4j

    # Initialize LLM
    llm = LLMFactory.create_llm(
        provider=LLMProvider.GEMINI,
        model_name=GEMINI_MODEL_NAME,
        api_key=GEMINI_API_KEY
    )
    app.state.llm = llm

    # Preload AI retrievers & BM25 caches
    get_cached_retrievers(None)
    get_cached_bm25()
    print("[Startup] Models and BM25 cache preloaded.")

    yield

    # Shutdown
    await neo4j.close()
    print("[Shutdown] Neo4j connection closed.")

# =========================
# FastAPI App
# =========================
app = FastAPI(lifespan=lifespan)
app.include_router(router)

# =========================
# Health Check
# =========================
@app.get("/healthz")
def healthz():
    return {"ok": True}

# =========================
# Citation Search Endpoint
# =========================
@app.post("/citation_search")
def citation_search(req: SearchRequest):
    try:
        hits = search(
            query=req.query,
            top_k_per_model=req.top_k,
            fuse=req.fuse,
            per=req.per,
            rrf_k=req.rrf_k,
            model_names=req.model_names,
            add_bm25=True
        )
        return {"hits": [h.__dict__ for h in hits]}
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# =========================
# Cypher Query Endpoint
# =========================
@app.post("/cypher_query")
async def execute_cypher_query(req: CypherQueryRequest, request: Request):
    try:
        neo4j_client = request.app.state.neo4j
        llm = request.app.state.llm
        result, cypher = await asyncio.wait_for(
            asyncio.to_thread(
                cypher_query,
                req.query,  # consistent with Pydantic model
                llm,
                neo4j_client
            ),
            timeout=60.0
        )
        return {
            "success": True,
            "data": result,
            "cypher_query": cypher,
            "query": req.query
        }
    except asyncio.TimeoutError:
        logger.error(f"Query timeout for: {req.query}")
        raise HTTPException(504, "Query exceeded 60 second timeout")
    except Exception as e:
        logger.error(f"Cypher query failed: {e}", exc_info=True)
        raise HTTPException(500, f"Cypher query failed: {str(e)}")

# =========================
# LipidBot Endpoint
# =========================
@app.post("/lipidbot")
async def lipidbot(req: LipidBotRequest, request: Request):
    try:
        neo4j_client = request.app.state.neo4j
        llm = request.app.state.llm

        # 1) Run Cypher query
        result, cypher = await asyncio.wait_for(
            cypher_query(req.query, llm, neo4j_client),
            timeout=60.0
        )

        # 2) Run citation search
        hits = await asyncio.to_thread(
            search,
            query=req.query,
            top_k_per_model=req.top_k,
            fuse=req.fuse,
            per=req.per,
            rrf_k=req.rrf_k,
            model_names=req.model_names,
            add_bm25=True
        )

        # 3) Format hits for LLM consumption
        citations = []
        for i, hit in enumerate(hits[:10], 1):  # Limit to top 10
            citations.append({
                "id": i,
                "text": getattr(hit, 'text', str(hit)),
                "source": getattr(hit, 'source', 'Unknown'),
                "score": getattr(hit, 'score', 0)
            })

        # 4) Create prompt for AI agent
        prompt = f"""You are a lipid biology expert assistant. A user asked: "{req.query}"

            We have two sources of information:

            **Database Query Results (Cypher):**
            {json.dumps(result, indent=2)}

            **Relevant Citations:**
            {json.dumps(citations, indent=2)}

            Please provide a comprehensive answer that:
            1. Directly answers the user's question
            2. Combines insights from both the database and citations
            3. Cites specific sources when making claims (use [1], [2], etc.)
            4. Is clear and actionable

            Format your response in a user-friendly way."""

        # 5) Call LLM to synthesize response
        ai_response = await llm.agenerate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Lower for more factual responses
        )
        
        formatted_answer = ai_response.generations[0][0].text

        # 6) Return structured response
        return {
            "success": True,
            "answer": formatted_answer,
            "sources": {
                "cypher_query": cypher,
                "cypher_result": result,
                "citations": citations
            }
        }
    except asyncio.TimeoutError:
        logger.error(f"LipidBot timeout for: {req.query}")
        raise HTTPException(504, "Request exceeded 60 second timeout")
    except Exception as e:
        logger.error(f"LipidBot failed: {e}", exc_info=True)
        raise HTTPException(500, f"LipidBot failed: {str(e)}")
