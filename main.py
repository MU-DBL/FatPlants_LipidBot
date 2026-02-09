import json
import os
import asyncio
from pathlib import Path
from cypher.ac import load_cache
from fastapi import FastAPI, HTTPException, Request, APIRouter
from lipidbot import classify_query_simple
from citation.search import search, get_cached_retrievers, get_cached_bm25
from config import HF_HOME, GEMINI_MODEL_NAME, GEMINI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OLLAMA_HOST, GPT_OSS_LLM_TYPE, LLAMA_LLM_TYPE
from llm_factory import LLMFactory
from data_service import LLMProvider
from cypher.cypher_query import cypher_query
from cypher.db_enginer import Neo4jClient
import logging
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import time
from data_service import SearchRequest, CypherQueryRequest, LipidBotRequest

# Set HF cache
os.environ["HF_HOME"] = HF_HOME

# Logger
logger = logging.getLogger("uvicorn.error")

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
    gemini_llm = LLMFactory.create_llm(
        provider=LLMProvider.GEMINI,
        model_name=GEMINI_MODEL_NAME,
        api_key=GEMINI_API_KEY
    )

    llama_llm = LLMFactory.create_llm(
        provider=LLMProvider.OLLAMA,
        model_name = LLAMA_LLM_TYPE,
        host = OLLAMA_HOST
    )

    gpt_oss_llm = LLMFactory.create_llm(
        provider=LLMProvider.OLLAMA,
        model_name = GPT_OSS_LLM_TYPE,
        host = OLLAMA_HOST
    )

    # app.state.gemini_llm = gemini_llm
    app.state.llama_llm = llama_llm
    app.state.gpt_oss_llm = gpt_oss_llm

    # Preload AI retrievers & BM25 caches
    get_cached_retrievers(None)
    get_cached_bm25()
    load_cache(None)
    print("[Startup] Models and BM25 cache preloaded.")

    yield

    # Shutdown
    await neo4j.close()
    print("[Shutdown] Neo4j connection closed.")

# =========================
# FastAPI App
# =========================
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI app and router
router = APIRouter()
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

    total_start = time.perf_counter()
    timings = {}
    
    try:
        # ===== INIT =====
        init_start = time.perf_counter()
        neo4j_client = request.app.state.neo4j
        llama_llm = request.app.state.llama_llm
        default_llm = (
            request.app.state.llama_llm 
            if req.llm_type == "llama" 
            else request.app.state.gpt_oss_llm
        )
        timings["init"] = time.perf_counter() - init_start
        
        # ===== CLASSIFICATION =====
        classify_start = time.perf_counter()
        classification = await asyncio.wait_for(
            classify_query_simple(req.query, default_llm),
            timeout=30.0
        )
        timings["classification"] = time.perf_counter() - classify_start
        
        logger.info(
            f"Classification: relevant={classification.is_relevant}, "
            f"needs_graph={classification.needs_graph}, "
            f"confidence={classification.confidence:.2f} | "
            f"{classification.reasoning}"
        )

        logger.info(f"[Classification] {timings['classification']:.3f}s")
        
        # ===== EARLY EXIT FOR IRRELEVANT QUERIES =====
        if not classification.is_relevant:
            return {
                "success": False,
                "answer": (
                    "I specialize in lipid biochemistry, metabolic pathways, genes, and enzymes. "
                    "Your question appears to be outside my domain. "
                    f"Reason: {classification.reasoning}"
                ),
                "classification": classification.dict(),
                "timings": timings
            }
        
        # ===== RETRIEVAL LOGIC =====
        needs_cypher = classification.needs_graph
        needs_citations = True  # Always get citations for relevant queries
        
        logger.info(f"Retrieval: Cypher={needs_cypher}, Citations={needs_citations}")
        
        # ===== PARALLEL RETRIEVAL =====
        async def run_cypher():
            if not needs_cypher:
                return [], ""
            
            cypher_start = time.perf_counter()
            result, cypher = await asyncio.wait_for(
                cypher_query(req.query, llama_llm, neo4j_client),
                timeout=40.0
            )
            timings["cypher_query"] = time.perf_counter() - cypher_start
            logger.info(f"[Timing] Cypher: {timings['cypher_query']:.3f}s")
            return result, cypher
        
        async def run_citation_search():
            if not needs_citations:
                return []
            
            search_start = time.perf_counter()
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
            timings["citation_search"] = time.perf_counter() - search_start
            logger.info(f"[Timing] Citations: {timings['citation_search']:.3f}s")
            return hits
        
        # Run in parallel
        (cypher_result, cypher_query_text), hits = await asyncio.gather(
            run_cypher(),
            run_citation_search()
        )
        
        # ===== FORMAT CITATIONS =====
        formatted_citations = ""
        if hits:
            citation_parts = []
            for i, hit in enumerate(hits[:10], 1):
                id_key = (
                    f"OpenAlex:{hit.citation_id}" 
                    if hit.citation_id.lower().startswith("w") 
                    else f"PubMed:{hit.citation_id}"
                )
                citation_parts.append(
                    f"SOURCE_ID: {id_key}\n"
                    f"TITLE: {hit.title}\n"
                    f"CONTENT: {hit.text}\n"
                )
            formatted_citations = "\n---\n".join(citation_parts)
        
        # ===== SYNTHESIZE RESPONSE =====
        synthesis_start = time.perf_counter()
        
        # Build context
        context_parts = []
        if cypher_result:
            context_parts.append(
                f"**Knowledge Graph Results:**\n"
                f"Query: {cypher_query_text}\n"
                f"```json\n{json.dumps(cypher_result, indent=2)}\n```"
            )
        if formatted_citations:
            context_parts.append(
                f"**Relevant Citations:**\n{formatted_citations}"
            )
        
        context = "\n\n".join(context_parts) if context_parts else "No data retrieved."
        
        synthesis_prompt = f"""You are a lipid biology expert assistant.

            **User Question:** "{req.query}"

            **Available Information:**
            {context}

            **Instructions:**
            1. Provide a direct, comprehensive answer to the user's question
            2. Integrate insights from both database results (if available) and citations
            3. Cite sources: [Graph] for database results, [1], [2], etc. for literature
            4. Be clear and actionable
            5. If information is limited, acknowledge gaps honestly

            **Your Response:**"""

        ai_response = await default_llm.agenerate(prompt=synthesis_prompt)
        
        timings["synthesis"] = time.perf_counter() - synthesis_start
        timings["total"] = time.perf_counter() - total_start
        
        # ===== RETURN =====
        logger.info(f"[Synthesis] {timings['synthesis']:.3f}s")
        logger.info(f"[Total] {timings['total']:.3f}s")
        
        return {
            "success": True,
            "answer": ai_response,
            # "classification": classification.dict(),
            # "sources": {
            #     "cypher_query": cypher_query_text,
            #     "cypher_result": cypher_result,
            #     "citations": formatted_citations,
            #     "retrieval_used": {
            #         "graph": needs_cypher,
            #         "citations": needs_citations
            #     }
            # },
            # "timings": timings
        }
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout for: {req.query}")
        raise HTTPException(504, "Request timeout")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))
