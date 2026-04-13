import os
import asyncio
import threading
from pathlib import Path
import uuid
from cypher.ac import load_cache
from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import StreamingResponse
from lipidbot import classify_query_simple, flatten_row,openai_chunk, safe_str
from citation.search import search, get_cached_retrievers, get_cached_bm25
from config import GEMINI_MODEL_NAME, GEMINI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OLLAMA_HOST, GPT_OSS_LLM_TYPE, LLAMA_LLM_TYPE, OPENROUTER_API_KEY
from llm_factory import LLMFactory
from data_service import LLMProvider
from cypher.cypher_query import cypher_query
from cypher.db_enginer import Neo4jClient
import logging
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import time
from data_service import LipidBotRequest

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
    # gemini_llm = LLMFactory.create_llm(
    #     provider=LLMProvider.GEMINI,
    #     model_name=GEMINI_MODEL_NAME,
    #     api_key=GEMINI_API_KEY
    # )

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

    # gpt_oss_llm = LLMFactory.create_llm(
    #     provider=LLMProvider.OPENROUTER,
    #     model_name = "openai/gpt-oss-20b",
    #     api_key = OPENROUTER_API_KEY,
    #     temperature=0.2
    # )

    # llama_llm = LLMFactory.create_llm(
    #     provider=LLMProvider.OPENROUTER,
    #     model_name = "meta-llama/llama-3.1-8b-instruct",
    #     api_key = OPENROUTER_API_KEY
    # )


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
    allow_origins=["https://fatplants.net", "https://ec2-100-31-63-120.compute-1.amazonaws.com", "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting: 100 requests per IP per day
_RATE_LIMIT = 100
_RATE_WINDOW = 86400  # seconds in 24 hours
_rate_store: dict = {}  # ip -> [count, reset_at]


# FastAPI app and router
router = APIRouter()
app.include_router(router)

# =========================
# Health Check
# =========================
@app.get("/health")
def health():
    return {"ok": True}

# =========================
# Cypher Query Endpoint
# =========================
@app.get("/lipidbot/cypher/")
async def run_cypher_query(query: str, request: Request):
    neo4j_client: Neo4jClient = request.app.state.neo4j
    records = await neo4j_client.run_query(query)
    return {"results": records}

# =========================
# LipidBot Streaming Endpoint
# =========================

@app.post("/lipidbot/stream")
async def lipidbot_stream(req: LipidBotRequest, request: Request):
    """Same as /lipidbot but streams the synthesis response token by token via SSE."""

    # ===== RATE LIMIT =====
    ip = req.client_ip or "unknown"
    now = time.time()
    entry = _rate_store.get(ip)
    if entry is None or now >= entry[1]:
        _rate_store[ip] = [1, now + _RATE_WINDOW]
    else:
        entry[0] += 1
        if entry[0] > _RATE_LIMIT:
            reset_in = int(entry[1] - now)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Resets in {reset_in}s.",
                headers={"Retry-After": str(reset_in)},
            )

    try:
        t0 = time.perf_counter()

        # ===== INIT =====
        default_llm = (
            request.app.state.llama_llm
            if req.llm_type == "llama"
            else request.app.state.gpt_oss_llm
        )

        neo4j_client = request.app.state.neo4j

        # ===== CLASSIFICATION + CITATIONS (parallel) =====
        t_parallel = time.perf_counter()
        classification, hits = await asyncio.gather(
            asyncio.to_thread(classify_query_simple, req.query, default_llm),
            asyncio.to_thread(
                search,
                query=req.query,
                top_k_per_model=req.top_k,
                fuse=req.fuse,
                per=req.per,
                rrf_k=req.rrf_k,
                model_names=req.model_names,
                add_bm25=True
            )
        )
        logger.info(
            f"[Timing] parallel(classification+citations)={time.perf_counter()-t_parallel:.3f}s"
            f"  relevant={classification.is_relevant} needs_graph={classification.needs_graph}"
            f"  hits={len(hits) if hits else 0}"
        )

        if not classification.is_relevant:
            _sid = f"chatcmpl-{uuid.uuid4().hex}"
            async def irrelevant_stream():
                msg = (
                    "I specialize in lipid biochemistry, metabolic pathways, genes, and enzymes. "
                    "Your question appears to be outside my domain."
                )
                yield openai_chunk(msg, _sid)
                yield openai_chunk("", _sid, finish=True)
                yield "data: [DONE]\n\n"
            return StreamingResponse(irrelevant_stream(), media_type="text/event-stream")

        # ===== RETRIEVAL =====
        async def run_cypher(needs_graph: bool):
            if not needs_graph:
                return [], ""
            result, cypher = await asyncio.wait_for(
                cypher_query(req.query, default_llm, neo4j_client),
                timeout=40.0
            )
            return result, cypher

        # Fire cypher in the background — it will run while LLM streams
        t_cypher_start = time.perf_counter()
        cypher_task = asyncio.create_task(run_cypher(classification.needs_graph))

        # ===== FORMAT CITATIONS =====
        formatted_citations = ""
        citation_list = []
        if hits:
            citation_parts = []
            for i, hit in enumerate(hits[:10], 1):
                id_key = (
                    f"OpenAlex:{hit.citation_id}"
                    if hit.citation_id.lower().startswith("w")
                    else f"PubMed:{hit.citation_id}"
                )
                citation_parts.append(
                    f"[{i}]\n"
                    f"SOURCE_ID: {id_key}\n"
                    f"TITLE: {hit.title}\n"
                    f"CONTENT:\n{hit.text}\n"
                )
                citation_list.append({
                    'number': i,
                    'id': id_key,
                    'title': hit.title,
                    'citation_id': hit.citation_id,
                })
            formatted_citations = "\n---\n".join(citation_parts)

        # ===== BUILD PROMPT =====
        context_parts = []
        if formatted_citations:
            context_parts.append(
                f"**Literature Sources:**\n"
                f"Cite these sources in your answer using [1], [2], [3] etc.\n"
                f"{formatted_citations}"
            )
        context = "\n\n".join(context_parts) if context_parts else "No data retrieved."

        synthesis_prompt = f"""You are a lipid biology expert assistant.

**User Question:** "{req.query}"

**Available Information:**
{context}

**CITATION PROTOCOL - READ CAREFULLY:**

Before adding ANY citation [N]:
1. Check: Is this fact explicitly stated in source [N]?
2. If YES → Add citation [N]
3. If NO → Either:
   - Find the correct source that states it
   - Or mention it without citation and note it's from general knowledge

**FORBIDDEN:**
❌ Citing sources based on relevance/topic match
❌ Adding citations to sound more authoritative
❌ Guessing which source might support a claim
❌ Citing [1] just because it's the first source

**REQUIRED:**
✓ Only cite when you can point to the exact sentence in that source
✓ Use phrases like "According to [1]..." when directly referencing
✓ Say "General lipid biology knowledge suggests..." for non-cited claims

**Instructions:**
1. Answer the user's question directly and comprehensively
2. **For each fact you state, mentally check: "Is this from a numbered source above? Which one exactly?"**
3. Only add [N] if you can trace the fact to that specific source
4. Be specific with enzyme names, gene IDs, and quantitative data
5. Acknowledge gaps honestly

**Your Response:**"""

        # ===== BUILD REFERENCES TEXT =====
        references_text = ""
        if citation_list:
            refs = "\n\n**Relevant references:**\n\n"
            for cite in citation_list:
                if cite['id'].startswith('PubMed:'):
                    pmid = cite['id'].replace('PubMed:', '')
                    link = f"[{cite['id']}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                elif cite['id'].startswith('OpenAlex:'):
                    oa_id = cite['id'].replace('OpenAlex:', '')
                    link = f"[{cite['id']}](https://openalex.org/{oa_id})"
                else:
                    link = cite['id']
                refs += f"[{cite['number']}] {cite['title']} {link}\n\n"
            references_text = refs

    except Exception as e:
        logger.error(f"[Stream] Pre-processing error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    # ===== STREAMING GENERATOR =====
    # Streams plain text tokens via SSE.  Client reads each "data: " line
    # and concatenates the values; references are appended as text at the end.
    # Special sentinel:  data: [DONE]  signals end of stream.
    # Async generator: LLM streams in a thread via queue while cypher_task
    # runs concurrently in the event loop. Cypher result is appended after
    # LLM finishes — no extra wait time for the user.
    async def stream_synthesis():
        stream_id = f"chatcmpl-{uuid.uuid4().hex}"
        try:
            # Bridge sync generate_stream → async via queue + thread
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def run_llm():
                try:
                    for token in default_llm.generate_stream(prompt=synthesis_prompt):
                        asyncio.run_coroutine_threadsafe(queue.put(token), loop).result()
                except Exception as exc:
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

            t_llm_start = time.perf_counter()
            threading.Thread(target=run_llm, daemon=True).start()

            # ---- stream LLM tokens while cypher runs in the background ----
            first_token = True
            while True:
                item = await queue.get()
                if item is None:        # sentinel — LLM done
                    logger.info(f"[Timing] llm_stream={time.perf_counter()-t_llm_start:.3f}s  total={time.perf_counter()-t0:.3f}s")
                    break
                if isinstance(item, Exception):
                    yield f"data: [ERROR] {str(item)}\n\n"
                    cypher_task.cancel()
                    return
                if first_token:
                    logger.info(f"[Timing] ttft={time.perf_counter()-t0:.3f}s")
                    first_token = False
                yield openai_chunk(item, stream_id)

            # ---- references block ----
            if references_text:
                yield openai_chunk("\n" + references_text + "\n", stream_id)

            # ---- cypher result (await the task that ran concurrently) ----
            cypher_result, cypher_query_text = await cypher_task
            logger.info(f"[Timing] cypher={time.perf_counter()-t_cypher_start:.3f}s, query={cypher_query_text}")
            if cypher_result:
                rows = [flatten_row(r) for r in cypher_result]

                # collect union of all columns
                all_keys = []
                for r in rows:
                    for k in r.keys():
                        if k not in all_keys:
                            all_keys.append(k)

                 # ---- markdown table ----
                header = "| " + " | ".join(all_keys) + " |"
                sep    = "| " + " | ".join("---" for _ in all_keys) + " |"

                body = "\n".join(
                    "| " + " | ".join(safe_str(r.get(k, "")) for k in all_keys) + " |"
                    for r in rows
                )

                cypher_block = (
                    f"\n\n**Pathway Graph Results (limited to 10 results):**\n\n"
                    f"{header}\n{sep}\n{body}\n"
                )

                yield openai_chunk(cypher_block, stream_id)

            # ---- finish ----
            yield openai_chunk("", stream_id, finish=True)
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(stream_synthesis(), media_type="text/event-stream")
