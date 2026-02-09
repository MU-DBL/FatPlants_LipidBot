import os
from pathlib import Path
import pandas as pd
from cypher.ac import load_cache
from config import (
    HF_HOME,
    GEMINI_MODEL_NAME,
    GEMINI_API_KEY,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    OLLAMA_HOST,
    GPT_OSS_LLM_TYPE,
    LLAMA_LLM_TYPE,
)
from llm_factory import LLMFactory
from data_service import LLMProvider
from cypher.db_enginer import Neo4jClient
from cypher.cypher_generator import SimpleCypherGenerator
import asyncio
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import ChatOllama
from langchain_neo4j import Neo4jGraph


async def langchain_cypher_generator(file_path):

    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USER, 
        password=NEO4J_PASSWORD
    )

    chat_llm = ChatOllama(model=GPT_OSS_LLM_TYPE, base_url=OLLAMA_HOST, temperature=0)


    chain = GraphCypherQAChain.from_llm(
        chat_llm,
        graph=graph,
        verbose=False,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )

    results = []
    output_file = "file/QAChain_cypher_results.csv"
    df = pd.read_csv(file_path)
    for _, row in df.iloc[169:].iterrows():

        question = row["question"]

        try:
            response = chain.invoke({"query": question})

            generated_cypher = response["intermediate_steps"][0]["query"]
            db_context = response["intermediate_steps"][1].get("context", [])

            entity_ids = []
            if isinstance(db_context, list):
                    for item in db_context:
                        if isinstance(item, dict):
                            node_values = list(item.values())[0]
                            if isinstance(node_values, dict):
                                eid = node_values.get("id")
                                if eid:
                                    entity_ids.append(eid)

            
            final_answer = entity_ids if entity_ids else response["result"]
            print(f'{_}, {question}, {generated_cypher}, {final_answer}')

        except Exception as e:
            final_answer = f"Exception: {type(e).__name__}"
            generated_cypher = "None"

        results.append(
            {
                "phase": row["phase"],
                "category": row["category"],
                "question": question,
                "cypher_executable": row["cypher_executable"],
                "generated_cypher": generated_cypher,
                "answer": final_answer,
            }
        )

        if (_ + 1) % 10 == 0:
            checkpoint_df = pd.DataFrame(results)
            header = not os.path.exists(output_file)
            checkpoint_df.to_csv(output_file, mode="a", index=False, header=header)
            results = []

    if results:
        pd.DataFrame(results).to_csv(
            output_file, mode="a", index=False, header=not os.path.exists(output_file)
        )


if __name__ == "__main__":
    file_path = "file/pathway_evaluation_complete_with_result.csv"
    asyncio.run(langchain_cypher_generator(file_path))
