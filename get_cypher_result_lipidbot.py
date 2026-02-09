import ast
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

llama_llm = LLMFactory.create_llm(
    provider=LLMProvider.OLLAMA, model_name=LLAMA_LLM_TYPE, host=OLLAMA_HOST
)

gpt_oss_llm = LLMFactory.create_llm(
    provider=LLMProvider.OLLAMA, model_name=GPT_OSS_LLM_TYPE, host=OLLAMA_HOST
)


async def main(file_path):

    neo4j = Neo4jClient(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

    await neo4j.connect()

    output_file = "file/lipidbot_cypher_results.csv"

    # df =pd.read_csv("file/pathway_evaluation_complete_with_result.csv")
    df = pd.read_csv(file_path)
    cypher_generator = SimpleCypherGenerator(llm=gpt_oss_llm)
    row_list = []
    results = []
    for _, row in df.iloc[0:].iterrows():

        question = row["question"]

        generated_cypher, metadata = cypher_generator.generate_query(question=question)

        print(f'{_}, {question}, {metadata["template_id"]}, {generated_cypher}')
        try:
            final_answer = await neo4j.run_query(cypher=generated_cypher)
            entity_ids = []
            if isinstance(final_answer, list):
                for item in final_answer:
                    if isinstance(item, dict):
                        node_values = list(item.values())[0]
                        if isinstance(node_values, dict):
                            eid = node_values.get("id")
                            if eid:
                                entity_ids.append(eid)

            final_answer = entity_ids if entity_ids else final_answer

        except Exception as e:
            final_answer = []
            print(e)
      
        results.append(
            {
                "phase": row["phase"],
                "category": row["category"],
                "question": question,
                "cypher_executable": row["cypher_executable"],
                "template_id": metadata["template_id"],
                "generated_cypher": generated_cypher,
                "answer": entity_ids,
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

    
# if __name__ == "__main__":
#     file_path = "file/pathway_evaluation_complete_with_result.csv"
#     # asyncio.run(main(file_path))
#     df = pd.read_csv("file/lipidbot_cypher_results.csv")
#     df["answer"] = df["answer"].astype(str).str[:1000]
#     df["if_correct"] = "1"
#     df.to_csv("lipidbot_cypher_results.csv", index=False)

#     df = pd.read_csv("file/QAChain_cypher_results.csv")
#     df["if_correct"] = "1"
#     df.to_csv("QAChain_cypher_results.csv", index=False)
