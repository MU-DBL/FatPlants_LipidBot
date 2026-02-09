import time
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
import concurrent.futures 
from cypher.ac import load_cache
from cypher.llm_cypher_generator import LLMCypherQueryGenerator
from cypher.entity_extractor import BioEntityExtractor
from llm_factory import BaseLLM
from cypher.db_enginer import Neo4jClient
from cypher.cypher_generator import SimpleCypherGenerator

async def cypher_query(question: str, llm:BaseLLM, neo4j_client: Neo4jClient):

    entity_extractor = BioEntityExtractor(llm=llm)
    # llm_query_engine = LLMCypherQueryGenerator(llm=llm)
    cypher_generator = SimpleCypherGenerator(llm=llm)

    entities = entity_extractor.extract_mentions(question) 
    # print(entities)
    # cypher, source = llm_query_engine.generate_query(
    #             question=question,
    #             entities=entities
    # )

    try:
        cypher_query, metadata = cypher_generator.generate_query(question=question)
        print(cypher_query)
        result = await neo4j_client.run_query(cypher=cypher_query)

    except Exception as e:
        print(e)
        return None, None

    return result, cypher_query

 