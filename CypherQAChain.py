import os
import pandas as pd
import time
from langchain_community.llms import Ollama
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any
import threading

# --- 1. Connect to Neo4j Graph ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ZTjgBxsK3KczbEq6uyTa")

graph = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD
)

# Refresh schema
graph.refresh_schema()
print("Graph Schema loaded successfully")
print("="*60)

# --- 2. Initialize the Ollama LLM ---
OLLAMA_MODEL = "llama3.1:8b" 
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = Ollama(
    model=OLLAMA_MODEL, 
    base_url=OLLAMA_BASE_URL,
    temperature=0
)

# --- 3. Function to create fresh chain (KEY FIX!) ---

def create_fresh_chain(query_timeout=30):
    """Create chain with timeout for langchain_community version"""
    
    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template="""Generate safe Cypher queries:
            1. ALWAYS add 'LIMIT 10'
            2. NEVER use -[:REL*]- (bidirectional)
            3. ALWAYS specify direction: -[:REL*1..3]->

            Schema: {schema}
            Question: {question}
            Cypher:"""
        )
    
    # Create base chain
    base_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        allow_dangerous_requests=True,
        return_intermediate_steps=True,
        top_k=10,
        cypher_prompt=cypher_prompt
    )
    
    # Override the internal _call method (this is what invoke uses internally)
    if hasattr(base_chain, '_call'):
        original_call = base_chain._call
        
        def _call_with_timeout(inputs: Dict[str, Any], run_manager=None) -> Dict[str, Any]:
            """Internal call with timeout protection"""
            
            result_container = {"result": None, "error": None, "done": False}
            
            def execute():
                try:
                    result_container["result"] = original_call(inputs, run_manager=run_manager)
                except Exception as e:
                    result_container["error"] = str(e)
                finally:
                    result_container["done"] = True
            
            thread = threading.Thread(target=execute, daemon=True)
            thread.start()
            thread.join(timeout=query_timeout)
            
            # Handle timeout
            if not result_container["done"]:
                print(f"\n⏱️ TIMEOUT after {query_timeout}s")
                return {
                    "result": f"Query timeout ({query_timeout}s). Please rephrase.",
                    "intermediate_steps": []
                }
            
            # Handle error
            if result_container["error"]:
                print(f"\n❌ ERROR: {result_container['error']}")
                return {
                    "result": f"Query error: {result_container['error']}",
                    "intermediate_steps": []
                }
            
            # Success
            print(f"✓ Query completed")
            return result_container["result"]
        
        base_chain._call = _call_with_timeout
    
    return base_chain

# --- 4. Read CSV file ---
input_csv = "evaluation/pathway_evaluation_complete_with_result.csv"
df = pd.read_csv(input_csv)
question_column = "question"

# --- 5. Process each question and store results ---
results = []
cypher_queries = []

for idx, row in df.iterrows():
    question = row[question_column]
    print(f"\nRow {idx + 1}/{len(df)}: {question}")
    
    try:
        start_time = time.time() 
        cypher_qa = create_fresh_chain()
        response = cypher_qa.invoke({"query": question})
        elapsed_time = time.time() - start_time
        
        answer = response['result']
        cypher_query = response['intermediate_steps'][0]['query']
        
        print(f"✓ Success in {elapsed_time:.2f}s")
        print(f"Cypher: {cypher_query[:100]}...")  # Show first 100 chars
        
    except Exception as e:
        error_msg = str(e)
        answer = f"Error: {error_msg[:200]}"
        cypher_query = "ERROR - Invalid query generated"
        
        print(f"✗ Error: {error_msg[:150]}...")
        
        # Try to extract Cypher from error message
        if "Generated Cypher:" in error_msg:
            try:
                cypher_query = error_msg.split("Generated Cypher:")[1].split("\n")[0].strip()
            except:
                pass
    
    results.append(answer)
    cypher_queries.append(cypher_query)
    if (idx + 1) % 10 == 0:
        temp_df = df.iloc[:idx+1].copy()
        temp_df['QAChain'] = cypher_queries[:idx+1]
        temp_df['QAChain_Answer'] = results[:idx+1]
        temp_df.to_csv('evaluation/pathway_evaluation_progress_1.csv', index=False)
        print(f"Progress saved: {idx + 1}/{len(df)} rows")

df['QAChain'] = cypher_queries
df['QAChain_Answer'] = results

output_csv = "evaluation/pathway_evaluation_complete_with_result_QAChain.csv"
df.to_csv(output_csv, index=False)
