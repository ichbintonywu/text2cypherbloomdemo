from  langchain_ollama  import OllamaLLM
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
import os
import streamlit as st
import re
from langchain_core.prompts.prompt import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from langchain.chains import GraphCypherQAChain
from langchain_openai import  AzureChatOpenAI
from langchain_community.cache import InMemoryCache
from langchain_community.cache import RedisCache
import redis
from langchain.globals import set_llm_cache

REDIS_URL = st.secrets["REDIS_URL"]

r = redis.Redis(host="localhost", port=6379, username="default", password="admin", db=0)
print(r.ping())

print(f"Connecting to Redis at: {REDIS_URL}")
redis_cache = RedisCache(r)
set_llm_cache(redis_cache)

NEO4J_HOST = st.secrets["NEO4J_AURA_YT"]+":"+st.secrets["NEO4J_PORT"]
NEO4J_USER = st.secrets["NEO4J_AURA_YT_USER"]
NEO4J_PASSWORD = st.secrets["NEO4J_AURA_YT_PASSWORD"]
AZURE_VERSION = st.secrets["AZURE_VERSION"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_4ODEPLOYMENT = st.secrets["AZURE_DEPLOYMENT"]
AZURE_API_KEY = st.secrets["AZURE_APIKEY"]
os.environ["NEO4J_URI"] = NEO4J_HOST
os.environ["NEO4J_USERNAME"] = NEO4J_USER
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
OLLAMA_MODEL = st.secrets["LLAMA_MODEL"]
OLLAMA_URL = st.secrets["LLAMA_BASE_URL"]

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"], username=os.environ["NEO4J_USERNAME"], password=os.environ["NEO4J_PASSWORD"]
)

llm = OllamaLLM(
    model = OLLAMA_MODEL,
    temperature = 0,
    base_url = OLLAMA_URL
)

api_version = AZURE_VERSION
endpoint = AZURE_ENDPOINT
gptdeployment = AZURE_4ODEPLOYMENT
api_key = AZURE_API_KEY

openaillm = AzureChatOpenAI(
    api_version=api_version,
    model=gptdeployment,
    api_key = api_key,
    azure_endpoint=endpoint,
    temperature=0.1,
    max_tokens = 3000
)

CYPHER_GENERATION_TEMPLATE = """
Task: Generate Cypher queries to query the graph database from the input question.
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:
# What app groups are present in the Department IT?
MATCH (d:DEPARTMENT {{name: "IT"}})<-[:DEPARTMENT_OF]-(ag:APPLICATION_GROUP) RETURN ag;

# What Departments are present in the Organisation Ack
MATCH (o:ORGANIZATION {{name: "Ack"}})<-[:ORGANIZATION_BY]-(d:DEPARTMENT) RETURN d.name AS departmentName;

# Clients have Transaction with FirstPartyFraudster return the entire path
MATCH path = (c:Client)-[:PERFORMED]->(t:Transaction)-[:TO]->(f:FirstPartyFraudster) RETURN path;

# FirstPartyFraudster transfer to other Clients, return the entire path?
MATCH path = (f:FirstPartyFraudster)-[:TRANSFER_TO]->(c:Client) RETURN path;

# Identify clients sharing PII
MATCH p=(c1:Client)-[:SHARED_IDENTIFIERS]->(x)<-[:SHARED_IDENTIFIERS]-(c2:Client) RETURN p;

The question is:
{question}
"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)
# Define function to trim Cypher query
def trim_query(query: str) -> str:
    """Trim the query to only include Cypher keywords."""
    keywords = (
        "CALL", "CREATE", "DELETE", "DETACH", "LIMIT", "MATCH", "MERGE",
        "OPTIONAL", "ORDER", "REMOVE", "RETURN", "SET", "SKIP", "UNWIND",
        "WITH", "WHERE", "//"
    )
    lines = query.split("\n")
    new_query = ""
    for line in lines:
        if line.strip().upper().startswith(keywords):
            new_query += line + "\n"
    return new_query.strip()

# Define function to extract Cypher code from text
def extract_cypher(text: str) -> str:
    """Extract Cypher code from text using Regex."""
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else text.strip()

# Define your custom chain class
class CustomGraphCypherQAChain(GraphCypherQAChain):
    captured_cypher: Optional[str] = None
    captured_context: Optional[str] = None

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate Cypher statement and context without executing the query."""
        _run_manager = run_manager or Any()
        question = inputs[self.input_key]

        intermediate_steps: List[Dict[str, Any]] = []

        # Ensure get_schema is used correctly
        schema = self.graph.get_schema() if callable(self.graph.get_schema) else self.graph.get_schema
        generated_cypher = self.cypher_generation_chain.run(
            {"question": question, "schema": schema}
        )

        # Clean up the generated Cypher
        generated_cypher = extract_cypher(generated_cypher)
        generated_cypher = trim_query(generated_cypher)

        intermediate_steps.append({"query": generated_cypher})

        # Capture the generated Cypher and context for return
        self.captured_cypher = generated_cypher
        self.captured_context = "Context placeholder"  # Replace with actual context extraction logic if needed

        # Example: Extracting context from generated Cypher query
        context = self.graph.query(generated_cypher)[: self.top_k]
        self.captured_context = context

        # Terminate the function as soon as valid values are generated
        if generated_cypher and context:
            return {self.output_key: {"cypher": self.captured_cypher, "context": self.captured_context}}
        else:
            return {"error": "Failed to generate valid Cypher query and context"}

    def get_captured_cypher_and_context(self):
        return self.captured_cypher, self.captured_context

# Instantiate the custom chain
def create_chain():
    return CustomGraphCypherQAChain.from_llm(
        graph=graph,
        # llm=llm,
        llm = openaillm,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
        verbose=True,
        validate_cypher=True,
        return_intermediate_steps=False,allow_dangerous_requests=True  # Ensure intermediate steps are not returned
    )

# Define the function to run a chain with a given question
def run_chain(chain, question):
    try:
        result = chain.invoke(question)['result']
        generated_cypher = result.get("cypher")
        context = result.get("context")
        if generated_cypher and context:
            return generated_cypher, context
    except Exception as e:
        print(f"Error invoking chain: {e}")
    return None, None

# Define the main function to run multiple chains in parallel
def run_parallel_chains(question, timeout=60):
    chains = [create_chain() for _ in range(5)]  # Adjust the number of chains as needed
    with ThreadPoolExecutor() as executor:
        future_to_chain = {executor.submit(run_chain, chain, question): chain for chain in chains}
        for future in as_completed(future_to_chain, timeout=timeout):
            result = future.result()
            if result and result[0] and result[1]:
                return result
    return "No valid response found within the time limit."

def custom_llama_qahelper(user_question):

    questions = user_question
    valid_cypher, valid_context, *_ = run_parallel_chains(questions)

    if valid_cypher and valid_context:
        print("Valid Cypher Query:")
        print(valid_cypher)
        print("\nValid Context:")
        print(valid_context)
        return valid_cypher,valid_context
    else:
        print("No valid Cypher query and context found within the specified iterations.")
