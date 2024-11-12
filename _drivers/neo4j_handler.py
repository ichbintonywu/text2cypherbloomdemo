from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import streamlit as st
import json
from openai import AzureOpenAI
    
AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZUER_VERSION = st.secrets["AZUER_VERSION"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT"]
AZURE_MODEL_EMBEDDING = st.secrets["AZURE_EMBEDDING_MODEL"]

client = AzureOpenAI(
    api_key = AZURE_API_KEY,  
    api_version = AZUER_VERSION,
    azure_endpoint = AZURE_ENDPOINT
    ) 
deployment_name = AZURE_DEPLOYMENT_NAME

host = "neo4j+s://"+ st.secrets["NEO4J_HOST"]+":"+st.secrets["NEO4J_PORT"]
user = st.secrets["NEO4J_USER"]
password = st.secrets["NEO4J_PASSWORD"]
db = st.secrets["NEO4J_DB"]

URI = host
AUTH = (user, password)
driver = GraphDatabase.driver(URI, auth=AUTH)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=AZURE_MODEL_EMBEDDING).data[0].embedding

def gdsrun(query):
    gds = GraphDataScience(
        host,
        auth=(user, password))
    gds.set_database(db)
    return gds.run_cypher(query)

def gdsrun_db(query,indb):
    gds = GraphDataScience(
        host,
        auth=(user, password))
    gds.set_database(indb)
    return gds.run_cypher(query)

def pairwise_list(input_list):
    return [(input_list[i], input_list[i + 1]) for i in range(len(input_list) - 1)]

def do_cypher_tx(tx,cypher):
    results = tx.run(cypher)
    values = []
    for record in results:
        values.append(record.values())
    return values

# @st.cache_resource
def exec_cypher_query(qry_str):
    with driver.session() as session:
        result = session.execute_read(do_cypher_tx,qry_str)
        return result

def write_movie_tx(tx, label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property):
    merge_str ="""
    MATCH (n:""" +label1 + """ {"""+label1_prop_name+""":"""+"\""+label1_prop_property+"\""+"""})
    MATCH (m:""" +label2 + """ {"""+label2_prop_name+""":"""+"\""+label2_prop_property+"\""+"""})
    MERGE (n)"""+ \
    """-[r:"""+rel_type+""" {"""+rel_prop_name+""":'"""+rel_prop_property+ \
    """'}]-> (m)
    """

    query = (merge_str)
    print(query)
    result = tx.run(query, label1=label1,label1_prop_name=label1_prop_name,label1_prop_property=label1_prop_property,
                   label2=label2,label2_prop_name=label2_prop_name,label2_prop_property=label2_prop_property,
                   rel_type=rel_type,rel_prop_name=rel_prop_name,rel_prop_property=rel_prop_property)
    record = result.single()
    return "SUCCESS"

def simple_write_tx(tx, cypher_write):
    query = (cypher_write)
    result = tx.run(query, )
    record = result.single()
    return "SUCCESS"

def exec_cypher_write(label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property):
    with driver.session() as session:
        cpyher_write_result = session.execute_write(write_movie_tx, label1,label1_prop_name,label1_prop_property,
                   label2,label2_prop_name,label2_prop_property,
                   rel_type,rel_prop_name,rel_prop_property)
        return cpyher_write_result

def exec_simple_cypher_write(cypher_write):
 with driver.session() as session:
        cpyher_write_result = session.execute_write(simple_write_tx,cypher_write)
        return cpyher_write_result   