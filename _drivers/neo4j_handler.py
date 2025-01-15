from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import json
    
host = "neo4j+s://"+ st.secrets["NEO4J_HOST"]+":"+st.secrets["NEO4J_PORT"]
user = st.secrets["NEO4J_USER"]
password = st.secrets["NEO4J_PASSWORD"]
db = st.secrets["NEO4J_DB"]

URI = host
AUTH = (user, password)
driver = GraphDatabase.driver(URI, auth=AUTH)


def do_cypher_tx(tx,cypher):
    results = tx.run(cypher)
    values = []
    for record in results:
        values.append(record.values())
    return values
    
def gdsrun(query):
    gds = GraphDataScience(
        host,
        auth=(user, password))
    gds.set_database(db)
    return gds.run_cypher(query)

def exec_cypher_query(qry_str):
    with driver.session() as session:
        result = session.execute_read(do_cypher_tx,qry_str)
        return result
    
def write_cypher_query(qry_str):
    with driver.session() as session:
        result = session.execute_write(do_cypher_tx,qry_str)
        return result
