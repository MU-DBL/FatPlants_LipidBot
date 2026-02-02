from neo4j import GraphDatabase

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, cypher: str, params: dict | None = None):
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [r.data() for r in result]
