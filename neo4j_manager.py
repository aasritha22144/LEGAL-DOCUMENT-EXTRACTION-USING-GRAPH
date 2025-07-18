from neo4j import GraphDatabase

class Neo4jGraphManager:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self._driver.close()
    
    def create_entity(self, entity):
        with self._driver.session() as session:
            session.write_transaction(self._create_entity_node, entity)
    
    @staticmethod
    def _create_entity_node(tx, entity):
        query = """
        MERGE (e:LegalEntity {name: $name, type: $type})
        ON CREATE SET e.id = randomUUID()
        RETURN e
        """
        tx.run(query, name=entity['word'], type=entity['entity'])

