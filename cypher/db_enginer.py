from neo4j import AsyncGraphDatabase

class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.auth = (user, password)
        self.driver = None

    async def connect(self):
        # This creates the actual driver instance asynchronously
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth)
        await self.driver.verify_connectivity()

    async def close(self):
        if self.driver:
            await self.driver.close()

    async def run_query(self, cypher: str, params: dict | None = None):
        # Note: Use 'async with' for the session
        async with self.driver.session() as session:
            result = await session.run(cypher, params or {})
            # We must iterate over the result asynchronously
            return await result.data()
