import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Text, JSON, Integer, TIMESTAMP, text
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
print("DATABASE_URL:", DATABASE_URL)
# ----------------------------------
# Async Engine
# ----------------------------------

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

# ----------------------------------
# Models
# ----------------------------------

class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True)
    embedding = Column(Vector(512))
    product_metadata = Column("metadata", JSON)


class ConversationMemory(Base):
    __tablename__ = "conversation_memories"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String)
    user_id = Column(String)
    content = Column(Text)
    embedding = Column(Vector(512))
    timestamp = Column(TIMESTAMP, server_default=func.now())


# ----------------------------------
# Initialize Database
# ----------------------------------

async def init_db():
    async with engine.begin() as conn:
        # Ensure extension exists
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

        # Optional cosine index
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS products_embedding_idx
            ON products
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))

    print("✅ Async DB initialized.")


# ----------------------------------
# Insert Product
# ----------------------------------

async def insert_product(product_id, embedding, metadata):
    async with AsyncSessionLocal() as session:
        product = Product(
            id=product_id,
            embedding=embedding,
            product_metadata=metadata
        )
        session.add(product)
        await session.commit()


# ----------------------------------
# Search Products (Cosine)
# ----------------------------------

async def search_products(query_embedding, limit=5):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT id,
                       metadata,
                       embedding <=> :embedding AS distance
                FROM products
                ORDER BY distance ASC
                LIMIT :limit
            """),
            {"embedding": query_embedding, "limit": limit},
        )

        return result.fetchall()


# ----------------------------------
# Search Memories
# ----------------------------------

async def search_memories(session_id, query_embedding, limit=5):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("""
                SELECT content,
                       embedding <=> :embedding AS distance
                FROM conversation_memories
                WHERE session_id = :session_id
                ORDER BY distance ASC
                LIMIT :limit
            """),
            {
                "embedding": query_embedding,
                "session_id": session_id,
                "limit": limit,
            },
        )

        return result.fetchall()
