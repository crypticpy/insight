# config.py

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Config(BaseSettings):
    OPENAI_API_KEY: str
    LANGCHAIN_API_KEY: str
    PERSIST_DIRECTORY: str = "./data/chroma_db"
    COLLECTION_NAME: str = "rag_documents"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RELATED_TOPICS: int = 3
    OPENAI_MODEL_NAME: str = "gpt-4o-2024-08-06"
    OPENAI_VISION_MODEL: str = "gpt-4o-2024-08-06"
    TEMPERATURE: float = 0.1
    HYBRID_SEARCH_WEIGHT: float = 0.5  # Weight for combining dense and sparse search results
    RERANK_TOP_K: int = 10  # Number of documents to re-rank
    USE_SEMANTIC_SEARCH: bool = True  # Whether to use semantic search by default

    class Config:
        env_file = ".env"
        extra = "ignore"

config = Config()
