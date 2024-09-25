from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever
from config import config
from database.vector_store import get_vectorstore
import logging

logger = logging.getLogger(__name__)
def get_retriever() -> BaseRetriever:
    """
    Creates and returns a simple retriever for debugging purposes.
    """
    vectorstore = get_vectorstore()
    print(f"Vectorstore initialized: {type(vectorstore)}")
    logger.info(f"Vectorstore initialized: {type(vectorstore)}")

    # Use a simple similarity search retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    print(f"Simple retriever initialized: {type(retriever)}")
    logger.info(f"Simple retriever initialized: {type(retriever)}")

    return retriever