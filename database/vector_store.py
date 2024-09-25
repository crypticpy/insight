# database/vector_store.py
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import config
import logging
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import uuid

nltk.download('punkt')

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    source: str
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None


class VectorStoreManager:
    """Manages interactions with the vector database."""

    def __init__(self):
        self._vectorstore = None
        self._embeddings = None
        self._tfidf_vectorizer = None
        self._bm25 = None
        self._cross_encoder = None
        self._contextual_tokenizer = None
        self._contextual_model = None
        self._initialize()

    def _initialize(self):
        """Initializes the vector store, embeddings, and other components."""
        logger.info("Initializing VectorStoreManager")
        try:
            self._embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
            self._vectorstore = Chroma(
                persist_directory=config.PERSIST_DIRECTORY,
                embedding_function=self._embeddings,
                collection_name=config.COLLECTION_NAME,
            )
            self._tfidf_vectorizer = TfidfVectorizer()
            self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self._contextual_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self._contextual_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self._initialize_sparse_retrieval()
            logger.info(f"Vectorstore initialized. Type: {type(self._vectorstore)}")
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {str(e)}")
            raise

    def _initialize_sparse_retrieval(self):
        """Initializes sparse retrieval components."""
        documents = self.get_all_documents()
        corpus = [doc.page_content for doc in documents]
        self._tfidf_vectorizer.fit(corpus)
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def get_vectorstore(self):
        if self._vectorstore is None:
            self._initialize()
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Adds a list of documents to the vector store and updates sparse retrieval components."""
        logger.info(f"Adding {len(documents)} documents to vectorstore")
        try:
            doc_ids = self._vectorstore.add_documents(documents)
            self._initialize_sparse_retrieval()  # Re-initialize sparse retrieval after adding new documents
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def _get_contextual_embedding(self, text: str) -> np.ndarray:
        """Generates contextual embeddings for a given text."""
        inputs = self._contextual_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = self._contextual_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def hybrid_search(
            self, query: str, k: int = 10, filter: Optional[Dict[str, str]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Performs a hybrid search combining dense and sparse retrieval methods.
        """
        logger.info(f"Performing hybrid search for query: {query}")
        try:
            # Dense retrieval
            dense_results = self._vectorstore.similarity_search_with_score(query, k=k * 2, filter=filter)

            # Filter out results with None content
            dense_results = [(doc, score) for doc, score in dense_results if doc.page_content is not None]

            # If no results from dense retrieval, return empty list
            if not dense_results:
                logger.warning(f"No valid results found for query: {query}")
                return []

            # Sparse retrieval
            tfidf_vector = self._tfidf_vectorizer.transform([query])
            bm25_scores = self._bm25.get_scores(query.lower().split())

            # Combine results
            all_docs = {doc.page_content: (doc, score) for doc, score in dense_results}

            # Get all document IDs from the vectorstore
            all_ids = self._vectorstore.get()["ids"]

            for doc_id, score in zip(all_ids, bm25_scores):
                if len(all_docs) >= k * 3:
                    break
                doc = self.get_document_by_id(doc_id)
                if doc and doc.page_content and doc.page_content not in all_docs:
                    all_docs[doc.page_content] = (doc, score)

            # If no documents found after combining, return empty list
            if not all_docs:
                logger.warning(f"No valid documents found after combining results for query: {query}")
                return []

            # Re-rank using cross-encoder
            rerank_inputs = [(query, doc.page_content) for doc, _ in all_docs.values()]
            rerank_scores = self._cross_encoder.predict(rerank_inputs)

            # Final ranking
            ranked_results = sorted(zip(all_docs.values(), rerank_scores), key=lambda x: x[1], reverse=True)

            return [(doc, score) for (doc, _), score in ranked_results[:k]]
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}", exc_info=True)
            return []  # Return an empty list in case of an error

    def get_collection_stats(self) -> Dict[str, int]:
        """Retrieves statistics about the vector store collection."""
        logger.info("Getting collection stats")
        try:
            collection_data = self._vectorstore.get()
            stats = {
                "total_documents": len(collection_data["ids"]),
                "total_chunks": sum(
                    len(doc.split()) for doc in collection_data["documents"]
                ),
            }
            logger.info(f"Collection stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise

    def clear_vectorstore(self) -> bool:
        """Clears the vector store and reinitializes it."""
        logger.info("Clearing vectorstore")
        try:
            self._vectorstore.delete_collection()
            self._initialize()
            logger.info("Vectorstore cleared and reinitialized")
            return True
        except Exception as e:
            logger.error(f"Error clearing vectorstore: {str(e)}")
            raise

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieves a document from the vector store by its ID."""
        logger.info(f"Getting document by ID: {doc_id}")
        try:
            results = self._vectorstore.get([doc_id])
            if results and results["ids"] and results["documents"][0] is not None:
                logger.info("Document found")
                return Document(
                    page_content=results["documents"][0] or "",  # Use empty string if None
                    metadata=results["metadatas"][0] or {},
                )
            logger.warning(f"No valid document found with ID: {doc_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting document by ID: {str(e)}")
            return None

    def find_related_documents(
            self, query: str, n_results: int = 3
    ) -> List[Tuple[Document, float]]:
        """Finds documents related to the query using hybrid search."""
        logger.info(f"Finding related documents for query: {query}")
        try:
            return self.hybrid_search(query, k=n_results)
        except Exception as e:
            logger.error(f"Error finding related documents: {str(e)}")
            raise

    def update_document(
            self, doc_id: str, new_content: str, metadata: Optional[Dict] = None
    ) -> bool:
        """Updates a document in the vector store and re-initializes sparse retrieval components."""
        logger.info(f"Updating document with ID: {doc_id}")
        try:
            self._vectorstore.update_document(doc_id, new_content, metadata)
            self._initialize_sparse_retrieval()  # Re-initialize sparse retrieval after updating a document
            logger.info(f"Document {doc_id} updated successfully")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document from the vector store and re-initializes sparse retrieval components."""
        logger.info(f"Deleting document with ID: {doc_id}")
        try:
            self._vectorstore.delete([doc_id])
            self._initialize_sparse_retrieval()  # Re-initialize sparse retrieval after deleting a document
            logger.info(f"Document {doc_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    def get_all_documents(self) -> List[Document]:
        """Retrieves all documents from the vector store."""
        logger.info("Getting all documents from the vector store")
        try:
            collection_data = self._vectorstore.get(include=["documents", "metadatas"])
            documents = [
                Document(page_content=content, metadata=metadata)
                for content, metadata in zip(
                    collection_data["documents"], collection_data["metadatas"]
                )
            ]
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error getting all documents: {str(e)}")
            raise

    def semantic_search(
            self, query: str, k: int = 4, filter: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Performs a semantic search using contextual embeddings and re-ranking.
        """
        logger.info(f"Performing semantic search for query: {query}")
        try:
            query_embedding = self._get_contextual_embedding(query)

            # Initial retrieval
            results = self._vectorstore.similarity_search_by_vector_with_relevance_scores(query_embedding, k=k * 2,
                                                                                          filter=filter)

            # Re-rank using cross-encoder
            rerank_inputs = [(query, doc.page_content) for doc, _ in results]
            rerank_scores = self._cross_encoder.predict(rerank_inputs)

            # Final ranking
            ranked_results = sorted(zip([doc for doc, _ in results], rerank_scores), key=lambda x: x[1], reverse=True)

            return [doc for doc, _ in ranked_results[:k]]
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise


vector_store_manager = VectorStoreManager()


# Convenience functions
def get_vectorstore():
    """Returns the vector store instance."""
    logger.info("Getting vectorstore")
    vectorstore = vector_store_manager.get_vectorstore()
    print(f"Vectorstore created with {len(vectorstore.get()['ids'])} documents")
    return vectorstore


def add_documents_to_vectorstore(documents: List[Document]) -> List[str]:
    """Adds documents to the vector store using the manager."""
    try:
        # Ensure each document has a doc_id
        for doc in documents:
            if "doc_id" not in doc.metadata:
                doc.metadata["doc_id"] = str(uuid.uuid4())

        doc_ids = vector_store_manager.add_documents(documents)
        logger.info(f"Added {len(doc_ids)} documents to the vector store.")
        return doc_ids
    except Exception as e:
        logger.error(f"Error adding documents to vector store: {str(e)}")
        return []


def clean_vectorstore():
    """Removes documents with None content from the vector store."""
    vectorstore = get_vectorstore()
    collection_data = vectorstore.get()

    ids_to_delete = []
    for i, content in enumerate(collection_data["documents"]):
        if content is None:
            ids_to_delete.append(collection_data["ids"][i])

    if ids_to_delete:
        vectorstore.delete(ids_to_delete)
        logger.info(f"Removed {len(ids_to_delete)} documents with None content from the vector store")
    else:
        logger.info("No documents with None content found in the vector store")

    return len(ids_to_delete)

def get_document_by_id(doc_id: str) -> Optional[Document]:
    """Retrieves a document using the manager."""
    return vector_store_manager.get_document_by_id(doc_id)


def get_collection_stats() -> Dict[str, int]:
    """Gets collection stats using the manager."""
    return vector_store_manager.get_collection_stats()


def clear_vectorstore() -> bool:
    """Clears the vector store using the manager."""
    return vector_store_manager.clear_vectorstore()


def hybrid_search(query: str, k: int = 10, filter: Optional[Dict[str, str]] = None) -> List[Tuple[Document, float]]:
    return vector_store_manager.hybrid_search(query, k, filter)


def find_related_documents(
        query: str, n_results: int = 3
) -> List[Tuple[Document, float]]:
    """Finds related documents using the manager."""
    return vector_store_manager.find_related_documents(query, n_results)


def update_document(
        doc_id: str, new_content: str, metadata: Optional[Dict] = None
) -> bool:
    """Updates a document using the manager."""
    return vector_store_manager.update_document(doc_id, new_content, metadata)


def delete_document(doc_id: str) -> bool:
    """Deletes a document using the manager."""
    return vector_store_manager.delete_document(doc_id)


def get_all_documents() -> List[Document]:
    """Retrieves all documents using the manager."""
    return vector_store_manager.get_all_documents()


def semantic_search(
        query: str, k: int = 4, filter: Optional[Dict[str, str]] = None
) -> List[Document]:
    """Performs a semantic search using the manager."""
    return vector_store_manager.semantic_search(query, k, filter)
