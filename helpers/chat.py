# helpers/chat.py
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from database.vector_store import hybrid_search, find_related_documents, semantic_search
from helpers.retrieval import get_retriever
from helpers.topic_extractor import topic_extractor
from config import config
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StdOutCallbackHandler
import logging

logger = logging.getLogger(__name__)

class Citation(BaseModel):
    source_id: int = Field(..., description="The integer ID of the source document")
    quote: str = Field(..., description="The relevant quote from the source")
    document_id: Optional[str] = Field(None, description="The ID of the document in the vector store")
    url: Optional[str] = Field(None, description="The URL of the source, if available")
    title: Optional[str] = Field(None, description="The title of the source, if available")

class AnswerWithSources(BaseModel):
    answer: str = Field(..., description="The answer to the user's question")
    citations: List[Citation] = Field(default_factory=list, description="Citations supporting the answer")

def format_docs(docs: List[Tuple[Document, float]]) -> str:
    """Formats documents for the LLM, including source IDs, metadata, and relevance information."""
    formatted_docs = []
    for i, (doc, score) in enumerate(docs):
        formatted_doc = (
            f"Source {i+1}:\n"
            f"Content: {doc.page_content}\n"
            f"Document ID: {doc.metadata.get('doc_id', 'Unknown')}\n"
            f"Source: {doc.metadata.get('source', 'Unknown')}\n"
            f"Chunk Type: {doc.metadata.get('chunk_type', 'Unknown')}\n"
            f"Relevance Score: {score}"
        )
        formatted_docs.append(formatted_doc)
    return "\n\n".join(formatted_docs)


def get_chat_response(
        prompt: str, chat_history: List[Tuple[str, str]] = []
) -> Tuple[str, List[Dict], List[Dict], List[Tuple[Document, float]]]:
    print(f"Received prompt: {prompt}")
    print(f"Chat history length: {len(chat_history)}")
    logger.info(f"Received prompt: {prompt}")
    logger.info(f"Chat history length: {len(chat_history)}")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Add chat history to memory
    for human, ai in chat_history:
        memory.chat_memory.add_user_message(human)
        memory.chat_memory.add_ai_message(ai)

    print(f"Memory initialized with {len(chat_history)} messages")
    logger.info(f"Memory initialized with {len(chat_history)} messages")

    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.

    The retrieved context includes diverse sources that may cover different aspects of the question.
    Analyze all provided sources and synthesize a comprehensive answer.

    When citing sources, refer to them by their Source ID. 
    For example: "According to Source 1, ..."

    Consider the relevance score of each source when deciding how to use it in your answer.
    Prioritize more relevant sources, but don't ignore less relevant ones if they provide unique insights.

    If sources contradict each other, acknowledge this in your answer and explain the different viewpoints.

    Only cite sources that are provided in the context.
    If there are no relevant sources, do not include any citations.

    Context:
    {context}

    Chat History:
    {chat_history}

    Human: {question}

    Assistant: """

    prompt_template = ChatPromptTemplate.from_template(system_prompt)
    print("Prompt template created")
    logger.info("Prompt template created")

    llm = ChatOpenAI(
        temperature=config.TEMPERATURE,
        model_name=config.OPENAI_MODEL_NAME,
    )
    print(f"LLM initialized: {config.OPENAI_MODEL_NAME}")
    logger.info(f"LLM initialized: {config.OPENAI_MODEL_NAME}")

    def get_relevant_documents(query):
        print(f"Retrieving documents for query: {query}")
        logger.info(f"Retrieving documents for query: {query}")
        docs = hybrid_search(query, k=5)  # Use hybrid_search instead of retriever
        print(f"Retrieved {len(docs)} documents")
        logger.info(f"Retrieved {len(docs)} documents")
        for i, (doc, score) in enumerate(docs):
            print(f"Document {i + 1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            print(f"Score: {score}")
            print("---")
            logger.info(
                f"Document {i + 1}: ID={doc.metadata.get('doc_id', 'Unknown')}, Content={doc.page_content[:100]}..., Score={score}")
        return docs

    rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(get_relevant_documents(x["question"]))
            )
            | prompt_template
            | llm
    )
    print("RAG chain created")
    logger.info("RAG chain created")

    try:
        print("Starting document retrieval and context formatting")
        logger.info("Starting document retrieval and context formatting")
        retrieved_documents = get_relevant_documents(prompt)
        context = format_docs(retrieved_documents)
        print(f"Formatted context (first 500 chars): {context[:500]}...")
        logger.info(f"Formatted context (first 500 chars): {context[:500]}...")

        print("Invoking RAG chain")
        logger.info("Invoking RAG chain")
        response = rag_chain.invoke(
            {"question": prompt, "chat_history": memory.load_memory_variables({})}
        )
        print(f"Raw response from language model: {response}")
        logger.info(f"Raw response from language model: {response}")
    except Exception as e:
        print(f"Error in chat response generation: {str(e)}")
        logger.error(f"Error in chat response generation: {str(e)}", exc_info=True)
        raise

    # Generate citations
    citations = []
    try:
        print("Generating citations")
        logger.info("Generating citations")
        for i, (doc, score) in enumerate(retrieved_documents):
            citations.append({
                "source_id": i + 1,
                "quote": doc.page_content[:100] + "...",  # First 100 characters as a quote
                "doc_id": doc.metadata.get("doc_id", "Unknown"),
                "url": doc.metadata.get("source", "Unknown"),
                "title": doc.metadata.get("title", "Unknown Title"),
                "score": score
            })
        print(f"Generated {len(citations)} citations")
        logger.info(f"Generated {len(citations)} citations")
    except Exception as e:
        print(f"Error generating citations: {str(e)}")
        logger.error(f"Error generating citations: {str(e)}", exc_info=True)

    try:
        print("Generating related topics")
        logger.info("Generating related topics")
        related_topics = generate_related_topics(prompt, response)
        print(f"Generated {len(related_topics)} related topics")
        logger.info(f"Generated {len(related_topics)} related topics")
    except Exception as e:
        print(f"Error generating related topics: {str(e)}")
        logger.error(f"Error generating related topics: {str(e)}", exc_info=True)
        related_topics = []

    print("Chat response generation completed")
    logger.info("Chat response generation completed")
    return response, citations, related_topics, retrieved_documents




def generate_related_topics(prompt: str, answer: str) -> List[Dict]:
    """Extracts and returns related topics from retrieved documents."""
    related_docs = find_related_documents(
        prompt, n_results=config.MAX_RELATED_TOPICS
    )

    all_topics = []
    for doc, score in related_docs:
        topics = topic_extractor.extract_topics(doc.page_content, n_topics=1)
        for topic in topics:
            all_topics.append(
                {
                    "topic": topic,
                    "source": doc.metadata.get("source", "Unknown"),
                    "doc_id": doc.metadata.get("doc_id", "Unknown"),
                }
            )

    # Remove duplicates while preserving order
    unique_topics = []
    seen = set()
    for topic in all_topics:
        if topic["topic"] not in seen:
            unique_topics.append(topic)
            seen.add(topic["topic"])

    return unique_topics[: config.MAX_RELATED_TOPICS]


def format_chat_history(messages: List[Dict]) -> str:
    """Formats the chat history for providing context to the LLM."""
    formatted_history = ""
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        formatted_history += f"{role}: {content}\n\n"
    return formatted_history.strip()


def get_chat_context(prompt: str, chat_history: List[Dict]) -> str:
    """Constructs the context for the current chat interaction."""
    formatted_history = format_chat_history(chat_history)
    context = f"""
    Chat History:
    {formatted_history}

    Current User Question:
    {prompt}
    """
    return context.strip()


def verify_model(llm):
    """Ensures the configured OpenAI model is being used."""
    if llm.model_name != config.OPENAI_MODEL_NAME:
        raise ValueError(
            f"Expected model {config.OPENAI_MODEL_NAME}, but got {llm.model_name}"
        )


# Initialize the LLM
handler = StdOutCallbackHandler()
callback_manager = CallbackManager([handler])

llm = ChatOpenAI(
    temperature=config.TEMPERATURE,
    model_name=config.OPENAI_MODEL_NAME,
    callback_manager=callback_manager,
)
verify_model(llm)