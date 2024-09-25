# helpers/chunking.py

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import config
import re

def get_adaptive_text_splitter(doc_type: str, content_length: int) -> RecursiveCharacterTextSplitter:
    """
    Returns an adaptive text splitter based on the document type and content length.

    Args:
        doc_type (str): The type of the document (e.g., 'pdf', 'docx', 'txt').
        content_length (int): The length of the document content.

    Returns:
        RecursiveCharacterTextSplitter: A text splitter with parameters optimized for the document type and length.
    """
    base_kwargs = {
        "separators": ["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        "chunk_size": adapt_chunk_size(content_length),
        "chunk_overlap": adapt_chunk_overlap(content_length),
    }

    if doc_type in ['pdf', 'docx']:
        return RecursiveCharacterTextSplitter(
            **base_kwargs,
            length_function=len,
        )
    elif doc_type == 'txt':
        return RecursiveCharacterTextSplitter(
            **base_kwargs,
            length_function=len,
            is_separator_regex=False,
        )
    elif doc_type in ['csv', 'json']:
        return RecursiveCharacterTextSplitter(
            **base_kwargs,
            length_function=len,
            is_separator_regex=True,
            separators=[",", "\n", " "],
        )
    else:
        # Default splitter for unknown document types
        return RecursiveCharacterTextSplitter(**base_kwargs)

def adapt_chunk_size(content_length: int) -> int:
    """
    Adapts the chunk size based on the content length.

    Args:
        content_length (int): The length of the document content.

    Returns:
        int: The adapted chunk size.
    """
    if content_length < 10000:
        return min(content_length, config.CHUNK_SIZE)
    elif content_length < 100000:
        return min(content_length // 10, config.CHUNK_SIZE)
    else:
        return min(content_length // 20, config.CHUNK_SIZE)

def adapt_chunk_overlap(content_length: int) -> int:
    """
    Adapts the chunk overlap based on the content length.

    Args:
        content_length (int): The length of the document content.

    Returns:
        int: The adapted chunk overlap.
    """
    return min(content_length // 50, config.CHUNK_OVERLAP)

def adaptive_split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits documents using an adaptive strategy based on document type and length.

    Args:
        documents (List[Document]): A list of documents to be split.

    Returns:
        List[Document]: A list of split documents.
    """
    split_docs = []
    for doc in documents:
        doc_type = doc.metadata.get('source', '').split('.')[-1].lower()
        content_length = len(doc.page_content)
        splitter = get_adaptive_text_splitter(doc_type, content_length)
        split_docs.extend(splitter.split_documents([doc]))
    return split_docs

def chunk_document(document: Document) -> List[Document]:
    """
    Chunks a single document using the adaptive splitting strategy.

    Args:
        document (Document): The document to be chunked.

    Returns:
        List[Document]: A list of chunked documents.
    """
    doc_type = document.metadata.get('source', '').split('.')[-1].lower()
    content_length = len(document.page_content)
    splitter = get_adaptive_text_splitter(doc_type, content_length)
    return splitter.split_documents([document])

def get_chunk_metadata(chunk: Document, total_chunks: int) -> Dict[str, Any]:
    """
    Generates metadata for a chunk, including its position in the original document.

    Args:
        chunk (Document): The document chunk.
        total_chunks (int): The total number of chunks for the document.

    Returns:
        Dict[str, Any]: Metadata for the chunk.
    """
    metadata = chunk.metadata.copy()
    metadata['chunk_index'] = getattr(chunk, 'chunk_index', None)
    metadata['total_chunks'] = total_chunks
    metadata['content_length'] = len(chunk.page_content)
    metadata['chunk_type'] = classify_chunk_content(chunk.page_content)
    return metadata

def classify_chunk_content(content: str) -> str:
    """
    Classifies the type of content in the chunk.

    Args:
        content (str): The content of the chunk.

    Returns:
        str: The classified content type.
    """
    if re.search(r'\b(table|column|row)\b', content, re.IGNORECASE):
        return 'tabular'
    elif re.search(r'\b(figure|image|diagram)\b', content, re.IGNORECASE):
        return 'visual'
    elif re.search(r'\b(equation|formula)\b', content, re.IGNORECASE):
        return 'mathematical'
    elif re.search(r'\b(code|function|variable)\b', content, re.IGNORECASE):
        return 'code'
    else:
        return 'text'
