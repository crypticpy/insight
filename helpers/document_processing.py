# helpers/document_processing.py

import uuid
import os
import base64
import csv
import io
from typing import List, Optional
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    WebBaseLoader,
    CSVLoader,
    JSONLoader,
)
from langchain_core.documents import Document
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from config import config
from database.vector_store import add_documents_to_vectorstore
from helpers.chunking import adaptive_split_documents, chunk_document, get_chunk_metadata
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

# Initialize GPT-4 Omni model
gpt4_omni = ChatOpenAI(model=config.OPENAI_MODEL_NAME, max_tokens=1000)

def process_uploaded_file(
        file_path: str = None, file_content: bytes = None, text: Optional[str] = None, url: Optional[str] = None
) -> List[Document]:
    """
    Processes uploaded files, text, or URLs to create enriched Document objects.

    Args:
        file_path (str, optional): Path to the uploaded file. Defaults to None.
        file_content (bytes, optional): Content of the file as bytes. Defaults to None.
        text (str, optional): Text content if no file is uploaded. Defaults to None.
        url (str, optional): URL source of the text content or for web scraping. Defaults to None.

    Returns:
        List[Document]: A list of processed Document objects ready for embedding.

    Raises:
        FileNotFoundError: If the file specified by file_path does not exist.
        ValueError: If an unsupported file type is encountered.
    """
    try:
        documents = []
        if file_path or file_content:
            if file_path and not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_type = file_path.split(".")[-1].lower() if file_path else "xlsx"
            if file_type == 'xlsx':
                documents = process_kb_excel(file_content or open(file_path, 'rb').read())
            elif file_type in ['jpg', 'jpeg', 'png', 'heif', 'heic']:
                documents = process_image(file_path)
            else:
                loader = get_loader(file_type, file_path)
                documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = file_path or "Uploaded Excel File"
                doc.metadata["title"] = os.path.basename(file_path) if file_path else "KB Article"

        elif text:
            documents = [Document(page_content=text, metadata={"source": url or "User Input"})]

        elif url:
            logger.info(f"Scraping content from URL: {url}")
            documents = scrape_web_content(url)

        # Apply adaptive chunking
        chunked_documents = adaptive_split_documents(documents)

        # Assign unique doc_ids and add chunk metadata
        total_chunks = len(chunked_documents)
        for i, doc in enumerate(chunked_documents):
            doc.metadata["doc_id"] = str(uuid.uuid4())
            doc.metadata.update(get_chunk_metadata(doc, total_chunks))
            doc.metadata["chunk_index"] = i
            doc.metadata["total_chunks"] = total_chunks

        logger.info(f"Processing {len(chunked_documents)} document chunks")

        return chunked_documents

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise


def process_kb_excel(file_content: bytes) -> List[Document]:
    """
    Process an Excel file containing KB articles and return a list of Document objects.

    Args:
        file_content (bytes): The content of the Excel file as bytes.

    Returns:
        List[Document]: A list of processed Document objects.
    """
    documents = []
    df = pd.read_excel(file_content)

    for _, row in df.iterrows():
        # Strip HTML from content fields
        introduction = BeautifulSoup(str(row['Introduction']), 'html.parser').get_text()
        instructions = BeautifulSoup(str(row['Instructions']), 'html.parser').get_text()
        internal = BeautifulSoup(str(row['Internal']), 'html.parser').get_text()

        # Combine relevant fields into a single content string
        content = f"""
        Title: {row['Title']}
        Category: {row['Category']}
        KB Article #: {row['KB Article #']}
        Version: {row['Version']}

        Introduction:
        {introduction}

        Instructions:
        {instructions}

        Internal:
        {internal}

        Keywords: {row['Keywords']}
        """

        # Create a Document object
        doc = Document(
            page_content=content.strip(),
            metadata={
                'source': f"KB{row['KB Article #']}",
                'title': row['Title'],
                'category': row['Category'],
                'kb_number': row['KB Article #'],
                'version': row['Version'],
                'keywords': row['Keywords'],
                'updated': str(row['Updated'])
            }
        )
        documents.append(doc)

    return documents

def process_image(file_path: str) -> List[Document]:
    """
    Process an image file using GPT-4 Omni vision capabilities.

    Args:
        file_path (str): Path to the image file.

    Returns:
        List[Document]: List of documents with image analysis.
    """
    try:
        with Image.open(file_path) as img:
            # Convert image to RGB if it's not
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Convert to JPEG format in memory
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        # Correct format for ChatOpenAI model
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this image in detail:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            )
        ]

        response = gpt4_omni.invoke(messages)

        # The response should be an AIMessage
        if isinstance(response, AIMessage):
            image_description = response.content
        else:
            # Fallback in case the response is not as expected
            image_description = str(response)

        # Create a single document for the image description
        document = Document(
            page_content=image_description,
            metadata={
                "source": file_path,
                "type": "image_analysis",
                "file_name": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "analysis_timestamp": datetime.now().isoformat()
            }
        )

        # Apply chunking to the image description
        return chunk_document(document)

    except Exception as e:
        logger.error(f"Error processing image {file_path}: {str(e)}")
        return [Document(
            page_content=f"Error processing image: {str(e)}",
            metadata={"source": file_path, "type": "image_processing_error"}
        )]

def get_loader(file_type: str, file_path: str):
    """
    Factory method to return the appropriate document loader based on file type.

    Args:
        file_type (str): The file extension (e.g., "pdf", "docx").
        file_path (str): The path to the file.

    Returns:
        DocumentLoader: The appropriate document loader instance.

    Raises:
        ValueError: If an unsupported file type is encountered.
    """
    loaders = {
        "pdf": PyPDFLoader,
        "docx": UnstructuredWordDocumentLoader,
        "doc": UnstructuredWordDocumentLoader,
        "pptx": UnstructuredPowerPointLoader,
        "ppt": UnstructuredPowerPointLoader,
        "html": UnstructuredHTMLLoader,
        "txt": TextLoader,
        "csv": CSVLoader,
        "json": JSONLoader,
    }

    loader_class = loaders.get(file_type)
    if loader_class:
        return loader_class(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def scrape_web_content(url: str) -> List[Document]:
    """
    Scrapes content from a web page using WebBaseLoader and extracts the main content.

    Args:
        url (str): The URL of the web page to scrape.

    Returns:
        List[Document]: A list containing a Document with the scraped main content.
    """
    try:
        # Use requests to fetch the content
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for bad status codes

        # Extract main content from the loaded document
        main_content = extract_main_content(response.text)

        # Create a new document with the extracted main content
        document = Document(page_content=main_content, metadata={"source": url, "title": "Web Page"})

        # Apply chunking to the web content
        return chunk_document(document)

    except requests.RequestException as e:
        logger.error(f"Error fetching content from {url}: {str(e)}")
        return [Document(page_content=f"Error fetching content: {str(e)}", metadata={"source": url, "error": str(e)})]
    except Exception as e:
        logger.error(f"Error scraping content from {url}: {str(e)}")
        return [Document(page_content=f"Error scraping content: {str(e)}", metadata={"source": url, "error": str(e)})]

def extract_main_content(html_content: str) -> str:
    """
    Extracts the main content from HTML, attempting to bypass headers, footers, and menu content.

    Args:
        html_content (str): The full HTML content of the web page.

    Returns:
        str: The extracted main content of the web page.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Remove header and footer
    for header in soup.find_all(['header', 'nav']):
        header.decompose()
    for footer in soup.find_all('footer'):
        footer.decompose()

    # Try to find the main content
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main|body'))

    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        # If no main content is found, fall back to the body
        body = soup.body
        if body:
            text = body.get_text(separator='\n', strip=True)
        else:
            # If there's no body, use the entire parsed content
            text = soup.get_text(separator='\n', strip=True)

    # Remove excessive newlines and spaces
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)

    return text.strip()

def batch_process_documents(directory_path: str) -> List[str]:
    """
    Processes all supported documents in a directory.

    Args:
        directory_path (str): Path to the directory containing documents.

    Returns:
        List[str]: List of all document IDs added to the vector store.
    """
    all_doc_ids = []
    supported_extensions = [
        ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html", ".txt",
        ".jpg", ".jpeg", ".png", ".heif", ".heic", ".csv", ".json"
    ]

    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in supported_extensions):
                    file_path = os.path.join(root, file)
                    documents = process_uploaded_file(file_path)
                    doc_ids = add_documents_to_vectorstore(documents)
                    all_doc_ids.extend(doc_ids)

        logger.info(
            f"Batch processed {len(all_doc_ids)} documents from {directory_path}"
        )
        return all_doc_ids

    except Exception as e:
        logger.error(f"Error in batch processing documents: {str(e)}")
        raise

def update_document_metadata(doc_id: str, metadata: dict) -> bool:
    """
    Updates metadata of a document in the vector store.

    Args:
        doc_id (str): The ID of the document to update.
        metadata (dict): The new metadata to apply.

    Returns:
        bool: True if update was successful, False otherwise.
    """
    try:
        from database.vector_store import update_document

        document = update_document(doc_id, None, metadata)
        if document:
            logger.info(f"Updated metadata for document {doc_id}")
            return True
        else:
            logger.warning(f"Document {doc_id} not found for metadata update")
            return False
    except Exception as e:
        logger.error(f"Error updating document metadata: {str(e)}")
        return False


def process_kb_csv(file_content: str) -> List[Document]:
    """
    Process a CSV file containing KB articles and return a list of Document objects.

    Args:
        file_content (str): The content of the CSV file as a string.

    Returns:
        List[Document]: A list of processed Document objects.
    """
    documents = []
    csv_file = io.StringIO(file_content)
    csv_reader = csv.DictReader(csv_file)

    for row in csv_reader:
        # Strip HTML from content fields
        introduction = BeautifulSoup(row['Introduction'], 'html.parser').get_text()
        instructions = BeautifulSoup(row['Instructions'], 'html.parser').get_text()
        internal = BeautifulSoup(row['Internal'], 'html.parser').get_text()

        # Combine relevant fields into a single content string
        content = f"""
        Title: {row['Title']}
        Category: {row['Category']}
        KB Article #: {row['KB Article #']}
        Version: {row['Version']}

        Introduction:
        {introduction}

        Instructions:
        {instructions}

        Internal:
        {internal}

        Keywords: {row['Keywords']}
        """

        # Create a Document object
        doc = Document(
            page_content=content.strip(),
            metadata={
                'source': f"KB{row['KB Article #']}",
                'title': row['Title'],
                'category': row['Category'],
                'kb_number': row['KB Article #'],
                'version': row['Version'],
                'keywords': row['Keywords'],
                'updated': row['Updated']
            }
        )
        documents.append(doc)

    return documents