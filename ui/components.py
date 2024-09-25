# ui/components.py
import streamlit as st
from database.vector_store import get_document_by_id
from typing import Dict, Optional, List

def display_chat_message(role: str, content: str, sources: Optional[List[Dict]] = None) -> None:
    """
    Display a chat message with optional sources.

    Args:
        role (str): The role of the message sender (e.g., 'user', 'assistant').
        content (str): The content of the message.
        sources (Optional[List[Dict]]): A list of source dictionaries, if any.
    """
    with st.chat_message(role):
        st.markdown(content)
        if sources:
            st.markdown("**Sources:**")
            for source in sources:
                st.markdown(f"- {source}")

def display_source(source: Dict) -> None:
    """
    Display information about a source.

    Args:
        source (Dict): A dictionary containing source information.
    """
    st.markdown(f"**Source:** {source.get('source', 'Unknown')}")
    st.markdown(f"**Document ID:** {source.get('doc_id', 'Unknown')}")
    if "content" in source:
        with st.expander("View source content"):
            st.markdown(source['content'])

def display_full_document(doc_id: str) -> None:
    """
    Display the full content of a document.

    Args:
        doc_id (str): The ID of the document to display.
    """
    document = get_document_by_id(doc_id)
    if document:
        st.markdown(f"**Source:** {document.metadata.get('source', 'Unknown')}")
        st.markdown(f"**Document ID:** {doc_id}")
        st.markdown(document.page_content)
    else:
        st.error(f"Document with ID {doc_id} not found.")

