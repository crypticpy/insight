# ui/debug_chat_page.py

import streamlit as st
from typing import List, Dict, Tuple, Optional

from langchain_core.documents import Document

from helpers.chat import get_chat_response
from ui.components import display_chat_message, display_source, display_full_document
from database.feedback_store import feedback_store
from langchain_core.messages import AIMessage, HumanMessage
from database.vector_store import get_document_by_id
import uuid
import logging
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

def render() -> None:
    """
    Render the debug chat page with enhanced functionality and error handling.
    """
    st.title("RAG Chat View")

    initialize_session_state()

    # Create a container for the chat history
    chat_container = st.container()

    # Create a container for the input and buttons
    input_container = st.container()

    with input_container:
        handle_user_input()
        display_chat_options()

    with chat_container:
        display_chat_history()

def initialize_session_state() -> None:
    """
    Initialize session state variables if they don't exist.
    """
    if "debug_messages" not in st.session_state:
        st.session_state.debug_messages = []
    if "display_full_doc" not in st.session_state:
        st.session_state.display_full_doc = None
    if "topic_query" not in st.session_state:
        st.session_state.topic_query = None

def display_chat_history() -> None:
    """
    Display the chat history with improved formatting for readability.
    """
    for message in st.session_state.debug_messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], (str, HumanMessage)):
                content = message["content"] if isinstance(message["content"], str) else message["content"].content
                st.markdown(content)
            elif isinstance(message["content"], AIMessage):
                content = message["content"].content

                # Split the content into paragraphs
                paragraphs = content.split("\n")

                for paragraph in paragraphs:
                    # Check if the paragraph is a numbered point
                    if paragraph.strip().startswith(tuple(f"{i}." for i in range(1, 10))):
                        st.markdown(f"**{paragraph.strip()}**")
                    else:
                        st.write(paragraph)

                # Display sources if available
                if 'sources' in message:
                    st.markdown("---")
                    st.markdown("**Sources:**")
                    for source in message['sources']:
                        if isinstance(source, dict):
                            with st.expander(f"Source {source.get('source_id', 'Unknown')}"):
                                st.markdown(f"**Title:** {source.get('title', 'Unknown')}")
                                st.markdown(
                                    f"**URL:** [{source.get('url', 'Unknown')}]({source.get('url', 'Unknown')})")
                                st.markdown(f"**Quote:** {source.get('quote', 'No quote available')[:200]}...")
                                st.markdown(f"**Document ID:** {source.get('doc_id', 'Unknown')}")
                        else:
                            st.markdown(f"- {str(source)}")
            else:
                st.write("Unsupported message type")

        if message["role"] == "assistant":
            with st.expander("Debug Information", expanded=False):
                tabs = st.tabs(["Retrieved Documents", "Metadata", "Related Topics"])

                with tabs[0]:
                    if 'retrieved_docs' in message:
                        for i, (doc, score) in enumerate(message['retrieved_docs'], start=1):
                            st.markdown(f"**Document {i}**")
                            st.write(f"**Content:** {doc.page_content[:200]}...")
                            st.json(doc.metadata)
                            st.write(f"**Score:** {score}")
                            display_document_feedback_buttons(doc.metadata.get('doc_id', 'Unknown'))
                            st.markdown("---")

                with tabs[1]:
                    st.json(message.get('metadata', {}))

                with tabs[2]:
                    display_related_topics(message.get('related_topics', []))

def display_document_feedback_buttons(doc_id: str) -> None:
    """
    Display feedback buttons for a retrieved document.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👍", key=f"upvote_{doc_id}_{uuid.uuid4()}"):
            feedback_store.add_feedback(doc_id, 1)
            st.success("Thank you for your feedback!")

    with col2:
        if st.button("👎", key=f"downvote_{doc_id}_{uuid.uuid4()}"):
            feedback_store.add_feedback(doc_id, -1)
            st.success("Thank you for your feedback!")

    with col3:
        if st.button("View Full", key=f"view_{doc_id}_{uuid.uuid4()}"):
            st.session_state.display_full_doc = doc_id
            st.rerun()

def handle_user_input() -> None:
    """
    Handle user input and generate a response.
    """
    user_input = st.chat_input("What would you like to know? (Debug Mode)")

    if user_input:
        process_user_input(user_input)
    elif st.session_state.topic_query:
        process_user_input(st.session_state.topic_query)
        st.session_state.topic_query = None  # Reset the topic query after processing

def process_user_input(user_input: str) -> None:
    """
    Process user input, generate a response, and update the session state.
    """
    st.session_state.debug_messages.append({"role": "user", "content": user_input})
    chat_history = prepare_chat_history()

    with st.spinner("Thinking..."):
        try:
            answer, citations, related_topics, retrieved_docs = get_chat_response(user_input, chat_history)

            update_session_state(answer, citations, related_topics, retrieved_docs)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logger.error(f"Error generating response: {str(e)}", exc_info=True)

    st.rerun()

def prepare_chat_history() -> List[Tuple[str, str]]:
    """
    Prepare the chat history for the model.
    """
    return [(msg["content"], st.session_state.debug_messages[i + 1]["content"])
            for i, msg in enumerate(st.session_state.debug_messages[:-1:2])]

def update_session_state(answer: str, citations: List[Dict], related_topics: List[Dict],
                         retrieved_docs: List[Tuple[Document, float]]) -> None:
    """
    Update the session state with the new response information.
    """
    st.session_state.debug_messages.append({
        "role": "assistant",
        "content": answer,
        "sources": citations,
        "retrieved_docs": retrieved_docs,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": "gpt-4",
            "version": "1.0"
        },
        "related_topics": related_topics
    })

def display_related_topics(related_topics: List[Dict]) -> None:
    """
    Display related topics if available.
    """
    if related_topics:
        for topic in related_topics:
            st.markdown(f"**{topic['topic']}**")
            st.write(f"**Source:** {topic.get('source', 'Unknown')}")
            st.write(f"**Document ID:** {topic.get('doc_id', 'Unknown')}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Full Document", key=f"view_full_{topic.get('doc_id', '')}_{uuid.uuid4()}"):
                    st.session_state.display_full_doc = topic.get('doc_id')
                    st.rerun()
            with col2:
                if st.button("Ask About This Topic", key=f"ask_{topic['topic']}_{uuid.uuid4()}"):
                    st.session_state.topic_query = generate_topic_query(topic['topic'])
                    st.rerun()
            st.markdown("---")
    else:
        st.write("No related topics available for this message.")

def generate_topic_query(topic: str) -> str:
    """
    Generate a query about the given topic based on the recent conversation.
    """
    last_user_message = next(
        (msg['content'] for msg in reversed(st.session_state.debug_messages) if msg['role'] == 'user'), None)
    last_assistant_message = next(
        (msg['content'] for msg in reversed(st.session_state.debug_messages) if msg['role'] == 'assistant'), None)

    if last_user_message and last_assistant_message:
        query = f"Based on our previous discussion where I asked '{last_user_message}' and you answered '{last_assistant_message[:50]}...', can you elaborate on how '{topic}' relates to this and provide more information about it?"
    else:
        query = f"Can you tell me more about '{topic}' and how it might be relevant to our conversation?"

    return query

def display_chat_options() -> None:
    """
    Display options to clear chat and download chat history.
    """
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Clear Chat", key="clear_chat_button"):
            st.session_state.debug_messages = []
            st.success("Chat cleared successfully!")
            st.rerun()

    with col2:
        if st.button("Download Chat History", key="download_chat_button"):
            chat_history = prepare_chat_history_for_download()
            st.download_button(
                label="Download as Text",
                data=chat_history,
                file_name="chat_history.txt",
                mime="text/plain",
                key="download_text_button"
            )

def prepare_chat_history_for_download() -> str:
    """
    Prepare the chat history as a text file for download.
    """
    chat_history = ""
    for message in st.session_state.debug_messages:
        chat_history += f"{message['role'].capitalize()}: {message['content']}\n\n"
        if 'sources' in message:
            chat_history += "Sources:\n"
            for source in message['sources']:
                chat_history += f"- {source['text']}\n"
            chat_history += "\n"
    return chat_history

if __name__ == "__main__":
    render()
