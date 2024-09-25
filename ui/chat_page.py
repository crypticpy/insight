# ui/chat_page.py

import streamlit as st
from typing import List, Dict
from helpers.chat import get_chat_response
from database.vector_store import get_document_by_id, get_collection_stats, find_related_documents
from database.feedback_store import feedback_store
from config import config
import uuid
import logging

logger = logging.getLogger(__name__)

def render():
    st.title("Chat with InSight")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "related_topics" not in st.session_state:
        st.session_state.related_topics = []
    if "current_citations" not in st.session_state:
        st.session_state.current_citations = []
    if "retrieved_docs" not in st.session_state:
        st.session_state.retrieved_docs = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False

    # Display sidebar content
    display_sidebar_content()

    # Display chat interface or sources page
    if not st.session_state.show_sources:
        display_chat_interface()
    else:
        display_sources_page(st.session_state.retrieved_docs, st.session_state.current_citations)

def display_sidebar_content():
    display_citations(st.session_state.current_citations)
    display_related_topics(st.session_state.related_topics)

def display_chat_interface():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("What would you like to know?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_history = [(msg["content"], st.session_state.messages[i + 1]["content"])
                        for i, msg in enumerate(st.session_state.messages[:-1:2])]

        with st.spinner("Thinking..."):
            answer, citations, related_topics, retrieved_docs = get_chat_response(prompt, chat_history)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.retrieved_docs = retrieved_docs
        st.session_state.current_citations = citations
        st.session_state.related_topics = related_topics

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.markdown(f"Sources found: {len(citations)}") # Displaying source count

        if related_topics:
            st.subheader("Related Topics")
            for topic in related_topics:
                st.write(f"- {topic['topic']}")

        if st.button("Why this answer?"):
            st.session_state.show_sources = True
            st.rerun()

def display_citations(citations):
    st.sidebar.title("Sources")
    if citations:
        for citation in citations:
            with st.sidebar.expander(f"Source {citation['source_id']}"):  # Use source_id for display
                st.write(f"**Document ID:** {citation['doc_id']}")  # Use doc_id internally
                st.write(f"**Quote:** {citation['quote']}")
                if citation.get("url"):
                    st.sidebar.markdown(f"**Source:** [{citation.get('title', 'Link')}]({citation['url']})")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍", key=f"upvote_{citation['source_id']}"):
                        feedback_store.add_feedback(citation["document_id"], 1)
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎", key=f"downvote_{citation['source_id']}"):
                        feedback_store.add_feedback(citation["document_id"], -1)
                        st.success("Thank you for your feedback!")
                with col3:
                    if st.button("View Full", key=f"view_{citation['source_id']}"):
                        st.session_state.display_full_doc = citation["document_id"]
                        st.rerun()
    else:
        st.sidebar.info("No sources were used for this answer.")

def display_related_topics(related_topics):
    """Displays related topics in the sidebar."""
    st.sidebar.title("Related Topics")
    for topic in related_topics:
        unique_key = f"{topic['doc_id']}_{uuid.uuid4()}"
        if st.sidebar.button(topic["topic"], key=f"topic_{unique_key}"):
            st.session_state.messages.append(
                {"role": "user", "content": f"Tell me about {topic['topic']}"}
            )
            st.rerun()

def display_sources_page(retrieved_docs, citations):
    st.title("Sources Used for the Answer")
    st.markdown(
        "The following sources were used by the LLM to generate the answer. "
        "The similarity score indicates how relevant each source is to your question."
    )

    for i, doc in enumerate(retrieved_docs):
        with st.expander(f"Source {i+1}"):
            highlighted_content = highlight_citations(doc.page_content, citations, i + 1)
            st.markdown(f"**Content:** {highlighted_content}", unsafe_allow_html=True)
            st.markdown(f"**Document ID:** {doc.metadata.get('doc_id', 'N/A')}")
            st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")

    if st.button("Back to Chat"):
        st.session_state.show_sources = False
        st.rerun()

def highlight_citations(content, citations, source_id):
    for citation in citations:
        if citation["source_id"] == source_id:
            content = content.replace(
                citation["quote"], f"<mark>{citation['quote']}</mark>"
            )
    return content

def generate_related_topics(prompt: str) -> List[Dict]:
    from helpers.topic_extractor import topic_extractor

    related_docs = find_related_documents(prompt, n_results=config.MAX_RELATED_TOPICS)

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

    unique_topics = []
    seen = set()
    for topic in all_topics:
        if topic["topic"] not in seen:
            unique_topics.append(topic)
            seen.add(topic["topic"])

    return unique_topics[: config.MAX_RELATED_TOPICS]

if __name__ == "__main__":
    render()
