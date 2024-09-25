# ui/admin_page.py
import streamlit as st
from helpers.document_processing import (
    process_uploaded_file,
    batch_process_documents,
    update_document_metadata,
    process_kb_excel
)
from database.vector_store import (
    add_documents_to_vectorstore,
    get_collection_stats,
    clear_vectorstore,
    get_all_documents,
    clean_vectorstore,  # Add this import
)
from database.feedback_store import feedback_store
import os
import tempfile
import pandas as pd
import plotly.express as px
import logging

logger = logging.getLogger(__name__)

def render():
    """Renders the admin dashboard with multiple tabs for document management,
    database statistics, and topic feedback.
    """
    st.title("Admin Dashboard")

    tabs = st.tabs(["Upload Documents", "Database Stats", "Topic Feedback", "Manage Documents"])

    with tabs[0]:
        render_upload_documents()

    with tabs[1]:
        render_database_stats()

    with tabs[2]:
        render_topic_feedback()

    with tabs[3]:
        render_manage_documents()


def render_upload_documents():
    """UI for document upload section, allowing file uploads, text input,
    and web scraping.
    """
    st.header("Upload Documents")

    uploaded_file = st.file_uploader(
        "Choose a file", type=["pdf", "docx", "doc", "pptx", "html", "txt", "jpg", "jpeg", "png", "xlsx", "json"]
    )

    text_input = st.text_area("Or paste text here")
    url_input = st.text_input("Source URL (for pasted text)")

    scrape_url = st.text_input("Or enter a URL to scrape")

    if st.button("Process and Add to Database"):
        if uploaded_file is not None:
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                handle_excel_upload(uploaded_file)
            else:
                handle_file_upload(uploaded_file)
        elif text_input and url_input:
            handle_text_input(text_input, url_input)
        elif scrape_url:
            handle_web_scraping(scrape_url)
        else:
            st.error(
                "Please upload a file, provide text and URL, or enter a URL to scrape."
            )


def handle_file_upload(uploaded_file):
    """Processes an uploaded file and adds it to the vector store."""
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File type: {uploaded_file.type}")
    st.write(f"File size: {uploaded_file.size} bytes")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"File saved temporarily at: {temp_file_path}")

        try:
            documents = process_uploaded_file(temp_file_path)
            st.write(f"Number of documents processed: {len(documents)}")

            for doc in documents:
                st.write(f"Document content preview: {doc.page_content[:100]}...")

            doc_ids = add_documents_to_vectorstore(documents)
            if doc_ids:
                st.success(f"Successfully added {len(doc_ids)} documents to the database.")
                for doc_id in doc_ids:
                    st.write(f"Added document with ID: {doc_id}")
            else:
                st.error("Failed to add documents to the database.")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.exception(e)


def handle_excel_upload(uploaded_file):
    st.write(f"Processing Excel file: {uploaded_file.name}")

    try:
        excel_content = uploaded_file.read()
        documents = process_kb_excel(excel_content)
        st.write(f"Number of KB articles processed: {len(documents)}")

        doc_ids = add_documents_to_vectorstore(documents)
        if doc_ids:
            st.success(f"Successfully added {len(doc_ids)} KB articles to the database.")
            for doc_id in doc_ids:
                st.write(f"Added document with ID: {doc_id}")
        else:
            st.error("Failed to add KB articles to the database.")

    except Exception as e:
        st.error(f"An error occurred while processing the Excel file: {str(e)}")
        st.exception(e)


def handle_text_input(text_input, url_input):
    """Processes pasted text and adds it to the vector store."""
    try:
        documents = process_uploaded_file(text=text_input, url=url_input)
        success = add_documents_to_vectorstore(documents)
        if success:
            st.success("Successfully added pasted text to the database.")
        else:
            st.error("Failed to add text to the database.")
    except Exception as e:
        st.error(f"An error occurred while processing the text: {str(e)}")
        st.exception(e)


def handle_web_scraping(scrape_url):
    """Scrapes content from a URL and adds it to the vector store."""
    try:
        documents = process_uploaded_file(url=scrape_url)
        st.write(f"Number of documents processed: {len(documents)}")

        success = add_documents_to_vectorstore(documents)
        if success:
            st.success(
                f"Successfully scraped and added content from {scrape_url} to the database."
            )
        else:
            st.error("Failed to add documents to the database.")
    except Exception as e:
        st.error(f"An error occurred while scraping the URL: {str(e)}")
        st.exception(e)


def render_database_stats():
    """Displays database statistics and management options."""
    st.header("Database Statistics and Management")

    try:
        stats = get_collection_stats()
        st.write(f"Total Documents: {stats['total_documents']}")
        st.write(f"Total Chunks: {stats['total_chunks']}")
    except Exception as e:
        st.error(f"Error fetching collection stats: {str(e)}")

    if st.button("Refresh Stats"):
        st.rerun()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Clear Vector Store")
        st.warning("Caution: This action will delete all documents in the vector store.")
        if st.button("Clear Vector Store"):
            try:
                if clear_vectorstore():
                    st.success("Vector store cleared successfully!")
                    st.rerun()
                else:
                    st.error("Failed to clear vector store.")
            except Exception as e:
                st.error(f"Error clearing vector store: {str(e)}")

    with col2:
        st.subheader("Clean Vector Store")
        st.info("This action will remove documents with None content from the vector store.")
        if st.button("Clean Vector Store"):
            try:
                removed_count = clean_vectorstore()
                if removed_count > 0:
                    st.success(f"Vector store cleaned. {removed_count} documents with None content have been removed.")
                else:
                    st.info("No documents with None content found in the vector store.")
                st.rerun()
            except Exception as e:
                st.error(f"Error cleaning vector store: {str(e)}")

    with col3:
        st.subheader("Feedback Data")
        st.warning("Caution: This action will delete all feedback data.")
        if st.button("Clear Feedback Data"):
            try:
                if feedback_store.clear_feedback_data():
                    st.success("Feedback data cleared successfully!")
                    st.rerun()
                else:
                    st.error("Failed to clear feedback data.")
            except Exception as e:
                st.error(f"Error clearing feedback data: {str(e)}")


def render_topic_feedback():
    """Displays topic feedback statistics."""
    st.header("Topic Feedback Statistics")

    topic_stats = feedback_store.get_topic_stats()

    if topic_stats:
        df = pd.DataFrame(topic_stats)

        fig = px.bar(
            df, x="topic", y="suggestion_count", title="Topic Suggestion Frequency"
        )
        st.plotly_chart(fig)

        fig = px.scatter(
            df,
            x="suggestion_count",
            y="avg_relevance",
            text="topic",
            title="Topic Relevance vs Suggestion Frequency",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig)

        st.subheader("Raw Data")
        st.dataframe(df)
    else:
        st.info("No topic feedback data available yet.")


def render_manage_documents():
    """UI for managing (viewing and deleting) documents in the vector store."""
    st.header("Manage Documents")

    all_docs = get_all_documents()

    if all_docs:
        df = pd.DataFrame(
            [
                {
                    "Document ID": doc.metadata.get("doc_id", "Unknown"),
                    "Title": doc.metadata.get("title", "N/A"),
                    "Source": doc.metadata.get("source", "N/A"),
                    "Content": doc.page_content[:100] + "..."  # Truncate content for display
                }
                for doc in all_docs
            ]
        )
        st.dataframe(df)

        # Filter out documents with unknown IDs
        valid_docs = [doc for doc in all_docs if doc.metadata.get("doc_id") != "Unknown"]

        if valid_docs:
            selected_doc_id = st.selectbox(
                "Select a Document to Delete",
                options=[doc.metadata["doc_id"] for doc in valid_docs],
                format_func=lambda x: f"{x} - {next((doc.metadata.get('title', 'N/A') for doc in valid_docs if doc.metadata['doc_id'] == x), 'N/A')}"
            )
            if st.button("Delete Document"):
                from database.vector_store import delete_document

                if delete_document(selected_doc_id):
                    st.success(f"Document {selected_doc_id} deleted successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to delete document {selected_doc_id}.")
        else:
            st.warning("No documents with valid IDs found in the database.")
    else:
        st.info("No documents found in the database.")


if __name__ == "__main__":
    render()
