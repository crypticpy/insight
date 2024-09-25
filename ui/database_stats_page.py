# ui/database_stats_page.py
import streamlit as st
from database.vector_store import get_collection_stats, clear_vectorstore
from database.feedback_store import feedback_store


def render():
    st.title("Database Statistics and Management")

    try:
        stats = get_collection_stats()
        st.write(f"Total Documents: {stats['total_documents']}")
        st.write(f"Total Chunks: {stats['total_chunks']}")
    except Exception as e:
        st.error(f"Error fetching collection stats: {str(e)}")

    if st.button("Refresh Stats"):
        st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vector Store Management")
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
        st.subheader("Feedback Data Management")
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

    st.markdown("---")

    st.subheader("Reset Entire System")
    st.warning("Caution: This action will delete all documents and feedback data.")
    if st.button("Reset Entire System"):
        try:
            vector_store_cleared = clear_vectorstore()
            feedback_cleared = feedback_store.clear_feedback_data()

            if vector_store_cleared and feedback_cleared:
                st.success("Entire system reset successfully!")
                st.rerun()
            else:
                st.error("Failed to reset the entire system. Please check the logs.")
        except Exception as e:
            st.error(f"Error resetting the system: {str(e)}")

