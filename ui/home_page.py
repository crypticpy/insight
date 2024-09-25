# ui/home_page.py
import streamlit as st
import numpy as np
from database.vector_store import get_collection_stats
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def render():
    """Renders the home page with an introduction to RAG and visualizations."""
    st.title("Welcome to InSight: Your RAG-Powered Knowledge Base")

    st.markdown(
        """
        **Retrieval Augmented Generation (RAG)** is a powerful technique that enhances the capabilities of Large Language Models (LLMs) by enabling them to access and retrieve relevant information from a vast knowledge base. 

        **InSight** leverages RAG to provide you with insightful answers to your questions, grounded in the information you've provided. 
        """
    )

    # Display database statistics
    stats = get_collection_stats()
    st.write(f"**Documents in Knowledge Base:** {stats['total_documents']}")

    # 3D Visualization of Embeddings (using t-SNE for dimensionality reduction)
    with st.expander("Explore the Semantic Space of Your Knowledge Base"):
        st.markdown(
            "This visualization represents the semantic relationships between the documents in your knowledge base. "
            "Documents closer together are semantically more similar."
        )
        try:
            df = create_embeddings_dataframe()
            fig = create_3d_embeddings_visualization(df)
            st.plotly_chart(fig)
        except Exception as e:
            logger.error(f"Error creating embedding visualization: {str(e)}")
            st.error("Embedding visualization is currently unavailable.")

def create_embeddings_dataframe():
    """Fetches embeddings and metadata from the vector store and creates a DataFrame."""
    from database.vector_store import get_vectorstore

    vectorstore = get_vectorstore()
    collection_data = vectorstore.get(include=["embeddings", "metadatas"])
    embeddings = collection_data["embeddings"]
    metadatas = collection_data["metadatas"]

    # Convert embeddings list to a NumPy array
    embeddings = np.array(embeddings)

    # Reduce embedding dimensionality for visualization
    tsne = TSNE(n_components=3, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    df = pd.DataFrame(reduced_embeddings, columns=["x", "y", "z"])
    for i, metadata in enumerate(metadatas):
        df.loc[i, "source"] = metadata.get("source", "N/A")
        df.loc[i, "title"] = metadata.get("title", "N/A")
    return df

def create_3d_embeddings_visualization(df):
    """Creates a 3D scatter plot of the embeddings using Plotly."""
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="source",
        hover_data=["title", "source"],
        title="3D Visualization of Document Embeddings",
    )
    return fig

if __name__ == "__main__":
    render()
