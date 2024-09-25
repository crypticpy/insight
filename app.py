# app.py
import streamlit as st
from ui import admin_page, chat_page, database_stats_page, debug_chat_page, home_page  # Import home_page
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Advanced RAG Chatbot", layout="wide")

def main():
    """Main function to control the application flow."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Home", "Chat", "Admin", "Database Stats"]  # Add "Home"
    )


    if page == "Admin":
        admin_page.render()
    elif page == "Database Stats":
        database_stats_page.render()
    elif page == "Chat":
        debug_chat_page.render()
    else:  # Default to the "Home" page
        home_page.render()

if __name__ == "__main__":
    main()