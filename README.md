# InSight: Advanced RAG-Powered Knowledge Base

InSight is a sophisticated Retrieval Augmented Generation (RAG) system that enhances the capabilities of Large Language Models (LLMs) by enabling them to access and retrieve relevant information from a vast knowledge base. This project combines state-of-the-art natural language processing techniques with an intuitive user interface to provide insightful answers grounded in your organization's specific information.

## Key Features

1. **Hybrid Search**: Combines dense and sparse retrieval methods for optimal document retrieval.
2. **Semantic Search**: Utilizes contextual embeddings and re-ranking for improved search relevance.
3. **Adaptive Document Processing**: Intelligently chunks and processes various document types (PDF, DOCX, TXT, etc.).
4. **Interactive Chat Interface**: Engage in natural conversations with the AI, complete with citations and related topics.
5. **Admin Dashboard**: Easily manage documents, view database statistics, and analyze user feedback.
6. **Feedback System**: Collects and analyzes user feedback on document relevance and suggested topics.
7. **Visualization Tools**: Explore the semantic space of your knowledge base through interactive 3D visualizations.
8. **API Integration**: Exposes key functionalities through a FastAPI-based API for easy integration with other systems.

## Core Functionality

### Document Ingestion and Processing

- Supports multiple file formats (PDF, DOCX, PPTX, HTML, TXT, images, CSV, JSON)
- Adaptive chunking based on document type and content length
- Extracts and stores relevant metadata
- Handles web scraping for URL-based content

### Vector Store Management

- Utilizes Chroma as the underlying vector database
- Implements hybrid search combining dense and sparse retrieval methods
- Supports semantic search with contextual embeddings
- Provides CRUD operations for document management

### Chat and Query Processing

- Implements a conversational interface using Streamlit
- Generates responses using advanced LLMs (e.g., GPT-4)
- Provides citations and sources for generated answers
- Suggests related topics to encourage exploration

### Admin and Analytics

- Offers a comprehensive admin dashboard for system management
- Displays database statistics and allows for data management
- Analyzes and visualizes user feedback and topic relevance

### API

- Exposes search, chat, and database management functionalities through a RESTful API
- Supports both GET and POST methods for flexible integration

## Technical Stack

- **Backend**: Python, FastAPI
- **Frontend**: Streamlit
- **Vector Database**: Chroma
- **Embeddings**: OpenAI Embeddings
- **LLM**: OpenAI GPT-4
- **NLP Libraries**: Langchain, NLTK, Sentence Transformers
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy, Scikit-learn

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in a `.env` file (see `config.py` for required variables)
4. Run the Streamlit app: `streamlit run app.py`
5. Access the admin dashboard to start ingesting documents and building your knowledge base

## Usage

1. **Chat Interface**: Navigate to the Chat page to start interacting with the system. Ask questions, and receive answers with citations and related topics.
2. **Admin Dashboard**: Use the Admin page to upload documents, view database statistics, and manage the system.
3. **API**: Integrate with other systems using the provided API endpoints (see `api_server.py` for available routes).

## Customization

InSight is designed to be highly customizable. Key areas for potential customization include:

- Embedding models (in `vector_store.py`)
- Chunking strategies (in `chunking.py`)
- LLM selection and parameters (in `config.py` and `chat.py`)
- UI components and layout (in the various files under the `ui/` directory)

## Future Enhancements

- Integration with more data sources and document types
- Advanced analytics and user behavior tracking
- Improved multi-modal capabilities (e.g., audio and video processing)
- Enhanced personalization and user-specific knowledge bases

InSight represents a powerful tool for organizations looking to leverage their internal knowledge and documentation through advanced AI techniques. By combining RAG with an intuitive interface and robust management tools, InSight offers a comprehensive solution for next-generation knowledge management and information retrieval.