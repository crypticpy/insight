# api_server.py

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from database.vector_store import hybrid_search, semantic_search
from helpers.chat import get_chat_response
import uvicorn
import logging
import numpy as np
from fastapi.encoders import jsonable_encoder

app = FastAPI(title="InSight API", description="API for querying the InSight vector store and retrieving results")

logger = logging.getLogger(__name__)

class Document(BaseModel):
    content: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    documents: List[Document]

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question")
    chat_history: Optional[List[List[str]]] = Field(None, description="Previous chat history")

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    related_topics: List[Dict[str, Any]]

def custom_jsonable_encoder(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    return obj

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": str(exc.detail)}
    )

@app.post("/search", response_model=SearchResponse)
async def search_vectorstore(
    query: str = Query(..., description="The search query"),
    num_results: int = Query(4, description="Number of results to return", ge=1, le=20),
    search_type: str = Query("hybrid", description="Type of search to perform: 'hybrid' or 'semantic'")
):
    try:
        if search_type == "hybrid":
            results = hybrid_search(query, k=num_results)
        elif search_type == "semantic":
            results = semantic_search(query, k=num_results)
        else:
            raise HTTPException(status_code=400, detail="Invalid search type. Use 'hybrid' or 'semantic'.")

        documents = [
            Document(
                content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": float(score)}
            ) for doc, score in results
        ]
        response_data = SearchResponse(documents=documents)
        json_compatible_data = jsonable_encoder(response_data, custom_encoder=custom_jsonable_encoder)
        return JSONResponse(content=json_compatible_data)
    except Exception as e:
        logger.error(f"Error searching vector store: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/search_get", response_model=SearchResponse)
async def search_vectorstore_get(
    query: str = Query(..., description="The search query"),
    num_results: int = Query(4, description="Number of results to return", ge=1, le=20)
):
    try:
        results = hybrid_search(query, k=num_results)
        if not results:
            return JSONResponse(content={"documents": []})

        documents = [
            Document(
                content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": float(score)}
            ) for doc, score in results
        ]
        response_data = SearchResponse(documents=documents)
        json_compatible_data = jsonable_encoder(response_data, custom_encoder={np.float32: float})
        return JSONResponse(content=json_compatible_data)
    except Exception as e:
        logger.error(f"Error searching vector store: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    try:
        chat_history = chat_request.chat_history or []
        answer, citations, related_topics, _ = get_chat_response(chat_request.query, chat_history)
        response_data = ChatResponse(
            answer=answer,
            citations=citations,
            related_topics=related_topics
        )
        json_compatible_data = jsonable_encoder(response_data, custom_encoder=custom_jsonable_encoder)
        return JSONResponse(content=json_compatible_data)
    except Exception as e:
        logger.error(f"Error in chat response: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
