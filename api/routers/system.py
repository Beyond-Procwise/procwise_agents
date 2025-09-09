# ProcWise/api/routers/system.py

from fastapi import APIRouter, Depends, HTTPException, Request
import ollama
from typing import Any, Dict, List, Union
from pydantic import BaseModel
from services.model_selector import RAGPipeline
from utils.gpu import configure_gpu

configure_gpu()


class ChatMessage(BaseModel):
    query: str
    answer: Union[str, Dict[str, Any], List[Any]]

router = APIRouter(prefix="/system", tags=["System Information & History"])

def get_rag_pipeline(request: Request) -> RAGPipeline:
    if not hasattr(request.app.state, 'rag_pipeline') or not request.app.state.rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline service is not available.")
    return request.app.state.rag_pipeline

@router.get("/models", summary="List available Ollama models")
def list_local_models():
    try:
        return {"models": ollama.list().get('models', [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not connect to Ollama service: {e}")

@router.get("/history/{user_id}", response_model=List[ChatMessage], summary="Retrieve chat history for a user from S3")
def get_user_chat_history(user_id: str, pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    try:
        return pipeline.history_manager.get_history(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history from S3: {e}")
