"""
config/llm_setup.py

Simplified setup for language models using Groq API for 3-agent system.
"""

import os
from typing import Dict, Any

from langchain_groq import ChatGroq

from settings import (
    GROQ_API_KEY, 
    SMALL_MODEL,
    LARGE_MODEL
)

# Import embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

class GroqLLMManager:
    """
    Manages Groq LLM instances for the 3-agent system.
    """
    
    def __init__(self):
        self.models = {}
        self._setup_models()
    
    def _setup_models(self):
        """Initialize the 3 required LLM models."""
        
        model_configs = {
            "router": {
                "model": SMALL_MODEL,
                "temperature": 0.1,
                "max_tokens": 512
            },
            "refund": {
                "model": LARGE_MODEL,
                "temperature": 0.2,
                "max_tokens": 1024
            },
            "issue_faq": {
                "model": LARGE_MODEL,
                "temperature": 0.3,
                "max_tokens": 1024
            }
        }
        
        for model_type, config in model_configs.items():
            self.models[model_type] = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                **config
            )
    
    def get_model(self, model_type: str) -> ChatGroq:
        """Get a specific model instance."""
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not found. Available: {list(self.models.keys())}")
        
        return self.models[model_type]


class EmbeddingManager:
    """
    Manages embedding models for the RAG system.
    """
    
    def __init__(self):
        self.embeddings = None
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Initialize the embedding model."""
        try:
            import torch
            # Force CPU and explicit dtype to avoid meta tensor issues
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    'device': 'cpu',
                    'torch_dtype': torch.float32,
                    'low_cpu_mem_usage': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            # Try alternative model
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                    model_kwargs={'device': 'cpu'}
                )
            except Exception as e2:
                raise e2
    
    def get_embeddings(self):
        """Get the embedding model instance."""
        return self.embeddings


def setup_llm_models() -> Dict[str, Any]:
    """
    Initialize and return all language models for the 3-agent system.
    """
    
    llm_manager = GroqLLMManager()
    
    # Get the 3 agent models
    router_llm = llm_manager.get_model("router")
    refund_llm = llm_manager.get_model("refund")
    issue_faq_llm = llm_manager.get_model("issue_faq")
    
    return {
        "router_llm": router_llm,
        "refund_llm": refund_llm,
        "issue_faq_llm": issue_faq_llm,
        "llm_manager": llm_manager
    }


def setup_embedding_model():
    """
    Initialize and return the embedding model for RAG system.
    """
    embedding_manager = EmbeddingManager()
    return embedding_manager.get_embeddings()


def test_groq_connection() -> bool:
    """Test the Groq API connection."""
    try:
        from langchain_groq import ChatGroq
        test_llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=SMALL_MODEL,
            max_tokens=10
        )
        response = test_llm.invoke("Test")
        return True
    except Exception:
        return False

# Initialize models when module is imported
try:
    llm_models = setup_llm_models()
    embedding_model = setup_embedding_model()
    
    if not test_groq_connection():
        print("⚠️ Groq API connection test failed")
        
except Exception as e:
    print(f"Error initializing models: {e}")
    llm_models = {}
    embedding_model = None