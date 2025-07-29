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

def test_groq_connection() -> bool:
    """Test the Groq API connection."""
    try:
        test_llm = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=GROQ_API_KEY,
            max_tokens=50
        )
        
        response = test_llm.invoke("Say 'Connection successful'")
        return "successful" in response.content.lower()
    
    except Exception as e:
        print(f"Groq connection test failed: {e}")
        return False

# Initialize models when module is imported
try:
    llm_models = setup_llm_models()
    print("✅ 3-Agent LLM models initialized successfully")
    
    if test_groq_connection():
        print("✅ Groq API connection verified")
    else:
        print("⚠️ Groq API connection test failed")
        
except Exception as e:
    print(f"❌ Error initializing models: {e}")
    llm_models = {}