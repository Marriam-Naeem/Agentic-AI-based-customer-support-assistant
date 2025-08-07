from typing import Dict, Any, Union
import litellm  # Add this missing import
from smolagents import OpenAIServerModel
from settings import GROQ_API_KEY, SMALL_MODEL, LARGE_MODEL, HUGGINGFACE_TOKEN, GEMINI_API_KEY
from ollamaModel import OllamaLLM
from langchain_community.llms import Ollama
import time
import random


try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings


class SmolAgentsLLMManager:
    def __init__(self):
        # Use Groq directly without fallback
        print("Using Gemini Models")
        try:
            self.models = {
                "router": OpenAIServerModel(model_id="gemini-1.5-flash", api_base="https://generativelanguage.googleapis.com/v1beta/openai/",api_key=GEMINI_API_KEY,),
                "refund": OpenAIServerModel(model_id="gemini-1.5-flash", api_base="https://generativelanguage.googleapis.com/v1beta/openai/",api_key=GEMINI_API_KEY,),
                "issue_faq": OpenAIServerModel(model_id="gemini-1.5-flash", api_base="https://generativelanguage.googleapis.com/v1beta/openai/",api_key=GEMINI_API_KEY,),
            }
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error initializing Gemini models: {e}")
            raise
    
    def get_model(self, model_type: str) -> OpenAIServerModel:
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not found. Available: {list(self.models.keys())}")
        return self.models[model_type]


class EmbeddingManager:
    def __init__(self):
        try:
            import torch
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu', 'torch_dtype': torch.float32, 'low_cpu_mem_usage': True},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
                model_kwargs={'device': 'cpu'}
            )
    
    def get_embeddings(self):
        return self.embeddings


def setup_llm_models() -> Dict[str, Any]:
    try:
        llm_manager = SmolAgentsLLMManager()
        return {
            "router_llm": llm_manager.get_model("router"),
            "refund_llm": llm_manager.get_model("refund"),
            "issue_faq_llm": llm_manager.get_model("issue_faq"),
            "llm_manager": llm_manager
        }
    except Exception as e:
        print(f"Error in setup_llm_models: {e}")
        raise


def setup_embedding_model():
    try:
        return EmbeddingManager().get_embeddings()
    except Exception as e:
        print(f"Error in setup_embedding_model: {e}")
        raise


# Initialize models at module level with proper error handling
try:
    llm_models = setup_llm_models()
    embedding_model = setup_embedding_model()
    print("All models initialized successfully")
except Exception as e:
    print(f"Error initializing models: {e}")
    llm_models = {}
    embedding_model = None