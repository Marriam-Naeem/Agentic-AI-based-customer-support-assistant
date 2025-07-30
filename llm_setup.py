from typing import Dict, Any
from langchain_groq import ChatGroq
from settings import GROQ_API_KEY, SMALL_MODEL, LARGE_MODEL

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

class GroqLLMManager:
    def __init__(self):
        self.models = {
            "router": ChatGroq(groq_api_key=GROQ_API_KEY, model=SMALL_MODEL, temperature=0.1, max_tokens=512),
            "refund": ChatGroq(groq_api_key=GROQ_API_KEY, model=LARGE_MODEL, temperature=0.2, max_tokens=1024),
            "issue_faq": ChatGroq(groq_api_key=GROQ_API_KEY, model=LARGE_MODEL, temperature=0.3, max_tokens=1024)
        }
    
    def get_model(self, model_type: str) -> ChatGroq:
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
    llm_manager = GroqLLMManager()
    return {
        "router_llm": llm_manager.get_model("router"),
        "refund_llm": llm_manager.get_model("refund"),
        "issue_faq_llm": llm_manager.get_model("issue_faq"),
        "llm_manager": llm_manager
    }

def setup_embedding_model():
    return EmbeddingManager().get_embeddings()

try:
    llm_models = setup_llm_models()
    embedding_model = setup_embedding_model()
except Exception as e:
    print(f"Error initializing models: {e}")
    llm_models = {}
    embedding_model = None