from typing import Dict, Any
from smolagents import OpenAIServerModel
from settings import GEMINI_API_KEY

try:
    from redis_cache_manager import create_cache_manager, setup_caching_for_llm
    REDIS_CACHING_AVAILABLE = True
except ImportError as e:
    print(f"Redis caching not available: {e}")
    REDIS_CACHING_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings

class SmolAgentsLLMManager:
    def __init__(self):
        print("Using Gemini Models")
        try:
            self.gemini_model = OpenAIServerModel(
                model_id="gemini-2.0-flash-lite", 
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=GEMINI_API_KEY,
            )
            self.models = {
                "router": self.gemini_model,
                "refund": self.gemini_model,
                "issue_faq": self.gemini_model,
            }
        except Exception as e:
            print(f"Error initializing Gemini models: {e}")
            raise
    
    def get_model(self, model_type: str) -> OpenAIServerModel:
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not found. Available: {list(self.models.keys())}")
        return self.models[model_type]

def setup_embedding_model():
    try:
        import torch
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu', 'torch_dtype': torch.float32, 'low_cpu_mem_usage': True},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
            model_kwargs={'device': 'cpu'}
        )

def setup_llm_models() -> Dict[str, Any]:
    try:
        llm_manager = SmolAgentsLLMManager()
        cache_manager = None
        
        if REDIS_CACHING_AVAILABLE:
            try:
                embedding_model = setup_embedding_model()
                cache_manager = create_cache_manager(embedding_model)
                for model_type in ["router", "refund", "issue_faq"]:
                    setup_caching_for_llm(llm_manager.get_model(model_type), cache_manager)
                print("Redis semantic caching enabled for all LLM models")
            except Exception as e:
                print(f"Failed to setup Redis caching: {e}")
        
        return {
            "router_llm": llm_manager.get_model("router"),
            "refund_llm": llm_manager.get_model("refund"),
            "issue_faq_llm": llm_manager.get_model("issue_faq"),
            "llm_manager": llm_manager,
            "cache_manager": cache_manager
        }
    except Exception as e:
        print(f"Error in setup_llm_models: {e}")
        raise