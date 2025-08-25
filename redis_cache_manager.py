import logging
from langchain.globals import set_llm_cache
from langchain.schema import Generation
from langchain_redis import RedisCache, RedisSemanticCache
from settings import REDIS_URL, REDIS_CACHE_TTL, REDIS_SEMANTIC_DISTANCE_THRESHOLD, REDIS_CACHE_ENABLED

logger = logging.getLogger(__name__)

class RedisCacheManager:
    """Redis-based semantic caching for LLM responses"""
    
    def __init__(self, embeddings_model=None):
        self.cache_enabled = REDIS_CACHE_ENABLED
        self.cache_hits = self.cache_misses = self.total_queries = 0
        
        if self.cache_enabled:
            self._init_cache(embeddings_model)
        else:
            self.current_cache = None
    
    def _init_cache(self, embeddings_model):
        """Initialize cache with embeddings model if available"""
        try:
            if embeddings_model:
                self.current_cache = RedisSemanticCache(
                    redis_url=REDIS_URL, embeddings=embeddings_model,
                    distance_threshold=REDIS_SEMANTIC_DISTANCE_THRESHOLD, ttl=REDIS_CACHE_TTL
                )
                logger.info(f"Semantic cache enabled (threshold: {REDIS_SEMANTIC_DISTANCE_THRESHOLD})")
            else:
                self.current_cache = RedisCache(redis_url=REDIS_URL, ttl=REDIS_CACHE_TTL)
                logger.info("Exact cache enabled")
            
            set_llm_cache(self.current_cache)
        except Exception as e:
            logger.error(f"Cache init failed: {e}")
            self.cache_enabled = False
            self.current_cache = None
    
    def create_cache_key(self, user_message: str, agent_name: str) -> str:
        """Create standard cache key"""
        return f"{agent_name}:{user_message[:100]}"
    
    def log_cache_operation(self, operation: str, prompt: str, llm_name: str, cache_hit: bool):
        """Log cache operation and update stats"""
        self.total_queries += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        hit_rate = (self.cache_hits / self.total_queries) * 100 if self.total_queries > 0 else 0
        cache_type = "Semantic" if isinstance(self.current_cache, RedisSemanticCache) else "Exact"
        
        log_msg = f"""
CACHE OPERATION: {operation.upper()}
   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}
   LLM: {llm_name}
   Cache Hit: {'✅ YES' if cache_hit else '❌ NO'}
   Hit Rate: {hit_rate:.1f}% ({self.cache_hits}/{self.total_queries})
   Cache Type: {cache_type}
        """
        logger.info(log_msg)
        # print(log_msg)
    
    def check_cache_and_store(self, prompt: str, llm_name: str, response: str = None):
        """Check cache and store response if provided"""
        if not self.cache_enabled or not self.current_cache:
            return None
        
        try:
            cached_result = self.current_cache.lookup(prompt, llm_name)
            
            if cached_result:
                self.log_cache_operation("HIT", prompt, llm_name, True)
                return cached_result[0].text
            else:
                if response:
                    self.current_cache.update(prompt, llm_name, [Generation(text=response)])
                    self.log_cache_operation("STORE", prompt, llm_name, False)
                else:
                    self.log_cache_operation("MISS", prompt, llm_name, False)
                return None
                
        except Exception as e:
            logger.error(f"Cache operation failed: {e}")
            return None
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if not self.cache_enabled:
            return {"status": "disabled"}
        
        try:
            hit_rate = (self.cache_hits / self.total_queries) * 100 if self.total_queries > 0 else 0
            return {
                "status": "enabled",
                "cache_type": "semantic" if isinstance(self.current_cache, RedisSemanticCache) else "exact",
                "performance": {
                    "total_queries": self.total_queries,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "hit_rate_percent": round(hit_rate, 2)
                }
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

def create_cache_manager(embeddings_model=None) -> RedisCacheManager:
    return RedisCacheManager(embeddings_model)

def setup_caching_for_llm(llm_instance, cache_manager: RedisCacheManager):
    if cache_manager.cache_enabled and cache_manager.current_cache:
        logger.info(f"Caching enabled for {type(llm_instance).__name__}")
    else:
        logger.warning("Caching not available")