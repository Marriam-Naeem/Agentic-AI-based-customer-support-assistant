import logging
import re
from langchain.globals import set_llm_cache
from langchain.schema import Generation
from langchain_redis import RedisCache, RedisSemanticCache
from settings import REDIS_URL, REDIS_CACHE_TTL, REDIS_SEMANTIC_DISTANCE_THRESHOLD, REDIS_CACHE_ENABLED

logger = logging.getLogger(__name__)

class RedisCacheManager:
    """Redis-based semantic caching for LLM responses"""
    
    def __init__(self, embeddings_model=None):
        self.cache_enabled = REDIS_CACHE_ENABLED
        if self.cache_enabled:
            self._init_cache(embeddings_model)
        else:
            self.current_cache = None
    
    def _init_cache(self, embeddings_model):
        """Initialize cache with embeddings model if available"""
        try:
            if embeddings_model:
                self.refund_cache = RedisSemanticCache(
                    redis_url=REDIS_URL, embeddings=embeddings_model,
                    distance_threshold=0.01, ttl=REDIS_CACHE_TTL
                )
                self.normal_cache = RedisSemanticCache(
                    redis_url=REDIS_URL, embeddings=embeddings_model,
                    distance_threshold=REDIS_SEMANTIC_DISTANCE_THRESHOLD, ttl=REDIS_CACHE_TTL
                )
                self.current_cache = self.normal_cache
                logger.info(f"Semantic cache enabled with dual thresholds: refund=0.01, normal={REDIS_SEMANTIC_DISTANCE_THRESHOLD}")
            else:
                self.current_cache = RedisCache(redis_url=REDIS_URL, ttl=REDIS_CACHE_TTL)
                logger.info("Exact cache enabled")
            set_llm_cache(self.current_cache)
        except Exception as e:
            logger.error(f"Cache init failed: {e}")
            self.cache_enabled = False
            self.current_cache = None
    
    def _is_refund_query(self, user_message: str) -> bool:
        """Check if the query is refund-related"""
        refund_keywords = ['refund', 'refunds', 'refunded', 'refunding', 'return', 'returns', 'returned', 'returning', 'money back', 'moneyback', 'reimbursement', 'cancel order', 'canceled order', 'cancelled order']
        return any(keyword in user_message.lower() for keyword in refund_keywords)
    
    def _extract_order_number(self, user_message: str) -> str:
        """Extract order number from refund queries"""
        matches = re.findall(r'[A-Za-z0-9_-]*\d+[A-Za-z0-9_-]*', user_message)
        return matches[0] if matches else ""
    
    def create_cache_key(self, user_message: str, agent_name: str) -> str:
        """Create cache key with special handling for refund queries"""
        if self._is_refund_query(user_message):
            order_number = self._extract_order_number(user_message)
            return f"refund:{order_number}" if order_number else f"refund_no_order:{user_message[:100]}"
        return user_message[:100]
    
    def adjust_cache_threshold_for_refund(self, user_message: str):
        """Dynamically switch between cache instances based on query type"""
        if not hasattr(self, 'refund_cache') or not hasattr(self, 'normal_cache'):
            return
        target_cache = self.refund_cache if self._is_refund_query(user_message) else self.normal_cache
        if self.current_cache != target_cache:
            self.current_cache = target_cache
            set_llm_cache(self.current_cache)
            threshold = "0.01 (refund cache)" if target_cache == self.refund_cache else f"{REDIS_SEMANTIC_DISTANCE_THRESHOLD} (normal cache)"
            logger.info(f"Cache SWITCHED to {'refund' if target_cache == self.refund_cache else 'normal'} cache with threshold {threshold}")
    
    def log_cache_operation(self, operation: str, prompt: str, llm_name: str, cache_hit: bool):
        """Log cache operation and update stats"""
        self.total_queries += 1
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        hit_rate = (self.cache_hits / self.total_queries) * 100 if self.total_queries > 0 else 0
        cache_type = "Semantic" if isinstance(self.current_cache, RedisSemanticCache) else "Exact"
        log_msg = f"CACHE OPERATION: {operation.upper()}\n   Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n   LLM: {llm_name}\n   Cache Hit: {'✅ YES' if cache_hit else '❌ NO'}\n   Hit Rate: {hit_rate:.1f}% ({self.cache_hits}/{self.total_queries})\n   Cache Type: {cache_type}"
        logger.info(log_msg)
    
    def check_cache_and_store(self, cache_key: str, llm_name: str, response: str = None, user_message: str = None):
        """Check cache and store response if provided"""
        if not self.cache_enabled or not self.current_cache:
            return None
        try:
            if user_message:
                self.adjust_cache_threshold_for_refund(user_message)
            cached_result = self.current_cache.lookup(cache_key, llm_name)
            if cached_result:
                self.log_cache_operation("HIT", cache_key, llm_name, True)
                return cached_result[0].text
            else:
                if response:
                    self.current_cache.update(cache_key, llm_name, [Generation(text=response)])
                    self.log_cache_operation("STORE", cache_key, llm_name, False)
                else:
                    self.log_cache_operation("MISS", cache_key, llm_name, False)
                return None
        except Exception as e:
            logger.error(f"Cache operation failed: {e}")
            return None

def create_cache_manager(embeddings_model=None) -> RedisCacheManager:
    return RedisCacheManager(embeddings_model)

def setup_caching_for_llm(llm_instance, cache_manager: RedisCacheManager):
    if cache_manager.cache_enabled and cache_manager.current_cache:
        logger.info(f"Caching enabled for {type(llm_instance).__name__}")
    else:
        logger.warning("Caching not available")