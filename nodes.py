import json
import time
import logging
from typing import Dict, Any, Optional
from smolagents import ToolCallingAgent, tool
from states import SupportState
from settings import (
    REFUND_AGENT_INSTRUCTIONS, 
    SUPPORT_AGENT_INSTRUCTIONS, 
    MANAGER_AGENT_PROMPT_TEMPLATE, 
    FORMATTER_AGENT_PROMPT_TEMPLATE,
    FALLBACK_RESPONSE,
    RATE_LIMIT_RESPONSE,
    RATE_LIMIT_KEYWORDS,
    MANAGED_AGENT_PROMPT_TASK
)
from refund_tools import get_refund_tools
from rag_tools import get_document_search_tools

try:
    from redis_cache_manager import create_cache_manager, setup_caching_for_llm
    REDIS_CACHING_AVAILABLE = True
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Redis caching not available: {e}")
    REDIS_CACHING_AVAILABLE = False
    logger = None

# Pre-load tools once at module level
refund_tools_list = get_refund_tools()
document_tools_list = get_document_search_tools()

@tool
def refund_verification_tool(order_id: str, customer_email: str = None) -> str:
    """Verify refund eligibility before processing.
    
    Args:
        order_id: The order ID to verify for refund eligibility
        customer_email: The customer's email address for verification
    """
    for tool in refund_tools_list:
        if tool.name == "refund_verification_tool":
            return tool.func(order_id=order_id, customer_email=customer_email)
    return json.dumps({"error": "Tool not found"})

@tool
def refund_processing_tool(order_id: str) -> str:
    """Execute actual refund processing after verification. ONLY call this after successful verification. WARNING: This tool should only be called if refund_verification_tool returned 'verified': true.
    
    Args:
        order_id: The order ID to process refund for
    """
    for tool in refund_tools_list:
        if tool.name == "refund_processing_tool":
            return tool.func(order_id=order_id)
    return json.dumps({"error": "Tool not found"})

@tool
def document_search_tool(query: str, max_results: int = 2) -> str:
    """Search through company documents using vector similarity from the configured vector store path.
    
    Args:
        query: The search query to find relevant documents
        max_results: Maximum number of results to return (default: 2)
    """
    for tool in document_tools_list:
        if tool.name == "document_search_tool":
            return tool.func(query=query, max_results=max_results)
    return json.dumps({"error": "Tool not found"})

def create_smolagents_system(models):
    """Create SmolAgents multi-agent system following the documentation pattern"""
    
    # Setup Redis caching for all agents if available
    cache_manager = models.get("cache_manager") if REDIS_CACHING_AVAILABLE else None
    
    if cache_manager:
        logger.info("Redis caching enabled for SmolAgents system")
    
    # Create agents with optimized configuration
    refund_agent = ToolCallingAgent(
        tools=[refund_verification_tool, refund_processing_tool],
        model=models["refund_llm"],
        max_steps=2, 
        name="refund_agent",
        description="""Agent that verifies refund eligibility and, if eligible, 
        processes the refund and generates a final answer with the decision.""",
        instructions=REFUND_AGENT_INSTRUCTIONS,
    )
     
    support_agent = ToolCallingAgent(
        tools=[document_search_tool],
        model=models["issue_faq_llm"],
        max_steps=3,  
        verbosity_level=1, 
        name="support_agent",
        description="""Support agent that answers user queries by searching company documents, 
        provides step-by-step answers, and never reveals the document source to the user.""",
        instructions=SUPPORT_AGENT_INSTRUCTIONS
    )

    manager_agent = ToolCallingAgent(
        tools=[],
        model=models["router_llm"],
        managed_agents=[refund_agent, support_agent],
        name="manager_agent",
        description="Manager agent that autonomously decides which specialized agent to call based on customer request analysis.",
        max_steps=3, 
        verbosity_level=1  
    )

    manager_agent.prompt_templates["system_prompt"] = manager_agent.initialize_system_prompt()
    support_agent.prompt_templates["system_prompt"] = SUPPORT_AGENT_INSTRUCTIONS
    refund_agent.prompt_templates["system_prompt"] = REFUND_AGENT_INSTRUCTIONS
    refund_agent.prompt_templates["managed_agent"]["task"] = MANAGED_AGENT_PROMPT_TASK
    support_agent.prompt_templates["managed_agent"]["task"] = MANAGED_AGENT_PROMPT_TASK

    formatter_agent = ToolCallingAgent(
        tools=[],
        model=models["router_llm"],
        name="formatter_agent",
        description="Formatter agent that formats the response from the manager agent into a professional email response with proper start and end."
    )

    return {
        "manager_agent": manager_agent,
        "refund_agent": refund_agent,
        "support_agent": support_agent,
        "formatter_agent": formatter_agent,
        "cache_manager": cache_manager
    }

def _handle_rate_limit_error(error: Exception) -> str:
    """Centralized rate limit error handling"""
    error_str = str(error).lower()
    return RATE_LIMIT_RESPONSE if any(keyword in error_str for keyword in RATE_LIMIT_KEYWORDS) else FALLBACK_RESPONSE

def _log_cache_status(cache_manager, operation: str, logger_enabled: bool = True):
    """Centralized cache status logging"""
    if not cache_manager or not REDIS_CACHING_AVAILABLE:
        return
    
    cache_stats = cache_manager.get_cache_stats()
    status = cache_stats.get('status', 'unknown')
    
    if logger_enabled and logger:
        logger.info(f"{operation} - Cache Status: {status}")
    print(f"{operation} - Cache Status: {status}")

def _log_performance_metrics(cache_manager, response_time: float, logger_enabled: bool = True):
    """Centralized performance metrics logging"""
    if not cache_manager or not REDIS_CACHING_AVAILABLE:
        return
    
    perf_stats = cache_manager.get_cache_stats().get("performance", {})
    hit_rate = perf_stats.get('hit_rate_percent', 0)
    total_queries = perf_stats.get('total_queries', 0)
    
    if logger_enabled and logger:
        logger.info(f"Response generated in {response_time:.3f}s, Cache Hit Rate: {hit_rate}%, Total Queries: {total_queries}")
    print(f"Response generated in {response_time:.3f}s | Cache Hit Rate: {hit_rate}% | Total Queries: {total_queries}")

def process_with_smolagents(smolagents_system: Dict[str, Any], user_message: str, user_message_only: Optional[str] = None) -> str:
    """Process user message with autonomous SmolAgents multi-agent system using semantic caching"""
    try:
        manager_agent = smolagents_system["manager_agent"]
        cache_manager = smolagents_system.get("cache_manager")
        
        # Log cache status
        _log_cache_status(cache_manager, "Processing query")
        
        prompt = MANAGER_AGENT_PROMPT_TEMPLATE.format(user_message=user_message)
        
        # Check cache first
        cached_response = None
        cache_key = None
        
        if cache_manager and REDIS_CACHING_AVAILABLE:
            cache_key = cache_manager.create_cache_key(user_message_only or user_message, "manager_agent")
            logger.info(f"Checking semantic cache with key: {cache_key[:80]}...")
            
            cached_response = cache_manager.check_cache_and_store(cache_key, "manager_agent", user_message=user_message_only or user_message)
            if cached_response:
                logger.info("Semantic cache hit - using cached response")
                return cached_response
        
        # Process with LLM if no cache hit
        start_time = time.time()
        result = manager_agent.run(prompt)
        response_time = time.time() - start_time
        
        # Store response in cache
        if cache_manager and REDIS_CACHING_AVAILABLE and cache_key:
            logger.info(f"Storing in semantic cache with key: {cache_key[:80]}...")
            cache_manager.check_cache_and_store(cache_key, "manager_agent", result, user_message=user_message_only or user_message)
        
        # Log performance metrics
        _log_performance_metrics(cache_manager, response_time)
        
        print(f"[DEBUG] Final email_content: {result[:500]}")
        return result
            
    except Exception as e:
        return _handle_rate_limit_error(e)

def format_with_smolagents(smolagents_system: Dict[str, Any], user_message: str) -> str:
    """Format the response from the manager agent into a professional email response."""
    try:
        prompt = FORMATTER_AGENT_PROMPT_TEMPLATE.format(user_message=user_message)
        return smolagents_system["formatter_agent"].run(prompt)
    except Exception as e:
        return _handle_rate_limit_error(e)

def _build_conversation_context(conversation_history: list, user_message: str) -> str:
    """Build conversation context efficiently"""
    if not conversation_history:
        return f"Current message: {user_message}\n\n"
    
    context_parts = ["Previous conversation:"]
    for user_msg, bot_response in conversation_history[-3:]:
        context_parts.extend([f"User: {user_msg}", f"Bot: {bot_response}", ""])
    context_parts.extend([f"Current message: {user_message}", ""])
    
    return "\n".join(context_parts)

class NodeFunctions:
    """Optimized NodeFunctions class with improved performance and maintainability"""
    
    def __init__(self, models: Dict[str, Any]):
        self.router_llm = models.get("router_llm")
        self.refund_llm = models.get("refund_llm")
        self.issue_faq_llm = models.get("issue_faq_llm")
        self.cache_manager = models.get("cache_manager")
        
        self.smolagents_system = create_smolagents_system(models)
        
        # Log cache integration status
        _log_cache_status(self.cache_manager, "NodeFunctions initialized")
    
    def router_agent(self, state: SupportState) -> dict:
        """Use autonomous SmolAgents multi-agent system for intelligent processing"""
        user_message = state.get("user_message", "")
        conversation_history = state.get("conversation_history", [])
        
        # Log cache performance at start
        _log_cache_status(self.cache_manager, "Router Agent", logger_enabled=False)
        
        # Build context efficiently
        context = _build_conversation_context(conversation_history, user_message)
        user_message_only = user_message.strip().lower()
        
        try:
            response_content = process_with_smolagents(self.smolagents_system, context, user_message_only)
            
            # Log cache performance after processing
            if self.cache_manager and REDIS_CACHING_AVAILABLE:
                end_stats = self.cache_manager.get_cache_stats()
                perf = end_stats.get("performance", {})
                hit_rate = perf.get('hit_rate_percent', 0)
                total_queries = perf.get('total_queries', 0)
                
                print(f"Router Agent Complete - Hit Rate: {hit_rate}% | Total Queries: {total_queries}")
                if logger:
                    logger.info(f"Router Agent complete - Hit Rate: {hit_rate}%, Total Queries: {total_queries}")
            
            state.update({"final_response": response_content})
            
        except Exception as e:
            final_response = _handle_rate_limit_error(e)
            state.update({"final_response": final_response})
        
        return state
    
    def formatter_agent(self, state: SupportState) -> dict:
        """Formatter agent that formats responses into professional emails."""
        final_response = state.get("final_response", "")
        user_message = state.get("user_message", "")
        
        try:
            formatted_email = format_with_smolagents(self.smolagents_system, final_response)
            print("DEBUG: FINAL GENERATED EMAIL: ", formatted_email)
            state.update({
                "final_email": formatted_email
            })
        except Exception as e:
            formatted_email = _handle_rate_limit_error(e)
            state.update({"final_email": formatted_email})
        
        # Update conversation history efficiently
        conversation_history = state.get("conversation_history", [])
        conversation_history.append([user_message, state.get("final_email", final_response)])
        state.update({"conversation_history": conversation_history})
        
        return state