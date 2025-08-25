from langgraph.graph import StateGraph, START, END
from states import SupportState
from llm_setup import setup_llm_models
from nodes import NodeFunctions
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

# Setup models with Redis caching
print("Setting up LangGraph workflow with Redis semantic caching...")
models = setup_llm_models()

# Check if caching is available
if "cache_manager" in models and models["cache_manager"]:
    cache_stats = models["cache_manager"].get_cache_stats()
    print(f"Redis caching enabled: {cache_stats.get('status', 'unknown')}")
    print(f"Cache Type: {cache_stats.get('cache_type', 'Unknown')}")
    print(f"TTL: {cache_stats.get('ttl_seconds', 0)} seconds")
    print(f"Distance Threshold: {cache_stats.get('distance_threshold', 'N/A')}")
else:
    print("Redis caching not available")

node_functions = NodeFunctions(models)

# Simplified graph for SmolAgents
builder = StateGraph(SupportState)
builder.add_node("router_node", node_functions.router_agent)
builder.add_node("formatter_node", node_functions.formatter_agent)
builder.add_edge(START, "router_node")
builder.add_edge("router_node", "formatter_node")
builder.add_edge("formatter_node", END)
graph = builder.compile(checkpointer=memory)

print("LangGraph workflow compiled successfully!")
print("=" * 60)