from langgraph.graph import StateGraph, START, END
from states import SupportState
from llm_setup import setup_llm_models
from nodes import NodeFunctions
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()

models = setup_llm_models()
node_functions = NodeFunctions(models)

# Simplified graph for SmolAgents
builder = StateGraph(SupportState)
builder.add_node("router_node", node_functions.router_agent)
builder.add_node("formatter_node", node_functions.formatter_agent)
builder.add_edge(START, "router_node")
builder.add_edge("router_node", "formatter_node")
builder.add_edge("formatter_node", END)
graph = builder.compile(checkpointer=memory)