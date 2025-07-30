from typing import Literal
from langgraph.graph import StateGraph, START
from states import SupportState
from llm_setup import setup_llm_models
from nodes import NodeFunctions

models = setup_llm_models()
node_functions = NodeFunctions(models)

def router_condition(state: SupportState) -> Literal["refund_node", "non_refund_node", "__end__"]:
    query_type = state.get("query_type", "")
    return "refund_node" if query_type == "refund" else "non_refund_node" if query_type in ["issue", "faq"] else "__end__"

def refund_condition(state: SupportState) -> Literal["refund_tools", "__end__"]:
    if state.get("final_response") or state.get("tool_execution_error"):
        return "__end__"
    last_llm_response = state.get("last_llm_response")
    if last_llm_response and hasattr(last_llm_response, 'tool_calls') and last_llm_response.tool_calls:
        return "refund_tools"
    verification_result = state.get("verification_result")
    return "__end__" if verification_result and (not verification_result.get("verified") or state.get("processing_result")) else "__end__"

def non_refund_condition(state: SupportState) -> Literal["rag_tools", "__end__"]:
    if state.get("final_response") or state.get("escalation_required"):
        return "__end__"
    last_llm_response = state.get("last_llm_response")
    if last_llm_response and hasattr(last_llm_response, 'tool_calls') and last_llm_response.tool_calls:
        return "rag_tools"
    return "non_refund_node" if state.get("search_results") and not state.get("final_response") else "__end__"

def refund_tools_condition(state: SupportState) -> Literal["refund_node", "__end__"]:
    return "refund_node"

def rag_tools_condition(state: SupportState) -> Literal["non_refund_node", "__end__"]:
    return "__end__" if state.get("escalation_required") or state.get("tool_execution_error") else "non_refund_node"

builder = StateGraph(SupportState)
builder.add_node("router_node", node_functions.router_agent)
builder.add_node("refund_node", node_functions.refund_node)
builder.add_node("non_refund_node", node_functions.non_refund_node)
builder.add_node("refund_tools", node_functions.execute_refund_tools)
builder.add_node("rag_tools", node_functions.execute_rag_tools)
builder.add_edge(START, "router_node")
builder.add_conditional_edges("router_node", router_condition)
builder.add_conditional_edges("refund_node", refund_condition)
builder.add_conditional_edges("refund_tools", refund_tools_condition)
builder.add_conditional_edges("non_refund_node", non_refund_condition)
builder.add_conditional_edges("rag_tools", rag_tools_condition)
graph = builder.compile()