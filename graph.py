"""
simple_workflow.py

Simple LangGraph workflow using existing nodes directly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Literal
from langgraph.graph import StateGraph, END, START
from IPython.display import Image, display

from states import SupportState, create_initial_state
from llm_setup import setup_llm_models
from nodes import NodeFunctions


# Initialize once
models = setup_llm_models()
node_functions = NodeFunctions(models)

def router_condition(state: SupportState) -> Literal["refund_node", "__end__"]:
    return "refund_node" if state.get("query_type") == "refund" else "__end__"

def refund_condition(state: SupportState) -> Literal["tools", "__end__"]:
    """Determine if we need to execute tools or end the workflow."""
    verification_result = state.get("verification_result")
    processing_result = state.get("processing_result")
    last_llm_response = state.get("last_llm_response")
    
    if state.get("final_response"):
        return "__end__"
    
    if state.get("error") or state.get("tool_execution_error"):
        return "__end__"
    
    if last_llm_response and hasattr(last_llm_response, 'tool_calls') and last_llm_response.tool_calls:
        return "tools"
    
    if verification_result and not verification_result.get("verified"):
        return "__end__"
    
    if verification_result and processing_result:
        return "__end__"
    
    return "__end__"

builder = StateGraph(SupportState)

builder.add_node("router_node", node_functions.router_agent)
builder.add_node("refund_node", node_functions.refund_node)
builder.add_node("tools", node_functions.execute_refund_tools)

builder.add_edge(START, "router_node")
builder.add_conditional_edges("router_node", router_condition)
builder.add_conditional_edges("refund_node", refund_condition)

def tools_condition(state: SupportState) -> Literal["refund_node", "__end__"]:
    """After tool execution, determine if we need to continue to refund_node for next step."""
    return "refund_node"

builder.add_edge("tools", "refund_node")

graph = builder.compile()

def test_workflow():
    """Test the workflow."""
    
    test_cases = [
        "I want a refund for order #12345, my email is john@example.com",
        "Refund for order 99999, email nobody@example.com", 
        "I need a refund for my order",
        "How do I track my order?"
    ]
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}: {message}")
        print('='*50)
        
        initial_state = create_initial_state(message, f"session_{i}")
        
        try:
            result = graph.invoke(initial_state)
            
            print(f"RESULT:")
            print(f"Query Type: {result.get('query_type')}")
            print(f"Resolved: {result.get('resolved', False)}")
            print(f"Escalation: {result.get('escalation_required', False)}")
            
            if result.get('final_response'):
                print(f"Final Response: {result.get('final_response')}")
            elif result.get('error'):
                print(f"Error: {result.get('error')}")
            elif result.get('verification_result'):
                if result.get('processing_result'):
                    print(f"Processing Complete: {result.get('processing_result', {}).get('message', 'Refund processed')}")
                else:
                    print(f"Verification Result: {result.get('verification_result', {}).get('message', 'Verification completed')}")
            
        except Exception as e:
            print(f"ERROR: {e}")

def visualize():
    """Show the graph."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(png_data)
        print("Graph saved as workflow.png")
        
        display(Image(png_data))
            
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    test_workflow()
    visualize()