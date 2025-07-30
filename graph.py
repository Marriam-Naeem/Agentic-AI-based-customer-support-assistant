"""
graph.py

Updated LangGraph workflow with RAG support for non-refund queries.
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

def router_condition(state: SupportState) -> Literal["refund_node", "non_refund_node", "__end__"]:
    """Route based on query type classification."""
    query_type = state.get("query_type", "")
    
    if query_type == "refund":
        return "refund_node"
    elif query_type in ["issue", "faq"]:
        return "non_refund_node"
    else:
        return "__end__"

def refund_condition(state: SupportState) -> Literal["refund_tools", "__end__"]:
    """Determine if we need to execute refund tools or end the workflow."""
    verification_result = state.get("verification_result")
    processing_result = state.get("processing_result")
    last_llm_response = state.get("last_llm_response")
    
    if state.get("final_response"):
        return "__end__"
    
    if state.get("tool_execution_error"):
        return "__end__"
    
    if last_llm_response and hasattr(last_llm_response, 'tool_calls') and last_llm_response.tool_calls:
        return "refund_tools"
    
    if verification_result and not verification_result.get("verified"):
        return "__end__"
    
    if verification_result and processing_result:
        return "__end__"
    
    return "__end__"

def non_refund_condition(state: SupportState) -> Literal["rag_tools", "__end__"]:
    """Determine if we need to execute RAG tools or end the workflow."""
    last_llm_response = state.get("last_llm_response")
    search_results = state.get("search_results", [])
    
    # If we have a final response, end
    if state.get("final_response"):
        return "__end__"
    
    # If escalation is required, end
    if state.get("escalation_required"):
        return "__end__"
    
    # If we have tool calls to execute, go to RAG tools
    if last_llm_response and hasattr(last_llm_response, 'tool_calls') and last_llm_response.tool_calls:
        return "rag_tools"
    
    # If we have search results but no final response, continue processing in non_refund_node
    if search_results and not state.get("final_response"):
        return "non_refund_node"  # Continue to non_refund_node to generate final response
    
    return "__end__"

def refund_tools_condition(state: SupportState) -> Literal["refund_node", "__end__"]:
    """After refund tool execution, determine if we need to continue processing."""
    return "refund_node"

def rag_tools_condition(state: SupportState) -> Literal["non_refund_node", "__end__"]:
    """After RAG tool execution, continue to non_refund_node for final answer generation."""
    if state.get("escalation_required") or state.get("tool_execution_error"):
        return "__end__"
    return "non_refund_node"

# Build the workflow graph
builder = StateGraph(SupportState)

# Add nodes
builder.add_node("router_node", node_functions.router_agent)
builder.add_node("refund_node", node_functions.refund_node)
builder.add_node("non_refund_node", node_functions.non_refund_node)
builder.add_node("refund_tools", node_functions.execute_refund_tools)
builder.add_node("rag_tools", node_functions.execute_rag_tools)

# Add edges
builder.add_edge(START, "router_node")
builder.add_conditional_edges("router_node", router_condition)

# Refund workflow edges
builder.add_conditional_edges("refund_node", refund_condition)
builder.add_conditional_edges("refund_tools", refund_tools_condition)

# Non-refund (RAG) workflow edges
builder.add_conditional_edges("non_refund_node", non_refund_condition)
builder.add_conditional_edges("rag_tools", rag_tools_condition)

# Compile the graph
graph = builder.compile()

def test_workflow():
    """Test the updated workflow with RAG support."""
    
    test_cases = [
        # Refund cases
        "I want a refund for order #12345, my email is john@example.com",
        "Refund for order 99999, email nobody@example.com",
        
        # Technical issue cases
        "My TechOffice Suite installation keeps failing with error 1603",
        "The software crashes every time I try to open it",
        "I can't login to my account, it says invalid password",
        
        # FAQ cases
        "How do I reset my password?",
        "What's your return policy?",
        "How do I track my order?",
        "What payment methods do you accept?",
        
        # Complex multi-part queries
        "My software won't install and I also need to know how to change my email address",
        "I'm having trouble with installation error 1603 and want to know about your refund policy"
    ]
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {message}")
        print('='*60)
        
        initial_state = create_initial_state(message, f"session_{i}")
        
        try:
            result = graph.invoke(initial_state)
            
            print(f"\nRESULT:")
            print(f"Query Type: {result.get('query_type')}")
            print(f"Resolved: {result.get('resolved', False)}")
            print(f"Escalation Required: {result.get('escalation_required', False)}")
            
            if result.get('subquestions'):
                print(f"Subquestions: {result.get('subquestions')}")
            
            if result.get('search_results'):
                print(f"Documents Found: {len(result.get('search_results', []))}")
            
            if result.get('final_response'):
                print(f"\nFinal Response:")
                print(f"{result.get('final_response')}")
            elif result.get('tool_execution_error'):
                print(f"\nError: {result.get('tool_execution_error')}")
            elif result.get('verification_result'):
                if result.get('processing_result'):
                    print(f"\nRefund Processed: {result.get('processing_result', {}).get('message', 'Refund completed')}")
                else:
                    print(f"\nVerification Result: {result.get('verification_result', {}).get('message', 'Verification completed')}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

def visualize():
    """Show the updated graph."""
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open("updated_workflow.png", "wb") as f:
            f.write(png_data)
        print("Updated graph saved as updated_workflow.png")
        
        display(Image(png_data))
            
    except Exception as e:
        print(f"Visualization failed: {e}")

def test_rag_functionality():
    """Test RAG functionality specifically."""
    print("\n" + "="*60)
    print("TESTING RAG FUNCTIONALITY")
    print("="*60)
    
    rag_test_cases = [
        "How do I fix TechOffice Suite installation error 1603?",
        "I need help with both password reset and changing my email address"
    ]
    
    for i, message in enumerate(rag_test_cases, 1):
        print(f"\nRAG TEST {i}: {message}")
        print("-" * 50)
        
        initial_state = create_initial_state(message, f"rag_session_{i}")
        
        try:
            result = graph.invoke(initial_state)
            
            print(f"Query Type: {result.get('query_type')}")
            print(f"Subquestions: {result.get('subquestions', [])}")
            print(f"Documents Found: {len(result.get('search_results', []))}")
            print(f"Resolved: {result.get('resolved', False)}")
            print(f"Escalation: {result.get('escalation_required', False)}")
            
            if result.get('final_response'):
                print(f"\nResponse Preview: {result.get('final_response')}")
            
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    # Test basic workflow
    # test_workflow()
    
    # Test RAG functionality specifically
    test_rag_functionality()
    
    # Visualize the graph
    visualize()