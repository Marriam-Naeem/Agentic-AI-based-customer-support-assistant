"""
graph/states.py

Updated state definitions for the 3-Agent Customer Support System with RAG support.
All agents in the workflow share and update this state.
"""

from typing import TypedDict, Optional, List, Dict, Any
from enum import Enum

class QueryType(str, Enum):
    """Enum for the 3 main query types"""
    REFUND = "refund"
    ISSUE = "issue" 
    FAQ = "faq"

class SupportState(TypedDict):
    """Updated state management for the 3-Agent Customer Support System with RAG."""
    
    # Core fields
    user_message: str
    query_type: str
    classification: str
    customer_info: Dict[str, Any]
    current_agent: str
    actions_taken: List[str]
    resolved: bool
    final_response: Optional[str]
    error: Optional[str]
    
    # Refund-specific fields
    verification_result: Optional[Any]
    processing_result: Optional[Any]
    last_llm_response: Optional[Any]
    tool_results: Optional[Any]
    last_tool_result: Optional[Any]
    tool_execution_error: Optional[str]
    refund_result: Optional[Dict[str, Any]]
    
    # RAG-specific fields
    search_results: List[Dict[str, Any]]
    subquestions: List[str]
    
    # Workflow control fields
    escalation_required: bool
    resolution_summary: str

def create_initial_state(user_message: str, session_id: str) -> SupportState:
    """Create an initial state for a new customer support session."""
    return SupportState(
        user_message=user_message,
        query_type="",
        classification="",
        customer_info={},
        current_agent="router",
        actions_taken=[],
        resolved=False,
        final_response=None,
        error=None,
        # Refund fields
        verification_result=None,
        processing_result=None,
        last_llm_response=None,
        tool_results=None,
        last_tool_result=None,
        tool_execution_error=None,
        refund_result=None,
        # RAG fields
        search_results=[],
        subquestions=[],
        # Workflow control
        escalation_required=False,
        resolution_summary=""
    )