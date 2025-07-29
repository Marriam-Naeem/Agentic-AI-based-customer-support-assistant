"""
graph/states.py

Simplified state definitions for the 3-Agent Customer Support System.
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
    """Simplified state management for the 3-Agent Customer Support System."""
    
    user_message: str
    query_type: str
    classification: str
    customer_info: Dict[str, Any]
    retrieved_documents: List[Dict[str, Any]]
    refund_result: Optional[Dict[str, Any]]
    last_llm_response: Optional[Any]
    tool_results: Optional[Any]
    last_tool_result: Optional[Any]
    tool_execution_error: Optional[str]
    verification_result: Optional[Any]
    processing_result: Optional[Any]
    current_agent: str
    actions_taken: List[str]
    resolved: bool
    resolution_summary: str
    email_sent: bool
    escalation_required: bool
    final_response: Optional[str]
    error: Optional[str]

def create_initial_state(user_message: str, session_id: str) -> SupportState:
    """Create an initial state for a new customer support session."""
    return SupportState(
        user_message=user_message,
        query_type="",
        classification="",
        customer_info={},
        retrieved_documents=[],
        refund_result=None,
        last_llm_response=None,
        tool_results=None,
        last_tool_result=None,
        tool_execution_error=None,
        verification_result=None,
        processing_result=None,
        current_agent="router",
        actions_taken=[],
        resolved=False,
        resolution_summary="",
        email_sent=False,
        escalation_required=False,
        final_response=None,
        error=None
    )