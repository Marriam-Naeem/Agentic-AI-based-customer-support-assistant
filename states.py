from typing import TypedDict, Optional, List, Dict, Any
from enum import Enum

class QueryType(str, Enum):
    REFUND = "refund"
    ISSUE = "issue" 
    FAQ = "faq"

class SupportState(TypedDict):
    # Core fields
    user_message: str
    query_type: str
    customer_info: Dict[str, Any]
    resolved: bool
    final_response: Optional[str]
    
    # Tool execution
    last_llm_response: Optional[Any]
    tool_execution_error: Optional[str]
    
    # Refund workflow
    verification_result: Optional[Any]
    processing_result: Optional[Any]
    
    # RAG workflow
    search_results: List[Dict[str, Any]]
    subquestions: List[str]
    
    # Control
    escalation_required: bool

def create_initial_state(user_message: str, session_id: str) -> SupportState:
    return SupportState(
        user_message=user_message,
        query_type="",
        customer_info={},
        resolved=False,
        final_response=None,
        last_llm_response=None,
        tool_execution_error=None,
        verification_result=None,
        processing_result=None,
        search_results=[],
        subquestions=[],
        escalation_required=False
    )