from typing import TypedDict, Optional, List

class SupportState(TypedDict):
    user_message: str
    final_response: Optional[str]
    final_email: str
    session_id: str
    conversation_history: List[List[str]]  # [[user_msg, bot_response], ...]

def create_initial_state(user_message: str, session_id: str, conversation_history: List[List[str]] = None) -> SupportState:
    if conversation_history is None:
        conversation_history = []
    
    return SupportState(
        user_message=user_message,
        final_response=None,
        final_email=None,
        session_id=session_id,
        conversation_history=conversation_history
    )