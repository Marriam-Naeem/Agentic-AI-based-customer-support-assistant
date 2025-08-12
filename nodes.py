import json
import re
import time
import random
from typing import Dict, Any
from smolagents import ToolCallingAgent, CodeAgent, tool
# from smolagents.utils import populate_template
# from smolagents import PromptTemplates, PlanningPromptTemplate, ManagedAgentPromptTemplate, FinalAnswerPromptTemplate
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


refund_tools_list = get_refund_tools()
document_tools_list = get_document_search_tools()

# def populate_template(template: str, variables: dict) -> str:
#     """Replace {placeholders} in the template with provided variables."""
#     if not template:
#         return template
#     for key, value in variables.items():
#         template = template.replace("{" + key + "}", str(value))
#     return template

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
        max_steps=3,  # Reduced to prevent duplication
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
        "formatter_agent": formatter_agent
    }

def process_with_smolagents(smolagents_system, user_message):
    """Process user message with autonomous SmolAgents multi-agent system"""
    try:

        manager_agent = smolagents_system["manager_agent"]
        
        prompt = MANAGER_AGENT_PROMPT_TEMPLATE.format(user_message=user_message)
      
        result = manager_agent.run(prompt)

        print(f"[DEBUG] Final email_content: {result[:500]}")   
        
        return result
            
    except Exception as e:
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in RATE_LIMIT_KEYWORDS):
            return RATE_LIMIT_RESPONSE
        return FALLBACK_RESPONSE

def format_with_smolagents(smolagents_system, user_message):
    """Format the response from the manager agent into a professional email response with proper start and end."""
    try:
        prompt = FORMATTER_AGENT_PROMPT_TEMPLATE.format(user_message=user_message)
        return smolagents_system["formatter_agent"].run(prompt)
    except Exception as e:
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in RATE_LIMIT_KEYWORDS):
            return RATE_LIMIT_RESPONSE
        # If formatting fails, return the original message with basic formatting
        return f"Dear Customer,\n\n{user_message}\n\nBest regards,\nTechCorps Support Team"

class NodeFunctions:
    
    def __init__(self, models: Dict[str, Any]):
        self.router_llm = models.get("router_llm")
        self.refund_llm = models.get("refund_llm")
        self.issue_faq_llm = models.get("issue_faq_llm")
        
        # Initialize SmolAgents multi-agent system
        self.smolagents_system = create_smolagents_system(models)
    
    def router_agent(self, state: SupportState) -> dict:
        """Use autonomous SmolAgents multi-agent system for intelligent processing"""
        user_message = state.get("user_message", "")
        conversation_history = state.get("conversation_history", [])
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            context = "Previous conversation:\n"
            for user_msg, bot_response in conversation_history[-3:]:  # Last 3 exchanges
                context += f"User: {user_msg}\nBot: {bot_response}\n\n"
            context += f"Current message: {user_message}\n\n"
        else:
            context = f"Current message: {user_message}\n\n"
        
        try:
            # Use the SmolAgents manager agent to autonomously process the message with context
            response_content = process_with_smolagents(self.smolagents_system, context)
            
            # Update state with autonomous SmolAgents results
            state.update({
                "final_response": response_content
            })
            
        except Exception as e:
            # Fallback response
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in RATE_LIMIT_KEYWORDS):
                final_response = RATE_LIMIT_RESPONSE
            else:
                final_response = FALLBACK_RESPONSE
            
            state.update({
                "final_response": final_response
            })
        
        return state
    
    def formatter_agent(self, state: SupportState) -> dict:
        """Formatter agent that formats the response from the manager agent into a professional email response with proper start and end."""
        final_response = state.get("final_response", "")
        user_message = state.get("user_message", "")
        
        try:
            formatted_email = format_with_smolagents(self.smolagents_system, final_response)
            print("DEBUG: FINAL GENERATED EMAIL: ", formatted_email)
            state.update({
                "final_email": formatted_email
            })
        except Exception as e:
            # If formatting fails, use the original response with basic formatting
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in RATE_LIMIT_KEYWORDS):
                formatted_email = RATE_LIMIT_RESPONSE
            else:
                formatted_email = f"Dear Customer,\n\n{final_response}\n\nBest regards,\nTechCorps Support Team"
            
            state.update({
                "final_email": formatted_email
            })
        
        # Update conversation history
        conversation_history = state.get("conversation_history", [])
        conversation_history.append([user_message, state.get("final_email", final_response)])
        state.update({
            "conversation_history": conversation_history
        })
        
        return state
