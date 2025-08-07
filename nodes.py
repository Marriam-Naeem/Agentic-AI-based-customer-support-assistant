import json
import re
import time
import random
from typing import Dict, Any, List
from smolagents import ToolCallingAgent, CodeAgent, tool

from states import SupportState
from settings import ROUTER_SYSTEM_PROMPT, REFUND_SYSTEM_PROMPT, NON_REFUND_SYSTEM_PROMPT
from refund_tools import get_refund_tools
from rag_tools import get_document_search_tools

# Get existing tools
refund_tools_list = get_refund_tools()
document_tools_list = get_document_search_tools()

def handle_rate_limit(func):
    """Decorator to handle rate limiting with exponential backoff"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate_limit" in str(e).lower() or "rate limit" in str(e).lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"Rate limit hit, waiting {delay:.1f} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(delay)
                        continue
                raise e
        return func(*args, **kwargs)
    return wrapper

# Create SmolAgents tools from existing tools
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

@handle_rate_limit
def create_smolagents_system(models):
    """Create SmolAgents multi-agent system following the documentation pattern"""
    
    refund_agent = ToolCallingAgent(
        tools=[refund_verification_tool, refund_processing_tool],
        model=models["refund_llm"],
        max_steps=2,
        name="refund_agent",
        description="Agent that verifies refund eligibility and, if eligible, processes the refund and sends a professional email response.",
        instructions="""
You are a refund processing agent that answers users queries. Your workflow is:
1. Call refund_verification_tool with the order_id (and customer_email if available).
2. If the result of refund_verification_tool is true (verified), call refund_processing_tool with the order_id.
3. Generate a professional email answer to the customer based on the results of the tool calls.
4. If any information is missing or there is a placeholder for that, just tell the customer that the information is missing.
Only process the refund if verification is successful. If verification fails, explain the reason to the customer in your email response.
"""
    )
     
    support_agent = CodeAgent(
        tools=[document_search_tool],
        model=models["issue_faq_llm"],
        max_steps=10,
        verbosity_level=2,
        name="support_agent",
        description="Support agent that answers user queries by searching company documents, provides step-by-step answers, and never reveals the document source to the user.",
        instructions="""
You are a support agent that answers user queries by searching company document but never tell the user from which document the answer was taken.

1. Use document_search_tool to find relevant information for the user's question.
2. Extract the most relevant content from the search results.
3. Generate a professional email response that contains a steps by step answers to the customer's question using the information found.
4. If no relevant information is found or if any information is missing, politely inform the customer that the information is not available or missing.
6. Always keep your response clear, concise, professional and strictly relevent to the customer's question.
"""
    )
    
    # Create manager agent that orchestrates the agents
    manager_agent = ToolCallingAgent(
        tools=[],
        model=models["router_llm"],
        managed_agents=[refund_agent, support_agent],
        name="manager_agent",
        description="Manager agent that autonomously decides which specialized agent to call based on customer request analysis."
    )

    formatter_agent = CodeAgent(
        tools=[],
        model=models["router_llm"],
        name="formatter_agent",
        description="Formatter agent that formats the response from the manager agent into a professional email response with proper start and end."
    )
    
    return {
        "manager_agent": manager_agent,
        "refund_agent": refund_agent,
        "support_agent": support_agent,
        "formatter_agent":formatter_agent
    }

   

@handle_rate_limit
def process_with_smolagents(smolagents_system, user_message):
    """Process user message with autonomous SmolAgents multi-agent system"""
    try:
        # Use the manager agent to autonomously decide which agent to call
        manager_agent = smolagents_system["manager_agent"]
        
        prompt = f"""
        Customer Request: {user_message}
        
        You are a manager agent responsible for autonomously deciding which specialized agent to call based on the customer's request.
        
        You have access to two managed agents:
        1. refund_agent: verifies refund eligibility and, if eligible, processes the refund and sends a professional email response.
        2. support_agent: Support agent that answers user queries by searching company documents, provides step-by-step answers, and never reveals the document source to the user.
        
        Analyze the customer's request and autonomously decide which agent is best suited to handle their inquiry.
        Consider the nature of the request, the customer's needs, and the capabilities of each agent.
        
        IMPORTANT: 
        - Make an intelligent, autonomous decision about which agent to call
        - Do not use simple keyword matching - analyze the actual intent and context
        - The chosen agent will format the response as a professional email
        - Return ONLY the final response from the chosen agent, not your reasoning
        - Do not add any additional text or explanations
        
        MANAGED AGENT CAPABILITIES:
        - refund_agent: verifies refund eligibility and, if eligible, processes the refund and sends a professional email response.
        - support_agent: Support agent that answers user queries by searching company documents, provides step-by-step answers, and never reveals the document source to the user.
        
        Execute the appropriate managed agent to handle this customer request and return their response directly.
        """
        
        # Run the manager agent which will autonomously decide and call the appropriate agent
        result = manager_agent.run(prompt)

        print(f"[DEBUG] Final email_content: {result[:500]}")   
        
        # # Extract the response - handle different response types
        # response_text = ""
        # if hasattr(result, 'content'):
        #     response_text = result.content
        # elif hasattr(result, 'message'):
        #     response_text = result.message
        # elif hasattr(result, 'text'):
        #     response_text = result.text
        # elif isinstance(result, str):
        #     response_text = result
        # else:
        #     response_text = str(result)
        
        # # --- Post-processing for customer-facing email ---
        # # Remove 'Task outcome' headers and meta-comments, extract main answer
        # import re
        
        # # Remove 'Task outcome' headers
        # response_text = re.sub(r"#+ ?[0-9]?\.?( )?Task outcome.*?:", "", response_text, flags=re.IGNORECASE)
        # response_text = re.sub(r"#+ ?[0-9]?\.?( )?Additional context.*?:", "", response_text, flags=re.IGNORECASE)
        # response_text = re.sub(r"Final answer:.*", "", response_text, flags=re.IGNORECASE)
        # response_text = re.sub(r"Here is the final answer from your managed agent.*?:", "", response_text, flags=re.IGNORECASE)
        
        # # Remove meta-comments about using the document search tool again
        # response_text = re.sub(r"I can use the document search tool again[^]*", "", response_text, flags=re.IGNORECASE)
        # response_text = re.sub(r"It would be helpful to investigate[^]*", "", response_text, flags=re.IGNORECASE)
        
        # # Remove any excessive blank lines
        # response_text = re.sub(r"\n{3,}", "\n\n", response_text)
        # response_text = response_text.strip()
        
        # Format as a proper email
        
        return result
            
    except Exception as e:
        return f"Dear Customer,\n\nI apologize for the technical difficulty. Please try rephrasing your request or contact our support team for immediate assistance.\n\nBest regards,\nTechCorps Support Agent"


def format_with_smolagents(smolagents_system, user_message):
    """Format the response from the manager agent into a professional email response with proper start and end."""
    prompt = f"""You are a professional email formatting specialist. Your role is to format the following given raw responses from support agents into polished, professional customer service emails.

FORMATTING GUIDELINES:
1. Structure the email with proper greeting, body, and closing
2. Ensure professional tone and clear communication
3. Remove any technical jargon or internal references
4. Clean up formatting issues and improve readability
5. Maintain the core message while enhancing presentation
6. Add appropriate context when needed
7. Ensure empathy and customer-centric language

EMAIL STRUCTURE:
- Greeting: "Dear [Customer Name]," or "Dear Customer," if name not provided
- Body: Well-formatted content with proper paragraphs and spacing
- Closing: "Best regards,\nTechCorps Support Team"

FORMATTING RULES:
- Remove debug text, tool references, or internal process mentions
- Fix grammar, punctuation, and spacing issues
- Break up long paragraphs for better readability
- Use bullet points or numbered lists for step-by-step instructions
- Ensure proper capitalization and professional language
- Remove redundant information
- Add transitional phrases for better flow

TONE REQUIREMENTS:
- Professional yet friendly
- Empathetic to customer concerns
- Clear and concise
- Helpful and solution-oriented
- Apologetic when appropriate

Raw Response: {user_message}
"""
    return smolagents_system["formatter_agent"].run(prompt)

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
        
        try:
            # Use the SmolAgents manager agent to autonomously process the message
            response_content = process_with_smolagents(self.smolagents_system, user_message)
            
            # Extract customer info using regex (this is still useful for state tracking)
            customer_info = {}
            if "order" in user_message.lower():
                order_match = re.search(r'order[:\s#]*([A-Z0-9-]+)', user_message, re.IGNORECASE)
                if order_match:
                    customer_info["order_id"] = order_match.group(1)
            
            if "@" in user_message:
                email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_message)
                if email_match:
                    customer_info["email"] = email_match.group(0)
            
            # Update state with autonomous SmolAgents results
            state.update({
                "query_type": "autonomous",  # Let the manager agent decide
                "customer_info": customer_info,
                "autonomous_decision": f"Manager agent autonomously orchestrated multi-agent system",
                "final_response": response_content,
                "resolved": True
            })
            
            # Record autonomous agent decisions
            state["agent_decisions"].append({
                "agent": "Manager_Agent",
                "decision": "autonomous_orchestration",
                "reasoning": "Intelligent analysis and autonomous agent selection",
                "timestamp": "now"
            })
            
        except Exception as e:
            # Fallback response
            state.update({
                "query_type": "autonomous",
                "customer_info": {},
                "autonomous_decision": f"Manager agent error: {str(e)}",
                "final_response": f"Dear Customer, I apologize for the technical difficulty. Please try rephrasing your request or contact our support team for immediate assistance. Best regards, TechCorps Support Agent"
            })
        
        return state
    
    def formatter_agent(self, state: SupportState) -> dict:
        """Formatter agent that formats the response from the manager agent into a professional email response with proper start and end."""
        final_response = state.get("final_response", "")
        formatted_email = format_with_smolagents(self.smolagents_system, final_response)
        print("DEBUG: FINAL GENERATED EMAIL: ",formatted_email)
        state.update({
            "final_email": formatted_email
        })
        return state
    
    def refund_node(self, state: SupportState) -> SupportState:
        """SmolAgents handles refund processing in the router_agent"""
        return state
    
    def non_refund_node(self, state: SupportState) -> SupportState:
        """SmolAgents handles support processing in the router_agent"""
        return state
    
    def execute_refund_tools(self, state: SupportState) -> SupportState:
        """SmolAgents handles tool execution autonomously"""
        return state
    
    def execute_rag_tools(self, state: SupportState) -> SupportState:
        """SmolAgents handles RAG tool execution autonomously"""
        return state