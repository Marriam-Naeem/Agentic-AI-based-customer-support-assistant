"""
graph/nodes.py

Updated node functions for the 3-Agent Customer Support System workflow.
Now includes refund nodes and RAG-based non-refund processing.
"""

import json
from typing import Dict, Any, List

from states import SupportState
from settings import ROUTER_SYSTEM_PROMPT, REFUND_SYSTEM_PROMPT, NON_REFUND_SYSTEM_PROMPT
from refund_tools import get_refund_tools
from rag_tools import get_document_search_tools, document_search_tool
from langgraph.prebuilt import ToolNode


class NodeFunctions:
    """Collection of node functions used in the 3-agent workflow graph."""
    
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize with required models.
        
        Args:
            models: Dictionary containing all required models from llm_setup.py
        """
        self.router_llm = models.get("router_llm")
        self.refund_llm = models.get("refund_llm")
        self.issue_faq_llm = models.get("issue_faq_llm")
        self.llm_manager = models.get("llm_manager")
        
        # Get refund tools and bind to LLM
        self.refund_tools = get_refund_tools()
        self.refund_llm_with_tools = self.refund_llm.bind_tools(self.refund_tools)
        
        # Get document search tools and bind to LLM
        self.document_search_tools = get_document_search_tools()
        self.issue_faq_llm_with_tools = self.issue_faq_llm.bind_tools(self.document_search_tools)
        
        # Create tool nodes for executing tools
        self.refund_tool_node = ToolNode(self.refund_tools)
        self.document_search_tool_node = ToolNode(self.document_search_tools)
    
    def router_agent(self, state: SupportState) -> dict:
        """
        Router Agent: Classify customer query into refund/issue/faq categories.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with classification and routing information
        """
        user_message = state.get("user_message", "")
        
        # Create the full prompt with system instructions and user message
        prompt = f"""{ROUTER_SYSTEM_PROMPT}

User Message: "{user_message}"

Please classify this customer query and respond with the JSON format specified above."""

        try:
            response = self.router_llm.invoke(prompt)
            
            # Parse the JSON response
            try:
                result = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback to default values if JSON parsing fails
                result = {
                    "query_type": "faq",
                    "classification": "General inquiry",
                    "customer_info": {}
                }
            
            # Update state with classification results
            state.update({
                "query_type": result.get("query_type", "faq"),
                "customer_info": result.get("customer_info", {})
            })
            
        except Exception as e:
            state.update({
                "query_type": "faq"  # Default fallback
            })
        
        return state
    
    def refund_node(self, state: SupportState) -> SupportState:
        """Process refund requests with tool calls."""
        
        customer_info = state.get("customer_info", {})
        order_id = customer_info.get("order_id", "unknown")
        email = customer_info.get("email", "unknown")
        
        conversation_context = f"Customer Message: {state.get('user_message', '')}\n"
        conversation_context += f"Order ID: {order_id}\n"
        conversation_context += f"Customer Email: {email}\n"
        
        verification_result = state.get("verification_result")
        processing_result = state.get("processing_result")
        
        if verification_result:
            conversation_context += f"\nVerification Result: {json.dumps(verification_result, indent=2)}\n"
        if processing_result:
            conversation_context += f"\nProcessing Result: {json.dumps(processing_result, indent=2)}\n"
        
        if not verification_result:
            action_needed = "VERIFICATION"
            tool_to_call = "refund_verification_tool"
            prompt_context = "You need to verify the refund request. Call the refund_verification_tool."
        elif verification_result and verification_result.get("verified") and not processing_result:
            action_needed = "PROCESSING"
            tool_to_call = "refund_processing_tool"
            prompt_context = "Verification is complete and successful. Now call the refund_processing_tool to process the refund."
        elif verification_result and not verification_result.get("verified"):
            action_needed = "FINAL_ANSWER"
            tool_to_call = None
            prompt_context = "Verification failed. Generate a final customer response explaining why the refund cannot be processed."
        else:
            action_needed = "FINAL_ANSWER"
            tool_to_call = None
            prompt_context = "Both verification and processing are complete. Generate a final customer response with the refund details."
        
        prompt = f"""{REFUND_SYSTEM_PROMPT}

CURRENT STATE:
{conversation_context}

ACTION REQUIRED: {action_needed}
{prompt_context}

CRITICAL RULES:
- If action is VERIFICATION: Call ONLY refund_verification_tool
- If action is PROCESSING: Call ONLY refund_processing_tool  
- If action is FINAL_ANSWER: Generate a customer response, NO tool calls
- NEVER call multiple tools at once
- NEVER call a tool that has already been executed
- If verification_result exists, DO NOT call verification tool again
- If processing_result exists, DO NOT call processing tool again

AVAILABLE TOOLS:
- refund_verification_tool: Verify customer and order eligibility
- refund_processing_tool: Process the actual refund

RESPONSE FORMAT:
- For tool calls: Use the exact tool name and provide required arguments
- For final answers: Provide a clear, helpful response to the customer

Customer Query: {state.get('user_message', '')}"""

        try:
            if action_needed in ["VERIFICATION", "PROCESSING"]:
                response = self.refund_llm_with_tools.invoke(prompt)
            else:
                response = self.refund_llm.invoke(prompt)
            
            tool_calls_present = hasattr(response, 'tool_calls') and response.tool_calls
            
            if tool_calls_present:
                state["last_llm_response"] = response
            else:
                state["last_llm_response"] = None
                state["final_response"] = response.content
                state["resolved"] = True
            
        except Exception as e:
            state["tool_execution_error"] = f"Error processing refund: {str(e)}"
        
        return state

    def execute_refund_tools(self, state: SupportState) -> SupportState:
        """Execute refund verification and processing tools."""
        last_llm_response = state.get("last_llm_response")
        
        if not last_llm_response or not hasattr(last_llm_response, "tool_calls") or not last_llm_response.tool_calls:
            return state
        
        try:
            verification_result = state.get("verification_result")
            processing_result = state.get("processing_result")
            
            allowed_tool = None
            if not verification_result:
                allowed_tool = "refund_verification_tool"
            elif verification_result and verification_result.get("verified") and not processing_result:
                allowed_tool = "refund_processing_tool"
            
            if not allowed_tool:
                return state
            
            filtered_tool_calls = [tc for tc in last_llm_response.tool_calls if tc["name"] == allowed_tool]
            
            if not filtered_tool_calls:
                return state
            
            from langchain_core.messages import AIMessage
            
            allowed_tool_call = filtered_tool_calls[0]
            
            filtered_ai_message = AIMessage(
                content="",
                tool_calls=[allowed_tool_call]
            )
            
            tool_state = {"messages": [filtered_ai_message]}
            tool_response = self.refund_tool_node.invoke(tool_state)
            
            for msg in tool_response["messages"]:
                if hasattr(msg, "name"):
                    if msg.name == "refund_verification_tool":
                        state["verification_result"] = json.loads(msg.content)
                    elif msg.name == "refund_processing_tool":
                        state["processing_result"] = json.loads(msg.content)
            
            state["last_llm_response"] = None
            
        except Exception as e:
            state["tool_execution_error"] = str(e)
        
        return state
    
    def non_refund_node(self, state: SupportState) -> SupportState:
        """
        Process non-refund queries (issues and FAQ) using RAG system.
        Breaks down complex queries into subquestions and searches for solutions.
        """
        user_message = state.get("user_message", "")
        query_type = state.get("query_type", "faq")
        
        # Check if we already have search results and need to generate final answer
        search_results = state.get("search_results", [])
        subquestions = state.get("subquestions", [])
        
        if search_results and not state.get("final_response"):
            # We have search results, generate final answer
            prompt = self._create_final_answer_prompt(user_message, query_type, search_results, subquestions)
            
            try:
                response = self.issue_faq_llm.invoke(prompt)
                state["final_response"] = response.content
                state["resolved"] = True
                return state
                
            except Exception as e:
                state["tool_execution_error"] = f"Error generating final answer: {str(e)}"
                return state
        
        # First time processing - break down query and search
        prompt = self._create_subquestion_prompt(user_message, query_type)
        
        try:
            response = self.issue_faq_llm_with_tools.invoke(prompt)
            
            tool_calls_present = hasattr(response, 'tool_calls') and response.tool_calls
            
            if tool_calls_present:
                state["last_llm_response"] = response
            else:
                # No tool calls - LLM provided direct answer or escalation
                if "escalate" in response.content.lower() or "human" in response.content.lower():
                    state["escalation_required"] = True
                    state["final_response"] = response.content
                else:
                    state["final_response"] = response.content
                    state["resolved"] = True
            
        except Exception as e:
            state["tool_execution_error"] = f"Error processing {query_type} query: {str(e)}"
        
        return state
    
    def execute_rag_tools(self, state: SupportState) -> SupportState:
        """Execute document search tools for RAG processing."""
        last_llm_response = state.get("last_llm_response")
        
        if not last_llm_response or not hasattr(last_llm_response, "tool_calls") or not last_llm_response.tool_calls:
            return state
        
        try:
            from langchain_core.messages import AIMessage
            
            # Execute all tool calls
            ai_message = AIMessage(
                content="",
                tool_calls=last_llm_response.tool_calls
            )
            
            tool_state = {"messages": [ai_message]}
            tool_response = self.document_search_tool_node.invoke(tool_state)
            
            # Process search results
            search_results = []
            subquestions = []
            
            for msg in tool_response["messages"]:
                if hasattr(msg, "name") and msg.name == "document_search_tool":
                    try:
                        search_data = json.loads(msg.content)
                        if search_data.get("success"):
                            search_results.extend(search_data.get("results", []))
                            # Extract the query as a subquestion
                            subquestions.append(search_data.get("query", ""))
                    except json.JSONDecodeError:
                        pass
            
            # Store results in state
            state["search_results"] = search_results
            state["subquestions"] = subquestions
            state["last_llm_response"] = None
            
            # Check if we found any solutions
            if not search_results:
                state["escalation_required"] = True
                state["final_response"] = "I couldn't find relevant information to answer your question. Let me connect you with a human agent who can help you better."
            
        except Exception as e:
            state["tool_execution_error"] = str(e)
            state["escalation_required"] = True
            state["final_response"] = "I encountered an error while searching for information. Let me connect you with a human agent."
        
        return state
    
    def _create_subquestion_prompt(self, user_message: str, query_type: str) -> str:
        """Create prompt for breaking down query into subquestions and searching."""
        return f"""{NON_REFUND_SYSTEM_PROMPT}

TASK: Analyze the customer query and break it down into specific subquestions, then search for each one.

CUSTOMER QUERY: "{user_message}"
QUERY TYPE: {query_type}

INSTRUCTIONS:
1. Analyze the customer query carefully
2. Break it down into specific searchable subquestions if it contains multiple parts
3. For each subquestion, call the document_search_tool with a clear, specific search query
4. If the query is simple and direct, make one targeted search
5. If no relevant information can be found, recommend escalation to human

SEARCH STRATEGY:
- Use specific, relevant keywords for each search
- Focus on the core problem or question
- Include product names, error codes, or specific terms mentioned
- Search for troubleshooting steps, procedures, or explanations

EXAMPLES:
- For "My TechOffice Suite won't install and shows error 1603": 
  Search for: "TechOffice Suite installation error 1603"
- For "How do I reset my password and change my email?":
  Search 1: "reset password procedure"
  Search 2: "change email address account"

Now analyze the customer query and perform the appropriate searches:"""

    def _create_final_answer_prompt(self, user_message: str, query_type: str, search_results: List[Dict], subquestions: List[str]) -> str:
        """Create prompt for generating final answer based on search results."""
        
        # Format search results for the prompt
        formatted_results = ""
        for i, result in enumerate(search_results, 1):
            formatted_results += f"\nResult {i}:\n"
            formatted_results += f"Content: {result.get('content', '')}\n"
            formatted_results += f"Source: {result.get('source', 'unknown')}\n"
            formatted_results += f"Relevance Score: {result.get('similarity_score', 0):.3f}\n"
            formatted_results += "---"
        
        subqs_text = "\n".join(f"- {sq}" for sq in subquestions)
        
        return f"""{NON_REFUND_SYSTEM_PROMPT}

TASK: Generate a comprehensive, helpful answer for the customer based on the search results.

ORIGINAL CUSTOMER QUERY: "{user_message}"
QUERY TYPE: {query_type}

SUBQUESTIONS ANALYZED:
{subqs_text}

SEARCH RESULTS FOUND:
{formatted_results}

INSTRUCTIONS:
1. Synthesize the search results into a clear, helpful response
2. Address all parts of the customer's original question
3. Provide step-by-step instructions when applicable
4. Include relevant details from the search results
5. Be empathetic and professional
6. If the search results don't fully answer the question, acknowledge limitations
7. If critical information is missing, recommend escalation to human support

RESPONSE REQUIREMENTS:
- Start with acknowledgment of the customer's issue
- Provide clear, actionable steps or information
- Reference specific details from search results when helpful
- End with offer for further assistance
- Keep response concise but comprehensive
- Use friendly, professional tone
- Do not mention what documents were searched or list search results

Generate the final customer response:"""