"""
graph/nodes.py

Simplified node functions for the 3-Agent Customer Support System workflow.
Updated with refund node (LLM with tools) and refund tools node (tool executor).
"""

import json
import re
from typing import Dict, Any

from states import SupportState
from settings import ROUTER_SYSTEM_PROMPT, REFUND_SYSTEM_PROMPT
from tools import get_refund_tools
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
        
        # Create tool node for executing tools
        self.refund_tool_node = ToolNode(self.refund_tools)
    
    def router_agent(self, state: SupportState) -> dict:
        """
        Router Agent: Classify customer query into refund/issue/faq categories.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with classification and routing information
        """
        print("\n[Node: Router Agent]")
        user_message = state.get("user_message", "")
        
        # Create the full prompt with system instructions and user message
        prompt = f"""{ROUTER_SYSTEM_PROMPT}

User Message: "{user_message}"

Analyze this message and classify it."""

        try:
            # Get classification from router LLM
            response = self.router_llm.invoke(prompt)
            classification_result = response.content
            
            print(f"Router LLM raw response: {classification_result}")
            
            # Parse JSON response
            try:
                parsed_result = json.loads(classification_result)
                
                # Update state with classification results
                state["query_type"] = parsed_result.get("query_type", "faq")
                state["classification"] = parsed_result.get("classification", "")
                state["customer_info"] = parsed_result.get("customer_info", {})
                
            except json.JSONDecodeError:
                print("Failed to parse JSON response, using fallback classification")
                # Fallback classification based on keywords
                message_lower = user_message.lower()
                
                if any(word in message_lower for word in ["refund", "money back", "return", "cancel"]):
                    state["query_type"] = "refund"
                elif any(word in message_lower for word in ["not working", "broken", "error", "bug", "problem", "wrong", "missing"]):
                    state["query_type"] = "issue"
                else:
                    state["query_type"] = "faq"
                
                state["classification"] = f"Fallback classification for {state['query_type']} query"
                state["customer_info"] = {}
            
            # Update workflow state
            state["current_agent"] = "router"
            state["actions_taken"].append("router_classification")
            
            print(f"Query classified as: {state['query_type']}")
            print(f"Classification: {state['classification']}")
            
        except Exception as e:
            print(f"Error in router agent: {e}")
            # Set fallback values
            state["query_type"] = "faq"
            state["classification"] = f"Error in classification: {str(e)}"
            state["customer_info"] = {}
        
        return state
    
    def refund_node(self, state: SupportState) -> SupportState:
        """Process refund requests with tool calls."""
        print(f"\n[Node: Refund Node - LLM Processing]")
        
        customer_info = state.get("customer_info", {})
        order_id = customer_info.get("order_id", "unknown")
        email = customer_info.get("email", "unknown")
        
        print(f"Processing refund request - Order ID: {order_id}, Email: {email}")
        
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
                print(f"[DEBUG] About to call LLM with tools for {action_needed.lower()}...")
                response = self.refund_llm_with_tools.invoke(prompt)
            else:
                print(f"[DEBUG] About to call LLM for final customer answer...")
                response = self.refund_llm.invoke(prompt)
            
            print(f"Refund LLM response: {response}")
            
            tool_calls_present = hasattr(response, 'tool_calls') and response.tool_calls
            print(f"Tool calls present: {tool_calls_present}")
            
            if tool_calls_present:
                print(f"[DEBUG] Tool calls in LLM response: {response.tool_calls}")
                state["last_llm_response"] = response
            else:
                state["last_llm_response"] = None
                state["final_response"] = response.content
                state["resolved"] = True
                state["resolution_summary"] = f"Refund request processed: {action_needed}"
            
            print(f"Refund node completed - Tool calls to execute: {len(response.tool_calls) if tool_calls_present else 0}")
            
        except Exception as e:
            print(f"Error in refund node: {e}")
            state["error"] = f"Error processing refund: {str(e)}"
        
        return state

    def execute_refund_tools(self, state: SupportState) -> SupportState:
        """Execute refund verification and processing tools."""
        print(f"\n[Node: Refund Tools - Tool Execution]")
        
        last_llm_response = state.get("last_llm_response")
        
        if not last_llm_response or not hasattr(last_llm_response, "tool_calls") or not last_llm_response.tool_calls:
            print("No tool calls to execute")
            return state
        
        print(f"Executing tool calls: {last_llm_response.tool_calls}")
        
        try:
            verification_result = state.get("verification_result")
            processing_result = state.get("processing_result")
            
            allowed_tool = None
            if not verification_result:
                allowed_tool = "refund_verification_tool"
            elif verification_result and verification_result.get("verified") and not processing_result:
                allowed_tool = "refund_processing_tool"
            
            if not allowed_tool:
                print("No tool execution allowed at this stage")
                return state
            
            filtered_tool_calls = [tc for tc in last_llm_response.tool_calls if tc["name"] == allowed_tool]
            
            if not filtered_tool_calls:
                print(f"No allowed tool call found. Expected: {allowed_tool}")
                return state
            
            from langchain_core.messages import AIMessage
            
            allowed_tool_call = filtered_tool_calls[0]
            
            filtered_ai_message = AIMessage(
                content="",
                tool_calls=[allowed_tool_call]
            )
            
            tool_state = {"messages": [filtered_ai_message]}
            tool_response = self.refund_tool_node.invoke(tool_state)
            
            print(f"Tool node response: {tool_response}")
            
            for msg in tool_response["messages"]:
                if hasattr(msg, "name"):
                    if msg.name == "refund_verification_tool":
                        state["verification_result"] = json.loads(msg.content)
                        print(f"Verification result stored: {state['verification_result']}")
                    elif msg.name == "refund_processing_tool":
                        state["processing_result"] = json.loads(msg.content)
                        print(f"Processing result stored: {state['processing_result']}")
            
            state["tool_results"] = tool_response
            state["last_tool_result"] = tool_response["messages"][-1] if tool_response["messages"] else None
            state["actions_taken"].append("refund_tools_execution")
            state["last_llm_response"] = None
            
        except Exception as e:
            print(f"Error executing tools: {e}")
            state["tool_execution_error"] = str(e)
        
        return state
        
    
    def _extract_order_id(self, message: str, customer_info: dict) -> str:
        """
        Extract order ID from message or customer info.
        
        Args:
            message: User message
            customer_info: Extracted customer information
            
        Returns:
            Order ID if found, empty string otherwise
        """
        # Check customer_info first
        if customer_info.get("order_id"):
            return customer_info["order_id"]
        
        # Look for order ID patterns in message
        order_patterns = [
            r'order\s*#?\s*(\d+)',
            r'order\s*id\s*:?\s*(\d+)',
            r'#(\d+)',
            r'\b(\d{4,6})\b'  # 4-6 digit numbers
        ]
        
        message_lower = message.lower()
        for pattern in order_patterns:
            match = re.search(pattern, message_lower)
            if match:
                return match.group(1)
        
        return ""