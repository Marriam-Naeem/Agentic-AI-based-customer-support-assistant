import json
from typing import Dict, Any, List

from states import SupportState
from settings import ROUTER_SYSTEM_PROMPT, REFUND_SYSTEM_PROMPT, NON_REFUND_SYSTEM_PROMPT
from refund_tools import get_refund_tools
from rag_tools import get_document_search_tools
from langgraph.prebuilt import ToolNode


class NodeFunctions:
    """Collection of node functions used in the agent workflow graph."""
    
    def __init__(self, models: Dict[str, Any]):
        self.router_llm = models.get("router_llm")
        self.refund_llm = models.get("refund_llm")
        self.issue_faq_llm = models.get("issue_faq_llm")
        
        self.refund_tools = get_refund_tools()
        self.refund_llm_with_tools = self.refund_llm.bind_tools(self.refund_tools)
        
        self.document_search_tools = get_document_search_tools()
        self.issue_faq_llm_with_tools = self.issue_faq_llm.bind_tools(self.document_search_tools)
        
        self.refund_tool_node = ToolNode(self.refund_tools)
        self.document_search_tool_node = ToolNode(self.document_search_tools)
    
    def router_agent(self, state: SupportState) -> dict:
        user_message = state.get("user_message", "")
        
        # Check if this is a follow-up message in a refund conversation
        # Look for previous refund-related state
        previous_query_type = state.get("query_type", "")
        customer_info = state.get("customer_info", {})
        
        # Smart context detection
        context_info = ""
        is_followup = False
        
        # Check if this looks like a follow-up message
        if (previous_query_type == "refund" or 
            customer_info.get("order_id") or 
            customer_info.get("email") or
            any(keyword in user_message.lower() for keyword in ["order", "email", "refund", "money back", "cancel"])):
            is_followup = True
            context_info = "\n\nCONTEXT: This appears to be a follow-up message in a refund conversation. Previous information indicates this is a refund request."
        
        # Enhanced prompt with context awareness
        prompt = f"{ROUTER_SYSTEM_PROMPT}\n\nCurrent User Message: \"{user_message}\"{context_info}\n\nIMPORTANT CLASSIFICATION RULES:\n1. If this message contains order numbers, emails, or appears to be continuing a refund conversation, classify it as 'refund'\n2. If this message is providing additional information for a refund request, classify it as 'refund'\n3. If this message is a standalone question about policies or general information, classify it as 'faq'\n4. If this message describes technical problems or errors, classify it as 'issue'\n\nPlease classify this customer query and respond with the JSON format specified above."
        
        print("\n" + "="*60)
        print("🔄 ROUTER LLM PROCESSING")
        print("="*60)
        print(f"📝 User Message: {user_message}")
        print(f"🤖 Router Model: {type(self.router_llm).__name__}")
        print(f"🔍 Previous Query Type: {previous_query_type}")
        print(f"🔍 Customer Info: {customer_info}")
        print(f"🔍 Is Follow-up: {is_followup}")
        
        try:
            # # Handle OllamaLLM vs LangChain models
            # if hasattr(self.router_llm, 'generate'):
            #     # OllamaLLM format
            #     response = self.router_llm.generate(prompt)
            #     response_content = response
            # else:
            #     # LangChain format (ChatGroq)
            #     response = self.router_llm.invoke(prompt)
            #     if hasattr(response, 'content'):
            #         response_content = response.content
            #     else:
            #         # ChatGroq sometimes returns string directly
            #         response_content = str(response)
            response = self.router_llm.invoke(prompt)
            # Handle both string responses (Ollama) and object responses (ChatGroq)
            if hasattr(response, 'content'):
                response_content = response.content
            else:
                response_content = str(response)
            
            print(f"Router Response: {response_content}")
            
            try:
                # Try to parse the response directly
                result = json.loads(response_content)
                print(f"📊 Parsed Result: {json.dumps(result, indent=2)}")
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    try:
                        extracted_json = json_match.group(0)
                        result = json.loads(extracted_json)
                        print(f"📊 Extracted JSON Result: {json.dumps(result, indent=2)}")
                    except json.JSONDecodeError:
                        result = {"query_type": "faq", "classification": "General inquiry", "customer_info": {}}
                        print(f"⚠️ JSON Extraction Failed - Using default: {result}")
                else:
                    result = {"query_type": "faq", "classification": "General inquiry", "customer_info": {}}
                    print(f"⚠️ No JSON found in response - Using default: {result}")
            
            state.update({
                "query_type": result.get("query_type", "faq"),
                "customer_info": result.get("customer_info", {})
            })
            print(f"🎯 Query Type: {result.get('query_type', 'faq')}")
            print("="*60)
        
        except Exception as e:
            print(f"❌ Router Error: {str(e)}")
            state.update({"query_type": "faq"})
            print("="*60)
        
        return state
    
    def refund_node(self, state: SupportState) -> SupportState:
        customer_info = state.get("customer_info", {})
        order_id = customer_info.get("order_id", "unknown")
        email = customer_info.get("email", "unknown")
        conversation_context = f"Customer Message: {state.get('user_message', '')}\nOrder ID: {order_id}\nCustomer Email: {email}\n"
        verification_result = state.get("verification_result")
        processing_result = state.get("processing_result")
        if verification_result:
            conversation_context += f"\nVerification Result: {json.dumps(verification_result, indent=2)}\n"
        if processing_result:
            conversation_context += f"\nProcessing Result: {json.dumps(processing_result, indent=2)}\n"
        if not verification_result:
            action_needed = "VERIFICATION"
            prompt_context = "You need to verify the refund request. Call the refund_verification_tool."
        elif verification_result and verification_result.get("verified") and not processing_result:
            action_needed = "PROCESSING"
            prompt_context = "Verification is complete and successful. Now call the refund_processing_tool to process the refund."
        elif verification_result and not verification_result.get("verified"):
            action_needed = "FINAL_ANSWER"
            prompt_context = "Verification failed. Generate a final customer response explaining why the refund cannot be processed."
        else:
            action_needed = "FINAL_ANSWER"
            prompt_context = "Both verification and processing are complete. Generate a final customer response with the refund details."
        prompt = f"{REFUND_SYSTEM_PROMPT}\n\nCURRENT STATE:\n{conversation_context}\nACTION REQUIRED: {action_needed}\n{prompt_context}\n\nCRITICAL RULES:\n- If action is VERIFICATION: Call ONLY refund_verification_tool\n- If action is PROCESSING: Call ONLY refund_processing_tool\n- If action is FINAL_ANSWER: Generate a customer response, NO tool calls\n- NEVER call multiple tools at once\n- NEVER call a tool that has already been executed\n- If verification_result exists, DO NOT call verification tool again\n- If processing_result exists, DO NOT call processing tool again\n\nAVAILABLE TOOLS:\n- refund_verification_tool: Verify customer and order eligibility\n- refund_processing_tool: Process the actual refund\n\nRESPONSE FORMAT:\n- For tool calls: Use the exact tool name and provide required arguments\n- For final answers: Provide a clear, helpful response to the customer\n\nCustomer Query: {state.get('user_message', '')}"
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
                # Handle both string responses (Ollama) and object responses (ChatGroq)
                response_content = response.content if hasattr(response, 'content') else str(response)
                cleaned_response = self._clean_response(response_content)
                state["final_response"] = cleaned_response
                state["resolved"] = True
        except Exception as e:
            state["tool_execution_error"] = f"Error processing refund: {str(e)}"
        return state

    def execute_refund_tools(self, state: SupportState) -> SupportState:
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
        user_message = state.get("user_message", "")
        query_type = state.get("query_type", "faq")
        search_results = state.get("search_results", [])
        subquestions = state.get("subquestions", [])
        if search_results and not state.get("final_response"):
            prompt = self._create_final_answer_prompt(user_message, query_type, search_results, subquestions)
            try:
                response = self.issue_faq_llm.invoke(prompt)
                # Handle both string responses (Ollama) and object responses (ChatGroq)
                response_content = response.content if hasattr(response, 'content') else str(response)
                cleaned_response = self._clean_response(response_content)
                state["final_response"] = cleaned_response
                state["resolved"] = True
                return state
            except Exception as e:
                state["tool_execution_error"] = f"Error generating final answer: {str(e)}"
                return state
        prompt = self._create_subquestion_prompt(user_message, query_type)
        try:
            response = self.issue_faq_llm_with_tools.invoke(prompt)
            tool_calls_present = hasattr(response, 'tool_calls') and response.tool_calls
            if tool_calls_present:
                state["last_llm_response"] = response
            else:
                # Handle both string responses (Ollama) and object responses (ChatGroq)
                response_content = response.content if hasattr(response, 'content') else str(response)
                if "escalate" in response_content.lower() or "human" in response_content.lower():
                    state["escalation_required"] = True
                    state["final_response"] = response_content
                else:
                    state["final_response"] = response_content
                    state["resolved"] = True
        except Exception as e:
            state["tool_execution_error"] = f"Error processing {query_type} query: {str(e)}"
        return state
    
    def execute_rag_tools(self, state: SupportState) -> SupportState:
        last_llm_response = state.get("last_llm_response")
        
        if not last_llm_response or not hasattr(last_llm_response, "tool_calls") or not last_llm_response.tool_calls:
            return state
        
        try:
            from langchain_core.messages import AIMessage
            

            ai_message = AIMessage(
                content="",
                tool_calls=last_llm_response.tool_calls
            )
            
            tool_state = {"messages": [ai_message]}
            tool_response = self.document_search_tool_node.invoke(tool_state)
            

            search_results = []
            subquestions = []
            
            for msg in tool_response["messages"]:
                if hasattr(msg, "name") and msg.name == "document_search_tool":
                    try:
                        search_data = json.loads(msg.content)
                        if search_data.get("success"):
                            search_results.extend(search_data.get("results", []))

                            subquestions.append(search_data.get("query", ""))
                    except json.JSONDecodeError:
                        pass
            

            state["search_results"] = search_results
            state["subquestions"] = subquestions
            state["last_llm_response"] = None
            

            if not search_results:
                state["escalation_required"] = True
                state["final_response"] = "I couldn't find relevant information to answer your question. Let me connect you with a human agent who can help you better."
            
        except Exception as e:
            state["tool_execution_error"] = str(e)
            state["escalation_required"] = True
            state["final_response"] = "I encountered an error while searching for information. Let me connect you with a human agent."
        
        return state
    
    def _create_subquestion_prompt(self, user_message: str, query_type: str) -> str:
        return f"{NON_REFUND_SYSTEM_PROMPT}\n\nTASK: Analyze the customer query and break it down into specific subquestions, then search for each one.\nCUSTOMER QUERY: \"{user_message}\"\nQUERY TYPE: {query_type}\nINSTRUCTIONS: 1. Analyze the customer query carefully 2. Break it down into specific searchable subquestions if it contains multiple parts 3. For each subquestion, call the document_search_tool with a clear, specific search query 4. If the query is simple and direct, make one targeted search 5. If no relevant information can be found, recommend escalation to human\nSEARCH STRATEGY: Use specific, relevant keywords for each search. Focus on the core problem or question. Include product names, error codes, or specific terms mentioned. Search for troubleshooting steps, procedures, or explanations.\nEXAMPLES: For 'My TechOffice Suite won't install and shows error 1603': Search for: 'TechOffice Suite installation error 1603'. For 'How do I reset my password and change my email?': Search 1: 'reset password procedure', Search 2: 'change email address account'. Now analyze the customer query and perform the appropriate searches:"

    def _create_final_answer_prompt(self, user_message: str, query_type: str, search_results: List[Dict], subquestions: List[str]) -> str:
        formatted_results = ""
        for i, result in enumerate(search_results, 1):
            formatted_results += f"\nResult {i}:\nContent: {result.get('content', '')}\nSource: {result.get('source', 'unknown')}\nRelevance Score: {result.get('similarity_score', 0):.3f}\n---"
        subqs_text = "\n".join(f"- {sq}" for sq in subquestions)
        return f"{NON_REFUND_SYSTEM_PROMPT}\n\nTASK: Generate a comprehensive, helpful answer for the customer based on the search results.\nORIGINAL CUSTOMER QUERY: \"{user_message}\"\nQUERY TYPE: {query_type}\nSUBQUESTIONS ANALYZED:\n{subqs_text}\nSEARCH RESULTS FOUND:\n{formatted_results}\nINSTRUCTIONS: 1. Synthesize the search results into a clear, helpful response 2. Address all parts of the customer's original question 3. Provide step-by-step instructions when applicable 4. Include relevant details from the search results 5. Be empathetic and professional 6. If the search results don't fully answer the question, acknowledge limitations 7. If critical information is missing, recommend escalation to human support RESPONSE REQUIREMENTS: Start with acknowledgment of the customer's issue. Provide clear, actionable steps or information. Reference specific details from search results when helpful. End with offer for further assistance. Keep response concise but comprehensive. Use friendly, professional tone. Do not mention what documents were searched or list search results. IMPORTANT: Do not include any thinking, reasoning, or internal thoughts in your response. Provide only the direct, helpful response to the customer. Generate the final customer response:"

    def _clean_response(self, response: str) -> str:
        import re

        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        thinking_patterns = [
            r'Okay, let\'s tackle.*?First, I need to.*?',
            r'Let me think about.*?',
            r'I need to analyze.*?',
            r'Looking at this.*?',
            r'Based on my analysis.*?',
            r'Let me break this down.*?'
        ]
        
        for pattern in thinking_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        response = response.strip()
        
        return response