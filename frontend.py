"""
frontend.py

Gradio frontend for the Customer Assistant chatbot.
Provides a user-friendly chat interface that integrates with the LangGraph workflow.
"""

import gradio as gr
import sys
import os
from typing import List, Tuple
import uuid
from datetime import datetime

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import graph
from states import create_initial_state


class ChatbotInterface:
    """Gradio chatbot interface for the customer assistant."""
    
    def __init__(self):
        self.chat_history = []
        self.session_id = str(uuid.uuid4())
    
    def process_message(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """
        Process a user message through the workflow and return the response.
        
        Args:
            message: User's input message
            history: Chat history from Gradio
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not message.strip():
            return "", history
        
        try:
            # Create initial state for the workflow
            initial_state = create_initial_state(message, self.session_id)
            
            # Process through the workflow
            result = graph.invoke(initial_state)
            
            # Extract the response
            response = self._extract_response(result)
            
            # Update chat history
            history.append([message, response])
            
            return "", history
            
        except Exception as e:
            error_response = f"Sorry, I encountered an error while processing your request: {str(e)}"
            history.append([message, error_response])
            return "", history
    
    def _extract_response(self, result: dict) -> str:
        """
        Extract the appropriate response from the workflow result.
        
        Args:
            result: The result from the workflow
            
        Returns:
            Formatted response string
        """
        # Check for errors first
        if result.get("error"):
            return f"❌ Error: {result['error']}"
        
        # Check for escalation
        if result.get("escalation_required"):
            escalation_reason = result.get("escalation_reason", "Complex issue detected")
            return f"🔄 **Escalation Required**\n\n{escalation_reason}\n\nYour request has been escalated to a human agent who will contact you soon."
        
        # Check for final response
        if result.get("final_response"):
            return result["final_response"]
        
        # Check for refund processing results
        if result.get("processing_result"):
            return f"✅ **Refund Processed Successfully**\n\n{result['processing_result'].get('message', 'Your refund has been processed.')}"
        
        # Check for verification results
        if result.get("verification_result"):
            verification = result["verification_result"]
            if verification.get("verified"):
                return f"✅ **Order Verified**\n\n{verification.get('message', 'Your order has been verified and is being processed.')}"
            else:
                return f"❌ **Verification Failed**\n\n{verification.get('message', 'We could not verify your order. Please check your order number and email.')}"
        
        # Check for search results (RAG responses)
        if result.get("search_results"):
            search_results = result["search_results"]
            if search_results:
                # Format search results as a response
                response_parts = ["📚 **Information Found:**\n"]
                for i, doc in enumerate(search_results[:3], 1):  # Limit to top 3 results
                    content = doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                    response_parts.append(f"{i}. {content}\n")
                return "\n".join(response_parts)
        
        # Default response if nothing specific found
        query_type = result.get("query_type", "unknown")
        return f"🤖 I've processed your {query_type} request. Please provide more details if you need specific assistance."


def create_chatbot_interface():
    """Create and return the Gradio chatbot interface."""
    
    chatbot = ChatbotInterface()
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    """
    
    # Create the Gradio interface
    with gr.Blocks(css=custom_css, title="Customer Assistant Chatbot") as interface:
        
        # Header
        gr.Markdown("""
        # 🤖 Customer Assistant Chatbot
        
        Welcome! I'm here to help you with:
        - **Refunds**: Process refund requests for your orders
        - **Technical Issues**: Help with software installation, login problems, and technical support
        - **General Questions**: Answer FAQs about policies, shipping, and account management
        
        Simply type your question below and I'll assist you!
        """)
        
        # Chat interface
        chatbot_component = gr.Chatbot(
            label="Chat History",
            height=500,
            show_label=True,
            container=True,
            bubble_full_width=False
        )
        
        # Input area
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your question here...",
                lines=2,
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Clear button
        clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        # Footer with information
        gr.Markdown("""
        ---
        **Examples you can try:**
        - "I want a refund for order #12345, my email is john@example.com"
        - "How do I fix TechOffice Suite installation error 1603?"
        - "What's your return policy?"
        - "I can't login to my account"
        """)
        
        # Event handlers
        def user_input(message, history):
            return chatbot.process_message(message, history)
        
        def clear_chat():
            return []
        
        # Connect events
        submit_btn.click(
            user_input,
            inputs=[msg, chatbot_component],
            outputs=[msg, chatbot_component]
        )
        
        msg.submit(
            user_input,
            inputs=[msg, chatbot_component],
            outputs=[msg, chatbot_component]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot_component]
        )
    
    return interface


def main():
    """Main function to launch the Gradio interface."""
    print("🚀 Starting Customer Assistant Chatbot...")
    print("📝 Loading workflow and models...")
    
    try:
        # Create and launch the interface
        interface = create_chatbot_interface()
        
        print("✅ Interface created successfully!")
        print("🌐 Launching web interface...")
        
        # Launch with specific settings
        interface.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,       # Default Gradio port
            share=False,            # Set to True if you want a public link
            debug=True,             # Enable debug mode
            show_error=True,        # Show errors in the interface
            quiet=False             # Show console output
        )
        
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 