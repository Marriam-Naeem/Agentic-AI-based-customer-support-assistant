import gradio as gr
import sys
import os
from typing import List, Tuple
import uuid

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph import graph
from states import create_initial_state
from settings import RATE_LIMIT_RESPONSE


class ChatbotInterface:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.session_id}}
    
    def process_message(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        if not message.strip():
            return "", history
        try:
            result = graph.invoke(
                create_initial_state(message, self.session_id),
                config=self.config
            )
            history.append([message, self._extract_response(result)])
            return "", history
        except Exception as e:
            error_message = str(e).lower()
            if any(keyword in error_message for keyword in ["rate_limit", "quota", "resource_exhausted", "429"]):
                history.append([message, RATE_LIMIT_RESPONSE])
            else:
                history.append([message, f"Sorry, I encountered an error while processing your request: {str(e)}"])
            return "", history
    
    def clear_session(self) -> Tuple[str, List[List[str]]]:
        self.session_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.session_id}}
        return "", []
    
    def _extract_response(self, result: dict) -> str:
        # Return the final email if available, otherwise a generic response
        if result.get("final_email"):
            return result["final_email"]
        
        # Fallback to final_response if available
        if result.get("final_response"):
            return result["final_response"]
        
        # Generic fallback
        return "I've processed your request. Please let me know if you need any clarification."


def create_chatbot_interface():
    chatbot = ChatbotInterface()
    custom_css = """
    .gradio-container { max-width: 800px !important; margin: 0 auto !important; }
    .chat-message { padding: 10px; border-radius: 10px; margin: 5px 0; }
    .user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
    .bot-message { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
    """
    
    with gr.Blocks(css=custom_css, title="Customer Assistant Chatbot") as interface:

        gr.Markdown("""
        # Customer Assistant Chatbot
        
        Welcome! I'm here to help you with:
        - **Refunds**: Process refund requests for your orders
        - **Technical Issues**: Help with software installation, login problems, and technical support
        - **General Questions**: Answer FAQs about policies, shipping, and account management
        
        Simply type your question below and I'll assist you!
        """)
        
        chatbot_component = gr.Chatbot(label="Chat History", height=500, show_label=True, container=True, bubble_full_width=False)
        with gr.Row():
            msg = gr.Textbox(label="Your Message", placeholder="Type your question here...", lines=2, scale=4)
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.Button("Clear Chat", variant="secondary")
        
        gr.Markdown("""
        ---
        **Examples you can try:**
        - "I want a refund for order #12345, my email is john@example.com"
        - "How do I fix TechOffice Suite installation error 1603?"
        - "What's your return policy?"
        - "I can't login to my account"
        """)
        
        def user_input(message, history):
            return chatbot.process_message(message, history)
        def clear_chat():
            new_msg, new_history = chatbot.clear_session()
            return new_msg, new_history
        submit_btn.click(user_input, inputs=[msg, chatbot_component], outputs=[msg, chatbot_component])
        msg.submit(user_input, inputs=[msg, chatbot_component], outputs=[msg, chatbot_component])
        clear_btn.click(clear_chat, outputs=[msg, chatbot_component])
    
    return interface

def main():
    try:
        interface = create_chatbot_interface()
        interface.launch(server_name="localhost", server_port=7860, share=False, debug=False)
    except Exception as e:
        print(f"Error launching interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()